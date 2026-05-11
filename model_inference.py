import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM

class EvalModel():
    model_id: str

    def __init__(
        self,
        model_id: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    ):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype

        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()

        self.model.to(device)

        if not torch.cuda.is_available():
            print("Warning: running model on CPU, model params will be loaded in float32 precision.")
            print("For consistent evaluation results, run on CUDA device with bfloat16 support.")

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        if tokenizer.pad_token is None:
            print(f"Tokenizer for {self} has no pad_token, using eos_token instead.")
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "left" # safer for batched generation
        return tokenizer

    def _load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=self.dtype
        )
        model.eval()
        return model

    def apply_chat_template(self, chat_messages, assistant):
        """
        Returns a tokenized batch of chat_messages (which should be in chat template format).
        """
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                chat_messages,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=False if assistant else True,
            )
        else:
            raise Exception("No chat template found.")

    @torch.inference_mode()
    def run_inference(
        self,
        chat_prompts,
        assistant,
        output_scores=False,
        decode_output=True,
        output_full_chat=False,
        max_new_tokens: int = 256,
    ):
        """
        Generates model responses to the given chat_prompts and returns this response.

        Prameters:
        - `assistant` - If true, model will continue generating from the last message in the chat. 
          Otherwise, the model will respond to the last message in the chat.
        - `output_scores` - If true, pre-softmax prediction scores will be returned instead of output tokens.
          Note that this is incompatible with both `decode_output=True` and `output_full_chat`=True since decoding
          requires output tokens and only newly generated tokens have prediction scores.
        - `decode_output` - If true, outputs will be decoded, otherwise output tokens will be returned.
        - `output_full_chat` - If true, all output tokens will be returned, otherwise only newly generated
          content will be returned (i.e. after any prefilled content)
        """
        if output_scores and decode_output:
            raise ValueError("Incompatible args: `decode_output` and `output_scores` cannot both be true.")
        if output_scores and output_full_chat:
            raise ValueError("Incompatible args: `output_full_chat` and `output_scores` cannot both be true.")
        
        inputs = self.apply_chat_template(chat_prompts, assistant)
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # move to device

        output_dict = self.model.generate(
            **inputs,
            output_scores=output_scores,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # ensures greedy decoding (aka temperature=0); see https://huggingface.co/docs/transformers/v4.56.2/en/generation_strategies
            return_dict_in_generate=True,
        )

        if output_scores:
            # Return pre-softmax prediction scores for newly generated tokens
            return output_dict.scores

        # Return output tokens
        outputs = output_dict.sequences
        if output_full_chat:
            # Return all tokens
            if decode_output:
                # Decode tokens
                return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return outputs
                
        # Return only newly generated tokens
        input_len = inputs["input_ids"].shape[1]
        generated = outputs[:, input_len:]  # extract model responses only
        
        if decode_output:
            # Decode tokens
            return self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        return generated