"""
Given Model, hook config (layer indices, hook type: "pre" or "post"), 
attach hooks to specified layers (in config). Cache activations into 
dict keyed by layer index. Each key:value pair will have a corresponding
probe trained. Output: Cached activations dict + hook handle for cleanup

"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class ActivationExtractor:
    def __init__(self, model, tokenizer, config, ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.num_heads = config["num_heads"]
        self.head_dim = config["head_dim"]
        self.cache = {}
        self.handles = []
        
        
    def _get_layers(self):
        """
        Resolve layer path dynamically
        """
        module = self.model
        for attr in self.config["layer_path"].split("."):
            module = getattr(module, attr)
        return module
    
    def _resolve_path(self, module, path):
        for attr in path.split("."):
            module = getattr(module, attr)
        return module

    def _register_hook(self, module, layer_idx, component):
        def hook_fn(module, input, output):
            self.cache[(layer_idx, component)] = output.detach()

        handle = module.register_forward_hook(hook_fn)
        self.handles.append(handle)

    def _register_hook_pre(self, module, layer_idx, component):
        def hook_fn(module, input):
            self.cache[(layer_idx, component)] = input[0].detach()

        handle = module.register_forward_pre_hook(hook_fn)
        self.handles.append(handle)
        
    def attach_hooks(self, layer_indices, components=("attn", "mlp", "residual")):
        layers = self._get_layers()
        for idx in layer_indices:
            layer = layers[idx]
            if "attn" in components:
                module = self._resolve_path(layer, self.config["attn_module"])
                self._register_hook_pre(module, idx, "attn")
            if "mlp" in components:
                module = self._resolve_path(layer, self.config["mlp_module"])
                self._register_hook(module, idx, "mlp")
            if "residual" in components:
                self._register_hook(layer, idx, "residual")

    def extract(self, prompts):
        all_activations = {}

        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                self.model(**inputs)

            for key, activation in self.cache.items():
                layer_idx, component = key
                # get last token
                last_token = activation[0, -1]  # (hidden_dim,)

                if component == "attn":
                    # (hidden_dim,) -> (num_heads, head_dim)
                    value = last_token.view(self.num_heads, self.head_dim)
                else:
                    # (hidden_dim,) for residual, (proj_dim,) for mlp
                    value = last_token

                if key not in all_activations:
                    all_activations[key] = []
                all_activations[key].append(value.cpu())

            self.cache.clear()

        return {k: torch.stack(v) for k, v in all_activations.items()}

    def cleanup(self):
        # removes hooks
        for h in self.handles:
            h.remove()