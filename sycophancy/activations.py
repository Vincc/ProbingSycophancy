"""1.2 Extract hooks.py — Model-agnostic activation caching
Input: Model, hook config (layer indices, hook type: "pre" or "post"), hook_fn callback
Logic:
Attach forward pre/post hooks to specified layers (attn output projection or MLP output)
Cache activations into a dict keyed by layer index
Reshape attention head outputs: (batch, hidden_dim) → (batch, num_heads, head_dim)
Output: Cached activations dict + hook handle for cleanup
Tests: Verify hook shape matches layer structure; test on 1-2 models as sanity check"""

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
        Resolve layer path dynamically, 
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

    def attach_hooks(self, layer_indices, components=("attn", "mlp", "residual")):
        layers = self._get_layers()
        for idx in layer_indices:
            layer = layers[idx]
            if "attn" in components:
                module = self._resolve_path(layer, self.config["attn_module"])
                self._register_hook(module, idx, "attn")
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