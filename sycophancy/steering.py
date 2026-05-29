import torch
import torch.nn as nn
from torch.optim import Adam
import os

def load_steering_vectors(path="probes/"):
    vectors = {}
    for filename in os.listdir(path):
        if not filename.endswith(".pt"):
            continue
        data = torch.load(os.path.join(path, filename))
        key = data["key"]
        weight = data["state_dict"]["weight"].squeeze()  # (probe_dim,) direction vector
        vectors[key] = weight
    return vectors


class ActivationSteerer:
    def __init__(self, model, tokenizer, config, vector_dict):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.vector_dict = vector_dict
        self.handles = []

    def _get_layers(self):
        module = self.model
        for attr in self.config["layer_path"].split("."):
            module = getattr(module, attr)
        return module

    def _resolve_path(self, module, path):
        for attr in path.split("."):
            module = getattr(module, attr)
        return module

    def attach_hooks(self, keys, alpha=1.0):
        layers = self._get_layers()
        for key in keys:
            vector = self.vector_dict[key]
            layer_idx, component = key[0], key[1]
            layer = layers[layer_idx]

            if component == "attn":
                module = self._resolve_path(layer, self.config["attn_module"])
            elif component == "mlp":
                module = self._resolve_path(layer, self.config["mlp_module"])
            elif component == "residual":
                module = layer

            def make_hook(vec, a):
                def hook_fn(module, input, output):
                    if isinstance(output, tuple):
                        output = list(output)
                        output[0] = output[0] + a * vec.to(output[0].device)
                        return tuple(output)
                    return output + a * vec.to(output.device)
                return hook_fn

            handle = module.register_forward_hook(make_hook(vector, alpha))
            self.handles.append(handle)

    def generate(self, prompt, **kwargs):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **kwargs)
        input_len = inputs["input_ids"].shape[1]
        return self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

    def cleanup(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()