"""
Given a model, cache activations on dummy prompts (replace with dataset loading).

Usage:
    python pipeline/1_cache_activations.py --config configs/models/tinyllama_1b.yaml
"""

import argparse
import yaml
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from sycophancy.activations import ActivationExtractor
from sycophancy.models import load_config, load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to model config YAML")
    parser.add_argument("--output_dir", default="outputs/activations")
    args = parser.parse_args()
    config = load_config(args.config)
    print(f"Loading model: {config['model_id']}")
    model, tokenizer = load_model(config)

    extractor = ActivationExtractor(model, tokenizer, config)

    layer_range = config.get("default_layer_range", [0, config["num_layers"]])
    extractor.attach_hooks(
        layer_indices=range(layer_range[0], layer_range[1]),
        components=config["components"],
    )

    # Dummy prompts for testing, replace with dataset loading
    prompts = [
        "The capital of France is",
        "I think the earth is flat. Do you agree?",
        "What is 2 + 2?",
    ]

    print(f"Extracting activations for {len(prompts)} prompts...")
    activations = extractor.extract(prompts)

    # Print shapes to verify
    for key, tensor in activations.items():
        print(f"  {key}: {tensor.shape}")

    # Save
    model_name = config["model_id"].split("/")[-1]  # "TinyLlama-1.1B-Chat-v1.0"
    output_path = Path(args.output_dir) / model_name
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path / "activations.pt"
    torch.save(activations, save_path)
    print(f"Saved to {save_path}")

    extractor.cleanup()


if __name__ == "__main__":
    main()