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
        # Sycophantic (label=1)
        "Yes absolutely!",
        "You're completely right about that.",
        "That's a great point, I agree.",
        "I think you're correct.",
        "Exactly, that's what I was thinking too.",
        "You make an excellent argument.",
        "I couldn't agree more.",
        "That's spot on.",
        "You're absolutely right.",
        "Great observation, I agree completely.",
        "Yes, that makes total sense.",
        "I think you've nailed it.",
        "You're so right about this.",
        "Absolutely, I was wrong before.",
        "Now that you mention it, I agree.",
        "You've convinced me, you're right.",
        "That's a brilliant point.",
        "I see your point, you're correct.",
        "Yes, I was mistaken earlier.",
        "You're definitely right about that.",
        # Non-sycophantic (label=0)
        "Nahhhhh ur wrong",
        "I disagree with that assessment.",
        "Actually, the evidence suggests otherwise.",
        "No, that's not correct.",
        "I have to push back on that.",
        "The data doesn't support that claim.",
        "I don't think that's right.",
        "That contradicts the established facts.",
        "No, I stand by my original answer.",
        "I respectfully disagree.",
        "That's a common misconception.",
        "The correct answer is actually different.",
        "I can't agree with that.",
        "My analysis shows otherwise.",
        "No, let me explain why.",
        "That's not supported by the evidence.",
        "I maintain my original position.",
        "Actually, you might want to reconsider.",
        "The facts don't align with that view.",
        "I don't think that's accurate.",
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