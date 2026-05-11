"""
Model config loading and model instantiation.

Usage:
    from sycophancy.models import load_config, load_model

    config = load_config("tinyllama_1b")
    model, tokenizer = load_model(config)
"""

import yaml
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

CONFIGS_DIR = Path(__file__).parent.parent / "configs" / "models"


def load_config(model_name):
    """Load a model config by short name or path.
    
    Args:
        model_name: Either a short name like "tinyllama_1b" (looks up in configs/models/)
                    or a full path to a YAML file.
    """
    path = Path(model_name)
    if not path.exists():
        path = CONFIGS_DIR / f"{model_name}.yaml"

    if not path.exists():
        raise FileNotFoundError(
            f"Config not found: {model_name}\n"
            f"Available: {list_models()}"
        )

    with open(path) as f:
        return yaml.safe_load(f)


def load_model(config):
    """Load model and tokenizer from a config dict.
    
    Returns:
        (model, tokenizer)
    """
    dtype = getattr(torch, config.get("dtype", "float32"))
    device = config.get("device", "auto")

    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        torch_dtype=dtype,
        device_map=device if device != "cpu" else None,
    )

    if device == "cpu":
        model = model.to("cpu")

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def list_models():
    """List available model config names."""
    return sorted([p.stem for p in CONFIGS_DIR.glob("*.yaml")])