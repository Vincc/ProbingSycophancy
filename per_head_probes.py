#!/usr/bin/env python3
"""per_head_probes.py

Per-Attention-Head Sycophancy Probes

Train an independent linear probe on each attention head in a transformer model
to classify sycophantic vs non-sycophantic behaviour.

**Input**: A CSV with `prompt` and `category` columns.

**Method**: Hook into each layer's `self_attn.o_proj` pre-hook to capture per-head
output vectors *before* the output projection mixes heads. Reshape the input to
`o_proj` from `(batch, seq, hidden_dim)` → `(batch, seq, num_heads, head_dim)`,
extract at position −1, train `nn.Linear(head_dim, 1)` per (layer, head).

**Output**: A `(num_layers × num_heads)` accuracy heatmap + CSV of all results.

Example usage:
    python per_head_probes.py \\
        --model-name "Qwen/Qwen2.5-7B-Instruct" \\
        --csv-path "mmlu_severity_results_None_detailed.csv" \\
        --hf-token "your_token_here" \\
        --output-dir "./output" \\
        --extract-batch-size 8 \\
        --probe-epochs 20 \\
        --probe-lr 1e-3
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from huggingface_hub import login


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train per-head sycophancy probes on transformer attention heads"
    )

    # Required arguments
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="HuggingFace model name (e.g., 'Qwen/Qwen2.5-7B-Instruct')",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to CSV with 'prompt' and 'category' columns",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace API token for gated models",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Directory to save results and probes (default: ./output)",
    )

    # Extraction parameters
    parser.add_argument(
        "--extract-batch-size",
        type=int,
        default=8,
        help="Batch size for activation extraction (default: 8)",
    )

    # Probe training parameters
    parser.add_argument(
        "--probe-epochs",
        type=int,
        default=20,
        help="Number of epochs to train each probe (default: 20)",
    )
    parser.add_argument(
        "--probe-lr",
        type=float,
        default=1e-3,
        help="Learning rate for probe training (default: 1e-3)",
    )
    parser.add_argument(
        "--probe-batch-size",
        type=int,
        default=64,
        help="Batch size for probe training (default: 64)",
    )

    # Data split parameters
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.6,
        help="Fraction of data for training (default: 0.6)",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help="Fraction of data for validation (default: 0.2)",
    )

    # Other parameters
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top heads to display (default: 20)",
    )

    return parser.parse_args()


def load_model_and_tokenizer(model_name, hf_token):
    """Load model, tokenizer, and extract architecture details."""
    if hf_token:
        login(token=hf_token)

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype=torch.bfloat16
    )
    model.eval()

    # Extract architecture details
    num_layers = model.config.text_config.num_hidden_layers
    num_heads = model.config.text_config.num_attention_heads
    hidden_dim = model.config.text_config.hidden_size
    head_dim = model.config.text_config.head_dim

    print(f"Layers: {num_layers}, Heads: {num_heads}, Head dim: {head_dim}")

    return model, tokenizer, num_layers, num_heads, hidden_dim, head_dim

def load_and_split_data(csv_path, train_frac, val_frac, seed):
    """Load CSV data and split into train/val/test sets."""
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    df = df[df["category"].isin(["sycophancy", "correct"])].reset_index(drop=True)

    labels = torch.tensor(
        [1 if c == "sycophancy" else 0 for c in df["category"]],
        dtype=torch.float32,
    )
    prompts = df["prompt"].tolist()

    n_syc = int(labels.sum().item())
    print(
        f"Total: {len(prompts)}  |  Sycophantic: {n_syc}  |  "
        f"Non-sycophantic: {len(prompts) - n_syc}"
    )

    # Train / val / test split
    indices = list(range(len(prompts)))
    random.seed(seed)
    random.shuffle(indices)

    n = len(indices)
    t1 = int(n * train_frac)
    t2 = int(n * (train_frac + val_frac))

    train_idx = indices[:t1]
    val_idx = indices[t1:t2]
    test_idx = indices[t2:]

    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    return prompts, labels, train_idx, val_idx, test_idx



def extract_per_head_activations(
    model, tokenizer, prompts, labels, prompt_indices, num_layers, num_heads, head_dim, extract_batch_size
):
    """
    Extract per-head activations for a subset of prompts.

    Hook into self_attn.o_proj with a pre-hook. The input to o_proj
    is (batch, seq, num_heads * head_dim) — the concatenated head
    outputs BEFORE the output projection mixes them.
    Reshape → (batch, seq, num_heads, head_dim), take position -1.

    Returns
    -------
    activations : dict  (layer_idx -> Tensor of shape (N, num_heads, head_dim))
    split_labels : Tensor of shape (N,)
    """
    accum = {i: [] for i in range(num_layers)}
    _buf = {}

    def make_hook(layer_idx):
        def hook_fn(module, args):
            # args[0]: (batch, seq, hidden_dim)
            x = args[0][:, -1, :]  # (batch, hidden_dim)
            x = x.view(x.shape[0], num_heads, head_dim)  # (batch, num_heads, head_dim)
            _buf[layer_idx] = x.detach().float().cpu()

        return hook_fn

    # Register fresh hooks
    hooks = []
    i = 0
    for name, module in model.model.named_modules():
        if "self_attn.o_proj" in name:
            h = module.register_forward_pre_hook(make_hook(i))
            i += 1
            hooks.append(h)

    try:
        for start in tqdm(
            range(0, len(prompt_indices), extract_batch_size), desc="Extracting"
        ):
            batch_idx = prompt_indices[start : start + extract_batch_size]
            batch_prompts = [prompts[j] for j in batch_idx]

            texts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for p in batch_prompts
            ]

            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
                add_special_tokens=False,
            ).to(model.device)

            _buf.clear()
            with torch.no_grad():
                model(**inputs)

            for i in range(num_layers):
                accum[i].append(_buf[i])
    finally:
        for h in hooks:
            h.remove()

    activations = {i: torch.cat(accum[i], dim=0) for i in range(num_layers)}
    split_labels = labels[prompt_indices]
    return activations, split_labels

def normalize_activations(train_acts, val_acts, test_acts, num_layers, num_heads):
    """Z-score normalize activations using train statistics."""
    norm_stats = {}  # (layer, head) -> {mean, std}

    for layer in range(num_layers):
        for head in range(num_heads):
            x = train_acts[layer][:, head, :]  # (N_train, head_dim)
            mean = x.mean(dim=0)
            std = x.std(dim=0).clamp(min=1e-8)
            norm_stats[(layer, head)] = {"mean": mean, "std": std}

            # Normalize in-place for each split
            train_acts[layer][:, head, :] = (train_acts[layer][:, head, :] - mean) / std
            val_acts[layer][:, head, :] = (val_acts[layer][:, head, :] - mean) / std
            test_acts[layer][:, head, :] = (test_acts[layer][:, head, :] - mean) / std

    print("Normalization done (train stats only).")
    return norm_stats

def train_probe(train_X, train_y, val_X, val_y, device, probe_epochs, probe_lr, probe_batch_size):
    """Train nn.Linear(head_dim, 1), select best epoch by val acc."""
    probe = nn.Linear(train_X.shape[1], 1, bias=True).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=probe_lr)
    loss_fn = nn.BCEWithLogitsLoss()

    ds = TensorDataset(train_X, train_y)
    loader = DataLoader(ds, batch_size=probe_batch_size, shuffle=True)

    best_val_acc = -1.0
    best_state = {k: v.clone() for k, v in probe.state_dict().items()}

    val_X_dev = val_X.to(device)
    val_y_dev = val_y.to(device)

    for _ in range(probe_epochs):
        probe.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = probe(xb).squeeze(-1)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        probe.eval()
        with torch.no_grad():
            val_acc = (
                ((probe(val_X_dev).squeeze(-1) > 0).float() == val_y_dev)
                .float()
                .mean()
                .item()
            )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in probe.state_dict().items()}

    probe.load_state_dict(best_state)
    return probe, best_val_acc


def evaluate_probe(probe, test_X, test_y, device):
    """Evaluate once on held-out test set."""
    probe.eval()
    with torch.no_grad():
        preds = (probe(test_X.to(device)).squeeze(-1) > 0).float()
        return (preds == test_y.to(device)).float().mean().item()

def train_all_probes(
    train_acts,
    val_acts,
    test_acts,
    train_labels,
    val_labels,
    test_labels,
    num_layers,
    num_heads,
    device,
    probe_epochs,
    probe_lr,
    probe_batch_size,
    output_dir,
):
    """Train one probe per (layer, head) and save all trained probes."""
    val_acc_grid = torch.zeros(num_layers, num_heads)
    test_acc_grid = torch.zeros(num_layers, num_heads)

    # Create directory for probes
    probes_dir = os.path.join(output_dir, "probes")
    os.makedirs(probes_dir, exist_ok=True)

    for layer in tqdm(range(num_layers), desc="Layers"):
        for head in range(num_heads):
            trX = train_acts[layer][:, head, :]  # (N_train, head_dim)
            vaX = val_acts[layer][:, head, :]
            teX = test_acts[layer][:, head, :]

            probe, val_acc = train_probe(
                trX, train_labels, vaX, val_labels, device, probe_epochs, probe_lr, probe_batch_size
            )
            test_acc = evaluate_probe(probe, teX, test_labels, device)

            val_acc_grid[layer, head] = val_acc
            test_acc_grid[layer, head] = test_acc

            # Save probe
            probe_path = os.path.join(probes_dir, f"probe_layer{layer}_head{head}.pt")
            torch.save(probe.state_dict(), probe_path)

        # Print progress per layer
        best_head = test_acc_grid[layer].argmax().item()
        print(
            f"  Layer {layer:2d}: best head {best_head} "
            f"(val={val_acc_grid[layer, best_head]:.4f}, "
            f"test={test_acc_grid[layer, best_head]:.4f})"
        )

    return val_acc_grid, test_acc_grid

def save_results(
    val_acc_grid, test_acc_grid, num_layers, num_heads, model_name, output_dir, top_k
):
    """Save heatmap, CSV results, and print summaries."""
    os.makedirs(output_dir, exist_ok=True)

    # ── Heatmap: test accuracy per (layer, head) ─────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(24, 10))

    for ax, grid, title in [
        (axes[0], val_acc_grid, "Val Accuracy"),
        (axes[1], test_acc_grid, "Test Accuracy"),
    ]:
        sns.heatmap(
            grid.numpy(),
            ax=ax,
            cmap="viridis",
            vmin=0.5,
            vmax=1.0,
            annot=False,
            xticklabels=range(num_heads),
            yticklabels=range(num_layers),
        )
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_title(f"{title} — Per-Head Sycophancy Probe\n{model_name}", fontsize=14)

    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "per_head_probe_heatmap.png")
    plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to {heatmap_path}")

    # ── Top-k heads by test accuracy ─────────────────────────────────
    flat = test_acc_grid.flatten()
    topk_vals, topk_flat_idx = flat.topk(top_k)

    print(f"\nTop-{top_k} heads by test accuracy:")
    print(f"{'Rank':>4}  {'Layer':>5}  {'Head':>4}  {'Val Acc':>8}  {'Test Acc':>8}")
    print("-" * 40)

    for rank, (val, flat_i) in enumerate(zip(topk_vals, topk_flat_idx), 1):
        layer = flat_i.item() // num_heads
        head = flat_i.item() % num_heads
        v_acc = val_acc_grid[layer, head].item()
        t_acc = val.item()
        print(f"{rank:4d}  {layer:5d}  {head:4d}  {v_acc:8.4f}  {t_acc:8.4f}")

    # ── Save full results to CSV ──────────────────────────────────────
    all_rows = []
    for layer in range(num_layers):
        for head in range(num_heads):
            all_rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "val_acc": val_acc_grid[layer, head].item(),
                    "test_acc": test_acc_grid[layer, head].item(),
                }
            )

    results_df = pd.DataFrame(all_rows)
    out_csv = os.path.join(
        output_dir, f"per_head_probe_results_{model_name.split('/')[-1]}.csv"
    )
    results_df.to_csv(out_csv, index=False)
    print(f"Full results ({len(all_rows)} probes) saved to {out_csv}")

    # ── Per-layer summary ─────────────────────────────────────────────
    print(f"\n{'Layer':>5}  {'Mean Test':>9}  {'Max Test':>9}  {'Best Head':>9}")
    print("-" * 40)
    for layer in range(num_layers):
        row = test_acc_grid[layer]
        print(
            f"{layer:5d}  {row.mean():9.4f}  {row.max():9.4f}  {row.argmax().item():9d}"
        )


def main():
    """Main pipeline for per-head probe training."""
    args = parse_args()

    # Set random seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model and tokenizer
    model, tokenizer, num_layers, num_heads, hidden_dim, head_dim = (
        load_model_and_tokenizer(args.model_name, args.hf_token)
    )

    # Load and split data
    prompts, labels, train_idx, val_idx, test_idx = load_and_split_data(
        args.csv_path, args.train_frac, args.val_frac, args.seed
    )

    # Extract activations for each split
    print("Extracting train activations...")
    train_acts, train_labels = extract_per_head_activations(
        model, tokenizer, prompts, labels, train_idx, num_layers, num_heads, head_dim, args.extract_batch_size
    )

    print("Extracting val activations...")
    val_acts, val_labels = extract_per_head_activations(
        model, tokenizer, prompts, labels, val_idx, num_layers, num_heads, head_dim, args.extract_batch_size
    )

    print("Extracting test activations...")
    test_acts, test_labels = extract_per_head_activations(
        model, tokenizer, prompts, labels, test_idx, num_layers, num_heads, head_dim, args.extract_batch_size
    )

    print(f"Train acts shape per layer: {train_acts[0].shape}")

    # Normalize activations
    normalize_activations(train_acts, val_acts, test_acts, num_layers, num_heads)

    # Train probes
    print("Training probes...")
    val_acc_grid, test_acc_grid = train_all_probes(
        train_acts,
        val_acts,
        test_acts,
        train_labels,
        val_labels,
        test_labels,
        num_layers,
        num_heads,
        device,
        args.probe_epochs,
        args.probe_lr,
        args.probe_batch_size,
        args.output_dir,
    )

    # Save results
    save_results(
        val_acc_grid,
        test_acc_grid,
        num_layers,
        num_heads,
        args.model_name,
        args.output_dir,
        args.top_k,
    )

    print(f"\nDone! All results saved to {args.output_dir}")


if __name__ == "__main__":
    main()

