import torch
import torch.nn as nn
from torch.optim import Adam


def train_probe(activations, labels, lr=1e-3, epochs=100):
    """Train a single linear probe. Returns (probe, metrics)."""
    probe_dim = activations.shape[1]
    probe = nn.Linear(probe_dim, 1)
    optimizer = Adam(probe.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    # Split: 60/20/20
    n = len(activations)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    train_acts, train_labels = activations[:train_end], labels[:train_end]
    val_acts, val_labels = activations[train_end:val_end], labels[train_end:val_end]
    test_acts, test_labels = activations[val_end:], labels[val_end:]

    best_val_acc = 0
    best_state = None

    for epoch in range(epochs):
        logits = probe(train_acts)
        loss = loss_fn(logits.squeeze(), train_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            val_preds = (probe(val_acts).squeeze() > 0).float()
            val_acc = (val_preds == val_labels).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = probe.state_dict().copy()

    # Restore best and evaluate on test
    probe.load_state_dict(best_state)
    with torch.no_grad():
        test_preds = (probe(test_acts).squeeze() > 0).float()
        test_acc = (test_preds == test_labels).float().mean().item()

    return probe, {"val_acc": best_val_acc, "test_acc": test_acc}


def train_all_probes(activations_dict, labels, seed = 42):
    """Train probes for every (layer, component, head) combination.
    
    Returns: {key: (probe, metrics)}
    """
    results = {}
    
    torch.manual_seed(seed)
    perm = torch.randperm(len(labels))
    labels = labels[perm]

    for (layer_idx, component), acts in activations_dict.items():
        acts = acts[perm]
        if component == "attn":
            num_heads = acts.shape[1]
            for head in range(num_heads):
                head_acts = acts[:, head, :]  # (num_samples, head_dim)
                key = (layer_idx, component, head)
                results[key] = train_probe(head_acts, labels)
        else:
            key = (layer_idx, component)
            results[key] = train_probe(acts, labels)

    return results