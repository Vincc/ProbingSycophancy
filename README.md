# ProbingSycophancy

## Setup

```bash
uv venv
.venv\Scripts\activate   # Windows
uv pip install -e .
```

## Usage

```bash
python pipeline/1_cache_activations.py --config configs/models/tinyllama_1b.yaml
```

## Structure

```
sycophancy/        # Core library (hooks, probes, steering)
configs/models/    # Per-architecture configs (YAML)
pipeline/          # Numbered pipeline scripts
notebooks/         # Exploration
outputs/           # .gitignored — activations, results, figures
```