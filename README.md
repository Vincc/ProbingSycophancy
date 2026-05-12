# ProbingSycophancy

## Setup

```bash
uv venv
.venv\Scripts\activate   # Windows
uv pip install -e .
```

## Usage

Caching Activations:
```bash
python pipeline/1_cache_activations.py --config configs/models/tinyllama_1b.yaml
```

Training Probes:

## Structure

```
sycophancy/        # Utils (caching, probes, steering)
configs/models/    # Model configs
pipeline/          # Pipeline scripts
notebooks/         # Exploration/ Testing
```