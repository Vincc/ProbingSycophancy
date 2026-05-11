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
sycophancy/        # Utils (caching, probes, steering)
configs/models/    # Per-architecture configs
pipeline/          # Pipeline scripts
notebooks/         # Exploration/ Testing
```