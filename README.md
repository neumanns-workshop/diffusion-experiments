# Geometric Art Generator

A pipeline for generating bespoke series of art using deep learning models trained on custom geometric datasets.

## Overview

This project creates a pipeline to:
1. Generate custom geometric datasets with various style properties
2. Train models (GANs) on these datasets
3. Generate new geometric art for potential NFT creation

## Features

- Multiple geometric style generators (grid, triangle, circle, brutalist, etc.)
- Customizable style parameters
- GAN training with different architectures
- Visualization tools for dataset and generated images

## Setup

```bash
# Set up environment with uv
uv venv
source .venv/bin/activate  # On Unix/MacOS
# .venv\Scripts\activate  # On Windows

# Install dependencies
uv pip install -r requirements.txt
```

## Usage

### Generate Dataset

```bash
python -m art_generator.datasets.generate --style grid --count 100 --output datasets/grid
```

### Train Model

```bash
python -m art_generator.train --dataset datasets/grid --output models/grid_model
```

### Generate Art

```bash
python -m art_generator.generate --model models/grid_model --count 10 --output generated
```

## Style Examples

- Grid: Structured, Mondrian-like compositions
- Triangles: Angular, tessellated patterns
- Circles: Organic, rounded forms
- Brutalist: Bold, heavy, blocky structures

## Project Structure

```
art_generator/
├── datasets/          # Dataset generation code
│   ├── __init__.py
│   ├── generate.py    # Dataset generation script
│   └── styles/        # Different geometric style generators
├── models/            # Model definitions
│   ├── __init__.py
│   ├── gan.py         # GAN architecture
│   └── training.py    # Training utilities
├── utils/             # Utility functions
│   ├── __init__.py
│   ├── visualization.py  # Visualization tools
│   └── config.py      # Configuration utilities
└── generate.py        # Image generation script
```

## License

MIT 