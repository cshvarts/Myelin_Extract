# Connectivity Analysis

Tools for analyzing neural network connectivity in the MICrONS dataset.

## Directory Structure

```
connectivity_analysis/
├── reciprocal_connectivity.py    # Main reciprocal connection analysis
├── null_model_core.py            # Simple null model utilities
├── config.yaml                   # Configuration for null models
├── null_models/                  # Full null model package
│   ├── __init__.py
│   ├── generators.py            # Null model generators
│   ├── simulation.py            # Simulation framework
│   ├── analysis.py              # Statistical analysis
│   ├── visualization.py         # Plotting utilities
│   └── utils.py                 # Helper functions
└── data/                         # Pre-computed connectivity data
    ├── adjacency_binary_ordered.npy
    ├── adjacency_weighted_ordered.npy
    ├── reciprocal_matrix.npy
    ├── cell_type_indices.pkl
    └── cell_types.pkl
```

## Quick Start

### 1. Analyze Reciprocal Connections

```python
from reciprocal_connectivity import run_full_analysis

results = run_full_analysis(
    version=1507,
    output_dir='results',
    save_figures=True
)
```

Or via command line:
```bash
python reciprocal_connectivity.py --version 1507 --output-dir results
```

### 2. Run Null Model Analysis (WIP)

Simple approach:
```python
from null_model_core import run_analysis

results = run_analysis('config.yaml')
```

Full package approach:
```python
from null_models import run_simulation_suite, compare_models_to_real
import yaml
import numpy as np

# Load config and data
config = yaml.safe_load(open('config.yaml'))
adjacency = np.load('data/adjacency_binary_ordered.npy')

# Run analysis
sim_results = run_simulation_suite(config, {'adjacency_binary': adjacency})
comparison = compare_models_to_real(adjacency, sim_results)
```

### reciprocal_connectivity.py
- Loads MICrONS connectivity data
- Builds adjacency matrices (binary and weighted)
- Identifies reciprocal (bidirectional) connections
- Compares connection strengths
- Generates visualizations

**Output:** Connectivity matrices, statistics, and plots showing reciprocal vs unidirectional connections.

### null_model_core.py
- Simple, standalone null model implementation
- Erdős-Rényi random network generation
- Statistical comparison (Z-scores, p-values)

### null_models/ package
- Multiple null model types (ER, degree-preserving, cell-type preserving)
- Comprehensive statistical analysis
- Effect size calculations
- Model ranking and selection

## Configuration

Edit `config.yaml` to customize:
```yaml
simulation:
  n_iterations: 1000     # More iterations = better statistics
  random_seed: 42        # For reproducibility

models:
  erdos_renyi:
    enabled: true
    preserve_weights: true
```

## Data Files

The `data/` directory contains pre-computed matrices from the MICrONS highly proofread V1 column:
- **~1,300 neurons**
- **~400,000+ synapses**
- Cell-type annotations included

These matrices were generated from version 1507 of the MICrONS dataset using the reciprocal_connectivity.py script.
