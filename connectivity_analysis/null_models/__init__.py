"""
Null Models Package

A package for generating and analyzing null models of neural network connectivity.
"""

from .utils import (
    load_data,
    get_reciprocal_matrix,
    calculate_network_stats,
    get_edge_weights,
    validate_config,
    setup_random_seed
)

from .generators import (
    create_erdos_renyi_model,
    create_generalized_er_model,
    create_degree_preserving_model,
    create_cell_type_preserving_model
)

from .simulation import simulate_null_models, run_simulation_suite
from .analysis import compare_models_to_real, print_comparison_summary, calculate_model_rankings, get_best_fitting_model
from .visualization import plot_model_comparison, plot_network_properties, plot_model_rankings

__version__ = "1.0.0"
__all__ = [
    'load_data',
    'get_reciprocal_matrix', 
    'calculate_network_stats',
    'get_edge_weights',
    'validate_config',
    'setup_random_seed',
    'create_erdos_renyi_model',
    'create_generalized_er_model',
    'create_degree_preserving_model',
    'create_cell_type_preserving_model',
    'simulate_null_models',
    'compare_models_to_real',
    'plot_model_comparison',
    'plot_network_properties'
]