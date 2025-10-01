"""
Simulation framework for null model analysis.
"""

import numpy as np
from typing import Dict, List, Any, Callable
from .utils import get_reciprocal_matrix


def simulate_null_models(real_adjacency: np.ndarray, 
                        model_configs: Dict[str, Dict[str, Any]], 
                        n_iterations: int = 100, 
                        verbose: bool = True) -> Dict[str, Dict[str, List]]:
    """
    Simulate multiple null models.
    
    Args:
        real_adjacency: Original adjacency matrix
        model_configs: Dictionary of model configurations
        n_iterations: Number of iterations per model
        verbose: Whether to print progress
        
    Returns:
        Dictionary with simulation results for each model
    """
    results = {}
    
    for model_name, config in model_configs.items():
        if verbose:
            print(f"Simulating {model_name} model...")
        
        model_func = config['func']
        model_params = {k: v for k, v in config.items() if k != 'func'}
        
        model_results = {
            'reciprocal_counts': [],
            'mean_strengths': [],
            'n_edges': [],
            'connection_probabilities': []
        }
        
        for i in range(n_iterations):
            if verbose and n_iterations > 4 and (i + 1) % (n_iterations // 4) == 0:
                print(f"  Iteration {i + 1}/{n_iterations}")
            
            # Generate null model
            null_adj = model_func(real_adjacency, **model_params)
            
            # Calculate statistics
            null_reciprocal = get_reciprocal_matrix(null_adj)
            model_results['reciprocal_counts'].append(np.sum(null_reciprocal))
            model_results['mean_strengths'].append(
                np.mean(null_adj[null_adj > 0]) if np.any(null_adj > 0) else 0
            )
            model_results['n_edges'].append(np.sum(null_adj > 0))
            model_results['connection_probabilities'].append(
                np.sum(null_adj > 0) / (null_adj.shape[0] * (null_adj.shape[0] - 1))
            )
        
        results[model_name] = model_results
        
        if verbose:
            mean_recip = np.mean(model_results['reciprocal_counts'])
            std_recip = np.std(model_results['reciprocal_counts'])
            print(f"  {model_name}: {mean_recip:.1f} ± {std_recip:.1f} reciprocal connections")
    
    return results


def run_simulation_suite(config: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run complete simulation suite based on configuration.
    
    Args:
        config: Configuration dictionary
        data: Data dictionary with loaded matrices
        
    Returns:
        Complete simulation results
    """
    from .generators import (
        create_erdos_renyi_model,
        create_generalized_er_model, 
        create_degree_preserving_model,
        create_cell_type_preserving_model
    )
    
    # Map model names to functions
    model_functions = {
        'erdos_renyi': create_erdos_renyi_model,
        'generalized_er': create_generalized_er_model,
        'degree_preserving': create_degree_preserving_model,
        'cell_type_preserving': create_cell_type_preserving_model
    }
    
    # Build model configurations
    model_configs = {}
    
    for model_name, model_config in config['models'].items():
        if model_config.get('enabled', False):
            config_dict = {'func': model_functions[model_name]}
            config_dict.update(model_config)
            config_dict.pop('enabled', None)
            
            # Add cell type data if needed
            if model_name == 'cell_type_preserving':
                config_dict['cell_type_indices'] = data['cell_type_indices']
                config_dict['cell_types'] = data['cell_types']
            
            model_configs[model_name] = config_dict
    
    # Run simulations
    simulation_results = simulate_null_models(
        data['adjacency_binary'],
        model_configs,
        config['simulation']['n_iterations'],
        config['output']['verbose']
    )
    
    return simulation_results