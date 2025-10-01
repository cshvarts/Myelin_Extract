"""
Utility functions for network analysis and null model generation.
"""

import numpy as np
import pickle
from typing import Dict, List, Tuple, Any


def load_data(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load network data from files specified in config.
    
    Args:
        config: Configuration dictionary with data file paths
        
    Returns:
        Dictionary containing loaded data
    """
    data = {}
    
    # Load adjacency matrices
    data['adjacency_binary'] = np.load(config['data']['adjacency_binary'])
    data['adjacency_weighted'] = np.load(config['data']['adjacency_weighted'])
    data['reciprocal_matrix'] = np.load(config['data']['reciprocal_matrix'])
    
    # Load cell type information
    with open(config['data']['cell_type_indices'], 'rb') as f:
        data['cell_type_indices'] = pickle.load(f)
    with open(config['data']['cell_types'], 'rb') as f:
        data['cell_types'] = pickle.load(f)
        
    return data


def get_reciprocal_matrix(adj_matrix: np.ndarray, threshold: float = 0) -> np.ndarray:
    """
    Create reciprocal connection matrix from adjacency matrix.
    
    Args:
        adj_matrix: Adjacency matrix
        threshold: Threshold for considering a connection
        
    Returns:
        Binary matrix indicating reciprocal connections
    """
    binary_adj = (adj_matrix > threshold).astype(int)
    reciprocal = binary_adj & binary_adj.T
    return reciprocal


def calculate_network_stats(adjacency_matrix: np.ndarray) -> Dict[str, float]:
    """
    Calculate basic network statistics.
    
    Args:
        adjacency_matrix: Network adjacency matrix
        
    Returns:
        Dictionary with network statistics
    """
    reciprocal_matrix = get_reciprocal_matrix(adjacency_matrix)
    n_nodes = adjacency_matrix.shape[0]
    
    stats = {
        'n_nodes': n_nodes,
        'n_edges': np.sum(adjacency_matrix > 0),
        'n_reciprocal': np.sum(reciprocal_matrix),
        'mean_strength': np.mean(adjacency_matrix[adjacency_matrix > 0]) if np.any(adjacency_matrix > 0) else 0,
        'connection_probability': np.sum(adjacency_matrix > 0) / (n_nodes * (n_nodes - 1))
    }
    return stats


def get_edge_weights(adjacency_matrix: np.ndarray) -> np.ndarray:
    """
    Extract all non-zero edge weights from adjacency matrix.
    
    Args:
        adjacency_matrix: Network adjacency matrix
        
    Returns:
        Array of edge weights
    """
    return adjacency_matrix[adjacency_matrix > 0]


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_sections = ['data', 'simulation', 'models', 'analysis']
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    required_data_files = ['adjacency_binary', 'adjacency_weighted', 'reciprocal_matrix']
    for file_key in required_data_files:
        if file_key not in config['data']:
            raise ValueError(f"Missing required data file: {file_key}")
            
    if config['simulation']['n_iterations'] <= 0:
        raise ValueError("n_iterations must be positive")


def setup_random_seed(seed: int = None) -> None:
    """
    Set up random seed for reproducibility.
    
    Args:
        seed: Random seed (None for random seed)
    """
    if seed is not None:
        np.random.seed(seed)