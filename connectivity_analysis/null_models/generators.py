"""
Null model generators for network analysis.
"""

import numpy as np
import random
from typing import Dict, List, Any
from .utils import get_edge_weights


def create_erdos_renyi_model(adjacency_matrix: np.ndarray, preserve_weights: bool = True, **kwargs) -> np.ndarray:
    """
    Create standard Erdős-Rényi null model.
    
    Args:
        adjacency_matrix: Original weighted adjacency matrix
        preserve_weights: If True, randomly reassign actual weights; if False, use binary
        
    Returns:
        Null adjacency matrix
    """
    n_nodes = adjacency_matrix.shape[0]
    
    # Calculate connection probability
    total_connections = np.sum(adjacency_matrix > 0)
    connection_probability = total_connections / (n_nodes * (n_nodes - 1))
    
    # Create random connectivity matrix
    random_matrix = np.random.random((n_nodes, n_nodes))
    null_binary = (random_matrix < connection_probability).astype(int)
    
    # Remove self-connections
    np.fill_diagonal(null_binary, 0)
    
    if preserve_weights:
        # Randomly redistribute the original weights
        null_weighted = null_binary.copy().astype(float)
        connection_indices = np.where(null_binary > 0)
        n_connections = len(connection_indices[0])
        
        if n_connections > 0:
            all_weights = get_edge_weights(adjacency_matrix)
            random_weights = np.random.choice(all_weights, size=n_connections, replace=True)
            null_weighted[connection_indices] = random_weights
    else:
        null_weighted = null_binary.astype(float)
    
    return null_weighted


def create_generalized_er_model(adjacency_matrix: np.ndarray, preserve_weights: bool = True, **kwargs) -> np.ndarray:
    """
    Create generalized ER model that preserves reciprocal edge count.
    Based on approach with p^uni and p^bi parameters.
    
    Args:
        adjacency_matrix: Original weighted adjacency matrix
        preserve_weights: If True, randomly reassign actual weights; if False, use binary
        
    Returns:
        Null adjacency matrix
    """
    n_nodes = adjacency_matrix.shape[0]
    
    # Calculate edge statistics from real network
    binary_adj = (adjacency_matrix > 0).astype(int)
    reciprocal_matrix = binary_adj & binary_adj.T
    
    # Count unidirectional and bidirectional edges
    total_edges = np.sum(binary_adj)
    reciprocal_edges = np.sum(reciprocal_matrix)
    unidirectional_edges = total_edges - reciprocal_edges
    
    # Calculate possible edge pairs (excluding diagonal)
    total_possible_pairs = n_nodes * (n_nodes - 1) // 2
    
    # Calculate probabilities
    p_bi = reciprocal_edges / (2 * total_possible_pairs)  # bidirectional probability
    p_uni = unidirectional_edges / (n_nodes * (n_nodes - 1))  # unidirectional probability
    
    # Generate null network
    null_adj = np.zeros((n_nodes, n_nodes))
    
    # For each pair of nodes (i,j) where i < j
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            rand = np.random.random()
            
            if rand < p_bi:
                # Create bidirectional connection
                null_adj[i, j] = 1
                null_adj[j, i] = 1
            elif rand < p_bi + 2 * p_uni:
                # Create unidirectional connection (randomly choose direction)
                if np.random.random() < 0.5:
                    null_adj[i, j] = 1
                else:
                    null_adj[j, i] = 1
    
    if preserve_weights:
        # Randomly redistribute the original weights
        connection_indices = np.where(null_adj > 0)
        n_connections = len(connection_indices[0])
        
        if n_connections > 0:
            all_weights = get_edge_weights(adjacency_matrix)
            random_weights = np.random.choice(all_weights, size=n_connections, replace=True)
            null_adj[connection_indices] = random_weights
    
    return null_adj


def create_degree_preserving_model(adjacency_matrix: np.ndarray, n_swaps: int = None, 
                                 preserve_weights: bool = True, **kwargs) -> np.ndarray:
    """
    Create degree-preserving null model using edge swapping (Maslov-Sneppén algorithm).
    Equivalent to Configuration Model (CFG).
    
    Args:
        adjacency_matrix: Original weighted adjacency matrix
        n_swaps: Number of edge swaps (default: 10x number of edges)
        preserve_weights: Whether to preserve edge weights
        
    Returns:
        Rewired adjacency matrix
    """
    adj = adjacency_matrix.copy()
    n_nodes = adj.shape[0]
    
    # Get all edges
    edges = list(zip(*np.where(adj > 0)))
    n_edges = len(edges)
    
    if n_swaps is None:
        n_swaps = 10 * n_edges
    
    successful_swaps = 0
    attempts = 0
    max_attempts = n_swaps * 10  # Prevent infinite loops
    
    while successful_swaps < n_swaps and attempts < max_attempts:
        attempts += 1
        
        # Randomly select two edges
        if len(edges) < 2:
            break
            
        edge_indices = random.sample(range(len(edges)), 2)
        edge1_idx, edge2_idx = edge_indices
        
        # Get the edges
        i, j = edges[edge1_idx]  # edge i->j
        k, l = edges[edge2_idx]  # edge k->l
        
        # Switch-and-hold: swap target endpoints
        # Check if we can swap: i->l and k->j (avoid self-loops and multiple edges)
        if (i != l and k != j and i != k and j != l and
            adj[i, l] == 0 and adj[k, j] == 0):
            
            # Store weights if preserving them
            if preserve_weights:
                weight_ij = adj[i, j]
                weight_kl = adj[k, l]
            else:
                weight_ij = weight_kl = 1
            
            # Perform the swap
            adj[i, j] = 0
            adj[k, l] = 0
            adj[i, l] = weight_ij
            adj[k, j] = weight_kl
            
            # Update edges list
            edges[edge1_idx] = (i, l)
            edges[edge2_idx] = (k, j)
            
            successful_swaps += 1
    
    return adj


def create_cell_type_preserving_model(adjacency_matrix: np.ndarray, 
                                    cell_type_indices: Dict[str, tuple],
                                    cell_types: List[str],
                                    preserve_within_type: bool = True,
                                    preserve_weights: bool = True, 
                                    **kwargs) -> np.ndarray:
    """
    Create cell-type preserving null model.
    
    Args:
        adjacency_matrix: Original weighted adjacency matrix
        cell_type_indices: Dictionary mapping cell types to (start, end) indices
        cell_types: List of cell type names
        preserve_within_type: If True, preserve within-cell-type connectivity exactly
        preserve_weights: Whether to preserve edge weights
        
    Returns:
        Cell-type preserving adjacency matrix
    """
    adj = np.zeros_like(adjacency_matrix)
    
    for ct1 in cell_types:
        start1, end1 = cell_type_indices[ct1]
        
        for ct2 in cell_types:
            start2, end2 = cell_type_indices[ct2]
            
            # Get the block between these cell types
            original_block = adjacency_matrix[start1:end1, start2:end2]
            
            if preserve_within_type and ct1 == ct2:
                # Keep within-cell-type connectivity exactly as is
                adj[start1:end1, start2:end2] = original_block
            else:
                # Randomize between-cell-type connectivity
                nonzero_weights = original_block[original_block > 0]
                
                if len(nonzero_weights) > 0:
                    # Calculate connection probability for this block
                    block_size = original_block.size
                    n_connections = len(nonzero_weights)
                    connection_prob = n_connections / block_size
                    
                    # Create random connectivity pattern
                    random_pattern = np.random.random(original_block.shape) < connection_prob
                    
                    if preserve_weights:
                        # Randomly redistribute the original weights
                        random_weights = np.random.choice(nonzero_weights, 
                                                        size=np.sum(random_pattern), 
                                                        replace=True)
                        new_block = np.zeros_like(original_block, dtype=float)
                        new_block[random_pattern] = random_weights
                    else:
                        new_block = random_pattern.astype(float)
                    
                    adj[start1:end1, start2:end2] = new_block
    
    return adj