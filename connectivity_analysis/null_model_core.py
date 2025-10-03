"""
Core null model functions - minimal and focused.
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml

def erdos_renyi_null(adjacency_matrix, n_iterations=100, preserve_weights=True):
    """
    Generate Erdős-Rényi null model.
    
    Args:
        adjacency_matrix: Original adjacency matrix
        n_iterations: Number of null networks to generate  
        preserve_weights: Whether to preserve edge weights
        
    Returns:
        List of null network statistics
    """
    n_nodes = adjacency_matrix.shape[0]
    total_edges = np.sum(adjacency_matrix > 0)
    p_connect = total_edges / (n_nodes * (n_nodes - 1))
    
    results = {
        'reciprocal_counts': [],
        'edge_counts': [],
        'mean_weights': []
    }
    
    if preserve_weights:
        all_weights = adjacency_matrix[adjacency_matrix > 0]
    
    for i in range(n_iterations):
        # Generate random connectivity
        null_binary = (np.random.random((n_nodes, n_nodes)) < p_connect).astype(int)
        np.fill_diagonal(null_binary, 0)
        
        if preserve_weights and len(all_weights) > 0:
            # Assign random weights from original distribution
            null_weighted = null_binary.astype(float)
            edge_indices = np.where(null_binary > 0)
            n_edges = len(edge_indices[0])
            if n_edges > 0:
                random_weights = np.random.choice(all_weights, size=n_edges, replace=True)
                null_weighted[edge_indices] = random_weights
        else:
            null_weighted = null_binary.astype(float)
        
        # Calculate reciprocal connections
        reciprocal_matrix = null_binary & null_binary.T
        
        results['reciprocal_counts'].append(np.sum(reciprocal_matrix))
        results['edge_counts'].append(np.sum(null_weighted > 0))
        results['mean_weights'].append(np.mean(null_weighted[null_weighted > 0]) if np.any(null_weighted > 0) else 0)
    
    return results

def compare_to_null(real_value, null_values, metric_name="metric"):
    """Compare real value to null distribution."""
    null_array = np.array(null_values)
    null_mean = np.mean(null_array)
    null_std = np.std(null_array)
    
    # Calculate statistics
    z_score = (real_value - null_mean) / null_std if null_std > 0 else np.inf
    p_value = np.mean(null_array >= real_value)  # one-tailed
    
    # Effect size (Cohen's d)
    cohens_d = z_score
    
    return {
        'real_value': real_value,
        'null_mean': null_mean,
        'null_std': null_std,
        'z_score': z_score,
        'p_value': p_value,
        'cohens_d': cohens_d
    }

def plot_comparison(real_value, null_values, title="Null Model Comparison", 
                   xlabel="Value", save_path=None):
    """Plot comparison between real and null values."""
    
    stats = compare_to_null(real_value, null_values)
    
    plt.figure(figsize=(10, 6))
    plt.hist(null_values, bins=30, alpha=0.7, density=True, color='skyblue',
             label=f'Null model (μ={stats["null_mean"]:.1f})')
    plt.axvline(real_value, color='red', linestyle='--', linewidth=2,
                label=f'Real network ({real_value})')
    
    plt.xlabel(xlabel)
    plt.ylabel('Density')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Z-score: {stats["z_score"]:.2f}\nP-value: {stats["p_value"]:.1e}'
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
             facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return stats

def run_analysis(config_path="config.yaml"):
    """Run complete null model analysis."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load data
    print("Loading data...")
    adjacency = np.load(config['data']['adjacency_binary'])
    
    # Calculate real network statistics
    n_nodes = adjacency.shape[0]
    n_edges = np.sum(adjacency > 0)
    binary_adj = (adjacency > 0).astype(int)
    reciprocal_matrix = binary_adj & binary_adj.T
    n_reciprocal = np.sum(reciprocal_matrix)
    
    print(f"\nReal Network:")
    print(f"  Nodes: {n_nodes:,}")
    print(f"  Edges: {n_edges:,}")
    print(f"  Reciprocal edges: {n_reciprocal:,}")
    print(f"  Reciprocity rate: {n_reciprocal/n_edges*100:.1f}%")
    
    # Run null model
    print(f"\nGenerating null model ({config['simulation']['n_iterations']} iterations)...")
    null_results = erdos_renyi_null(
        adjacency, 
        n_iterations=config['simulation']['n_iterations'],
        preserve_weights=config['models']['erdos_renyi']['preserve_weights']
    )
    
    # Compare reciprocal connections
    print("\nAnalyzing reciprocal connections...")
    recip_stats = compare_to_null(n_reciprocal, null_results['reciprocal_counts'])
    
    print(f"Results:")
    print(f"  Real: {recip_stats['real_value']}")
    print(f"  Null: {recip_stats['null_mean']:.1f} ± {recip_stats['null_std']:.1f}")
    print(f"  Z-score: {recip_stats['z_score']:.2f}")
    print(f"  P-value: {recip_stats['p_value']:.6f}")
    
    # Plot
    if config['output']['save_plots']:
        plot_comparison(
            n_reciprocal, 
            null_results['reciprocal_counts'],
            title="Reciprocal Connections: Real vs Erdős-Rényi Null",
            xlabel="Number of Reciprocal Connections",
            save_path="null_model_comparison.png"
        )
    
    return {
        'real_stats': {'n_reciprocal': n_reciprocal, 'n_edges': n_edges},
        'null_results': null_results,
        'comparison': recip_stats
    }