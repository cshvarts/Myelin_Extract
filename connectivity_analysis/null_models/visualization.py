"""
Visualization functions for null model analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple, Optional
from .utils import calculate_network_stats


def plot_model_comparison(real_adjacency: np.ndarray, 
                         simulation_results: Dict[str, Dict[str, List]], 
                         comparison_results: Dict[str, Dict[str, Dict]],
                         metrics: List[str] = ['reciprocal_counts', 'mean_strengths'],
                         figsize: Tuple[int, int] = (16, 12),
                         save_path: str = None,
                         show: bool = True) -> plt.Figure:
    """
    Create comprehensive comparison plots for null models.
    
    Args:
        real_adjacency: Original adjacency matrix
        simulation_results: Results from simulate_null_models
        comparison_results: Results from compare_models_to_real
        metrics: List of metrics to plot
        figsize: Figure size
        save_path: Path to save figure (optional)
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    n_metrics = len(metrics)
    n_models = len(simulation_results)
    
    # Create figure with subplots: 2 columns per metric (distribution + effect size)
    fig, axes = plt.subplots(n_metrics, 3, figsize=figsize)
    if n_metrics == 1:
        axes = axes.reshape(1, -1)
    
    # Colors for different models
    colors = plt.cm.Set3(np.linspace(0, 1, n_models))
    model_names = list(simulation_results.keys())
    
    # Real network stats
    real_stats = calculate_network_stats(real_adjacency)
    real_values = {
        'reciprocal_counts': real_stats['n_reciprocal'],
        'mean_strengths': real_stats['mean_strength'],
        'n_edges': real_stats['n_edges'],
        'connection_probabilities': real_stats['connection_probability']
    }
    
    metric_labels = {
        'reciprocal_counts': 'Reciprocal Connections',
        'mean_strengths': 'Mean Connection Strength',
        'n_edges': 'Number of Edges',
        'connection_probabilities': 'Connection Probability'
    }
    
    for i, metric in enumerate(metrics):
        # Plot 1: Null distributions with real value
        ax_dist = axes[i, 0]
        
        for j, model_name in enumerate(model_names):
            values = simulation_results[model_name][metric]
            z_score = comparison_results[model_name][metric]['z_score']
            
            ax_dist.hist(values, bins=30, alpha=0.6, density=True,
                        label=f"{model_name.replace('_', ' ')} (z={z_score:.2f})",
                        color=colors[j])
        
        real_val = real_values[metric]
        ax_dist.axvline(real_val, color='red', linestyle='--', linewidth=2,
                       label=f'Real network')
        
        ax_dist.set_xlabel(metric_labels.get(metric, metric))
        ax_dist.set_ylabel('Density')
        ax_dist.set_title(f'Null Distributions: {metric_labels.get(metric, metric)}')
        ax_dist.legend()
        ax_dist.grid(True, alpha=0.3)
        
        # Plot 2: Effect sizes (Cohen's d)
        ax_effect = axes[i, 1]
        effect_sizes = [comparison_results[model][metric]['cohens_d'] for model in model_names]
        
        bars = ax_effect.barh(range(len(model_names)), effect_sizes,
                             color=colors, alpha=0.7)
        ax_effect.axvline(0, color='black', linestyle='-', alpha=0.5)
        ax_effect.axvline(0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect')
        ax_effect.axvline(0.8, color='gray', linestyle='--', alpha=0.5, label='Large effect')
        ax_effect.axvline(-0.2, color='gray', linestyle='--', alpha=0.5)
        ax_effect.axvline(-0.8, color='gray', linestyle='--', alpha=0.5)
        
        ax_effect.set_yticks(range(len(model_names)))
        ax_effect.set_yticklabels([m.replace('_', ' ') for m in model_names])
        ax_effect.set_xlabel("Effect Size (Cohen's d)")
        ax_effect.set_title(f'Effect Sizes: {metric_labels.get(metric, metric)}')
        ax_effect.legend()
        ax_effect.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, effect in zip(bars, effect_sizes):
            width = bar.get_width()
            ax_effect.text(width + 0.05 if width >= 0 else width - 0.05, 
                          bar.get_y() + bar.get_height()/2.,
                          f'{effect:.2f}', ha='left' if width >= 0 else 'right', 
                          va='center', fontsize=9)
        
        # Plot 3: P-values
        ax_pval = axes[i, 2]
        p_vals = [comparison_results[model][metric]['p_value_two_tailed'] for model in model_names]
        
        # Use log scale for p-values
        log_p_vals = [-np.log10(max(p, 1e-10)) for p in p_vals]  # Avoid log(0)
        
        bars = ax_pval.bar(range(len(model_names)), log_p_vals,
                          color=colors, alpha=0.7)
        ax_pval.axhline(-np.log10(0.05), color='red', linestyle='--', alpha=0.7, label='α = 0.05')
        ax_pval.axhline(-np.log10(0.01), color='orange', linestyle='--', alpha=0.7, label='α = 0.01')
        ax_pval.axhline(-np.log10(0.001), color='darkred', linestyle='--', alpha=0.7, label='α = 0.001')
        
        ax_pval.set_xticks(range(len(model_names)))
        ax_pval.set_xticklabels([m.replace('_', ' ') for m in model_names], rotation=45)
        ax_pval.set_ylabel('-log₁₀(p-value)')
        ax_pval.set_title(f'Significance: {metric_labels.get(metric, metric)}')
        ax_pval.legend()
        ax_pval.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, p_val in zip(bars, p_vals):
            height = bar.get_height()
            ax_pval.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{p_val:.1e}' if p_val < 0.001 else f'{p_val:.3f}', 
                        ha='center', va='bottom', fontsize=9, rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
        
    return fig


def plot_network_properties(adjacency_matrix: np.ndarray, 
                           cell_type_indices: Dict[str, Tuple[int, int]] = None,
                           cell_types: List[str] = None,
                           figsize: Tuple[int, int] = (20, 15),
                           save_path: str = None,
                           show: bool = True) -> plt.Figure:
    """
    Create comprehensive visualization of network properties.
    
    Args:
        adjacency_matrix: Network adjacency matrix
        cell_type_indices: Dictionary mapping cell types to indices
        cell_types: List of cell type names
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    from .utils import get_reciprocal_matrix
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    # 1. Degree distribution
    ax1 = axes[0, 0]
    in_degrees = np.sum(adjacency_matrix > 0, axis=0)
    out_degrees = np.sum(adjacency_matrix > 0, axis=1)
    
    ax1.hist(in_degrees, bins=50, alpha=0.7, label='In-degree', density=True)
    ax1.hist(out_degrees, bins=50, alpha=0.7, label='Out-degree', density=True)
    ax1.set_xlabel('Degree')
    ax1.set_ylabel('Density')
    ax1.set_title('Degree Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Weight distribution
    ax2 = axes[0, 1]
    weights = adjacency_matrix[adjacency_matrix > 0]
    ax2.hist(weights, bins=50, density=True, alpha=0.7)
    ax2.set_xlabel('Connection Strength')
    ax2.set_ylabel('Density')
    ax2.set_title('Connection Weight Distribution')
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 3. Reciprocity analysis
    ax3 = axes[0, 2]
    reciprocal_matrix = get_reciprocal_matrix(adjacency_matrix)
    
    total_edges = np.sum(adjacency_matrix > 0)
    reciprocal_edges = np.sum(reciprocal_matrix)
    unidirectional_edges = total_edges - reciprocal_edges
    
    labels = ['Reciprocal', 'Unidirectional']
    sizes = [reciprocal_edges, unidirectional_edges]
    colors = ['lightcoral', 'lightskyblue']
    
    ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax3.set_title('Edge Reciprocity')
    
    # 4. Connection matrix visualization (if cell types provided)
    ax4 = axes[1, 0]
    if cell_type_indices and cell_types:
        # Create cell type connection matrix
        n_types = len(cell_types)
        conn_matrix = np.zeros((n_types, n_types))
        
        for i, ct1 in enumerate(cell_types):
            start1, end1 = cell_type_indices[ct1]
            for j, ct2 in enumerate(cell_types):
                start2, end2 = cell_type_indices[ct2]
                block = adjacency_matrix[start1:end1, start2:end2]
                conn_matrix[i, j] = np.mean(block[block > 0]) if np.any(block > 0) else 0
        
        im = ax4.imshow(conn_matrix, cmap='viridis', aspect='auto')
        ax4.set_xticks(range(n_types))
        ax4.set_yticks(range(n_types))
        ax4.set_xticklabels(cell_types, rotation=45)
        ax4.set_yticklabels(cell_types)
        ax4.set_title('Cell Type Connection Matrix\n(Mean Connection Strength)')
        plt.colorbar(im, ax=ax4)
    else:
        ax4.text(0.5, 0.5, 'Cell type data\nnot provided', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=14)
        ax4.set_title('Cell Type Connection Matrix')
    
    # 5. Strength vs degree correlation
    ax5 = axes[1, 1]
    node_strengths = np.sum(adjacency_matrix, axis=1)  # Out-strength
    node_degrees = np.sum(adjacency_matrix > 0, axis=1)  # Out-degree
    
    # Remove zeros for better visualization
    nonzero_mask = (node_degrees > 0) & (node_strengths > 0)
    if np.any(nonzero_mask):
        ax5.scatter(node_degrees[nonzero_mask], node_strengths[nonzero_mask], 
                   alpha=0.6, s=20)
        ax5.set_xlabel('Out-degree')
        ax5.set_ylabel('Out-strength')
        ax5.set_title('Strength vs Degree Correlation')
        ax5.grid(True, alpha=0.3)
        
        # Add correlation coefficient
        if len(node_degrees[nonzero_mask]) > 1:
            corr = np.corrcoef(node_degrees[nonzero_mask], node_strengths[nonzero_mask])[0, 1]
            ax5.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax5.transAxes,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    else:
        ax5.text(0.5, 0.5, 'No connections found', 
                ha='center', va='center', transform=ax5.transAxes)
    
    # 6. Network statistics summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    stats = calculate_network_stats(adjacency_matrix)
    
    stats_text = f"""Network Statistics:
    
Nodes: {stats['n_nodes']:,}
Edges: {stats['n_edges']:,}
Reciprocal edges: {stats['n_reciprocal']:,}
Connection probability: {stats['connection_probability']:.6f}
Mean strength: {stats['mean_strength']:.3f}

Reciprocity rate: {stats['n_reciprocal']/stats['n_edges']*100:.1f}%
Density: {stats['n_edges']/(stats['n_nodes']*(stats['n_nodes']-1))*100:.3f}%
"""
    
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
        
    return fig


def plot_model_rankings(comparison_results: Dict[str, Dict[str, Dict]],
                       metrics: List[str] = ['reciprocal_counts', 'mean_strengths'],
                       figsize: Tuple[int, int] = (12, 8),
                       save_path: str = None,
                       show: bool = True) -> plt.Figure:
    """
    Plot model rankings based on different metrics.
    
    Args:
        comparison_results: Results from compare_models_to_real
        metrics: List of metrics to rank by
        figsize: Figure size
        save_path: Path to save figure
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
    """
    from .analysis import calculate_model_rankings
    
    fig, ax = plt.subplots(figsize=figsize)
    
    model_names = list(comparison_results.keys())
    n_models = len(model_names)
    n_metrics = len(metrics)
    
    # Calculate rankings for each metric
    all_rankings = {}
    for metric in metrics:
        rankings = calculate_model_rankings(comparison_results, metric)
        all_rankings[metric] = rankings
    
    # Create bar chart
    x = np.arange(n_models)
    width = 0.35
    colors = plt.cm.Set2(np.linspace(0, 1, n_metrics))
    
    for i, metric in enumerate(metrics):
        rankings = all_rankings[metric]
        z_scores = [rankings[model] for model in model_names]
        
        offset = (i - n_metrics/2 + 0.5) * width / n_metrics
        bars = ax.bar(x + offset, z_scores, width/n_metrics, 
                     label=metric.replace('_', ' ').title(),
                     color=colors[i], alpha=0.8)
        
        # Add value labels
        for bar, score in zip(bars, z_scores):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{score:.2f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Null Models')
    ax.set_ylabel('|Z-score| (lower = better fit)')
    ax.set_title('Model Performance Rankings\n(Distance from Real Network)')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ') for m in model_names], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
        
    return fig