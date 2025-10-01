"""
Analysis functions for comparing null models to real networks.
"""

import numpy as np
from typing import Dict, List, Any
from .utils import calculate_network_stats


def compare_models_to_real(real_adjacency: np.ndarray, 
                          simulation_results: Dict[str, Dict[str, List]], 
                          metrics: List[str] = ['reciprocal_counts', 'mean_strengths']) -> Dict[str, Dict[str, Dict]]:
    """
    Compare simulation results to real network.
    
    Args:
        real_adjacency: Original adjacency matrix
        simulation_results: Results from simulate_null_models
        metrics: List of metrics to compare
        
    Returns:
        Dictionary with p-values and effect sizes for each model and metric
    """
    real_stats = calculate_network_stats(real_adjacency)
    
    # Map metrics to real values
    real_values = {
        'reciprocal_counts': real_stats['n_reciprocal'],
        'mean_strengths': real_stats['mean_strength'],
        'n_edges': real_stats['n_edges'],
        'connection_probabilities': real_stats['connection_probability']
    }
    
    comparison_results = {}
    
    for model_name, model_results in simulation_results.items():
        model_comparison = {}
        
        for metric in metrics:
            if metric in model_results and metric in real_values:
                null_values = np.array(model_results[metric])
                real_value = real_values[metric]
                
                # Calculate p-values
                p_value_greater = np.mean(null_values >= real_value)
                p_value_lesser = np.mean(null_values <= real_value)
                p_value_two_tailed = 2 * min(p_value_greater, p_value_lesser)
                
                # Calculate effect size (Cohen's d)
                null_mean = np.mean(null_values)
                null_std = np.std(null_values)
                cohens_d = (real_value - null_mean) / null_std if null_std > 0 else np.inf
                
                # Calculate z-score
                z_score = (real_value - null_mean) / null_std if null_std > 0 else np.inf
                
                model_comparison[metric] = {
                    'real_value': real_value,
                    'null_mean': null_mean,
                    'null_std': null_std,
                    'null_median': np.median(null_values),
                    'null_min': np.min(null_values),
                    'null_max': np.max(null_values),
                    'p_value_greater': p_value_greater,
                    'p_value_lesser': p_value_lesser,
                    'p_value_two_tailed': p_value_two_tailed,
                    'cohens_d': cohens_d,
                    'z_score': z_score,
                    'null_values': null_values.tolist()  # For detailed analysis
                }
        
        comparison_results[model_name] = model_comparison
    
    return comparison_results


def print_comparison_summary(comparison_results: Dict[str, Dict[str, Dict]], 
                           significance_levels: List[float] = [0.001, 0.01, 0.05]) -> None:
    """
    Print comprehensive summary of null model comparisons.
    
    Args:
        comparison_results: Results from compare_models_to_real
        significance_levels: List of significance levels for interpretation
    """
    print("\n" + "="*80)
    print("NULL MODEL COMPARISON SUMMARY")
    print("="*80)
    
    for model_name, model_results in comparison_results.items():
        print(f"\n{model_name.replace('_', ' ').title()}")
        print("-" * 50)
        
        for metric, stats in model_results.items():
            real_val = stats['real_value']
            null_mean = stats['null_mean']
            null_std = stats['null_std']
            null_median = stats['null_median']
            p_val_greater = stats['p_value_greater']
            p_val_two_tailed = stats['p_value_two_tailed']
            cohens_d = stats['cohens_d']
            z_score = stats['z_score']
            
            print(f"\n  {metric.replace('_', ' ').title()}:")
            print(f"    Real network: {real_val:.3f}")
            print(f"    Null model mean: {null_mean:.3f} ± {null_std:.3f}")
            print(f"    Null model median: {null_median:.3f}")
            print(f"    Z-score: {z_score:.3f}")
            print(f"    Effect size (Cohen's d): {cohens_d:.3f}")
            print(f"    P-value (greater): {p_val_greater:.6f}")
            print(f"    P-value (two-tailed): {p_val_two_tailed:.6f}")
            
            # Determine significance
            if p_val_two_tailed < 0.001:
                significance = "*** (p < 0.001)"
            elif p_val_two_tailed < 0.01:
                significance = "** (p < 0.01)"
            elif p_val_two_tailed < 0.05:
                significance = "* (p < 0.05)"
            else:
                significance = "ns (not significant)"
            
            print(f"    Significance: {significance}")
            
            # Interpret effect size
            if abs(cohens_d) < 0.2:
                effect_interpretation = "negligible"
            elif abs(cohens_d) < 0.5:
                effect_interpretation = "small"
            elif abs(cohens_d) < 0.8:
                effect_interpretation = "medium"
            else:
                effect_interpretation = "large"
                
            direction = "higher" if cohens_d > 0 else "lower"
            print(f"    Effect interpretation: {effect_interpretation} effect ({direction} than null)")


def calculate_model_rankings(comparison_results: Dict[str, Dict[str, Dict]], 
                           metric: str = 'reciprocal_counts') -> Dict[str, float]:
    """
    Rank models by how well they explain the real network metric.
    
    Args:
        comparison_results: Results from compare_models_to_real
        metric: Metric to use for ranking
        
    Returns:
        Dictionary with model rankings (lower rank = better fit)
    """
    rankings = {}
    
    for model_name, model_results in comparison_results.items():
        if metric in model_results:
            # Use absolute z-score as ranking criterion (closer to 0 = better fit)
            z_score = abs(model_results[metric]['z_score'])
            rankings[model_name] = z_score
    
    # Sort by z-score (ascending - lower is better)
    sorted_rankings = dict(sorted(rankings.items(), key=lambda x: x[1]))
    
    return sorted_rankings


def get_best_fitting_model(comparison_results: Dict[str, Dict[str, Dict]], 
                          metric: str = 'reciprocal_counts') -> str:
    """
    Get the best fitting model for a given metric.
    
    Args:
        comparison_results: Results from compare_models_to_real
        metric: Metric to evaluate
        
    Returns:
        Name of best fitting model
    """
    rankings = calculate_model_rankings(comparison_results, metric)
    return list(rankings.keys())[0] if rankings else None