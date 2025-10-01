"""
Reciprocal Connectivity Analysis for MICrONS Dataset

Analyzes reciprocal connections in the highly proofread V1 neural column.
Key analyses:
- Build connectivity matrices from synaptic data
- Identify reciprocal connections
- Compare reciprocal vs non-reciprocal connection strengths
- Cell-type specific connectivity patterns
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from caveclient import CAVEclient
from typing import Optional, Tuple
import pickle


def load_microns_data(client: CAVEclient, version: int = 1507) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load connectivity data from MICrONS dataset.

    Parameters:
    -----------
    client : CAVEclient
        Initialized CAVE client for minnie65_public
    version : int
        Materialization version

    Returns:
    --------
    tuple of (synapses_df, neurons_df, cell_types_df)
    """
    client.materialize.version = version

    print(f"Loading data from MICrONS (version {version})...")

    # Load proofreading table
    proof_df = client.materialize.query_table('proofreading_status_public_release')

    # Filter for highly proofread neurons (status_axon==True)
    proof_df = proof_df[proof_df['status_axon'] == True].copy()
    print(f"  Found {len(proof_df)} highly proofread neurons")

    # Load neuron metadata
    neurons_df = client.materialize.query_table('nucleus_detection_v0')

    # Merge
    proof_neurons_df = pd.merge(proof_df, neurons_df, on='pt_root_id', how='inner')
    print(f"  Merged: {len(proof_neurons_df)} neurons with metadata")

    # Load synapse table
    print("  Loading synapses (this may take a minute)...")
    synapses_df = client.materialize.query_table('synapses_pni_2')

    # Filter to only synapses among proofread neurons
    proof_ids = set(proof_neurons_df['pt_root_id'].values)
    syn_proof_only_df = synapses_df[
        synapses_df['pre_pt_root_id'].isin(proof_ids) &
        synapses_df['post_pt_root_id'].isin(proof_ids)
    ].copy()

    print(f"  Filtered to {len(syn_proof_only_df)} synapses among proofread neurons")

    # Load cell types
    cell_types_df = client.materialize.query_table('allen_v1_column_types_slanted_ref')

    return syn_proof_only_df, proof_neurons_df, cell_types_df


def build_connectivity_matrix(synapses_df: pd.DataFrame,
                               neurons_df: pd.DataFrame,
                               weighted: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build connectivity matrix from synapse data.

    Parameters:
    -----------
    synapses_df : pd.DataFrame
        Synapse table with pre_pt_root_id, post_pt_root_id, size
    neurons_df : pd.DataFrame
        Neuron table with pt_root_id
    weighted : bool
        If True, use synapse counts/sizes. If False, binary connectivity.

    Returns:
    --------
    tuple of (adjacency_matrix, neuron_ids)
        adjacency_matrix[i, j] = connection strength from neuron i to neuron j
        neuron_ids = array of pt_root_ids corresponding to matrix indices
    """
    # Create neuron ID to index mapping
    neuron_ids = np.sort(neurons_df['pt_root_id'].unique())
    id_to_idx = {nid: idx for idx, nid in enumerate(neuron_ids)}
    n_neurons = len(neuron_ids)

    print(f"Building {n_neurons}x{n_neurons} connectivity matrix...")

    # Aggregate synapse counts/weights per connection
    if weighted:
        syn_weights = synapses_df.groupby(['pre_pt_root_id', 'post_pt_root_id']).agg({
            'size': 'sum'  # Total synaptic strength
        }).reset_index()
    else:
        syn_weights = synapses_df.groupby(['pre_pt_root_id', 'post_pt_root_id']).size().reset_index(name='size')

    # Build matrix
    adjacency_matrix = np.zeros((n_neurons, n_neurons))

    for _, row in syn_weights.iterrows():
        pre_idx = id_to_idx.get(row['pre_pt_root_id'])
        post_idx = id_to_idx.get(row['post_pt_root_id'])

        if pre_idx is not None and post_idx is not None:
            adjacency_matrix[pre_idx, post_idx] = row['size']

    print(f"  Total edges: {np.sum(adjacency_matrix > 0)}")
    print(f"  Connection density: {np.sum(adjacency_matrix > 0) / (n_neurons * (n_neurons - 1)) * 100:.2f}%")

    return adjacency_matrix, neuron_ids


def identify_reciprocal_connections(adjacency_matrix: np.ndarray) -> np.ndarray:
    """
    Identify reciprocal connections in adjacency matrix.

    Parameters:
    -----------
    adjacency_matrix : np.ndarray
        NxN connectivity matrix

    Returns:
    --------
    reciprocal_matrix : np.ndarray
        Binary matrix where [i,j]=1 if both i->j and j->i connections exist
    """
    binary_adj = (adjacency_matrix > 0).astype(int)
    reciprocal_matrix = binary_adj & binary_adj.T

    n_reciprocal = np.sum(reciprocal_matrix) // 2  # Divide by 2 since symmetric
    n_edges = np.sum(binary_adj)

    print(f"\nReciprocal connections:")
    print(f"  Total edges: {n_edges}")
    print(f"  Reciprocal pairs: {n_reciprocal}")
    print(f"  Reciprocity rate: {n_reciprocal * 2 / n_edges * 100:.1f}%")

    return reciprocal_matrix


def compare_reciprocal_vs_unidirectional_strength(adjacency_matrix: np.ndarray,
                                                   reciprocal_matrix: np.ndarray) -> pd.DataFrame:
    """
    Compare connection strengths between reciprocal and unidirectional connections.

    Parameters:
    -----------
    adjacency_matrix : np.ndarray
        Weighted connectivity matrix
    reciprocal_matrix : np.ndarray
        Binary reciprocal connection matrix

    Returns:
    --------
    results_df : pd.DataFrame
        Statistics comparing reciprocal vs non-reciprocal connections
    """
    binary_adj = (adjacency_matrix > 0).astype(int)

    # Get weights for reciprocal connections
    recip_mask = reciprocal_matrix.astype(bool)
    recip_weights = adjacency_matrix[recip_mask]

    # Get weights for non-reciprocal (unidirectional) connections
    non_recip_mask = (binary_adj > 0) & (~recip_mask)
    non_recip_weights = adjacency_matrix[non_recip_mask]

    results = {
        'connection_type': ['Reciprocal', 'Unidirectional'],
        'n_connections': [len(recip_weights), len(non_recip_weights)],
        'mean_strength': [np.mean(recip_weights), np.mean(non_recip_weights)],
        'median_strength': [np.median(recip_weights), np.median(non_recip_weights)],
        'std_strength': [np.std(recip_weights), np.std(non_recip_weights)]
    }

    results_df = pd.DataFrame(results)

    print("\nConnection strength comparison:")
    print(results_df.to_string(index=False))

    # Statistical test
    from scipy import stats
    t_stat, p_val = stats.ttest_ind(recip_weights, non_recip_weights)
    print(f"\nT-test: t={t_stat:.3f}, p={p_val:.2e}")

    return results_df


def plot_reciprocal_connections(adjacency_matrix: np.ndarray,
                                reciprocal_matrix: np.ndarray,
                                neuron_ids: np.ndarray,
                                cell_types_df: Optional[pd.DataFrame] = None,
                                save_path: Optional[str] = None):
    """
    Visualize reciprocal connections.

    Parameters:
    -----------
    adjacency_matrix : np.ndarray
        Connectivity matrix
    reciprocal_matrix : np.ndarray
        Binary reciprocal connection matrix
    neuron_ids : np.ndarray
        Array of pt_root_ids
    cell_types_df : pd.DataFrame, optional
        Cell type annotations
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Full adjacency matrix
    ax = axes[0]
    binary_adj = (adjacency_matrix > 0).astype(int)
    im = ax.imshow(binary_adj, cmap='binary', aspect='auto')
    ax.set_title(f'Full Connectivity Matrix\n({np.sum(binary_adj)} edges)', fontsize=12)
    ax.set_xlabel('Post-synaptic Neuron')
    ax.set_ylabel('Pre-synaptic Neuron')

    # Plot 2: Reciprocal connections only
    ax = axes[1]
    im = ax.imshow(reciprocal_matrix, cmap='Reds', aspect='auto')
    n_recip = np.sum(reciprocal_matrix) // 2
    ax.set_title(f'Reciprocal Connections Only\n({n_recip} pairs)', fontsize=12)
    ax.set_xlabel('Neuron')
    ax.set_ylabel('Neuron')

    # Plot 3: Connection strength distribution
    ax = axes[2]
    recip_weights = adjacency_matrix[reciprocal_matrix.astype(bool)]
    non_recip_mask = (binary_adj > 0) & (~reciprocal_matrix.astype(bool))
    non_recip_weights = adjacency_matrix[non_recip_mask]

    ax.hist(non_recip_weights, bins=50, alpha=0.6, label='Unidirectional', density=True)
    ax.hist(recip_weights, bins=50, alpha=0.6, label='Reciprocal', density=True)
    ax.set_xlabel('Connection Strength')
    ax.set_ylabel('Density')
    ax.set_title('Connection Strength Distribution')
    ax.legend()
    ax.set_yscale('log')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to {save_path}")

    plt.show()


def save_results(adjacency_binary: np.ndarray,
                adjacency_weighted: np.ndarray,
                reciprocal_matrix: np.ndarray,
                neuron_ids: np.ndarray,
                output_dir: str = "."):
    """
    Save connectivity analysis results.

    Parameters:
    -----------
    adjacency_binary : np.ndarray
        Binary adjacency matrix
    adjacency_weighted : np.ndarray
        Weighted adjacency matrix
    reciprocal_matrix : np.ndarray
        Reciprocal connection matrix
    neuron_ids : np.ndarray
        Neuron IDs
    output_dir : str
        Directory to save results
    """
    import os

    print(f"\nSaving results to {output_dir}...")

    np.save(os.path.join(output_dir, 'adjacency_binary.npy'), adjacency_binary)
    np.save(os.path.join(output_dir, 'adjacency_weighted.npy'), adjacency_weighted)
    np.save(os.path.join(output_dir, 'reciprocal_matrix.npy'), reciprocal_matrix)
    np.save(os.path.join(output_dir, 'neuron_ids.npy'), neuron_ids)

    print("  Saved:")
    print(f"    - adjacency_binary.npy ({adjacency_binary.shape})")
    print(f"    - adjacency_weighted.npy ({adjacency_weighted.shape})")
    print(f"    - reciprocal_matrix.npy ({reciprocal_matrix.shape})")
    print(f"    - neuron_ids.npy ({len(neuron_ids)} neurons)")


def run_full_analysis(client: Optional[CAVEclient] = None,
                     version: int = 1507,
                     output_dir: str = "connectivity_results",
                     save_figures: bool = True,
                     save_data: bool = True) -> dict:
    """
    Run complete reciprocal connectivity analysis.

    Parameters:
    -----------
    client : CAVEclient, optional
        If None, will create new client for minnie65_public
    version : int
        Materialization version
    output_dir : str
        Directory to save outputs
    save_figures : bool
        Whether to save figures
    save_data : bool
        Whether to save matrices

    Returns:
    --------
    results : dict
        Dictionary containing all analysis results
    """
    import os

    if client is None:
        print("Initializing CAVE client...")
        client = CAVEclient('minnie65_public')

    # Create output directory
    if save_figures or save_data:
        os.makedirs(output_dir, exist_ok=True)

    # Load data
    synapses_df, neurons_df, cell_types_df = load_microns_data(client, version)

    # Build matrices
    adjacency_weighted, neuron_ids = build_connectivity_matrix(synapses_df, neurons_df, weighted=True)
    adjacency_binary = (adjacency_weighted > 0).astype(int)

    # Identify reciprocal connections
    reciprocal_matrix = identify_reciprocal_connections(adjacency_weighted)

    # Compare strengths
    strength_comparison = compare_reciprocal_vs_unidirectional_strength(adjacency_weighted, reciprocal_matrix)

    # Visualize
    if save_figures:
        plot_path = os.path.join(output_dir, 'reciprocal_analysis.png')
    else:
        plot_path = None

    plot_reciprocal_connections(adjacency_weighted, reciprocal_matrix, neuron_ids,
                               cell_types_df, save_path=plot_path)

    # Save results
    if save_data:
        save_results(adjacency_binary, adjacency_weighted, reciprocal_matrix,
                    neuron_ids, output_dir)

    return {
        'adjacency_weighted': adjacency_weighted,
        'adjacency_binary': adjacency_binary,
        'reciprocal_matrix': reciprocal_matrix,
        'neuron_ids': neuron_ids,
        'strength_comparison': strength_comparison,
        'synapses_df': synapses_df,
        'neurons_df': neurons_df,
        'cell_types_df': cell_types_df
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze reciprocal connectivity in MICrONS dataset")
    parser.add_argument('--version', type=int, default=1507, help='Materialization version')
    parser.add_argument('--output-dir', type=str, default='connectivity_results', help='Output directory')
    parser.add_argument('--no-figures', action='store_true', help='Skip figure generation')
    parser.add_argument('--no-save', action='store_true', help='Skip saving matrices')

    args = parser.parse_args()

    results = run_full_analysis(
        version=args.version,
        output_dir=args.output_dir,
        save_figures=not args.no_figures,
        save_data=not args.no_save
    )

    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)
