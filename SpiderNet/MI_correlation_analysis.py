## Package import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, leaves_list


def run_analysis(file_savepath_main, Factor_envir_use_path, show=True):
    """
    Compute and visualize correlation among MI factors.

    Args:
        file_savepath_main (str): Directory to save the result figure.
        Factor_envir_use_path (str): Path to the .npy file containing factor data.
        show (bool): Whether to display the heatmap inline (for Jupyter).
    """
    # Verify paths
    if not os.path.exists(Factor_envir_use_path):
        raise FileNotFoundError(f"File not found: {Factor_envir_use_path}")

    # Load array
    Factor_envir_use = np.load(Factor_envir_use_path)
    if Factor_envir_use.ndim == 1:
        Factor_envir_use = Factor_envir_use.reshape(-1, 1)

    # Compute correlation
    Factor_envir_use_corr = np.corrcoef(Factor_envir_use, rowvar=False)
    Factor_envir_use_corr = np.nan_to_num(Factor_envir_use_corr, nan=0.0, posinf=0.0, neginf=0.0)

    # Clustering
    dist_matrix = 1 - Factor_envir_use_corr
    row_linkage = linkage(squareform(dist_matrix, checks=False), method='average')
    col_linkage = linkage(squareform(dist_matrix.T, checks=False), method='average')

    row_order = leaves_list(row_linkage)
    col_order = leaves_list(col_linkage)

    Factor_envir_use_corr = pd.DataFrame(
        Factor_envir_use_corr,
        index=[f"MI-{i+1}" for i in range(Factor_envir_use_corr.shape[0])],
        columns=[f"MI-{i+1}" for i in range(Factor_envir_use_corr.shape[0])]
    )

    # Reorder
    corr_clustered = Factor_envir_use_corr.iloc[row_order, :].iloc[:, col_order]

    # Plot
    plt.figure(figsize=(10, 10))
    sns.heatmap(
        corr_clustered,
        cmap="viridis",
        annot=True,
        fmt=".2f",
        square=True,
        cbar_kws={"shrink": .8},
        mask=np.eye(len(corr_clustered))
    )

    plt.title("Correlation of MI Factors (Clustered)")
    save_path = os.path.join(file_savepath_main, "MI_correlation_heatmap_clustered.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    if show:
        plt.show()

    plt.close()
    print(f"✅ Saved heatmap to {save_path}")


# Allow both import and direct execution
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        raise ValueError("Usage: python MI_correlation_analysis.py <file_savepath_main> <Factor_envir_use_path>")
    run_analysis(sys.argv[1], sys.argv[2], show=False)
