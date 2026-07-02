"""
PHASE 3: UNSUPERVISED CLUSTERING & VALIDATION (HEADLESS CLUSTER & THESIS READY)
Takes an N x N symmetric Hausdorff distance matrix from Phase 2 (SUPPLY or DEMAND),
performs Agglomerative Hierarchical Clustering, evaluates optimal k via Silhouette,
identifies/plots representative medoids, and profiles cluster distributions.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe for headless cluster/SSH sessions, no display needed
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import time
import json
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import silhouette_score

# ============================================================================
# CONFIGURATION
# ============================================================================
CURVES_DICT_PATH = Path("simplified_curves_dict.joblib")
OUTPUT_DIR = Path("phase3_output")
OUTPUT_DIR.mkdir(exist_ok=True)

K_MIN = 3
K_MAX = 8

# Running inside Slurm? Automatically bypass interactive inputs if True
IS_SLURM = "SLURM_JOB_ID" in os.environ
RUN_SUBSAMPLE_CHECK_FIRST = True if not IS_SLURM else False
SUBSAMPLE_SIZE = 2000


# ============================================================================
# 0. ARGUMENT HANDLING
# ============================================================================
def get_side_from_args():
    if len(sys.argv) != 2 or sys.argv[1].upper() not in ("SUPPLY", "DEMAND"):
        print("ERROR: must specify which side to cluster.")
        print("Usage: python Phase_3.py SUPPLY")
        print("       python Phase_3.py DEMAND")
        sys.exit(1)
    return sys.argv[1].upper()


# ============================================================================
# 1. DATA LOADING
# ============================================================================
def detect_side_key(curves_dict, side):
    target_letter = 'V' if side == 'SUPPLY' else 'C'
    sample_key = next(iter(curves_dict.keys()))
    
    candidates = []
    if isinstance(sample_key, tuple):
        for pos in range(len(sample_key)):
            try:
                matched = [k for k in curves_dict.keys() if k[pos] == target_letter]
                if matched:
                    candidates.append((pos, matched))
            except (IndexError, TypeError):
                continue

    if not candidates:
        raise ValueError(f"Could not auto-detect '{target_letter}' marker.")

    candidates.sort(key=lambda c: c[0])
    pos, matched_keys = candidates[-1]
    return matched_keys


def load_data(side, distance_matrix_path):
    print(f"Loading {side} distance matrix and curve metadata...")
    curves_dict = joblib.load(CURVES_DICT_PATH)
    side_keys = detect_side_key(curves_dict, side)

    # Force the path to look for our new .npy file
    raw_path = distance_matrix_path.with_suffix('.npy')
    print(f"Loading raw matrix array from {raw_path}...")
    dist_matrix = np.load(raw_path)

    print(f"Loaded distance matrix of shape: {dist_matrix.shape}")

    if dist_matrix.shape[0] != len(side_keys):
        raise ValueError("MISMATCH between distance matrix rows and keys.")

    return dist_matrix, side_keys, curves_dict


# ============================================================================
# 2. HIERARCHICAL LINKAGE & RECONSTRUCTION
# ============================================================================
def compute_linkage(dist_matrix):
    print("Computing hierarchical linkage (Complete Linkage)...")
    t0 = time.time()
    symmetric_matrix = (dist_matrix + dist_matrix.T) / 2
    np.fill_diagonal(symmetric_matrix, 0)
    condensed_dist = squareform(symmetric_matrix, checks=False)
    Z = linkage(condensed_dist, method='complete')
    print(f"  Linkage computed in {time.time() - t0:.1f}s")
    return Z


def plot_dendrogram(Z, side):
    print("Plotting dendrogram...")
    plt.figure(figsize=(15, 7))
    plt.title(f'Hierarchical Clustering Dendrogram ({side.title()} Curves)', fontsize=14, fontweight='bold')
    plt.xlabel('Market Curves (Truncated Leaf Nodes)', fontsize=12)
    plt.ylabel('Hausdorff Distance Threshold', fontsize=12)
    
    dendrogram(Z, truncate_mode='lastp', p=40, leaf_rotation=90., leaf_font_size=10., show_contracted=False)
    
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    out_path = OUTPUT_DIR / f"{side}_Hierarchical_Dendrogram.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  Saved to {out_path}")


# ============================================================================
# 3. PARTITIONING & MEDOID EXTRACTION
# ============================================================================
def optimize_partitions(Z, dist_matrix, side):
    print(f"Evaluating optimal 'k' via Silhouette analysis...")
    best_k = K_MIN
    best_score = -1
    best_labels = None
    silhouette_scores = []

    for k in range(K_MIN, K_MAX + 1):
        t0 = time.time()
        labels = fcluster(Z, t=k, criterion='maxclust')
        score = silhouette_score(dist_matrix, labels, metric='precomputed')
        silhouette_scores.append(score)
        print(f"  k={k} | Silhouette Score: {score:.4f} | ({time.time() - t0:.1f}s)")

        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    plt.figure(figsize=(8, 5))
    plt.plot(range(K_MIN, K_MAX + 1), silhouette_scores, marker='o', linestyle='--', color='tab:blue', linewidth=2)
    plt.title(f'Silhouette Method Optimization Profile ({side.title()})', fontsize=12, fontweight='bold')
    plt.xlabel('Number of Clusters (k)', fontsize=11)
    plt.ylabel('Mean Silhouette Coefficient', fontsize=11)
    plt.axvline(x=best_k, color='tab:red', linestyle=':', label=f'Optimal k={best_k}')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    out_path = OUTPUT_DIR / f"{side}_Silhouette_Optimization.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    print(f"[DONE] Optimal k determined: {best_k} (Score: {best_score:.4f})")
    return best_labels, best_k


def extract_and_save_medoids(dist_matrix, labels, best_k, keys, side):
    print("Identifying and saving empirical cluster centers (medoids)...")
    medoid_indices = {}
    medoid_metadata = {}

    for cluster_id in range(1, best_k + 1):
        cluster_indices = np.where(labels == cluster_id)[0]
        sub_matrix = dist_matrix[cluster_indices][:, cluster_indices]
        distance_sums = sub_matrix.sum(axis=1)
        best_sub_idx = np.argmin(distance_sums)
        global_idx = cluster_indices[best_sub_idx]
        
        medoid_indices[cluster_id] = int(global_idx)
        actual_key = keys[global_idx]
        
        # Format key strings safely for JSON representation
        key_str = "_".join(map(str, actual_key)) if isinstance(actual_key, tuple) else str(actual_key)
        
        # FIX: Convert the tuple elements to strings so JSON can serialize them
        safe_market_key = [str(item) for item in actual_key] if isinstance(actual_key, tuple) else str(actual_key)
        
        medoid_metadata[f"Cluster_{cluster_id}"] = {
            "global_index": int(global_idx),
            "market_key": safe_market_key,  # <--- Use the safe string version here
            "key_string": key_str
        }

    with open(OUTPUT_DIR / f"{side}_cluster_medoids_metadata.json", "w") as f:
        json.dump(medoid_metadata, f, indent=4)
        
    print(f"[DONE] Saved medoid mappings to JSON.")
    return medoid_indices


# ============================================================================
# 4. THESIS PROFILE VISUALIZATION (SPAGHETTI PLOTS)
# ============================================================================
def plot_cluster_profiles(curves_dict, keys, labels, medoid_map, best_k, side):
    """
    Generates academic cluster profile figures: Member shapes plotted in high
    transparency background with the mathematical archetype (Medoid) highlighted.
    """
    print("Generating cluster shape profile figures for thesis text...")
    
    # Calculate grid canvas proportions dynamically
    cols = 2
    rows = (best_k + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows), sharex=True, sharey=True)
    axes = axes.flatten()

    for cluster_id in range(1, best_k + 1):
        ax = axes[cluster_id - 1]
        cluster_indices = np.where(labels == cluster_id)[0]
        
        # Draw background shapes (sample maximum of 80 curves to maintain clarity)
        np.random.seed(42)
        sample_indices = np.random.choice(cluster_indices, size=min(80, len(cluster_indices)), replace=False)
        
        for idx in sample_indices:
            curve_df = curves_dict[keys[idx]]
            ax.plot(curve_df.iloc[:, 0], curve_df.iloc[:, 1], color='gray', alpha=0.08, linewidth=1)
        
        # Overlaid bold Medoid curve
        medoid_idx = medoid_map[cluster_id]
        medoid_df = curves_dict[keys[medoid_idx]]
        medoid_label = keys[medoid_idx][0] if isinstance(keys[medoid_idx], tuple) else "Medoid"
        
        ax.plot(medoid_df.iloc[:, 0], medoid_df.iloc[:, 1], color='tab:red', linewidth=2.5, 
                label=f'Archetype ({medoid_label})')
        
        ax.set_title(f'Cluster {cluster_id} Profile (N={len(cluster_indices)})', fontsize=11, fontweight='bold')
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(loc='upper right', fontsize=9)

    # Clean up empty subplots if any exist
    for idx in range(best_k, len(axes)):
        fig.delaxes(axes[idx])

    fig.text(0.5, 0.01, 'Energy Volume / Quantity (MW)', ha='center', fontsize=12, fontweight='bold')
    fig.text(0.01, 0.5, 'Market Bidding Price (€/MWh)', va='center', rotation='vertical', fontsize=12, fontweight='bold')
    
    plt.suptitle(f'Empirical Structural Form Archetypes -- {side.title()} Side Strategy Groups', 
                 fontsize=14, fontweight='bold', y=0.99)
    plt.tight_layout()
    out_path = OUTPUT_DIR / f"{side}_Cluster_Structural_Profiles.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved master structural profile grid to {out_path}")


# ============================================================================
# 5. EXPORT FILE MAPPING & SORT DISTRIBUTION ANALYSIS
# ============================================================================
def export_cluster_mapping(keys, labels, best_k, side):
    print("Exporting final cluster assignments and structural sorting...")
    sample = keys[0]
    
    if isinstance(sample, tuple) and len(sample) >= 2:
        df = pd.DataFrame(keys, columns=['Date', 'Hour', 'Side'][:len(sample)])
    else:
        df = pd.DataFrame({'Key': keys})

    df['Cluster_ID'] = labels
    output_csv = OUTPUT_DIR / f"{side}_Cluster_Assignments_k{best_k}.csv"
    df.to_csv(output_csv, index=False)
    
    print("\n--- Structural Cluster Distribution ---")
    print(df['Cluster_ID'].value_counts().sort_index())
    
    # If structural hourly indices exist, provide a summary table for thesis contexts
    if 'Hour' in df.columns:
        print("\n--- Cross-Tabulation: Cluster Presence across Market Hours ---")
        hourly_dist = pd.crosstab(df['Hour'], df['Cluster_ID'], normalize='index') * 100
        print(hourly_dist.round(1).head(10)) # displaying first 10 rows
        hourly_dist.to_csv(OUTPUT_DIR / f"{side}_Hourly_Cluster_Distribution_Matrix.csv")


# ============================================================================
# MAIN PIPELINE
# ============================================================================
if __name__ == "__main__":
    side = get_side_from_args()
    distance_matrix_path = Path(f"phase2_output/{side}/distance_matrix_{side}.npz")

    print("=" * 60)
    print(f"PHASE 3 RUN TRIGGERED FOR: {side}")
    if IS_SLURM:
        print("  Running non-interactively under Slurm Batch Scheduler context.")
    print("=" * 60)

    if not distance_matrix_path.exists():
        print(f"ERROR: distance matrix not found at {distance_matrix_path}")
        sys.exit(1)

    dist_matrix, keys, curves_dict = load_data(side, distance_matrix_path)

    Z = compute_linkage(dist_matrix)
    plot_dendrogram(Z, side)
    
    best_labels, best_k = optimize_partitions(Z, dist_matrix, side)
    medoid_map = extract_and_save_medoids(dist_matrix, best_labels, best_k, keys, side)
    
    # Thesis visualization generation pass
    plot_cluster_profiles(curves_dict, keys, best_labels, medoid_map, best_k, side)
    export_cluster_mapping(keys, best_labels, best_k, side)

    print(f"\n[DONE] PHASE 3 COMPLETION PROTOCOL VERIFIED FOR {side}.\n")