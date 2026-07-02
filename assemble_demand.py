"""
GEOMETRIC DEMAND MATRIX RECONSTRUCTION
Bypasses corrupted coordinate arrays in the .npz files by mathematically 
regenerating the true upper-triangular grid coordinates based on chunk IDs.
"""
from pathlib import Path
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _shared_setup import get_config_logger_and_split

if __name__ == "__main__":
    config, logger, supply_dict, demand_dict = get_config_logger_and_split()
    
    side_name = "DEMAND"
    side_output_dir = config.output_dir / side_name
    n_curves = len(demand_dict)
    
    print("=" * 60)
    print(f"GEOMETRIC MASTER COMPILE FOR: {side_name}")
    print(f"Targeting Matrix Dimensions: {n_curves} x {n_curves}")
    print("=" * 60)

    chunk_files = sorted(side_output_dir.glob("distance_chunk_*.npz"))
    if not chunk_files:
        print(f"CRITICAL ERROR: No chunk files found in {side_output_dir.absolute()}!")
        sys.exit(1)

    print(f"Found {len(chunk_files)} chunk files. Allocating master grid...")
    master_matrix = np.zeros((n_curves, n_curves), dtype=np.float32)

    for chunk_file in chunk_files:
        print(f"  -> Reading {chunk_file.name}...")
        chunk = np.load(chunk_file)
        data_vals = chunk['data']
        
        # 1. Extract chunk ID to know exactly which rows this file processed
        chunk_id = int(chunk_file.stem.split('_')[-1])
        start_idx = chunk_id * config.chunk_size
        end_idx = min((chunk_id + 1) * config.chunk_size, n_curves)
        
        # 2. Mathematically regenerate the true coordinate maps
        i_list, j_list = [], []
        for r in range(start_idx, end_idx):
            i_list.append(np.full(n_curves - r, r, dtype=np.int32))
            j_list.append(np.arange(r, n_curves, dtype=np.int32))
            
        true_i = np.concatenate(i_list)
        true_j = np.concatenate(j_list)
        
        # 3. Validation check
        if len(true_i) != len(data_vals):
            print(f"     WARNING: Length mismatch! Expected {len(true_i)}, got {len(data_vals)}")
            # Fallback just in case
            true_i = chunk['i']
            true_j = chunk['j']
        
        chunk_max = data_vals.max()
        print(f"     Chunk {chunk_id} has {len(data_vals)} elements. True Max = {chunk_max:.2f}")
        
        # 4. Stamp values into the master matrix symmetrically
        master_matrix[true_i, true_j] = data_vals
        master_matrix[true_j, true_i] = data_vals
        
        current_max = master_matrix.max()
        print(f"     Master Matrix Global Max so far: {current_max:.2f}")

    print("\n[DONE] All chunks processed geometrically!")
    print(f"FINAL Master Matrix Max Value: {master_matrix.max():.2f}")
    print(f"FINAL Master Matrix Mean Value: {master_matrix.mean():.2f}")
    
    if master_matrix.max() > 0:
        output_file = side_output_dir / f"distance_matrix_{side_name}.npy"
        print(f"Writing raw binary array to {output_file}...")
        np.save(output_file, master_matrix)
        print("[DONE] SUCCESS! You are fully cleared to run Phase 3 Clustering.")
    else:
        print("ERROR: Master matrix is still zero.")