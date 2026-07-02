"""
PHASE 2: DISTANCE MATRIX COMPUTATION (WITH CURVE FIXING)

IMPORTANT: This version includes pre-processing to add missing origin points
to curves that don't start at energy = 0.

Demand curves (C) get point [0, max_price] at the beginning
Supply curves (V) get point [0, min_price] at the beginning

This ensures economically correct curve representation.
"""

import numpy as np
import pandas as pd
import joblib
import json
import time
import sys
from datetime import datetime
from scipy.spatial.distance import directed_hausdorff
from pathlib import Path
from joblib import Parallel, delayed

# ============================================================================
# SECTION 3A: CURVE PREPROCESSING
# ============================================================================

def fix_curve_origin_points(curves_dict, logger):
    """
    Energy market curves technically represent a continuous 
    function of willingness to buy/sell starting from zero volume. If a curve is 
    missing its (0, Y) intercept, distance algorithms misinterpret its shape.
    This function forces all curves to anchor at the Y-axis.
    """
    logger.log("Fixing curves that don't start at energy=0...")
    
    curves_modified = 0
    stats = {'supply_modified': 0, 'demand_modified': 0, 'total_processed': 0}
    
    for key, curve in curves_dict.items():
        date, hour, side = key
        stats['total_processed'] += 1
        
        min_energy = curve['CumEnergy'].iloc[0]
        
        # If the curve doesn't start at 0 MWh, we project it backward horizontally.
        if min_energy > 0:
            if side == 'C': # Demand (Compra): Origin is the maximum willing price
                origin_price = curve['Price'].max()
                stats['demand_modified'] += 1
            else: # Supply (Venta): Origin is the minimum asking price
                origin_price = curve['Price'].min()
                stats['supply_modified'] += 1
            
            # Included 'Energy': [0.0] so pandas doesn't create NaNs!
            origin_point = pd.DataFrame({
                'CumEnergy': [0.0],
                'Price': [float(origin_price)],
                'Energy': [0.0]  
            })
            
            curve_fixed = pd.concat([origin_point, curve], ignore_index=True)
            curves_dict[key] = curve_fixed
            curves_modified += 1
            
    logger.log(f"[DONE] Preprocessing complete:")
    logger.log(f"  Total curves processed: {stats['total_processed']:,}")
    logger.log(f"  Supply curves modified: {stats['supply_modified']:,}")
    logger.log(f"  Demand curves modified: {stats['demand_modified']:,}")
    logger.log(f"  Total curves with origin point added: {curves_modified:,}")
    
    return curves_dict, stats

# ============================================================================
# SECTION 4: DISTANCE METRIC IMPLEMENTATIONS
# ============================================================================

def hausdorff_distance_directed(curve1, curve2):
    """
    Compute directed Hausdorff distance. 
    Measures the maximum distance from a point in curve1 to the closest point in curve2.
    """
    try:
        dist, _ = directed_hausdorff(curve1, curve2)
        return float(dist)
    except:
        return np.inf

def hausdorff_distance_symmetric(curve1, curve2):
    """
    Compute symmetric Hausdorff distance.
    """
    d12 = hausdorff_distance_directed(curve1, curve2)
    d21 = hausdorff_distance_directed(curve2, curve1)
    return max(d12, d21)

def compute_distance(curve1, curve2, metric='hausdorff', directed=True):
    """Wrapper for distance computation to allow easy swapping of metrics later if needed"""
    try:
        if metric == 'hausdorff':
            if directed:
                return hausdorff_distance_directed(curve1, curve2)
            else:
                return hausdorff_distance_symmetric(curve1, curve2)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    except Exception as e:
        print(f"Error computing distance: {e}")
        return np.nan

# ============================================================================
# SECTION 5: DATA LOADING & VALIDATION
# ============================================================================

def load_simplified_curves(input_file, logger, test_mode=False, test_size=100):
    """Load simplified curves from Phase 1 joblib file"""
    logger.log(f"Loading simplified curves from {input_file}...")
    
    try:
        curves_dict = joblib.load(input_file)
        logger.log(f"[DONE] Loaded {len(curves_dict):,} curves")
        
        # Test mode creation
        if test_mode:
            logger.log(f"  Running in TEST MODE (using {test_size} curves)")
            
            # Explicitly find all Supply and Demand keys
            supply_keys = [k for k in curves_dict.keys() if (isinstance(k, tuple) and k[-1] == 'V') or (isinstance(k, str) and k.endswith('V'))]
            demand_keys = [k for k in curves_dict.keys() if (isinstance(k, tuple) and k[-1] == 'C') or (isinstance(k, str) and k.endswith('C'))]
            
            # Calculate a 50/50 split to ensure balanced testing
            half_size = test_size // 2
            
            sample_supply = supply_keys[:half_size]
            sample_demand = demand_keys[:half_size]
            
            sample_keys = sample_supply + sample_demand
            curves_dict = {k: curves_dict[k] for k in sample_keys}
            
            logger.log(f"  Sampled {len(sample_supply)} Supply (V) and {len(sample_demand)} Demand (C) curves for testing")
        
        return curves_dict
        
    except FileNotFoundError:
        logger.log(f"ERROR: File not found: {input_file}", level="ERROR")
        sys.exit(1)

def validate_curves(curves_dict, logger):
    """
    Ensure no malformed DataFrames crash the parallel workers. 
    A single NaN could ruin an entire row in the distance matrix.
    """
    logger.log("Validating curves...")
    
    stats = {
        'total_curves': len(curves_dict),
        'min_points': float('inf'),
        'max_points': 0,
        'mean_points': 0,
        'by_side': {'V': 0, 'C': 0},
        'curves_starting_at_zero': 0,
        'invalid_curves': []
    }
    
    total_points = 0
    
    for key, curve in curves_dict.items():
        date, hour, side = key
        
        # Structural checks
        if not isinstance(curve, pd.DataFrame):
            stats['invalid_curves'].append((key, "Not DataFrame"))
            continue
        
        if 'CumEnergy' not in curve.columns or 'Price' not in curve.columns:
            stats['invalid_curves'].append((key, "Missing columns"))
            continue
        
        if len(curve) < 2:
            stats['invalid_curves'].append((key, "Less than 2 points"))
            continue
        
        if curve.isnull().any().any():
            stats['invalid_curves'].append((key, "Contains NaN"))
            continue
        
        # Economic structural check
        if curve['CumEnergy'].iloc[0] == 0:
            stats['curves_starting_at_zero'] += 1
        
        n_points = len(curve)
        total_points += n_points
        stats['min_points'] = min(stats['min_points'], n_points)
        stats['max_points'] = max(stats['max_points'], n_points)
        stats['by_side'][side] += 1
    
    stats['mean_points'] = total_points / len(curves_dict) if len(curves_dict) > 0 else 0
    
    logger.log(f"[DONE] Validation complete:")
    logger.log(f"  Total curves: {stats['total_curves']:,}")
    logger.log(f"  Supply curves (V): {stats['by_side']['V']:,}")
    logger.log(f"  Demand curves (C): {stats['by_side']['C']:,}")
    logger.log(f"  Curves starting at energy=0: {stats['curves_starting_at_zero']:,}")
    logger.log(f"  Points per curve: min={stats['min_points']}, max={stats['max_points']}, mean={stats['mean_points']:.1f}")
    
    return stats

# ============================================================================
# SECTION 6: DISTANCE COMPUTATION 
# ============================================================================

def compute_single_row(i, index_to_key, curves_dict, n_total, metric, directed):
    """
    Compute distances for one row in the matrix.
    Because the distance matrix is symmetric, only compute the upper triangle
    to save 50% of the computation time.
    """
    i_idx, j_idx, dists = [], [], []
    key_i = index_to_key[i]
    curve_i = curves_dict[key_i][['CumEnergy', 'Price']].values
    
    for j in range(i, n_total):
        key_j = index_to_key[j]
        curve_j = curves_dict[key_j][['CumEnergy', 'Price']].values
        
        if not directed:
            d1 = directed_hausdorff(curve_i, curve_j)[0]
            d2 = directed_hausdorff(curve_j, curve_i)[0]
            dist = max(d1, d2)
        else:
            dist = directed_hausdorff(curve_i, curve_j)[0]
            
        i_idx.append(i)
        j_idx.append(j)
        dists.append(dist)
    
    return i_idx, j_idx, dists

def compute_single_cross_row(i, supply_curve, demand_curves_list, directed):
    """Computes a single row safely in memory mapping for cross-side calculation"""
    row_dists = np.zeros(len(demand_curves_list), dtype=np.float32)
    for j, demand_curve in enumerate(demand_curves_list):
        if not directed:
            d1 = directed_hausdorff(supply_curve, demand_curve)[0]
            d2 = directed_hausdorff(demand_curve, supply_curve)[0]
            row_dists[j] = max(d1, d2)
        else:
            row_dists[j] = directed_hausdorff(supply_curve, demand_curve)[0]
    return i, row_dists

def compute_cross_side_distances(supply_dict, demand_dict, config, logger):
    """
    Compute distances between supply and demand curves. 
    """
    logger.log(f"\n{'='*80}")
    logger.log("--- PROCESSING CROSS-SIDE DISTANCES ---")
    logger.log(f"{'='*80}")
    
    if len(supply_dict) == 0 or len(demand_dict) == 0:
        logger.log("WARNING: Missing Supply or Demand curves. Skipping...")
        return
        
    cross_output_dir = config.output_dir / "CROSS"
    cross_output_dir.mkdir(exist_ok=True)
    
    supply_keys = sorted(supply_dict.keys())
    demand_keys = sorted(demand_dict.keys())
    n_supply = len(supply_keys)
    n_demand = len(demand_keys)
    
    logger.log(f"Computing {n_supply} supply x {n_demand} demand")
    logger.log(f"Total distances: {n_supply * n_demand:,}")
    
    cross_matrix = np.full((n_supply, n_demand), np.inf, dtype=config.dtype)
    
    logger.log("Pre-loading all curves into memory...")
    supply_list = [supply_dict[k][['CumEnergy', 'Price']].values for k in supply_keys]
    demand_list = [demand_dict[k][['CumEnergy', 'Price']].values for k in demand_keys]
    logger.log("[DONE] Curves loaded")
    
    logger.log(f"Computing cross-side distances using {config.n_jobs if config.n_jobs > 0 else 'ALL'} cores...")
    start_time = time.time()
    
    # Parallelise row computation across CPU cores
    results = Parallel(n_jobs=config.n_jobs)(
        delayed(compute_single_cross_row)(
            i, supply_list[i], demand_list, config.use_directed
        ) for i in range(n_supply)
    )
    
    for i, row_dists in results:
        cross_matrix[i, :] = row_dists
        
    elapsed = time.time() - start_time
    logger.log(f"[DONE] Cross-side computation complete in {elapsed/60:.2f} minutes")
    
    logger.log("Validating cross-side matrix...")
    nan_count = np.isnan(cross_matrix).sum()
    inf_count = np.isinf(cross_matrix).sum()
    logger.log(f"   Shape: {cross_matrix.shape}")
    logger.log(f"   Min: {np.nanmin(cross_matrix):.6f}")
    logger.log(f"   Max: {np.nanmax(cross_matrix):.6f}")
    logger.log(f"   Mean: {np.nanmean(cross_matrix):.6f}")
    logger.log(f"   NaN count: {nan_count}")
    logger.log(f"   Inf count: {inf_count}")
    
    if nan_count > 0 or inf_count > 0:
        logger.log("WARNING: Invalid values found in cross matrix!", level="WARNING")
        
    output_file = cross_output_dir / "distance_matrix_CROSS.npz"
    if config.compress_output:
        np.savez_compressed(output_file, distance_matrix=cross_matrix)
    else:
        np.savez(output_file, distance_matrix=cross_matrix)
        
    logger.log(f"[DONE] Saved to {output_file}")
    
    return cross_matrix

def compute_distance_matrix_chunk(curves_dict, index_to_key, start_idx, end_idx, 
                                 metric='hausdorff', directed=False, n_jobs=-1, logger=None):
    """
    Compute distance matrix chunk with parallelization.
    Chunking prevents Out-Of-Memory (OOM) crashes. An O(N^2) distance matrix for 
    77,000 curves requires holding billions of floats in RAM simultaneously. 
    Breaking it into chunks ensures stability on standard hardware.
    """
    n_total = len(index_to_key)
    chunk_size = end_idx - start_idx
    
    if logger:
        logger.log(f"Computing distances for curves {start_idx:,} to {end_idx-1:,} "
                  f"({chunk_size} curves) using {n_jobs if n_jobs > 0 else 'ALL'} cores...")
    
    start_time = time.time()
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(compute_single_row)(i, index_to_key, curves_dict, n_total, metric, directed)
        for i in range(start_idx, end_idx)
    )
    
    i_indices = np.concatenate([r[0] for r in results])
    j_indices = np.concatenate([r[1] for r in results])
    distances = np.concatenate([r[2] for r in results])
        
    elapsed = time.time() - start_time
    if logger:
        logger.log(f"[DONE] Chunk complete in {elapsed/60:.2f} minutes")
    
    return i_indices, j_indices, distances

def save_distance_matrix_chunk(chunk_data, chunk_id, output_dir, dtype):
    """Save chunk to disk to clear memory for the next processing block."""
    i_indices, j_indices, distances = chunk_data
    file_path = output_dir / f"distance_chunk_{chunk_id:05d}.npz"
    np.savez_compressed(file_path, i=i_indices, j=j_indices, data=distances)
    return str(file_path)

# ============================================================================
# SECTION 7: DISTANCE MATRIX ASSEMBLY
# ============================================================================

def assemble_distance_matrix(n_curves, output_dir, logger, dtype=np.float32):
    """
    Once all chunks are written to disk, this function loads them back one by one 
    and stitches them into the final continuous NumPy array.
    """
    logger.log("Assembling distance matrix from chunks...")
    
    distance_matrix = np.zeros((n_curves, n_curves), dtype=dtype)
    
    chunk_files = sorted(output_dir.glob("distance_chunk_*.npz"))
    logger.log(f"Found {len(chunk_files)} chunk files")
    
    for chunk_file in chunk_files:
        logger.log(f"Loading {chunk_file.name}...")
        data = np.load(chunk_file)
        i_indices = data['i']
        j_indices = data['j']
        values = data['data']
        
        # Populate upper and lower triangles to complete the symmetric matrix
        distance_matrix[i_indices, j_indices] = values
        distance_matrix[j_indices, i_indices] = values
    
    logger.log(f"[DONE] Distance matrix assembled: {distance_matrix.shape}")
    
    return distance_matrix

def validate_distance_matrix(distance_matrix, logger):
    """
    Mathematical validation check: Ensures the assembled matrix meets all algebraic 
    requirements for downstream clustering algorithms.
    """
    logger.log("Validating distance matrix...")
    
    # Distance from a curve to itself should be exactly 0
    diagonal = np.diag(distance_matrix)
    logger.log(f"  Diagonal: min={diagonal.min():.6f}, max={diagonal.max():.6f}, "
              f"mean={diagonal.mean():.6f}")
    
    # The matrix must mirror exactly across the diagonal
    is_symmetric = np.allclose(distance_matrix, distance_matrix.T, rtol=1e-5)
    logger.log(f"  Symmetry: {'SYMMETRIC' if is_symmetric else 'NOT SYMMETRIC'}")
    
    nan_count = np.isnan(distance_matrix).sum()
    inf_count = np.isinf(distance_matrix).sum()
    logger.log(f"  Invalid values: NaN={nan_count}, Inf={inf_count}")
    
    valid_mask = ~(np.isnan(distance_matrix) | np.isinf(distance_matrix))
    valid_distances = distance_matrix[valid_mask]
    
    if len(valid_distances) > 0:
        logger.log(f"  Distance statistics:")
        logger.log(f"    min={valid_distances.min():.6f}")
        logger.log(f"    max={valid_distances.max():.6f}")
        logger.log(f"    mean={valid_distances.mean():.6f}")
        logger.log(f"    median={np.median(valid_distances):.6f}")
        logger.log(f"    std={valid_distances.std():.6f}")
    
    return is_symmetric and nan_count == 0 and inf_count == 0

# ============================================================================
# SECTION 8: MAIN EXECUTION
# ============================================================================

class DistanceMatrixConfig:
    """
    Master configuration object. Keeping all parameters strictly defined in one 
    class ensures your thesis pipeline is 100% reproducible for reviewers.
    """
    def __init__(self):
        self.input_file = "simplified_curves_dict.joblib"
        self.output_dir = Path("phase2_output")
        self.output_dir.mkdir(exist_ok=True)
        
        self.distance_metric = "hausdorff" 
        self.use_directed = False
        self.symmetric_matrix = True  
        
        self.n_jobs = -1        # -1 utilizes all available CPU cores
        self.batch_size = 1000  
        self.chunk_size = 500   
        self.save_intermediate = True
        
        self.dtype = np.float32  # f32 uses exactly half the RAM of default f64
        self.compress_output = True
        
        self.normalize_prices = False  
        self.normalize_energy = False
        self.validate_monotonicity = True
        self.test_mode = False
        self.test_sample_size = 100
        
        self.compute_cross_side = True
        self.fix_curve_origins = True  # Activates the economic origin fix
        
def setup_logging(output_dir):
    """Setup logging to save a permanent console output record for auditing"""
    log_file = output_dir / f"phase2_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    class Logger:
        def __init__(self, file):
            self.file = file
            self.start_time = time.time()
            
        def log(self, message, level="INFO"):
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            elapsed = time.time() - self.start_time
            log_line = f"[{timestamp}][{elapsed:.1f}s][{level}] {message}"
            print(log_line)
            with open(self.file, 'a', encoding='utf-8') as f:
                f.write(log_line + "\n")
    
    return Logger(log_file)

def process_market_side(side_name, side_curves_dict, config, logger):
    """Process supply or demand curves independently"""
    logger.log(f"\n--- PROCESSING {side_name} CURVES ---")
    
    if len(side_curves_dict) == 0:
        logger.log(f"WARNING: 0 curves found for {side_name}. Skipping...")
        return
        
    # Create specific output directory for this side
    side_output_dir = config.output_dir / side_name
    side_output_dir.mkdir(exist_ok=True)
    
    # HOUSEKEEPING: Delete old chunks from previous runs to prevent mixing old and new data
    for old_chunk in side_output_dir.glob("distance_chunk_*.npz"):
        old_chunk.unlink()
    
    # Create index for just this side
    index_to_key = sorted(side_curves_dict.keys())
    n_curves = len(index_to_key)
    n_pairs = (n_curves * (n_curves + 1)) // 2
    logger.log(f"Estimation: {n_pairs:,} pairwise distances for {side_name}")
    
    num_chunks = (n_curves + config.chunk_size - 1) // config.chunk_size
    logger.log(f"Processing {num_chunks} chunks of size {config.chunk_size}")
    
    for chunk_id in range(num_chunks):
        start_idx = chunk_id * config.chunk_size
        end_idx = min((chunk_id + 1) * config.chunk_size, n_curves)
        
        chunk_distances = compute_distance_matrix_chunk(
            side_curves_dict, index_to_key, start_idx, end_idx,
            metric=config.distance_metric,
            directed=config.use_directed,
            n_jobs=config.n_jobs,
            logger=logger
        )
        
        save_distance_matrix_chunk(chunk_distances, chunk_id, side_output_dir, config.dtype)
        logger.log(f"[DONE] Saved chunk {chunk_id}/{num_chunks-1}")
    
    # Assemble & Save
    distance_matrix = assemble_distance_matrix(n_curves, side_output_dir, logger, config.dtype)
    validate_distance_matrix(distance_matrix, logger)
        
    output_file = side_output_dir / f"distance_matrix_{side_name}.npz"
    np.savez_compressed(output_file, distance_matrix=distance_matrix)
    logger.log(f"[DONE] Saved {side_name} matrix to {output_file}")

def run_phase2(config=None, logger=None):
    """Main execution controller"""
    if config is None: 
        config = DistanceMatrixConfig()
    
    if logger is None: 
        logger = setup_logging(config.output_dir)
    
    logger.log("="*80)
    logger.log("PHASE 2: DISTANCE MATRIX COMPUTATION")
    logger.log("="*80)
    
    # Load curves
    curves_dict = load_simplified_curves(
        config.input_file, logger, test_mode=config.test_mode, test_size=config.test_sample_size
    )
    
    # FIX: Add origin points if needed before ANY math happens
    if config.fix_curve_origins:
        curves_dict, fix_stats = fix_curve_origin_points(curves_dict, logger)
    
    # Validate curves
    validate_curves(curves_dict, logger)
    
    # Split into supply and demand
    supply_dict = {k: v for k, v in curves_dict.items() if (isinstance(k, tuple) and k[-1] == 'V')}
    demand_dict = {k: v for k, v in curves_dict.items() if (isinstance(k, tuple) and k[-1] == 'C')}
    
    logger.log(f"\n[DONE] Split curves:")
    logger.log(f"  Supply curves: {len(supply_dict):,}")
    logger.log(f"  Demand curves: {len(demand_dict):,}")
    
    # Process supply and demand
    process_market_side("SUPPLY", supply_dict, config, logger)
    process_market_side("DEMAND", demand_dict, config, logger)
    
    # Process cross-side if requested
    if config.compute_cross_side:
        compute_cross_side_distances(supply_dict, demand_dict, config, logger)
    
    logger.log("="*80)
    logger.log("PHASE 2 COMPLETE")
    logger.log("="*80)
    
    # Document execution in metadata JSON (vital for thesis reproducibility)
    metadata = {
        'creation_date': datetime.now().isoformat(),
        'supply_curves': len(supply_dict),
        'demand_curves': len(demand_dict),
        'total_curves': len(curves_dict),
        'distance_metric': config.distance_metric,
        'directed': config.use_directed,
        'curve_origins_fixed': config.fix_curve_origins,
        'cross_side_distances': config.compute_cross_side,
        'files_generated': [
            'SUPPLY/distance_matrix_SUPPLY.npz',
            'DEMAND/distance_matrix_DEMAND.npz',
            'CROSS/distance_matrix_CROSS.npz' if config.compute_cross_side else None
        ]
    }
    with open(config.output_dir / "distance_matrix_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.log("[DONE] Metadata saved")
    logger.log("\nPHASE 2 EXECUTION COMPLETE - ALL DISTANCES COMPUTED WITH CORRECTED CURVES")

if __name__ == "__main__":
    config = DistanceMatrixConfig()
    
    # Set to False when running the actual pipeline
    # Uncomment for test mode
    # config.test_mode = True
    # config.test_sample_size = 100
    
    run_phase2(config)