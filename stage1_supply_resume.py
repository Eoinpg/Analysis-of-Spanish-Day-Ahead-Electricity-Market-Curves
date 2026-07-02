"""
SUPPLY RESUME SCRIPT
====================
Resumes the SUPPLY distance matrix computation from the last saved chunk,
without deleting existing chunks or restarting from chunk 0.

Uses Phase_2.py unchanged -- only bypasses the chunk deletion and
chunk loop start point inside process_market_side.

USAGE:
    python stage1_supply_resume.py
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _shared_setup import get_config_logger_and_split
from Phase_2 import (
    compute_distance_matrix_chunk,
    save_distance_matrix_chunk,
    assemble_distance_matrix,
    validate_distance_matrix,
)


def resume_market_side(side_name, side_curves_dict, config, logger):
    """
    Resumes chunk computation for a given side from the last saved chunk.
    Does NOT delete existing chunks -- picks up exactly where the job left off.
    """
    logger.log(f"\n--- RESUMING {side_name} CURVES ---")

    side_output_dir = config.output_dir / side_name
    side_output_dir.mkdir(exist_ok=True)

    index_to_key = sorted(side_curves_dict.keys())
    n_curves = len(index_to_key)
    num_chunks = (n_curves + config.chunk_size - 1) // config.chunk_size

    # Find which chunks already exist on disk
    existing_chunks = sorted(side_output_dir.glob("distance_chunk_*.npz"))
    completed_ids = set()
    for f in existing_chunks:
        try:
            chunk_id = int(f.stem.replace("distance_chunk_", ""))
            completed_ids.add(chunk_id)
        except ValueError:
            continue

    if completed_ids:
        last_done = max(completed_ids)
        resume_from = last_done + 1
        logger.log(f"Found {len(completed_ids)} existing chunks (0 to {last_done})")
        logger.log(f"Resuming from chunk {resume_from} of {num_chunks - 1}")
    else:
        resume_from = 0
        logger.log("No existing chunks found -- starting from chunk 0")

    if resume_from >= num_chunks:
        logger.log(f"All {num_chunks} chunks already complete -- skipping to assembly")
    else:
        remaining = num_chunks - resume_from
        logger.log(f"Remaining chunks to compute: {remaining}")

        for chunk_id in range(resume_from, num_chunks):
            start_idx = chunk_id * config.chunk_size
            end_idx = min((chunk_id + 1) * config.chunk_size, n_curves)

            chunk_distances = compute_distance_matrix_chunk(
                side_curves_dict, index_to_key, start_idx, end_idx,
                metric=config.distance_metric,
                directed=config.use_directed,
                n_jobs=config.n_jobs,
                logger=logger
            )

            save_distance_matrix_chunk(
                chunk_distances, chunk_id, side_output_dir, config.dtype
            )
            logger.log(f"[DONE] Saved chunk {chunk_id}/{num_chunks - 1}")

    # Assemble all chunks (existing + newly computed) into final matrix
    logger.log("\nAssembling final distance matrix from all chunks...")
    distance_matrix = assemble_distance_matrix(
        n_curves, side_output_dir, logger, config.dtype
    )
    validate_distance_matrix(distance_matrix, logger)

    output_file = side_output_dir / f"distance_matrix_{side_name}.npz"
    np.savez_compressed(output_file, distance_matrix=distance_matrix)
    logger.log(f"[DONE] Saved {side_name} matrix to {output_file}")


if __name__ == "__main__":
    config, logger, supply_dict, demand_dict = get_config_logger_and_split()

    logger.log("=" * 80)
    logger.log("STAGE: SUPPLY RESUME")
    logger.log("=" * 80)

    resume_market_side("SUPPLY", supply_dict, config, logger)

    logger.log("\n[DONE] SUPPLY RESUME COMPLETE")