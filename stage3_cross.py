"""
STAGE 3: Compute CROSS-side (supply x demand) distance matrix only.
Independent of SUPPLY and DEMAND stages -- needs only the dicts, not their
already-computed distance matrices, so it can run in parallel with them
if you wanted, or after them.
"""
from _shared_setup import get_config_logger_and_split
from Phase_2 import compute_cross_side_distances

if __name__ == "__main__":
    config, logger, supply_dict, demand_dict = get_config_logger_and_split()

    logger.log("\n" + "=" * 80)
    logger.log("RUNNING STAGE: CROSS ONLY")
    logger.log("=" * 80)

    compute_cross_side_distances(supply_dict, demand_dict, config, logger)

    logger.log("\n[DONE] STAGE COMPLETE: CROSS")
