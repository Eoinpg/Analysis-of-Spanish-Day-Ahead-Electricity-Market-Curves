"""
STAGE 2: Compute DEMAND-side distance matrix only.
Independent of SUPPLY and CROSS stages -- can run in any order relative to them.
"""
from _shared_setup import get_config_logger_and_split
from Phase_2 import process_market_side

if __name__ == "__main__":
    config, logger, supply_dict, demand_dict = get_config_logger_and_split()

    logger.log("\n" + "=" * 80)
    logger.log("RUNNING STAGE: DEMAND ONLY")
    logger.log("=" * 80)

    process_market_side("DEMAND", demand_dict, config, logger)

    logger.log("\n[DONE] STAGE COMPLETE: DEMAND")
