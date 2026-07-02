"""
STAGE 1: Compute SUPPLY-side distance matrix only.
Run this first. Independent of DEMAND and CROSS stages.
"""
from _shared_setup import get_config_logger_and_split
from Phase_2 import process_market_side

if __name__ == "__main__":
    config, logger, supply_dict, demand_dict = get_config_logger_and_split()

    logger.log("\n" + "=" * 80)
    logger.log("RUNNING STAGE: SUPPLY ONLY")
    logger.log("=" * 80)

    process_market_side("SUPPLY", supply_dict, config, logger)

    logger.log("\n[DONE] STAGE COMPLETE: SUPPLY")
