"""
Shared setup used by all 3 stage drivers (supply / demand / cross).
Re-running this each time is cheap (seconds) -- it's just loading the
joblib file, fixing origins, validating, and splitting into supply/demand.
The expensive part (distance computation) happens only in the stage
you actually invoke.
"""
import sys
from pathlib import Path

# Make sure Phase_2.py (with all its function/class definitions) is importable.
# Assumes this file lives in the same directory as Phase_2.py, or that
# directory is on PYTHONPATH. Adjust if needed.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from Phase_2 import (
    DistanceMatrixConfig,
    setup_logging,
    load_simplified_curves,
    fix_curve_origin_points,
    validate_curves,
)


def get_config_logger_and_split():
    """Load config, set up logging, load+fix+validate curves, split into supply/demand."""
    config = DistanceMatrixConfig()
    logger = setup_logging(config.output_dir)

    logger.log("=" * 80)
    logger.log("STAGE SETUP: loading and preparing curves")
    logger.log("=" * 80)

    curves_dict = load_simplified_curves(
        config.input_file, logger,
        test_mode=config.test_mode, test_size=config.test_sample_size
    )

    if config.fix_curve_origins:
        curves_dict, fix_stats = fix_curve_origin_points(curves_dict, logger)

    validate_curves(curves_dict, logger)

    supply_dict = {k: v for k, v in curves_dict.items() if (isinstance(k, tuple) and k[-1] == 'V')}
    demand_dict = {k: v for k, v in curves_dict.items() if (isinstance(k, tuple) and k[-1] == 'C')}

    logger.log(f"\n[DONE] Split curves:")
    logger.log(f"  Supply curves: {len(supply_dict):,}")
    logger.log(f"  Demand curves: {len(demand_dict):,}")

    return config, logger, supply_dict, demand_dict
