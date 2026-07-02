# Create test_phase2.py
import sys
sys.path.insert(0, '.')
from Phase_2 import DistanceMatrixConfig, run_phase2, setup_logging

# Configure test mode
config = DistanceMatrixConfig()
config.test_mode = True           # Only 100 curves
config.test_sample_size = 100
config.chunk_size = 50            # Smaller chunks for testing
config.n_jobs = 4                 # Use 4 cores for test

# Create logger
logger = setup_logging(config.output_dir)

# Run test
logger.log("Starting Phase 2 LOCAL TEST")
run_phase2(config, logger)
logger.log("✓ Local test complete!")