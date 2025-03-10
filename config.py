from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.resolve()

# Datasets - raw, processed
DATA_DIR = PROJECT_ROOT / 'data'

# Models - pretrained, checkpoints
MODELS_DIR = PROJECT_ROOT / 'models'