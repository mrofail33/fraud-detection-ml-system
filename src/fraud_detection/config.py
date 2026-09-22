from pathlib import Path

TARGET_COLUMN = "Class"
DEFAULT_RANDOM_STATE = 42
DEFAULT_TEST_SIZE = 0.2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
MODELS_DIR = OUTPUT_DIR / "models"

MODEL_FILENAME = "best_model.joblib"
