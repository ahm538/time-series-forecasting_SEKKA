from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "sekka_outputs"

# Files
DEFAULT_CSV = DATA_DIR / "sekka_mansoura_synthetic_dataset_2years_v2.csv"
TRAINING_REPORT = PROJECT_ROOT / "training_report.csv"

# Modeling
FORECAST_HORIZON_DAYS = 7
TEST_DAYS = 30
SEASONALITY_MODE = "additive"  # or "multiplicative"
DAILY_FOURIER = 15
WEEKLY_FOURIER = 10
CHANGEPOINT_PRIOR_SCALE = 0.5
SEASONALITY_PRIOR_SCALE = 10.0

# Target bounds
Y_MIN = 0.0
Y_MAX = 10.0

# FastAPI
API_HOST = "0.0.0.0"
API_PORT = 8000
