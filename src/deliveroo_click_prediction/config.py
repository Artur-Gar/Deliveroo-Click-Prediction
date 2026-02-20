"""Project configuration and shared constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
MODELS_DIR = PROJECT_ROOT / "models"

TARGET_COL = "Clicks_Conversion"
LOCAL_RDATA_FILENAME = "DeliveryAdClick.RData"
LOCAL_RDATA_PATH = RAW_DATA_DIR / LOCAL_RDATA_FILENAME

SESSION_KEY_DF_TRAIN = "df_train"
SESSION_KEY_DF_NEW = "df_new"
SESSION_KEY_X_TRAIN_FULL = "X_train_full"
SESSION_KEY_Y_TRAIN_FULL = "y_train_full"
SESSION_KEY_MODEL = "model"
SESSION_KEY_METRICS = "metrics"
SESSION_KEY_X_TEST = "X_test"
SESSION_KEY_Y_TEST = "y_test"
SESSION_KEY_Y_PRED = "y_pred"
SESSION_KEY_Y_PROB = "y_prob"
