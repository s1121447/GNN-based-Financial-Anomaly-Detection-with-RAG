import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "cache")
GRAPH_CACHE_DIR = os.path.join(CACHE_DIR, "graphs")
DATASET_CACHE_DIR = os.path.join(CACHE_DIR, "datasets")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

os.makedirs(GRAPH_CACHE_DIR, exist_ok=True)
os.makedirs(DATASET_CACHE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

GEMINI_MODEL = "gemini-2.5-flash"

LOOKBACK_PERIOD = "2y"
LABEL_HORIZON_DAYS = 5
ANOMALY_THRESHOLD = -0.08   # 未來 5 日最大跌幅 <= -8% 視為異常
MIN_COMMON_DATES = 120
CORR_EDGE_THRESHOLD = 0.30
CORR_EDGE_WINDOW = 60

FEATURE_COLUMNS = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "vol_change_1d",
    "vol_ratio_5",
    "volatility_5",
    "volatility_20",
    "ma5_bias",
    "ma20_bias",
    "rsi_14",
    "hl_range",
]