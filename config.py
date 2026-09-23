SYMBOL = "BTC-USD"
INTERVAL = "1m"

# How much history to collect for training.
HISTORY_CANDLES = 10000

# Coinbase candles are requested in chunks.
COINBASE_CANDLE_LIMIT = 300

HORIZONS = (1, 5, 15)
MIN_TRAIN_ROWS = 2000

MODEL_DIR = "models"
DATA_DIR = "data"

REFRESH_SECONDS = 10
