# config.py
from pathlib import Path
from typing import Final, List
from dotenv import load_dotenv
from datetime import datetime, timezone
import pytz
import os

# --------------------------- #
# 1. Load .env
# --------------------------- #
_ENV_PATH = ".env"
load_dotenv(dotenv_path=_ENV_PATH)

# --------------------------- #
# 2. Environment Variables
# --------------------------- #
MT5_SERVER:   Final[str] = os.getenv("MT5_SERVER", "FTMO-Demo")
MT5_LOGIN:    Final[int] = int(os.getenv("MT5_LOGIN", "0"))
MT5_PASSWORD: Final[str] = os.getenv("MT5_PASSWORD", "")
DB_PATH:      Final[str] = os.getenv("DB_PATH", "data/trading.db")
MODEL_DIR:      Final[str] = os.getenv("MODEL_DIR", "models")
BINANCE_BASE: Final[str] = os.getenv("BINANCE_BASE", "https://api.binance.com").rstrip("/")

# --------------------------- #
# 3. UTC Timezone (fixed)
# --------------------------- #
TZ: Final[timezone] = timezone.utc
MT5_TZ: Final[timezone] = pytz.FixedOffset(120)   # GMT+2 (Egypt Standard Time)

# --------------------------- #
# 3. Loguru Setup
# --------------------------- #
from loguru import logger

# Remove default handler
logger.remove()

# Add colorful console handler
logger.add(
    sink=lambda msg: print(msg, end=""),
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
           "<level>{message}</level>",
    level="INFO",
    colorize=True,
)

# Optional: Add file logging
log_file = Path("logs") / "trading.log"
log_file.parent.mkdir(exist_ok=True)
logger.add(
    sink=str(log_file),
    rotation="7 days",
    retention="30 days",
    compression="zip",
    level="DEBUG",
    encoding="utf-8"
)

# ------------------------------------------------------------------ #
# 4. Portfolio
# ------------------------------------------------------------------ #
PORTFOLIO: Final[List[str]] = [
    "BTCUSD",   # Bitcoin
    "ETHUSD",   # Ethereum
    "BNBUSD",   # BNB
    #"XRPUSD",
    #"SOLUSD",   # Solana
    #"DOGEUSD",  # Dogecoin
    #"ADAUSD",
    #"LTCUSD"
]

# ------------------------------------------------------------------ #
# 5. Rebalance Settings
# ------------------------------------------------------------------ #
MAX_HISTORY_HOURS: Final[int] = 24 * 360 * 2  # 2 years of hourly data
MAX_POSITION_PCT: Final[float] = 2.0          # ±200% max exposure
LOT_PRECISION: Final[int] = 2                 # 2 decimals for lots
