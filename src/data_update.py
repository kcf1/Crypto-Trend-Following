from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import List
import pytz

from config import logger, PORTFOLIO, TZ, MAX_HISTORY_HOURS
from api_mt5 import get_mt5_bars,get_mt5_symbols,is_mt5_ready
from api_bnb import get_klines,get_recent_trades
from db_read import read_mtbars,read_klines,read_symbols,read_trades,get_binance_symbol
from db_load import save_mtbars,save_klines,save_symbols

# ------------------------------------------------------------------ #
# Update MT5 Bars
# ------------------------------------------------------------------ #
def update_mt5_bars(symbol: str) -> None:
    """Download and save latest 1H MT5 bars."""
    logger.info(f"Updating MT5 bars: {symbol}")

    # Get last saved timestamp
    try:
        df = read_mtbars(symbol, limit=1)
        if not df.empty:
            last_time = df.index[-1].to_pydatetime()
            date_from = last_time - timedelta(hours=10)  # overlap 10 bar for safety
        else:
            date_from = datetime.now(TZ) - timedelta(hours=MAX_HISTORY_HOURS)
    except Exception as e:
        logger.warning(f"Failed to read last MT5 bar for {symbol}: {e}")
        date_from = datetime.now(TZ) - timedelta(hours=MAX_HISTORY_HOURS)

    date_to = datetime.now(TZ)
    if date_to - date_from < timedelta(hours=10):
        logger.info(f"No new klines for {symbol}")

    try:
        new_bars = get_mt5_bars(symbol,interval='1h',date_from=date_from,date_to=date_to)
        if not new_bars.empty:
            saved = save_mtbars(new_bars, symbol)
            logger.success(f"Saved {saved} new MT5 bars for {symbol}")
        else:
            logger.info(f"No new MT5 bars for {symbol}")
    except Exception as e:
        logger.error(f"Failed to update MT5 bars for {symbol}: {e}")


# ------------------------------------------------------------------ #
# Update Binance Klines
# ------------------------------------------------------------------ #
def update_binance_klines(symbol: str) -> None:
    """Download and save latest 1H Binance klines."""
    bnb_symbol = get_binance_symbol(symbol)
    if not bnb_symbol:
        logger.warning(f"No Binance mapping for {symbol} — skipping klines")
        return

    logger.info(f"Updating Binance klines: {symbol} → {bnb_symbol}")

    try:
        df = read_klines(bnb_symbol, limit=1)
        if not df.empty:
            last_time = df.index[-1].to_pydatetime()
            date_from = last_time - timedelta(hours=10)  # overlap 10 bar for safety
        else:
            date_from = datetime.now(TZ) - timedelta(hours=MAX_HISTORY_HOURS)
    except Exception as e:
        logger.warning(f"Failed to read last kline for {bnb_symbol}: {e}")
        date_from = datetime.now(TZ) - timedelta(hours=MAX_HISTORY_HOURS)

    date_to = datetime.now(TZ)
    if date_to - date_from < timedelta(hours=1):
        logger.info(f"No new klines for {bnb_symbol}")

    try:
        new_klines = get_klines(bnb_symbol,interval='1h',start_time=date_from,end_time=date_to)
        if not new_klines.empty:
            saved = save_klines(new_klines, bnb_symbol)
            logger.success(f"Saved {saved} new klines for {bnb_symbol}")
        else:
            logger.info(f"No new klines for {bnb_symbol}")
    except Exception as e:
        logger.error(f"Failed to update klines for {bnb_symbol}: {e}")


# ------------------------------------------------------------------ #
# Update Symbol Specs
# ------------------------------------------------------------------ #
def update_symbol_specs() -> None:
    """Update symbols table from MT5."""
    logger.info("Updating symbol specifications...")

    try:
        df = get_mt5_symbols(asset_class='Crypto')
        if not df.empty:
            saved = save_symbols(df)
            logger.success(f"Updated {saved} symbols")
        else:
            logger.warning("No symbols fetched from MT5")
    except Exception as e:
        logger.error(f"Failed to update symbols: {e}")


# ------------------------------------------------------------------ #
# Main Update Routine
# ------------------------------------------------------------------ #
def update_all_data() -> None:
    """Update all required data before rebalancing."""

    # 1. MT5 connection
    if not is_mt5_ready():
        logger.critical("MT5 init failed — aborting update")
        return

    try:
        # 2. Update symbols
        update_symbol_specs()

        # 3. Update MT5 bars + Binance klines
        for symbol in PORTFOLIO:
            update_mt5_bars(symbol)
            update_binance_klines(symbol)

        logger.success("All data updated successfully!")

    except Exception as e:
        logger.critical(f"Data update failed: {e}", exc_info=True)