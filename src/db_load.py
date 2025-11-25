# database.py
import sqlite3
from pathlib import Path
from config import DB_PATH, logger
import pandas as pd
from db_schema import CREATE_TRADES_TABLE,INDEX_TRADES
from db_schema import CREATE_KLINES_TABLE,INDEX_KLINES
from db_schema import CREATE_SYMBOLS_TABLE,INDEX_SYMBOLS
from db_schema import CREATE_MTBARS_TABLE,INDEX_MTBARS

# ------------------------------------------------------------------ #
# Ensure DB folder exists
# ------------------------------------------------------------------ #
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------ #
# Init
# ------------------------------------------------------------------ #
def init_db() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        logger.info("Creating klines (1h) and trades tables...")
        cur.execute(CREATE_KLINES_TABLE)
        cur.execute(INDEX_KLINES)
        cur.execute(CREATE_TRADES_TABLE)
        cur.execute(INDEX_TRADES)
        cur.execute(CREATE_SYMBOLS_TABLE)
        cur.execute(INDEX_SYMBOLS)
        cur.execute(CREATE_MTBARS_TABLE)
        cur.execute(INDEX_MTBARS)
        conn.commit()
    logger.success(f"DB ready: {DB_PATH}")

# ------------------------------------------------------------------ #
# Save klines (interval fixed = 1h)
# ------------------------------------------------------------------ #
def save_klines(df: pd.DataFrame, symbol: str) -> int:
    if df.empty:
        logger.warning("Empty klines DataFrame")
        return 0

    # Must have open_time (klines) from get_bars
    if "open_time" not in df.columns:
        raise ValueError("DataFrame must contain 'open_time' (milliseconds)")

    df = df.copy()
    df["symbol"] = symbol.upper()

    cols = [
        "symbol", "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "num_trades",
        "taker_buy_base_vol", "taker_buy_quote_vol"
    ]
    data = df[cols]

    # Ensure integer klines
    data["open_time"] = data["open_time"].astype(int)
    data["close_time"] = data["close_time"].astype(int)
    data["num_trades"] = data["num_trades"].astype(int)

    upsert = f"""
    INSERT INTO klines ({",".join(data.columns)})
    VALUES ({",".join(["?"] * len(data.columns))})
    ON CONFLICT(symbol, open_time) DO UPDATE SET
        open = excluded.open,
        high = excluded.high,
        low = excluded.low,
        close = excluded.close,
        volume = excluded.volume,
        close_time = excluded.close_time,
        quote_volume = excluded.quote_volume,
        num_trades = excluded.num_trades,
        taker_buy_base_vol = excluded.taker_buy_base_vol,
        taker_buy_quote_vol = excluded.taker_buy_quote_vol
    """

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        rows = data.to_records(index=False).tolist()
        cur.executemany(upsert, rows)
        conn.commit()

    count = len(rows)
    logger.info(f"Saved {count} 1h klines for {symbol}")
    return count

# ------------------------------------------------------------------ #
# Save trades (unchanged)
# ------------------------------------------------------------------ #
def save_trades(df: pd.DataFrame, symbol: str) -> int:
    if df.empty:
        return 0

    df = df.copy()
    df["symbol"] = symbol.upper()
    df["is_buyer_maker"] = df["is_buyer_maker"].astype(int)
    df["is_best_match"] = df["is_best_match"].astype(int)

    cols = ["id", "symbol", "price", "qty", "quote_qty", "time", "is_buyer_maker", "is_best_match"]
    data = df[cols]

    insert = f"""
    INSERT OR IGNORE INTO trades ({",".join(cols)})
    VALUES ({",".join(["?"] * len(cols))})
    """

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        rows = data.to_records(index=False).tolist()
        cur.executemany(insert, rows)
        conn.commit()

    count = cur.rowcount
    logger.info(f"Saved {count} new trades for {symbol}")
    return count

def save_symbols(df: pd.DataFrame) -> int:
    """
    Save symbols to DB with upsert.
    
    Returns:
        Number of rows inserted/updated
    """
    if df.empty:
        logger.warning("Empty DataFrame → nothing to save")
        return 0

    df = df.copy()

    cols = [
        'symbol', 'symbol_bnb', 'class', 'currency_base', 'currency_profit',
        'price', 'spread', 'trade_contract_value', 'trade_contract_size',
        'trade_tick_value', 'trade_tick_size'
    ]
    data = df[cols]

    # Upsert SQL
    placeholders = ",".join(["?"] * len(cols))
    insert = f"""
    INSERT INTO symbols ({",".join(cols)})
    VALUES ({placeholders})
    ON CONFLICT(symbol) DO UPDATE SET
        symbol_bnb = excluded.symbol_bnb,
        class = excluded.class,
        currency_base = excluded.currency_base,
        currency_profit = excluded.currency_profit,
        price = excluded.price,
        spread = excluded.spread,
        trade_contract_value = excluded.trade_contract_value,
        trade_contract_size = excluded.trade_contract_size,
        trade_tick_value = excluded.trade_tick_value,
        trade_tick_size = excluded.trade_tick_size
    """

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        rows = data.to_records(index=False).tolist()
        cur.executemany(insert, rows)
        conn.commit()

    count = len(rows)
    logger.info(f"Saved/updated {count} symbols in DB")

def save_mtbars(
    df: pd.DataFrame,
    symbol: str
) -> int:
    """
    Save 1H bars to mtbars table with upsert.
    """
    if df.empty:
        logger.warning("Empty DataFrame")
        return 0

    df = df.copy()
    df['symbol'] = symbol.upper()

    cols = [
        'symbol', 'time', 'open', 'high', 'low', 'close',
        'tick_volume', 'spread', 'real_volume'
    ]
    data = df[cols]

    insert = f"""
    INSERT INTO mtbars ({",".join(cols)})
    VALUES ({",".join(["?"] * len(cols))})
    ON CONFLICT(symbol, time) DO UPDATE SET
        open = excluded.open,
        high = excluded.high,
        low = excluded.low,
        close = excluded.close,
        tick_volume = excluded.tick_volume,
        spread = excluded.spread,
        real_volume = excluded.real_volume
    """

    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        rows = data.to_records(index=False).tolist()
        cur.executemany(insert, rows)
        conn.commit()

    count = len(rows)
    logger.info(f"Saved {count} 1H mtbars")

if __name__ == "__main__":
    init_db()