# db_reader.py
import sqlite3
import pandas as pd
from typing import Optional, List
from config import DB_PATH, logger
from pathlib import Path
from utils import to_milliseconds

# ------------------------------------------------------------------ #
# 1. Read klines (1h candles)
# ------------------------------------------------------------------ #
def read_klines(
    symbol: str,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Read 1h klines for a symbol.

    Args:
        symbol: e.g. "BTCUSDT"
        start_time: ms (inclusive)
        end_time: ms (inclusive)
        limit: max rows (latest first)

    Returns:
        pd.DataFrame with open_time as index (UTC)
    """
    query = "SELECT * FROM klines WHERE symbol = ?"
    params: List = [symbol.upper()]

    if start_time:
        query += " AND open_time >= ?"
        params.append(to_milliseconds(start_time))
    if end_time:
        query += " AND open_time <= ?"
        params.append(to_milliseconds(end_time))

    query += " ORDER BY open_time DESC"
    if limit:
        query += " LIMIT ?"
        params.append(limit)

    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        if df.empty:
            logger.debug(f"No klines found for {symbol}")
            return df

        # Convert ms to datetime index
        df['open_time_dt'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        df = df.set_index('open_time_dt').sort_index()
        df.index.name = None
        
        logger.info(f"Read {len(df)} klines for {symbol}")
        return df

    except Exception as e:
        logger.error(f"Failed to read klines: {e}")
        raise


# ------------------------------------------------------------------ #
# 2. Read recent trades
# ------------------------------------------------------------------ #
def read_trades(
    symbol: str,
    since_id: Optional[int] = None,
    start_time: Optional[int] = None,
    limit: Optional[int] = 1000
) -> pd.DataFrame:
    """
    Read recent trades for a symbol.

    Args:
        symbol: e.g. "BTCUSDT"
        since_id: get trades with id > since_id
        start_time: ms
        limit: max rows

    Returns:
        pd.DataFrame with time_dt index
    """
    query = "SELECT * FROM trades WHERE symbol = ?"
    params: List = [symbol.upper()]

    if since_id:
        query += " AND id > ?"
        params.append(since_id)
    if start_time:
        query += " AND time >= ?"
        params.append(to_milliseconds(start_time))

    query += " ORDER BY time DESC"
    if limit:
        query += " LIMIT ?"
        params.append(limit)

    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        if df.empty:
            return df

        df['time_dt'] = pd.to_datetime(df['time'], unit='ms', utc=True)
        df = df.set_index('time_dt').sort_index()

        # Convert bool
        df['is_buyer_maker'] = df['is_buyer_maker'].astype(bool)
        df['is_best_match'] = df['is_best_match'].astype(bool)

        logger.info(f"Read {len(df)} trades for {symbol}")
        return df

    except Exception as e:
        logger.error(f"Failed to read trades: {e}")
        raise


# ------------------------------------------------------------------ #
# 3. Read symbols (full or filtered)
# ------------------------------------------------------------------ #
def read_symbols(
    asset_class: Optional[str] = None,
    currency_base: Optional[str] = None,
    currency_profit: Optional[str] = None
) -> pd.DataFrame:
    """
    Read symbols table with optional filters.

    Args:
        asset_class: e.g. "Forex"
        currency_base: e.g. "USD"
        currency_profit: e.g. "USDT"

    Returns:
        pd.DataFrame
    """
    query = "SELECT * FROM symbols"
    params: List = []
    conditions = []

    if asset_class:
        conditions.append("class = ?")
        params.append(asset_class)
    if currency_base:
        conditions.append("currency_base = ?")
        params.append(currency_base)
    if currency_profit:
        conditions.append("currency_profit = ?")
        params.append(currency_profit)

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(query, conn, params=params)
        
        logger.info(f"Read {len(df)} symbols")
        return df

    except Exception as e:
        logger.error(f"Failed to read symbols: {e}")
        raise

def get_binance_symbol(symbol: str) -> Optional[float]:
    """
    Read trade_contract_value for a symbol from the 'symbols' table.

    Args:
        symbol: str (e.g. "BTCUSD")

    Returns:
        float: trade_contract_value, or None if not found
    """
    query = """
    SELECT symbol_bnb 
    FROM symbols 
    WHERE symbol = ? 
    LIMIT 1
    """
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute(query, (symbol.upper(),))
            row = cursor.fetchone()
        
        if row is None:
            logger.warning(f"Symbol not found in DB: {symbol}")
            return None
        
        value = str(row[0])
        logger.debug(f"{symbol}: symbol_bnb = {value}")
        return value

    except Exception as e:
        logger.error(f"Failed to read binance symbol for {symbol}: {e}")
        raise

def get_contract_value(symbol: str) -> Optional[float]:
    """
    Read trade_contract_value for a symbol from the 'symbols' table.

    Args:
        symbol: str (e.g. "BTCUSD")

    Returns:
        float: trade_contract_value, or None if not found
    """
    query = """
    SELECT trade_contract_value 
    FROM symbols 
    WHERE symbol = ? 
    LIMIT 1
    """
    
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.execute(query, (symbol.upper(),))
            row = cursor.fetchone()
        
        if row is None:
            logger.warning(f"Symbol not found in DB: {symbol}")
            return None
        
        value = float(row[0])
        logger.debug(f"{symbol}: trade_contract_value = {value}")
        return value

    except Exception as e:
        logger.error(f"Failed to read contract value for {symbol}: {e}")
        raise
    
def read_mtbars(
    symbol: str,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    limit: Optional[int] = None
) -> pd.DataFrame:
    """
    Read 1H MT5 bars from 'mtbars' table.

    Args:
        symbol: e.g. "EURUSD"
        start_time: ms (UTC), inclusive
        end_time: ms (UTC), inclusive
        limit: max rows (latest first)

    Returns:
        pd.DataFrame with UTC datetime index
    """
    query = "SELECT * FROM mtbars WHERE symbol = ?"
    params = [symbol.upper()]

    if start_time:
        query += " AND time >= ?"
        params.append(to_milliseconds(start_time))
    if end_time:
        query += " AND time <= ?"
        params.append(to_milliseconds(end_time))

    query += " ORDER BY time DESC"
    if limit:
        query += " LIMIT ?"
        params.append(limit)

    try:
        with sqlite3.connect(DB_PATH) as conn:
            df = pd.read_sql_query(query, conn, params=params)

        if df.empty:
            logger.debug(f"No mtbars for {symbol}")
            return df

        # Convert ms → UTC datetime index
        df['time_dt'] = pd.to_datetime(df['time'], unit='ms', utc=True)
        df = df.set_index('time_dt').sort_index()
        df.index.name = None
        df = df.drop(columns=['time', 'symbol'])

        logger.info(f"Read {len(df)} mtbars for {symbol}")
        return df

    except Exception as e:
        logger.error(f"Failed to read mtbars: {e}")
        raise