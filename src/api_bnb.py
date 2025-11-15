import requests
import pandas as pd
from typing import Union, Optional, List
from datetime import datetime
import time
from config import BINANCE_BASE, logger

def _to_milliseconds(
    dt: Union[str, datetime, pd.Timestamp, int, float]
) -> int:
    """
    Convert various datetime formats to milliseconds since epoch (UTC).
    Supported inputs:
        - int/float: milliseconds or seconds
        - str: ISO format, e.g. '2025-01-01', '2025-01-01 12:00:00'
        - datetime, pd.Timestamp
    """
    if dt is None:
        return None

    if isinstance(dt, (int, float)):
        if dt > 10**12:  # likely milliseconds
            return int(dt)
        else:  # likely seconds
            return int(dt * 1000)

    if isinstance(dt, str):
        # Handle common formats
        dt = pd.to_datetime(dt, utc=True)
    elif isinstance(dt, datetime):
        dt = pd.to_datetime(dt, utc=True)
    elif isinstance(dt, pd.Timestamp):
        dt = dt.tz_convert('UTC') if dt.tzinfo else dt.tz_localize('UTC')
    else:
        raise ValueError(f"Unsupported time type: {type(dt)}")

    return int(dt.timestamp() * 1000)

def get_klines(
    symbol: str,
    interval: str = '1h',
    start_time: Optional[Union[str, datetime, pd.Timestamp, int, float]] = None,
    end_time: Optional[Union[str, datetime, pd.Timestamp, int, float]] = None,
    limit: int = 1000,
    time_zone: str = "0",
) -> pd.DataFrame:
    """
    Download Kline/Candlestick data from Binance API.

    Parameters:
    -----------
    symbol : str
        Trading pair symbol, e.g., 'BTCUSDT'
    interval : str
        Kline interval: '1m', '5m', '1h', '1d', etc.
    start_time : int, optional
        Start time in milliseconds (UTC). If None, fetches from most recent.
    end_time : int, optional
        End time in milliseconds (UTC). If None, fetches up to now.
    limit : int, default 1000
        Number of klines per request. Max 1000.
    time_zone : str, default "0"
        Timezone offset like '0', '-5:30', '+08:00'. Only affects interval parsing.
    base_url : str
        Binance API base URL (use testnet if needed)

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns:
        ['open_time', 'open', 'high', 'low', 'close', 'volume',
         'close_time', 'quote_volume', 'num_trades',
         'taker_buy_base_vol', 'taker_buy_quote_vol']
        Datetime index based on open_time (UTC).
    """
    endpoint = "/api/v3/klines"
    url = BINANCE_BASE + endpoint
    # Validate inputs
    valid_intervals = [
        '1s', '1m', '3m', '5m', '15m', '30m',
        '1h', '2h', '4h', '6h', '8h', '12h',
        '1d', '3d', '1w', '1M'
    ]
    if interval not in valid_intervals:
        raise ValueError(f"Invalid interval. Choose from: {', '.join(valid_intervals)}")

    if limit > 1000 or limit < 1:
        raise ValueError("Limit must be between 1 and 1000")

    params = {
        'symbol': symbol.upper(),
        'interval': interval,
        'limit': limit,
        'timeZone': time_zone
    }
    if start_time:
        start_time = _to_milliseconds(start_time)
        params['startTime'] = start_time
    if end_time:
        end_time = _to_milliseconds(end_time)
        params['endTime'] = end_time

    data = []
    while True:
        response = requests.get(url, params=params)
        if response.status_code != 200:
            raise Exception(f"API error {response.status_code}: {response.text}")

        batch = response.json()
        if not batch:
            break

        data.extend(batch)

        # Check if we got fewer than limit -> last batch
        if len(batch) < limit:
            break

        # Update startTime to continue from last kline
        last_open_time = batch[-1][0]  # open time of last kline
        params['startTime'] = last_open_time + 1  # next millisecond

        # Respect rate limits (Binance weight: 2 per request)
        time.sleep(0.01)

        # Break if end_time is reached
        if end_time and last_open_time >= end_time:
            break

    if not data:
        return pd.DataFrame()

    # Convert to DataFrame
    columns = [
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'num_trades',
        'taker_buy_base_vol', 'taker_buy_quote_vol', 'ignore'
    ]
    df = pd.DataFrame(data, columns=columns)

    # Drop unused column
    df = df.drop(columns=['ignore'])

    # Convert types
    numeric_cols = ['open', 'high', 'low', 'close', 'volume',
                    'quote_volume', 'taker_buy_base_vol', 'taker_buy_quote_vol']
    df[numeric_cols] = df[numeric_cols].astype(float)

    time_cols = ['open_time', 'close_time']
    df[time_cols] = df[time_cols].astype(int)

    # Convert timestamps to datetime
    #df['open_time_dt'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
    #df['close_time_dt'] = pd.to_datetime(df['close_time'], unit='ms', utc=True)

    # Set index
    df = df.sort_values('open_time')

    # Optional: filter by end_time if provided
    if end_time:
        #end_dt = pd.to_datetime(end_time, unit='ms', utc=True)
        df = df.loc[df['close_time'] <= end_time]

    logger.success(f"[{symbol}-{df['open_time'].min()}-{df['open_time'].max()}] Downloaded {len(df)} klines")
    return df

def get_recent_trades(
    symbol: str,
    limit: int = 500
) -> pd.DataFrame:
    """
    Fetch recent trades for a symbol from Binance.

    Parameters:
    -----------
    symbol : str
        Trading pair, e.g. 'BTCUSDT'
    limit : int
        Number of trades to return. Default: 500, Max: 1000.

    Returns:
    --------
    pd.DataFrame
        Columns:
        - id: int
        - price: float
        - qty: float
        - quote_qty: float
        - time: int (ms)
        - time_dt: pd.Timestamp (UTC)
        - is_buyer_maker: bool
        - is_best_match: bool
        Sorted by time (ascending)
    """
    endpoint = "/api/v3/trades"
    url = BINANCE_BASE + endpoint

    # Validate limit
    if limit < 1 or limit > 1000:
        raise ValueError("limit must be between 1 and 1000")

    params = {
        'symbol': symbol.upper(),
        'limit': limit
    }

    # Rate limit safety (weight: 25 → be conservative)
    time.sleep(0.01)

    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Binance API error {response.status_code}: {response.text}")

    data = response.json()

    if not data:
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Rename and type-convert
    df = df.rename(columns={
        'quoteQty': 'quote_qty',
        'isBuyerMaker': 'is_buyer_maker',
        'isBestMatch': 'is_best_match'
    })

    # Type conversions
    df['id'] = df['id'].astype(int)
    df['price'] = df['price'].astype(float)
    df['qty'] = df['qty'].astype(float)
    df['quote_qty'] = df['quote_qty'].astype(float)
    df['time'] = df['time'].astype(int)
    df['is_buyer_maker'] = df['is_buyer_maker'].astype(bool)
    df['is_best_match'] = df['is_best_match'].astype(bool)

    # Add datetime column
    #df['time_dt'] = pd.to_datetime(df['time'], unit='ms', utc=True)

    # Sort by time (oldest first)
    df = df.sort_values('time').reset_index(drop=True)

    logger.success(f"[{symbol}-{df['time'].min()}-{df['time'].max()}] Downloaded {len(df)} recent trades")
    return df

