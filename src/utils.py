# ml_utils.py
import pandas as pd
import numpy as np
from config import logger
from typing import Union
from tqdm import tqdm
from datetime import timedelta,datetime
import time
from scipy.stats.mstats import winsorize as scipy_winsorize

def to_milliseconds(
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

def align_idx(
    X: pd.Series | pd.DataFrame,
    y: pd.Series
) -> tuple[pd.Series, pd.Series | pd.DataFrame]:
    """
    Align y and X on common index, drop any row with NaN.

    Args:
        y: Target Series
        X: Features (Series or DataFrame)

    Returns:
        (y_clean, X_clean) — same index, no NaN
    """
    # Find common index
    common_idx = y.index.intersection(X.index)
    y = y.loc[common_idx]
    X = X.loc[common_idx]

    # Drop NaN in y
    mask = y.notna()

    # Drop NaN in X
    if isinstance(X, pd.DataFrame):
        mask &= X.notna().all(axis=1)
    else:  # X is Series
        mask &= X.notna()

    y_clean = y[mask]
    X_clean = X[mask]

    #logger.info(f"Aligned: {len(y_clean)} samples (from {len(common_idx)})")
    return X_clean, y_clean

def reverse_sigmoid(vol, center=0.75, steepness=10, floor=0.1):
    """
    [0,1] scaler: LOW in calm, HIGH in volatile regimes
    - center: percentile where boost starts (e.g. 75th %ile of vol)
    - steepness: how fast to ramp up
    - floor: minimum position in calm markets
    """
    pct = pd.Series(vol).expanding().rank(pct=True)  # [0,1] rank
    boost = 1 / (1 + np.exp(-steepness * (pct - center)))
    return floor + (1 - floor) * boost

def onhour_offset(offset_mins=0,offset_secs=0):
    now = datetime.now()
    tar_time = (now + timedelta(minutes=30)).replace(minute=0,second=0,microsecond=0) + timedelta(minutes=offset_mins) + timedelta(seconds=offset_secs)
    # Seconds delayed
    #tar_time = tar_time.replace(second=3)
    time_left = tar_time - now
    secs = int(time_left / timedelta(seconds=1))
    logger.info(f'Wait until {tar_time.strftime("%H:%M:%S")}...')
    for _ in tqdm(range(secs)): time.sleep(1)
    
def triple_barrier(
    returns: pd.Series,
    vol: float = 0.01,           # Fixed volatility per bar (e.g. 1%)
    tp_multiplier: float = 2.0,  # TP = +2%
    sl_multiplier: float = 1.0,  # SL = -1%
    max_holding: int = 10
) -> pd.DataFrame:
    """
    Sequential Triple Barrier Labeling (Non-overlapping, No Look-ahead)
    
    Returns ONLY one label per completed trade.
    Perfect for ML training, meta-labeling, and backtesting.
    """
    returns = returns.sort_index()
    idx = returns.index
    N = len(returns)
    
    events = []
    i = 0
    
    while i < N:
        start_idx = i
        start_time = idx[i]
        cum_ret = 0.0
        
        upper =  vol * tp_multiplier
        lower = -vol * sl_multiplier
        
        hit_tp = hit_sl = False
        exit_idx = min(i + max_holding, N)
        
        # Walk forward until barrier hit or max_holding reached
        for j in range(i + 1, i + max_holding + 1):
            if j >= N:
                exit_idx = N - 1
                break
                
            cum_ret += returns.iloc[j]
            
            if cum_ret >= upper:
                exit_idx = j
                hit_tp = True
                break
            if cum_ret <= lower:
                exit_idx = j
                hit_sl = True
                break
        
        # Record the completed trade
        exit_time = idx[exit_idx]
        final_return = cum_ret if hit_tp or hit_sl else \
                       (returns.iloc[i+1:exit_idx+1].sum() if exit_idx > i else 0.0)
        
        label = 1 if hit_tp else (-1 if hit_sl else 0)
        reason = "tp" if hit_tp else ("sl" if hit_sl else "timeout")
        
        events.append({
            'start_time': start_time,
            'end_time': exit_time,
            'duration': exit_idx - start_idx,
            'return': final_return,
            'label': label,
            'reason': reason
        })
        
        # CRITICAL: jump to day AFTER exit
        i = exit_idx + 1
        #i = i + 1
    
    return pd.DataFrame(events).set_index('start_time')

def winsorize(data: pd.Series | np.ndarray, 
                      lower_limit: float = 0.05, 
                      upper_limit: float = 0.05) -> pd.Series:
    """
    Returns a winsorized version of the input series.
    
    Parameters:
    - data: Input pandas Series or NumPy array.
    - lower_limit: Proportion to winsorize from the lower end (e.g., 0.05 for 5th percentile).
    - upper_limit: Proportion to winsorize from the upper end (e.g., 0.05 for 95th percentile).
    
    Returns:
    - Winsorized pandas Series.
    
    Note: Requires scipy installed.
    """
    if not isinstance(data, (pd.Series, np.ndarray)):
        raise ValueError("Input must be a pandas Series or NumPy array.")
    
    # winsorize returns a masked array if there are NaNs, but we convert back
    winsorized_array = scipy_winsorize(data, limits=[lower_limit, upper_limit])
    
    # If input is Series, preserve index and name
    if isinstance(data, pd.Series):
        return pd.Series(winsorized_array.data if np.ma.is_masked(winsorized_array) else winsorized_array,
                         index=data.index, name=data.name)
    
    return pd.Series(winsorized_array,data.index)