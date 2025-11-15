# ml_utils.py
import pandas as pd
import numpy as np
from config import logger
from tqdm import tqdm
from datetime import timedelta,datetime
import time

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

    logger.info(f"Aligned: {len(y_clean)} samples (from {len(common_idx)})")
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

def onhour_offset(offset_mins=0):
    now = datetime.now()
    tar_time = (now + timedelta(minutes=30)).replace(minute=0,second=0,microsecond=0) + timedelta(minutes=offset_mins)
    time_left = tar_time - now
    secs = int(time_left / timedelta(seconds=1))
    logger.info(f'Wait until {tar_time.strftime("%H:%M:%S")}...')
    for _ in tqdm(range(secs)): time.sleep(1)