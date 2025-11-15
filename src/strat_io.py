# model_io.py
import joblib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from config import logger,MODEL_DIR
import os
import time

def save_model(
    model: Any,
    symbol: str,
    model_name: str = "strategy"
) -> str:
    """
    Save model with metadata.

    Args:
        model: fitted strategy object
        symbol: e.g. "BTCUSD"
        stats: dict from .get_pnl_stats()
        params: dict of init params
        model_name: base filename

    Returns:
        str: saved filepath
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{symbol}.joblib"
    filedir = f"{MODEL_DIR}/{symbol}"
    filepath = f"{filedir}/{filename}"
    if not os.path.exists(filedir):
        os.mkdir(filedir)

    # Package everything
    data = {
        'model': model,
        'symbol': symbol,
        'saved_at': datetime.now().isoformat(),
        'version': '1.0'
    }

    joblib.dump(data, filepath)
    logger.success(f"Model saved: {filepath}")
    return str(filepath)


def load_model(filepath: str) -> Dict[str, Any]:
    """
    Load model and metadata.

    Args:
        filepath: path to .joblib file

    Returns:
        dict with 'model', 'symbol', etc.
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Model not found: {filepath}")

    data = joblib.load(filepath)
    
    required = ['model', 'symbol']
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"Corrupted model file. Missing: {missing}")

    logger.success(f"Model loaded: {filepath} | Symbol: {data['symbol']}")
    return data