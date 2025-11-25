from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from config import TZ, MODEL_DIR, logger
from api_mt5 import is_mt5_ready, read_balance, adjust_position
from db_read import get_contract_value, get_binance_symbol, read_klines, read_mtbars
from strat_models import BaseStrategy
from strat_io import load_model
from config import PORTFOLIO,MAX_HISTORY_HOURS,MAX_POSITION_PCT

# ------------------------------------------------------------------ #
# Core Rebalance Logic
# ------------------------------------------------------------------ #
def rebalance_portfolio() -> None:
    """Rebalance all assets in portfolio."""

    # --- Initialize MT5 ---
    if not is_mt5_ready():
        logger.critical("MT5 initialization failed. Aborting.")
        return

    # --- Get account equity ---
    try:
        account_stats = read_balance()
        #capital = account_stats['equity']
        capital = 10000 # fixed capital since lots too sensitive
        logger.info(f"Account equity: {capital:,.2f} {account_stats['currency']}")
    except Exception as e:
        logger.critical(f"Failed to read balance: {e}")
        return

    # --- Allocate capital ---
    n_assets = len(PORTFOLIO)
    capital_per_asset = capital / n_assets
    logger.info(f"Capital per asset: {capital_per_asset:,.2f} ({n_assets} assets)")

    # --- Process each symbol ---
    for symbol in PORTFOLIO:
        try:
            _rebalance_asset(symbol, capital_per_asset)
        except Exception as e:
            logger.error(f"Failed to rebalance {symbol}: {e}", exc_info=True)

    logger.success("Rebalancing complete!")


def _rebalance_asset(symbol: str, capital_allocation: float) -> None:
    """Rebalance a single asset."""
    logger.info(f"Processing {symbol}...")

    # --- 1. Get contract value ---
    contract_value = get_contract_value(symbol)
    if contract_value is None:
        logger.warning(f"Skipping {symbol}: no contract value in DB")
        return

    all_in_lots = capital_allocation / contract_value
    logger.debug(f"{symbol}: all-in lots = {all_in_lots:.2f}")

    # --- 2. Get Binance symbol & price data ---
    symbol_bnb = get_binance_symbol(symbol)
    if not symbol_bnb:
        logger.warning(f"Skipping {symbol}: no Binance mapping")
        return

    try:
        #bars = read_klines(symbol_bnb, limit=MAX_HISTORY_HOURS)#.iloc[:-1]
        bars = read_mtbars(symbol, limit=MAX_HISTORY_HOURS)#.iloc[:-1]
        bars['volume'] = bars['tick_volume']
        if bars.empty or len(bars) < 100:
            logger.warning(f"Skipping {symbol}: insufficient data ({len(bars)} rows)")
            return
        latest_price = bars['close'].iloc[-1]
        logger.debug(f"{symbol} → {symbol_bnb}: latest price = {latest_price:,.6f}")
    except Exception as e:
        logger.error(f"Failed to read klines for {symbol_bnb}: {e}")
        return

    # --- 3. Load all models for this symbol ---
    model_dir = Path(f'{MODEL_DIR}/{symbol}')
    if not model_dir.exists():
        logger.warning(f"No models found for {symbol}: {model_dir}")
        return

    models: List[BaseStrategy] = []
    for model_file in model_dir.glob("*.joblib"):
        try:
            data = load_model(str(model_file))
            model = data['model']
            if not isinstance(model, BaseStrategy):
                logger.warning(f"Invalid model type in {model_file}")
                continue
            models.append(model)
            logger.debug(f"Loaded model: {model_file.name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_file}: {e}")

    if not models:
        logger.warning(f"No valid models for {symbol}")
        return

    # --- 4. Compute aggregate signal ---
    try:
        # Use latest close for all models
        #close_series = bars['close']
        position_pct = sum(strat.one_step_predict(bars) for strat in models)
        #position_pct = max(min(position_pct, 2.0), -2.0)  # cap at ±200%
        logger.info(f"{symbol}: aggregate signal = {position_pct:+.2%}")
    except Exception as e:
        logger.error(f"Signal computation failed for {symbol}: {e}")
        return

    # --- 5. Convert to lots ---
    target_lots = round(position_pct*10)/10 * all_in_lots
    target_lots = round(target_lots, 2)  # MT5 lot precision
    logger.info(f"{symbol}: target lots = {target_lots:+.2f}")

    # --- 6. Adjust position ---
    try:
        adjust_position(
            symbol=symbol,
            target_volume=target_lots,
            comment=f"trend"
        )
    except Exception as e:
        logger.error(f"Failed to adjust {symbol}: {e}")