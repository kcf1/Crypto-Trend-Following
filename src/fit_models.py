from datetime import datetime
from config import TZ,PORTFOLIO
from config import logger
import api_bnb as bn
import api_mt5 as mt
import db_read as db
from strat_models import EmaVolStrategy,AccelVolStrategy,BreakVolStrategy,BlockVolStrategy,WedThuStrategy,RevStrategy,OrthAlphaStrategy
from strat_io import save_model,load_model,reset_models

logger.info("Starting model refitting session...")
target_vol = 0.30

for symbol in PORTFOLIO:
    contract_value = db.get_contract_value(symbol)
    symbol_bnb = db.get_binance_symbol(symbol)
    #bars = db.read_klines(symbol_bnb,limit=24*360*10)
    bars = db.read_mtbars(symbol,limit=24*360*10)
    bars['volume'] = bars['tick_volume']

    reset_models(symbol)
    # ------------------------------------------------------------------ #
    # 1. EMA Standardized × Vol State → Position (% of capital)
    # ------------------------------------------------------------------ #
    strat_weight = 0.20
    variants = [24,48,96,192]
    variant_weight = strat_weight / len(variants)
    if strat_weight > 1e-6:
        for fast_ema_window in variants:
            strat = EmaVolStrategy(
                fast_ema_window=fast_ema_window,
                slow_ema_multiplier=2,
                vol_window=24*30,
                weibull_c=2,
                alpha=1.0,
                fit_decay=True,
                target_vol=target_vol,
                strat_weight=variant_weight
            )
            strat.fit(bars)
            save_model(
                model=strat,
                symbol=symbol,
                model_name=f'EmaVolStrategy_{fast_ema_window:0>4}'
            )
            
    # ------------------------------------------------------------------ #
    # 2. EMA Acceleration × Vol State → Position (% of capital)
    # ------------------------------------------------------------------ #
    strat_weight = 0.15
    variants = [24,48,96,192]
    variant_weight = strat_weight / len(variants)
    if strat_weight > 1e-6:
        for fast_ema_window in variants:
            strat = AccelVolStrategy(
                fast_ema_window=fast_ema_window,
                slow_ema_multiplier=2,
                diff_multiplier=1.0,
                vol_window=24*30,
                weibull_c=2,
                alpha=1.0,
                fit_decay=True,
                target_vol=target_vol,
                strat_weight=variant_weight
            )
            strat.fit(bars)
            save_model(
                model=strat,
                symbol=symbol,
                model_name=f'AccelVolStrategy_{fast_ema_window:0>4}'
            )
        
    # ------------------------------------------------------------------ #
    # 3. Breakout (smoothed) × Vol State → Position (% of capital)
    # ------------------------------------------------------------------ #
    strat_weight = 0.20
    variants = [48,96,192,384]
    variant_weight = strat_weight / len(variants)
    if strat_weight > 1e-6:
        for breakout_window in variants:
            strat = BreakVolStrategy(
                breakout_window=breakout_window,
                smooth_window=12,
                vol_window=24*30,
                weibull_c=2,
                alpha=1.0,
                fit_decay=True,
                target_vol=target_vol,
                strat_weight=variant_weight
            ) 
            strat.fit(bars)
            save_model(
                model=strat,
                symbol=symbol,
                model_name=f'BreakoutVolStrategy_{breakout_window:0>4}'
            )
            
    # ------------------------------------------------------------------ #
    # 4. Block Momentum (Higher High + Higher Low) × Vol Tilt → Position
    # ------------------------------------------------------------------ #
    strat_weight = 0.15
    variants = [48,96,192,384]
    variant_weight = strat_weight / len(variants)
    if strat_weight > 1e-6:
        for block_window in variants:
            strat = BlockVolStrategy(
                block_window=block_window,
                smooth_window=12,
                vol_window=24*30,
                weibull_c=2,
                alpha=1.0,
                fit_decay=True,
                target_vol=target_vol,
                strat_weight=variant_weight
            ) 
            strat.fit(bars)
            save_model(
                model=strat,
                symbol=symbol,
                model_name=f'BlockVolStrategy_{block_window:0>4}'
            )
        
    # ------------------------------------------------------------------ #
    # 5. Long Wed / Short Thu → Position
    # ------------------------------------------------------------------ #
    strat_weight = 0.10
    variants = [24*60,24*180]
    variant_weight = strat_weight / len(variants)
    if strat_weight > 1e-6:
        for vol_window in variants:
            strat = WedThuStrategy(
                vol_window=vol_window,
                target_vol=target_vol,
                strat_weight=variant_weight
            )
            strat.fit(bars)
            save_model(
                model=strat,
                symbol=symbol,
                model_name=f'WedThuStrategy_{vol_window:0>4}'
            )
            
    # ------------------------------------------------------------------ #
    # 6. Short-term Reversal → Position
    # ------------------------------------------------------------------ #
    strat_weight = 0.10
    variants = [6,9,12,15]
    variant_weight = strat_weight / len(variants)
    if strat_weight > 1e-6:
        for reversal_window in variants:
            strat = RevStrategy(
                vol_window=24*30,
                reversal_window=reversal_window,
                reversal_threshold=2.0,
                volume_threshold=0.3,
                target_vol=target_vol,
                strat_weight=variant_weight
            )
            strat.fit(bars)
            save_model(
                model=strat,
                symbol=symbol,
                model_name=f'RevStrategy_{reversal_window:0>4}'
            )
            
    # ------------------------------------------------------------------ #
    # 7. Momentum-Orthogonal Alpha → Position
    # ------------------------------------------------------------------ #
    strat_weight = 0.20
    variants = [24,48,96,192]
    variant_weight = strat_weight / len(variants)
    if strat_weight > 1e-6:
        for forward_window in variants:
            strat = OrthAlphaStrategy(
                forward_window=forward_window,
                vol_window=24*30,
                regression_window=24*30,
                alpha=1.0,
                fit_decay=True,
                target_vol=target_vol,
                strat_weight=variant_weight
            )
            strat.fit(bars)
            save_model(
                model=strat,
                symbol=symbol,
                model_name=f'OrthAlphaStrategy_{forward_window:0>4}'
            )
            
logger.success("Done!")