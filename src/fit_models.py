from datetime import datetime
from config import TZ
from config import logger
import api_bnb as bn
import api_mt5 as mt
import db_read as db
from strat_models import EmaVolStrategy
from strat_io import save_model,load_model

logger.info("Starting rebalancing session...")
mt.init_mt5()

capital = mt.read_balance()['equity']
portfolio = [
    "BTCUSD",   # Bitcoin
    "ETHUSD",   # Ethereum
    "BNBUSD",   # BNB
    "XRPUSD",
    "SOLUSD",   # Solana
    "DOGEUSD",  # Dogecoin
    "ADAUSD",
    "LTCUSD"
]
n_asset = len(portfolio)
capital_per_asset = capital / n_asset

for symbol in portfolio:
    contract_value = db.get_contract_value(symbol)
    symbol_bnb = db.get_binance_symbol(symbol)
    bars = db.read_klines(symbol_bnb,limit=24*360*10)

    for fast in [12,24,48,96]:
        strat = EmaVolStrategy(
            fast_ema_window=fast,
            slow_ema_multiplier=2,
            vol_window=24*30,
            weibull_c=2,
            target_vol=0.5,
            strat_weight=0.25
        ) 
        strat.fit(bars['close'])
        save_model(
            model=strat,
            symbol=symbol,
            model_name=f'EmaVolStrategy{fast}'
        )

logger.success("Done!")