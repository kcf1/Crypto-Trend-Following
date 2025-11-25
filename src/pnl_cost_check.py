import pandas as pd
import numpy as np
from api_bnb import get_klines
from db_read import read_klines,read_mtbars,read_symbols
from db_load import save_klines
from config import TZ
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
from statsmodels.api import OLS,add_constant
from utils import align_idx
from plotting import pnl_curve,scatter,hist,qqplot,realtime_pnl
from scipy.stats import gamma,weibull_min,lognorm
from strat_models import EmaVolStrategy,VolScaleStrategy,BreakVolStrategy
from sklearn.model_selection import train_test_split,GridSearchCV,TimeSeriesSplit
import seaborn as sns
from tqdm import tqdm
from strat_io import load_model
import matplotlib.ticker as mtick
from config import TZ, MODEL_DIR, logger,PORTFOLIO, MAX_HISTORY_HOURS
import os
from scipy.stats import norm

symbols = [
    "BTCUSD",   # Bitcoin
    "ETHUSD",   # Ethereum
    "BNBUSD",   # BNB
    "SOLUSD",   # Solana
    "DOGEUSD",  # Dogecoin
    "ADAUSD",
]


r,c = 2,3
port = pd.DataFrame()
port_lots = pd.DataFrame()
trading_start = datetime(2025,11,17,1,tzinfo=TZ)
#trading_start = datetime(2025,6,17,1,tzinfo=TZ)
trading_end = datetime.now(TZ)
days = (trading_end - trading_start)/timedelta(1)

capital = 10000
n_assets = len(symbols)
capital_per_asset = capital / n_assets
logger.info(f"Capital per asset: {capital_per_asset:,.2f} ({n_assets} assets)")
symb_df = read_symbols(asset_class='Crypto').set_index('symbol')
symb_df['notional'] = symb_df['trade_tick_value']/symb_df['trade_tick_size']
notional = symb_df['notional']
spread = symb_df['spread']

tcs = pd.Series()
for i,symbol in enumerate(symbols):
    poss = pd.DataFrame()
    df = read_mtbars(symbol,limit=360*24*4)
    ret = df['close'].diff().shift(-1)
    vol = ret.std()
    contract_value = notional[symbol] * df['close']
    slippage = 0.0010
    round_trip_cost = (spread[symbol] + slippage) * notional[symbol]
    commission = 0.0650
    ret = contract_value.diff().shift(-1)
    all_in_lots = capital_per_asset / contract_value
    tc = round_trip_cost / 2 + commission
    tc = (tc / contract_value).max()


    tcs[symbol] = tc/vol

print(tcs*95)