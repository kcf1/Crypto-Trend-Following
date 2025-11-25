import pandas as pd
import numpy as np
from api_bnb import get_klines
from db_read import read_klines,read_mtbars,read_symbols
from db_load import save_klines
from config import TZ
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.api import OLS,add_constant
from utils import align_idx
from plotting import pnl_curve,scatter,hist,qqplot
from scipy.stats import gamma,weibull_min,lognorm
from strat_models import EmaVolStrategy,EmaVlmVolStrategy,VolScaleStrategy,BreakVolStrategy
from sklearn.model_selection import train_test_split,GridSearchCV,TimeSeriesSplit
import seaborn as sns
from tqdm import tqdm

symbols = [
    "BTCUSD",   # Bitcoin
    "ETHUSD",   # Ethereum
    "BNBUSD",   # BNB
    #"XRPUSD",
    "SOLUSD",   # Solana
    "DOGEUSD",  # Dogecoin
    "ADAUSD",
    #"LTCUSD"
]
#symbols = ["BTCUSD","BNBUSD","ETHUSD","SOLUSD","FETUSD","XLMUSD","DASHUSD","DOGEUSD","BARUSD","ADAUSD"]
#symbols = read_symbols('Crypto')['symbol']
st_time = datetime(2015,1,1,tzinfo=TZ)
ed_time = datetime.now(TZ)
for symbol in tqdm(symbols[:]):
    try:
        bars = get_klines(symbol+'T',start_time=st_time,end_time=ed_time)
        save_klines(bars,symbol+'T')
    except Exception as e:
        print(e)