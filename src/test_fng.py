import pandas as pd
import numpy as np
from db_read import read_klines,read_mtbars
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
from statsmodels.api import OLS,add_constant
from utils import align_idx,reverse_sigmoid
from performance import regress_incremental
from plotting import pnl_curve,scatter,hist,qqplot,feat_stationarity
from scipy.stats import gamma,weibull_min,lognorm,norm
from fast_func import rolling_slope_vec,rolling_pca1,rolling_pca2
from statsmodels.regression.rolling import RollingOLS
from config import TZ
from ta import add_all_ta_features,add_momentum_ta
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from strat_models import EmaVolStrategy
from scipy import stats
import pytz

symbols = [
    "BTCUSD",   # Bitcoin
    "ETHUSD",   # Ethereum
    "BNBUSD",   # BNB
    #"SOLUSD",   # Solana
    #"DOGEUSD",  # Dogecoin
    #"ADAUSD",
    #"XRPUSD",
    #"LTCUSD"
]

local_tz = pytz.FixedOffset(60*8)

fng = pd.read_csv('data/fng.csv',skiprows=1,index_col=0)
fng.index.name = None
fng.columns = [0]
fng.index = (pd.to_datetime(fng.index) + timedelta(1,hours=1))
fng = fng.tz_localize(local_tz).tz_convert(TZ)
fng = fng[0]

r,c = 2,3
fig,axes = plt.subplots(r,c)
port = pd.DataFrame()
for i,symbol in enumerate(symbols[:]):
    bars = read_mtbars(symbol,limit=24*360*5)
    prc = bars['close'].reindex(fng.index)

    #fng = fng.diff()
    fng = fng.diff() / fng.diff().ewm(span=4*12).std()
    #fng = fng.mask(fng <= 10, np.nan)
    ret = np.log(prc).diff().shift(-1)
    vol = ret.ewm(span=4).std()

    pnl = fng * ret# / vol

    pnl_curve(pnl,ax=axes[0,i%c],freq=24*7)
    #scatter(fng,ret)
    continue

    #print(pos1.corr(pos2))


#(pc1-pc1.iat[0]).plot(secondary_y=True)
#ax = pnl_curve(port.mean(axis=1))
#port.cumsum().plot(ax=ax,secondary_y=True,alpha=0.5,linestyle='--')
plt.show()