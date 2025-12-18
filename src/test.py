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
from itertools import combinations,product
from tqdm import tqdm
import seaborn as sns
from api_mt5 import get_mt5_bars,init_mt5
from fast_func import rogers_satchell_volatility


symbols = [
    "BTCUSD",   # Bitcoin
    "ETHUSD",   # Ethereum
    "BNBUSD",   # BNB
]

all_prc = pd.DataFrame()
start_date = datetime(2018,1,1)
end_date = datetime(2025,12,1)

pnls = pd.DataFrame()

r,c = 2,4
fig,axes = plt.subplots(r,c)
for i,symbol in tqdm(enumerate(symbols)):
    #bars = read_klines(symbol+'T',limit=24*360*5)
    #vlm_bnb = bars['volume']
    bars = read_mtbars(symbol,limit=24*360*5)
    #vlm_mt = bars['tick_volume']

    #corrs = []
    #for n in range(1,97):
    #    n = 3
    #    vlm_bnb = vlm_bnb.ewm(span=n).mean()
    #    vlm_bnb = vlm_bnb / vlm_bnb.ewm(span=24*30).std()
    #    vlm_mt = vlm_mt.ewm(span=n).mean()
    #    vlm_mt = vlm_mt / vlm_mt.ewm(span=24*30).std()
    #    corrs.append(vlm_bnb.corr(vlm_mt))
    #corrs = pd.Series(corrs,index=range(1,97))

    #ax1,ax2 = axes[0,i%c],axes[1,i%c]
    #corrs.plot()
    #vlm_bnb.rolling(24*360).corr(vlm_mt).plot(ax=ax1)

    prc = bars['close']
    vlm = bars['tick_volume']

    n = 34
    vlm_z = vlm.ewm(span=24).mean()
    vlm_z = (vlm_z - vlm_z.ewm(span=24*30).mean()) / vlm_z.ewm(span=24*30).std()
    vlm_z = vlm_z
    #vlm_z = vlm_z.clip(-2,2)
    vlm_tilt = (1 - pd.Series(norm.cdf(vlm_z),index=vlm_z.index).clip(lower=0.8))
    #vlm_tilt.plot()
    #plt.show()
    dp = prc.ewm(span=24).mean() - prc.ewm(span=48).mean()
    dp = dp / dp.ewm(span=24*30).std()
    dp = dp.clip(-2,2)
    vol = rogers_satchell_volatility(bars,window=24*12)

    regress_incremental(vlm_z,bars,horizon=24)

    r1 = np.log(prc).diff().shift(-1)
    pos = (dp*vlm_tilt) * .3 / 95 / vol
    pnl1 = pos * r1 - pos.diff().abs() * 0.0010

    ax1,ax2 = axes[0,i%c],axes[1,i%c]
    pnl_curve(pnl1,ax=ax1)
    pnls[symbol] = pnl1
#pnl_curve(pnls.mean(axis=1))
plt.show()