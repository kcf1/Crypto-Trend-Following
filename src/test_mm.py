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

start_date = datetime(2020,1,1)
end_date = datetime(2025,12,1)
init_mt5()
btc = get_mt5_bars('BTCUSD',date_from=start_date,date_to=end_date)['close']
neo = get_mt5_bars('NEOUSD',date_from=start_date,date_to=end_date)['close']

nr = np.log(neo).diff()
br = np.log(btc).diff()

print(nr.describe())
hist(nr)
plt.show()

nr.rolling(24).corr(br).plot()
plt.show()
exit()


all_prc = pd.DataFrame()
start_date = datetime(2018,1,1)
end_date = datetime(2025,12,1)

pnls = pd.DataFrame()

r,c = 2,4
fig,axes = plt.subplots(r,c)
for i,symbol in tqdm(enumerate(symbols)):
    bars = read_klines(symbol+'T',limit=24*360*2)
    prc = bars['close']

    ret = np.log(prc).diff().shift(-1)
    vol = rogers_satchell_volatility(bars,window=24*12)
    #ret = ret / vol
    for day,ret in ret.groupby(ret.index.weekday):
        ax = axes[day//4,day%4]
        hour_ret = ret.groupby(ret.index.hour).mean()
        hour_ret.plot(kind='bar',ax=ax)
        ax.axhline(hour_ret.mean()+2*hour_ret.std(),linestyle='-',color='red',alpha=0.5)
        ax.axhline(hour_ret.mean(),linestyle='--',color='black',alpha=0.5)
        ax.axhline(hour_ret.mean()-2*hour_ret.std(),linestyle='-',color='red',alpha=0.5)
    plt.show()
    exit()

    buy = (prc.index.hour >= 20) & (prc.index)
    we = fri | sat | sun | mon
    is_we = np.where(we, 1, -1)
    
    r = np.log(prc).diff()#.shift(-1)
    v = np.log(prc).diff().ewm(span=24*30).std()
    y = r/v

    regress_incremental(is_we,bars,horizon=24)

    r1 = np.log(prc).diff().shift(-1)
    pos = (y.rolling(24*5).mean())[we] * .3 / 95 / v
    pnl1 = pos * r1 - pos.diff().abs() * 0.0010

    ax1,ax2 = axes[0,i%c],axes[1,i%c]
    pnl_curve(pnl1,ax=ax1)
    pnls[symbol] = pnl1
plt.show()
pnl_curve(pnls.mean(axis=1))
plt.show()