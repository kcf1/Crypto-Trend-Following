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
from strat_models import *
from sklearn.model_selection import train_test_split,GridSearchCV,TimeSeriesSplit
import seaborn as sns
from tqdm import tqdm

symbols = [
    "BTCUSD",   # Bitcoin
    "ETHUSD",   # Ethereum
    "BNBUSD",   # BNB
    #"XRPUSD",
    #"SOLUSD",   # Solana
    #"DOGEUSD",  # Dogecoin
    #"ADAUSD",
    #"LTCUSD"
]
#symbols = ["BTCUSD","BNBUSD","ETHUSD","SOLUSD","FETUSD","XLMUSD","DASHUSD","DOGEUSD","BARUSD","ADAUSD"]

r,c = 2,3
fig,axes = plt.subplots(r,c)
port = pd.DataFrame()
stats = pd.DataFrame()
all_prc = pd.DataFrame({symbol:read_klines(symbol+'T')['close'] for symbol in symbols})

for i,symbol in enumerate(symbols):
    pnls = pd.DataFrame()
    poss = pd.DataFrame()
    #df = read_klines(symbol+'T',limit=24*360*4)
    #df = pd.concat([df,all_prc],axis=1).dropna()
    df = read_mtbars(symbol)
    df['volume'] = df['tick_volume']
    #prc = df['close']
    ret = np.log(df['close']).diff().shift(-1)
    
    if True:
    #try:
        x,y = align_idx(df,ret)
        x1, x2, y1, y2 = train_test_split(
            x, y, 
            test_size=0.5,
            shuffle=False  # This ensures sequential split
        )
        strat_weight = 0.10
        target_vol = 0.3
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
                strat.fit(x)
                pos = strat.predict(x)
                poss[reversal_window] = pos
                pnl = pos * y - pos.diff().abs() * .0010
                pnls[reversal_window] = pnl

        strat = VolScaleStrategy(
            vol_window=24*30,
            target_vol=0.5,
            strat_weight=1
        )
        strat.fit(x)
        pos = strat.predict(x)
        ax1 = axes[0,i%c]
        pnl = pos * y - pos.diff().abs() * .0010
        #pnl.cumsum().plot(linestyle='--',color='grey',alpha=0.5,ax=ax1)
        pos = poss.sum(axis=1)
        pnl = pos * y - pos.diff().abs() * .0010
        pnl_curve(pnl,ax=ax1)
        ax1.set_title(symbol+' (10bps tc)')
        pnls.cumsum().plot(linestyle='--',alpha=0.5,ax=ax1,secondary_y=True)
        #ax1.legend(['VolScaled','EmaDiff'])
        port[symbol] = pnl
        ax2 = axes[1,i%c]
        sns.heatmap(pnls.corr(),annot=True,ax=ax2)
    #except: pass
ax = pnl_curve(port.mean(axis=1))
port.cumsum().plot(linestyle='--',ax=ax,alpha=0.5)
#stats_dist = stats.T.describe().T#.round(2)
#stats_dist.to_csv('plot/stats.csv')
plt.show()
#sns.heatmap(port.corr(),annot=True)