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
from plotting import pnl_curve,scatter,hist,qqplot,corr_dendrogram,monthly_sharpe
from scipy.stats import gamma,weibull_min,lognorm
from strat_models import EmaVolStrategy,VolScaleStrategy,BreakVolStrategy,SlopeVolStrategy,BlockVolStrategy,AccelVolStrategy,WedThuStrategy,RevStrategy
from sklearn.model_selection import train_test_split,GridSearchCV,TimeSeriesSplit
import seaborn as sns
from tqdm import tqdm
from strat_io import load_model
from config import TZ, MODEL_DIR, logger,PORTFOLIO
import os
from scipy.stats import norm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from performance import calcaulte_pnl_stats

symbols = PORTFOLIO

r,c = 3,len(symbols)
fig,axes = plt.subplots(r,c)
port = pd.DataFrame()
bnch = pd.DataFrame()
for i,symbol in enumerate(symbols):
    poss = dict()
    df = read_mtbars(symbol,limit=24*360*3)
    df['volume'] = df['tick_volume']
    ret = np.log(df['close']).diff().shift(-1)
    #df = read_klines(symbol+'T',limit=24*360*5)
    
    if True:
    #try:
        x,y = align_idx(df,ret)
        
        #funding_hour = pd.Series(np.where((x.index.hour == 23)|(x.index.hour == 7)|(x.index.hour == 15), 1, 0), x.index)
        #x = x.loc[funding_hour==1].reindex(x.index).ffill()
        model_dir = f'{MODEL_DIR}/{symbol}'
        for name in os.listdir(model_dir):
            strat = load_model(f'{model_dir}/{name}')['model']
            pos = strat.predict(x)
            name,variant = name.split('_')[:2]
            poss[(name,variant)] = pos

        if False:
            strat_weight = 0.05
            variants = [394,788]
            variant_weight = 1 / len(variants) * strat_weight
            
            for fast_ema_window in variants:
                strat = EmaVolStrategy(
                    fast_ema_window=fast_ema_window,
                    slow_ema_multiplier=2,
                    vol_window=24*30,
                    weibull_c=2,
                    alpha=1.0,
                    target_vol=.3,
                    strat_weight=variant_weight
                )
                strat.fit(df)
                pos = strat.predict(x)
                poss[('NewEmaVolStrategy',fast_ema_window)] = pos

        strat = VolScaleStrategy(
            vol_window=24*30,
            target_vol=0.3,
            strat_weight=1
        )
        strat.fit(x)
        pos = strat.predict(x)
        benchmark = pos * y - pos.diff().abs() * .0010
        ax1 = axes[0,i%c]
        ax1.set_title(symbol+' (10bps tc)')
        benchmark.cumsum().plot(linestyle='--',alpha=0.5,ax=ax1,color='grey')

        poss = pd.DataFrame(poss)
        pos = poss.sum(axis=1)
        pos = pos#[(pos.index.weekday != 5)].reindex(pos.index).ffill()
        port_pnl = pos * y - pos.diff().abs() * .0010
        pnl_curve(port_pnl,ax=ax1)
        #monthly_sharpe(port_pnl,ax=ax1)
        ax1.legend(['VolScaled','Combined'])
        port[symbol] = port_pnl
        bnch[symbol] = benchmark

        ax2 = axes[1,i%c]
        ax2.set_title(symbol+' by Strategy (10bps tc)')
        pos = poss.groupby(level=0,axis=1).sum()#[(poss.index.weekday != 5)].reindex(poss.index).ffill()
        pnl = pos.mul(y,axis=0) - pos.diff().abs() * .0010
        pnl = pnl * .3 / 95 / pnl.std()
        pnl.cumsum().plot(linestyle='--',ax=ax2)

        ax3 = axes[2,i%c]
        ax3.set_title(symbol+' Strat Corr')
        corr_dendrogram(poss.corr().fillna(0),ax=ax3)
    #except: pass
ax = pnl_curve(port.mean(axis=1))
stats = calcaulte_pnl_stats(port.mean(axis=1),benchmark=bnch.mean(axis=1))
ax.set_title('Portfolio (10bps tc)')
#port.cumsum().plot(linestyle='--',ax=ax)
plt.show()