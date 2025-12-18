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
from plotting import pnl_curve,scatter,hist,qqplot,corr_dendrogram,monthly_sharpe,realtime_pnl
from scipy.stats import gamma,weibull_min,lognorm
from strat_models import *
from sklearn.model_selection import train_test_split,GridSearchCV,TimeSeriesSplit
import seaborn as sns
from tqdm import tqdm
from strat_io import load_model
from config import TZ, MODEL_DIR, logger,PORTFOLIO
import os
from scipy.stats import norm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from performance import calcaulte_pnl_stats
import matplotlib.ticker as mtick

symbols = PORTFOLIO
trading_start = datetime(2025,11,17,1,tzinfo=TZ)
#trading_start = datetime(2024,6,17,1,tzinfo=TZ)
trading_end = datetime.now(TZ)

info = read_symbols()
info = info.set_index('symbol')
info['commission'] = np.where(info['class']=='Crypto',.00065,0)
spread = info['spread']/info['price']
slippage = .0001
commission = info['commission']
cost = spread / 2 + slippage + commission
print(cost.sort_values())
print(commission.sort_values())
print(spread.sort_values())
#exit()


r,c = 3,len(symbols)
fig,axes = plt.subplots(r,c)
port = pd.DataFrame()
bnch = pd.DataFrame()
for i,symbol in enumerate(symbols):
    tc = cost[symbol]
    poss = dict()
    df = read_mtbars(symbol,limit=24*360*5)
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
            target_vol = 0.3
            strat_weight = 0.20
            variants = [24,48,96,192]
            variant_weight = 1 / len(variants) * strat_weight
            
            for forward_window in variants:
                strat = OrthAlphaStrategy(
                    forward_window=forward_window,
                    vol_window=24*30,
                    regression_window=24*30,
                    smooth_window=12,
                    fit_decay=True,
                    alpha=1.0,
                    target_vol=target_vol,
                    strat_weight=variant_weight
                )
                strat.fit(x)
                pos = strat.predict(x)
                poss[('OrthResidStrategy',forward_window)] = pos

        strat = VolScaleStrategy(
            vol_window=24*30,
            target_vol=0.20,
            strat_weight=1
        )
        strat.fit(x)
        pos = strat.predict(x)
        benchmark = pos * y - pos.diff().abs() * tc
        ax1 = axes[0,i%c]
        ax1.set_title(symbol+' (10bps tc)')
        benchmark.cumsum().plot(linestyle='--',alpha=0.5,ax=ax1,color='grey')

        poss = pd.DataFrame(poss)
        pos = poss.sum(axis=1)
        pos = pos#[(pos.index.weekday != 5)].reindex(pos.index).ffill()
        port_pnl = pos * y - pos.diff().abs() * tc

        pnl_curve(port_pnl,ax=ax1)
        #pnl_curve(port_pnl.loc[trading_start:],ax=ax1)
        #monthly_sharpe(port_pnl,ax=ax1)
        ax1.legend(['VolScaled','Combined'])
        port[symbol] = port_pnl
        bnch[symbol] = benchmark

        ax2 = axes[1,i%c]
        ax2.set_title(symbol+' by Strategy (10bps tc)')
        pos = poss.groupby(level=0,axis=1).sum()#[(poss.index.weekday != 5)].reindex(poss.index).ffill()
        pnl = pos.mul(y,axis=0) - pos.diff().abs() * tc
        pnl = pnl * 0.20 / 95 / pnl.std()
        pnl.cumsum().plot(linestyle='--',ax=ax2)
        #pnl.loc[trading_start:].cumsum().plot(linestyle='--',ax=ax2)

        ax3 = axes[2,i%c]
        ax3.set_title(symbol+' Strat Corr')
        corr_dendrogram(poss.corr().fillna(0),ax=ax3)
        #corr_dendrogram(poss.loc[trading_start:].corr().fillna(0),ax=ax3)
    #except: pass
port.mean(axis=1).to_csv('data/tf_single.csv')
ax = pnl_curve(port.mean(axis=1))
stats = calcaulte_pnl_stats(port.mean(axis=1),benchmark=bnch.mean(axis=1),print_results=True)
#ax = pnl_curve(port.mean(axis=1).loc[trading_start:])
#stats = calcaulte_pnl_stats(port.mean(axis=1).loc[trading_start:],benchmark=bnch.mean(axis=1).loc[trading_start:])
ax.set_title('Portfolio (10bps tc)')
#port.cumsum().plot(linestyle='--',ax=ax)
#plt.show()

if False:
    btc = read_mtbars(symbol,limit=24*360*5)['close']
    for i,pnl in port.resample('M'):
        #print(i)
        #print(pnl)
        idx = pnl.index
        prc = btc.reindex(idx)
        ax = realtime_pnl(pnl.mean(axis=1))
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=2))
        #pnl.cumsum().plot(linestyle='--',ax=ax,alpha=0.5,secondary_y=True)
        prc.plot(linestyle='--',ax=ax,alpha=0.5,secondary_y=True)
        dt = i.strftime('%Y%m')
        plt.savefig(f'plot/portfolio/monthly/CombStrategy_{dt}.png', dpi=300, bbox_inches='tight', facecolor='white')