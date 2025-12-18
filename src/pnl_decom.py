import pandas as pd
import numpy as np
from api_bnb import get_klines
from db_read import read_klines,read_mtbars,read_symbols
from db_load import save_klines
from config import TZ
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.api import OLS,add_constant
from utils import align_idx,triple_barrier
from plotting import pnl_curve,scatter,hist,qqplot,corr_dendrogram,monthly_sharpe
from scipy.stats import gamma,weibull_min,lognorm
from strat_models import *
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split,GridSearchCV,TimeSeriesSplit
import seaborn as sns
from tqdm import tqdm
from strat_io import load_model
from config import TZ, MODEL_DIR, logger,PORTFOLIO
import os
from scipy.stats import norm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from performance import calcaulte_pnl_stats
from scipy.stats import ttest_1samp
from ta import add_all_ta_features
from itertools import combinations
from strat_cv import BlockBootstrap
from multiprocessing import Pool

symbols = PORTFOLIO
info = read_symbols()
info = info.set_index('symbol')
info['commission'] = np.where(info['class']=='Crypto',.00065,0)
spread = info['spread']/info['price']
slippage = .0001
commission = info['commission']
cost = spread / 2 + slippage + commission

port = pd.DataFrame()
bnch = pd.DataFrame()
for i,symbol in enumerate(symbols):
    poss = dict()
    sigs = dict()
    df = read_mtbars(symbol,limit=24*360*5)
    df['volume'] = df['tick_volume']
    ret = np.log(df['close']).diff().shift(-1)
    #df = read_klines(symbol+'T',limit=24*360*5)
    #ta = add_all_ta_features(df.copy(),'open','high','low','close','volume',vectorized=True)
    #ta = ta.loc[:,ta.isnull().mean()<=.3]
    
    if True:
    #try:
        x,y = align_idx(df,ret)
        
        #funding_hour = pd.Series(np.where((x.index.hour == 23)|(x.index.hour == 7)|(x.index.hour == 15), 1, 0), x.index)
        #x = x.loc[funding_hour==1].reindex(x.index).ffill()
        model_dir = f'{MODEL_DIR}/{symbol}'
        for name in os.listdir(model_dir):
            strat = load_model(f'{model_dir}/{name}')['model']
            pos = strat.predict(x)
            #sig = strat.signal(x)
            name,variant = name.split('_')[:2]
            poss[name+variant] = pos
            #sigs[name+variant] = sig

        tc = cost[symbol]
        poss = pd.DataFrame(poss).dropna()
        #igs = pd.DataFrame(sigs).dropna()
        pos = poss.sum(axis=1)
        #sig = sigs.sum(axis=1)
        pnl = pos * y - pos.diff().abs() * .0010
        port[symbol] = pnl
        
        strat = VolScaleStrategy(
            vol_window=24*30,
            target_vol=0.20,
            strat_weight=1
        )
        strat.fit(x)
        pos = strat.predict(x)
        benchmark = pos * y - pos.diff().abs() * .0010
        bnch[symbol] = benchmark

bnch_pnl = bnch.mean(axis=1)
port_pnl = port.mean(axis=1)
bcv = BlockBootstrap(
    block_size=24*90, n_bootstraps=1000
)
stats = []
for trn,tes in bcv.split(port_pnl):
    #trn = trn[:24*90]
    pnl = port_pnl.iloc[trn]
    benchmark = bnch_pnl.iloc[trn]
    stat = calcaulte_pnl_stats(pnl,benchmark=benchmark,print_results=False)
    stat = stat.dropna()
    #stat['max_loss'] = pnl.cumsum().min()
    #stat['time_passing'] = (pnl.cumsum() >= .15).argmax()
    stats.append(stat)
stats = pd.concat(stats,axis=1).T
print(stats.describe())

r,c = 3,5
fig,axes = plt.subplots(r,c)
for i,col in enumerate(stats.columns):
    #if i >= 12: break
    ax = axes[i//c,i%c]
    ax.set_title(col)
    hist(stats[col].dropna(),ax=ax)
plt.show()