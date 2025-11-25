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
from strat_models import EmaVolStrategy,VolScaleStrategy,BreakVolStrategy,SlopeVolStrategy,BlockVolStrategy,AccelVolStrategy,WedThuStrategy,RevStrategy
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

symbols = PORTFOLIO

r,c = 2,len(symbols)
fig,axes = plt.subplots(r,c)
for i,symbol in enumerate(symbols):
    poss = dict()
    sigs = dict()
    df = read_mtbars(symbol,limit=24*360*6)
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
            sig = strat.signal(x)
            name,variant = name.split('_')[:2]
            poss[name+variant] = pos
            sigs[name+variant] = sig

        poss = pd.DataFrame(poss).dropna()
        sigs = pd.DataFrame(sigs).dropna()
        pos = poss.sum(axis=1)
        sig = sigs.sum(axis=1)
        port_pnl = pos * y #- pos.diff().abs() * .0010
        port_sig = sig * y #- sig.diff().abs() * .0010

        sigs['sr30'] = (port_pnl.ewm(span=24*30).mean() / port_pnl.ewm(span=24*30).std()).shift(24*30)

        #sigs['vol'] = y.ewm(span=24*30).std()

        y = port_pnl.reindex(sigs.dropna().index).dropna()
        x = sigs.reindex(y.index)
        y1,y2 = train_test_split(y,test_size=0.5,random_state=42,shuffle=False)
        x1,x2 = x.reindex(y1.index),x.reindex(y2.index)
        y1 = triple_barrier(
            returns=y1,
            vol=0.01,
            tp_multiplier=5,
            sl_multiplier=1,
            max_holding=48,
        )['label']
        x1 = x1.reindex(y1.index)

        #coefs = pd.DataFrame(mod_sign.coef_,index=mod_sign.classes_,columns=mod_sign.feature_names_in_)
        #coefs.T.plot(kind='bar',alpha=0.5)

        #mod_size = LogisticRegressionCV(penalty='l2',cv=5,n_jobs=-1,random_state=42,fit_intercept=True,class_weight='balanced')
        mod_size = RandomForestClassifier(n_estimators=100,random_state=42,n_jobs=-1,class_weight='balanced')
        mod_size.fit(x1,y1)

        prob = pd.DataFrame(mod_size.predict_proba(x),index=x.index,columns=mod_size.classes_)

        f = prob[1]
        f = np.sign(f - f.expanding(24*90).median()).clip(0).ewm(span=12).mean()
        yh = (f * pos).loc[x2.index[0]:]
        y = pos.loc[x2.index[0]:]
        yh = yh * ret - yh.diff().abs()*.0010
        y = y * ret - y.diff().abs()*.0010

        ax1,ax2 = axes[0,i%c],axes[1,i%c]
        pnl_curve(y,ax=ax1)
        pnl_curve(yh,ax=ax2)

        print(y.corr(yh))
        #ax.axvline(x=x2.index[0],linestyle='--',color='black')

        #coefs = pd.DataFrame(mod.coef_,index=mod.classes_,columns=mod.feature_names_in_)
        #coefs.T.plot(kind='bar',alpha=0.5)

plt.show()