import pandas as pd
import numpy as np
from db_read import read_klines,read_mtbars
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.api import OLS,add_constant
from utils import align_idx,reverse_sigmoid
from plotting import pnl_curve,scatter,hist,qqplot
from scipy.stats import gamma,weibull_min,lognorm,norm
from fast_func import rolling_slope_vec,rolling_pca1,rolling_pca2
from statsmodels.regression.rolling import RollingOLS
from config import PORTFOLIO
from performance import regress_incremental

symbols = [
    "BTCUSD",   # Bitcoin
    "ETHUSD",   # Ethereum
    #"BNBUSD",   # BNB
    #"SOLUSD",   # Solana
    #"DOGEUSD",  # Dogecoin
    #"ADAUSD",
    #"XRPUSD",
    #"LTCUSD"
]
#symbols = PORTFOLIO

prcs = pd.DataFrame()
vlms = pd.DataFrame()
for i,symbol in enumerate(symbols[:]):
    #df = read_mtbars(symbol)
    #prc = df['close']
    #vlm = df['tick_volume']
    df = read_klines(symbol+'T')
    prc = df['close']
    vlm = df['volume']
    prcs[symbol] = prc
    vlms[symbol] = vlm
prcs = prcs.ffill().dropna()

pc1 = pd.DataFrame(rolling_pca2(np.log(prcs).values,window=24),prcs.index)
#pc1 = np.log(prcs).diff()
#pc1 = pc1 / pc1.ewm(span=24*30).std()
#pc1 = pc1.mean(axis=1).ewm(span=24).mean()

r,c = 2,6
fig,axes = plt.subplots(r,c)
port = pd.DataFrame()
for i,symbol in enumerate(symbols[:]):
    bars = read_klines(symbol+'T')
    prc = prcs[symbol]
    vlm = vlms[symbol]
    vlm = np.log((vlm.ewm(span=24).mean()) / vlm.ewm(span=24*30).std())
    #hist(vlm)
    #plt.show()

    log = np.log(prc)
    #(log-log.iat[0]).plot()
    #continue
    x = add_constant(pc1)
    y = log.diff()#.ewm(span=24).mean() #/ log.diff().ewm(span=24*30).std()
    #x,y = align_idx(x,y)
    reg = RollingOLS(y,x,window=24).fit()
    p = reg.params
    e = y - (p*x.drop(columns='const')).sum(axis=1)
    a = p['const']
    #r2 = reg.rsquared

    S = a.ewm(span=24).mean()
    s = S / S.ewm(span=24*30).std()
    #pos = s * (1-v_cdf) * .5/v

    s = s.iloc[24*30*2:].clip(-2,2)

    r = log.diff(24).shift(-24)*360
    v = log.diff().ewm(span=24*30).std()*95
    
    s = s.clip(-2,2)
    regress_incremental(s,bars)

    pos = s * .5/v
    pnl = pos * log.diff().shift(-1) - pos.diff().abs()*.0005/2
    port[symbol] = pnl
    #pnl_curve(pnl,ax=axes[0,i%c])

    #pos = s * (1-v_cdf) * .5/v
    #pnl = pos * log.diff().shift(-1) - pos.diff().abs()*.0005/2
    #pnl_curve(pnl,ax=axes[1,i%c])

    #print(model.summary())
#(pc1-pc1.iat[0]).plot(secondary_y=True)
ax = pnl_curve(port.mean(axis=1))
port.cumsum().plot(ax=ax,secondary_y=True,alpha=0.5,linestyle='--')
plt.show()