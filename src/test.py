import pandas as pd
import numpy as np
from db_read import read_klines,read_mtbars
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.api import OLS,add_constant
from utils import align_idx,reverse_sigmoid
from plotting import pnl_curve,scatter,hist,qqplot
from scipy.stats import gamma,weibull_min,lognorm

symbols = [
    "BTCUSD",   # Bitcoin
    "ETHUSD",   # Ethereum
    "BNBUSD",   # BNB
    "XRPUSD",
    "SOLUSD",   # Solana
    "DOGEUSD",  # Dogecoin
    "ADAUSD",
    "LTCUSD"
]
#symbols = [s+"T" for s in symbols]

r,c = 2,8
fig,axes = plt.subplots(r,c)
for i,symbol in enumerate(symbols):
    df = read_klines(symbol+'T')
    prc = df['close']
    log = np.log(prc)
    S = prc.ewm(span=24).mean() - prc.ewm(span=24*2).mean()
    s = S / S.ewm(span=24*30).std()
    s = s.iloc[24*30*2:]

    D = -(prc - prc.ewm(span=24).mean())
    d = D / S.ewm(span=24*30).std()
    d = d.iloc[24*30*2:]

    r = log.diff(24).shift(-24)*360
    v = log.diff().ewm(span=24*30).std()*95
    weibull_cdf = pd.Series(weibull_min.cdf(v, c=2, scale=v.iloc[:v.shape[0]//5].median()*1.5),v.index)
    rsigmoid_cdf = pd.Series(reverse_sigmoid(v, center=0.75, steepness=10, floor=0.1), v.index)
    s = s.clip(-2,2).mul((1 - weibull_cdf),axis=0)

    x = d.mask(s*d<0,0).clip(-2,2)#(v/v.expanding().mean()/2).clip(0,1)
    x = pd.concat([x*v],axis=1)
    y = (r/v).iloc[::24]
    x,y = align_idx(x,y)
    #model = OLS(y,add_constant(x)).fit()
    f = d.mask(s*d<0,0).clip(-2,2) * weibull_cdf
    f = f*.1 + s*.9
    pos = .5/v * f
    pnl = pos * log.diff().shift(-1)
    pnl = pnl.loc['2025':]
    pnl_curve(pnl,ax=axes[0,i%c])
    
    f = d.mask(s*d<0,0).clip(-2,2) * weibull_cdf
    pos = .5/v * s
    pnl = pos * log.diff().shift(-1)
    pnl = pnl.loc['2025':]
    pnl_curve(pnl,ax=axes[1,i%c])

    #qqplot(v,'gamma',ax=axes[0,i%c])
    #qqplot(v,'weibull',ax=axes[1,i%c])
    #hist(np.log(v),ax=axes[0,i%c])
    #scatter(x,y,ax=axes[1,i%c])

    #print(model.summary())
plt.show()