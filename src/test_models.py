import pandas as pd
import numpy as np
from db_read import read_klines,read_mtbars,read_symbols
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.api import OLS,add_constant
from utils import align_idx
from plotting import pnl_curve,scatter,hist,qqplot
from scipy.stats import gamma,weibull_min,lognorm
from strat_models import EmaVolStrategy
from sklearn.model_selection import train_test_split,GridSearchCV,TimeSeriesSplit
import seaborn as sns

symbols = [
    "BTCUSD",   # Bitcoin
    "ETHUSD",   # Ethereum
    "BNBUSD",   # BNB
    #"XRPUSD",
    "SOLUSD",   # Solana
    "DOGEUSD",  # Dogecoin
    "ADAUSD",
    #"LTCUSD"
]
#symbols = [s+"T" for s in symbols]
#symbols = read_symbols('Crypto')['symbol']

#r,c = 3,3
#fig,axes = plt.subplots(r,c)
port = pd.DataFrame()
for i,symbol in enumerate(symbols):
    pnls = pd.DataFrame()
    df = read_klines(symbol+'T')
    #prc = df['close']
    #df = read_mtbars(symbol)
    prc = df['close']
    ret = np.log(df['close']).diff().shift(-1)
    
    x,y = align_idx(prc,ret)
    x1, x2, y1, y2 = train_test_split(
        x, y, 
        test_size=0.5,
        shuffle=False  # This ensures sequential split
    )
    try:
        for fast in [12,24,48,96]:
            tscv = TimeSeriesSplit(n_splits=5)
            strat = EmaVolStrategy(
                fast_ema_window=fast,
                slow_ema_multiplier=2,
                vol_window=24*30,
                weibull_c=2,
                target_vol=0.5,
                strat_weight=1
            )
            strat.fit(x1)
            pos = strat.predict(x2)
            pnl = np.around(pos*10)/10 * y2
            pnls[fast] = pnl
        #sns.heatmap(pnls.corr(),annot=True)
        #plt.show()
        pnls = pnls.loc['2025-10':]
        #ax = axes[i//c,i%c]
        #ax.set_title(symbol)
        #pnl_curve(,ax=ax)
        #pnls.cumsum().plot(ax=ax,secondary_y=True)
        pnl = pnls.mean(axis=1)
        port[symbol] = pnl
    except: pass
ax = pnl_curve(port.mean(axis=1))
port.cumsum().plot(linestyle='--',ax=ax)
plt.show()
sns.heatmap(port.corr(),annot=True)
plt.show()