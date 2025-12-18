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
from strat_models import *
from scipy import stats
from itertools import combinations,product
from tqdm import tqdm
import seaborn as sns
from api_mt5 import get_mt5_bars,init_mt5


symbols = [
    "BTCUSD",   # Bitcoin
    "ETHUSD",   # Ethereum
    "BNBUSD",   # BNB
]

all_prc = pd.DataFrame()
start_date = datetime(2018,1,1)
end_date = datetime(2025,12,1)

rets = pd.DataFrame()
port = pd.DataFrame()
new_bars = []
for i,symbol in tqdm(enumerate(symbols)):
    bars = read_klines(symbol+'T',limit=24*360*5)
    h,l,o,c = bars['high'],bars['low'],bars['open'],bars['close']
    h,l,o,c = np.log(h/c.shift(1)),np.log(l/c.shift(1)),np.log(o/c.shift(1)),np.log(c/c.shift(1))
    strat = VolScaleStrategy(
        vol_window=24*30,
        target_vol=0.3,
        strat_weight=1/len(symbols)
    )
    strat.fit(bars)
    pos = strat.predict(bars)
    port[symbol] = pos
    rets[symbol] = np.log(bars['close']).diff().shift(-1)
    bars = pd.DataFrame({
        'open':o,
        'high':h,
        'low':l,
        'close':c
    })
    new_bars.append(bars.mul(pos.shift(1),axis=0))
ind = (pd.concat(new_bars,axis=1)).groupby(axis=1,level=0).sum()
ind['close'] = ind['close'].cumsum()
ind['open'] = ind['close'] + ind['open']
ind['high'] = ind['close'] + ind['high']
ind['low'] = ind['close'] + ind['low']
ind = np.exp(ind)

ind_pos = pd.DataFrame()


target_vol = 0.35
strat_weight = 0.05
variants = [24,48,96,192]
variant_weight = strat_weight / len(variants)
if strat_weight > 1e-6:
    for fast_ema_window in variants:
        strat = EmaVolStrategy(
            fast_ema_window=fast_ema_window,
            slow_ema_multiplier=2,
            vol_window=24*30,
            weibull_c=2,
            alpha=1.0,
            fit_decay=True,
            target_vol=target_vol,
            strat_weight=variant_weight
        )
        strat.fit(ind)
        ind_pos[f'EmaVol_{fast_ema_window}'] = strat.predict(ind)
        
# ------------------------------------------------------------------ #
# 2. EMA Acceleration × Vol State → Position (% of capital)
# ------------------------------------------------------------------ #
strat_weight = 0.05
variants = [24,48,96,192]
variant_weight = strat_weight / len(variants)
if strat_weight > 1e-6:
    for fast_ema_window in variants:
        strat = AccelVolStrategy(
            fast_ema_window=fast_ema_window,
            slow_ema_multiplier=2,
            diff_multiplier=1.0,
            vol_window=24*30,
            weibull_c=2,
            alpha=1.0,
            fit_decay=True,
            target_vol=target_vol,
            strat_weight=variant_weight
        )
        strat.fit(ind)
        ind_pos[f'AccelVol_{fast_ema_window}'] = strat.predict(ind)
    
# ------------------------------------------------------------------ #
# 3. Breakout (smoothed) × Vol State → Position (% of capital)
# ------------------------------------------------------------------ #
strat_weight = 0.35
variants = [48,96,192,384]
variant_weight = strat_weight / len(variants)
if strat_weight > 1e-6:
    for breakout_window in variants:
        strat = BreakVolStrategy(
            breakout_window=breakout_window,
            smooth_window=12,
            vol_window=24*30,
            weibull_c=2,
            alpha=1.0,
            fit_decay=True,
            target_vol=target_vol,
            strat_weight=variant_weight
        ) 
        strat.fit(ind)
        ind_pos[f'BreakVol_{fast_ema_window}'] = strat.predict(ind)
        
# ------------------------------------------------------------------ #
# 4. Block Momentum (Higher High + Higher Low) × Vol Tilt → Position
# ------------------------------------------------------------------ #
strat_weight = 0.25
variants = [48,96,192,384]
variant_weight = strat_weight / len(variants)
if strat_weight > 1e-6:
    for block_window in variants:
        strat = BlockVolStrategy(
            block_window=block_window,
            smooth_window=12,
            vol_window=24*30,
            weibull_c=2,
            alpha=1.0,
            fit_decay=True,
            target_vol=target_vol,
            strat_weight=variant_weight
        ) 
        strat.fit(ind)
        ind_pos[f'BlockVol_{fast_ema_window}'] = strat.predict(ind)
    
# ------------------------------------------------------------------ #
# 5. Long Wed / Short Thu → Position
# ------------------------------------------------------------------ #
strat_weight = 0.10
variants = [24*60,24*180]
variant_weight = strat_weight / len(variants)
if strat_weight > 1e-6:
    for vol_window in variants:
        strat = WedThuStrategy(
            vol_window=vol_window,
            target_vol=target_vol,
            strat_weight=variant_weight
        )
        strat.fit(ind)
        ind_pos[f'WedThu_{fast_ema_window}'] = strat.predict(ind)
        
# ------------------------------------------------------------------ #
# 6. Short-term Reversal → Position
# ------------------------------------------------------------------ #
strat_weight = 0.10
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
        strat.fit(ind)
        ind_pos[f'Rev_{fast_ema_window}'] = strat.predict(ind)
        
# ------------------------------------------------------------------ #
# 7. Momentum-Orthogonal Alpha → Position
# ------------------------------------------------------------------ #
strat_weight = 0.20
variants = [96,192,384]
variant_weight = strat_weight / len(variants)
if strat_weight > 1e-6:
    for forward_window in variants:
        strat = OrthAlphaStrategy(
            forward_window=forward_window,
            vol_window=24*30,
            regression_window=24*30,
            smooth_window=12,
            fit_decay=True,
            target_vol=target_vol,
            strat_weight=variant_weight
        )
        strat.fit(ind)
        ind_pos[f'OrthAlpha_{fast_ema_window}'] = strat.predict(ind)

pos = ind_pos.sum(axis=1)
poss = port.mul(pos,axis=0)
pnl = poss * rets.shift(-1) - poss.diff().abs() * .0010
ax = pnl_curve(pnl.sum(axis=1))
pnl.cumsum().plot(ax=ax)
plt.show()
exit()




r,c = 2,3
fig,axes = plt.subplots(r,c)
for i,symbol in tqdm(enumerate(symbols)):
    bars = read_klines(symbol+'T')
    prc = bars['close']

    fri = (prc.index.weekday == 4) & (prc.index.hour >= 23)
    sat = (prc.index.weekday == 5)
    sun = (prc.index.weekday == 6)
    mon = (prc.index.weekday == 0) & (prc.index.hour <= 11)
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