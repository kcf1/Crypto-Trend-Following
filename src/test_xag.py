import pandas as pd
import numpy as np
from db_read import read_klines,read_mtbars
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
from statsmodels.api import OLS,add_constant
from utils import align_idx,reverse_sigmoid
from performance import regress_incremental
from plotting import pnl_curve,scatter,hist,qqplot,feat_stationarity,corr_dendrogram
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
from api_mt5 import get_mt5_bars,init_mt5,get_mt5_symbols

init_mt5()
df = get_mt5_symbols('Agriculture')
df = df.set_index('symbol')
spread = df['spread']/df['price']
slippage = .0003
commission = .000014
cost = spread/2 + slippage + commission

print(cost)
#exit()



start_date = datetime(2020,1,1)
end_date = datetime(2025,12,1)
bars = get_mt5_bars('COFFEE.c',date_from=start_date,date_to=end_date)
bars['volume'] = bars['tick_volume']
bars.index = pd.to_datetime(bars['time'],unit='ms')
poss = dict()

target_vol = 0.35
strat_weight = 0.20
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
        strat.fit(bars)
        poss[('EmaVolStrategy',fast_ema_window)] = strat.predict(bars)
        
# ------------------------------------------------------------------ #
# 2. EMA Acceleration × Vol State → Position (% of capital)
# ------------------------------------------------------------------ #
strat_weight = 0.15
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
        strat.fit(bars)
        poss[('AccelVolStrategy',fast_ema_window)] = strat.predict(bars)
    
# ------------------------------------------------------------------ #
# 3. Breakout (smoothed) × Vol State → Position (% of capital)
# ------------------------------------------------------------------ #
strat_weight = 0.20
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
        strat.fit(bars)
        poss[('BreakVolStrategy',breakout_window)] = strat.predict(bars)
        
# ------------------------------------------------------------------ #
# 4. Block Momentum (Higher High + Higher Low) × Vol Tilt → Position
# ------------------------------------------------------------------ #
strat_weight = 0.15
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
        strat.fit(bars)
        poss[('BlockVolStrategy',block_window)] = strat.predict(bars)
        
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
        strat.fit(bars)
        poss[('RevStrategy',reversal_window)] = strat.predict(bars)
        
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
        strat.fit(bars)
        poss[('OrthAlphaStrategy',forward_window)] = strat.predict(bars)

fig,axes = plt.subplots(1,3)

poss = pd.DataFrame(poss)
print(poss)
ret = np.log(bars['close']).diff().shift(-1)

pos = poss.sum(axis=1)
pnl = pos * ret - pos.diff().abs() * .0005
print(pnl)
ax = pnl_curve(pnl,ax=axes[0])

pnls = poss.groupby(axis=1,level=0).sum().mul(ret,axis=0)
pnls = pnls / pnls.std() * target_vol / 95
pnls.cumsum().plot(ax=axes[1])

corr_dendrogram(poss.corr(),ax=axes[2])

plt.show()

pnl.to_csv('data/tf_xau.csv')
exit()
