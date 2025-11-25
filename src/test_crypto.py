import pandas as pd
import numpy as np
from api_bnb import get_klines
from db_read import read_klines,read_mtbars,read_symbols
from api_mt5 import get_mt5_bars,get_mt5_symbols,init_mt5
from db_load import save_klines, save_mtbars
from config import TZ,PORTFOLIO
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

init_mt5()

df = get_mt5_symbols('Crypto')
#df = df.loc[df['currency_profit']=='USD']
df['bps'] = 2*df['spread']/df['price']*10000 #+ 5/(df['price']*df['trade_contract_size'])
print(df.sort_values('bps'))
bps = df.set_index('symbol')['bps'].to_dict()
#exit()

symbols = df['symbol'].to_list()
symbols = PORTFOLIO + ['AAVUSD','LNKUSD','UNIUSD','FETUSD','XLMUSD','BARUSD','ICPUSD','MANUSD','GRTUSD','DOGEUSD']
symbols = ['AAVUSD','LNKUSD']
symbols = ['GRTUSD','DOGEUSD']
start_date,end_date = datetime(2020,1,1), datetime(2025,12,1)

r,c = 2,len(symbols)
fig,axes = plt.subplots(r,c)
port = pd.DataFrame()
stats = pd.DataFrame()
all_prc = pd.DataFrame({symbol:read_klines(symbol+'T')['close'] for symbol in symbols})

for i,symbol in enumerate(symbols[:]):
    cost = bps[symbol]/10000
    pnls = pd.DataFrame()
    poss = pd.DataFrame()
    #df = read_klines(symbol+'T',limit=24*360*4)
    #df = pd.concat([df,all_prc],axis=1).dropna()
    df = get_mt5_bars(symbol,date_from=start_date,date_to=end_date)
    df.index = pd.to_datetime(df['time'],unit='ms')
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
        target_vol = 0.3
        strat_weight = 0.10
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
                    target_vol=.3,
                    strat_weight=variant_weight
                )
                strat.fit(x)
                pos = strat.predict(x)
                poss[f'EmaVol_{fast_ema_window}'] = pos
                pnl = pos * y - pos.diff().abs() * cost
                pnls[f'EmaVol_{fast_ema_window}'] = pnl
                
        # ------------------------------------------------------------------ #
        # 2. EMA Acceleration × Vol State → Position (% of capital)
        # ------------------------------------------------------------------ #
        strat_weight = 0.10
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
                    target_vol=.3,
                    strat_weight=variant_weight
                )
                strat.fit(x)
                pos = strat.predict(x)
                poss[f'AccelVol_{fast_ema_window}'] = pos
                pnl = pos * y - pos.diff().abs() * cost
                pnls[f'AccelVol_{fast_ema_window}'] = pnl
            
        # ------------------------------------------------------------------ #
        # 3. Breakout (smoothed) × Vol State → Position (% of capital)
        # ------------------------------------------------------------------ #
        strat_weight = 0.40
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
                    target_vol=target_vol,
                    strat_weight=variant_weight
                ) 
                strat.fit(x)
                pos = strat.predict(x)
                poss[f'BreakoutVol_{breakout_window}'] = pos
                pnl = pos * y - pos.diff().abs() * cost
                pnls[f'BreakoutVol_{breakout_window}'] = pnl
                
        # ------------------------------------------------------------------ #
        # 4. Block Momentum (Higher High + Higher Low) × Vol Tilt → Position
        # ------------------------------------------------------------------ #
        strat_weight = 0.30
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
                    target_vol=target_vol,
                    strat_weight=variant_weight
                ) 
                strat.fit(x)
                pos = strat.predict(x)
                poss[f'BlockVol_{block_window}'] = pos
                pnl = pos * y - pos.diff().abs() * cost
                pnls[f'BlockVol_{block_window}'] = pnl
            
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
                strat.fit(x)
                pos = strat.predict(x)
                poss[f'WedThu_{vol_window}'] = pos
                pnl = pos * y - pos.diff().abs() * cost
                pnls[f'WedThu_{vol_window}'] = pnl
                
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
                strat.fit(x)
                pos = strat.predict(x)
                poss[f'Rev_{reversal_window}'] = pos
                pnl = pos * y - pos.diff().abs() * cost
                pnls[f'Rev_{reversal_window}'] = pnl

        strat = VolScaleStrategy(
            vol_window=24*30,
            target_vol=0.5,
            strat_weight=1
        )
        strat.fit(x)
        pos = strat.predict(x)
        ax1 = axes[0,i%c]
        pnl = pos * y - pos.diff().abs() * cost
        #pnl.cumsum().plot(linestyle='--',color='grey',alpha=0.5,ax=ax1)
        pos = poss.sum(axis=1)
        pnl = pos * y - pos.diff().abs() * cost
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