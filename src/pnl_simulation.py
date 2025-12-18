import pandas as pd
import numpy as np
from api_bnb import get_klines
from db_read import read_klines,read_mtbars,read_symbols
from db_load import save_klines
from config import TZ
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
from statsmodels.api import OLS,add_constant
from utils import align_idx
from plotting import pnl_curve,scatter,hist,qqplot,realtime_pnl
from scipy.stats import gamma,weibull_min,lognorm
from strat_models import EmaVolStrategy,VolScaleStrategy,BreakVolStrategy
from sklearn.model_selection import train_test_split,GridSearchCV,TimeSeriesSplit
import seaborn as sns
from tqdm import tqdm
from strat_io import load_model
import matplotlib.ticker as mtick
from config import TZ, MODEL_DIR, logger,PORTFOLIO, MAX_HISTORY_HOURS
import os
from scipy.stats import norm

symbols = [
    "BTCUSD",   # Bitcoin
    "ETHUSD",   # Ethereum
    "BNBUSD",   # BNB
    "DOGEUSD",  # Dogecoin
    "ADAUSD",
]


r,c = 2,3
port = pd.DataFrame()
port_lots = pd.DataFrame()
port_tcs = pd.DataFrame()
port_fcs = pd.DataFrame()
trading_start = datetime(2025,11,17,1,tzinfo=TZ)
#trading_start = datetime(2020,6,17,1,tzinfo=TZ)
trading_end = datetime.now(TZ)
days = (trading_end - trading_start)/timedelta(1)

capital = 10000
n_assets = len(PORTFOLIO)
capital_per_asset = capital / n_assets
logger.info(f"Capital per asset: {capital_per_asset:,.2f} ({n_assets} assets)")
symb_df = read_symbols(asset_class='Crypto').set_index('symbol')
symb_df['notional'] = symb_df['trade_tick_value']/symb_df['trade_tick_size']
notional = symb_df['notional']
spread = symb_df['spread']

#fig,axes = plt.subplots(2,3)
for i,symbol in enumerate(PORTFOLIO):
    poss = pd.DataFrame()
    df = read_mtbars(symbol,limit=MAX_HISTORY_HOURS*6)
    df['volume'] = df['tick_volume']
    ret = df['close'].diff().shift(-1)
    contract_value = notional[symbol] * df['close']
    slippage = 0.0005
    round_trip_cost = (spread[symbol] + slippage) * notional[symbol]
    commission = 0.000650 * contract_value
    ret = contract_value.diff().shift(-1)
    all_in_lots = capital_per_asset / contract_value
    tc = round_trip_cost / 2 + commission
    #tc = .0010
    swap = .30 / (24*365)  # 0.30% annualized
    fc = swap * contract_value
    
    if True:
        x,y = align_idx(df,ret)
        model_dir = f'{MODEL_DIR}/{symbol}'
        for name in os.listdir(model_dir):
            #if '24' in name: continue
            strat = load_model(f'{model_dir}/{name}')['model']
            pos = strat.predict(x)
            name,variant = name.split('_')[:2]
            poss[(name,variant)] = pos
            
        strat = VolScaleStrategy(
            vol_window=24*30,
            target_vol=0.30,
            strat_weight=1  
        )
        strat.fit(x)
        pos = strat.predict(x)

        pos = poss.sum(axis=1)
        lots = np.around(pos * all_in_lots * 100)/100
        pnl = lots * y
        trading_cost = lots.diff().abs() * tc
        funding_cost = lots.abs() * fc
        pnl_net = pnl - trading_cost - funding_cost
        port_lots[symbol] = lots.loc[trading_start:]
        port_tcs[symbol] = trading_cost.loc[trading_start:]
        port_fcs[symbol] = funding_cost.loc[trading_start:]
        port[symbol] = pnl_net.loc[trading_start:]

    print(f'{(tc/contract_value).max()*10000:.2f}bps')
    #ax1 = axes[0, i%3]
    #ax2 = axes[1, i%3]
    #pnl_curve(pnl,ax=ax1)
    #lots.plot(ax=ax2)
#plt.show()

print(port_lots.diff().abs().sum())
print(port_tcs.sum())
print(port_fcs.sum())
#ax = realtime_pnl(port.sum(axis=1)/capital)
ax = pnl_curve(port.sum(axis=1)/capital)
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=2))
port.cumsum().plot(linestyle='--',ax=ax,alpha=0.5,secondary_y=True)
ax.set_title(f'Simulation PnL Since Inception- Day {days:.2f} (35% targeted vol)')
#plt.show()
plt.savefig(f'plot/simulation_pnl/day_{days:04.0f}.png', dpi=300, bbox_inches='tight', facecolor='white')

print(port_lots.tail(20))