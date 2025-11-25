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
from strat_models import EmaVolStrategy
from scipy import stats
from itertools import combinations,product
from tqdm import tqdm
import seaborn as sns


symbols = [
    "BTCUSD",   # Bitcoin
    "ETHUSD",   # Ethereum
    "BNBUSD",   # BNB
    "SOLUSD",   # Solana
    "DOGEUSD",  # Dogecoin
    "ADAUSD",
    "XRPUSD",
    "LTCUSD"
]

all_prc = pd.DataFrame()
start_date = datetime(2020,1,1)
end_date = datetime(2023,1,1)
for symbol in symbols:
    all_prc[symbol] = read_klines(symbol+'T',start_time=start_date,end_time=end_date)['close']

r,c = 2,3
#fig,axes = plt.subplots(r,c)
pair_r = pd.DataFrame(index=symbols,columns=symbols)
v_r = []
r_r = []
pair = []
for i,(s1,s2) in tqdm(enumerate(product(['BTCUSD'],['BNBUSD','SOLUSD','DOGEUSD','ADAUSD','XRPUSD','LTCUSD']))):
    #if i >= 5: break
    p1,p2 = all_prc[s1],all_prc[s2]
    l1,l2 = np.log(p1),np.log(p2)
    r1,r2 = l1.diff(),l2.diff()
    v1,v2 = r1.ewm(span=24*30).std(),r2.ewm(span=24*30).std()

    x,y = v1,v2
    x,y = align_idx(x,y)
    #mod = OLS(y,add_constant(x)).fit()
    #pair_r.loc[s1,s2] = mod.rsquared
    #if mod.rsquared >= .65:
    #v_r.append(mod.rsquared)

    #x,y = r1,r2
    #x,y = align_idx(x,y)
    #mod = OLS(y,add_constant(x)).fit()
    #pair_r.loc[s1,s2] = mod.rsquared
    #r_r.append(mod.rsquared)

    #pair.append((s1,s2))

    #print(s1,s2,f'{mod.rsquared:.2%}')

    mod = RollingOLS(y,add_constant(x),window=24*30,min_nobs=24*30).fit()
    hedge = mod.params.ewm(span=24).mean()
    w1 = hedge.iloc[:,1]
    w2 = pd.Series(-1,hedge.index)
    c1 = r1.mul(w1,axis=0)
    c2 = r2.mul(w2,axis=0)
    a = hedge.iloc[:,0]
    resid = -pd.concat([c1,c2,a],axis=1).sum(axis=1)
    mse = mod.mse_resid
    z = (resid / mse)
    #z = z#.ewm(span=12).mean()
    #z = z / z.ewm(span=24*30).std()
    #z.plot()
    #plt.show()
    #exit()

    #print(resid1.corr(resid2))

    tf = pd.Series(np.where(z.abs() >= z.abs().expanding(24*30).quantile(.95), np.sign(z), 0), z.index)
    p1 = tf.mul(w1)
    p2 = tf.mul(w2)
    p1 = p1.mul(r1.shift(-1),axis=0) - p1.diff().abs() * .0003
    p2 = p2.mul(r2.shift(-1),axis=0) - p2.diff().abs() * .0003
    o = p1+p2
    #ax = scatter(tf,o)
    o.cumsum().plot()
    #p1.cumsum().plot()
    #p2.cumsum().plot()
    #ax.set_title(str(s1)+str(s2))
plt.show()

#ax = scatter(pd.Series(v_r),pd.Series(r_r),annot=pd.Series(pair))
#ax.set_title('Vol R2 (x) vs Ret R2 (y)')
#sns.heatmap(pair_r.astype(float),annot=True)
#plt.show()
