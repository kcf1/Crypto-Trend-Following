import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from scipy import stats

def pnl_curve(y,freq=1,ax=None):
    cur = y.cumsum()
    scaler = 24/freq

    m,s = y.mean()*(360*scaler),y.std()*(360*scaler)**0.5
    skew = y.skew()
    sr = m/s
    hit = (y >= 0).mean()
    dd05 = (cur - cur.cummax()).quantile(0.05)
    
    if ax is None: fig,ax = plt.subplots()
    cur.plot(ax=ax)
    ax.annotate(f'sr: {sr:.2f}, hit: {hit*100:.2f}%',xycoords='axes fraction',xy=(0.05,0.9))
    ax.annotate(f'mean: {m*100:.2f}%, vol: {s*100:.2f}%',xycoords='axes fraction',xy=(0.05,0.8))
    ax.annotate(f'dd05: {dd05*100:.2f}%, skew: {skew:.2f}',xycoords='axes fraction',xy=(0.05,0.7))
    return ax

def hist(y,ax=None):
    median = y.median()
    mean = y.mean()
    q05,q95 = y.quantile(0.05),y.quantile(0.95)

    if ax is None: fig,ax = plt.subplots()
    ax.hist(y,bins=min(50,len(y)//5))
    ax.axvline(median, color='black', linestyle='--')
    ax.axvspan(xmin=q05,xmax=q95,color='grey',alpha=0.2)
    ax.annotate(f'median: {median:.2f}, mean: {mean:.2f}',xycoords='axes fraction',xy=(0.05,0.9))
    ax.annotate(f'q05: {q05:.2f}, q95: {q95:.2f}',xycoords='axes fraction',xy=(0.05,0.8))
    return ax

def qqplot(data, 
           dist='norm', 
           fit=True,
           ax=None):
    """
    Q-Q plot with:
      - 45° reference line
      - 5th and 95th percentiles highlighted
      - Median and mean annotated
      - Optional MLE fit (for parametric dists)
    
    Parameters
    ----------
    data : array-like
        Empirical data (e.g., volatility series)
    dist : str or scipy.stats distribution
        'norm', 'lognorm', 'weibull', 'gamma', or any scipy.stats.rv_continuous
    fit : bool
        If True, fit distribution parameters to data
    ax : matplotlib Axes
        Optional axis to plot on
    title : str
        Custom title
    label_data, label_theory : str
        Legend labels
    """
    data = np.asarray(data)
    data = data[~np.isnan(data)]  # drop NaN

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    # --- Get theoretical quantiles ---
    if isinstance(dist, str):
        dist = dist.lower()
        if dist == 'norm':
            dist_obj = stats.norm
        elif dist == 'lognorm':
            dist_obj = stats.lognorm
        elif dist == 'weibull':
            dist_obj = stats.weibull_min
        elif dist == 'gamma':
            dist_obj = stats.gamma
        else:
            raise ValueError("dist must be 'norm', 'lognorm', 'weibull', 'gamma', or scipy dist")
    else:
        dist_obj = dist

    # Fit if requested
    if fit:
        params = dist_obj.fit(data)
    else:
        params = ()

    # Sample quantiles
    quantiles = np.linspace(0.01, 0.99, len(data))
    theoretical = dist_obj.ppf(quantiles, *params)
    empirical = np.quantile(data, quantiles)

    # --- Stats ---
    median_emp = np.median(data)
    mean_emp = np.mean(data)
    q05_emp = np.quantile(data, 0.05)
    q95_emp = np.quantile(data, 0.95)

    # Find theoretical counterparts
    #q05_theory = dist_obj.ppf(0.05, *params)
    #q95_theory = dist_obj.ppf(0.95, *params)

    # --- Plot ---
    ax.scatter(theoretical, empirical, alpha=0.7, s=20)
    #ax.plot([0, 1], [0, 1], 'r--', label='y = x', linewidth=1.5)  # 45° line

    # Highlight 5th–95th band
    #ax.axvspan(q05_theory, q95_theory, color='grey', alpha=0.1, label='5th–95th')
    #ax.axhspan(q05_emp, q95_emp, color='grey', alpha=0.1)

    # Annotate
    ax.annotate(f'median: {median_emp:.3f}', xycoords='axes fraction', xy=(0.05, 0.93), fontsize=9)
    ax.annotate(f'mean:   {mean_emp:.3f}', xycoords='axes fraction', xy=(0.05, 0.86), fontsize=9)
    ax.annotate(f'q05: {q05_emp:.3f}, q95: {q95_emp:.3f}', xycoords='axes fraction', xy=(0.05, 0.79), fontsize=9)
    return ax

def scatter(x,y,ax=None):
    idx = x.dropna().index.intersection(y.dropna().index)
    x,y = x.reindex(idx),y.reindex(idx)

    coef = np.polyfit(x,y,1)
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = np.polyval(coef, x_fit)

    #tss = ((y-y.mean())**2).mean()
    #rss = ((y-y_fit)**2).mean()

    #r2 = (1 - rss / tss)*100
    beta = coef[0]
    pearson = x.corr(y, method='pearson')*100
    spearman = x.corr(y, method='spearman')*100


    if ax is None: fig,ax = plt.subplots()
    ax.scatter(x,y)
    ax.plot(x_fit,y_fit,color='black',linestyle='--')
    ax.annotate(f'beta: {beta:.2f}',xycoords='axes fraction',xy=(0.05,0.9))
    ax.annotate(f'pearson: {pearson:.2f}%',xycoords='axes fraction',xy=(0.05,0.8))
    ax.annotate(f'spearman: {spearman:.2f}%',xycoords='axes fraction',xy=(0.05,0.7))
    return ax

def bin_scatter(x,y,ax=None):
    idx = x.dropna().index.intersection(y.dropna().index)
    x,y = x.reindex(idx),y.reindex(idx)

    x_bin = pd.qcut(x,10,labels=range(10))
    xcs, ycs, cs = [], [], []
    for xg,yg in zip(x.groupby(x_bin),y.groupby(x_bin)):
        xb,yb = xg[1],yg[1]
        xc,yc = xb.mean(),yb.mean()
        corr = xb.corr(yb, method='spearman')

        xcs.append(xc)
        ycs.append(yc)
        cs.append(corr)
    x = pd.Series(xcs,index=range(10))
    y = pd.Series(ycs,index=range(10))
    c = pd.Series(cs,index=range(10)) / 0.01

    coef = np.polyfit(x,y,1)
    x_fit = np.linspace(x.min(), x.max(), 100)
    y_fit = np.polyval(coef, x_fit)

    beta = coef[0]
    pearson = x.corr(y, method='pearson')*100
    spearman = x.corr(y, method='spearman')*100

    if ax is None: fig,ax = plt.subplots()
    ax.scatter(xcs,ycs,s=c.abs()*20,c=np.where(c>0,'C0','C3'))
    ax.plot(x_fit,y_fit,color='black',linestyle='--')
    ax.annotate(f'beta: {beta:.2f}',xycoords='axes fraction',xy=(0.05,0.9))
    ax.annotate(f'pearson: {pearson:.2f}%',xycoords='axes fraction',xy=(0.05,0.8))
    ax.annotate(f'spearman: {spearman:.2f}%',xycoords='axes fraction',xy=(0.05,0.7))
    return ax

def feat_stationarity(x):
    r,c = x.shape[1]//5+1, 5
    fig,axes = plt.subplots(r,c)
    for i,name in enumerate(x.columns):
        s = x[name]
        mean = s.mean()
        std = s.std()
        skew = s.skew()
        res = adfuller(s)
        adf,p = res[0],res[1]

        ax = axes[i//c,i%c]
        s.plot(ax=ax,title=name)
        ax.annotate(f'adf: {adf:.2f}, p: {p*100:.2f}%',xycoords='axes fraction',xy=(0.05,0.9))
        ax.annotate(f'mean: {mean:.2f}',xycoords='axes fraction',xy=(0.05,0.8))
        ax.annotate(f'std: {std:.2f}',xycoords='axes fraction',xy=(0.05,0.7))
        ax.annotate(f'skew: {skew:.2f}',xycoords='axes fraction',xy=(0.05,0.6))

    plt.show()