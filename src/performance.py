import pandas as pd
import numpy as np
from config import logger
from strat_models import BaseStrategy, EmaVolStrategy
from statsmodels.api import OLS,add_constant
from utils import align_idx


def calcaulte_pnl_stats(pnl: pd.Series, pos_raw: pd.Series=None, benchmark: pd.Series=None, print_results: bool=True) -> None:
    """Calculate in-sample PnL + turnover."""
    pnl_clean = pnl.dropna()
    pnl_daily = pnl_clean.resample('d').sum()
    if len(pnl_clean) == 0:
        raise ValueError("No valid PnL for stats")

    if pos_raw is not None:
        # --- Position change for turnover ---
        pos_change = pos_raw.diff().dropna()
        turnover_hourly = pos_change.abs().mean()
        turnover_ann = turnover_hourly * 24 * 360 # annual turnover (% of capital)

        # --- Time in Long vs Short ---
        time_in_long = (pos_raw > 0).mean()      # % of bars with long position
        time_in_short = (pos_raw < 0).mean()     # % of bars with short position
        time_in_neutral = 1 - time_in_long - time_in_short
    else:
        turnover_ann = np.nan
        time_in_long = np.nan      # % of bars with long position
        time_in_short = np.nan     # % of bars with short position
        time_in_neutral = np.nan


    # --- Metrics ---
    mean_ann = pnl_daily.mean() * 360
    vol_ann = pnl_daily.std() * 360**0.5
    downside = pnl_daily[pnl_daily < 0]
    down_vol = ((downside**2).mean()**0.5) * 360**0.5 if len(downside) > 0 else 1e-9

    var95 = np.percentile(pnl_daily, 5)
    cvar95 = pnl_daily[pnl_daily <= var95].mean()

    cum_ret = pnl_daily.cumsum()
    drawdown = cum_ret - cum_ret.cummax()
    dd95 = np.percentile(drawdown, 5)
    cdd95 = drawdown[drawdown <= dd95].mean()

    skew_mth = pnl_daily.resample('ME').sum().skew()
    kurt_mth = pnl_daily.resample('ME').sum().kurtosis()

    sharpe_ann = mean_ann / vol_ann if vol_ann != 0 else 0
    sortino_ann = mean_ann / down_vol if down_vol != 0 else 0
    calmar_ann = mean_ann / -cdd95 if cdd95 != 0 else 0

    if benchmark is not None:
        benchmark_daily = benchmark.resample('d').sum()
        x,y = align_idx(pnl_daily,benchmark_daily)
        cov = np.cov(y, x)[0, 1]
        var_bench = np.var(x)
        beta = cov / var_bench if var_bench != 0 else np.nan
        alpha_ann = (y.mean() - beta * x.mean()) * 360
    else:
        beta = np.nan
        alpha_ann = np.nan

    hit_rate = (pnl_clean > 0).mean()

    gross_profit = pnl_daily[pnl_daily > 0].sum()
    gross_loss = abs(pnl_daily[pnl_daily < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf

    # --- Store ---
    pnl_stats = {
        'mean_ann': mean_ann,
        'vol_ann': vol_ann,
        'cvar95': cvar95,
        'cdd95': cdd95,
        'skew_mth': skew_mth,
        'kurt_mth': kurt_mth,
        'sharpe_ann': sharpe_ann,
        'sortino_ann': sortino_ann,
        'calmar_ann': calmar_ann,
        'alpha_ann': alpha_ann,
        'beta': beta,
        'hit_rate': hit_rate,
        'profit_factor': profit_factor,
        'turnover_ann': turnover_ann,        # NEW
        'time_in_long': time_in_long,
        'time_in_short': time_in_short,
        'time_in_neutral': time_in_neutral,
        'n_samples': len(pnl_clean)
    }

    # --- Log ---
    pnl_stats_series = pd.Series(pnl_stats)
    if print_results:
        logger.info(f"In-sample ({len(pnl_clean)}) PnL & Turnover Stats:")
        logger.info(f"  Mean (ann):     {mean_ann:+.2%}")
        logger.info(f"  Vol (ann):      {vol_ann:.2%}")
        logger.info(f"  CVaR95:         {cvar95:+.2%}")
        logger.info(f"  CDD95:          {cdd95:+.2%}")
        logger.info(f"  Skew (mth):     {skew_mth:.2f}")
        logger.info(f"  Kurt (mth):     {kurt_mth:.2f}")
        logger.info(f"  Sharpe:         {sharpe_ann:.2f}")
        logger.info(f"  Sortino:        {sortino_ann:.2f}")
        logger.info(f"  Calmar:         {calmar_ann:.2f}")
        logger.info(f"  Alpha (ann):    {alpha_ann:+.2%}")
        logger.info(f"  Beta:           {beta:.2f}")
        logger.info(f"  Hit Rate:       {hit_rate:.2%}")
        logger.info(f"  Profit Factor:  {profit_factor:.2f}")
        logger.info(f"  Turnover (ann): {turnover_ann:.2f}")
        logger.info(f"  Time in Long:   {time_in_long:.2%}")
        logger.info(f"  Time in Short:  {time_in_short:.2%}")
        logger.info(f"  Time in Neutral:{time_in_neutral:.2%}")
    return pnl_stats_series

def regress_incremental(new_signal: pd.Series, bars: pd.DataFrame, horizon: int=24, interaction_only: bool=False, base: BaseStrategy=None) -> None:
    if base is None:
        base = EmaVolStrategy(
            fast_ema_window=24,
            slow_ema_multiplier=2,
            vol_window=24*30,
            weibull_c=2,
            alpha=1.0,
            target_vol=1.0,
            strat_weight=1.0
        )
    base.fit(bars)
    base_signal = base.signal(bars)

    prc = bars['close']
    r = np.log(prc).diff(horizon).shift(-horizon)*(24*360/horizon)
    v = np.log(prc).diff().ewm(span=24*30).std()*(24*360)**0.5
    y = (r / v).iloc[::horizon]

    x = pd.DataFrame({
        'base': base_signal,
        'new': new_signal,
        'inter': base_signal * new_signal
    })
    if interaction_only: x = x.drop(columns='new')
    x = x
    x,y = align_idx(x,y)
    model = OLS(y,add_constant(x)).fit()
    print(model.summary())
    print(x.corr())