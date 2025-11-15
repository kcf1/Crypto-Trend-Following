# strategy.py
import pandas as pd
import numpy as np
from scipy.stats import weibull_min
from sklearn.base import BaseEstimator
from typing import Tuple, Dict, Any
from config import logger
from utils import align_idx
import warnings
warnings.filterwarnings("ignore")

# Annualization: 24H bars → 360 trading days
ANNUAL_BARS = 24 * 360 # ≈ 18.973

class BaseStrategy(BaseEstimator):
    def __init__(self):
        super().__init__()
        
    def calculate_pnl_stats(self, pnl: pd.Series, pos_raw: pd.Series, benchmark: pd.Series) -> None:
        """Calculate in-sample PnL + turnover."""
        pnl_clean = pnl.dropna()
        if len(pnl_clean) == 0:
            raise ValueError("No valid PnL for stats")

        # --- Position change for turnover ---
        pos_change = pos_raw.diff().dropna()
        turnover_hourly = pos_change.abs().mean()
        turnover_ann = turnover_hourly * ANNUAL_BARS # annual turnover (% of capital)
        
        # --- Time in Long vs Short ---
        time_in_long = (pos_raw > 0).mean()      # % of bars with long position
        time_in_short = (pos_raw < 0).mean()     # % of bars with short position
        time_in_neutral = 1 - time_in_long - time_in_short

        # --- Metrics ---
        mean_ann = pnl_clean.mean() * ANNUAL_BARS
        vol_ann = pnl_clean.std() * ANNUAL_BARS**0.5
        downside = pnl_clean.clip(upper=0)
        down_vol = downside.std() * ANNUAL_BARS**0.5 if len(downside) > 0 else 1e-9

        var95 = np.percentile(pnl_clean, 5)
        cvar95 = pnl_clean[pnl_clean <= var95].mean()

        cum_ret = pnl_clean.cumsum()
        drawdown = cum_ret - cum_ret.cummax()
        dd95 = np.percentile(drawdown, 5)
        cdd95 = drawdown[drawdown <= dd95].mean()

        skew_ann = pnl_clean.skew()
        kurt_ann = pnl_clean.kurtosis()

        sharpe_ann = mean_ann / vol_ann if vol_ann != 0 else 0
        sortino_ann = mean_ann / down_vol if down_vol != 0 else 0
        calmar_ann = mean_ann / -cdd95 if cdd95 != 0 else 0

        x,y = align_idx(pnl_clean,benchmark)
        cov = np.cov(y, x)[0, 1]
        var_bench = np.var(x)
        beta = cov / var_bench if var_bench != 0 else np.nan
        alpha_ann = (y.mean() - beta * x.mean()) * ANNUAL_BARS

        hit_rate = (pnl_clean > 0).mean()

        gross_profit = pnl_clean[pnl_clean > 0].sum()
        gross_loss = abs(pnl_clean[pnl_clean < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf

        # --- Store ---
        self.pnl_stats = {
            'mean_ann': mean_ann,
            'vol_ann': vol_ann,
            'cvar95': cvar95,
            'cdd95': cdd95,
            'skew_ann': skew_ann,
            'kurt_ann': kurt_ann,
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
        logger.info(f"In-sample ({len(pnl_clean)}) PnL & Turnover Stats:")
        logger.info(f"  Mean (ann):     {mean_ann:+.2%}")
        logger.info(f"  Vol (ann):      {vol_ann:.2%}")
        logger.info(f"  CVaR95:         {cvar95:+.2%}")
        logger.info(f"  CDD95:          {cdd95:+.2%}")
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

    def one_step_predict(self, X: pd.Series | pd.DataFrame) -> float:
        """
        Predict next step position.
        
        Args:
            X: Required predictors
        
        Returns:
            float: Position as % of capital (e.g. 0.87 = 87%)
        """

        # Reuse full predict logic
        pos_series = self.predict(X)

        # Extract latest position
        one_step_pos = pos_series.iat[-1]

        logger.info(f"One-step position: {one_step_pos:+.2%}")
        return float(one_step_pos)

class EmaVolStrategy(BaseStrategy):
    """
    EMA Standardized × Vol State → Position (% of capital)
    
    signal_f = ema_standardized * vol_state_scaler[0,1]
    pos = signal_f * target_vol / asset_vol * to_target_scaler * strat_weight
    """
    
    def __init__(
        self,
        fast_ema_window: int = 24,
        slow_ema_multiplier: int = 2,
        vol_window: int = 24 * 30,
        weibull_c: float = 2.0,
        target_vol: float = 0.5,
        strat_weight: float = 1.0
    ):
        # --- Strategy Parameters ---
        self.fast_ema_window = fast_ema_window
        self.slow_ema_multiplier = slow_ema_multiplier
        self.vol_window = vol_window
        self.weibull_c = weibull_c
        self.target_vol = target_vol
        self.strat_weight = np.clip(strat_weight, 0.0, 1.0)
        
        # --- Fit Parameters (set in .fit()) ---
        self.cdf_median: float = 0.0
        self.to_target_scaler: float = 1.0
        
        # --- Intermediate Series ---
        self.s: pd.Series = pd.Series()
        self.v: pd.Series = pd.Series()
        self.v_cdf: pd.Series = pd.Series()
        self.scaler: pd.Series = pd.Series()
        self.f: pd.Series = pd.Series()
        self.pos_raw: pd.Series = pd.Series()
        self.final_pos: pd.Series = pd.Series()

        # Stats
        self.pnl_stats: Dict[str, float] = {}
        
        logger.info("EmaVolStrategy initialized")

    def _check_data_length(self, price: pd.Series, method_name: str) -> int:
        min_required = self.vol_window + self.fast_ema_window * self.slow_ema_multiplier

        if len(price) <= min_required:
            raise ValueError(
                f"Data too short for {method_name}. "
                f"Need > {min_required} rows, got {len(price)}. "
            )
        return min_required

    def fit(self, X: pd.Series, y: pd.Series = None) -> 'EmaVolStrategy':
        """
        Fit CDF median and to-target scaler from raw price.
        """
        self._check_data_length(X, ".fit()")

        prc = X
        log = np.log(prc)
        
        # 1. EMA Signal: S = fast_ema - slow_ema
        fast_ema = prc.ewm(span=self.fast_ema_window).mean()
        slow_ema = prc.ewm(span=self.fast_ema_window * self.slow_ema_multiplier).mean()
        self.s = fast_ema - slow_ema
        
        # 2. Standardize
        self.s = self.s / self.s.ewm(span=self.vol_window).std()
        self.s = self.s.iloc[self.vol_window + self.fast_ema_window * self.slow_ema_multiplier:]  # burn-in
        self.s = self.s.clip(-2, 2)
        
        # 3. Annualized hourly vol
        self.v = log.diff().ewm(span=self.vol_window).std() * ANNUAL_BARS**0.5
        
        # 4. Fit CDF median
        self.cdf_median = self.v.median()
        logger.info(f"Fit: CDF median = {self.cdf_median:.6f}")        
        
        # Simulate PnL to get to-target scaler
        self.v_cdf = pd.Series(
            weibull_min.cdf(self.v, c=self.weibull_c, scale=self.cdf_median * 1.5),
            index=self.v.index
        )
        self.scaler = 1 - self.v_cdf
        self.f = self.s * self.scaler
        pos_raw = self.target_vol / self.v * self.f
        pnl = pos_raw * log.diff().shift(-1)
        pnl = pnl.dropna()
        if len(pnl) == 0:
            raise ValueError("No valid PnL after alignment")
        
        actual_vol = pnl.std() * ANNUAL_BARS**0.5
        self.to_target_scaler = self.target_vol / actual_vol if not np.isnan(actual_vol) else 1

        logger.info(f"Fit: to_target_scaler = {self.to_target_scaler:.6f}")
        logger.success("Strategy fitted")
        
        # Final position
        final_pos = pos_raw * self.to_target_scaler * self.strat_weight
        final_pos = final_pos.reindex(prc.index)
        final_pnl = final_pos * log.diff().shift(-1)
        self.calculate_pnl_stats(final_pnl,final_pos,log.diff().shift(-1))

        return self

    def predict(self, X: pd.Series) -> pd.Series:
        """
        Generate position (% of capital) from latest data.
        """
        self._check_data_length(X, ".fit()")

        if not hasattr(self, 'cdf_median'):
            raise RuntimeError("Call .fit() first")
        
        prc = X
        log = np.log(prc)
        
        # Recompute intermediates on full data
        fast_ema = prc.ewm(span=self.fast_ema_window).mean()
        slow_ema = prc.ewm(span=self.fast_ema_window * self.slow_ema_multiplier).mean()
        s = fast_ema - slow_ema
        s = s / s.ewm(span=self.vol_window).std()
        s = s.iloc[self.vol_window + self.fast_ema_window * self.slow_ema_multiplier:]  # burn-in
        s = s.clip(-2, 2)

        v = log.diff().ewm(span=self.vol_window).std() * ANNUAL_BARS**0.5
        
        # CDF scaler
        v_cdf = pd.Series(
            weibull_min.cdf(v, c=self.weibull_c, scale=self.cdf_median * 1.5),
            index=v.index
        )
        scaler = 1 - v_cdf
        
        # Signal
        f = s * scaler
        
        # Raw position
        pos_raw = self.target_vol / v * f
        
        # Final position
        final_pos = pos_raw * self.to_target_scaler * self.strat_weight
        final_pos = final_pos.reindex(prc.index)
        
        #logger.info(f"Latest position: {final_pos.iloc[-1]:+.2%}")
        return final_pos

    def get_intermediates(self) -> Dict[str, Any]:
        """Return all intermediate series for inspection."""
        return {
            's': self.s,
            'v': self.v,
            'v_cdf': self.v_cdf if hasattr(self, 'v_cdf') else None,
            'scaler': self.scaler,
            'f': self.f,
            'pos_raw': self.pos_raw,
            'final_pos': self.final_pos,
            'params': {
                'fast_ema_window': self.fast_ema_window,
                'slow_ema_multiplier': self.slow_ema_multiplier,
                'vol_window': self.vol_window,
                'weibull_c': self.weibull_c,
                'target_vol': self.target_vol,
                'strat_weight': self.strat_weight,
                'cdf_median': self.cdf_median,
                'to_target_scaler': self.to_target_scaler
            },
            'pnl_stats': self.pnl_stats
        }