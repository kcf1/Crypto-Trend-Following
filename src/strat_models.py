# strategy.py
import pandas as pd
import numpy as np
from scipy.stats import weibull_min,norm
from sklearn.base import BaseEstimator
from typing import Tuple, Dict, Any, List
from statsmodels.regression.rolling import RollingOLS
from statsmodels.api import OLS,add_constant
from config import logger
from utils import align_idx
from fast_func import rolling_slope_vec,rolling_ols_coef_fast,rogers_satchell_volatility
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

        skew_mth = pnl_clean.resample('ME').sum().skew()
        kurt_mth = pnl_clean.resample('ME').sum().kurtosis()

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
        print(pd.Series(self.pnl_stats))

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

class VolScaleStrategy(BaseStrategy):
    """
    Simple volatility scaling:
    pos = target_vol / vol_forecast * strat_weight
    """
    
    def __init__(
        self,
        vol_window: int = 24 * 30,
        target_vol: float = 0.5,
        strat_weight: float = 1.0
    ):
        # --- Parameters ---
        self.vol_window = vol_window
        self.target_vol = target_vol
        self.strat_weight = np.clip(strat_weight, 0.0, 1.0)
        
        # --- Fit Parameters ---
        self.to_target_scaler: float = 1.0

        # Stats
        self.pnl_stats: Dict[str, float] = {}
        
        logger.info("VolScaleStrategy initialized")

    def _check_data_length(self, price: pd.Series, method_name: str) -> int:
        min_required = self.vol_window

        if len(price) <= min_required:
            raise ValueError(
                f"Data too short for {method_name}. "
                f"Need > {min_required} rows, got {len(price)}. "
            )
        return min_required

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'EmaVolStrategy':
        """
        Fit to-target scaler from in-sample PnL.
        """
        self._check_data_length(X, ".fit()")

        if not all(col in X.columns for col in ['close']):
            raise ValueError("X must contain 'close'")

        prc = X['close']
        log = np.log(prc)

        # 1. Volatility forecast (EWM std)
        #v = log.diff().ewm(span=self.vol_window).std() * ANNUAL_BARS**0.5
        v = rogers_satchell_volatility(X,window=round(self.vol_window*.4)) * ANNUAL_BARS**0.5
        v = v.iloc[self.vol_window:]  # burn-in

        # 2. Raw position
        pos_raw = self.target_vol / v

        # 3. Simulate PnL
        pnl = pos_raw * log.diff().shift(-1)
        pnl = pnl.dropna()
        if len(pnl) == 0:
            raise ValueError("No valid PnL after alignment")

        # 4. Fit to-target scaler
        actual_vol = pnl.std() * ANNUAL_BARS**0.5
        self.to_target_scaler = self.target_vol / actual_vol if actual_vol > 0 else 1.0

        logger.info(f"Fit: to_target_scaler = {self.to_target_scaler:.6f}")

        # 5. Final position + stats
        final_pos = pos_raw * self.to_target_scaler * self.strat_weight
        final_pos = final_pos.reindex(prc.index)
        final_pnl = final_pos * log.diff().shift(-1)
        
        #self.calculate_pnl_stats(final_pnl, final_pos, log.diff().shift(-1))

        logger.success("Strategy fitted")

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate position from latest price.
        """
        self._check_data_length(X, ".predict()")
        if not hasattr(self, 'to_target_scaler'):
            raise RuntimeError("Call .fit() first")

        prc = X['close']
        log = np.log(prc)

        # Recompute volatility
        #v = log.diff().ewm(span=self.vol_window).std() * ANNUAL_BARS**0.5
        v = rogers_satchell_volatility(X,window=round(self.vol_window*.4)) * ANNUAL_BARS**0.5
        v = v.iloc[self.vol_window:]

        # Raw position
        pos_raw = self.target_vol / v

        # Final position
        final_pos = pos_raw * self.to_target_scaler * self.strat_weight
        final_pos = final_pos.reindex(prc.index)
        
        #logger.info(f"Latest position: {final_pos.iloc[-1]:+.2%}")
        return final_pos

    def get_intermediates(self) -> Dict[str, Any]:
        """Return all intermediate series for inspection."""
        return {
            'params': {
                'vol_window': self.vol_window,
                'target_vol': self.target_vol,
                'strat_weight': self.strat_weight,
                'to_target_scaler': self.to_target_scaler
            },
            'pnl_stats': self.pnl_stats
        }
    
class WedThuStrategy(BaseStrategy):
    """
    Crazy long wed / short thu strategy (why??):
    wed_pos = 1 / wed_vol * strat_weight
    thu_pos = 1 / thu_vol * strat_weight
    """
    
    def __init__(
        self,
        vol_window: int = 24 * 30,
        target_vol: float = 0.5,
        strat_weight: float = 1.0
    ):
        # --- Parameters ---
        self.vol_window = vol_window
        self.target_vol = target_vol
        self.strat_weight = np.clip(strat_weight, 0.0, 1.0)
        
        # --- Fit Parameters ---
        self.to_target_scaler: float = 1.0

        # Stats
        self.pnl_stats: Dict[str, float] = {}
        
        logger.info("WedThuStrategy initialized")

    def _check_data_length(self, price: pd.Series, method_name: str) -> int:
        min_required = self.vol_window

        if len(price) <= min_required:
            raise ValueError(
                f"Data too short for {method_name}. "
                f"Need > {min_required} rows, got {len(price)}. "
            )
        return min_required

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'EmaVolStrategy':
        """
        Fit to-target scaler from in-sample PnL.
        """
        self._check_data_length(X, ".fit()")

        if not all(col in X.columns for col in ['close']):
            raise ValueError("X must contain 'close'")

        prc = X['close']
        log = np.log(prc)

        # 1. Wed long / Thu short
        wed = prc.index.weekday==2
        thu = prc.index.weekday==3
        s = pd.Series(0, prc.index)
        s[wed] = 1
        s[thu] = -1
        
        # 2. Corresponding volatility
        scaled_vol_window = int(self.vol_window * 1/7) # correct vol window to capture 1 day in the week
        v_wed = log.diff()[wed].ewm(span=scaled_vol_window).std()
        v_thu = log.diff()[thu].ewm(span=scaled_vol_window).std()
        v_wed = v_wed.iloc[self.vol_window:] * ANNUAL_BARS**0.5
        v_thu = v_thu.iloc[self.vol_window:] * ANNUAL_BARS**0.5
        v = pd.concat([v_wed,v_thu])
        v = v.reindex(s.index).fillna(1)

        # 3. Raw position
        pos_raw = s * self.target_vol / v

        # 4. Simulate PnL
        pnl = pos_raw * log.diff().shift(-1)
        pnl = pnl.dropna()
        if len(pnl) == 0:
            raise ValueError("No valid PnL after alignment")

        # 4. Fit to-target scaler
        actual_vol = pnl.std() * ANNUAL_BARS**0.5
        self.to_target_scaler = self.target_vol / actual_vol if actual_vol > 0 else 1.0

        logger.info(f"Fit: to_target_scaler = {self.to_target_scaler:.6f}")

        # 5. Final position + stats
        final_pos = pos_raw * self.to_target_scaler * self.strat_weight
        final_pos = final_pos.reindex(prc.index)
        final_pnl = final_pos * log.diff().shift(-1)
        
        #self.calculate_pnl_stats(final_pnl, final_pos, log.diff().shift(-1))

        logger.success("Strategy fitted")

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate position from latest price.
        """
        self._check_data_length(X, ".predict()")
        if not hasattr(self, 'to_target_scaler'):
            raise RuntimeError("Call .fit() first")

        prc = X['close']
        log = np.log(prc)

        # 1. Wed long / Thu short
        wed = prc.index.weekday==2
        thu = prc.index.weekday==3
        s = pd.Series(0, prc.index)
        s[wed] = 1
        s[thu] = -1
        
        # 2. Corresponding volatility
        scaled_vol_window = int(self.vol_window * 1/7) # correct vol window to capture 1 day in the week
        v_wed = log.diff()[wed].ewm(span=scaled_vol_window).std()
        v_thu = log.diff()[thu].ewm(span=scaled_vol_window).std()
        v_wed = v_wed.iloc[self.vol_window:] * ANNUAL_BARS**0.5
        v_thu = v_thu.iloc[self.vol_window:] * ANNUAL_BARS**0.5
        v = pd.concat([v_wed,v_thu])
        v = v.reindex(s.index).fillna(1)

        # 3. Raw position
        pos_raw = s * self.target_vol / v

        # Final position
        final_pos = pos_raw * self.to_target_scaler * self.strat_weight
        final_pos = final_pos.reindex(prc.index)
        
        #logger.info(f"Latest position: {final_pos.iloc[-1]:+.2%}")
        return final_pos
    
    def signal(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate raw signal from latest data.
        """
        self._check_data_length(X, ".predict()")
        if not hasattr(self, 'to_target_scaler'):
            raise RuntimeError("Call .fit() first")

        prc = X['close']
        log = np.log(prc)

        # 1. Wed long / Thu short
        wed = prc.index.weekday==2
        thu = prc.index.weekday==3
        s = pd.Series(0, prc.index)
        s[wed] = 1
        s[thu] = -1
        
        # 2. Corresponding volatility
        scaled_vol_window = int(self.vol_window * 1/7) # correct vol window to capture 1 day in the week
        v_wed = log.diff()[wed].ewm(span=scaled_vol_window).std()
        v_thu = log.diff()[thu].ewm(span=scaled_vol_window).std()
        v_wed = v_wed.iloc[self.vol_window:] * ANNUAL_BARS**0.5
        v_thu = v_thu.iloc[self.vol_window:] * ANNUAL_BARS**0.5
        v = pd.concat([v_wed,v_thu])
        v = v.reindex(s.index).fillna(1)

        # 3. Raw position
        pos_raw = s * self.target_vol / v * self.to_target_scaler * self.strat_weight
        f = pos_raw.reindex(prc.index)
        
        #logger.info(f"Latest position: {final_pos.iloc[-1]:+.2%}")
        return f

    def get_intermediates(self) -> Dict[str, Any]:
        """Return all intermediate series for inspection."""
        return {
            'params': {
                'vol_window': self.vol_window,
                'target_vol': self.target_vol,
                'strat_weight': self.strat_weight,
                'to_target_scaler': self.to_target_scaler
            },
            'pnl_stats': self.pnl_stats
        }

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
        alpha: float = 1.0,              # Weight of vol_tilt (0 = no tilt, 1 = full tilt)
        target_vol: float = 0.5,
        strat_weight: float = 1.0
    ):
        # --- Strategy Parameters ---
        self.fast_ema_window = fast_ema_window
        self.slow_ema_multiplier = slow_ema_multiplier
        self.vol_window = vol_window
        self.weibull_c = weibull_c
        self.alpha = np.clip(alpha, 0.0, 1.0)
        self.target_vol = target_vol
        self.strat_weight = np.clip(strat_weight, 0.0, 1.0)
        
        # --- Fit Parameters (set in .fit()) ---
        self.cdf_median: float = 0.0
        self.to_target_scaler: float = 1.0

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

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'EmaVolStrategy':
        """
        Fit CDF median and to-target scaler from raw price.
        """
        self._check_data_length(X, ".fit()")

        if not all(col in X.columns for col in ['close']):
            raise ValueError("X must contain 'close'")

        prc = X['close']
        log = np.log(prc)
        
        # 1. EMA Signal: S = fast_ema - slow_ema
        fast_ema = prc.ewm(span=self.fast_ema_window).mean()
        slow_ema = prc.ewm(span=self.fast_ema_window * self.slow_ema_multiplier).mean()
        s = fast_ema - slow_ema
        
        # 2. Standardize
        s = s / s.ewm(span=self.vol_window).std()
        s = s.iloc[self.vol_window + self.fast_ema_window * self.slow_ema_multiplier:]  # burn-in
        s = s.clip(-2, 2)
        
        # 3. Annualized hourly vol
        #v = log.diff().ewm(span=self.vol_window).std() * ANNUAL_BARS**0.5
        v = rogers_satchell_volatility(X,window=round(self.vol_window*.4)) * ANNUAL_BARS**0.5
        
        # 4. Fit CDF median
        self.cdf_median = v.median()
        logger.info(f"Fit: CDF median = {self.cdf_median:.6f}")        
        
        # Simulate PnL to get to-target scaler
        v_cdf = pd.Series(
            weibull_min.cdf(v, c=self.weibull_c, scale=self.cdf_median * 1.5),
            index=v.index
        )
        vol_tilt = 1 - v_cdf
        # 5. Combined Signal with alpha blending
        tilt_component = (1.0 - self.alpha) + self.alpha * vol_tilt

        f = s * tilt_component
        pos_raw = self.target_vol / v * f
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
        #self.calculate_pnl_stats(final_pnl,final_pos,log.diff().shift(-1))

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate position (% of capital) from latest data.
        """
        self._check_data_length(X, ".fit()")

        if not all(col in X.columns for col in ['close']):
            raise ValueError("X must contain 'close'")

        if not hasattr(self, 'cdf_median'):
            raise RuntimeError("Call .fit() first")
        
        prc = X['close']
        log = np.log(prc)
        
        # Recompute intermediates on full data
        fast_ema = prc.ewm(span=self.fast_ema_window).mean()
        slow_ema = prc.ewm(span=self.fast_ema_window * self.slow_ema_multiplier).mean()
        s = fast_ema - slow_ema
        s = s / s.ewm(span=self.vol_window).std()
        s = s.iloc[self.vol_window + self.fast_ema_window * self.slow_ema_multiplier:]  # burn-in
        s = s.clip(-2, 2)

        #v = log.diff().ewm(span=self.vol_window).std() * ANNUAL_BARS**0.5
        v = rogers_satchell_volatility(X,window=round(self.vol_window*.4)) * ANNUAL_BARS**0.5
        
        # CDF scaler
        v_cdf = pd.Series(
            weibull_min.cdf(v, c=self.weibull_c, scale=self.cdf_median * 1.5),
            index=v.index
        )
        vol_tilt = 1 - v_cdf
        # 5. Combined Signal with alpha blending
        tilt_component = (1.0 - self.alpha) + self.alpha * vol_tilt
        
        # Signal
        f = s * tilt_component
        
        # Raw position
        pos_raw = self.target_vol / v * f
        
        # Final position
        final_pos = pos_raw * self.to_target_scaler * self.strat_weight
        final_pos = final_pos.reindex(prc.index)
        
        #logger.info(f"Latest position: {final_pos.iloc[-1]:+.2%}")
        return final_pos
    
    def signal(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate raw signal from latest data.
        """
        self._check_data_length(X, ".fit()")

        if not all(col in X.columns for col in ['close']):
            raise ValueError("X must contain 'close'")

        if not hasattr(self, 'cdf_median'):
            raise RuntimeError("Call .fit() first")
        
        prc = X['close']
        log = np.log(prc)
        
        # Recompute intermediates on full data
        fast_ema = prc.ewm(span=self.fast_ema_window).mean()
        slow_ema = prc.ewm(span=self.fast_ema_window * self.slow_ema_multiplier).mean()
        s = fast_ema - slow_ema
        s = s / s.ewm(span=self.vol_window).std()
        s = s.iloc[self.vol_window + self.fast_ema_window * self.slow_ema_multiplier:]  # burn-in
        s = s.clip(-2, 2)

        #v = log.diff().ewm(span=self.vol_window).std() * ANNUAL_BARS**0.5
        v = rogers_satchell_volatility(X,window=round(self.vol_window*.4)) * ANNUAL_BARS**0.5
        
        # CDF scaler
        v_cdf = pd.Series(
            weibull_min.cdf(v, c=self.weibull_c, scale=self.cdf_median * 1.5),
            index=v.index
        )
        vol_tilt = 1 - v_cdf
        # 5. Combined Signal with alpha blending
        tilt_component = (1.0 - self.alpha) + self.alpha * vol_tilt
        
        # Signal
        f = s * tilt_component * self.to_target_scaler * self.strat_weight
        f = f.reindex(prc.index)
        
        #logger.info(f"Latest position: {final_pos.iloc[-1]:+.2%}")
        return f

    def get_intermediates(self) -> Dict[str, Any]:
        """Return all intermediate series for inspection."""
        return {
            'params': {
                'fast_ema_window': self.fast_ema_window,
                'slow_ema_multiplier': self.slow_ema_multiplier,
                'vol_window': self.vol_window,
                'weibull_c': self.weibull_c,
                'alpha': self.alpha,
                'target_vol': self.target_vol,
                'strat_weight': self.strat_weight,
                'cdf_median': self.cdf_median,
                'to_target_scaler': self.to_target_scaler
            },
            'pnl_stats': self.pnl_stats
        }
    
class AccelVolStrategy(BaseStrategy):
    """
    EMA Crossover Acceleration Strategy (Trend Momentum-of-Momentum)

    Signal flow:
        1. s_level     = (fast_ema - slow_ema) / vol_std          → standardized crossover
        2. s_accel_raw = s_level.diff(fast_ema_window * diff_multiplier)
        3. s_accel     = s_accel_raw / s_accel_raw.ewm(vol_window).std()   → RE-STANDARDIZED
        4. final_f     = s_accel * ((1-alpha) + alpha * vol_tilt)
        pos            = final_f * target_vol / asset_vol * scaler * weight
    """
    
    def __init__(
        self,
        fast_ema_window: int = 24,
        slow_ema_multiplier: int = 2,
        diff_multiplier: float = 0.5,        # NEW: controls diff lookback = fast * multiplier
        vol_window: int = 24 * 30,
        weibull_c: float = 2.0,
        alpha: float = 1.0,
        target_vol: float = 0.5,
        strat_weight: float = 1.0
    ):
        self.fast_ema_window = fast_ema_window
        self.slow_ema_multiplier = slow_ema_multiplier
        self.diff_multiplier = diff_multiplier
        self.vol_window = vol_window
        self.weibull_c = weibull_c
        self.alpha = np.clip(alpha, 0.0, 1.0)
        self.target_vol = target_vol
        self.strat_weight = np.clip(strat_weight, 0.0, 1.0)

        # Fit params
        self.cdf_median = 0.0
        self.to_target_scaler = 1.0

        logger.info("AccelVolStrategy initialized")

    def _check_data_length(self, df: pd.DataFrame, method_name: str):
        min_req = self.vol_window + self.fast_ema_window * self.slow_ema_multiplier \
                  + int(self.fast_ema_window * self.diff_multiplier) + 200
        if len(df) <= min_req:
            raise ValueError(f"Data too short for {method_name}: need >{min_req}, got {len(df)}")

    def fit(self, X: pd.DataFrame, y=None) -> 'AccelVolStrategy':
        self._check_data_length(X, ".fit()")
        if 'close' not in X.columns:
            raise ValueError("X must contain 'close'")

        prc = X['close']
        log = np.log(prc)
        diff_window = int(self.fast_ema_window * self.diff_multiplier)

        # 1. Level signal (standardized EMA crossover)
        fast = prc.ewm(span=self.fast_ema_window).mean()
        slow = prc.ewm(span=self.fast_ema_window * self.slow_ema_multiplier).mean()
        s_level = fast - slow
        s_level = s_level / s_level.ewm(span=self.vol_window).std()

        # 2. Acceleration = diff of standardized level
        s_accel_raw = s_level.diff(diff_window)

        # 3. CRITICAL: re-standardize the acceleration
        s_accel = s_accel_raw / s_accel_raw.ewm(span=self.vol_window).std()
        s_accel = s_accel.clip(-2, 2)

        burn_in = self.vol_window + self.fast_ema_window * self.slow_ema_multiplier + diff_window
        s_accel = s_accel.iloc[burn_in:]

        # 4. Volatility
        #v = log.diff().ewm(span=self.vol_window).std() * ANNUAL_BARS**0.5
        v = rogers_satchell_volatility(X,window=round(self.vol_window*.4)) * ANNUAL_BARS**0.5

        # 5. Vol tilt (Weibull)
        self.cdf_median = v.median()
        scale = self.cdf_median * 1.5
        v_cdf = pd.Series(weibull_min.cdf(v, c=self.weibull_c, scale=scale), index=v.index)
        vol_tilt = 1 - v_cdf.clip(0, 1)

        # 6. Final blended signal
        tilt_comp = (1.0 - self.alpha) + self.alpha * vol_tilt
        f = s_accel * tilt_comp

        # Align
        idx = f.index.intersection(v.index)
        f = f.loc[idx]
        v = v.loc[idx]

        # 7. Calibrate scaler
        pos_raw = self.target_vol / v * f
        pnl = pos_raw * log.diff().shift(-1).loc[idx].dropna()
        actual_vol = pnl.std() * ANNUAL_BARS**0.5
        self.to_target_scaler = self.target_vol / actual_vol if actual_vol > 1e-8 else 1.0

        logger.info(f"EmaDiffVolStrategy fitted | "
                    f"diff_window={diff_window}h | "
                    f"alpha={self.alpha:.2f} | "
                    f"scaler={self.to_target_scaler:.3f}")

        # Stats
        final_pos = pos_raw * self.to_target_scaler * self.strat_weight
        final_pos = final_pos.reindex(prc.index)
        final_pnl = final_pos * log.diff().shift(-1)
        #self.calculate_pnl_stats(final_pnl.dropna(), final_pos, log.diff().shift(-1))

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        self._check_data_length(X, ".predict()")
        if not hasattr(self, 'cdf_median'):
            raise RuntimeError("Call .fit() first")

        prc = X['close']
        log = np.log(prc)
        diff_window = int(self.fast_ema_window * self.diff_multiplier)

        fast = prc.ewm(span=self.fast_ema_window).mean()
        slow = prc.ewm(span=self.fast_ema_window * self.slow_ema_multiplier).mean()
        s_level = (fast - slow) / (fast - slow).ewm(span=self.vol_window).std()

        s_accel_raw = s_level.diff(diff_window)
        s_accel = s_accel_raw / s_accel_raw.ewm(span=self.vol_window).std()
        s_accel = s_accel.clip(-2, 2)

        #v = log.diff().ewm(span=self.vol_window).std() * ANNUAL_BARS**0.5
        v = rogers_satchell_volatility(X,window=round(self.vol_window*.4)) * ANNUAL_BARS**0.5

        scale = self.cdf_median * 1.5
        v_cdf = pd.Series(weibull_min.cdf(v, c=self.weibull_c, scale=scale), index=v.index)
        vol_tilt = 1 - v_cdf.clip(0, 1)

        tilt_comp = (1.0 - self.alpha) + self.alpha * vol_tilt
        f = s_accel * tilt_comp

        idx = f.index.intersection(v.index)
        pos = pd.Series(0.0, index=X.index)
        pos.loc[idx] = f.loc[idx] * self.target_vol / v.loc[idx]
        pos = pos * self.to_target_scaler * self.strat_weight

        return pos.reindex(X.index, fill_value=0.0)
    
    def signal(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate raw signal from latest data.
        """
        
        self._check_data_length(X, ".predict()")
        if not hasattr(self, 'cdf_median'):
            raise RuntimeError("Call .fit() first")

        prc = X['close']
        log = np.log(prc)
        diff_window = int(self.fast_ema_window * self.diff_multiplier)

        fast = prc.ewm(span=self.fast_ema_window).mean()
        slow = prc.ewm(span=self.fast_ema_window * self.slow_ema_multiplier).mean()
        s_level = (fast - slow) / (fast - slow).ewm(span=self.vol_window).std()

        s_accel_raw = s_level.diff(diff_window)
        s_accel = s_accel_raw / s_accel_raw.ewm(span=self.vol_window).std()
        s_accel = s_accel.clip(-2, 2)

        #v = log.diff().ewm(span=self.vol_window).std() * ANNUAL_BARS**0.5
        v = rogers_satchell_volatility(X,window=round(self.vol_window*.4)) * ANNUAL_BARS**0.5

        scale = self.cdf_median * 1.5
        v_cdf = pd.Series(weibull_min.cdf(v, c=self.weibull_c, scale=scale), index=v.index)
        vol_tilt = 1 - v_cdf.clip(0, 1)

        tilt_comp = (1.0 - self.alpha) + self.alpha * vol_tilt
        f = s_accel * tilt_comp * self.to_target_scaler * self.strat_weight
        f = f.reindex(prc.index)
        
        #logger.info(f"Latest position: {final_pos.iloc[-1]:+.2%}")
        return f

    def get_intermediates(self) -> Dict[str, Any]:
        return {
            'params': {
                'fast_ema_window': self.fast_ema_window,
                'slow_ema_multiplier': self.slow_ema_multiplier,
                'diff_multiplier': self.diff_multiplier,
                'diff_window_actual': int(self.fast_ema_window * self.diff_multiplier),
                'alpha': self.alpha,
                'target_vol': self.target_vol,
                'to_target_scaler': self.to_target_scaler,
            },
            'pnl_stats': self.pnl_stats
        }
    
class BreakVolStrategy(BaseStrategy):
    """
    Breakout (smoothed) × Vol State → Position (% of capital)
    
    signal_f = breakout_standardized * vol_state_scaler[0,1]
    pos = signal_f * target_vol / asset_vol * to_target_scaler * strat_weight
    """
    
    def __init__(
        self,
        breakout_window: int = 24 * 2,
        smooth_window: int = 12,         # NEW: smooth signal
        vol_window: int = 24 * 30,
        weibull_c: float = 2.0,
        alpha: float = 1.0,              # Weight of vol_tilt (0 = no tilt, 1 = full tilt)
        target_vol: float = 0.5,
        strat_weight: float = 1.0
    ):
        # --- Parameters ---
        self.breakout_window = breakout_window
        self.smooth_window = smooth_window
        self.vol_window = vol_window
        self.weibull_c = weibull_c
        self.alpha = np.clip(alpha, 0.0, 1.0)
        self.target_vol = target_vol
        self.strat_weight = np.clip(strat_weight, 0.0, 1.0)
        
        # --- Fit Parameters ---
        self.cdf_median: float = 0.0
        self.to_target_scaler: float = 1.0

        # Stats
        self.pnl_stats: Dict[str, float] = {}
        
        logger.info("BreakVolStrategy initialized")

    def _check_data_length(self, df: pd.DataFrame, method_name: str) -> int:
        min_required = max(self.breakout_window, self.vol_window, self.smooth_window) + 1
        if len(df) <= min_required:
            raise ValueError(
                f"Data too short for {method_name}. "
                f"Need > {min_required} rows, got {len(df)}."
            )
        return min_required

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'BreakVolStrategy':
        self._check_data_length(X, ".fit()")

        if not all(col in X.columns for col in ['close']):
            raise ValueError("X must contain 'close'")

        prc = X['close']
        log = np.log(prc)

        # 1. Rolling high/low
        h = prc.rolling(self.breakout_window).max()
        l = prc.rolling(self.breakout_window).min()

        # 2. Raw breakout signal
        mid = (h + l) / 2
        rng = h - l
        s = ((prc - mid) / rng * 2).replace([np.inf, -np.inf], np.nan)
        s = s.iloc[self.breakout_window:]

        # 3. Smooth signal
        s_smooth = s.ewm(span=self.smooth_window).mean()
        s_smooth = s_smooth.iloc[self.smooth_window:]

        # 4. Annualized hourly vol
        #v = log.diff().ewm(span=self.vol_window).std() * ANNUAL_BARS**0.5
        v = rogers_satchell_volatility(X,window=round(self.vol_window*.4)) * ANNUAL_BARS**0.5
        v = v.iloc[self.vol_window:]

        # 5. Fit CDF median
        self.cdf_median = v.median()
        logger.info(f"Fit: CDF median = {self.cdf_median:.6f}")

        # 6. Vol tilt
        v_cdf = pd.Series(
            weibull_min.cdf(v, c=self.weibull_c, scale=self.cdf_median * 1.5),
            index=v.index
        )
        vol_tilt = 1 - v_cdf
        # 5. Combined Signal with alpha blending
        tilt_component = (1.0 - self.alpha) + self.alpha * vol_tilt

        # 7. Final signal
        f = s_smooth * tilt_component
        pos_raw = self.target_vol / v * f

        # 8. Simulate PnL
        pnl = pos_raw * log.diff().shift(-1)
        pnl = pnl.dropna()
        if len(pnl) == 0:
            raise ValueError("No valid PnL after alignment")

        # 9. Fit to-target scaler
        actual_vol = pnl.std() * ANNUAL_BARS**0.5
        self.to_target_scaler = self.target_vol / actual_vol if actual_vol > 0 else 1.0

        logger.info(f"Fit: to_target_scaler = {self.to_target_scaler:.6f}")

        # 10. Final position + stats
        final_pos = pos_raw * self.to_target_scaler * self.strat_weight
        final_pos = final_pos.reindex(prc.index)
        final_pnl = final_pos * log.diff().shift(-1)
        
        #self.calculate_pnl_stats(final_pnl, final_pos, log.diff().shift(-1))

        logger.success("BreakVolStrategy fitted")
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        self._check_data_length(X, ".predict()")
        if not hasattr(self, 'cdf_median'):
            raise RuntimeError("Call .fit() first")

        if not all(col in X.columns for col in ['close']):
            raise ValueError("X must contain 'close'")

        prc = X['close']
        log = np.log(prc)

        # Rolling high/low
        h = prc.rolling(self.breakout_window).max()
        l = prc.rolling(self.breakout_window).min()

        # Raw breakout
        mid = (h + l) / 2
        rng = h - l
        s = ((prc - mid) / rng * 2).replace([np.inf, -np.inf], np.nan)
        s = s.iloc[self.breakout_window:]

        # Smooth
        s_smooth = s.ewm(span=self.smooth_window).mean()
        s_smooth = s_smooth.iloc[self.smooth_window:]

        # Vol forecast
        #v = log.diff().ewm(span=self.vol_window).std() * ANNUAL_BARS**0.5
        v = rogers_satchell_volatility(X,window=round(self.vol_window*.4)) * ANNUAL_BARS**0.5
        v = v.iloc[self.vol_window:]

        # Vol tilt
        v_cdf = pd.Series(
            weibull_min.cdf(v, c=self.weibull_c, scale=self.cdf_median * 1.5),
            index=v.index
        )
        vol_tilt = 1 - v_cdf
        # 5. Combined Signal with alpha blending
        tilt_component = (1.0 - self.alpha) + self.alpha * vol_tilt

        # Final signal
        f = s_smooth * tilt_component
        pos_raw = self.target_vol / v * f

        # Final position
        final_pos = pos_raw * self.to_target_scaler * self.strat_weight
        final_pos = final_pos.reindex(prc.index)

        logger.info(f"Latest position: {final_pos.iloc[-1]:+.2%}")
        return final_pos
    
    def signal(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate raw signal from latest data.
        """
        self._check_data_length(X, ".predict()")
        if not hasattr(self, 'cdf_median'):
            raise RuntimeError("Call .fit() first")

        if not all(col in X.columns for col in ['close']):
            raise ValueError("X must contain 'close'")

        prc = X['close']
        log = np.log(prc)

        # Rolling high/low
        h = prc.rolling(self.breakout_window).max()
        l = prc.rolling(self.breakout_window).min()

        # Raw breakout
        mid = (h + l) / 2
        rng = h - l
        s = ((prc - mid) / rng * 2).replace([np.inf, -np.inf], np.nan)
        s = s.iloc[self.breakout_window:]

        # Smooth
        s_smooth = s.ewm(span=self.smooth_window).mean()
        s_smooth = s_smooth.iloc[self.smooth_window:]

        # Vol forecast
        #v = log.diff().ewm(span=self.vol_window).std() * ANNUAL_BARS**0.5
        v = rogers_satchell_volatility(X,window=round(self.vol_window*.4)) * ANNUAL_BARS**0.5
        v = v.iloc[self.vol_window:]

        # Vol tilt
        v_cdf = pd.Series(
            weibull_min.cdf(v, c=self.weibull_c, scale=self.cdf_median * 1.5),
            index=v.index
        )
        vol_tilt = 1 - v_cdf
        # 5. Combined Signal with alpha blending
        tilt_component = (1.0 - self.alpha) + self.alpha * vol_tilt

        # Final signal
        f = s_smooth * tilt_component * self.to_target_scaler * self.strat_weight
        f = f.reindex(prc.index)
        
        #logger.info(f"Latest position: {final_pos.iloc[-1]:+.2%}")
        return f

    def get_intermediates(self) -> Dict[str, Any]:
        return {
            'params': {
                'breakout_window': self.breakout_window,
                'smooth_window': self.smooth_window,
                'vol_window': self.vol_window,
                'weibull_c': self.weibull_c,
                'alpha': self.alpha,
                'target_vol': self.target_vol,
                'strat_weight': self.strat_weight,
                'cdf_median': self.cdf_median,
                'to_target_scaler': self.to_target_scaler
            },
            'pnl_stats': self.pnl_stats,
            'pnl_stats_series': self.pnl_stats_series
        }
    
class BlockVolStrategy(BaseStrategy):
    """
    Block Momentum Strategy (Higher High + Higher Low)
    
    Core signal:
        h, l       = rolling high/low over block_window
        hh, ll     = h.diff(block_window), l.diff(block_window)
        s_raw      = (hh + ll) / 2 / (h - l)     → normalized block momentum
        s          = EMA(s_raw)                 → smoothed
        final_f    = s * ((1-alpha) + alpha * vol_tilt)
        pos        = final_f * target_vol / asset_vol * scaler * weight
    
    Captures sustained directional block structure (trend continuation).
    """
    
    def __init__(
        self,
        block_window: int = 48,          # Lookback for high/low block (e.g. 48h = 2 days)
        smooth_window: int = 12,         # Smoothing EMA span
        vol_window: int = 24 * 30,
        weibull_c: float = 2.0,
        alpha: float = 1.0,              # Weight of vol_tilt (0 = no tilt, 1 = full)
        target_vol: float = 0.5,
        strat_weight: float = 1.0
    ):
        # --- Strategy Parameters ---
        self.block_window = block_window
        self.smooth_window = smooth_window
        # New
        self.vol_window = vol_window
        self.weibull_c = weibull_c
        self.alpha = np.clip(alpha, 0.0, 1.0)
        self.target_vol = target_vol
        self.strat_weight = np.clip(strat_weight, 0.0, 1.0)
        
        # --- Fit Parameters ---
        self.cdf_median: float = 0.0
        self.to_target_scaler: float = 1.0

        # Stats
        self.pnl_stats: Dict[str, float] = {}
        
        logger.info("BlockVolStrategy initialized")

    def _check_data_length(self, df: pd.DataFrame, method_name: str) -> int:
        min_required = self.vol_window + 2 * self.block_window + self.smooth_window + 100
        if len(df) <= min_required:
            raise ValueError(
                f"Data too short for {method_name}. "
                f"Need > {min_required} rows, got {len(df)}."
            )
        return min_required

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'BlockVolStrategy':
        self._check_data_length(X, ".fit()")

        if 'close' not in X.columns:
            raise ValueError("X must contain 'close'")

        prc = X['close']
        log = np.log(prc)

        # === Block Momentum Signal ===
        h = prc.rolling(window=self.block_window).max()
        l = prc.rolling(window=self.block_window).min()
        rng = h - l

        hh = h.diff(self.block_window)   # How much higher is current high vs N periods ago
        ll = l.diff(self.block_window)   # How much higher is current low vs N periods ago

        # Normalized symmetric block momentum
        s_raw = (hh + ll) / 2 / rng.replace(0, np.nan)

        # Smooth
        s = s_raw.ewm(span=self.smooth_window).mean()

        # Burn-in
        burn_in = self.vol_window + 2 * self.block_window + self.smooth_window
        s = s.iloc[burn_in:]
        s = s.clip(-2, 2)

        # Volatility
        #v = log.diff().ewm(span=self.vol_window).std() * ANNUAL_BARS**0.5
        v = rogers_satchell_volatility(X,window=round(self.vol_window*.4)) * ANNUAL_BARS**0.5

        # Fit Weibull CDF
        self.cdf_median = v.median()
        logger.info(f"BlockVolStrategy | CDF median = {self.cdf_median:.6f}")

        # Vol tilt
        scale = self.cdf_median * 1.5
        v_cdf = pd.Series(
            weibull_min.cdf(v, c=self.weibull_c, scale=scale),
            index=v.index
        )
        vol_tilt = 1 - v_cdf.clip(0, 1)

        # Alpha-blended signal
        tilt_component = (1.0 - self.alpha) + self.alpha * vol_tilt
        f = s * tilt_component

        # Align
        idx = f.index.intersection(v.index)
        f = f.loc[idx]
        v = v.loc[idx]

        # Raw position & scaler calibration
        pos_raw = self.target_vol / v * f
        pnl = pos_raw * log.diff().shift(-1).loc[idx]
        pnl = pnl.dropna()

        if len(pnl) == 0:
            raise ValueError("No valid PnL")

        actual_vol = pnl.std() * ANNUAL_BARS**0.5
        self.to_target_scaler = self.target_vol / actual_vol if actual_vol > 1e-8 else 1.0

        logger.info(f"BlockVolStrategy fitted | "
                    f"block={self.block_window}h | "
                    f"smooth={self.smooth_window} | "
                    f"alpha={self.alpha:.2f} | "
                    f"scaler={self.to_target_scaler:.4f}")

        # Final stats
        final_pos = pos_raw * self.to_target_scaler * self.strat_weight
        final_pos = final_pos.reindex(prc.index)
        final_pnl = final_pos * log.diff().shift(-1)
        #self.calculate_pnl_stats(final_pnl.dropna(), final_pos, log.diff().shift(-1))

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        self._check_data_length(X, ".predict()")

        if not hasattr(self, 'cdf_median'):
            raise RuntimeError("Call .fit() first")

        prc = X['close']
        log = np.log(prc)

        h = prc.rolling(window=self.block_window).max()
        l = prc.rolling(window=self.block_window).min()
        rng = h - l

        hh = h.diff(self.block_window)
        ll = l.diff(self.block_window)

        s_raw = (hh + ll) / 2 / rng.replace(0, np.nan)
        s = s_raw.ewm(span=self.smooth_window).mean()
        s = s.clip(-2, 2)

        #v = log.diff().ewm(span=self.vol_window).std() * ANNUAL_BARS**0.5
        v = rogers_satchell_volatility(X,window=round(self.vol_window*.4)) * ANNUAL_BARS**0.5

        # Vol tilt
        scale = self.cdf_median * 1.5
        v_cdf = pd.Series(weibull_min.cdf(v, c=self.weibull_c, scale=scale), index=v.index)
        vol_tilt = 1 - v_cdf.clip(0, 1)

        # Blended signal
        tilt_component = (1.0 - self.alpha) + self.alpha * vol_tilt
        f = s * tilt_component

        # Position
        idx = f.index.intersection(v.index)
        pos = pd.Series(0.0, index=X.index)
        pos.loc[idx] = f.loc[idx] * self.target_vol / v.loc[idx]
        pos = pos * self.to_target_scaler * self.strat_weight

        return pos.reindex(X.index, fill_value=0.0)

    def signal(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate raw signal from latest data.
        """
        self._check_data_length(X, ".predict()")

        if not hasattr(self, 'cdf_median'):
            raise RuntimeError("Call .fit() first")

        prc = X['close']
        log = np.log(prc)

        h = prc.rolling(window=self.block_window).max()
        l = prc.rolling(window=self.block_window).min()
        rng = h - l

        hh = h.diff(self.block_window)
        ll = l.diff(self.block_window)

        s_raw = (hh + ll) / 2 / rng.replace(0, np.nan)
        s = s_raw.ewm(span=self.smooth_window).mean()
        s = s.clip(-2, 2)

        #v = log.diff().ewm(span=self.vol_window).std() * ANNUAL_BARS**0.5
        v = rogers_satchell_volatility(X,window=round(self.vol_window*.4)) * ANNUAL_BARS**0.5

        # Vol tilt
        scale = self.cdf_median * 1.5
        v_cdf = pd.Series(weibull_min.cdf(v, c=self.weibull_c, scale=scale), index=v.index)
        vol_tilt = 1 - v_cdf.clip(0, 1)

        # Blended signal
        tilt_component = (1.0 - self.alpha) + self.alpha * vol_tilt
        f = s * tilt_component * self.to_target_scaler * self.strat_weight
        f = f.reindex(prc.index)
        
        #logger.info(f"Latest position: {final_pos.iloc[-1]:+.2%}")
        return f
    
    def get_intermediates(self) -> Dict[str, Any]:
        return {
            'params': {
                'block_window': self.block_window,
                'smooth_window': self.smooth_window,
                'vol_window': self.vol_window,
                'weibull_c': self.weibull_c,
                'alpha': self.alpha,
                'target_vol': self.target_vol,
                'strat_weight': self.strat_weight,
                'cdf_median': self.cdf_median,
                'to_target_scaler': self.to_target_scaler,
            },
            'pnl_stats': self.pnl_stats,
        }
    
class SlopeVolStrategy(BaseStrategy):
    """
    Slope signal × Vol Tilt → Position
    """
    
    def __init__(
        self,
        regression_window: int = 24,   # 5-day slope
        vol_window: int = 24 * 30,
        weibull_c: float = 2.0,
        alpha: float = 1.0,              # Weight of vol_tilt (0 = no tilt, 1 = full tilt)
        target_vol: float = 0.5,
        strat_weight: float = 1.0
    ):
        # --- Parameters ---
        self.regression_window = regression_window
        self.vol_window = vol_window
        self.weibull_c = weibull_c
        self.alpha = np.clip(alpha, 0.0, 1.0)
        self.target_vol = target_vol
        self.strat_weight = np.clip(strat_weight, 0.0, 1.0)
        
        # --- Fit Parameters ---
        self.cdf_median: float = 0.0
        self.to_target_scaler: float = 1.0
        
        # Stats
        self.pnl_stats: Dict[str, float] = {}
        
        logger.info("SlopeVolStrategy initialized")

    def _check_data_length(self, price: pd.Series, method_name: str) -> int:
        min_required = max(self.regression_window, self.vol_window) + 1
        if len(price) <= min_required:
            raise ValueError(
                f"Data too short for {method_name}. "
                f"Need > {min_required} rows, got {len(price)}."
            )
        return min_required

    def fit(self, X: pd.Series, y: pd.Series = None) -> 'SlopeVolStrategy':
        """
        Fit CDF median and to-target scaler.
        """
        self._check_data_length(X, ".predict()")
        if not hasattr(self, 'cdf_median'):
            raise RuntimeError("Call .fit() first")

        if not all(col in X.columns for col in ['close']):
            raise ValueError("X must contain 'close'")

        prc = X['close']
        log = np.log(prc)

        # 1. Rolling slope
        slope,r2 = rolling_slope_vec(prc, self.regression_window)
        S = slope * r2
        S = S.iloc[self.regression_window:]

        # 2. Standardize
        s = S / S.ewm(span=self.vol_window).std()
        s = s.iloc[self.vol_window:]

        # 3. Annualized hourly vol
        #v = log.diff().ewm(span=self.vol_window).std() * ANNUAL_BARS**0.5
        v = rogers_satchell_volatility(X,window=round(self.vol_window*.4)) * ANNUAL_BARS**0.5
        v = v.iloc[self.vol_window:]

        # 4. Fit CDF median
        self.cdf_median = v.median()
        logger.info(f"Fit: CDF median = {self.cdf_median:.6f}")

        # 5. Vol tilt
        v_cdf = pd.Series(
            weibull_min.cdf(v, c=self.weibull_c, scale=self.cdf_median * 1.5),
            index=v.index
        )
        vol_tilt = 1 - v_cdf
        # 5. Combined Signal with alpha blending
        tilt_component = (1.0 - self.alpha) + self.alpha * vol_tilt

        # 6. Final signal
        f = s * tilt_component
        pos_raw = self.target_vol / v * f

        # 7. Simulate PnL
        pnl = pos_raw * log.diff().shift(-1)
        pnl = pnl.dropna()
        if len(pnl) == 0:
            raise ValueError("No valid PnL after alignment")

        # 8. Fit to-target scaler
        actual_vol = pnl.std() * ANNUAL_BARS**0.5
        self.to_target_scaler = self.target_vol / actual_vol if actual_vol > 0 else 1.0

        logger.info(f"Fit: to_target_scaler = {self.to_target_scaler:.6f}")

        # 9. Final position + stats
        final_pos = pos_raw * self.to_target_scaler * self.strat_weight
        final_pos = final_pos.reindex(prc.index)
        final_pnl = final_pos * log.diff().shift(-1)
        
        #self.calculate_pnl_stats(final_pnl, final_pos, log.diff().shift(-1))

        logger.success("SlopeVolStrategy fitted")
        return self

    def predict(self, X: pd.Series) -> pd.Series:
        """
        Generate position from latest price.
        """
        self._check_data_length(X, ".predict()")
        if not hasattr(self, 'cdf_median'):
            raise RuntimeError("Call .fit() first")

        if not all(col in X.columns for col in ['close']):
            raise ValueError("X must contain 'close'")

        prc = X['close']
        log = np.log(prc)

        # Rolling slope
        slope,r2 = rolling_slope_vec(prc, self.regression_window)
        S = slope * r2
        S = S.iloc[self.regression_window:]

        # Standardize
        s = S / S.ewm(span=self.vol_window).std()
        s = s.iloc[self.vol_window:]

        # Vol forecast
        #v = log.diff().ewm(span=self.vol_window).std() * ANNUAL_BARS**0.5
        v = rogers_satchell_volatility(X,window=round(self.vol_window*.4)) * ANNUAL_BARS**0.5
        v = v.iloc[self.vol_window:]

        # Vol tilt
        v_cdf = pd.Series(
            weibull_min.cdf(v, c=self.weibull_c, scale=self.cdf_median * 1.5),
            index=v.index
        )
        vol_tilt = 1 - v_cdf
        # 5. Combined Signal with alpha blending
        tilt_component = (1.0 - self.alpha) + self.alpha * vol_tilt

        # Final signal
        f = s * tilt_component
        pos_raw = self.target_vol / v * f

        # Final position
        final_pos = pos_raw * self.to_target_scaler * self.strat_weight
        final_pos = final_pos.reindex(prc.index)

        logger.info(f"Latest position: {final_pos.iloc[-1]:+.2%}")
        return final_pos

    def signal(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate raw signal from latest data.
        """
        self._check_data_length(X, ".predict()")
        if not hasattr(self, 'cdf_median'):
            raise RuntimeError("Call .fit() first")

        if not all(col in X.columns for col in ['close']):
            raise ValueError("X must contain 'close'")

        prc = X['close']
        log = np.log(prc)

        # Rolling slope
        slope,r2 = rolling_slope_vec(prc, self.regression_window)
        S = slope * r2
        S = S.iloc[self.regression_window:]

        # Standardize
        s = S / S.ewm(span=self.vol_window).std()
        s = s.iloc[self.vol_window:]

        # Vol forecast
        #v = log.diff().ewm(span=self.vol_window).std() * ANNUAL_BARS**0.5
        v = rogers_satchell_volatility(X,window=round(self.vol_window*.4)) * ANNUAL_BARS**0.5
        v = v.iloc[self.vol_window:]

        # Vol tilt
        v_cdf = pd.Series(
            weibull_min.cdf(v, c=self.weibull_c, scale=self.cdf_median * 1.5),
            index=v.index
        )
        vol_tilt = 1 - v_cdf
        # 5. Combined Signal with alpha blending
        tilt_component = (1.0 - self.alpha) + self.alpha * vol_tilt

        # Final signal
        f = s * tilt_component * self.to_target_scaler * self.strat_weight
        f = f.reindex(prc.index)
        
        #logger.info(f"Latest position: {final_pos.iloc[-1]:+.2%}")
        return f

    def get_intermediates(self) -> Dict[str, Any]:
        return {
            'params': {
                'regression_window': self.regression_window,
                'vol_window': self.vol_window,
                'weibull_c': self.weibull_c,
                'alpha': self.alpha,
                'target_vol': self.target_vol,
                'strat_weight': self.strat_weight,
                'cdf_median': self.cdf_median,
                'to_target_scaler': self.to_target_scaler
            },
            'pnl_stats': self.pnl_stats
        }
    
class BuySellVolStrategy(BaseStrategy):
    """
    Buy/Sell Ratio Strategy with optional Volatility Tilt
    
    Core signal:
        bsr_raw      = EMA(taker_buy_base_vol) / EMA(total_volume)
        bsr_z        = z-score of bsr_raw
        vol_tilt     = 1 - Weibull_CDF(vol)   → lower during high-vol regimes
        final_signal = bsr_z * ((1 - alpha) + alpha * vol_tilt)
        pos          = final_signal * 0.5 / annualized_vol * scaler * weight
    
    alpha = 0.0 → pure BSR (no tilt)
    alpha = 1.0 → full volatility tilt (same as original EmaVolStrategy style)
    """
    
    def __init__(
        self,
        volume_window: int = 24,
        smooth_window: int = 12,         # NEW: smooth signal
        vol_window: int = 24 * 30,
        weibull_c: float = 2.0,          # Shape parameter for Weibull vol CDF
        alpha: float = 1.0,              # Weight of vol_tilt (0 = no tilt, 1 = full tilt)
        target_vol: float = 0.5,
        strat_weight: float = 1.0
    ):
        # --- Strategy Parameters ---
        self.volume_window = volume_window
        self.smooth_window = smooth_window
        self.vol_window = vol_window
        self.weibull_c = weibull_c
        self.alpha = np.clip(alpha, 0.0, 1.0)
        self.target_vol = target_vol
        self.strat_weight = np.clip(strat_weight, 0.0, 1.0)
        
        # --- Fit Parameters ---
        self.cdf_median: float = 0.0
        self.to_target_scaler: float = 1.0

        # Stats
        self.pnl_stats: Dict[str, float] = {}
        
        logger.info(f"BuySellVolStrategy initialized | alpha={self.alpha:.2f}")

    def _check_data_length(self, df: pd.DataFrame, method_name: str) -> None:
        min_required = self.vol_window + self.volume_window + 200
        if len(df) <= min_required:
            raise ValueError(
                f"Data too short for {method_name}. "
                f"Need > {min_required} rows, got {len(df)}."
            )

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'BuySellVolStrategy':
        self._check_data_length(X, ".fit()")

        required = ['close', 'taker_buy_base_vol', 'volume']
        if not all(col in X.columns for col in required):
            missing = [c for c in required if c not in X.columns]
            raise ValueError(f"Missing columns: {missing}")

        close = X['close']
        log_ret = np.log(close).diff()

        # 1. Buy/Sell Ratio + Z-score
        buy_ema = X['taker_buy_base_vol'].ewm(span=self.volume_window).mean()
        vol_ema = X['volume'].ewm(span=self.volume_window).mean()
        bsr_raw = (buy_ema / vol_ema.replace(0, np.nan)).ewm(span=self.smooth_window).mean()
        bsr_raw = bsr_raw.iloc[self.smooth_window:]

        bsr_mean = bsr_raw.ewm(span=self.vol_window).mean()
        bsr_std  = bsr_raw.ewm(span=self.vol_window).std()
        bsr_z = ((bsr_raw - bsr_mean) / bsr_std).clip(-2,2)

        # 2. Volatility
        #v = log_ret.ewm(span=self.vol_window).std() * ANNUAL_BARS**0.5
        v = rogers_satchell_volatility(X,window=round(self.vol_window*.4)) * ANNUAL_BARS**0.5

        # 3. Fit Weibull CDF median (same method as original EmaVolStrategy)
        self.cdf_median = v.median()
        logger.info(f"Fit: Weibull CDF median = {self.cdf_median:.6f}")

        # 4. Volatility Tilt
        scale = self.cdf_median * 1.5
        v_cdf = pd.Series(
            weibull_min.cdf(v, c=self.weibull_c, scale=scale),
            index=v.index
        )
        vol_tilt = 1.0 - v_cdf.clip(0, 1)

        # 5. Combined Signal with alpha blending
        tilt_component = (1.0 - self.alpha) + self.alpha * vol_tilt
        signal_f = bsr_z * tilt_component

        # Burn-in
        burn = self.vol_window + self.volume_window
        signal_f = signal_f.iloc[burn:]
        v = v.iloc[burn:]

        # Align
        idx = signal_f.index.intersection(v.index)
        signal_f = signal_f.loc[idx]
        v = v.loc[idx]

        # 6. Raw position & scaler calibration
        pos_raw = signal_f * self.target_vol / v
        pnl = pos_raw * log_ret.shift(-1).loc[idx]

        actual_vol = pnl.std() * ANNUAL_BARS**0.5
        self.to_target_scaler = self.target_vol / actual_vol if actual_vol > 1e-8 else 1.0

        logger.info(f"BuySellVolStrategy fitted | "
                    f"alpha={self.alpha:.2f} | "
                    f"to_target_scaler={self.to_target_scaler:.4f} | "
                    f"target_vol={self.target_vol:.1%}")

        # Final backtest for stats
        final_pos = pos_raw * self.to_target_scaler * self.strat_weight
        final_pos = final_pos.reindex(close.index)
        final_pnl = final_pos * log_ret.shift(-1)
        #self.calculate_pnl_stats(final_pnl.dropna(), final_pos, log_ret.shift(-1))

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        self._check_data_length(X, ".predict()")

        if not hasattr(self, 'to_target_scaler'):
            raise RuntimeError("Call .fit() first.")

        close = X['close']
        log_ret = np.log(close).diff()

        # Recompute full history
        buy_ema = X['taker_buy_base_vol'].ewm(span=self.volume_window).mean()
        vol_ema = X['volume'].ewm(span=self.volume_window).mean()
        bsr_raw = (buy_ema / vol_ema.replace(0, np.nan)).ewm(span=self.smooth_window).mean()
        bsr_raw = bsr_raw.iloc[self.smooth_window:]

        bsr_mean = bsr_raw.ewm(span=self.vol_window).mean()
        bsr_std  = bsr_raw.ewm(span=self.vol_window).std()
        bsr_z = ((bsr_raw - bsr_mean) / bsr_std).clip(-2,2)

        #v = log_ret.ewm(span=self.vol_window).std() * ANNUAL_BARS**0.5
        v = rogers_satchell_volatility(X,window=round(self.vol_window*.4)) * ANNUAL_BARS**0.5

        # Vol tilt
        scale = self.cdf_median * 1.5
        v_cdf = pd.Series(weibull_min.cdf(v, c=self.weibull_c, scale=scale), index=v.index)
        vol_tilt = 1.0 - v_cdf.clip(0, 1)

        # Blended signal
        tilt_component = (1.0 - self.alpha) + self.alpha * vol_tilt
        signal_f = bsr_z * tilt_component

        # Position
        aligned_idx = signal_f.index.intersection(v.index)
        pos = pd.Series(0.0, index=X.index)
        pos.loc[aligned_idx] = signal_f.loc[aligned_idx] * self.target_vol / v.loc[aligned_idx]
        pos = pos * self.to_target_scaler * self.strat_weight

        return pos.reindex(X.index, fill_value=0.0)

    def signal(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate raw signal from latest data.
        """
        self._check_data_length(X, ".predict()")

        if not hasattr(self, 'to_target_scaler'):
            raise RuntimeError("Call .fit() first.")

        close = X['close']
        log_ret = np.log(close).diff()

        # Recompute full history
        buy_ema = X['taker_buy_base_vol'].ewm(span=self.volume_window).mean()
        vol_ema = X['volume'].ewm(span=self.volume_window).mean()
        bsr_raw = (buy_ema / vol_ema.replace(0, np.nan)).ewm(span=self.smooth_window).mean()
        bsr_raw = bsr_raw.iloc[self.smooth_window:]

        bsr_mean = bsr_raw.ewm(span=self.vol_window).mean()
        bsr_std  = bsr_raw.ewm(span=self.vol_window).std()
        bsr_z = ((bsr_raw - bsr_mean) / bsr_std).clip(-2,2)

        #v = log_ret.ewm(span=self.vol_window).std() * ANNUAL_BARS**0.5
        v = rogers_satchell_volatility(X,window=round(self.vol_window*.4)) * ANNUAL_BARS**0.5

        # Vol tilt
        scale = self.cdf_median * 1.5
        v_cdf = pd.Series(weibull_min.cdf(v, c=self.weibull_c, scale=scale), index=v.index)
        vol_tilt = 1.0 - v_cdf.clip(0, 1)

        # Blended signal
        tilt_component = (1.0 - self.alpha) + self.alpha * vol_tilt
        signal_f = bsr_z * tilt_component * self.to_target_scaler * self.strat_weight
        f = signal_f
        f = f.reindex(close.index)
        
        #logger.info(f"Latest position: {final_pos.iloc[-1]:+.2%}")
        return f

    def get_intermediates(self) -> Dict[str, Any]:
        return {
            'params': {
                'volume_window': self.volume_window,
                'vol_window': self.vol_window,
                'smooth_window': self.smooth_window,
                'weibull_c': self.weibull_c,
                'alpha': self.alpha,
                'target_vol': self.target_vol,
                'strat_weight': self.strat_weight,
                'cdf_median': self.cdf_median,
                'to_target_scaler': self.to_target_scaler,
            },
            'pnl_stats': self.pnl_stats,
        }
    

class RevStrategy(BaseStrategy):
    """
    Short-term reversal with volume filter
    rev = -sign(ret_z) only when |ret_z| > threshold AND volume regime is high
    pos = rev * 0.5 / vol   → scaled to target volatility
    """

    def __init__(
        self,
        vol_window: int = 24 * 30,
        reversal_window: int = 6,
        reversal_threshold: float = 2.0,
        volume_threshold: float = 0.8,
        target_vol: float = 0.5,
        strat_weight: float = 1.0
    ):
        self.vol_window = int(vol_window)
        self.reversal_window = int(reversal_window)
        self.reversal_threshold = float(reversal_threshold)
        self.volume_threshold = float(volume_threshold)
        self.target_vol = float(target_vol)
        self.strat_weight = np.clip(float(strat_weight), 0.0, 1.0)

        self.to_target_scaler: float = 1.0
        self.pnl_stats: Dict[str, float] = {}

        logger.info("RevStrategy initialized")

    def _check_data_length(self, price: pd.Series, method_name: str) -> None:
        if len(price) <= self.vol_window + 100:
            raise ValueError(f"Data too short for {method_name}")

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'RevStrategy':
        self._check_data_length(X['close'], ".fit()")
        if not {'close', 'volume'}.issubset(X.columns):
            raise ValueError("X must have 'close' and 'volume'")

        prc = X['close']
        vlm = X['volume']

        log_ret = np.log(prc).diff()

        # volatility
        #v = log_ret.ewm(span=self.vol_window).std()
        v = rogers_satchell_volatility(X,window=round(self.vol_window*.4)) #* ANNUAL_BARS**0.5

        # volume regime
        vlm_z = vlm.ewm(span=self.reversal_window).mean()
        vlm_z = (vlm_z - vlm_z.rolling(self.vol_window).min()) / (
                vlm_z.rolling(self.vol_window).max() - vlm_z.rolling(self.vol_window).min() + 1e-8)

        # momentum z-score
        ret_z = log_ret.ewm(span=self.reversal_window).mean() / v.shift(self.reversal_window)

        # reversal signal
        rev = -np.sign(ret_z).mask(ret_z.abs() <= self.reversal_threshold, 0)
        rev = rev.mask(vlm_z < self.volume_threshold, 0)

        # raw position
        pos_raw = rev * self.target_vol / (v.shift(self.reversal_window) * ANNUAL_BARS**0.5)

        # to-target scaling from in-sample PnL
        pnl = pos_raw * log_ret.shift(-1)
        pnl = pnl.dropna()
        if len(pnl) == 0:
            raise ValueError("No PnL after alignment")

        actual_vol = pnl.std() * ANNUAL_BARS**0.5
        self.to_target_scaler = self.target_vol / actual_vol if actual_vol > 0 else 1.0

        logger.info(f"RevStrategy fit → scaler = {self.to_target_scaler:.4f}")

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        self._check_data_length(X['close'], ".predict()")
        if not hasattr(self, 'to_target_scaler'):
            raise RuntimeError("Call .fit() first")

        prc = X['close']
        vlm = X['volume']

        log_ret = np.log(prc).diff()

        #v = log_ret.ewm(span=self.vol_window).std()
        v = rogers_satchell_volatility(X,window=round(self.vol_window*.4)) #* ANNUAL_BARS**0.5
        vlm_z = vlm.ewm(span=self.reversal_window).mean()
        vlm_z = (vlm_z - vlm_z.rolling(self.vol_window).min()) / (
                vlm_z.rolling(self.vol_window).max() - vlm_z.rolling(self.vol_window).min() + 1e-8)

        # momentum z-score
        ret_z = log_ret.ewm(span=self.reversal_window).mean() / v.shift(self.reversal_window)

        rev = -np.sign(ret_z).mask(ret_z.abs() <= self.reversal_threshold, 0)
        rev = rev.mask(vlm_z < self.volume_threshold, 0)

        pos_raw = rev * self.target_vol / (v.shift(self.reversal_window) * ANNUAL_BARS**0.5)
        pos = pos_raw * self.to_target_scaler * self.strat_weight

        return pos.reindex(X.index).fillna(0)

    def signal(self, X: pd.DataFrame) -> pd.Series:
        self._check_data_length(X['close'], ".signal()")

        prc = X['close']
        vlm = X['volume']

        log_ret = np.log(prc).diff()
        #v = log_ret.ewm(span=self.vol_window).std()
        v = rogers_satchell_volatility(X,window=round(self.vol_window*.4)) #* ANNUAL_BARS**0.5

        vlm_z = vlm.ewm(span=self.reversal_window).mean()
        vlm_z = (vlm_z - vlm_z.rolling(self.vol_window).min()) / (
                vlm_z.rolling(self.vol_window).max() - vlm_z.rolling(self.vol_window).min() + 1e-8)

        # momentum z-score
        ret_z = log_ret.ewm(span=self.reversal_window).mean() / v.shift(self.reversal_window)

        rev = -np.sign(ret_z).mask(ret_z.abs() <= self.reversal_threshold, 0)
        rev = rev.mask(vlm_z < self.volume_threshold, 0)
        f = rev.reindex(X.index).fillna(0) * self.to_target_scaler * self.strat_weight

        return f

    def get_intermediates(self) -> Dict[str, Any]:
        return {
            'params': {
                'vol_window': self.vol_window,
                'reversal_window': self.reversal_window,
                'reversal_threshold': self.reversal_threshold,
                'volume_threshold': self.volume_threshold,
                'target_vol': self.target_vol,
                'strat_weight': self.strat_weight,
                'to_target_scaler': self.to_target_scaler,
            },
            'pnl_stats': self.pnl_stats
        }