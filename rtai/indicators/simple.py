"""
rtai.indicators.simple
======================

Self-contained "basic" RSI-style oscillators plus a DebounceManager.

• BaseOscillator – EWMA-variance, regression-band, z-score normalisation
• WallRatioOsc   – bid/ask wall dominance            (order-book depth)
• TradeImbalanceOsc – buy/sell trade-flow imbalance  (agg. trades)
• LiquidationOsc – liquidation magnitude / count     (exchange feed)
• DIROsc         – Depth-Imbalance-Ratio             (order-book depth)
• FundingAccelOsc – acceleration of funding rate     (funding feed)
• DebounceManager – generic cooldown helper

All oscillators expose:

    update_*() → Optional[float]        # returns bounded osc ∈ (-1, +1)
    is_extreme(z_threshold=2.0) → {HIGH|LOW|None}
    get_state() → dict                  # serialisable snapshot
"""

from __future__ import annotations

import math
import statistics
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
#                               Base oscillator                               #
# --------------------------------------------------------------------------- #
class BaseOscillator:
    """Trend-adjusted, EWMA-variance, percentile-aware oscillator."""

    def __init__(
        self,
        *,
        window: int = 60,
        min_points: int | None = None,
        ewma_alpha: float = 0.10,
    ) -> None:
        self.window: int = window
        self.min_points: int = min_points or max(10, window // 2)
        self.ewma_alpha: float = ewma_alpha

        self.values: Deque[Tuple[float, float]] = deque(maxlen=window)
        self.ewma_var: Optional[float] = None

        # runtime state
        self.trend_slope: float = 0.0
        self.z_score: float = 0.0
        self.osc: float = 0.0
        self.last_update: float = 0.0
        self.threshold_high: Optional[float] = None
        self.threshold_low: Optional[float] = None

    # --------------------------------------------------------------------- #
    #                          common update routine                        #
    # --------------------------------------------------------------------- #
    def update_raw(self, ts: float, raw_value: float) -> float | None:
        """Internal: push raw value, update state, return bounded osc or None."""
        self.values.append((ts, raw_value))

        if len(self.values) < self.min_points:
            return None

        # --- regression-band de-trending --------------------------------- #
        x_vals = [t for t, _ in self.values]
        y_vals = [v for _, v in self.values]
        n = len(x_vals)
        mean_x = sum(x_vals) / n
        mean_y = sum(y_vals) / n
        cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, y_vals))
        var_x = sum((x - mean_x) ** 2 for x in x_vals) or 1.0
        self.trend_slope = cov / var_x
        intercept = mean_y - self.trend_slope * mean_x
        current_resid = raw_value - (self.trend_slope * ts + intercept)

        # --- EWMA variance ------------------------------------------------ #
        resid_var = current_resid**2
        if self.ewma_var is None:
            self.ewma_var = resid_var
        else:
            self.ewma_var = (
                (1 - self.ewma_alpha) * self.ewma_var + self.ewma_alpha * resid_var
            )

        std = math.sqrt(self.ewma_var) if self.ewma_var > 1e-12 else 1.0
        self.z_score = current_resid / std

        # compress to (-1, +1) for RSI-like behaviour
        self.osc = math.tanh(self.z_score / 2.0)
        self.last_update = ts

        # adaptive percentile thresholds
        if len(self.values) >= self.window:
            sorted_vals = sorted(v for _, v in self.values)
            hi_idx = int(len(sorted_vals) * 0.975) - 1
            lo_idx = int(len(sorted_vals) * 0.025)
            self.threshold_high = sorted_vals[hi_idx]
            self.threshold_low = sorted_vals[lo_idx]

        return self.osc

    # --------------------------------------------------------------------- #
    #                          convenience helpers                          #
    # --------------------------------------------------------------------- #
    def is_extreme(self, z_threshold: float = 2.0) -> Optional[str]:
        if abs(self.z_score) >= z_threshold:
            return "HIGH" if self.z_score > 0 else "LOW"
        return None

    def get_state(self) -> Dict[str, Any]:
        return {
            "osc": self.osc,
            "z_score": self.z_score,
            "trend_slope": self.trend_slope,
            "last_update": self.last_update,
            "data_points": len(self.values),
        }


# --------------------------------------------------------------------------- #
#                         Specialised basic oscillators                       #
# --------------------------------------------------------------------------- #
class WallRatioOsc(BaseOscillator):
    """Bid-vs-ask wall dominance (depth imbalance)."""

    def __init__(
        self, *, window: int = 100, price_levels: int = 3, size_cutoff: float = 100.0
    ) -> None:
        super().__init__(window=window)
        self.price_levels = price_levels
        self.size_cutoff = size_cutoff

    def update_depth(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        timestamp: float,
    ) -> Optional[float]:
        if not bids or not asks:
            return None
        bid_wall = sum(
            s for _, s in bids[: self.price_levels] if s >= self.size_cutoff
        )
        ask_wall = sum(
            s for _, s in asks[: self.price_levels] if s >= self.size_cutoff
        )
        total = bid_wall + ask_wall
        if total == 0:
            return None
        raw = (bid_wall - ask_wall) / total
        return self.update_raw(timestamp, raw)


class TradeImbalanceOsc(BaseOscillator):
    """Imbalance of buy vs sell aggressive volume."""

    def __init__(self, *, window: int = 60, bucket_seconds: int = 1) -> None:
        super().__init__(window=window)
        self.bucket_seconds = bucket_seconds
        self._cur_bucket: Optional[int] = None
        self._buy: float = 0.0
        self._sell: float = 0.0

    def update_trade(
        self, qty: float, is_buyer: bool, timestamp: float
    ) -> Optional[float]:
        bucket = int(timestamp // self.bucket_seconds)
        if self._cur_bucket is None:
            self._cur_bucket = bucket

        if bucket == self._cur_bucket:
            if is_buyer:
                self._buy += qty
            else:
                self._sell += qty
            return None

        # finish old bucket
        total = self._buy + self._sell
        res = None
        if total > 0:
            raw = (self._buy - self._sell) / total
            res = self.update_raw(self._cur_bucket * self.bucket_seconds, raw)

        # reset
        self._cur_bucket = bucket
        self._buy = qty if is_buyer else 0.0
        self._sell = qty if not is_buyer else 0.0
        return res


class LiquidationOsc(BaseOscillator):
    """Magnitude of liquidations (USD) and count combined."""

    def update_liquidation(self, usd_amt: float, timestamp: float) -> Optional[float]:
        # In your feeds you can pre-normalise usd_amt or keep as raw.
        return self.update_raw(timestamp, usd_amt)


class DIROsc(BaseOscillator):
    """Depth-Imbalance-Ratio (DIR) - log normalised."""

    def update_depth(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        timestamp: float,
    ) -> Optional[float]:
        bid_vol = sum(s for _, s in bids)
        ask_vol = sum(s for _, s in asks)
        if bid_vol == 0 or ask_vol == 0:
            return None
        raw = math.log((bid_vol + 1e-9) / (ask_vol + 1e-9))
        return self.update_raw(timestamp, raw)


class FundingAccelOsc(BaseOscillator):
    """Second-derivative (acceleration) of funding rate."""

    def __init__(self, *, window: int = 50) -> None:
        super().__init__(window=window)
        self._prev_rate: Optional[float] = None
        self._prev_ts: Optional[float] = None

    def update_funding(self, rate: float, timestamp: float) -> Optional[float]:
        if self._prev_rate is None:
            self._prev_rate, self._prev_ts = rate, timestamp
            return None
        # first derivative
        d1 = (rate - self._prev_rate) / (timestamp - self._prev_ts + 1e-6)
        self._prev_rate, self._prev_ts = rate, timestamp
        return self.update_raw(timestamp, d1)


# --------------------------------------------------------------------------- #
#                               Debounce helper                               #
# --------------------------------------------------------------------------- #
class DebounceManager:
    """Global cooldown storage to prevent double-fire of extreme signals."""

    def __init__(self) -> None:
        self._last: Dict[str, float] = {}

    def should_fire(self, key: str, cooldown: float) -> bool:
        now = time.time()
        if key not in self._last or now - self._last[key] >= cooldown:
            self._last[key] = now
            return True
        return False


# --------------------------------------------------------------------------- #
#                               public exports                                #
# --------------------------------------------------------------------------- #
__all__ = [
    "BaseOscillator",
    "WallRatioOsc",
    "TradeImbalanceOsc",
    "LiquidationOsc",
    "DIROsc",
    "FundingAccelOsc",
    "DebounceManager",
]