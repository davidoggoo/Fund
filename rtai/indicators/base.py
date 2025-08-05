"""Consolidated indicator classes for RTAI system

This module contains all the core trading indicators:
- OFI: Order Flow Imbalance with z-score normalization  
- VPIN: Volume-Synchronized Probability of Informed Trading
- KyleLambda: Market impact coefficient
- LPI: Liquidation Pressure Index

Enhanced with health monitoring and structured logging.
"""

from __future__ import annotations

import statistics
import time
from collections import deque
from typing import Deque

import numpy as np
from loguru import logger
from ..state import StateAdapter, create_state_store

# Health monitoring imports
try:
    from ..utils.health import record_indicator_calculation, record_error
    from ..utils.structured_logging import log_indicator_calculation
    HEALTH_MONITORING_AVAILABLE = True
except ImportError:
    HEALTH_MONITORING_AVAILABLE = False
    def record_indicator_calculation(*args, **kwargs):
        pass
    def record_error(*args, **kwargs):
        pass
    def log_indicator_calculation(*_a, **_kw):
        return None
    
# Temporarily disable structured logging due to logger bug
HEALTH_MONITORING_AVAILABLE = False
def log_indicator_calculation(*_a, **_kw):
    return None


class OFI:
    """Order Flow Imbalance indicator with EWMA-based z-score normalization for 1-minute edge"""

    def __init__(self, alpha: float = 0.15, mad_alpha: float = 0.3):  # P2-1 FIX: Optimized for 1-minute edge
        # EWMA parameters for mean and MAD (robust deviation)
        self.alpha = alpha  # Increased to 0.15 for better 1-minute reactivity
        self.mad_alpha = mad_alpha  # Increased to 0.3 for reduced noise from small prints
        
        # EWMA state variables
        self.ewma_mean: float | None = None
        self.ewma_mad: float | None = None  # Mean Absolute Deviation
        
        # Volatility filter for quality signals
        self.recent_values: Deque[float] = deque(maxlen=20)
        self.value: float | None = None
        self.last_update: float = 0.0
        
        # Micro-print filtering for 1-minute edge
        self.micro_print_threshold = 0.001  # Filter prints < 0.1% of typical size

    def update_oi_estimate(self, oi_value: float):
        """Update Open Interest estimate from external feed"""
        if oi_value > 0:
            self.total_oi_estimate = oi_value
            self.oi_last_update = time.time()
    
    def update(self, trade_qty: float, current_price: float) -> float | None:
        """Update OFI with signed quantity using EWMA z-score with comprehensive validation"""
        
        start_time = time.perf_counter() if HEALTH_MONITORING_AVAILABLE else None
        
        try:
            # Enhanced input validation
            qty_signed = float(trade_qty)
            price = float(current_price)
            
            if not np.isfinite(qty_signed) or not np.isfinite(price):
                if HEALTH_MONITORING_AVAILABLE:
                    record_error("ofi_invalid_input")
                logger.warning(f"OFI: Non-finite inputs: qty={qty_signed}, price={price}")
                return self.value  # Return current value without update
                
            if price <= 0:
                if HEALTH_MONITORING_AVAILABLE:
                    record_error("ofi_invalid_price")
                logger.warning(f"OFI: Invalid price: {price}")
                return self.value
            
            # Filter micro-prints for better signal quality with enhanced validation
            if abs(qty_signed) < self.micro_print_threshold:
                return self.value  # Return current value without update
            
            # Initialize EWMA on first update with validation
            if self.ewma_mean is None:
                self.ewma_mean = float(qty_signed)
                self.ewma_mad = float(abs(qty_signed))
                self.recent_values.append(float(qty_signed))
                return None
            
            # Validate current EWMA state
            if not np.isfinite(self.ewma_mean) or not np.isfinite(self.ewma_mad):
                if HEALTH_MONITORING_AVAILABLE:
                    record_error("ofi_corrupted_state")
                logger.warning(f"OFI: Corrupted EWMA state: mean={self.ewma_mean}, mad={self.ewma_mad}")
                # Reset EWMA
                self.ewma_mean = float(qty_signed)
                self.ewma_mad = float(abs(qty_signed))
                return None
            
            # Update EWMA mean with overflow protection
            try:
                new_mean = self.alpha * qty_signed + (1 - self.alpha) * self.ewma_mean
                if not np.isfinite(new_mean):
                    if HEALTH_MONITORING_AVAILABLE:
                        record_error("ofi_ewma_overflow")
                    logger.warning(f"OFI: EWMA mean overflow, resetting")
                    self.ewma_mean = float(qty_signed)
                    return None
                self.ewma_mean = new_mean
                
            except (OverflowError, FloatingPointError) as e:
                if HEALTH_MONITORING_AVAILABLE:
                    record_error("ofi_ewma_error") 
                logger.warning(f"OFI: EWMA mean calculation error: {e}")
                return self.value
            
            # Update EWMA MAD (robust deviation measure) with validation
            try:
                abs_deviation = abs(qty_signed - self.ewma_mean)
                if not np.isfinite(abs_deviation):
                    logger.warning(f"OFI: Invalid deviation: {abs_deviation}")
                    return self.value
                    
                new_mad = self.mad_alpha * abs_deviation + (1 - self.mad_alpha) * self.ewma_mad
                if not np.isfinite(new_mad) or new_mad < 0:
                    logger.warning(f"OFI: EWMA MAD calculation error: {new_mad}")
                    return self.value
                self.ewma_mad = new_mad
                
            except (OverflowError, FloatingPointError) as e:
                logger.warning(f"OFI: EWMA MAD calculation error: {e}")
                return self.value
            
            # Calculate z-score using robust MAD scaling with comprehensive validation
            if self.ewma_mad > 1e-8:  # Avoid division by zero
                try:
                    # MAD scaling factor (1.4826 for normal distribution)
                    mad_scale = self.ewma_mad * 1.4826
                    if mad_scale <= 0 or not np.isfinite(mad_scale):
                        logger.warning(f"OFI: Invalid MAD scale: {mad_scale}")
                        return self.value
                        
                    z_score = (qty_signed - self.ewma_mean) / mad_scale
                    
                    if not np.isfinite(z_score):
                        logger.warning(f"OFI: Non-finite z-score: {z_score}")
                        return self.value
                    
                    # Sanity check for extreme z-scores
                    if abs(z_score) > 50.0:  # Extreme outlier
                        logger.warning(f"OFI: Extreme z-score: {z_score:.2f}, capping")
                        z_score = np.clip(z_score, -50.0, 50.0)
                    
                except (ZeroDivisionError, OverflowError, FloatingPointError) as e:
                    logger.warning(f"OFI: Z-score calculation error: {e}")
                    return self.value
                
                # Store for volatility filtering with validation
                try:
                    abs_qty = abs(qty_signed)
                    if np.isfinite(abs_qty) and abs_qty >= 0:
                        self.recent_values.append(abs_qty)
                except Exception as e:
                    logger.warning(f"OFI: Error storing recent value: {e}")
                
                # Apply volatility filter - only signal when recent activity > median
                if len(self.recent_values) >= 10:
                    try:
                        recent_list = list(self.recent_values)
                        valid_values = [v for v in recent_list if np.isfinite(v) and v >= 0]
                        
                        if len(valid_values) >= 5:
                            recent_median = float(statistics.median(valid_values))
                            current_volume = abs(qty_signed)
                            
                            if not np.isfinite(recent_median) or recent_median < 0:
                                logger.warning(f"OFI: Invalid median: {recent_median}")
                                return self.value
                            
                            # Enhanced signal when volume > p50 and significant z-score
                            if current_volume > recent_median and abs(z_score) > 1.6:  # Strong signal
                                self.value = float(z_score)
                                self.last_update = time.time()
                                
                                # Log significant signals
                                if abs(z_score) > 3.0:
                                    logger.info(f"🚨 OFI: Strong signal: {z_score:.2f}σ "
                                              f"(qty={qty_signed:.2f}, vol={current_volume:.2f})")
                                
                                # Record performance metrics
                                if HEALTH_MONITORING_AVAILABLE and start_time:
                                    duration_ms = (time.perf_counter() - start_time) * 1000
                                    record_indicator_calculation()
                                    log_indicator_calculation("OFI", "BTCUSDT", self.value, duration_ms,
                                                            signal_strength="strong", z_score=z_score)
                                
                                return self.value
                                
                            # Still return z-score for monitoring, but flag as weak
                            elif abs(z_score) >= 1.0:
                                self.value = float(z_score * 0.8)  # Dampened weak signals
                                self.last_update = time.time()
                                return self.value
                                
                    except (ValueError, TypeError, statistics.StatisticsError) as e:
                        logger.warning(f"OFI: Volatility filter error: {e}")
                        # Return z-score without filter
                        if abs(z_score) > 1.6:
                            self.value = float(z_score)
                            self.last_update = time.time()
                            return self.value
                else:
                    # For initial samples (< 10), be less restrictive for testing
                    if abs(z_score) >= 1.0:  # Lower threshold for early samples
                        self.value = float(z_score)
                        self.last_update = time.time()
                        
                        if HEALTH_MONITORING_AVAILABLE and start_time:
                            duration_ms = (time.perf_counter() - start_time) * 1000
                            record_indicator_calculation()
                            log_indicator_calculation("OFI", "BTCUSDT", self.value, duration_ms)
                        
                        return self.value
                        
            return None
            
        except Exception as e:
            if HEALTH_MONITORING_AVAILABLE:
                record_error("ofi_unexpected_error")
            logger.error(f"OFI: Unexpected error: {e}")
            return self.value  # Return current value as fallback
        
        finally:
            # Record performance metrics for all updates
            if HEALTH_MONITORING_AVAILABLE and start_time:
                duration_ms = (time.perf_counter() - start_time) * 1000
                record_indicator_calculation()
                if self.value is not None:
                    log_indicator_calculation("OFI", "BTCUSDT", self.value, duration_ms)
    
    def get_value(self) -> float | None:
        """Get current OFI z-score value"""
        return self.value


class VPIN:
    """Volume-Synchronized Probability of Informed Trading with dynamic ATR buckets"""

    def __init__(self, base_bucket_usdt: float = 50.0, win_buckets: int = 12):  # P2-2 FIX: Optimized for minute-scale
        self.base_bucket = base_bucket_usdt  # Increased to 50 USDT for better signal quality
        self.win = deque(maxlen=win_buckets)  # Reduced to 12 for 3-5min flow focus
        self.buy = self.sell = 0.0
        self.value: float | None = None
        self.last_price = 0.0  # Initialize with safe default, updated from first trade
        self.last_update: float = 0.0
        
        # ATR calculation for dynamic buckets
        self.price_history: Deque[float] = deque(maxlen=900)  # 15min at 1-second resolution
        self.atr_15m: float | None = None
        self.dynamic_threshold: float = 0.8  # Will be updated from 24h p95
        
        # Threshold tracking for dynamic updates (P2-2 FIX: 6h instead of 24h)
        self.vpin_24h: Deque[float] = deque(maxlen=21600)  # 6h history for current regime sensitivity
        self.last_threshold_update: float = 0.0

    def _update_atr(self, price: float) -> None:
        """Update 15-minute ATR for dynamic bucket sizing"""
        self.price_history.append(price)
        
        if len(self.price_history) >= 60:  # Need sufficient price history
            prices = np.array(self.price_history)
            # Calculate true range approximation using price differences
            price_diffs = np.abs(np.diff(prices))
            
            if len(price_diffs) > 0:
                # ATR as average of recent price movements
                self.atr_15m = np.mean(price_diffs[-900:])  # 15min window

    def _update_dynamic_threshold(self, current_time: float) -> None:
        """Update dynamic threshold hourly based on 24h p98 - Enhanced with comprehensive validation"""
        try:
            current_time_safe = float(current_time)
            last_update_safe = float(self.last_threshold_update)
            
            # Validate timestamps
            if not np.isfinite(current_time_safe) or not np.isfinite(last_update_safe):
                logger.warning("VPIN: Invalid timestamps for threshold update")
                return
            
            # Check if update is needed (hourly)
            time_since_update = current_time_safe - last_update_safe
            if time_since_update < 3600:  # Less than 1 hour
                return
                
            # Validate we have sufficient history
            if len(self.vpin_24h) < 100:
                logger.debug(f"VPIN: Insufficient history for threshold update: {len(self.vpin_24h)} samples")
                return
            
            # Clean and validate VPIN history data
            valid_vpin_values = []
            for vpin_val in self.vpin_24h:
                try:
                    val = float(vpin_val)
                    if np.isfinite(val) and 0.0 <= val <= 1.0:  # VPIN should be [0,1]
                        valid_vpin_values.append(val)
                except (ValueError, TypeError):
                    continue
            
            if len(valid_vpin_values) < 50:  # Need minimum valid samples
                logger.warning(f"VPIN: Too few valid samples for threshold: {len(valid_vpin_values)}")
                return
            
            # Calculate p98 from valid VPIN values
            vpin_array = np.array(valid_vpin_values)
            new_threshold = float(np.percentile(vpin_array, 98))
            
            # Validate new threshold is reasonable
            if not np.isfinite(new_threshold):
                logger.warning(f"VPIN: Non-finite threshold calculated: {new_threshold}")
                return
                
            # Ensure threshold is within reasonable bounds (0.5 to 0.98)
            old_threshold = self.dynamic_threshold
            self.dynamic_threshold = float(np.clip(new_threshold, 0.5, 0.98))
            self.last_threshold_update = current_time_safe
            
            # Log threshold updates
            logger.info(f"📊 VPIN: Dynamic threshold updated: {old_threshold:.3f} → {self.dynamic_threshold:.3f} "
                       f"(p98 from {len(valid_vpin_values)} valid samples)")
            
        except (ValueError, TypeError, OverflowError) as e:
            logger.error(f"VPIN: Threshold update error: {e}")
            # Keep existing threshold, don't update
        except Exception as e:
            logger.error(f"VPIN: Unexpected threshold update error: {e}")
            # Keep existing threshold

    def update(self, qty_signed: float, px: float) -> float | None:
        """Update VPIN with comprehensive error handling and validation"""
        try:
            # Enhanced input validation
            qty_signed = float(qty_signed)
            px = float(px)
            
            if not np.isfinite(qty_signed) or not np.isfinite(px):
                logger.warning(f"VPIN: Non-finite inputs: qty={qty_signed}, px={px}")
                return None
                
            if px <= 0:
                logger.warning(f"VPIN: Invalid price: {px}")
                return None
            
            # Initialize last_price from first trade if not set
            if self.last_price == 0.0:
                self.last_price = px
            else:
                self.last_price = px
                
            current_time = time.time()
            
            # Update ATR for dynamic bucket sizing with validation
            try:
                self._update_atr(px)
            except Exception as e:
                logger.warning(f"VPIN: ATR update error: {e}")
            
            # Calculate dynamic bucket size with enhanced validation
            try:
                if self.atr_15m is not None and np.isfinite(self.atr_15m) and self.atr_15m > 0:
                    dynamic_bucket = float(self.atr_15m * 150)
                    if np.isfinite(dynamic_bucket) and dynamic_bucket > 0:
                        bucket_size = max(self.base_bucket, dynamic_bucket)
                    else:
                        bucket_size = max(self.base_bucket, 0.01 * px)
                else:
                    # Fallback: use 1% of price as reasonable bucket size
                    bucket_size = max(self.base_bucket, 0.01 * px) 
                    
                # Validate bucket size is reasonable
                if not np.isfinite(bucket_size) or bucket_size <= 0:
                    logger.warning(f"VPIN: Invalid bucket size: {bucket_size}, using base")
                    bucket_size = self.base_bucket
                    
            except (ValueError, TypeError, OverflowError) as e:
                logger.warning(f"VPIN: Bucket size calculation error: {e}")
                bucket_size = self.base_bucket

            # Calculate volume with overflow protection  
            try:
                vol = abs(qty_signed) * px
                if not np.isfinite(vol) or vol < 0:
                    logger.warning(f"VPIN: Invalid volume calculation: {vol}")
                    return None
                    
            except (OverflowError, FloatingPointError) as e:
                logger.warning(f"VPIN: Volume calculation overflow: {e}")
                return None
            
            # Update buy/sell buckets with validation
            try:
                if qty_signed > 0:
                    self.buy += vol
                else:
                    self.sell += vol
                    
                # Validate accumulated volumes
                if not np.isfinite(self.buy) or not np.isfinite(self.sell):
                    logger.warning(f"VPIN: Non-finite accumulated volumes: buy={self.buy}, sell={self.sell}")
                    self.buy = self.sell = 0.0
                    return None
                    
            except (OverflowError, FloatingPointError) as e:
                logger.warning(f"VPIN: Volume accumulation error: {e}")
                self.buy = self.sell = 0.0
                return None
                
            # Check if bucket is complete
            total_volume = self.buy + self.sell
            if total_volume >= bucket_size:
                try:
                    if total_volume > 0:  # Avoid division by zero
                        imb = abs(self.buy - self.sell) / total_volume
                        
                        # Validate imbalance
                        if np.isfinite(imb) and 0 <= imb <= 1:
                            self.win.append(float(imb))
                            
                            # Store for 24h threshold calculation with validation
                            if len(self.win) >= 5:
                                current_vpin = sum(self.win) / len(self.win)
                                if np.isfinite(current_vpin) and 0 <= current_vpin <= 1:
                                    self.vpin_24h.append(current_vpin)
                        else:
                            logger.warning(f"VPIN: Invalid imbalance: {imb}")
                            
                except (ValueError, TypeError, OverflowError, ZeroDivisionError) as e:
                    logger.warning(f"VPIN: Imbalance calculation error: {e}")
                
                # Reset buckets
                self.buy = self.sell = 0.0
                
                # Calculate and return VPIN if we have sufficient data
                if len(self.win) >= 5:
                    try:
                        self.value = float(sum(self.win) / len(self.win))
                        
                        # Validate VPIN value
                        if not np.isfinite(self.value) or not (0 <= self.value <= 1):
                            logger.warning(f"VPIN: Invalid VPIN value: {self.value}")
                            self.value = 0.5  # Neutral fallback
                        
                        self.last_update = current_time
                        
                        # Update dynamic threshold
                        try:
                            self._update_dynamic_threshold(current_time)
                        except Exception as e:
                            logger.warning(f"VPIN: Threshold update error: {e}")
                        
                        return self.value
                        
                    except (ValueError, TypeError, OverflowError, ZeroDivisionError) as e:
                        logger.warning(f"VPIN: Final calculation error: {e}")
                        return None
                        
            return None
            
        except Exception as e:
            logger.error(f"VPIN: Unexpected error: {e}")
            return None
    
    def get_value(self) -> float | None:
        """Get current VPIN value"""
        return self.value


class KyleLambda:
    """Kyle's Lambda - market impact coefficient with EWMA smoothing and volatility normalization"""

    def __init__(self, window: int = 120):  # P2-3 FIX: Reduced from 300 to 120 for 30min decay
        self.dp: Deque[float] = deque(maxlen=window)
        self.dv: Deque[float] = deque(maxlen=window)
        self._last_px: float | None = None
        self.value: float | None = None
        self.last_update: float = 0.0
        self.ewma_alpha: float = 0.25  # P2-3 FIX: Increased from 0.1 to 0.25 for faster decay
        
        # P3-4 FIX: Add volatility tracking for normalization
        self.price_returns: Deque[float] = deque(maxlen=60)  # 1-minute volatility window
        self.volatility: float = 1e-8  # Current volatility estimate

    def update(self, qty_signed: float, px: float) -> float | None:
        """Update Kyle's Lambda with comprehensive validation and error handling"""
        try:
            # Enhanced input validation
            qty_signed = float(qty_signed)
            px = float(px)
            
            if not np.isfinite(qty_signed) or not np.isfinite(px):
                logger.warning(f"KyleLambda: Non-finite inputs: qty={qty_signed}, px={px}")
                return self.value
                
            if px <= 0:
                logger.warning(f"KyleLambda: Invalid price: {px}")
                return self.value
            
            # Initialize price on first update
            if self._last_px is None:
                self._last_px = float(px)
                return None
            
            # Validate last price
            if not np.isfinite(self._last_px) or self._last_px <= 0:
                logger.warning(f"KyleLambda: Invalid last price: {self._last_px}, resetting")
                self._last_px = float(px)
                return None
            
            # Calculate price impact and return with validation
            try:
                impact = abs(px - self._last_px)
                
                if not np.isfinite(impact):
                    logger.warning(f"KyleLambda: Non-finite impact: {impact}")
                    self._last_px = float(px)
                    return self.value
                
                # Calculate price return with overflow protection
                if self._last_px > 0:
                    price_return = (px - self._last_px) / self._last_px
                else:
                    price_return = 0.0
                    
                if not np.isfinite(price_return):
                    logger.warning(f"KyleLambda: Non-finite price return: {price_return}")
                    price_return = 0.0
                
                # Sanity check for extreme price returns
                if abs(price_return) > 0.5:  # >50% price move
                    logger.warning(f"KyleLambda: Extreme price return: {price_return:.4f}, capping")
                    price_return = np.clip(price_return, -0.5, 0.5)
                    
            except (OverflowError, FloatingPointError, ZeroDivisionError) as e:
                logger.warning(f"KyleLambda: Price calculation error: {e}")
                self._last_px = float(px)
                return self.value
            
            # Enhanced volatility tracking with validation
            try:
                self.price_returns.append(float(price_return))
                
                if len(self.price_returns) >= 10:
                    returns_list = list(self.price_returns)
                    valid_returns = [r for r in returns_list if np.isfinite(r)]
                    
                    if len(valid_returns) >= 5:
                        volatility = float(np.std(valid_returns))
                        
                        # Validate volatility calculation
                        if np.isfinite(volatility) and volatility >= 0:
                            # Higher volatility floor to prevent over-sensitivity
                            self.volatility = max(volatility, 2e-7)  # Increased minimum
                        else:
                            logger.warning(f"KyleLambda: Invalid volatility: {volatility}")
                            self.volatility = 2e-7  # Fallback
                    else:
                        self.volatility = 2e-7  # Fallback for insufficient data
                        
            except Exception as e:
                logger.warning(f"KyleLambda: Volatility calculation error: {e}")
                self.volatility = 2e-7  # Safe fallback
            
            # Update last price
            self._last_px = float(px)
            
            # Calculate notional volume with validation
            try:
                notional = abs(qty_signed) * px
                
                if not np.isfinite(notional) or notional < 0:
                    logger.warning(f"KyleLambda: Invalid notional: {notional}")
                    return self.value
                
                # Prevent division by zero in lambda calculation
                if notional < 1e-12:  # Very small trade
                    logger.debug(f"KyleLambda: Micro-trade ignored: {notional}")
                    return self.value
                    
            except (OverflowError, FloatingPointError) as e:
                logger.warning(f"KyleLambda: Notional calculation error: {e}")
                return self.value
            
            # Enhanced volatility-normalized Kyle's lambda calculation
            try:
                # Calculate raw lambda
                raw_lambda = impact / notional
                
                if not np.isfinite(raw_lambda) or raw_lambda < 0:
                    logger.warning(f"KyleLambda: Invalid raw lambda: {raw_lambda}")
                    return self.value
                
                # Apply volatility normalization
                lam = raw_lambda / self.volatility
                
                if not np.isfinite(lam):
                    logger.warning(f"KyleLambda: Non-finite lambda after normalization: {lam}")
                    return self.value
                
                # Sanity check for extreme lambda values
                if lam > 1e6:  # Extremely high market impact
                    logger.warning(f"KyleLambda: Extreme lambda: {lam:.2e}, capping")
                    lam = 1e6
                
            except (OverflowError, FloatingPointError, ZeroDivisionError) as e:
                logger.warning(f"KyleLambda: Lambda calculation error: {e}")
                return self.value
            
            # Apply EWMA smoothing with enhanced validation
            try:
                if self.value is None:
                    self.value = float(lam)
                else:
                    if not np.isfinite(self.value):
                        logger.warning(f"KyleLambda: Corrupted existing value: {self.value}, resetting")
                        self.value = float(lam)
                    else:
                        new_value = self.ewma_alpha * lam + (1 - self.ewma_alpha) * self.value
                        
                        if np.isfinite(new_value):
                            self.value = float(new_value)
                        else:
                            logger.warning(f"KyleLambda: EWMA resulted in non-finite value: {new_value}")
                            return self.value
                
                self.last_update = time.time()
                
                # Log significant market impact events
                if self.value > 1000:  # High market impact threshold
                    logger.info(f"💥 KyleLambda: High market impact: {self.value:.2f} "
                               f"(impact={impact:.6f}, notional={notional:.2f})")
                
                return self.value
                
            except Exception as e:
                logger.warning(f"KyleLambda: EWMA smoothing error: {e}")
                return self.value
                
        except Exception as e:
            logger.error(f"KyleLambda: Unexpected error: {e}")
            return self.value
    
    def get_value(self) -> float | None:
        """Get current Kyle's Lambda value"""
        return self.value


class LPI:
    """Enhanced Liquidation Pressure Index with multi-venue aggregation and Gaussian decay"""

    def __init__(self, window_seconds: int = 60):
        # Gaussian decay for 60-second window instead of fixed record count
        self.window_seconds = window_seconds
        self.liquidations: Deque[tuple[float, float, float]] = deque()  # (timestamp, long_qty, short_qty)
        
        # Multi-venue tracking (Binance, Bybit, Deribit, BitMEX)
        # Open Interest tracking - MUST be set from external feed
        self.total_oi_estimate = 0.0  # Initialize with safe default, update from external feed
        self.oi_last_update = 0.0
        self.venue_weights = {'binance': 0.4, 'bybit': 0.3, 'deribit': 0.2, 'bitmex': 0.1}
        
        self.value: float | None = None
        self.last_update: float = 0.0

    def _apply_gaussian_decay(self, current_time: float) -> tuple[float, float]:
        """Apply Gaussian decay to liquidation data within window with comprehensive error handling"""
        long_weighted = short_weighted = 0.0
        
        try:
            current_time_safe = float(current_time)
            sigma = float(self.window_seconds) / 3.0  # 3-sigma coverage
            cutoff_time = current_time_safe - float(self.window_seconds)
            
            # Validate time parameters
            if sigma <= 0:
                logger.warning(f"LPI: Invalid sigma: {sigma}")
                return 0.0, 0.0
            
            # Remove old data outside window with enhanced error handling
            removed_count = 0
            while self.liquidations and removed_count < len(self.liquidations):  # Prevent infinite loop
                try:
                    if len(self.liquidations) == 0:
                        break
                        
                    first_item = self.liquidations[0]
                    if len(first_item) != 3:
                        logger.warning(f"LPI: Malformed liquidation data: {first_item}")
                        self.liquidations.popleft()
                        removed_count += 1
                        continue
                    
                    timestamp = float(first_item[0])
                    
                    # Validate timestamp
                    if not np.isfinite(timestamp):
                        logger.warning(f"LPI: Invalid timestamp: {timestamp}")
                        self.liquidations.popleft()
                        removed_count += 1
                        continue
                    
                    if timestamp < cutoff_time:
                        self.liquidations.popleft()
                        removed_count += 1
                    else:
                        break
                        
                except (ValueError, TypeError, IndexError, OverflowError) as e:
                    logger.warning(f"LPI: Error removing old data: {e}")
                    # Remove corrupted item and continue
                    if self.liquidations:
                        self.liquidations.popleft()
                    removed_count += 1
                    continue
            
            # Apply Gaussian weights to remaining data with comprehensive validation
            processed_count = 0
            for liquidation_data in list(self.liquidations):  # Create copy to avoid modification during iteration
                try:
                    if len(liquidation_data) != 3:
                        logger.warning(f"LPI: Skipping malformed liquidation: {liquidation_data}")
                        continue
                        
                    timestamp, long_qty, short_qty = liquidation_data
                    
                    # Enhanced type conversion with overflow protection
                    timestamp = float(timestamp)
                    long_qty = float(long_qty) 
                    short_qty = float(short_qty)
                    
                    # Validate all values are finite
                    if not all(np.isfinite([timestamp, long_qty, short_qty])):
                        logger.warning(f"LPI: Non-finite values: {timestamp}, {long_qty}, {short_qty}")
                        continue
                    
                    # Validate quantities are non-negative
                    if long_qty < 0 or short_qty < 0:
                        logger.warning(f"LPI: Negative quantities: {long_qty}, {short_qty}")
                        continue
                    
                    age = current_time_safe - timestamp
                    
                    # Validate age is reasonable
                    if age < 0:
                        logger.warning(f"LPI: Future timestamp detected: {timestamp} > {current_time_safe}")
                        continue
                        
                    if age <= self.window_seconds:
                        # Gaussian weight: exp(-0.5 * (age/sigma)^2) with overflow protection
                        normalized_age = age / sigma
                        
                        # Prevent overflow in exponential
                        if abs(normalized_age) > 10.0:  # exp(-50) ≈ 0
                            weight = 0.0
                        else:
                            try:
                                weight = float(np.exp(-0.5 * normalized_age ** 2))
                            except (OverflowError, FloatingPointError):
                                weight = 0.0
                        
                        # Validate weight
                        if not np.isfinite(weight) or weight < 0:
                            logger.warning(f"LPI: Invalid weight: {weight}")
                            continue
                            
                        # Apply weighted contribution with overflow protection
                        try:
                            long_contribution = long_qty * weight
                            short_contribution = short_qty * weight
                            
                            # Check for overflow
                            if not np.isfinite(long_contribution) or not np.isfinite(short_contribution):
                                logger.warning(f"LPI: Overflow in contribution calculation")
                                continue
                                
                            long_weighted += long_contribution
                            short_weighted += short_contribution
                            
                        except (OverflowError, FloatingPointError) as e:
                            logger.warning(f"LPI: Overflow in weighted sum: {e}")
                            continue
                    
                    processed_count += 1
                    
                except (ValueError, TypeError, IndexError, OverflowError) as e:
                    logger.warning(f"LPI: Error processing liquidation data: {e}")
                    continue
            
            # Final validation of results
            if not np.isfinite(long_weighted):
                logger.warning(f"LPI: Non-finite long_weighted: {long_weighted}")
                long_weighted = 0.0
                
            if not np.isfinite(short_weighted):
                logger.warning(f"LPI: Non-finite short_weighted: {short_weighted}")
                short_weighted = 0.0
            
            # Log processing summary for debugging
            if processed_count > 0:
                logger.debug(f"LPI: Processed {processed_count} liquidations, "
                           f"weighted sums: L={long_weighted:.2f}, S={short_weighted:.2f}")
            
            return float(long_weighted), float(short_weighted)
            
        except Exception as e:
            logger.error(f"LPI: Unexpected error in Gaussian decay: {e}")
            return 0.0, 0.0

    def update(self, *args, venue: str = 'binance') -> float | None:
        """Update with liquidation data from specified venue.

        Accepts either the old (side: str, qty: float) signature *or*
        the new (long_qty: float, short_qty: float) signature with venue tracking.
        
        Args:
            *args: Either (long_qty, short_qty) or (side, qty)
            venue: Exchange venue identifier for weighting
            
        Returns:
            Updated LPI value or None if invalid input
        """
        current_time = time.time()

        # ------------------------------------------------------------------
        # Enhanced argument parsing with comprehensive validation
        # ------------------------------------------------------------------
        try:
            if len(args) == 2:
                # Type 1: New call style (long_qty, short_qty)
                if isinstance(args[0], (int, float)) and isinstance(args[1], (int, float)):
                    long_qty = float(args[0])
                    short_qty = float(args[1])
                    
                    # Validate quantities are non-negative
                    if long_qty < 0 or short_qty < 0:
                        logger.warning(f"LPI: Negative liquidation quantities not allowed: {long_qty}, {short_qty}")
                        return None
                        
                # Type 2: Old call style (side, qty)
                elif isinstance(args[0], str) and isinstance(args[1], (int, float)):
                    side = str(args[0]).strip().lower()
                    qty = float(args[1])
                    
                    # Validate quantity is positive
                    if qty <= 0:
                        logger.warning(f"LPI: Invalid liquidation quantity: {qty}")
                        return None
                    
                    # Map side to long/short quantities
                    if side in ("long", "buy", "l", "b"):
                        long_qty, short_qty = qty, 0.0
                    elif side in ("short", "sell", "s", "sl"):
                        long_qty, short_qty = 0.0, qty
                    else:
                        logger.warning(f"LPI: Unknown liquidation side: {side}")
                        return None
                else:
                    logger.warning(f"LPI: Invalid argument types: {type(args[0])}, {type(args[1])}")
                    return None
            else:
                logger.warning(f"LPI: Expected 2 arguments, got {len(args)}")
                return None

        except (ValueError, TypeError, OverflowError) as e:
            logger.warning(f"LPI: Argument parsing error: {e}")
            return None

        # ------------------------------------------------------------------
        # Venue weighting with validation
        # ------------------------------------------------------------------
        try:
            venue_safe = str(venue).strip().lower()
            venue_weight = self.venue_weights.get(venue_safe, 1.0)
            
            # Validate venue weight
            if not (0.0 < venue_weight <= 2.0):
                logger.warning(f"LPI: Invalid venue weight {venue_weight} for {venue_safe}, using 1.0")
                venue_weight = 1.0
                
            long_qty *= venue_weight
            short_qty *= venue_weight
            
        except (ValueError, TypeError, OverflowError) as e:
            logger.warning(f"LPI: Venue weighting error: {e}")
            return None

        # ------------------------------------------------------------------
        # Enhanced data storage with comprehensive validation
        # ------------------------------------------------------------------
        try:
            # Validate all components before storage
            timestamp_safe = float(current_time)
            long_qty_safe = float(long_qty)
            short_qty_safe = float(short_qty)
            
            # Additional safety checks
            if not (0 <= timestamp_safe <= 2**53):  # JavaScript safe integer limit
                logger.warning(f"LPI: Invalid timestamp: {timestamp_safe}")
                return None
                
            if not (0 <= long_qty_safe <= 1e12):  # Sanity check for liquidation size
                logger.warning(f"LPI: Excessive long liquidation: {long_qty_safe}")
                return None
                
            if not (0 <= short_qty_safe <= 1e12):
                logger.warning(f"LPI: Excessive short liquidation: {short_qty_safe}")
                return None
            
            # Store validated data
            self.liquidations.append((timestamp_safe, long_qty_safe, short_qty_safe))
            
        except (ValueError, TypeError, OverflowError) as e:
            logger.warning(f"LPI: Data validation error: {e}")
            return None

        # ------------------------------------------------------------------
        # Enhanced LPI calculation with comprehensive error handling
        # ------------------------------------------------------------------
        try:
            # Calculate weighted liquidation pressure using Gaussian decay
            long_weighted, short_weighted = self._apply_gaussian_decay(current_time)
            
            total_liquidated = long_weighted + short_weighted
            
            # Enhanced validation checks
            if total_liquidated <= 0:
                logger.debug("LPI: No liquidation data in window")
                return None
                
            if self.total_oi_estimate <= 0:
                logger.warning("LPI: Invalid or missing OI estimate, cannot calculate pressure")
                return None
            
            # Calculate net pressure with bounds checking
            net_pressure = (short_weighted - long_weighted) / self.total_oi_estimate
            
            # Validate net pressure is reasonable
            if abs(net_pressure) > 10.0:  # Sanity check
                logger.warning(f"LPI: Extreme net pressure detected: {net_pressure:.6f}, capping")
                net_pressure = np.clip(net_pressure, -10.0, 10.0)
            
            # Enhanced sensitivity for 1-minute trading with validation
            sensitivity_multiplier = 10.0  # Increased from 5.0 for minute-edge
            
            # Apply tanh transformation for bounded range [-2, +2]
            pressure_scaled = net_pressure * sensitivity_multiplier
            
            # Additional bounds checking before tanh
            if abs(pressure_scaled) > 50.0:  # Prevent overflow in tanh
                pressure_scaled = np.clip(pressure_scaled, -50.0, 50.0)
            
            self.value = float(np.tanh(pressure_scaled) * 2.0)
            
            # Final validation of output
            if not np.isfinite(self.value):
                logger.warning("LPI: Non-finite value generated, resetting")
                self.value = 0.0
                
            self.last_update = current_time
            
            # Log significant pressure events
            if abs(self.value) > 1.5:
                logger.info(f"🚨 LPI: High liquidation pressure detected: {self.value:.3f} "
                           f"(L:{long_weighted:.0f}, S:{short_weighted:.0f}, OI:{self.total_oi_estimate:.0f})")
            
            return self.value
            
        except (ValueError, TypeError, OverflowError, np.core._exceptions._ArrayMemoryError) as e:
            logger.error(f"LPI: Calculation error: {e}")
            self.value = 0.0  # Safe fallback
            return None
        except Exception as e:
            logger.error(f"LPI: Unexpected error: {e}")
            self.value = 0.0
            return None

    def update_oi_estimate(self, total_oi: float) -> None:
        """Update total open interest estimate for normalization with comprehensive validation"""
        try:
            # Enhanced validation of OI input
            oi_safe = float(total_oi)
            
            # Validate OI is finite and positive
            if not np.isfinite(oi_safe):
                logger.warning(f"LPI: Non-finite OI estimate: {total_oi}")
                return
                
            if oi_safe <= 0:
                logger.warning(f"LPI: Non-positive OI estimate: {oi_safe}")
                return
            
            # Sanity check for reasonable OI values (between 1M and 100B)
            if not (1e6 <= oi_safe <= 1e11):
                logger.warning(f"LPI: Unusual OI estimate: {oi_safe:,.0f}, using bounds")
                oi_safe = np.clip(oi_safe, 1e6, 1e11)
            
            # Update with validated value
            old_oi = self.total_oi_estimate
            self.total_oi_estimate = oi_safe
            self.oi_last_update = time.time()
            
            # Log significant OI changes
            if old_oi > 0 and abs(oi_safe - old_oi) / old_oi > 0.1:  # >10% change
                logger.info(f"📊 LPI: Significant OI change: {old_oi:,.0f} → {oi_safe:,.0f} "
                           f"({((oi_safe/old_oi - 1)*100):+.1f}%)")
            
        except (ValueError, TypeError, OverflowError) as e:
            logger.error(f"LPI: OI update error: {e}")
            # Keep existing OI estimate, don't update
    
    def get_value(self) -> float | None:
        """Get current LPI value"""
        return self.value


__all__ = ["OFI", "VPIN", "KyleLambda", "LPI", "Liquidations", "TopOfBookImbalance", "WallRatio", "FundingBasis", "TradeImbalance", "ZBand", "AdvancedZBandWrapper"]


# ------------------------------------------------------------------
# ⬇︎ ATOMIC 1-minute metrics (NO smoothing here)
# ------------------------------------------------------------------

class Liquidations:
    """Track liquidation count and notional USD value per minute"""
    
    def __init__(self):
        self.reset(0)
        self.value: float | None = None
        self.last_update: float = 0.0
    
    def reset(self, timestamp: float) -> None:
        """Reset counters for new minute"""
        self.cnt = 0
        self.notional = 0.0
        self.last_update = timestamp
    
    def on_liq(self, qty: float, price: float) -> None:
        """Record liquidation event with comprehensive validation"""
        try:
            qty_safe = float(qty)
            price_safe = float(price)
            
            # Validate inputs
            if not np.isfinite(qty_safe) or not np.isfinite(price_safe):
                logger.warning(f"Liquidations: Non-finite inputs: qty={qty}, price={price}")
                return
                
            if price_safe <= 0:
                logger.warning(f"Liquidations: Invalid price: {price_safe}")
                return
                
            if qty_safe == 0:
                logger.debug(f"Liquidations: Zero quantity liquidation ignored")
                return
            
            # Calculate notional with overflow protection
            try:
                notional_value = abs(qty_safe * price_safe)
                
                if not np.isfinite(notional_value):
                    logger.warning(f"Liquidations: Non-finite notional: {notional_value}")
                    return
                
                # Sanity check for reasonable liquidation sizes
                if notional_value > 1e12:  # > $1 trillion
                    logger.warning(f"Liquidations: Excessive liquidation size: ${notional_value:,.0f}")
                    return
                
                self.cnt += 1
                self.notional += notional_value
                self.value = float(self.notional)
                
                # Log large liquidations
                if notional_value > 1e6:  # > $1M
                    logger.info(f"💥 Large liquidation: ${notional_value:,.0f} "
                               f"(qty={qty_safe:.2f}, px=${price_safe:,.2f})")
                
            except (OverflowError, FloatingPointError) as e:
                logger.warning(f"Liquidations: Calculation error: {e}")
                return
                
        except (ValueError, TypeError) as e:
            logger.warning(f"Liquidations: Input validation error: {e}")
            return
    
    @property
    def count(self) -> int:
        """Number of liquidations in current minute"""
        return self.cnt


class TopOfBookImbalance:
    """Top-of-book imbalance: bid_qty1 / (bid_qty1 + ask_qty1)"""
    
    def __init__(self):
        self.reset(0)
        self.value: float | None = None
        self.last_update: float = 0.0
    
    def reset(self, timestamp: float) -> None:
        """Reset for new minute"""
        self.bid = 0.0
        self.ask = 0.0
        self.last_update = timestamp
    
    def on_best(self, bid_qty: float, ask_qty: float) -> None:
        """Update with best bid/ask quantities with comprehensive validation"""
        try:
            bid_safe = float(bid_qty)
            ask_safe = float(ask_qty)
            
            # Validate inputs
            if not np.isfinite(bid_safe) or not np.isfinite(ask_safe):
                logger.warning(f"TopOfBookImbalance: Non-finite inputs: bid={bid_qty}, ask={ask_qty}")
                self.value = 0.5  # Neutral fallback
                return
                
            if bid_safe < 0 or ask_safe < 0:
                logger.warning(f"TopOfBookImbalance: Negative quantities: bid={bid_safe}, ask={ask_safe}")
                self.value = 0.5  # Neutral fallback
                return
            
            self.bid = bid_safe
            self.ask = ask_safe
            
            # Calculate imbalance with validation
            tot = self.bid + self.ask
            if tot > 1e-9:  # Avoid division by zero
                imbalance = self.bid / tot
                
                # Validate result is in [0,1] range
                if 0 <= imbalance <= 1:
                    self.value = float(imbalance)
                else:
                    logger.warning(f"TopOfBookImbalance: Invalid imbalance: {imbalance}")
                    self.value = 0.5  # Neutral fallback
            else:
                self.value = 0.5  # Neutral when no liquidity
                
        except (ValueError, TypeError, OverflowError) as e:
            logger.warning(f"TopOfBookImbalance: Calculation error: {e}")
            self.value = 0.5  # Default to neutral


class WallRatio:
    """Wall ratio: share of "large" orders vs total orders"""
    
    def __init__(self, threshold: float = 50.0):
        self.threshold = threshold
        self.reset(0)
        self.value: float | None = None
        self.last_update: float = 0.0
    
    def reset(self, timestamp: float) -> None:
        """Reset counters for new minute"""
        self.big = 0
        self.tot = 0
        self.last_update = timestamp
    
    def on_level(self, size: float) -> None:
        """Process orderbook level with comprehensive validation"""
        try:
            size_safe = float(size)
            
            # Validate input
            if not np.isfinite(size_safe):
                logger.warning(f"WallRatio: Non-finite size: {size}")
                return
                
            if size_safe < 0:
                logger.warning(f"WallRatio: Negative size: {size_safe}")
                return
            
            self.tot += 1
            if size_safe >= self.threshold:
                self.big += 1
                
            # Enhanced reactivity for 1-minute trading with validation
            reactivity_multiplier = 0.6  # Reduced from default to increase sensitivity
            
            # Update value with enhanced reactivity and validation
            if self.tot > 0:
                try:
                    imbalance_ratio = self.big / self.tot
                    
                    # Validate ratio
                    if not (0 <= imbalance_ratio <= 1):
                        logger.warning(f"WallRatio: Invalid ratio: {imbalance_ratio}")
                        self.value = 0.0
                        return
                    
                    # Apply non-linear transformation for better signal detection
                    if reactivity_multiplier > 0:
                        transformed_value = imbalance_ratio ** reactivity_multiplier
                        
                        if np.isfinite(transformed_value) and 0 <= transformed_value <= 1:
                            self.value = float(transformed_value)
                        else:
                            logger.warning(f"WallRatio: Invalid transformed value: {transformed_value}")
                            self.value = float(imbalance_ratio)  # Fallback to original ratio
                    else:
                        self.value = float(imbalance_ratio)
                        
                except (OverflowError, FloatingPointError, ZeroDivisionError) as e:
                    logger.warning(f"WallRatio: Calculation error: {e}")
                    self.value = 0.0
            else:
                self.value = float("nan")  # No data yet
                
        except (ValueError, TypeError) as e:
            logger.warning(f"WallRatio: Input validation error: {e}")
            return


class FundingBasis:
    """Funding basis: spot - perpetual price difference"""
    
    def __init__(self):
        self.reset(0)
        self.value: float | None = None
        self.last_update: float = 0.0
    
    def reset(self, timestamp: float) -> None:
        """Reset for new minute"""
        self.basis = float("nan")
        self.last_update = timestamp
    
    def on_funding(self, spot: float, perp: float) -> None:
        """Update with spot and perp prices with comprehensive validation"""
        try:
            spot_safe = float(spot)
            perp_safe = float(perp)
            
            # Validate inputs
            if not np.isfinite(spot_safe) or not np.isfinite(perp_safe):
                logger.warning(f"FundingBasis: Non-finite prices: spot={spot}, perp={perp}")
                self.value = 0.0
                return
                
            if spot_safe <= 0 or perp_safe <= 0:
                logger.warning(f"FundingBasis: Invalid prices: spot={spot_safe}, perp={perp_safe}")
                self.value = 0.0
                return
            
            # Calculate basis with overflow protection
            try:
                basis = spot_safe - perp_safe
                
                if not np.isfinite(basis):
                    logger.warning(f"FundingBasis: Non-finite basis: {basis}")
                    self.value = 0.0
                    return
                
                # Sanity check for reasonable basis values
                basis_pct = abs(basis) / spot_safe
                if basis_pct > 0.1:  # >10% basis seems extreme
                    logger.warning(f"FundingBasis: Extreme basis: {basis:.2f} ({basis_pct:.2%})")
                
                self.basis = basis
                self.value = basis
                
                # Log significant basis events
                if abs(basis) > spot_safe * 0.005:  # >0.5% basis
                    logger.info(f"📊 FundingBasis: Significant basis: ${basis:.2f} "
                               f"({basis/spot_safe:.3%})")
                
            except (OverflowError, FloatingPointError) as e:
                logger.warning(f"FundingBasis: Calculation error: {e}")
                self.value = 0.0
                
        except (ValueError, TypeError) as e:
            logger.warning(f"FundingBasis: Input validation error: {e}")
            self.value = 0.0


class TradeImbalance:
    """Trade imbalance: (#buy - #sell) / total_trades"""
    
    def __init__(self):
        self.reset(0)
        self.value: float | None = None
        self.last_update: float = 0.0
    
    def reset(self, timestamp: float) -> None:
        """Reset counters for new minute"""
        self.buy_count = 0
        self.sell_count = 0
        self.last_update = timestamp
    
    def on_tick(self, side: str) -> None:
        """Process trade tick with comprehensive validation"""
        try:
            side_safe = str(side).strip().upper()
            
            # Validate side input
            if not side_safe:
                logger.warning(f"TradeImbalance: Empty side string")
                return
            
            # Enhanced side mapping with more variations
            if side_safe in ('B', 'BUY', 'BID', 'LONG', 'L'):
                self.buy_count += 1
            elif side_safe in ('S', 'SELL', 'ASK', 'SHORT', 'SL', 'A'):
                self.sell_count += 1
            else:
                logger.warning(f"TradeImbalance: Unknown side: '{side_safe}'")
                return
            
            # Update value with validation
            total = self.buy_count + self.sell_count
            if total > 0:
                try:
                    imbalance = (self.buy_count - self.sell_count) / total
                    
                    # Validate imbalance is in [-1, 1] range
                    if -1 <= imbalance <= 1:
                        self.value = float(imbalance)
                    else:
                        logger.warning(f"TradeImbalance: Invalid imbalance: {imbalance}")
                        # Recalculate from scratch
                        if total > 0:
                            self.value = float((self.buy_count - self.sell_count) / total)
                        else:
                            self.value = 0.0
                            
                except (OverflowError, FloatingPointError, ZeroDivisionError) as e:
                    logger.warning(f"TradeImbalance: Calculation error: {e}")
                    self.value = 0.0
            else:
                self.value = 0.0
                
        except (ValueError, TypeError) as e:
            logger.warning(f"TradeImbalance: Input validation error: {e}")
            return


# ------------------------------------------------------------------
# Advanced Z-Band Analysis (Items 26-30)
# ------------------------------------------------------------------

class AdvancedZBandWrapper:
    """
    Advanced Z-band wrapper with multi-timeframe analysis and regime detection
    Items 26-30: Advanced Z-band features
    """
    
    def __init__(self, window_size: int = 120, timeframes: List[int] = None):
        self.window_size = window_size
        self.timeframes = timeframes or [60, 120, 300, 600]  # 1m, 2m, 5m, 10m
        
        # Multi-timeframe Z-score tracking
        self.z_scores = {tf: deque(maxlen=tf) for tf in self.timeframes}
        self.regime_states = {tf: 'normal' for tf in self.timeframes}
        
        # Advanced statistics
        self.values = deque(maxlen=window_size)
        self.volatility_adjusted = deque(maxlen=window_size)
        self.momentum_score = 0.0
        self.regime_confidence = 0.0
        
        # Regime detection thresholds
        self.extreme_threshold = 2.5
        self.momentum_threshold = 1.8
        self.volatility_window = 20
        
        self.value = 0.0
        self.raw_value = 0.0
        self.confidence = 0.0
    
    def update(self, raw_value: float, timestamp: float = None) -> Dict[str, float]:
        """
        Update with advanced multi-timeframe Z-band analysis
        Returns comprehensive analysis including regime detection
        """
        try:
            self.raw_value = raw_value
            self.values.append(raw_value)
            
            if len(self.values) < 10:
                self.value = 0.0
                return self._get_analysis_output()
            
            # Calculate multi-timeframe Z-scores
            results = {}
            
            for timeframe in self.timeframes:
                if len(self.values) >= min(timeframe, len(self.values)):
                    # Get values for this timeframe
                    tf_values = list(self.values)[-timeframe:]
                    
                    if len(tf_values) >= 5:
                        mean_val = np.mean(tf_values)
                        std_val = np.std(tf_values)
                        
                        if std_val > 1e-10:
                            z_score = (raw_value - mean_val) / std_val
                            self.z_scores[timeframe].append(z_score)
                            
                            # Detect regime for this timeframe
                            regime = self._detect_regime(z_score, timeframe)
                            self.regime_states[timeframe] = regime
                            
                            results[f'z_{timeframe}s'] = z_score
                            results[f'regime_{timeframe}s'] = 1.0 if regime == 'extreme' else 0.0
            
            # Calculate primary Z-score (120s default)
            primary_tf = 120
            if primary_tf in results:
                self.value = results[f'z_{primary_tf}s']
            else:
                # Fallback to simple Z-score
                mean_val = np.mean(self.values)
                std_val = np.std(self.values)
                self.value = (raw_value - mean_val) / std_val if std_val > 1e-10 else 0.0
            
            # Calculate momentum and volatility-adjusted scores
            self._calculate_advanced_metrics()
            
            return self._get_analysis_output()
            
        except Exception as e:
            logger.warning(f"AdvancedZBandWrapper error: {e}")
            self.value = 0.0
            return self._get_analysis_output()
    
    def _detect_regime(self, z_score: float, timeframe: int) -> str:
        """Detect market regime based on Z-score patterns"""
        try:
            recent_z = list(self.z_scores[timeframe])[-5:] if len(self.z_scores[timeframe]) >= 5 else []
            
            if not recent_z:
                return 'normal'
            
            # Extreme regime: sustained high Z-scores
            if abs(z_score) > self.extreme_threshold:
                extreme_count = sum(1 for z in recent_z if abs(z) > self.momentum_threshold)
                if extreme_count >= 3:
                    return 'extreme'
            
            # Momentum regime: consistent directional movement
            if len(recent_z) >= 3:
                positive_momentum = sum(1 for z in recent_z if z > self.momentum_threshold)
                negative_momentum = sum(1 for z in recent_z if z < -self.momentum_threshold)
                
                if positive_momentum >= 3 or negative_momentum >= 3:
                    return 'momentum'
            
            return 'normal'
            
        except Exception:
            return 'normal'
    
    def _calculate_advanced_metrics(self):
        """Calculate advanced momentum and volatility metrics"""
        try:
            if len(self.values) < self.volatility_window:
                return
            
            recent_values = list(self.values)[-self.volatility_window:]
            
            # Momentum score: rate of change in Z-scores
            if len(self.values) >= 5:
                recent_z = [self.value] + [v for v in list(self.values)[-4:]]
                momentum_changes = np.diff(recent_z)
                self.momentum_score = np.mean(momentum_changes) if len(momentum_changes) > 0 else 0.0
            
            # Regime confidence: consistency of regime detection across timeframes
            extreme_regimes = sum(1 for regime in self.regime_states.values() if regime == 'extreme')
            total_regimes = len(self.regime_states)
            self.regime_confidence = extreme_regimes / total_regimes if total_regimes > 0 else 0.0
            
            # Volatility-adjusted Z-score
            volatility = np.std(recent_values) if len(recent_values) > 1 else 1.0
            vol_adjustment = min(2.0, max(0.5, 1.0 / (volatility + 0.1)))
            
            vol_adjusted_z = self.value * vol_adjustment
            self.volatility_adjusted.append(vol_adjusted_z)
            
        except Exception as e:
            logger.warning(f"Advanced metrics calculation error: {e}")
    
    def _get_analysis_output(self) -> Dict[str, float]:
        """Get comprehensive analysis output"""
        return {
            'z_score': self.value,
            'raw_value': self.raw_value,
            'momentum_score': self.momentum_score,
            'regime_confidence': self.regime_confidence,
            'volatility_adjusted': list(self.volatility_adjusted)[-1] if self.volatility_adjusted else 0.0,
            'regime_state': 1.0 if any(r == 'extreme' for r in self.regime_states.values()) else 0.0
        }
    
    def get_regime_summary(self) -> Dict[str, str]:
        """Get regime summary across all timeframes"""
        return self.regime_states.copy()


# ------------------------------------------------------------------
# Rolling z-score band (regression-band‐lite)
# ------------------------------------------------------------------

class ZBand:
    """Rolling Z-score wrapper for any atomic indicator"""
    
    def __init__(self, src, window: int = 120):
        from collections import deque
        self.src = src
        self.buf = deque(maxlen=window)
        self._z = 0.0
        self.window_size = window
        self.last_update: float = 0.0
    
    def reset(self, timestamp: float) -> None:
        """Keep buffer across resets"""
        self.last_update = timestamp
    
    def update(self) -> None:
        """Update Z-score based on source value with comprehensive validation"""
        try:
            # Validate source exists and has value
            if not hasattr(self.src, 'value') or self.src.value is None:
                self._z = 0.0
                return
            
            # Get and validate source value
            try:
                v = float(self.src.value)
                if not np.isfinite(v):
                    logger.warning(f"ZBand: Non-finite source value: {self.src.value}")
                    self._z = 0.0
                    return
            except (ValueError, TypeError) as e:
                logger.warning(f"ZBand: Invalid source value: {self.src.value}, error: {e}")
                self._z = 0.0
                return
            
            # Add to buffer
            self.buf.append(v)
            
            # Calculate Z-score if we have sufficient data
            if len(self.buf) >= 20:  # Minimum data for statistical relevance
                try:
                    buf_array = np.array(list(self.buf))
                    
                    # Remove any non-finite values
                    finite_values = buf_array[np.isfinite(buf_array)]
                    
                    if len(finite_values) < 10:  # Need minimum finite values
                        logger.warning(f"ZBand: Insufficient finite values: {len(finite_values)}")
                        self._z = 0.0
                        return
                    
                    mean_val = float(np.mean(finite_values))
                    std_val = float(np.std(finite_values))
                    
                    # Validate statistics
                    if not np.isfinite(mean_val) or not np.isfinite(std_val):
                        logger.warning(f"ZBand: Non-finite statistics: mean={mean_val}, std={std_val}")
                        self._z = 0.0
                        return
                    
                    if std_val > 1e-9:  # Avoid division by zero
                        z_score = (v - mean_val) / std_val
                        
                        if np.isfinite(z_score):
                            # Sanity check for extreme Z-scores
                            if abs(z_score) > 50.0:
                                logger.warning(f"ZBand: Extreme Z-score: {z_score:.2f}, capping")
                                z_score = np.clip(z_score, -50.0, 50.0)
                                
                            self._z = float(z_score)
                        else:
                            logger.warning(f"ZBand: Non-finite Z-score: {z_score}")
                            self._z = 0.0
                    else:
                        self._z = 0.0  # No variance in data
                        
                except Exception as e:
                    logger.warning(f"ZBand: Statistics calculation error: {e}")
                    self._z = 0.0
            else:
                self._z = 0.0  # Not enough data
                
        except Exception as e:
            logger.error(f"ZBand: Unexpected error: {e}")
            self._z = 0.0
    
    @property
    def value(self) -> float:
        """Current Z-score"""
        return self._z
    
    def save_state(self) -> list:
        """Save buffer state for persistence"""
        return list(self.buf)
    
    def load_state(self, state_data: list) -> None:
        """Load buffer state from persistence with comprehensive validation"""
        try:
            if not isinstance(state_data, list):
                logger.warning(f"ZBand: Invalid state data type: {type(state_data)}")
                return
                
            if not state_data:
                logger.debug("ZBand: Empty state data, starting fresh")
                return
            
            self.buf.clear()
            
            # Validate and load state data
            loaded_count = 0
            for val in state_data[-self.window_size:]:  # Keep only recent values
                try:
                    val_float = float(val)
                    if np.isfinite(val_float):
                        self.buf.append(val_float)
                        loaded_count += 1
                    else:
                        logger.warning(f"ZBand: Skipping non-finite state value: {val}")
                except (ValueError, TypeError) as e:
                    logger.warning(f"ZBand: Skipping invalid state value: {val}, error: {e}")
                    continue
            
            if loaded_count > 0:
                logger.info(f"ZBand: Loaded {loaded_count} valid state values")
            else:
                logger.warning("ZBand: No valid state values loaded")
                
        except Exception as e:
            logger.error(f"ZBand: State loading error: {e}")
            # Clear buffer on error to start fresh
            self.buf.clear()
