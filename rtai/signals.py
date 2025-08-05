"""Trading signal detection and alert system"""

import time
import math
from typing import Dict, Optional, List
from dataclasses import dataclass
from loguru import logger
from rtai.telegram_bot import send_trading_signal


@dataclass
class SignalConfig:
    """Configuration for signal thresholds - Phase 2 Optimized + Basic Oscillators"""
    ofi_buy_threshold: float = -1.8    # Phase 2: Tightened from -2.0 to cut noise
    ofi_sell_threshold: float = 1.8    # Phase 2: Tightened from 2.0 to cut noise  
    vpin_threshold: float = 0.98       # Phase 2: Dynamic P98 threshold (up from static 0.8)
    kyle_threshold: float = 0.1        # Kyle lambda above this = high impact
    lpi_buy_threshold: float = -0.5    # LPI below this = liquidation pressure favors longs
    lpi_sell_threshold: float = 0.5    # LPI above this = liquidation pressure favors shorts
    
    # Basic Oscillators Thresholds - RSI-style extreme detection
    wall_ratio_extreme: float = 2.0    # |z-score| > 2.0 = extreme wall imbalance
    trade_imbalance_extreme: float = 2.0  # |z-score| > 2.0 = extreme trade imbalance
    liquidation_extreme: float = 1.8    # |z-score| > 1.8 = extreme liquidation activity
    dir_extreme: float = 2.2           # |z-score| > 2.2 = extreme depth imbalance ratio
    funding_accel_extreme: float = 2.5  # |z-score| > 2.5 = extreme funding acceleration
    
    cooldown_seconds: int = 300        # 5 minutes between signals for same symbol
    basic_osc_cooldown: int = 120      # 2 minutes between basic oscillator signals
    funding_accel_extreme: float = 2.5   # |z-score| > 2.5 = extreme funding acceleration
    
    cooldown_seconds: int = 300        # 5 minutes between signals for same symbol
    basic_osc_cooldown: int = 180      # 3 minutes between basic oscillator signals


class SignalDetector:
    """Detects trading signals based on indicator values"""
    
    def __init__(self, config: SignalConfig = None):
        try:
            # Enhanced input validation
            if config is not None and not isinstance(config, SignalConfig):
                logger.warning(f"SignalDetector: Invalid config type: {type(config)}")
                config = None
            
            self.config = config or SignalConfig()
            self.last_signals: Dict[str, Dict[str, float]] = {}  # symbol -> {indicator: timestamp}
            
        except Exception as e:
            logger.error(f"SignalDetector.__init__ error: {e}")
            # Set safe defaults
            self.config = SignalConfig()
            self.last_signals = {}
    
    def _can_send_signal(self, symbol: str, indicator: str, use_basic_cooldown: bool = False) -> bool:
        """Check if enough time has passed since last signal"""
        try:
            # Enhanced input validation
            if not isinstance(symbol, str):
                logger.warning(f"SignalDetector: Invalid symbol type: {type(symbol)}")
                return False
            
            if not isinstance(indicator, str):
                logger.warning(f"SignalDetector: Invalid indicator type: {type(indicator)}")
                return False
            
            if not symbol.strip():
                logger.warning("SignalDetector: Empty symbol")
                return False
            
            if not indicator.strip():
                logger.warning("SignalDetector: Empty indicator")
                return False
            
            now = time.time()
            
            if symbol not in self.last_signals:
                self.last_signals[symbol] = {}
            
            last_time = self.last_signals[symbol].get(indicator, 0)
            
            # Validate last_time
            if not isinstance(last_time, (int, float)) or not math.isfinite(last_time):
                logger.warning(f"SignalDetector: Invalid last_time for {symbol}.{indicator}: {last_time}")
                last_time = 0
            
            # Enhanced cooldown calculation
            try:
                cooldown = self.config.basic_osc_cooldown if use_basic_cooldown else self.config.cooldown_seconds
                
                if not isinstance(cooldown, (int, float)) or not math.isfinite(cooldown) or cooldown < 0:
                    logger.warning(f"SignalDetector: Invalid cooldown: {cooldown}")
                    cooldown = 300  # Default 5 minutes
                
                time_passed = now - last_time
                return time_passed >= cooldown
                
            except Exception as e:
                logger.warning(f"SignalDetector: Cooldown calculation error: {e}")
                return True  # Allow signal on error
            
        except Exception as e:
            logger.error(f"SignalDetector._can_send_signal error: {e}")
            return False
    
    def _record_signal(self, symbol: str, indicator: str):
        """Record that a signal was sent"""
        try:
            # Enhanced input validation
            if not isinstance(symbol, str) or not isinstance(indicator, str):
                logger.warning(f"SignalDetector: Invalid types for record_signal - symbol: {type(symbol)}, indicator: {type(indicator)}")
                return
            
            if not symbol.strip() or not indicator.strip():
                logger.warning("SignalDetector: Empty symbol or indicator in record_signal")
                return
            
            if symbol not in self.last_signals:
                self.last_signals[symbol] = {}
            
            current_time = time.time()
            self.last_signals[symbol][indicator] = current_time
            
        except Exception as e:
            logger.error(f"SignalDetector._record_signal error: {e}")
            return
    
    async def check_ofi_signal(self, symbol: str, ofi_value: float) -> Optional[str]:
        """Check for OFI-based trading signals"""
        try:
            # Enhanced input validation
            if not isinstance(symbol, str):
                logger.warning(f"SignalDetector: Invalid symbol type in check_ofi_signal: {type(symbol)}")
                return None
            
            if not symbol.strip():
                logger.warning("SignalDetector: Empty symbol in check_ofi_signal")
                return None
            
            if not isinstance(ofi_value, (int, float)):
                logger.warning(f"SignalDetector: Invalid ofi_value type: {type(ofi_value)}")
                return None
            
            ofi_f = float(ofi_value)
            
            if not math.isfinite(ofi_f):
                logger.warning(f"SignalDetector: Non-finite ofi_value: {ofi_f}")
                return None
            
            # Enhanced threshold validation
            try:
                buy_threshold = float(self.config.ofi_buy_threshold)
                sell_threshold = float(self.config.ofi_sell_threshold)
                
                if not (math.isfinite(buy_threshold) and math.isfinite(sell_threshold)):
                    logger.warning(f"SignalDetector: Invalid thresholds - buy: {buy_threshold}, sell: {sell_threshold}")
                    return None
                
            except (ValueError, TypeError):
                logger.warning("SignalDetector: Error converting OFI thresholds")
                return None
            
            if ofi_f <= buy_threshold:
                if self._can_send_signal(symbol, "OFI"):
                    try:
                        await send_trading_signal(
                            symbol=symbol,
                            signal_type="BUY", 
                            indicator="OFI-z",
                            value=ofi_f,
                            reason=f"OFI extremely negative ({ofi_f:.2f} ≤ {buy_threshold})"
                        )
                        self._record_signal(symbol, "OFI")
                        return "BUY"
                    except Exception as e:
                        logger.error(f"SignalDetector: Error sending OFI BUY signal: {e}")
                        return None
            
            elif ofi_f >= sell_threshold:
                if self._can_send_signal(symbol, "OFI"):
                    try:
                        await send_trading_signal(
                            symbol=symbol,
                            signal_type="SELL",
                            indicator="OFI-z", 
                            value=ofi_f,
                            reason=f"OFI extremely positive ({ofi_f:.2f} ≥ {sell_threshold})"
                        )
                        self._record_signal(symbol, "OFI")
                        return "SELL"
                    except Exception as e:
                        logger.error(f"SignalDetector: Error sending OFI SELL signal: {e}")
                        return None
            
            return None
            
        except Exception as e:
            logger.error(f"SignalDetector.check_ofi_signal critical error: {e}")
            return None
    
    async def check_vpin_signal(self, symbol: str, vpin_value: float) -> Optional[str]:
        """Check for VPIN-based signals (high informed trading)"""
        if vpin_value >= self.config.vpin_threshold:
            if self._can_send_signal(symbol, "VPIN"):
                await send_trading_signal(
                    symbol=symbol,
                    signal_type="ALERT",
                    indicator="VPIN",
                    value=vpin_value,
                    reason=f"High informed trading detected ({vpin_value:.3f} ≥ {self.config.vpin_threshold})"
                )
                self._record_signal(symbol, "VPIN")
                return "ALERT"
        
        return None
    
    async def check_lpi_signal(self, symbol: str, lpi_value: float) -> Optional[str]:
        """Check for LPI-based signals (liquidation pressure)"""
        if lpi_value <= self.config.lpi_buy_threshold:
            if self._can_send_signal(symbol, "LPI"):
                await send_trading_signal(
                    symbol=symbol,
                    signal_type="BUY",
                    indicator="LPI",
                    value=lpi_value,
                    reason=f"Strong long liquidation pressure ({lpi_value:.3f} ≤ {self.config.lpi_buy_threshold})"
                )
                self._record_signal(symbol, "LPI")
                return "BUY"
        
        elif lpi_value >= self.config.lpi_sell_threshold:
            if self._can_send_signal(symbol, "LPI"):
                await send_trading_signal(
                    symbol=symbol,
                    signal_type="SELL", 
                    indicator="LPI",
                    value=lpi_value,
                    reason=f"Strong short liquidation pressure ({lpi_value:.3f} ≥ {self.config.lpi_sell_threshold})"
                )
                self._record_signal(symbol, "LPI")
                return "SELL"
        
        return None
    
    async def check_basic_oscillator_signals(self, symbol: str, oscillator_values: Dict[str, float]) -> List[str]:
        """Check Basic Oscillators for extreme signals (RSI-style approach)"""
        signals = []
        
        # Wall Ratio Oscillator - orderbook wall detection
        if "wall_ratio" in oscillator_values and oscillator_values["wall_ratio"] is not None:
            value = oscillator_values["wall_ratio"]
            if abs(value) >= self.config.wall_ratio_extreme:
                if self._can_send_signal(symbol, "WALL_RATIO", use_basic_cooldown=True):
                    signal_type = "SELL" if value > 0 else "BUY"  # Positive = ask wall dominance = bearish
                    await send_trading_signal(
                        symbol=symbol,
                        signal_type=signal_type,
                        indicator="Wall-Ratio-Z",
                        value=value,
                        reason=f"Extreme orderbook wall imbalance (z={value:.2f}, threshold=±{self.config.wall_ratio_extreme})"
                    )
                    self._record_signal(symbol, "WALL_RATIO")
                    signals.append(f"WALL-{signal_type}")
        
        # Trade Imbalance Oscillator - buy/sell pressure
        if "trade_imbalance" in oscillator_values and oscillator_values["trade_imbalance"] is not None:
            value = oscillator_values["trade_imbalance"]
            if abs(value) >= self.config.trade_imbalance_extreme:
                if self._can_send_signal(symbol, "TRADE_IMBALANCE", use_basic_cooldown=True):
                    signal_type = "BUY" if value > 0 else "SELL"  # Positive = buy pressure
                    await send_trading_signal(
                        symbol=symbol,
                        signal_type=signal_type,
                        indicator="Trade-Imbalance-Z",
                        value=value,
                        reason=f"Extreme trade flow imbalance (z={value:.2f}, threshold=±{self.config.trade_imbalance_extreme})"
                    )
                    self._record_signal(symbol, "TRADE_IMBALANCE")
                    signals.append(f"TRADE-{signal_type}")
        
        # Liquidation Oscillator - liquidation magnitude tracking
        if "liquidation" in oscillator_values and oscillator_values["liquidation"] is not None:
            value = oscillator_values["liquidation"]
            if abs(value) >= self.config.liquidation_extreme:
                if self._can_send_signal(symbol, "LIQUIDATION", use_basic_cooldown=True):
                    signal_type = "ALERT"  # Liquidations are generally warning signals
                    await send_trading_signal(
                        symbol=symbol,
                        signal_type=signal_type,
                        indicator="Liquidation-Z",
                        value=value,
                        reason=f"Extreme liquidation activity (z={value:.2f}, threshold=±{self.config.liquidation_extreme})"
                    )
                    self._record_signal(symbol, "LIQUIDATION")
                    signals.append(f"LIQ-{signal_type}")
        
        # DIR Oscillator - depth imbalance ratio
        if "dir" in oscillator_values and oscillator_values["dir"] is not None:
            value = oscillator_values["dir"]
            if abs(value) >= self.config.dir_extreme:
                if self._can_send_signal(symbol, "DIR", use_basic_cooldown=True):
                    signal_type = "BUY" if value < 0 else "SELL"  # Negative = bid depth dominance = bullish
                    await send_trading_signal(
                        symbol=symbol,
                        signal_type=signal_type,
                        indicator="DIR-Z",
                        value=value,
                        reason=f"Extreme depth imbalance (z={value:.2f}, threshold=±{self.config.dir_extreme})"
                    )
                    self._record_signal(symbol, "DIR")
                    signals.append(f"DIR-{signal_type}")
        
        # Funding Acceleration Oscillator - funding rate acceleration
        if "funding_accel" in oscillator_values and oscillator_values["funding_accel"] is not None:
            value = oscillator_values["funding_accel"]
            if abs(value) >= self.config.funding_accel_extreme:
                if self._can_send_signal(symbol, "FUNDING_ACCEL", use_basic_cooldown=True):
                    signal_type = "SELL" if value > 0 else "BUY"  # Positive acceleration = increasing long bias = contrarian sell
                    await send_trading_signal(
                        symbol=symbol,
                        signal_type=signal_type,
                        indicator="Funding-Accel-Z",
                        value=value,
                        reason=f"Extreme funding rate acceleration (z={value:.2f}, threshold=±{self.config.funding_accel_extreme})"
                    )
                    self._record_signal(symbol, "FUNDING_ACCEL")
                    signals.append(f"FUND-{signal_type}")
        
        return signals

    async def check_all_signals(self, symbol: str, indicators: Dict[str, float], basic_oscillators: Dict[str, float] = None):
        """Check all indicators for signals"""
        signals = []
        
        # Legacy advanced indicators
        if "ofi" in indicators and indicators["ofi"] is not None:
            signal = await self.check_ofi_signal(symbol, indicators["ofi"])
            if signal:
                signals.append(f"OFI-{signal}")
        
        if "vpin" in indicators and indicators["vpin"] is not None:
            signal = await self.check_vpin_signal(symbol, indicators["vpin"])
            if signal:
                signals.append(f"VPIN-{signal}")
        
        if "lpi" in indicators and indicators["lpi"] is not None:
            signal = await self.check_lpi_signal(symbol, indicators["lpi"])
            if signal:
                signals.append(f"LPI-{signal}")
        
        # New Basic Oscillators
        if basic_oscillators:
            osc_signals = await self.check_basic_oscillator_signals(symbol, basic_oscillators)
            signals.extend(osc_signals)
        
        return signals


# Global signal detector instance
signal_detector = SignalDetector()
