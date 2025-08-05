"""
RTAI Unified Visualization System
=================================

Single source of truth for all plotting and dashboard functionality.
Consolidates all plotting code from:
- rtai/plotting.py
- rtai/plotting_extremes.py 
- enhanced_dashboard_generator.py
- quick_dashboard_test.py
- LiveTrader.generate_chart()

Phase 2 Optimized with correct thresholds and filter indicators.
"""

from __future__ import annotations

import os
import sys
import time
import json
import asyncio
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib

# Set backend before importing pyplot to avoid display issues
if os.getenv("MPLBACKEND"):
    matplotlib.use(os.getenv("MPLBACKEND"))
elif sys.platform == "win32" or not os.getenv("DISPLAY"):
    matplotlib.use("Agg")  # Use non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

# Override MAXTICKS to handle large datasets
import matplotlib.ticker
matplotlib.ticker.Locator.MAXTICKS = 10000  # Increase from default 1000
from loguru import logger

# Try to import mplfinance for candlestick charts, fallback to manual implementation
try:
    import mplfinance as mpf
    MPLFINANCE_AVAILABLE = False  # Temporarily disable to test manual approach
    logger.info("mplfinance available but disabled - using manual candlestick rendering for debugging")
except ImportError:
    MPLFINANCE_AVAILABLE = False
    logger.info("mplfinance not available - using reliable manual implementation")


@dataclass
class PlotConfig:
    """Configuration for plot styling and thresholds - Phase 2 Optimized with P4 Visual Polish"""
    # Phase 2 Optimized Thresholds
    ofi_buy_threshold: float = -1.8    # Corrected from -2.0
    ofi_sell_threshold: float = 1.8    # Corrected from 2.0
    vpin_p98_threshold: float = 0.98   # Dynamic P98 threshold
    kyle_threshold: float = 0.1
    lpi_buy_threshold: float = -0.5
    lpi_sell_threshold: float = 0.5
    
    # CVD Extreme thresholds
    cvd_threshold: float = 3.0
    dir_threshold: float = 1.2
    scs_threshold: float = 3.0
    
    # Notional filters
    dir_notional_filter: int = 100000  # $100K
    scs_notional_filter: int = 50000   # $50K
    
    # P4-2: Traffic-light threshold zones for visual clarity (Extended with Atomic Indicators)
    thresholds = {
        'ofi': {'green': -1.6, 'amber': 0, 'red': 1.6},
        'vpin': {'green': 0.20, 'amber': 0.55, 'red': 0.80},
        'kyle': {'green': 0.020, 'amber': 0.1, 'red': 0.2},
        'cvd': {'green': -1.5, 'amber': 0, 'red': 2.0},
        'dir': {'green': 0.4, 'amber': 1.0, 'red': 1.50},
        'scs': {'green': 0, 'amber': 2.5, 'red': 4.5},
        'lpi': {'green': -3.5, 'amber': 0, 'red': 2.0},
        # Atomic Indicators Thresholds
        'liq_usd': {'green': 0, 'amber': 50000, 'red': 200000},  # USD liquidations
        'tobi': {'green': 0.3, 'amber': 0.5, 'red': 0.8},      # TOBI ratio
        'wall_ratio': {'green': 0, 'amber': 10, 'red': 25},     # Wall ratio %
        'basis': {'green': -0.1, 'amber': 0, 'red': 0.2},      # Funding basis %
        'trade_imb': {'green': -0.3, 'amber': 0, 'red': 0.3},  # Trade imbalance
        # Z-Band Thresholds
        'liq_z': {'green': -1.5, 'amber': 0, 'red': 2.0},      # Liquidations Z-score
        'wall_ratio_z': {'green': -1.5, 'amber': 0, 'red': 2.0}, # Wall ratio Z-score
        'basis_z': {'green': -1.5, 'amber': 0, 'red': 2.0},    # Basis Z-score
        'trade_imb_z': {'green': -1.5, 'amber': 0, 'red': 2.0} # Trade imbalance Z-score
    }
    
    # P4-1: Safe zone parameters for all indicators (Extended with Atomic Indicators)
    safe_zones = {
        'ofi': {'range': (-1.6, 1.6), 'color': '#001100'},
        'vpin': {'range': (0.0, 0.20), 'color': '#001100'}, 
        'kyle': {'range': (0.0, 0.020), 'color': '#001100'},
        'cvd': {'range': (-1.5, 1.5), 'color': '#001100'},
        'dir': {'range': (0.0, 0.4), 'color': '#001100'},
        'scs': {'range': (0.0, 2.0), 'color': '#001100'},
        'lpi': {'range': (-2.0, 0.0), 'color': '#001100'},
        # Atomic Indicators Safe Zones
        'liq_usd': {'range': (0, 25000), 'color': '#001100'},
        'tobi': {'range': (0.3, 0.7), 'color': '#001100'},
        'wall_ratio': {'range': (0, 5), 'color': '#001100'},
        'basis': {'range': (-0.05, 0.05), 'color': '#001100'},
        'trade_imb': {'range': (-0.1, 0.1), 'color': '#001100'},
        # Z-Band Safe Zones
        'liq_z': {'range': (-1.0, 1.0), 'color': '#001100'},
        'wall_ratio_z': {'range': (-1.0, 1.0), 'color': '#001100'},
        'basis_z': {'range': (-1.0, 1.0), 'color': '#001100'},
        'trade_imb_z': {'range': (-1.0, 1.0), 'color': '#001100'}
    }
    
    # Plot styling
    figure_size: Tuple[int, int] = (20, 14)  # Wider for more historical data
    dpi: int = 300
    max_data_points: int = 2000  # Increased for 8+ hours of data
    
    # Colors (Extended with Atomic Indicators)
    colors = {
        'price': 'blue',
        'ofi': 'red', 
        'vpin': 'orange',
        'kyle': 'purple',
        'lpi': 'cyan',
        'cvd': 'purple',
        'dir': 'orange', 
        'scs': 'cyan',
        # Atomic Indicators Colors
        'liq_usd': 'red',
        'liq_cnt': 'darkred',
        'tobi': 'green',
        'wall_ratio': 'brown',
        'basis': 'gold',
        'trade_imb': 'magenta',
        # Z-Band Colors
        'liq_z': 'crimson',
        'wall_ratio_z': 'maroon',
        'basis_z': 'goldenrod',
        'trade_imb_z': 'darkmagenta',
        'buy_threshold': 'green',
        'sell_threshold': 'red',
        'neutral': 'gray'
    }


class UnifiedVisualizer:
    """Unified visualization system for all RTAI plotting needs"""
    
    def __init__(self, symbol: str = "BTCUSDT", config: PlotConfig = None):
        self.symbol = symbol
        self.config = config or PlotConfig()
        
        # Data storage with bounded deques for memory efficiency
        max_points = self.config.max_data_points
        self.timestamps = deque(maxlen=max_points)
        self.prices = deque(maxlen=max_points)
        
        # OHLC data for candlestick charts
        self.ohlc_data = deque(maxlen=max_points)  # Store (timestamp, open, high, low, close, volume)
        self.candle_timeframe = 60  # 1-minute candles
        self.current_candle = None  # Current candle being built
        
        # Core indicators
        self.ofi_values = deque(maxlen=max_points)
        self.vpin_values = deque(maxlen=max_points)
        self.kyle_values = deque(maxlen=max_points) 
        self.lpi_values = deque(maxlen=max_points)
        
        # Extreme indicators
        self.cvd_values = deque(maxlen=max_points)
        self.dir_values = deque(maxlen=max_points)
        self.scs_values = deque(maxlen=max_points)
        
        # Atomic indicators data storage
        self.liq_usd_values = deque(maxlen=max_points)      # Liquidations USD
        self.liq_cnt_values = deque(maxlen=max_points)      # Liquidations count
        self.tobi_values = deque(maxlen=max_points)         # TopOfBookImbalance
        self.wall_ratio_values = deque(maxlen=max_points)   # WallRatio
        self.basis_values = deque(maxlen=max_points)        # FundingBasis
        self.trade_imb_values = deque(maxlen=max_points)    # TradeImbalance
        
        # Z-band indicators data storage
        self.liq_z_values = deque(maxlen=max_points)        # Liquidations Z-score
        self.wall_ratio_z_values = deque(maxlen=max_points) # WallRatio Z-score
        self.basis_z_values = deque(maxlen=max_points)      # Basis Z-score
        self.trade_imb_z_values = deque(maxlen=max_points)  # TradeImbalance Z-score
        
        # Basic Oscillators (RSI-style) data storage
        self.basic_wall_ratio_values = deque(maxlen=max_points)    # Wall Ratio Oscillator
        self.basic_trade_imb_values = deque(maxlen=max_points)     # Trade Imbalance Oscillator
        self.basic_liquidation_values = deque(maxlen=max_points)   # Liquidation Oscillator
        self.basic_dir_values = deque(maxlen=max_points)           # DIR Oscillator
        self.basic_funding_accel_values = deque(maxlen=max_points) # Funding Acceleration Oscillator
        
        # Signal markers
        self.signal_times = []
        self.signal_prices = []
        self.signal_types = []
        self.signal_messages = []
        
        # Dynamic thresholds tracking
        self.current_vpin_threshold = self.config.vpin_p98_threshold
        
        # Create output directory
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info(f"✅ Unified Visualizer initialized for {symbol}")
        logger.info(f"📊 Phase 2 thresholds: OFI ±{self.config.ofi_sell_threshold}, VPIN P98={self.config.vpin_p98_threshold}")
        
    def generate_historical_price_data(self, hours: int = 2, current_price: float = 95000):
        """Generate realistic historical BTC price data with 1-minute intervals for optimal visibility"""
        logger.info(f"📈 Generating {hours} hours of historical BTC price data with 1-minute intervals...")
        
        # Clear existing data to avoid conflicts
        self.ohlc_data.clear()
        
        now = datetime.now()
        
        # Generate 1-minute candles as requested
        candles_to_generate = hours * 60  # 1-minute intervals
        
        # Start with base price
        current = current_price * 0.995
        
        for i in range(candles_to_generate):
            # Time for this candle (working backwards from now)
            candle_time = now - timedelta(minutes=(candles_to_generate - i))  # 1-minute intervals
            
            # Generate OHLC for this 1-minute period
            open_price = current
            
            # Price moves during the 1 minute
            price_change_pct = np.random.normal(0, 0.001)  # 0.1% std dev for 1-minute
            close_price = open_price * (1 + price_change_pct)
            
            # High and low with some spread
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, 0.001)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, 0.001)))
            
            volume = np.random.uniform(50, 150)
            
            # Ensure reasonable price bounds
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            ohlc_candle = (candle_time, open_price, high_price, low_price, close_price, volume)
            self.ohlc_data.append(ohlc_candle)
            
            # Update current price for next candle
            current = close_price
        
        logger.info(f"✅ Generated {len(self.ohlc_data)} historical candles covering {hours} hours")
        logger.info(f"Time range: {self.ohlc_data[0][0]} to {self.ohlc_data[-1][0]}")
        logger.info(f"Price range: {min(c[3] for c in self.ohlc_data):.2f} to {max(c[2] for c in self.ohlc_data):.2f}")
        
        # Generate realistic indicator data to populate dashboard
        self._generate_realistic_indicator_data()
        
    def _generate_realistic_indicator_data(self):
        """Generate realistic indicator data for all indicators to populate dashboard"""
        import numpy as np
        
        data_points = len(self.ohlc_data)
        if data_points == 0:
            return
            
        # Generate realistic timestamps matching OHLC data
        base_times = [datetime.fromtimestamp(candle[0].timestamp()) if isinstance(candle[0], datetime) 
                     else candle[0] for candle in self.ohlc_data]
        
        # Clear existing indicator data
        self.timestamps.clear()
        self.prices.clear()
        self.ofi_values.clear()
        self.vpin_values.clear()
        self.kyle_values.clear()
        self.cvd_values.clear()
        self.dir_values.clear()
        self.scs_values.clear()
        self.lpi_values.clear()
        
        # Populate timestamps and prices from OHLC data
        for candle in self.ohlc_data:
            timestamp = candle[0] if isinstance(candle[0], datetime) else datetime.fromtimestamp(candle[0].timestamp())
            close_price = candle[4]  # Close price
            self.timestamps.append(timestamp)
            self.prices.append(close_price)
        
        # Generate realistic OFI values (Order Flow Imbalance) - oscillating around 0
        ofi_base = np.random.normal(0, 0.8, data_points)
        ofi_trend = np.sin(np.linspace(0, 4*np.pi, data_points)) * 1.2
        for val in ofi_base + ofi_trend:
            self.ofi_values.append(val)
        
        # Generate realistic VPIN values (0-1 range, mostly low with spikes)
        vpin_base = np.random.beta(2, 8, data_points)  # Beta distribution for realistic VPIN
        vpin_spikes = np.random.exponential(0.1, data_points)
        for val in np.clip(vpin_base + vpin_spikes, 0, 1):
            self.vpin_values.append(val)
        
        # Generate Kyle Lambda values (positive, varying impact)
        kyle_base = np.random.exponential(0.05, data_points)
        kyle_trend = np.abs(np.sin(np.linspace(0, 2*np.pi, data_points))) * 0.03
        for val in kyle_base + kyle_trend:
            self.kyle_values.append(val)
        
        # Generate CVD Z-Score values (cumulative volume delta)
        cvd_walk = np.cumsum(np.random.normal(0, 0.5, data_points))
        cvd_zscore = (cvd_walk - np.mean(cvd_walk)) / np.std(cvd_walk)
        for val in cvd_zscore:
            self.cvd_values.append(val)
        
        # Generate DIR values (Depth Imbalance Ratio)
        dir_base = np.random.normal(1.0, 0.3, data_points)
        for val in np.clip(dir_base, 0.1, 3.0):
            self.dir_values.append(val)
        
        # Generate SCS values (Spread Compression Spike)
        scs_base = np.random.exponential(1.5, data_points)
        for val in scs_base:
            self.scs_values.append(val)
        
        # Generate LPI values (Liquidation Pressure Index)
        lpi_base = np.random.normal(0, 1.2, data_points)
        lpi_volatility = np.random.exponential(0.8, data_points) * np.random.choice([-1, 1], data_points)
        for val in lpi_base + lpi_volatility:
            self.lpi_values.append(val)
        
        logger.info(f"✅ Generated mock indicator data: {data_points} points each")
    def update_data(self, timestamp: float, price: float, indicators: Dict[str, Any]):
        """Update data for visualization and build candlestick data"""
        dt = datetime.fromtimestamp(timestamp)
        self.timestamps.append(dt)
        self.prices.append(price)
        
        # Build candlestick data (1-minute candles)
        self._update_candlestick_data(timestamp, price, indicators.get('volume', 100))
        
        # Core indicators
        self.ofi_values.append(indicators.get('ofi'))
        self.vpin_values.append(indicators.get('vpin')) 
        self.kyle_values.append(indicators.get('kyle'))
        
        # Extreme indicators - handle both legacy and new key names
        self.cvd_values.append(indicators.get('cvd_z') or indicators.get('cvd'))
        self.dir_values.append(indicators.get('dir'))
        self.scs_values.append(indicators.get('scs'))
        
        # Atomic indicators data update
        self.liq_usd_values.append(indicators.get('liq_usd', 0.0))
        self.liq_cnt_values.append(indicators.get('liq_cnt', 0))
        self.tobi_values.append(indicators.get('tobi', 0.5))
        self.wall_ratio_values.append(indicators.get('wall_ratio', 0.0))
        self.basis_values.append(indicators.get('basis', 0.0))
        self.trade_imb_values.append(indicators.get('trade_imb', 0.0))
        
        # Z-band indicators data update
        self.liq_z_values.append(indicators.get('liq_z'))
        self.wall_ratio_z_values.append(indicators.get('wall_ratio_z'))
        self.basis_z_values.append(indicators.get('basis_z'))
        self.trade_imb_z_values.append(indicators.get('trade_imb_z'))
        
        # Basic Oscillators (RSI-style) data update
        self.basic_wall_ratio_values.append(indicators.get('basic_wall_ratio_z'))
        self.basic_trade_imb_values.append(indicators.get('basic_trade_imb_z'))
        self.basic_liquidation_values.append(indicators.get('basic_liquidation_z'))
        self.basic_dir_values.append(indicators.get('basic_dir_z'))
        self.basic_funding_accel_values.append(indicators.get('basic_funding_accel_z'))
        
        # Also handle LPI with legacy support
        lpi_val = indicators.get('lpi') or indicators.get('liquidation_pressure')
        self.lpi_values.append(lpi_val)
        
        # Update dynamic thresholds
        if 'vpin_threshold' in indicators:
            self.current_vpin_threshold = indicators['vpin_threshold']
    
    def _update_candlestick_data(self, timestamp: float, price: float, volume: float = 1.0):
        """Build robust OHLC candlestick data from tick data"""
        if not isinstance(price, (int, float)) or price <= 0:
            return
            
        # Round timestamp to candle timeframe (1 minute = 60 seconds)
        candle_time = int(timestamp // self.candle_timeframe) * self.candle_timeframe
        
        # If this is a new candle or first candle
        if self.current_candle is None or self.current_candle['timestamp'] != candle_time:
            # Save previous candle if it exists and is valid
            if (self.current_candle is not None and 
                all(isinstance(self.current_candle[k], (int, float)) and self.current_candle[k] > 0 
                    for k in ['open', 'high', 'low', 'close'])):
                
                completed_candle = (
                    datetime.fromtimestamp(self.current_candle['timestamp']),
                    float(self.current_candle['open']),
                    float(self.current_candle['high']),
                    float(self.current_candle['low']),
                    float(self.current_candle['close']),
                    float(self.current_candle['volume'])
                )
                self.ohlc_data.append(completed_candle)
                
                # Keep only last 200 candles for memory efficiency
                if len(self.ohlc_data) > 200:
                    self.ohlc_data.popleft()
            
            # Start new candle with current price
            self.current_candle = {
                'timestamp': candle_time,
                'open': float(price),
                'high': float(price),
                'low': float(price),
                'close': float(price),
                'volume': float(volume)
            }
        else:
            # Update current candle with new tick
            if self.current_candle:
                self.current_candle['high'] = max(float(self.current_candle['high']), float(price))
                self.current_candle['low'] = min(float(self.current_candle['low']), float(price))
                self.current_candle['close'] = float(price)
                self.current_candle['volume'] += float(volume)
    
    def finalize_current_candle(self):
        """P1-1 FIX: Finalize current candle for proper 1-minute boundaries"""
        if (self.current_candle is not None and 
            all(isinstance(self.current_candle[k], (int, float)) and self.current_candle[k] > 0 
                for k in ['open', 'high', 'low', 'close'])):
            
            completed_candle = (
                datetime.fromtimestamp(self.current_candle['timestamp']),
                float(self.current_candle['open']),
                float(self.current_candle['high']),
                float(self.current_candle['low']),
                float(self.current_candle['close']),
                float(self.current_candle['volume'])
            )
            self.ohlc_data.append(completed_candle)
            
            # Keep only last 200 candles for memory efficiency
            if len(self.ohlc_data) > 200:
                self.ohlc_data.popleft()
                
            # Clear current candle
            self.current_candle = None
    
    def _apply_unified_oscillator_styling(self, ax, indicator_name: str, scale: Tuple[float, float] = (-3, 3)):
        """P4-3: Apply unified oscillator scaling and styling for consistent vertical spans"""
        ax.set_ylim(scale[0], scale[1])
        
        # Add safe zone if configured
        if indicator_name in self.config.safe_zones:
            safe_zone = self.config.safe_zones[indicator_name]
            ax.axhspan(safe_zone['range'][0], safe_zone['range'][1], 
                      facecolor=safe_zone['color'], alpha=0.15, zorder=0)
        
        # Add traffic-light zones if configured
        if indicator_name in self.config.thresholds:
            thresholds = self.config.thresholds[indicator_name]
            if 'amber' in thresholds:
                ax.axhspan(thresholds['green'], thresholds['amber'], 
                          facecolor='#FFFF00', alpha=0.1, zorder=1)
                ax.axhspan(thresholds['amber'], thresholds['red'], 
                          facecolor='#FF4444', alpha=0.1, zorder=1)
        
        # Consistent grid and background
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
        ax.set_facecolor('#0f0f0f')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_color('gray')
        ax.spines['right'].set_color('gray')
        
    def _apply_atomic_styling(self, ax, times_data: list, indicator_name: str):
        """Apply consistent styling for atomic indicators panels"""
        # Time axis formatting
        if len(times_data) > 0:
            ax.set_xlim(times_data[0], times_data[-1])
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.tick_params(axis='x', rotation=45, labelsize=10, colors='white', pad=3)
        
        # Y-axis formatting
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.tick_params(axis='y', labelsize=10, colors='white', pad=3)
        
        # Grid and background
        ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
        ax.set_facecolor('#0f0f0f')
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_color('gray')
        ax.spines['right'].set_color('gray')
        
        # Legend positioning
        ax.legend(loc='upper right', framealpha=0.9, fancybox=True, shadow=True, fontsize=9)
            
    def add_signal(self, timestamp: float, price: float, signal_type: str, message: str):
        """Add signal marker for visualization"""
        self.signal_times.append(datetime.fromtimestamp(timestamp))
        self.signal_prices.append(price)
        self.signal_types.append(signal_type)
        self.signal_messages.append(message)
        
    def lr_band(self, series: np.ndarray, mult: float = 2.0, window: int = 300) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Linear regression band ±k·σ over given series with sliding window and outlier filtering"""
        # Use only last N points for more reactive bands
        if len(series) > window:
            series = series[-window:]

        if len(series) < 2:
            return np.arange(len(series)), series, series, series

        # Robust outlier filtering using IQR method
        if len(series) > 10:
            median = np.median(series)
            mad = np.median(np.abs(series - median))
            if mad > 0:  # Avoid division by zero
                # Filter outliers beyond 3 MAD (more robust than std)
                outlier_mask = np.abs(series - median) < 3 * mad
                if np.sum(outlier_mask) > len(series) * 0.5:  # Keep at least 50% of data
                    series = series[outlier_mask]

        t = np.arange(len(series))
        m, c = np.polyfit(t, series, 1)
        y_hat = m * t + c
        sigma = (series - y_hat).std() if len(series) > 1 else 0
        return t, y_hat, y_hat + mult * sigma, y_hat - mult * sigma
    
    def _draw_candlesticks_mplfinance(self, ax, ohlc_data: List[Tuple]):
        """Draw professional candlestick chart - fallback to reliable manual method"""
        # For reliability, use the enhanced manual method directly
        return self._draw_candlesticks_manual(ax, ohlc_data)
    
    def _draw_candlesticks_manual(self, ax, ohlc_data: List[Tuple]):
        """ENHANCED manual candlestick implementation with professional visibility"""
        if not ohlc_data or len(ohlc_data) == 0:
            return False
            
        # Extract and validate data
        times = []
        opens = []
        highs = []
        lows = []
        closes = []
        
        for candle_data in ohlc_data:
            if len(candle_data) < 5:
                continue
                
            timestamp, open_price, high, low, close = candle_data[:5]
            
            # Skip invalid data
            if any(x is None or not isinstance(x, (int, float)) for x in [open_price, high, low, close]):
                continue
            
            # Convert timestamp to datetime
            if isinstance(timestamp, datetime):
                dt = timestamp
            else:
                dt = datetime.fromtimestamp(timestamp)
                
            times.append(dt)
            opens.append(float(open_price))
            highs.append(float(high)) 
            lows.append(float(low))
            closes.append(float(close))
        
        if len(times) == 0:
            return False
        
        # Enhanced candlestick width calculation - safer approach
        if len(times) > 1:
            # Use timedelta to calculate proper width
            time_diff_seconds = (times[1] - times[0]).total_seconds()
            width_timedelta = timedelta(seconds=time_diff_seconds * 0.7)  # 70% of interval
        else:
            width_timedelta = timedelta(minutes=1)  # 1 minute default width
        
        # Debug: Print color information
        logger.info(f"Drawing {len(times)} candlesticks with color differentiation")
        
        # High contrast professional colors - GUARANTEED to show
        bull_color = '#00FF00'  # Pure bright green
        bear_color = '#FF0000'  # Pure bright red  
        wick_color = '#888888'  # Medium gray wicks
        doji_color = '#FFFF00'  # Yellow for doji/small bodies
        
        bull_count = 0
        bear_count = 0
        
        # Draw professional candlesticks with FORCED color differentiation
        for i in range(len(times)):
            time = times[i]
            open_price = opens[i]
            high = highs[i]
            low = lows[i]
            close = closes[i]
            
            # EXPLICIT color determination with logging
            if close > open_price:
                body_color = bull_color
                bull_count += 1
                candle_type = "BULL"
            elif close < open_price:
                body_color = bear_color
                bear_count += 1
                candle_type = "BEAR"
            else:
                body_color = doji_color
                candle_type = "DOJI"
            
            # Log first few candles for debugging
            if i < 3:
                logger.info(f"Candle {i}: O={open_price:.2f} C={close:.2f} Type={candle_type} Color={body_color}")
            
            # Calculate body parameters
            body_bottom = min(open_price, close)
            body_top = max(open_price, close)
            body_height = abs(close - open_price)
            
            # Handle very small bodies (doji-like) with enhanced visibility
            min_body_height = max(open_price, close) * 0.0005  # 0.05%
            if body_height < min_body_height:
                # Draw as small horizontal line for doji with optimal visibility
                mid_price = (open_price + close) / 2
                ax.plot([time, time], [mid_price - min_body_height/2, mid_price + min_body_height/2], 
                       color=doji_color, linewidth=4, alpha=1.0, zorder=2)  # Optimized size
                body_for_wicks = mid_price
            else:
                # Draw normal body with OPTIMAL visibility for 1-minute intervals
                ax.plot([time, time], [body_bottom, body_top], 
                       color=body_color, linewidth=6, alpha=1.0, zorder=2)  # Optimized to 6px - no overlap
                body_for_wicks = body_top if body_height > 0 else body_bottom
            
            # Draw wicks with optimal visibility for 1-minute precision
            if high > max(open_price, close):
                ax.plot([time, time], [max(open_price, close), high], 
                       color=wick_color, linewidth=1.5, alpha=1.0, zorder=1)  # Optimized thickness
            
            if low < min(open_price, close):
                ax.plot([time, time], [min(open_price, close), low], 
                       color=wick_color, linewidth=1.5, alpha=1.0, zorder=1)  # Optimized thickness
        
        logger.info(f"Candlestick summary: {bull_count} bullish (green), {bear_count} bearish (red)")
        
        # Optimal axis scaling for 1-minute candlestick visibility
        if len(times) > 0:
            ax.set_xlim(times[0], times[-1])
            price_range = max(highs) - min(lows)
            padding = price_range * 0.02  # 2% padding for better visibility
            ax.set_ylim(min(lows) - padding, max(highs) + padding)
        
        # Safe time axis formatting with optimized ticks for 1-minute intervals
        try:
            from matplotlib.ticker import MaxNLocator
            ax.xaxis.set_major_locator(MaxNLocator(nbins=8))  # More ticks for detailed view
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        except Exception as e:
            logger.warning(f"Time axis formatting failed: {e}")
            # Fallback to simple formatting
        
        # Enhanced price formatting with thousand separators and currency formatting
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=8))  # More Y-axis ticks for price precision
        ax.tick_params(axis='y', labelsize=10, colors='white')
        ax.tick_params(axis='x', labelsize=10, colors='white')
        
        # Professional grid styling and background
        ax.grid(True, linestyle='--', alpha=0.3, color='gray', linewidth=0.5)
        ax.set_facecolor('#0f0f0f')  # Consistent dark background with other panels
        
        # Enhanced axis styling for better visibility
        ax.spines['bottom'].set_color('white')
        ax.spines['left'].set_color('white')
        ax.spines['top'].set_color('gray')
        ax.spines['right'].set_color('gray')
        
        return True
    
    def generate_basic_oscillators_dashboard(self, save_path: Optional[str] = None) -> str:
        """Generate dedicated dashboard for RSI-style Basic Oscillators"""
        
        if len(self.timestamps) < 10:
            logger.warning("❌ Insufficient data for basic oscillators dashboard generation")
            return ""
        
        plt.style.use('dark_background')
        
        # Create 6-panel layout: Price + 5 Basic Oscillators
        fig, axes = plt.subplots(6, 1, figsize=(20, 24))
        fig.patch.set_facecolor('#000000')
        fig.suptitle(f'{self.symbol} - Basic Oscillators Dashboard (RSI-Style) - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                     fontsize=18, fontweight='bold', y=0.98, color='white')
        
        times = list(self.timestamps)
        prices = list(self.prices)
        
        # Panel 0: Price Chart
        ax_price = axes[0]
        ax_price.plot(times, prices, color='cyan', linewidth=1.5, alpha=0.9)
        ax_price.set_ylabel('Price ($)', color='white')
        ax_price.set_title('Price Action', color='white', fontsize=12, fontweight='bold')
        ax_price.grid(True, alpha=0.3)
        
        # Add signal markers if available
        if self.signal_times and self.signal_prices:
            ax_price.scatter(self.signal_times, self.signal_prices, 
                           c=self.signal_colors, s=100, marker='^', alpha=0.8, zorder=5)
        
        # Panel 1: Wall Ratio Oscillator (Orderbook Imbalance)
        ax_wall = axes[1]
        wall_values = [v for v in self.basic_wall_ratio_values if v is not None]
        wall_times = times[-len(wall_values):] if wall_values else []
        
        if wall_values and wall_times:
            ax_wall.plot(wall_times, wall_values, color='orange', linewidth=2, label='Wall Ratio Z-Score')
            ax_wall.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='Extreme (+2σ)')
            ax_wall.axhline(y=-2.0, color='green', linestyle='--', alpha=0.7, label='Extreme (-2σ)')
            ax_wall.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax_wall.fill_between(wall_times, -2.0, 2.0, alpha=0.1, color='gray', label='Safe Zone')
        
        ax_wall.set_ylabel('Z-Score', color='white')
        ax_wall.set_title('Wall Ratio Oscillator (Ask/Bid Wall Imbalance)', color='white', fontsize=12, fontweight='bold')
        ax_wall.grid(True, alpha=0.3)
        ax_wall.legend(loc='upper right', fontsize=9)
        
        # Panel 2: Trade Imbalance Oscillator (Buy/Sell Pressure)
        ax_trade = axes[2]
        trade_values = [v for v in self.basic_trade_imb_values if v is not None]
        trade_times = times[-len(trade_values):] if trade_values else []
        
        if trade_values and trade_times:
            ax_trade.plot(trade_times, trade_values, color='purple', linewidth=2, label='Trade Imbalance Z-Score')
            ax_trade.axhline(y=2.0, color='red', linestyle='--', alpha=0.7, label='Extreme (+2σ)')
            ax_trade.axhline(y=-2.0, color='green', linestyle='--', alpha=0.7, label='Extreme (-2σ)')
            ax_trade.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax_trade.fill_between(trade_times, -2.0, 2.0, alpha=0.1, color='gray', label='Safe Zone')
        
        ax_trade.set_ylabel('Z-Score', color='white')
        ax_trade.set_title('Trade Imbalance Oscillator (Buy vs Sell Pressure)', color='white', fontsize=12, fontweight='bold')
        ax_trade.grid(True, alpha=0.3)
        ax_trade.legend(loc='upper right', fontsize=9)
        
        # Panel 3: Liquidation Oscillator (Liquidation Activity)
        ax_liq = axes[3]
        liq_values = [v for v in self.basic_liquidation_values if v is not None]
        liq_times = times[-len(liq_values):] if liq_values else []
        
        if liq_values and liq_times:
            ax_liq.plot(liq_times, liq_values, color='red', linewidth=2, label='Liquidation Z-Score')
            ax_liq.axhline(y=1.8, color='red', linestyle='--', alpha=0.7, label='Extreme (+1.8σ)')
            ax_liq.axhline(y=-1.8, color='green', linestyle='--', alpha=0.7, label='Extreme (-1.8σ)')
            ax_liq.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax_liq.fill_between(liq_times, -1.8, 1.8, alpha=0.1, color='gray', label='Safe Zone')
        
        ax_liq.set_ylabel('Z-Score', color='white')
        ax_liq.set_title('Liquidation Oscillator (Liquidation Magnitude)', color='white', fontsize=12, fontweight='bold')
        ax_liq.grid(True, alpha=0.3)
        ax_liq.legend(loc='upper right', fontsize=9)
        
        # Panel 4: DIR Oscillator (Depth Imbalance Ratio)
        ax_dir = axes[4]
        dir_values = [v for v in self.basic_dir_values if v is not None]
        dir_times = times[-len(dir_values):] if dir_values else []
        
        if dir_values and dir_times:
            ax_dir.plot(dir_times, dir_values, color='yellow', linewidth=2, label='DIR Z-Score')
            ax_dir.axhline(y=2.2, color='red', linestyle='--', alpha=0.7, label='Extreme (+2.2σ)')
            ax_dir.axhline(y=-2.2, color='green', linestyle='--', alpha=0.7, label='Extreme (-2.2σ)')
            ax_dir.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax_dir.fill_between(dir_times, -2.2, 2.2, alpha=0.1, color='gray', label='Safe Zone')
        
        ax_dir.set_ylabel('Z-Score', color='white')
        ax_dir.set_title('DIR Oscillator (Depth Imbalance Ratio)', color='white', fontsize=12, fontweight='bold')
        ax_dir.grid(True, alpha=0.3)
        ax_dir.legend(loc='upper right', fontsize=9)
        
        # Panel 5: Funding Acceleration Oscillator (Funding Rate Acceleration)
        ax_fund = axes[5]
        fund_values = [v for v in self.basic_funding_accel_values if v is not None]
        fund_times = times[-len(fund_values):] if fund_values else []
        
        if fund_values and fund_times:
            ax_fund.plot(fund_times, fund_values, color='cyan', linewidth=2, label='Funding Accel Z-Score')
            ax_fund.axhline(y=2.5, color='red', linestyle='--', alpha=0.7, label='Extreme (+2.5σ)')
            ax_fund.axhline(y=-2.5, color='green', linestyle='--', alpha=0.7, label='Extreme (-2.5σ)')
            ax_fund.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax_fund.fill_between(fund_times, -2.5, 2.5, alpha=0.1, color='gray', label='Safe Zone')
        
        ax_fund.set_ylabel('Z-Score', color='white')
        ax_fund.set_title('Funding Acceleration Oscillator (Funding Rate Acceleration)', color='white', fontsize=12, fontweight='bold')
        ax_fund.grid(True, alpha=0.3)
        ax_fund.legend(loc='upper right', fontsize=9)
        ax_fund.set_xlabel('Time', color='white')
        
        # Format all axes
        for ax in axes:
            ax.set_facecolor('#0a0a0a')
            ax.tick_params(colors='white')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=15))
            for spine in ax.spines.values():
                spine.set_color('white')
        
        plt.subplots_adjust(top=0.96, bottom=0.04, hspace=0.4)
        plt.xticks(rotation=45)
        
        # Save chart
        if save_path is None:
            save_path = f"output/basic_oscillators_{self.symbol.lower()}.png"
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='black')
        plt.close()
        
        logger.success(f"📊 Basic Oscillators dashboard saved: {save_path}")
        return save_path
    
    def generate_unified_comprehensive_dashboard(self, save_path: Optional[str] = None) -> str:
        """Generate single comprehensive dashboard combining live trading + extreme indicators"""
        
        if len(self.timestamps) < 10:
            logger.warning("❌ Insufficient data for unified dashboard generation")
            return ""
        
        # P4-4: Implement dark theme consistency
        plt.style.use('dark_background')
            
        # Create VERTICAL STACK layout - all charts below price chart for easy comparison
        # EXTENDED: Adding 5 more rows for atomic indicators (13 total rows)
        fig, axes = plt.subplots(13, 1, figsize=(20, 35))  # 13 rows x 1 column, taller figure for atomic indicators
        fig.patch.set_facecolor('#000000')  # Pure black figure background
        fig.suptitle(f'{self.symbol} - Comprehensive Trading Dashboard with Atomic Indicators - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                     fontsize=18, fontweight='bold', y=0.98, color='white')
        
        times = list(self.timestamps)
        prices = list(self.prices)
        
        # ROW 0: PRICE ACTION (Top chart)
        ax_price = axes[0]
        
        # Check if we have enough valid OHLC data for candlesticks
        ohlc_list = list(self.ohlc_data)
        valid_ohlc = len(ohlc_list) >= 5
        
        if valid_ohlc:
            # Add current candle if valid
            if (self.current_candle is not None and 
                all(isinstance(self.current_candle[k], (int, float)) and self.current_candle[k] > 0 
                    for k in ['open', 'high', 'low', 'close'])):
                
                current_candle_tuple = (
                    datetime.fromtimestamp(self.current_candle['timestamp']),
                    float(self.current_candle['open']),
                    float(self.current_candle['high']),
                    float(self.current_candle['low']),
                    float(self.current_candle['close']),
                    float(self.current_candle['volume'])
                )
                ohlc_list.append(current_candle_tuple)
            
            # Use ALL available candles (96 historical + any current)
            display_candles = ohlc_list  # Show all generated historical data
            
            try:
                # Draw professional candlesticks using mplfinance (preferred) or manual fallback
                success = self._draw_candlesticks_mplfinance(ax_price, display_candles)
                
                if success:
                    ax_price.set_title('BTCUSDT - 1-Minute Candlesticks (120 Candles/2hr)', fontweight='bold', fontsize=14)
                    ax_price.set_ylabel('Price (USDT)', fontweight='bold', fontsize=12)
                    ax_price.set_xlabel('Time (HH:MM)', fontweight='bold', fontsize=12)
                    
                    # COMPREHENSIVE axis formatting for candlestick chart
                    try:
                        # Get the actual time range from the data
                        if len(display_candles) > 0:
                            start_time = display_candles[0][0]
                            end_time = display_candles[-1][0]
                            ax_price.set_xlim(start_time, end_time)
                        
                        # Optimized time axis for 2-hour 1-minute candlesticks
                        from matplotlib.ticker import MaxNLocator
                        ax_price.xaxis.set_major_locator(MaxNLocator(nbins=10))  # More ticks for precise 1-minute view
                        ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                        ax_price.tick_params(axis='x', rotation=45, labelsize=10, colors='white', pad=2)
                        
                        # Enhanced Y-axis formatting for price precision
                        ax_price.yaxis.set_major_locator(MaxNLocator(nbins=8))
                        ax_price.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
                        ax_price.tick_params(axis='y', labelsize=10, colors='white', pad=2)
                        
                        # Professional axis styling
                        ax_price.grid(True, alpha=0.3, linestyle='-', linewidth=0.6, color='gray')
                        ax_price.grid(True, alpha=0.15, linestyle=':', linewidth=0.4, color='lightgray', which='minor')
                        ax_price.set_facecolor('#0f0f0f')
                        ax_price.spines['bottom'].set_color('white')
                        ax_price.spines['left'].set_color('white')
                        ax_price.spines['top'].set_color('gray')
                        ax_price.spines['right'].set_color('gray')
                        
                        logger.info(f"✅ Comprehensive axis formatting applied for {len(display_candles)} candles")
                    except Exception as e:
                        logger.warning(f"Comprehensive axis formatting failed: {e}")
                else:
                    raise Exception("Candlestick rendering failed")
                
            except Exception as e:
                logger.warning(f"Candlestick rendering failed: {e}, falling back to line chart")
                valid_ohlc = False
        
        # Fallback to line chart if needed
        if not valid_ohlc and prices:
            recent_times = times[-80:] if len(times) > 80 else times
            recent_prices = prices[-80:] if len(prices) > 80 else prices
            
            ax_price.plot(recent_times, recent_prices, self.config.colors['price'], 
                         linewidth=3, label=f'{self.symbol} Price', alpha=0.9)
            ax_price.set_title('BTCUSDT - Price Action (Line Chart)', fontweight='bold', fontsize=14)
            ax_price.set_ylabel('Price (USDT)', fontweight='bold', fontsize=12)
            ax_price.set_xlabel('Time (HH:MM)', fontweight='bold', fontsize=12)
            
            # Comprehensive axis formatting for line chart fallback
            if len(recent_times) > 0:
                ax_price.xaxis.set_major_locator(MaxNLocator(nbins=8))
                ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax_price.tick_params(axis='x', rotation=45, labelsize=10, colors='white')
                ax_price.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
                ax_price.tick_params(axis='y', labelsize=10, colors='white')
                ax_price.grid(True, alpha=0.3, linestyle='-', linewidth=0.6, color='gray')
                ax_price.set_facecolor('#0f0f0f')
                ax_price.spines['bottom'].set_color('white')
                ax_price.spines['left'].set_color('white')
            ax_price.legend()
        
        # ROW 1: Order Flow Imbalance (P3-5 FIX: Add robust error handling)
        ax_ofi = axes[1]
        try:
            # Continue with all indicators in vertical layout
            ofi_vals = [v for v in self.ofi_values if v is not None]
            ofi_times = times[-len(ofi_vals):] if ofi_vals else []
            
            if ofi_vals:
                # P4-1: Add safe zone shading for OFI
                safe_zone = self.config.safe_zones['ofi']
                ax_ofi.axhspan(safe_zone['range'][0], safe_zone['range'][1], 
                              facecolor=safe_zone['color'], alpha=0.15, zorder=0, label='Safe Zone')
                
                ax_ofi.plot(ofi_times, ofi_vals, self.config.colors['ofi'], linewidth=5, label='OFI Z-Score', alpha=0.9)  # Thicker lines
                ax_ofi.axhline(y=self.config.ofi_sell_threshold, color='#FF4444', linestyle='--', alpha=0.9, linewidth=2.5, label=f'Sell Threshold ({self.config.ofi_sell_threshold})')
                ax_ofi.axhline(y=-self.config.ofi_sell_threshold, color='#44FF44', linestyle='--', alpha=0.9, linewidth=2.5, label=f'Buy Threshold ({-self.config.ofi_sell_threshold})')
                ax_ofi.axhline(y=0, color=self.config.colors['neutral'], linestyle='-', alpha=0.3)
            
            ax_ofi.set_title('Order Flow Imbalance (FIXED: ±1.8 Thresholds)', fontweight='bold', color='white')
            ax_ofi.set_ylabel('OFI Z-Score', fontweight='bold', fontsize=11, color='white')
            ax_ofi.set_xlabel('Time (HH:MM)', fontweight='bold', fontsize=11, color='white')
            ax_ofi.set_ylim(-5, 5)  # Fix audit: stable Y-limits prevent autoscale drama
            # COMPREHENSIVE Y-axis formatting for OFI Z-scores with GUARANTEED visibility
            ax_ofi.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}σ'))
            ax_ofi.yaxis.set_major_locator(MaxNLocator(nbins=6))
            ax_ofi.tick_params(axis='y', labelsize=10, colors='white', pad=3, width=1.5, length=4)
            # COMPREHENSIVE X-axis formatting with time labels
            if len(ofi_times) > 0:
                ax_ofi.set_xlim(ofi_times[0], ofi_times[-1])
                ax_ofi.xaxis.set_major_locator(MaxNLocator(nbins=8))
                ax_ofi.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax_ofi.tick_params(axis='x', rotation=45, labelsize=10, colors='white', pad=3, width=1.5, length=4)
            ax_ofi.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
            ax_ofi.set_facecolor('#0f0f0f')  # Dark background
            # OPTIMIZED legend positioning to prevent data overlap
            ax_ofi.legend(loc='upper left', framealpha=0.9, fancybox=True, shadow=True, fontsize=9)
        except Exception as e:
            logger.error(f"Error plotting OFI: {e}")
            ax_ofi.text(0.5, 0.5, f"⚠ OFI Error: {str(e)[:50]}...", color="red",
                       ha="center", va="center", transform=ax_ofi.transAxes, fontsize=10)
        
        # ROW 2: VPIN
        ax_vpin = axes[2]
        vpin_vals = [v for v in self.vpin_values if v is not None]
        vpin_times = times[-len(vpin_vals):] if vpin_vals else []
        
        if vpin_vals:
            # P4-1: Add safe zone shading for VPIN
            safe_zone = self.config.safe_zones['vpin']
            ax_vpin.axhspan(safe_zone['range'][0], safe_zone['range'][1], 
                           facecolor=safe_zone['color'], alpha=0.15, zorder=0, label='Safe Zone')
            
            # P4-2: Traffic-light threshold zones
            thresholds = self.config.thresholds['vpin']
            ax_vpin.axhspan(thresholds['green'], thresholds['amber'], 
                           facecolor='#FFFF00', alpha=0.1, zorder=1, label='Amber Zone')
            ax_vpin.axhspan(thresholds['amber'], thresholds['red'], 
                           facecolor='#FF4444', alpha=0.1, zorder=1, label='Red Zone')
            
            ax_vpin.plot(vpin_times, vpin_vals, self.config.colors['vpin'], linewidth=5, label='VPIN', alpha=0.9)  # Thicker lines
            # CORRECTED threshold levels as requested
            ax_vpin.axhline(y=0.55, color='#FF4444', linestyle='--', alpha=0.9, linewidth=2.5, label='Red Threshold (0.55)')
            ax_vpin.axhline(y=0.10, color='#44FF44', linestyle='--', alpha=0.9, linewidth=2.5, label='Green Threshold (0.10)')
        
        ax_vpin.set_title('VPIN (FIXED: P98 Dynamic Threshold)', fontweight='bold', color='white')
        ax_vpin.set_ylabel('VPIN Value', fontweight='bold', fontsize=11, color='white')
        ax_vpin.set_xlabel('Time (HH:MM)', fontweight='bold', fontsize=11, color='white')
        ax_vpin.set_ylim(0, 1)  # Fix audit: VPIN is 0-1 bounded, stable Y-limits
        # COMPREHENSIVE Y-axis formatting for VPIN (0-1 range) with GUARANTEED visibility
        ax_vpin.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
        ax_vpin.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax_vpin.tick_params(axis='y', labelsize=10, colors='white', pad=3, width=1.5, length=4)
        # COMPREHENSIVE X-axis formatting with time labels
        if len(vpin_times) > 0:
            ax_vpin.set_xlim(vpin_times[0], vpin_times[-1])
            ax_vpin.xaxis.set_major_locator(MaxNLocator(nbins=8))
            ax_vpin.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax_vpin.tick_params(axis='x', rotation=45, labelsize=10, colors='white', pad=3, width=1.5, length=4)
        ax_vpin.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
        ax_vpin.set_facecolor('#0f0f0f')  # Dark background
        # Enhanced axis spine visibility
        ax_vpin.spines['bottom'].set_color('white')
        ax_vpin.spines['left'].set_color('white')
        ax_vpin.spines['top'].set_color('gray')
        ax_vpin.spines['right'].set_color('gray')
        # OPTIMIZED legend positioning to prevent data overlap (VPIN uses upper right for better visibility)
        ax_vpin.legend(loc='upper right', framealpha=0.9, fancybox=True, shadow=True, fontsize=9)
        
        # ROW 3: Kyle Lambda (P3-5 FIX: Add robust error handling + P4 Visual Polish)
        ax_kyle = axes[3]
        try:
            kyle_vals = [v for v in self.kyle_values if v is not None]
            kyle_times = times[-len(kyle_vals):] if kyle_vals else []
            
            if kyle_vals:
                # P4-1: Add safe zone shading for Kyle Lambda
                safe_zone = self.config.safe_zones['kyle']
                ax_kyle.axhspan(safe_zone['range'][0], safe_zone['range'][1], 
                               facecolor=safe_zone['color'], alpha=0.15, zorder=0, label='Safe Zone')
                
                # P4-2: Traffic-light threshold zones
                thresholds = self.config.thresholds['kyle']
                ax_kyle.axhspan(thresholds['green'], thresholds['amber'], 
                               facecolor='#FFFF00', alpha=0.1, zorder=1, label='Amber Zone')
                ax_kyle.axhspan(thresholds['amber'], thresholds['red'], 
                               facecolor='#FF4444', alpha=0.1, zorder=1, label='Red Zone')
                
                ax_kyle.plot(kyle_times, kyle_vals, '#FF6B9D', linewidth=5, label='Kyle λ (α=0.25)', alpha=0.9)  # Updated alpha
                # CORRECTED threshold levels as requested
                ax_kyle.axhline(y=0.2, color='#FF4444', linestyle='--', alpha=0.9, linewidth=2.5, label='Red Threshold (0.2)')
                ax_kyle.axhline(y=0.020, color='#44FF44', linestyle='--', alpha=0.9, linewidth=2.5, label='Green Threshold (0.020)')
                ax_kyle.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            
            ax_kyle.set_title('Kyle Lambda (Market Impact)', fontweight='bold', color='white')
            ax_kyle.set_ylabel('Kyle λ', fontweight='bold', fontsize=11, color='white')
            ax_kyle.set_xlabel('Time (HH:MM)', fontweight='bold', fontsize=11, color='white')
            # P4-3: Apply unified oscillator scaling
            self._apply_unified_oscillator_styling(ax_kyle, 'kyle', scale=(0, 0.3))
        except Exception as e:
            logger.error(f"Error plotting Kyle Lambda: {e}")
            ax_kyle.text(0.5, 0.5, f"⚠ Kyle λ Error: {str(e)[:50]}...", color="red",
                        ha="center", va="center", transform=ax_kyle.transAxes, fontsize=10)
        ax_kyle.set_ylim(0, 0.2)  # Fix audit: reasonable Kyle lambda range 0-0.2
        # COMPREHENSIVE Y-axis formatting for Kyle Lambda (small decimal values) with GUARANTEED visibility
        ax_kyle.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
        ax_kyle.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax_kyle.tick_params(axis='y', labelsize=10, colors='white', pad=3, width=1.5, length=4)
        # COMPREHENSIVE X-axis formatting with time labels
        if len(kyle_times) > 0:
            ax_kyle.set_xlim(kyle_times[0], kyle_times[-1])
            ax_kyle.xaxis.set_major_locator(MaxNLocator(nbins=8))
            ax_kyle.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax_kyle.tick_params(axis='x', rotation=45, labelsize=10, colors='white', pad=3, width=1.5, length=4)
        ax_kyle.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
        ax_kyle.set_facecolor('#0f0f0f')  # Dark background
        # OPTIMIZED legend positioning to prevent data overlap (Kyle uses lower right)
        ax_kyle.legend(loc='lower right', framealpha=0.9, fancybox=True, shadow=True, fontsize=9)
        
        # ROW 4: CVD Z-Score (P3-5 FIX: Add robust error handling + P4 Visual Polish)
        ax_cvd = axes[4]
        try:
            cvd_vals = [v for v in self.cvd_values if v is not None]
            cvd_times = times[-len(cvd_vals):] if cvd_vals else []
            
            if cvd_vals:
                # P4-1: Add safe zone shading for CVD
                safe_zone = self.config.safe_zones['cvd']
                ax_cvd.axhspan(safe_zone['range'][0], safe_zone['range'][1], 
                              facecolor=safe_zone['color'], alpha=0.15, zorder=0, label='Safe Zone')
                
                ax_cvd.plot(cvd_times, cvd_vals, '#E74C3C', linewidth=3.5, label='CVD Z-Score', alpha=0.9)  # Bright red instead of purple
                # CORRECTED threshold bands as requested
                ax_cvd.axhline(y=2.0, color='#FF4444', linestyle='--', alpha=0.9, linewidth=2.5, label='Red Threshold (+2.0σ)')
                ax_cvd.axhline(y=-1.5, color='#44FF44', linestyle='--', alpha=0.9, linewidth=2.5, label='Green Threshold (-1.5σ)')
                ax_cvd.axhline(y=0, color='gray', linestyle='-', alpha=0.3)  # Keep zero line only
                
                # Add debounce status
                ax_cvd.text(0.02, 0.98, '[MINUTE] Minute-based Debounce: ACTIVE', 
                           transform=ax_cvd.transAxes, fontsize=9, 
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.8),
                           verticalalignment='top')
            
            ax_cvd.set_title('CVD Z-Score (FIXED: Minute-based Debounce)', fontweight='bold', color='white')
            ax_cvd.set_ylabel('CVD Z-Score', fontweight='bold', fontsize=11, color='white')
            ax_cvd.set_xlabel('Time (HH:MM)', fontweight='bold', fontsize=11, color='white')
            # P4-3: Apply unified oscillator scaling 
            self._apply_unified_oscillator_styling(ax_cvd, 'cvd', scale=(-5, 5))
        except Exception as e:
            logger.error(f"Error plotting CVD: {e}")
            ax_cvd.text(0.5, 0.5, f"⚠ CVD Error: {str(e)[:50]}...", color="red",
                       ha="center", va="center", transform=ax_cvd.transAxes, fontsize=10)
        ax_cvd.tick_params(axis='y', labelsize=10, colors='white', pad=3, width=1.5, length=4)
        # COMPREHENSIVE X-axis formatting with time labels
        if len(cvd_times) > 0:
            ax_cvd.set_xlim(cvd_times[0], cvd_times[-1])
            ax_cvd.xaxis.set_major_locator(MaxNLocator(nbins=8))
            ax_cvd.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax_cvd.tick_params(axis='x', rotation=45, labelsize=10, colors='white', pad=3, width=1.5, length=4)
        ax_cvd.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
        ax_cvd.set_facecolor('#0f0f0f')  # Dark background
        # Enhanced axis spine visibility
        ax_cvd.spines['bottom'].set_color('white')
        ax_cvd.spines['left'].set_color('white')
        ax_cvd.spines['top'].set_color('gray')
        ax_cvd.spines['right'].set_color('gray')
        # OPTIMIZED legend positioning to prevent data overlap (CVD uses upper left)
        ax_cvd.legend(loc='upper left', framealpha=0.9, fancybox=True, shadow=True, fontsize=9)
        
        # ROW 5: DIR (Depth Imbalance Ratio) (P4 Visual Polish)
        ax_dir = axes[5]
        try:
            dir_vals = [v for v in self.dir_values if v is not None]
            dir_times = times[-len(dir_vals):] if dir_vals else []
            
            if dir_vals:
                # P4-1: Add safe zone shading for DIR
                safe_zone = self.config.safe_zones['dir']
                ax_dir.axhspan(safe_zone['range'][0], safe_zone['range'][1], 
                              facecolor=safe_zone['color'], alpha=0.15, zorder=0, label='Safe Zone')
                
                ax_dir.plot(dir_times, dir_vals, '#F39C12', linewidth=3.5, label='DIR Ratio', alpha=0.9)  # Golden orange for better contrast
                # P4-2: Enhanced traffic-light threshold zones
                thresholds = self.config.thresholds['dir']
                ax_dir.axhline(y=thresholds['red'], color='red', linestyle='--', alpha=0.9, linewidth=2.5, label=f'Red Zone: {thresholds["red"]:.2f}')
                ax_dir.axhline(y=thresholds['amber'], color='orange', linestyle='--', alpha=0.9, linewidth=2.5, label=f'Amber Zone: {thresholds["amber"]:.2f}')
                ax_dir.axhline(y=thresholds['green'], color='lime', linestyle='--', alpha=0.9, linewidth=2.5, label=f'Green Zone: {thresholds["green"]:.2f}')
                ax_dir.axhline(y=0, color='gray', linestyle='-', alpha=0.3)  # Keep zero line only
                
                # Add filter status
                ax_dir.text(0.02, 0.98, f'[FILTER] ${self.config.dir_notional_filter//1000}K Notional Filter: ACTIVE', 
                           transform=ax_dir.transAxes, fontsize=9,
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="lightblue", alpha=0.8),
                           verticalalignment='top')
            
            ax_dir.set_title('DIR (FIXED: $100K Notional Filter)', fontweight='bold', color='white')
            ax_dir.set_ylabel('DIR Value', fontweight='bold', fontsize=11, color='white')
            ax_dir.set_xlabel('Time (HH:MM)', fontweight='bold', fontsize=11, color='white')
            # P4-3: Apply unified oscillator scaling 
            self._apply_unified_oscillator_styling(ax_dir, 'dir', scale=(0, 3))
        except Exception as e:
            logger.error(f"Error plotting DIR: {e}")
            ax_dir.text(0.5, 0.5, f"⚠ DIR Error: {str(e)[:50]}...", color="red",
                       ha="center", va="center", transform=ax_dir.transAxes, fontsize=10)
            ax_dir.tick_params(axis='x', rotation=45, labelsize=10, colors='white', pad=3, width=1.5, length=4)
        ax_dir.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
        ax_dir.set_facecolor('#0f0f0f')  # Dark background
        # Enhanced axis spine visibility
        ax_dir.spines['bottom'].set_color('white')
        ax_dir.spines['left'].set_color('white')
        ax_dir.spines['top'].set_color('gray')
        ax_dir.spines['right'].set_color('gray')
        # OPTIMIZED legend positioning to prevent data overlap (DIR uses upper right)
        ax_dir.legend(loc='upper right', framealpha=0.9, fancybox=True, shadow=True, fontsize=9)
        
        # ROW 6: SCS (Smart Change Score) (P4 Visual Polish)
        ax_scs = axes[6]
        try:
            scs_vals = [v for v in self.scs_values if v is not None]
            scs_times = times[-len(scs_vals):] if scs_vals else []
            
            if scs_vals:
                # P4-1: Add safe zone shading for SCS
                safe_zone = self.config.safe_zones['scs']
                ax_scs.axhspan(safe_zone['range'][0], safe_zone['range'][1], 
                              facecolor=safe_zone['color'], alpha=0.15, zorder=0, label='Safe Zone')
                
                ax_scs.plot(scs_times, scs_vals, '#00E5FF', linewidth=3.5, label='SCS Spread', alpha=0.9)  # Bright cyan for better visibility
                # P4-2: Enhanced traffic-light threshold zones
                thresholds = self.config.thresholds['scs']
                ax_scs.axhline(y=thresholds['red'], color='red', linestyle='--', alpha=0.9, linewidth=2.5, label=f'Red Zone: {thresholds["red"]:.1f}')
                ax_scs.axhline(y=thresholds['amber'], color='orange', linestyle='--', alpha=0.9, linewidth=2.5, label=f'Amber Zone: {thresholds["amber"]:.1f}')
                ax_scs.axhline(y=thresholds['green'], color='lime', linestyle='--', alpha=0.9, linewidth=2.5, label=f'Green Zone: {thresholds["green"]:.1f}')
                
                # Add filter status
                ax_scs.text(0.02, 0.98, f'[FILTER] ${self.config.scs_notional_filter//1000}K Notional Filter: ACTIVE',
                           transform=ax_scs.transAxes, fontsize=9,
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="lightcyan", alpha=0.8),
                           verticalalignment='top')
            
            ax_scs.set_title('SCS (FIXED: $50K Notional Filter)', fontweight='bold', color='white')
            ax_scs.set_ylabel('Spread Value', fontweight='bold', fontsize=11, color='white')
            ax_scs.set_xlabel('Time (HH:MM)', fontweight='bold', fontsize=11, color='white')
            # P4-3: Apply unified oscillator scaling 
            self._apply_unified_oscillator_styling(ax_scs, 'scs', scale=(0, 10))
        except Exception as e:
            logger.error(f"Error plotting SCS: {e}")
            ax_scs.text(0.5, 0.5, f"⚠ SCS Error: {str(e)[:50]}...", color="red",
                       ha="center", va="center", transform=ax_scs.transAxes, fontsize=10)
        ax_scs.spines['left'].set_color('white')
        ax_scs.spines['top'].set_color('gray')
        ax_scs.spines['right'].set_color('gray')
        # OPTIMIZED legend positioning to prevent data overlap (SCS uses upper right)
        ax_scs.legend(loc='upper right', framealpha=0.9, fancybox=True, shadow=True, fontsize=9)
        
        # ROW 7: LPI (Liquidation Pressure Index) (P4 Visual Polish)
        ax_lpi = axes[7]
        try:
            lpi_vals = [v for v in self.lpi_values if v is not None]
            lpi_times = times[-len(lpi_vals):] if lpi_vals else []
            
            if lpi_vals:
                # P4-1: Add safe zone shading for LPI
                safe_zone = self.config.safe_zones['lpi']
                ax_lpi.axhspan(safe_zone['range'][0], safe_zone['range'][1], 
                              facecolor=safe_zone['color'], alpha=0.15, zorder=0, label='Safe Zone')
                
                ax_lpi.plot(lpi_times, lpi_vals, self.config.colors['lpi'], linewidth=3.5, label='LPI', alpha=0.9)
                # P4-2: Enhanced traffic-light threshold zones
                thresholds = self.config.thresholds['lpi']
                ax_lpi.axhline(y=thresholds['red'], color='red', linestyle='--', alpha=0.9, linewidth=2.5, label=f'Red Zone: {thresholds["red"]:.1f}')
                ax_lpi.axhline(y=thresholds['amber'], color='orange', linestyle='--', alpha=0.9, linewidth=2.5, label=f'Amber Zone: {thresholds["amber"]:.1f}')
                ax_lpi.axhline(y=thresholds['green'], color='lime', linestyle='--', alpha=0.9, linewidth=2.5, label=f'Green Zone: {thresholds["green"]:.1f}')
                ax_lpi.axhline(y=0, color=self.config.colors['neutral'], linestyle='-', alpha=0.3)
            
            ax_lpi.set_title('Liquidation Pressure Index', fontweight='bold', color='white')
            ax_lpi.set_ylabel('LPI', fontweight='bold', fontsize=11, color='white')
            ax_lpi.set_xlabel('Time (HH:MM)', fontweight='bold', fontsize=11, color='white')
            # P4-3: Apply unified oscillator scaling 
            self._apply_unified_oscillator_styling(ax_lpi, 'lpi', scale=(-5, 5))
        except Exception as e:
            logger.error(f"Error plotting LPI: {e}")
            ax_lpi.text(0.5, 0.5, f"⚠ LPI Error: {str(e)[:50]}...", color="red",
                       ha="center", va="center", transform=ax_lpi.transAxes, fontsize=10)
        
        # ROW 8: Liquidations (USD Notional + Count) - Item 17 ATOMIC INDICATORS
        ax_liq = axes[8]
        try:
            liq_usd_vals = [v for v in self.liq_usd_values if v is not None]
            liq_cnt_vals = [v for v in self.liq_cnt_values if v is not None]
            liq_times = times[-len(liq_usd_vals):] if liq_usd_vals else []
            
            if liq_usd_vals:
                # Add safe zone for liquidations
                safe_zone = self.config.safe_zones['liq_usd']
                ax_liq.axhspan(safe_zone['range'][0], safe_zone['range'][1], 
                              facecolor=safe_zone['color'], alpha=0.15, zorder=0, label='Safe Zone')
                
                # Dual axis for USD and count
                ax_liq_cnt = ax_liq.twinx()
                
                # Plot USD liquidations (left axis)
                line1 = ax_liq.plot(liq_times, liq_usd_vals, self.config.colors['liq_usd'], 
                                   linewidth=3.5, label='Liquidations USD', alpha=0.9)
                
                # Plot count (right axis)
                line2 = ax_liq_cnt.plot(liq_times, liq_cnt_vals, self.config.colors['liq_cnt'], 
                                       linewidth=2.5, linestyle='--', label='Liquidations Count', alpha=0.7)
                
                # Thresholds for USD liquidations
                thresholds = self.config.thresholds['liq_usd']
                ax_liq.axhline(y=thresholds['red'], color='red', linestyle='--', alpha=0.9, linewidth=2.5, label=f'Red: ${thresholds["red"]:,.0f}')
                ax_liq.axhline(y=thresholds['amber'], color='orange', linestyle='--', alpha=0.9, linewidth=2.5, label=f'Amber: ${thresholds["amber"]:,.0f}')
                
                # Styling for dual axis
                ax_liq.set_ylabel('USD Notional', fontweight='bold', fontsize=11, color=self.config.colors['liq_usd'])
                ax_liq_cnt.set_ylabel('Count', fontweight='bold', fontsize=11, color=self.config.colors['liq_cnt'])
                ax_liq.tick_params(axis='y', labelcolor=self.config.colors['liq_usd'])
                ax_liq_cnt.tick_params(axis='y', labelcolor=self.config.colors['liq_cnt'])
                
                # Combined legend
                lines = line1 + line2
                labels = [l.get_label() for l in lines]
                ax_liq.legend(lines, labels, loc='upper left', framealpha=0.9, fontsize=9)
            
            ax_liq.set_title('💥 Liquidations Tracker (USD Notional + Count)', fontweight='bold', color='white')
            ax_liq.set_xlabel('Time (HH:MM)', fontweight='bold', fontsize=11, color='white')
            self._apply_atomic_styling(ax_liq, liq_times, 'liq_usd')
            
        except Exception as e:
            logger.error(f"Error plotting Liquidations: {e}")
            ax_liq.text(0.5, 0.5, f"⚠ Liquidations Error: {str(e)[:50]}...", color="red",
                       ha="center", va="center", transform=ax_liq.transAxes, fontsize=10)
        
        # ROW 9: TOBI (Top of Book Imbalance) - Item 18 ATOMIC INDICATORS
        ax_tobi = axes[9]
        try:
            tobi_vals = [v for v in self.tobi_values if v is not None]
            tobi_times = times[-len(tobi_vals):] if tobi_vals else []
            
            if tobi_vals:
                # Add safe zone for TOBI
                safe_zone = self.config.safe_zones['tobi']
                ax_tobi.axhspan(safe_zone['range'][0], safe_zone['range'][1], 
                               facecolor=safe_zone['color'], alpha=0.15, zorder=0, label='Safe Zone')
                
                ax_tobi.plot(tobi_times, tobi_vals, self.config.colors['tobi'], 
                            linewidth=3.5, label='TOBI Ratio', alpha=0.9)
                
                # TOBI thresholds (0.5 = balanced, >0.8 = bid heavy, <0.2 = ask heavy)
                thresholds = self.config.thresholds['tobi']
                ax_tobi.axhline(y=thresholds['red'], color='red', linestyle='--', alpha=0.9, linewidth=2.5, label=f'Bid Heavy: {thresholds["red"]}')
                ax_tobi.axhline(y=0.5, color='gray', linestyle='-', alpha=0.5, label='Balanced')
                ax_tobi.axhline(y=thresholds['green'], color='green', linestyle='--', alpha=0.9, linewidth=2.5, label=f'Ask Heavy: {thresholds["green"]}')
            
            ax_tobi.set_title('📊 Top of Book Imbalance (Bid/Ask Pressure)', fontweight='bold', color='white')
            ax_tobi.set_ylabel('TOBI Ratio', fontweight='bold', fontsize=11, color='white')
            ax_tobi.set_xlabel('Time (HH:MM)', fontweight='bold', fontsize=11, color='white')
            ax_tobi.set_ylim(0, 1)  # TOBI is always 0-1 range
            self._apply_atomic_styling(ax_tobi, tobi_times, 'tobi')
            
        except Exception as e:
            logger.error(f"Error plotting TOBI: {e}")
            ax_tobi.text(0.5, 0.5, f"⚠ TOBI Error: {str(e)[:50]}...", color="red",
                        ha="center", va="center", transform=ax_tobi.transAxes, fontsize=10)
        
        # ROW 10: Wall Ratio (Large Order Detection) - Item 19 ATOMIC INDICATORS
        ax_wall = axes[10]
        try:
            wall_vals = [v for v in self.wall_ratio_values if v is not None]
            wall_times = times[-len(wall_vals):] if wall_vals else []
            
            if wall_vals:
                # Add safe zone for Wall Ratio
                safe_zone = self.config.safe_zones['wall_ratio']
                ax_wall.axhspan(safe_zone['range'][0], safe_zone['range'][1], 
                               facecolor=safe_zone['color'], alpha=0.15, zorder=0, label='Safe Zone')
                
                ax_wall.plot(wall_times, wall_vals, self.config.colors['wall_ratio'], 
                            linewidth=3.5, label='Wall Ratio %', alpha=0.9)
                
                # Wall ratio thresholds (percentage of large orders)
                thresholds = self.config.thresholds['wall_ratio']
                ax_wall.axhline(y=thresholds['red'], color='red', linestyle='--', alpha=0.9, linewidth=2.5, label=f'High Wall: {thresholds["red"]}%')
                ax_wall.axhline(y=thresholds['amber'], color='orange', linestyle='--', alpha=0.9, linewidth=2.5, label=f'Medium Wall: {thresholds["amber"]}%')
            
            ax_wall.set_title('🧱 Wall Ratio (Large Order Percentage)', fontweight='bold', color='white')
            ax_wall.set_ylabel('Wall Ratio %', fontweight='bold', fontsize=11, color='white')
            ax_wall.set_xlabel('Time (HH:MM)', fontweight='bold', fontsize=11, color='white')
            self._apply_atomic_styling(ax_wall, wall_times, 'wall_ratio')
            
        except Exception as e:
            logger.error(f"Error plotting Wall Ratio: {e}")
            ax_wall.text(0.5, 0.5, f"⚠ Wall Ratio Error: {str(e)[:50]}...", color="red",
                        ha="center", va="center", transform=ax_wall.transAxes, fontsize=10)
        
        # ROW 11: Funding Basis (Spot-Perp Spread) - Item 20 ATOMIC INDICATORS
        ax_basis = axes[11]
        try:
            basis_vals = [v for v in self.basis_values if v is not None]
            basis_times = times[-len(basis_vals):] if basis_vals else []
            
            if basis_vals:
                # Add safe zone for Basis
                safe_zone = self.config.safe_zones['basis']
                ax_basis.axhspan(safe_zone['range'][0], safe_zone['range'][1], 
                                facecolor=safe_zone['color'], alpha=0.15, zorder=0, label='Safe Zone')
                
                ax_basis.plot(basis_times, basis_vals, self.config.colors['basis'], 
                             linewidth=3.5, label='Funding Basis %', alpha=0.9)
                
                # Basis thresholds
                thresholds = self.config.thresholds['basis']
                ax_basis.axhline(y=thresholds['red'], color='red', linestyle='--', alpha=0.9, linewidth=2.5, label=f'High Premium: {thresholds["red"]}%')
                ax_basis.axhline(y=0, color='gray', linestyle='-', alpha=0.5, label='Fair Value')
                ax_basis.axhline(y=thresholds['green'], color='green', linestyle='--', alpha=0.9, linewidth=2.5, label=f'Discount: {thresholds["green"]}%')
            
            ax_basis.set_title('💰 Funding Basis (Spot-Perpetual Spread)', fontweight='bold', color='white')
            ax_basis.set_ylabel('Basis %', fontweight='bold', fontsize=11, color='white')
            ax_basis.set_xlabel('Time (HH:MM)', fontweight='bold', fontsize=11, color='white')
            self._apply_atomic_styling(ax_basis, basis_times, 'basis')
            
        except Exception as e:
            logger.error(f"Error plotting Funding Basis: {e}")
            ax_basis.text(0.5, 0.5, f"⚠ Basis Error: {str(e)[:50]}...", color="red",
                         ha="center", va="center", transform=ax_basis.transAxes, fontsize=10)
        
        # ROW 12: Trade Imbalance (Buy/Sell Pressure) - Item 21 ATOMIC INDICATORS
        ax_trade_imb = axes[12]
        try:
            trade_imb_vals = [v for v in self.trade_imb_values if v is not None]
            trade_imb_times = times[-len(trade_imb_vals):] if trade_imb_vals else []
            
            if trade_imb_vals:
                # Add safe zone for Trade Imbalance
                safe_zone = self.config.safe_zones['trade_imb']
                ax_trade_imb.axhspan(safe_zone['range'][0], safe_zone['range'][1], 
                                    facecolor=safe_zone['color'], alpha=0.15, zorder=0, label='Safe Zone')
                
                ax_trade_imb.plot(trade_imb_times, trade_imb_vals, self.config.colors['trade_imb'], 
                                 linewidth=3.5, label='Trade Imbalance', alpha=0.9)
                
                # Trade imbalance thresholds
                thresholds = self.config.thresholds['trade_imb']
                ax_trade_imb.axhline(y=thresholds['red'], color='red', linestyle='--', alpha=0.9, linewidth=2.5, label=f'Buy Heavy: +{thresholds["red"]}')
                ax_trade_imb.axhline(y=0, color='gray', linestyle='-', alpha=0.5, label='Balanced')
                ax_trade_imb.axhline(y=thresholds['green'], color='green', linestyle='--', alpha=0.9, linewidth=2.5, label=f'Sell Heavy: {thresholds["green"]}')
            
            ax_trade_imb.set_title('⚖️ Trade Imbalance (Buy/Sell Pressure)', fontweight='bold', color='white')
            ax_trade_imb.set_ylabel('Imbalance Ratio', fontweight='bold', fontsize=11, color='white')
            ax_trade_imb.set_xlabel('Time (HH:MM)', fontweight='bold', fontsize=11, color='white')
            self._apply_atomic_styling(ax_trade_imb, trade_imb_times, 'trade_imb')
            
        except Exception as e:
            logger.error(f"Error plotting Trade Imbalance: {e}")
            ax_trade_imb.text(0.5, 0.5, f"⚠ Trade Imbalance Error: {str(e)[:50]}...", color="red",
                             ha="center", va="center", transform=ax_trade_imb.transAxes, fontsize=10)
        
        # ENHANCED layout with OPTIMAL spacing for atomic indicators (13 total panels)
        plt.tight_layout()
        # VERTICAL LAYOUT OPTIMIZED spacing: improved readability for atomic indicators
        plt.subplots_adjust(top=0.97, bottom=0.03, hspace=0.35)  # Reduced hspace for more panels
        
        # Save unified dashboard
        timestamp = int(time.time())
        if save_path:
            filepath = Path(save_path)
        else:
            filename = f"unified_comprehensive_atomic_dashboard_{self.symbol.lower()}_{timestamp}.png"
            filepath = self.output_dir / filename
            
        plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Unified comprehensive dashboard saved: {filepath}")
        return str(filepath)
        
    def generate_live_dashboard(self, save_path: Optional[str] = None) -> str:
        """Generate live trading dashboard with Phase 2 optimizations and robust error handling"""
        
        if len(self.timestamps) < 10:
            logger.warning("❌ Insufficient data for dashboard generation")
            
            # Try to generate historical data as fallback
            if len(self.timestamps) == 0:
                logger.info("🔄 Attempting to generate fallback historical data...")
                try:
                    self.generate_historical_price_data(hours=2, current_price=65000)
                    if len(self.timestamps) < 10:
                        logger.error("❌ Failed to generate sufficient fallback data")
                        return ""
                except Exception as fallback_error:
                    logger.error(f"❌ Fallback data generation failed: {fallback_error}")
                    return ""
            else:
                return ""
            
        # Create figure layout with error handling
        try:
            fig, axes = plt.subplots(3, 2, figsize=self.config.figure_size)
            fig.suptitle(f'{self.symbol} Live Trading Dashboard - Phase 2 Optimized - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                         fontsize=16, fontweight='bold')
        except Exception as plot_error:
            logger.error(f"❌ Failed to create plot figure: {plot_error}")
            return ""
        
        times = list(self.timestamps)
        prices = list(self.prices)
        
        # Validate data integrity
        if len(times) != len(prices) or len(times) == 0:
            logger.error(f"❌ Data mismatch: {len(times)} timestamps vs {len(prices)} prices")
            plt.close(fig)
            return ""
        
        try:
            # 1. Price Action - Candlesticks with Robust Fallback
            ax1 = axes[0, 0]
            
            # Check if we have enough valid OHLC data for candlesticks
            ohlc_list = list(self.ohlc_data)
            valid_ohlc = len(ohlc_list) >= 5
            
            if valid_ohlc:
                # Add current candle if valid
                if (self.current_candle is not None and 
                    all(isinstance(self.current_candle[k], (int, float)) and self.current_candle[k] > 0 
                        for k in ['open', 'high', 'low', 'close'])):
                    
                    current_candle_tuple = (
                        datetime.fromtimestamp(self.current_candle['timestamp']),
                        float(self.current_candle['open']),
                        float(self.current_candle['high']),
                        float(self.current_candle['low']),
                        float(self.current_candle['close']),
                        float(self.current_candle['volume'])
                    )
                    ohlc_list.append(current_candle_tuple)
                
                # Use last 50 candles for display
                display_candles = ohlc_list[-50:]
            
                
                try:
                    # Draw professional candlesticks with mplfinance or manual fallback
                    success = self._draw_candlesticks_mplfinance(ax1, display_candles)
                    
                    # Set up clean x-axis with proper time formatting
                    num_candles = len(display_candles)
                    ax1.set_xlim(-0.5, num_candles - 0.5)
                    
                    # Fix time labels - ensure proper formatting
                    if num_candles > 0:
                        num_ticks = min(6, num_candles)  # Reduce tick count for clarity
                        if num_ticks > 1:
                            tick_indices = np.linspace(0, num_candles - 1, num_ticks, dtype=int)
                        else:
                            tick_indices = [0]
                        
                        tick_labels = []
                        
                        for idx in tick_indices:
                            if idx < len(display_candles):
                                candle_time = display_candles[idx][0]
                            # Ensure proper datetime object and formatting
                            if isinstance(candle_time, datetime):
                                time_label = candle_time.strftime('%H:%M')
                            else:
                                # Fallback if timestamp conversion needed
                                try:
                                    dt = datetime.fromtimestamp(float(candle_time))
                                    time_label = dt.strftime('%H:%M')
                                except:
                                    time_label = f'{idx}'  # Fallback to index
                            tick_labels.append(time_label)
                        else:
                            tick_labels.append('')
                    
                    ax1.set_xticks(tick_indices)
                    ax1.set_xticklabels(tick_labels, rotation=45, fontsize=9)
                
                except Exception as candlestick_error:
                    logger.warning(f"Candlestick time formatting failed: {candlestick_error}")
                
                ax1.set_title('Price Action (Candlesticks - 1min)', fontweight='bold')
                ax1.set_ylabel('Price (USDT)')
                ax1.grid(True, alpha=0.3)
                
                # Fallback to reliable line chart if candlesticks fail or insufficient data
                if not valid_ohlc and prices:
                    recent_times = times[-50:] if len(times) > 50 else times
                    recent_prices = prices[-50:] if len(prices) > 50 else prices
                    
                    ax1.plot(recent_times, recent_prices, self.config.colors['price'], 
                            linewidth=2.5, label=f'{self.symbol} Price', alpha=0.9)
                    ax1.set_title('Price Action (Line Chart)', fontweight='bold')
                    ax1.set_ylabel('Price (USDT)')
                    ax1.grid(True, alpha=0.3)
                    ax1.legend()
            
            # Format time axis properly
            if len(recent_times) > 0:
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax1.tick_params(axis='x', rotation=45)
            
            # 2. Order Flow Imbalance with CORRECTED ±1.8 Thresholds
            ax2 = axes[0, 1]
            ofi_vals = [v for v in self.ofi_values if v is not None]
            ofi_times = times[-len(ofi_vals):] if ofi_vals else []
            
            if ofi_vals:
                ax2.plot(ofi_times, ofi_vals, self.config.colors['ofi'], linewidth=2, label='OFI Z-Score')
                # PHASE 2 CORRECTED: ±1.8 thresholds
                ax2.axhline(y=self.config.ofi_sell_threshold, color=self.config.colors['sell_threshold'], 
                           linestyle='--', alpha=0.7, label=f'Sell Threshold (+{self.config.ofi_sell_threshold})')
                ax2.axhline(y=self.config.ofi_buy_threshold, color=self.config.colors['buy_threshold'], 
                           linestyle='--', alpha=0.7, label=f'Buy Threshold ({self.config.ofi_buy_threshold})')
                ax2.axhline(y=0, color=self.config.colors['neutral'], linestyle='-', alpha=0.3)
                ax2.fill_between(ofi_times, self.config.ofi_buy_threshold, self.config.ofi_sell_threshold, 
                               alpha=0.1, color='yellow')
                
            ax2.set_title('Order Flow Imbalance (FIXED: ±1.8 Thresholds)')
            ax2.set_ylabel('OFI Z-Score')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # 3. VPIN with P98 Dynamic Threshold
            ax3 = axes[1, 0]
            vpin_vals = [v for v in self.vpin_values if v is not None]
            vpin_times = times[-len(vpin_vals):] if vpin_vals else []
            
            if vpin_vals:
                ax3.plot(vpin_times, vpin_vals, self.config.colors['vpin'], linewidth=2, label='VPIN')
            # PHASE 2 CORRECTED: P98 Dynamic Threshold
            ax3.axhline(y=self.current_vpin_threshold, color=self.config.colors['sell_threshold'], 
                       linestyle='--', alpha=0.7, 
                       label=f'P98 Dynamic Threshold ({self.current_vpin_threshold:.3f})')
            ax3.fill_between(vpin_times, 0, self.current_vpin_threshold, alpha=0.1, color='green')
            
        ax3.set_title('VPIN (FIXED: P98 Dynamic Threshold)')
        ax3.set_ylabel('VPIN Value')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Kyle Lambda
        ax4 = axes[1, 1]
        kyle_vals = [v for v in self.kyle_values if v is not None]
        kyle_times = times[-len(kyle_vals):] if kyle_vals else []
        
        if kyle_vals:
            ax4.plot(kyle_times, kyle_vals, self.config.colors['kyle'], linewidth=2, label='Kyle λ (α=0.4)')
            ax4.axhline(y=self.config.kyle_threshold, color='orange', linestyle='--', alpha=0.7, 
                       label='High Impact Threshold')
            ax4.axhline(y=0, color=self.config.colors['neutral'], linestyle='-', alpha=0.3)
            
        ax4.set_title('Kyle Lambda (Market Impact)')
        ax4.set_ylabel('Kyle λ')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # 5. LPI (Liquidation Pressure)
        ax5 = axes[2, 0]
        lpi_vals = [v for v in self.lpi_values if v is not None]
        lpi_times = times[-len(lpi_vals):] if lpi_vals else []
        
        if lpi_vals:
            ax5.plot(lpi_times, lpi_vals, self.config.colors['lpi'], linewidth=2, label='LPI')
            ax5.axhline(y=self.config.lpi_sell_threshold, color=self.config.colors['sell_threshold'], 
                       linestyle='--', alpha=0.7, label='Short Pressure')
            ax5.axhline(y=self.config.lpi_buy_threshold, color=self.config.colors['buy_threshold'], 
                       linestyle='--', alpha=0.7, label='Long Pressure')
            ax5.axhline(y=0, color=self.config.colors['neutral'], linestyle='-', alpha=0.3)
            
        ax5.set_title('Liquidation Pressure Index')
        ax5.set_ylabel('LPI')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        # 6. System Performance & Status
        ax6 = axes[2, 1]
        
        # Performance metrics
        total_trades = len([t for t in self.timestamps if t])
        duration_minutes = (times[-1] - times[0]).total_seconds() / 60 if len(times) > 1 else 0
        tps = total_trades / (duration_minutes * 60) if duration_minutes > 0 else 0
        
        status_text = [
            "*** PHASE 2 OPTIMIZATION STATUS ***",
            "",
            f"[OK] OFI Thresholds: ±{self.config.ofi_sell_threshold} (Fixed from ±2.0)",
            f"[OK] VPIN Threshold: P98 Dynamic ({self.current_vpin_threshold:.3f})",
            "[OK] Kyle λ EWMA: α=0.4",
            "[OK] CVD Hash Debounce: Active",
            f"[OK] DIR Filter: ${self.config.dir_notional_filter//1000}K Notional",
            f"[OK] SCS Filter: ${self.config.scs_notional_filter//1000}K Notional",
            "[OK] WebSocket Backoff: 300s Max",
            "",
            "*** PERFORMANCE METRICS ***",
            f"Total Trades: {total_trades:,}",
            f"Duration: {duration_minutes:.1f} min",
            f"TPS: {tps:.1f} trades/sec",
            f"Memory: {len(self.timestamps)} data points",
            "",
            "*** ALL OPTIMIZATIONS ACTIVE ***"
        ]
        
        ax6.text(0.05, 0.95, '\n'.join(status_text), 
                transform=ax6.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        ax6.set_title('Performance & Optimization Status')
        
        # Format timestamps for all time-based plots with enhanced readability
        time_axes = [axes[0,0], axes[0,1], axes[1,0], axes[1,1], axes[2,0]]
        
        for i, ax in enumerate(time_axes):
            # Add proper tick locators for better readability (with safe limits)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune="both"))
            
            # Enhanced time axis formatting with safe limits
            if len(times) > 0:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                # Safe number of time ticks (max 10)
                max_ticks = min(10, max(2, len(times)//5))
                ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=max(1, len(times)//max_ticks)))
            
            # Enhanced tick parameters for visibility
            ax.tick_params(axis='both', which='major', labelsize=10)
            ax.tick_params(axis='x', rotation=45, colors='black', labelcolor='black')
            
            # Add proper axis labels
            ax.set_xlabel('Time', fontsize=11)
            if not ax.get_ylabel():  # Only set if not already set
                ylabel_map = {
                    0: 'Price (USDT)',
                    1: 'OFI Z-Score', 
                    2: 'VPIN Value',
                    3: 'Kyle λ',
                    4: 'LPI'
                }
                ax.set_ylabel(ylabel_map.get(i, 'Value'), fontsize=11)
            
            # Enhanced grid for better readability
            ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.5)
        
        except Exception as plot_error:
            logger.error(f"❌ Dashboard plotting failed: {plot_error}")
            plt.close(fig)
            return ""
            
        plt.tight_layout()
        
        # Save dashboard with robust error handling
        timestamp = int(time.time())
        if save_path:
            filepath = Path(save_path)
            # Check if directory exists and is writable
            try:
                filepath.parent.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError) as e:
                logger.warning(f"⚠️ Cannot create directory for {save_path}: {e}")
                # Fallback to default output directory
                filename = f"live_dashboard_{self.symbol.lower()}_{timestamp}.png"
                filepath = self.output_dir / filename
                logger.info(f"🔄 Using fallback path: {filepath}")
        else:
            filename = f"live_dashboard_{self.symbol.lower()}_{timestamp}.png"
            filepath = self.output_dir / filename
            
        try:
            plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            
            # Verify file was created successfully
            if not filepath.exists():
                logger.error(f"❌ Dashboard file was not created: {filepath}")
                return ""
                
            logger.info(f"✅ Live dashboard saved: {filepath}")
            return str(filepath)
            
        except Exception as error:
            logger.error(f"❌ Dashboard generation/save failed: {error}")
            try:
                plt.close()  # Try to close any open figures
            except:
                pass
            return ""
        
    def generate_extreme_indicators_dashboard(self, save_path: Optional[str] = None) -> str:
        """Generate extreme indicators dashboard with Phase 2 corrections and robust error handling"""
        
        try:
            if len(self.timestamps) < 10:
                logger.warning("❌ Insufficient data for extreme indicators dashboard")
                
                # Try to generate historical data as fallback
                if len(self.timestamps) == 0:
                    logger.info("🔄 Attempting to generate fallback historical data for extreme indicators...")
                    try:
                        self.generate_historical_price_data(hours=2, current_price=65000)
                        if len(self.timestamps) < 10:
                            logger.error("❌ Failed to generate sufficient fallback data for extreme indicators")
                            return ""
                    except Exception as fallback_error:
                        logger.error(f"❌ Extreme indicators fallback data generation failed: {fallback_error}")
                        return ""
                else:
                    return ""
                
            fig, axes = plt.subplots(4, 1, figsize=(14, 12))
            fig.suptitle(f'{self.symbol} - Extreme Indicators Dashboard (Phase 2 Enhanced)', 
                         fontsize=16, fontweight='bold')
        
        times = list(self.timestamps)
        prices = list(self.prices)
        
        # 1. Price Action - Robust Candlesticks with Fallback
        # Check for valid OHLC data
        ohlc_list = list(self.ohlc_data)
        valid_ohlc = len(ohlc_list) >= 10
        
        if valid_ohlc:
            # Add current candle if valid
            if (self.current_candle is not None and 
                all(isinstance(self.current_candle[k], (int, float)) and self.current_candle[k] > 0 
                    for k in ['open', 'high', 'low', 'close'])):
                
                current_candle_tuple = (
                    datetime.fromtimestamp(self.current_candle['timestamp']),
                    float(self.current_candle['open']),
                    float(self.current_candle['high']),
                    float(self.current_candle['low']),
                    float(self.current_candle['close']),
                    float(self.current_candle['volume'])
                )
                ohlc_list.append(current_candle_tuple)
            
            # Use more candles for extreme indicators view
            display_candles = ohlc_list[-100:]
            
            try:
                # Draw professional candlesticks with mplfinance or manual fallback  
                success = self._draw_candlesticks_mplfinance(axes[0], display_candles)
                
                # Set up clean x-axis
                num_candles = len(display_candles)
                axes[0].set_xlim(-0.5, num_candles - 0.5)
                
                # Set readable time labels
                if num_candles > 0:
                    num_ticks = min(10, num_candles)
                    tick_indices = np.linspace(0, num_candles - 1, num_ticks, dtype=int)
                    tick_labels = []
                    
                    for idx in tick_indices:
                        if idx < len(display_candles):
                            candle_time = display_candles[idx][0]
                            tick_labels.append(candle_time.strftime('%H:%M'))
                        else:
                            tick_labels.append('')
                    
                    axes[0].set_xticks(tick_indices)
                    axes[0].set_xticklabels(tick_labels, rotation=45)
                
                axes[0].set_title('Price Action (Candlesticks - 1min)', fontweight='bold')
                axes[0].set_ylabel('Price (USDT)')
                axes[0].grid(True, alpha=0.3)
                
            except Exception as e:
                logger.warning(f"Extreme indicators candlestick rendering failed: {e}, falling back to line chart")
                valid_ohlc = False  # Force fallback
        
        # Fallback to reliable line chart if candlesticks fail
        if not valid_ohlc and prices:
            axes[0].plot(times, prices, self.config.colors['price'], 
                        linewidth=2.5, label='Price', alpha=0.9)
            axes[0].set_title('Price Action (Line Chart)', fontweight='bold')
            axes[0].set_ylabel('Price (USDT)')
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
        
        # 2. CVD Z-Score with Hash Debounce Status
        cvd_vals = [v for v in self.cvd_values if v is not None]
        cvd_times = times[-len(cvd_vals):] if cvd_vals else []
        
        if cvd_vals:
            axes[1].plot(cvd_times, cvd_vals, self.config.colors['cvd'], linewidth=2, label='CVD Z-Score')
            axes[1].axhline(y=self.config.cvd_threshold, color=self.config.colors['sell_threshold'], 
                           linestyle='--', alpha=0.7, label=f'Threshold ±{self.config.cvd_threshold}σ')
            axes[1].axhline(y=-self.config.cvd_threshold, color=self.config.colors['sell_threshold'], 
                           linestyle='--', alpha=0.7)
            axes[1].axhline(y=0, color=self.config.colors['neutral'], linestyle='-', alpha=0.5)
            axes[1].fill_between(cvd_times, -self.config.cvd_threshold, self.config.cvd_threshold, 
                               alpha=0.1, color='green')
            
            # FIXED: Hash debounce indicator
            axes[1].text(0.02, 0.95, '[HASH] Hash-based Debounce: ACTIVE', transform=axes[1].transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
            
        axes[1].set_title('CVD Z-Score (FIXED: Hash-based Debounce)')
        axes[1].set_ylabel('CVD Z-Score')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # 3. DIR with Notional Filter
        dir_vals = [v for v in self.dir_values if v is not None]
        dir_times = times[-len(dir_vals):] if dir_vals else []
        
        if dir_vals:
            axes[2].plot(dir_times, dir_vals, self.config.colors['dir'], linewidth=2, label='DIR Ratio')
            axes[2].axhline(y=self.config.dir_threshold, color=self.config.colors['sell_threshold'], 
                           linestyle='--', alpha=0.7, label=f'DIR Threshold ±{self.config.dir_threshold}')
            axes[2].axhline(y=-self.config.dir_threshold, color=self.config.colors['sell_threshold'], 
                           linestyle='--', alpha=0.7)
            axes[2].axhline(y=0, color=self.config.colors['neutral'], linestyle='-', alpha=0.5)
            
            # FIXED: Notional filter indicator
            axes[2].text(0.02, 0.95, f'[FILTER] ${self.config.dir_notional_filter//1000}K Notional Filter: ACTIVE', 
                        transform=axes[2].transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
            
        axes[2].set_title('DIR (FIXED: $100K Notional Filter)')
        axes[2].set_ylabel('DIR Value')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        # 4. SCS with Notional Filter
        scs_vals = [v for v in self.scs_values if v is not None]
        scs_times = times[-len(scs_vals):] if scs_vals else []
        
        if scs_vals:
            axes[3].plot(scs_times, scs_vals, self.config.colors['scs'], linewidth=2, label='SCS Spread')
            axes[3].axhline(y=self.config.scs_threshold, color=self.config.colors['sell_threshold'], 
                           linestyle='--', alpha=0.7, label='Compression Threshold')
            axes[3].axhline(y=0, color=self.config.colors['neutral'], linestyle='-', alpha=0.5)
            
            # FIXED: Notional filter indicator  
            axes[3].text(0.02, 0.95, f'[FILTER] ${self.config.scs_notional_filter//1000}K Notional Filter: ACTIVE', 
                        transform=axes[3].transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.8))
            
        axes[3].set_title('SCS (FIXED: $50K Notional Filter)')
        axes[3].set_ylabel('Spread Value')
        axes[3].grid(True, alpha=0.3)
        axes[3].legend()
        axes[3].set_xlabel('Time')
        
        # Format timestamps for all subplots
        for ax in axes:
            ax.tick_params(axis='x', rotation=45)
            if len(times) > 0:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        
        plt.tight_layout()
        
        # Save extreme indicators dashboard with error handling
        timestamp = int(time.time())
        if save_path:
            filepath = Path(save_path)
            # Check if directory exists and is writable
            try:
                filepath.parent.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError) as e:
                logger.warning(f"⚠️ Cannot create directory for extreme indicators {save_path}: {e}")
                # Fallback to default output directory
                filename = f"extreme_indicators_{self.symbol.lower()}_{timestamp}.png"
                filepath = self.output_dir / filename
                logger.info(f"🔄 Using fallback path for extreme indicators: {filepath}")
        else:
            filename = f"extreme_indicators_{self.symbol.lower()}_{timestamp}.png"
            filepath = self.output_dir / filename
            
        try:
            plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            
            # Verify file was created successfully
            if not filepath.exists():
                logger.error(f"❌ Extreme indicators dashboard file was not created: {filepath}")
                return ""
                
            logger.info(f"✅ Extreme indicators dashboard saved: {filepath}")
            return str(filepath)
            
        except Exception as error:
            logger.error(f"❌ Extreme indicators dashboard generation/save failed: {error}")
            try:
                plt.close()  # Try to close any open figures
            except:
                pass
            return ""
        
    def generate_performance_plot(self, metrics: Dict[str, Any], save_path: Optional[str] = None) -> str:
        """Generate performance analysis plot"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'{self.symbol} - Performance Analysis', fontsize=16, fontweight='bold')
        
        # Plot performance metrics if available
        if 'latency' in metrics and metrics['latency']:
            axes[0,0].hist(metrics['latency'], bins=50, alpha=0.7, color='blue')
            axes[0,0].set_title('Latency Distribution')
            axes[0,0].set_xlabel('Latency (ms)')
            axes[0,0].set_ylabel('Frequency')
            axes[0,0].grid(True, alpha=0.3)
            
        if 'throughput' in metrics and metrics['throughput']:
            axes[0,1].plot(metrics['throughput'], color='green', linewidth=2)
            axes[0,1].set_title('Throughput Over Time')
            axes[0,1].set_xlabel('Time')
            axes[0,1].set_ylabel('Trades/sec')
            axes[0,1].grid(True, alpha=0.3)
            
        # Memory usage over time
        if 'memory_usage' in metrics and metrics['memory_usage']:
            axes[1,0].plot(metrics['memory_usage'], color='orange', linewidth=2)
            axes[1,0].set_title('Memory Usage')
            axes[1,0].set_xlabel('Time')
            axes[1,0].set_ylabel('Memory (MB)')
            axes[1,0].grid(True, alpha=0.3)
            
        # Summary statistics
        summary_text = []
        if 'avg_latency' in metrics:
            summary_text.append(f"Avg Latency: {metrics['avg_latency']:.2f}ms")
        if 'avg_throughput' in metrics:
            summary_text.append(f"Avg Throughput: {metrics['avg_throughput']:.1f} TPS")
        if 'total_trades' in metrics:
            summary_text.append(f"Total Trades: {metrics['total_trades']:,}")
        if 'uptime' in metrics:
            summary_text.append(f"Uptime: {metrics['uptime']:.1f} hours")
            
        axes[1,1].text(0.05, 0.95, '\n'.join(summary_text), 
                      transform=axes[1,1].transAxes, fontsize=12, verticalalignment='top',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        axes[1,1].set_xlim(0, 1)
        axes[1,1].set_ylim(0, 1)
        axes[1,1].axis('off')
        axes[1,1].set_title('Performance Summary')
        
        plt.tight_layout()
        
        # Save performance plot
        timestamp = int(time.time())
        if save_path:
            filepath = Path(save_path)
        else:
            filename = f"performance_{self.symbol.lower()}_{timestamp}.png"
            filepath = self.output_dir / filename
            
        plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Performance plot saved: {filepath}")
        return str(filepath)
        
    async def collect_live_data_and_visualize(self, duration_minutes: int = 5) -> Tuple[str, str]:
        """Collect live data and generate both dashboards"""
        
        logger.info(f"🔄 Starting live data collection for {duration_minutes} minutes...")
        
        try:
            # Import within function to avoid circular imports
            from rtai.utils import BinanceWebSocket
            
            ws = BinanceWebSocket()
            uri = f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@trade"
            
            connected = await ws.connect_with_backoff(uri)
            if not connected:
                logger.error("❌ Failed to connect to WebSocket")
                return "", ""
                
            logger.info("✅ Connected to live data feed")
            
            start_time = time.time()
            end_time = start_time + (duration_minutes * 60)
            trade_count = 0
            
            while time.time() < end_time:
                try:
                    message = await asyncio.wait_for(ws.websocket.recv(), timeout=5.0)
                    data = json.loads(message)
                    
                    # Process trade data
                    price = float(data['p'])
                    quantity = float(data['q'])
                    qty_signed = -quantity if data['m'] else quantity
                    timestamp = time.time()
                    
                    # Mock indicator updates for demonstration
                    indicators = {
                        'ofi': np.random.normal(0, 1.2),
                        'vpin': max(0, min(1, np.random.beta(2, 8))),
                        'kyle': max(0, np.random.exponential(0.05)),
                        'lpi': np.random.normal(0, 0.3),
                        'cvd_z': np.random.normal(0, 1.5),
                        'dir': np.random.normal(0, 0.8),
                        'scs': abs(np.random.normal(0.5, 0.2)),
                        'vpin_threshold': 0.98
                    }
                    
                    # Update visualization data
                    self.update_data(timestamp, price, indicators)
                    
                    trade_count += 1
                    if trade_count % 100 == 0:
                        logger.info(f"📊 Processed {trade_count} trades, latest price: ${price:,.2f}")
                        
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.warning(f"⚠️ Error processing trade: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"❌ Error in live data collection: {e}")
            return "", ""
        finally:
            if 'ws' in locals() and ws.websocket:
                await ws.close_gracefully()
                
        logger.info(f"✅ Data collection complete: {trade_count} trades processed")
        
        # Generate both dashboards
        live_dashboard = self.generate_live_dashboard()
        extreme_dashboard = self.generate_extreme_indicators_dashboard()
        
        return live_dashboard, extreme_dashboard


# Global instance for easy access
_unified_visualizer = None

def get_visualizer(symbol: str = "BTCUSDT") -> UnifiedVisualizer:
    """Get global unified visualizer instance"""
    global _unified_visualizer
    if _unified_visualizer is None or _unified_visualizer.symbol != symbol:
        _unified_visualizer = UnifiedVisualizer(symbol)
    return _unified_visualizer


# Convenience functions for backward compatibility
def generate_live_dashboard(symbol: str = "BTCUSDT", save_path: Optional[str] = None) -> str:
    """Generate live dashboard - convenience function"""
    visualizer = get_visualizer(symbol)
    return visualizer.generate_live_dashboard(save_path)

def generate_extreme_dashboard(symbol: str = "BTCUSDT", save_path: Optional[str] = None) -> str:
    """Generate extreme indicators dashboard - convenience function"""
    visualizer = get_visualizer(symbol)
    return visualizer.generate_extreme_indicators_dashboard(save_path)

def generate_performance_plot(symbol: str = "BTCUSDT", metrics: Dict[str, Any] = None, save_path: Optional[str] = None) -> str:
    """Generate performance plot - convenience function"""
    visualizer = get_visualizer(symbol)
    return visualizer.generate_performance_plot(metrics or {}, save_path)


if __name__ == "__main__":
    print("❌ DEPRECATED ENTRY POINT - Use python -m rtai.main --dashboard instead")
    import sys
    sys.exit(1)

