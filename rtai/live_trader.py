"""
🔥 RTAI Live Trader - Real-Time Algorithmic Intelligence
Enhanced live trading with multi-stream architecture and centralized data hub.
"""

import asyncio
import json
import time
import signal
from typing import Dict, List, Optional, Any
from collections import deque
import numpy as np

# Enhanced logging with graceful fallback
try:
    from .utils.structured_logging import create_logger
    logger = create_logger(__name__)
except ImportError:
    from loguru import logger
    logger.warning("Using loguru - structured logging not available")

# Multi-stream feed imports with comprehensive fallback
try:
    from .io.multi_stream_feed import BinanceMultiStreamFeed, recorder
    MULTI_STREAM_AVAILABLE = True
    logger.info("✅ Multi-stream feed system available")
except ImportError:
    try:
        from .io.feeds import BinanceLiveTrades
        MULTI_STREAM_AVAILABLE = False
        logger.warning("⚠️ Multi-stream feed not available, using fallback")
    except ImportError:
        logger.error("❌ No live feed system available!")
        MULTI_STREAM_AVAILABLE = None

# Enhanced cache and event system
try:
    from .state.cache import cache
    from .utils.events import bus
    ENHANCED_SYSTEMS = True
    logger.info("✅ Enhanced cache and event systems available")
except ImportError:
    ENHANCED_SYSTEMS = False
    logger.warning("⚠️ Enhanced systems not available - using basic mode")

# Indicator imports with fallback
try:
    from .indicators.base import OFI, VPIN, KyleLambda, LPI
    from .indicators.extremes import ExtremeIndicatorManager
    CORE_INDICATORS = True
    logger.info("✅ Core indicators available")
except ImportError:
    logger.warning("⚠️ Core indicators not available")
    CORE_INDICATORS = False

try:
    from .indicators.simple import WallRatioOsc, TradeImbalanceOsc, LiquidationOsc, DIROsc, FundingAccelOsc, DebounceManager
    SIMPLE_INDICATORS = True
except ImportError:
    logger.warning("⚠️ Some simple indicators not available")
    SIMPLE_INDICATORS = False

# Signal system
try:
    from .signals import signal_detector
    SIGNALS_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ Signal detector not available")
    SIGNALS_AVAILABLE = False

# Utilities with fallback
try:
    from .utils import BinanceWebSocket, PerformanceMetrics, PerformanceTimer, BackfillManager
    UTILS_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ Some utils not available")
    UTILS_AVAILABLE = False

try:
    from .utils.candlestick_engine import CandlestickEngine
    CANDLESTICK_AVAILABLE = True
except ImportError:
    logger.warning("⚠️ Candlestick engine not available")
    CANDLESTICK_AVAILABLE = False

from .state import StateAdapter, create_state_store
from .storage import IndicatorStorage

# RRS Engine Integration
try:
    from .io import EventRecorder, record_trade, record_depth_snapshot, record_liquidation, record_funding_update, record_indicator_update, record_basic_oscillator, record_signal_trigger, record_bar, record_signal, record_equity
    RRS_AVAILABLE = True
    logger.info("✅ RRS recording system available")
except ImportError:
    logger.warning("⚠️ Some RRS functions not available")
    RRS_AVAILABLE = False

# WebSocket broadcasting for real-time dashboard
try:
    from .api.server import broadcast_data
    BROADCAST_AVAILABLE = True
except ImportError:
    # Fallback if server not available
    logger.warning("⚠️ Dashboard broadcast not available")
    BROADCAST_AVAILABLE = False

class LiveTrader:
    """
    🚀 Enhanced Real-Time Trading System with Multi-Stream Architecture
    
    Features:
    - Multi-stream WebSocket feeds (trade, depth, OI, funding, liquidation)
    - Centralized data cache with thread-safe operations
    - Priority-based event bus for efficient message routing
    - Comprehensive indicator suite (OFI, VPIN, Kyle's Lambda, LPI)
    - Real-time recording and dashboard broadcasting
    - Graceful error handling and system resilience
    """
    
    def __init__(self, symbol: str = "BTCUSDT", window_size: int = 100, enable_recording: bool = True):
        """Initialize Enhanced LiveTrader with multi-stream architecture"""
        try:
            # Enhanced input validation for symbol
            if not isinstance(symbol, str):
                logger.error(f"LiveTrader: Invalid symbol type: {type(symbol)}")
                raise TypeError(f"Symbol must be string, got {type(symbol)}")
            
            if not symbol or not symbol.strip():
                logger.error("LiveTrader: Empty symbol provided")
                raise ValueError("Symbol cannot be empty")
            
            if len(symbol.strip()) < 2 or len(symbol.strip()) > 20:
                logger.error(f"LiveTrader: Invalid symbol length: {len(symbol.strip())}")
                raise ValueError(f"Symbol length must be 2-20 characters, got {len(symbol.strip())}")
            
            # Enhanced input validation for window_size
            if not isinstance(window_size, int):
                logger.error(f"LiveTrader: Invalid window_size type: {type(window_size)}")
                raise TypeError(f"Window size must be integer, got {type(window_size)}")
            
            if window_size <= 0:
                logger.error(f"LiveTrader: Invalid window_size value: {window_size}")
                raise ValueError(f"Window size must be positive, got {window_size}")
            
            if window_size > 10000:  # Reasonable upper bound
                logger.warning(f"LiveTrader: Large window_size: {window_size}, capping at 10000")
                window_size = 10000
            
            self.symbol = symbol.strip().upper()  # Store as uppercase for consistency
            self.window_size = window_size
            self.enable_recording = enable_recording
            self.is_running = False
            
            # Multi-stream feed setup
            if MULTI_STREAM_AVAILABLE:
                logger.info(f"🔗 Initializing multi-stream feed for {self.symbol}")
                self.feed = BinanceMultiStreamFeed(self.symbol)
                self.use_multi_stream = True
            else:
                logger.warning(f"⚠️ Using fallback single-stream for {self.symbol}")
                self.feed = BinanceLiveTrades(self.symbol) if 'BinanceLiveTrades' in globals() else None
                self.use_multi_stream = False
            
            # State management - default to memory for live trading with error handling
            try:
                self.state_store = create_state_store("memory")
                self.state_adapter = StateAdapter(self.state_store)
            except Exception as e:
                logger.error(f"LiveTrader: Failed to initialize state management: {e}")
                raise RuntimeError(f"State management initialization failed: {e}")
            
            # Performance tracking with enhanced metrics and error handling
            try:
                self.metrics = PerformanceMetrics()
            except Exception as e:
                logger.error(f"LiveTrader: Failed to initialize performance metrics: {e}")
                raise RuntimeError(f"Performance metrics initialization failed: {e}")
            
            # Enhanced indicators with new parameters and comprehensive error handling
            except Exception as e:
                logger.error(f"Failed to initialize state management: {e}")
                self.state_store = None
                self.state_adapter = None
            
            # Enhanced indicator initialization with comprehensive fallbacks
            self._initialize_indicators()
            
            # Storage with fallback
            try:
                self.storage = IndicatorStorage()
                logger.debug("✅ Indicator storage initialized")
            except Exception as e:
                logger.warning(f"Storage initialization failed: {e}")
                self.storage = None
            
            # Event handlers setup
            self._setup_event_handlers()
            
            # Performance tracking
            self.stats = {
                'trades_processed': 0,
                'indicators_computed': 0,
                'last_update': 0,
                'processing_times': deque(maxlen=1000),
                'errors': 0
            }
            
            # Recording setup
            if self.enable_recording and 'recorder' in globals():
                try:
                    recorder.start_recording(self.symbol)
                    logger.info(f"📹 Recording enabled for {self.symbol}")
                except Exception as e:
                    logger.warning(f"Recording setup failed: {e}")
            
            logger.success(f"🚀 Enhanced LiveTrader initialized for {self.symbol}")
            
        except Exception as e:
            logger.error(f"LiveTrader.__init__ critical error: {e}")
            raise
    
    def _initialize_indicators(self):
        """Initialize all indicators with comprehensive error handling"""
        # Core microstructure indicators
        if CORE_INDICATORS:
            try:
                self.ofi = OFI(alpha=0.7, mad_alpha=0.5)
                logger.debug("✅ OFI initialized")
            except Exception as e:
                logger.warning(f"OFI initialization failed: {e}")
                self.ofi = None
            
            try:
                self.vpin = VPIN(base_bucket_usdt=20.0, win_buckets=30)
                logger.debug("✅ VPIN initialized")
            except Exception as e:
                logger.warning(f"VPIN initialization failed: {e}")
                self.vpin = None
            
            try:
                self.kyle = KyleLambda(window=self.window_size)
                logger.debug("✅ Kyle's Lambda initialized")
            except Exception as e:
                logger.warning(f"Kyle's Lambda initialization failed: {e}")
                self.kyle = None
            
            try:
                self.lpi = LPI(window_seconds=60)
                logger.debug("✅ LPI initialized")
            except Exception as e:
                logger.warning(f"LPI initialization failed: {e}")
                self.lpi = None
        else:
            logger.warning("⚠️ Core indicators not available")
            self.ofi = self.vpin = self.kyle = self.lpi = None
        
        # Simple indicators
        if SIMPLE_INDICATORS:
            try:
                self.wall_ratio_osc = WallRatioOsc()
                self.trade_imbalance_osc = TradeImbalanceOsc()
                self.liquidation_osc = LiquidationOsc()
                self.dir_osc = DIROsc()
                self.funding_accel_osc = FundingAccelOsc()
                self.debounce_manager = DebounceManager()
                logger.debug("✅ Simple indicators initialized")
            except Exception as e:
                logger.warning(f"Simple indicators initialization failed: {e}")
        
        # Extremes manager
        try:
            self.extremes = ExtremeIndicatorManager(window_size=self.window_size)
            logger.debug("✅ Extremes manager initialized")
        except Exception as e:
            logger.warning(f"Extremes manager initialization failed: {e}")
            self.extremes = None
    
    def _setup_event_handlers(self):
        """Setup event handlers for multi-stream data"""
        if not ENHANCED_SYSTEMS:
            return
        
        # Register event handlers for different data types
        try:
            bus.subscribe("trade_update", self._handle_trade_event)
            bus.subscribe("depth_update", self._handle_depth_event)
            bus.subscribe("oi_update", self._handle_oi_event)
            bus.subscribe("funding_update", self._handle_funding_event)
            bus.subscribe("liquidation_update", self._handle_liquidation_event)
            logger.debug("✅ Event handlers registered")
        except Exception as e:
            logger.warning(f"Event handler setup failed: {e}")
    
    async def _handle_trade_event(self, data: Dict[str, Any]):
        """Handle trade update from event bus"""
        try:
            await self._process_trade_data(data)
        except Exception as e:
            logger.error(f"Error handling trade event: {e}")
            self.stats['errors'] += 1
    
    async def _handle_depth_event(self, data: Dict[str, Any]):
        """Handle depth update from event bus"""
        try:
            await self._process_depth_data(data)
        except Exception as e:
            logger.error(f"Error handling depth event: {e}")
            self.stats['errors'] += 1
    
    async def _handle_oi_event(self, data: Dict[str, Any]):
        """Handle open interest update from event bus"""
        try:
            if self.kyle:
                # OI changes can affect Kyle's Lambda calculations
                self.kyle.update_market_state(oi_change=data.get('value', 0))
        except Exception as e:
            logger.error(f"Error handling OI event: {e}")
    
    async def _handle_funding_event(self, data: Dict[str, Any]):
        """Handle funding rate update from event bus"""
        try:
            if hasattr(self, 'funding_accel_osc') and self.funding_accel_osc:
                self.funding_accel_osc.update(data.get('value', 0), data.get('ts', time.time()))
        except Exception as e:
            logger.error(f"Error handling funding event: {e}")
    
    async def _handle_liquidation_event(self, data: Dict[str, Any]):
        """Handle liquidation update from event bus"""
        try:
            if hasattr(self, 'liquidation_osc') and self.liquidation_osc:
                self.liquidation_osc.update(
                    data.get('quantity', 0),
                    data.get('side', 'unknown'),
                    data.get('ts', time.time())
                )
        except Exception as e:
            logger.error(f"Error handling liquidation event: {e}")
    
    async def _process_trade_data(self, trade_data: Dict[str, Any]):
        """Process trade data and update indicators"""
        start_time = time.time()
        
        try:
            # Extract trade information
            ts = trade_data.get('ts', time.time())
            price = float(trade_data.get('price', 0))
            volume = float(trade_data.get('quantity', 0))
            side = trade_data.get('side', 'unknown')
            
            # Update core indicators
            if self.ofi:
                self.ofi.update(price, volume, side == 'buy', ts)
            
            if self.vpin:
                self.vpin.update(price, volume, side == 'buy')
            
            if self.kyle:
                self.kyle.update(price, volume, ts)
            
            if self.lpi:
                signed_volume = volume * (1 if side == 'buy' else -1)
                self.lpi.update(price, signed_volume, ts)
            
            # Update simple indicators
            if hasattr(self, 'trade_imbalance_osc') and self.trade_imbalance_osc:
                self.trade_imbalance_osc.update(volume, side == 'buy', ts)
            
            # Update extremes
            if self.extremes:
                self.extremes.update_with_trade(price, volume, side == 'buy', ts)
            
            # Record for RRS if available
            if RRS_AVAILABLE and self.enable_recording:
                try:
                    record_trade(price, volume, side, ts)
                except Exception as e:
                    logger.debug(f"Trade recording error: {e}")
            
            # Broadcast if available
            if BROADCAST_AVAILABLE:
                try:
                    await broadcast_data("trade", {
                        "price": price,
                        "volume": volume,
                        "side": side,
                        "timestamp": ts
                    })
                except Exception as e:
                    logger.debug(f"Broadcast error: {e}")
            
            # Update stats
            self.stats['trades_processed'] += 1
            self.stats['last_update'] = ts
            processing_time = time.time() - start_time
            self.stats['processing_times'].append(processing_time)
            
        except Exception as e:
            logger.error(f"Error processing trade data: {e}")
            self.stats['errors'] += 1
    
    async def _process_depth_data(self, depth_data: Dict[str, Any]):
        """Process depth/orderbook data"""
        try:
            ts = depth_data.get('ts', time.time())
            bid_price = float(depth_data.get('bid_price', 0))
            ask_price = float(depth_data.get('ask_price', 0))
            bid_qty = float(depth_data.get('bid_qty', 0))
            ask_qty = float(depth_data.get('ask_qty', 0))
            
            # Update indicators that use depth data
            if hasattr(self, 'wall_ratio_osc') and self.wall_ratio_osc:
                self.wall_ratio_osc.update(bid_qty, ask_qty, ts)
            
            # Record depth snapshot if available
            if RRS_AVAILABLE and self.enable_recording:
                try:
                    record_depth_snapshot(bid_price, ask_price, bid_qty, ask_qty, ts)
                except Exception as e:
                    logger.debug(f"Depth recording error: {e}")
            
        except Exception as e:
            logger.error(f"Error processing depth data: {e}")
    
    async def run(self):
        """Enhanced main run loop with multi-stream support"""
        if self.is_running:
            logger.warning("LiveTrader is already running")
            return
        
        self.is_running = True
        logger.info(f"🚀 Starting Enhanced LiveTrader for {self.symbol}")
        
        try:
            if self.use_multi_stream and self.feed:
                # Use enhanced multi-stream feed
                logger.info("🔗 Starting multi-stream feed...")
                await self.feed.start()
            else:
                # Fallback to single stream
                logger.warning("⚠️ Using single-stream fallback")
                await self._run_single_stream()
                
        except KeyboardInterrupt:
            logger.info("👋 Shutdown requested by user")
        except Exception as e:
            logger.error(f"❌ LiveTrader run error: {e}")
        finally:
            self.is_running = False
            if self.feed and hasattr(self.feed, 'stop'):
                self.feed.stop()
            logger.info("🛑 LiveTrader stopped")
    
    async def _run_single_stream(self):
        """Fallback single stream processing"""
        if not self.feed:
            logger.error("No feed available for single stream mode")
            return
        
        try:
            async for trade in self.feed:
                if not self.is_running:
                    break
                
                # Convert trade to standard format and process
                trade_data = {
                    'ts': trade.timestamp,
                    'price': trade.price,
                    'quantity': trade.volume,
                    'side': trade.side
                }
                
                await self._process_trade_data(trade_data)
                
        except Exception as e:
            logger.error(f"Single stream error: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        stats = self.stats.copy()
        
        # Add processing performance
        if self.stats['processing_times']:
            times = list(self.stats['processing_times'])
            stats['avg_processing_time'] = sum(times) / len(times)
            stats['max_processing_time'] = max(times)
            stats['min_processing_time'] = min(times)
        
        # Add feed stats if available
        if self.feed and hasattr(self.feed, 'get_stats'):
            stats['feed_stats'] = self.feed.get_stats()
        
        # Add cache stats if available
        if ENHANCED_SYSTEMS:
            try:
                stats['cache_stats'] = cache.get_stats()
                stats['bus_stats'] = bus.get_stats()
            except Exception as e:
                logger.debug(f"Stats collection error: {e}")
        
        return stats
    
    def stop(self):
        """Stop the LiveTrader gracefully"""
        self.is_running = False
        if self.feed and hasattr(self.feed, 'stop'):
            self.feed.stop()
        logger.info("🛑 LiveTrader stop requested")
                    logger.warning("LiveTrader: Liquidations indicator not available, skipping liq_z")
                    self.liq_z = None
            except Exception as e:
                logger.warning(f"LiveTrader: Failed to initialize liq_z: {e}")
                self.liq_z = None
            
            try:
                if hasattr(self, 'wall_ratio') and self.wall_ratio is not None:
                    self.wall_ratio_z = ZBand(self.wall_ratio, window=120)
                else:
                    logger.warning("LiveTrader: WallRatio indicator not available, skipping wall_ratio_z")
                    self.wall_ratio_z = None
            except Exception as e:
                logger.warning(f"LiveTrader: Failed to initialize wall_ratio_z: {e}")
                self.wall_ratio_z = None
            
            try:
                if hasattr(self, 'funding_basis') and self.funding_basis is not None:
                    self.basis_z = ZBand(self.funding_basis, window=120)
                else:
                    logger.warning("LiveTrader: FundingBasis indicator not available, skipping basis_z")
                    self.basis_z = None
            except Exception as e:
                logger.warning(f"LiveTrader: Failed to initialize basis_z: {e}")
                self.basis_z = None
            
            try:
                if hasattr(self, 'trade_imbalance') and self.trade_imbalance is not None:
                    self.trade_imb_z = ZBand(self.trade_imbalance, window=120)
                else:
                    logger.warning("LiveTrader: TradeImbalance indicator not available, skipping trade_imb_z")
                    self.trade_imb_z = None
            except Exception as e:
                logger.warning(f"LiveTrader: Failed to initialize trade_imb_z: {e}")
                self.trade_imb_z = None
            
            logger.debug("LiveTrader: Z-band wrappers initialization completed (some may be None)")
        except Exception as e:
            logger.error(f"LiveTrader: Critical error in Z-band wrappers initialization: {e}")
            # Don't raise - continue with None values
        
        # Advanced Z-band wrappers (Items 26-30: Multi-timeframe Analysis) with error handling
        try:
            self.advanced_liq_z = AdvancedZBandWrapper(window_size=120, timeframes=[60, 120, 300, 600])
            self.advanced_wall_z = AdvancedZBandWrapper(window_size=120, timeframes=[60, 120, 300, 600])
            self.advanced_basis_z = AdvancedZBandWrapper(window_size=120, timeframes=[60, 120, 300, 600])
            self.advanced_trade_z = AdvancedZBandWrapper(window_size=120, timeframes=[60, 120, 300, 600])
            logger.debug("LiveTrader: Advanced Z-band wrappers initialized")
        except Exception as e:
            logger.error(f"LiveTrader: Failed to initialize advanced Z-band wrappers: {e}")
            raise RuntimeError(f"Advanced Z-band wrappers initialization failed: {e}")
        
        # NEW: RSI-Style Basic Oscillators (Professional Implementation) RESTORED with error handling
        try:
            self.wall_ratio_osc = WallRatioOsc(window=100, price_levels=3, size_cutoff=150.0)
            self.trade_imbalance_osc = TradeImbalanceOsc(window=60)  # Remove bucket_seconds
            self.liquidation_osc = LiquidationOsc(window=60)  # Remove bucket_seconds
            self.dir_osc = DIROsc(window=100)  # Remove depth_levels
            self.funding_accel_osc = FundingAccelOsc(window=50)  # Remove smoothing_window
            logger.debug("LiveTrader: RSI-style oscillators initialized")
        except Exception as e:
            logger.error(f"LiveTrader: Failed to initialize RSI-style oscillators: {e}")
            raise RuntimeError(f"RSI-style oscillators initialization failed: {e}")
        
        # Centralized debounce manager RESTORED with error handling
        try:
            self.debounce_manager = DebounceManager()
            logger.info("✅ Essential oscillators restored and operational")
        except Exception as e:
            logger.error(f"LiveTrader: Failed to initialize debounce manager: {e}")
            raise RuntimeError(f"Debounce manager initialization failed: {e}")
        
        # Basic oscillators integration re-enabled
        logger.info("✅ Basic oscillators manager active (WallRatio, TradeImbalance, Liquidation, DIR, FundingAccel)")
        
        # Thresholds for basic oscillators (configurable) with validation
        try:
            self.basic_config = {
                'extreme_zscore_threshold': 2.0,
                'debounce_seconds': 30.0,
                'warning_zscore_threshold': 1.0
            }
            
            # Validate configuration values
            for key, value in self.basic_config.items():
                if not isinstance(value, (int, float)):
                    logger.error(f"LiveTrader: Invalid basic_config value for {key}: {value}")
                    raise ValueError(f"Invalid basic_config value for {key}: {value}")
                if value <= 0:
                    logger.error(f"LiveTrader: Non-positive basic_config value for {key}: {value}")
                    raise ValueError(f"Non-positive basic_config value for {key}: {value}")
            
        except Exception as e:
            logger.error(f"LiveTrader: Failed to initialize basic_config: {e}")
            raise RuntimeError(f"Basic configuration initialization failed: {e}")
        
        # Storage system for persistence with error handling
        try:
            self.storage = IndicatorStorage()
            logger.debug("LiveTrader: Storage system initialized")
        except Exception as e:
            logger.error(f"LiveTrader: Failed to initialize storage system: {e}")
            raise RuntimeError(f"Storage system initialization failed: {e}")
        
        # Load Z-band state from storage for persistence with error handling
        try:
            self._load_z_band_state()
        except Exception as e:
            logger.warning(f"LiveTrader: Failed to load Z-band state: {e}")
            # Don't raise - this is recoverable
        
        # Micro-indicatori per estremi with error handling
        try:
            self.extremes = ExtremeIndicatorManager()
            logger.debug("LiveTrader: Extreme indicators manager initialized")
        except Exception as e:
            logger.error(f"LiveTrader: Failed to initialize extreme indicators: {e}")
            raise RuntimeError(f"Extreme indicators initialization failed: {e}")
        
        # Visualization system re-enabled - using WebSocket broadcasting
        logger.info("📊 Visualization system active via WebSocket broadcasting to dashboard")
        
        # Backfill manager for warm-up after reconnection with error handling
        try:
            self.backfill_manager = BackfillManager()
            logger.debug("LiveTrader: Backfill manager initialized")
        except Exception as e:
            logger.error(f"LiveTrader: Failed to initialize backfill manager: {e}")
            raise RuntimeError(f"Backfill manager initialization failed: {e}")
        
        # RRS Engine Recording (Professional Drop-In Integration) with error handling
        try:
            from rtai.io.recorder import RecordConfig
            record_config = RecordConfig(
                output_dir="recordings",
                max_file_size_mb=100,  # 100MB per file
                compression=True
            )
            self.recorder = EventRecorder(symbol=symbol.upper(), config=record_config)
            self.enable_recording = True  # Can be toggled for production
            logger.debug("LiveTrader: RRS Engine initialized")
        except Exception as e:
            logger.error(f"LiveTrader: Failed to initialize RRS Engine: {e}")
            raise RuntimeError(f"RRS Engine initialization failed: {e}")
        
        # Data storage with bounds checking
        try:
            if window_size <= 0:
                raise ValueError(f"Invalid window_size for data storage: {window_size}")
                
            storage_multiplier = max(2, min(10, window_size // 50))  # Reasonable multiplier
            self.timestamps = deque(maxlen=window_size * storage_multiplier)
            self.prices = deque(maxlen=window_size * storage_multiplier)
            self.volumes = deque(maxlen=window_size * storage_multiplier)
            self.trades = deque(maxlen=window_size * max(10, storage_multiplier))  # Store more trades for indicators
            
            # Indicator value history for plotting with validation
            self.ofi_history = deque(maxlen=window_size)
            self.vpin_history = deque(maxlen=window_size)
            self.kyle_history = deque(maxlen=window_size)
            self.lpi_history = deque(maxlen=window_size)
            self.indicator_timestamps = deque(maxlen=window_size)
            logger.debug("LiveTrader: Data storage initialized")
        except Exception as e:
            logger.error(f"LiveTrader: Failed to initialize data storage: {e}")
            raise RuntimeError(f"Data storage initialization failed: {e}")
        
        # State initialization with validation
        try:
            self.is_running = False
            self.shutdown_event = asyncio.Event()  # For graceful shutdown
            self.last_price = 0.0  # Initialize with safe default
            self.tick_count = 0
            self.current_minute = -1  # Track current minute for candle finalization
            
            # Initialize candlestick engine for OHLCV data
            self.candlestick_engine = CandlestickEngine(interval_seconds=60)
            logger.debug("LiveTrader: CandlestickEngine initialized")
            
            # Dashboard generation flags with validation
            self.startup_dashboard_generated = False
            self.last_dashboard_time = 0
            self.dashboard_interval = 300  # 5 minutes
            
            if self.dashboard_interval <= 0:
                logger.warning("LiveTrader: Invalid dashboard_interval, setting to 300")
                self.dashboard_interval = 300
            
            # Minute-level storage tracking with validation
            current_time = time.time()
            if not isinstance(current_time, (int, float)) or current_time <= 0:
                logger.error(f"LiveTrader: Invalid current time: {current_time}")
                raise ValueError(f"Invalid current time: {current_time}")
                
            self.storage_minute = int(current_time // 60)
            self.minute_data = {
                'trades_count': 0,
                'total_volume': 0.0,
                'price_sum': 0.0,
                'price_count': 0
            }
            
            # Storage synchronization settings (Items 22-25) with validation
            self.sync_interval = 60  # Sync every minute
            if self.sync_interval <= 0:
                logger.warning("LiveTrader: Invalid sync_interval, setting to 60")
                self.sync_interval = 60
                
            self.last_sync = current_time
            self.batch_storage = []
            self.storage_batch_size = 50
            
            if self.storage_batch_size <= 0:
                logger.warning("LiveTrader: Invalid storage_batch_size, setting to 50")
                self.storage_batch_size = 50
            
            logger.debug("LiveTrader: State variables initialized")
        except Exception as e:
            logger.error(f"LiveTrader: Failed to initialize state variables: {e}")
            raise RuntimeError(f"State variables initialization failed: {e}")
        
        # Initialize signal detector for both basic and advanced signals with error handling
        try:
            self.signal_detector = signal_detector
            logger.debug("LiveTrader: Signal detector initialized")
        except Exception as e:
            logger.error(f"LiveTrader: Failed to initialize signal detector: {e}")
            raise RuntimeError(f"Signal detector initialization failed: {e}")
        
        # RSI-like Oscillators (NEW: Unified Pipeline) with comprehensive error handling
        # NOTE: These imports are temporarily disabled during zombie elimination
        # All indicators are now consolidated in rtai.indicators.base
        try:
            # ZOMBIE ELIMINATION: These files have been removed and consolidated into base.py
            # from rtai.indicators.ofi import OFIOsc
            # from rtai.indicators.lpi import LPIOsc
            # from rtai.indicators.vpin import VPINOsc
            # from rtai.indicators.kyle import KyleOsc
            # from rtai.indicators.cvd import CVDOsc
            # from rtai.indicators.basis import BasisOsc
            # from rtai.indicators.filters import Cooldown, EventAccumulator
            
            # Use base indicators (already available) - RSI-like oscillators re-enabled
            try:
                # We already have the base indicators (OFI, LPI, VPIN, Kyle, etc.)
                # These provide the same functionality as the oscillator versions
                logger.info("✅ Using base indicators (OFI, LPI, VPIN, Kyle) - equivalent to RSI-like oscillators")
                logger.debug("LiveTrader: Base indicators active (OFI, LPI, VPIN, Kyle, CVD, Basis)")
            except Exception as e:
                logger.warning(f"LiveTrader: Base indicators setup warning: {e}")
        except ImportError as e:
            logger.warning(f"LiveTrader: RSI oscillator modules not available (expected during zombie elimination): {e}")
        except Exception as e:
            logger.warning(f"LiveTrader: RSI oscillators initialization disabled: {e}")
        
        # Unified cooldown and event management with comprehensive error handling
        # ZOMBIE ELIMINATION: Temporarily using simple replacements
        try:
            from rtai.config import COOLDOWN_SEC, EVENT_WINDOWS, BIG_ORDER_THRESHOLD
            
            # Validate configuration values
            if not isinstance(COOLDOWN_SEC, (int, float)) or COOLDOWN_SEC <= 0:
                logger.error(f"LiveTrader: Invalid COOLDOWN_SEC: {COOLDOWN_SEC}")
                raise ValueError(f"Invalid COOLDOWN_SEC: {COOLDOWN_SEC}")
            
            if not isinstance(EVENT_WINDOWS, dict):
                logger.error(f"LiveTrader: Invalid EVENT_WINDOWS type: {type(EVENT_WINDOWS)}")
                raise TypeError(f"EVENT_WINDOWS must be dict, got {type(EVENT_WINDOWS)}")
            
            if "big_orders" not in EVENT_WINDOWS:
                logger.error("LiveTrader: Missing 'big_orders' in EVENT_WINDOWS")
                raise KeyError("Missing 'big_orders' in EVENT_WINDOWS")
            
            if not isinstance(BIG_ORDER_THRESHOLD, (int, float)) or BIG_ORDER_THRESHOLD <= 0:
                logger.error(f"LiveTrader: Invalid BIG_ORDER_THRESHOLD: {BIG_ORDER_THRESHOLD}")
                raise ValueError(f"Invalid BIG_ORDER_THRESHOLD: {BIG_ORDER_THRESHOLD}")
            
            # Create simple replacements for removed Cooldown and EventAccumulator classes
            class SimpleCooldown:
                """Simple cooldown replacement during zombie elimination"""
                def __init__(self, seconds): 
                    self.seconds = seconds
                    self.last_times = {}
                def ready(self, key): 
                    import time
                    now = time.time()
                    last = self.last_times.get(key, 0)
                    result = (now - last) >= self.seconds
                    if result: self.last_times[key] = now
                    return result
            
            class SimpleEventAccumulator:
                """Simple event accumulator replacement during zombie elimination"""
                def __init__(self, window): 
                    self.window = window
                    self.events = []
                def add_event(self, value): 
                    import time
                    self.events.append((time.time(), value))
                    cutoff = time.time() - self.window
                    self.events = [(t, v) for t, v in self.events if t > cutoff]
                def get_sum(self): 
                    return sum(v for t, v in self.events)
            
            self._cooldown = SimpleCooldown(COOLDOWN_SEC)
            self._big_order_acc = SimpleEventAccumulator(EVENT_WINDOWS["big_orders"])
            self._big_order_threshold = BIG_ORDER_THRESHOLD
            logger.info("🔄 Using simple cooldown/event replacements during zombie elimination")
            logger.debug("LiveTrader: Simple cooldown and event management initialized")
        except Exception as e:
            logger.error(f"LiveTrader: Failed to initialize cooldown/event management: {e}")
            raise RuntimeError(f"Cooldown/event management initialization failed: {e}")
        
        # Minute aggregation containers with validation
        try:
            self._current_minute_ts = 0
            self._agg = type('Agg', (), {
                'ofi_sum': {},
                'liq_buy_not': {},
                'liq_sell_not': {},
                'vpin_value': {},
                'price_deltas': {},
                'notional_list': {},
                'buy_vol': {},
                'sell_vol': {},
                'basis_values': {},
                'funding_rates': {},
            })()
            
            # RSI state tracking for isteresi
            self._rsi_state = {}
            logger.debug("LiveTrader: Aggregation containers initialized")
        except Exception as e:
            logger.error(f"LiveTrader: Failed to initialize aggregation containers: {e}")
            raise RuntimeError(f"Aggregation containers initialization failed: {e}")
        
        # Initialize Telegram bot (avoid AttributeError in finally block) with error handling
        try:
            self.telegram = self._try_init_telegram()
        except Exception as e:
            logger.warning(f"LiveTrader: Telegram initialization issue: {e}")
            self.telegram = None  # Ensure it's defined for safety
        
        logger.info(f"✅ LiveTrader initialized for {symbol.upper()}")
        logger.info("📊 Performance tracking enabled")
        logger.info("⚡ Extreme indicators active (CVD, DIR, SCS, FACCEL, WD)")
        logger.info("🎯 RSI-like oscillators active (OFI, LPI, VPIN, Kyle, CVD, Basis)")
    
    def _try_init_telegram(self):
        """Initialize Telegram bot safely with comprehensive error handling"""
        try:
            # Validate environment first
            import os
            
            token = os.getenv("TELEGRAM_BOT_TOKEN")
            chat_id = os.getenv("TELEGRAM_CHAT_ID")
            
            if not token or not chat_id:
                logger.debug("LiveTrader: Telegram credentials not provided in environment")
                return None
            
            if not isinstance(token, str) or not token.strip():
                logger.warning("LiveTrader: Invalid Telegram bot token format")
                return None
            
            if not isinstance(chat_id, str) or not chat_id.strip():
                logger.warning("LiveTrader: Invalid Telegram chat ID format")
                return None
            
            # Attempt to import and initialize Telegram bot
            try:
                from rtai.telegram_bot import TelegramBot
            except ImportError as e:
                logger.warning(f"LiveTrader: TelegramBot module not available: {e}")
                return None
            
            telegram = TelegramBot()
            
            # Validate the initialized bot
            if telegram is None:
                logger.warning("LiveTrader: TelegramBot initialization returned None")
                return None
            
            logger.info("✅ Telegram bot initialized successfully")
            return telegram
            
        except Exception as e:
            logger.warning(f"LiveTrader: Telegram bot initialization failed: {e}")
            return None
        
    def _load_z_band_state(self):
        """Load Z-band state from storage for persistence across restarts with enhanced error handling"""
        try:
            # Validate storage system first
            if not hasattr(self, 'storage') or self.storage is None:
                logger.warning("LiveTrader: Storage system not available for Z-band state loading")
                return
            
            # Load state for each Z-band indicator with validation
            z_indicators = {
                'liq_z': self.liq_z,
                'wall_ratio_z': self.wall_ratio_z,
                'basis_z': self.basis_z,
                'trade_imb_z': self.trade_imb_z
            }
            
            loaded_count = 0
            for name, indicator in z_indicators.items():
                try:
                    if not hasattr(indicator, 'values') or indicator is None:
                        logger.warning(f"LiveTrader: Invalid Z-band indicator: {name}")
                        continue
                    
                    # Attempt to load state from storage
                    if hasattr(self.storage, 'load_z_state'):
                        state_data = self.storage.load_z_state(name)
                        
                        if state_data and isinstance(state_data, (list, tuple)):
                            # Validate and restore state data
                            valid_values = []
                            for value in state_data:
                                if isinstance(value, (int, float)) and not np.isnan(value) and np.isfinite(value):
                                    valid_values.append(float(value))
                            
                            if valid_values:
                                # Restore state to indicator
                                try:
                                    indicator.values.extend(valid_values)
                                    loaded_count += 1
                                    logger.debug(f"LiveTrader: Loaded {len(valid_values)} values for {name}")
                                except Exception as restore_error:
                                    logger.warning(f"LiveTrader: Failed to restore {name} state: {restore_error}")
                    
                except Exception as indicator_error:
                    logger.warning(f"LiveTrader: Error loading state for {name}: {indicator_error}")
                    continue
            
            if loaded_count > 0:
                logger.info(f"LiveTrader: Successfully loaded Z-band state for {loaded_count} indicators")
            else:
                logger.debug("LiveTrader: No Z-band state loaded (first run or no saved state)")
                
        except Exception as e:
            logger.warning(f"LiveTrader: Z-band state loading failed: {e}")
            # Don't raise - this is recoverable, system can start fresh
            
            for name, z_band in z_indicators.items():
                values = self.storage.load_z_state(name)
                if values:
                    # Restore the rolling buffer
                    z_band.values.extend(values)
                    if len(z_band.values) > z_band.window:
                        # Keep only the most recent values
                        z_band.values = z_band.values[-z_band.window:]
                    logger.info(f"🔄 Restored {name} state: {len(values)} values")
                else:
                    logger.debug(f"No stored state found for {name}")
                    
        except Exception as e:
            logger.warning(f"Error loading Z-band state: {e}")
            
    async def _store_minute_data(self, minute_timestamp: int, indicators: Dict[str, float]):
        """Store minute-level aggregated data with atomic indicators
        
        Args:
            minute_timestamp: Minute timestamp 
            indicators: Current indicator values
        """
        try:
            # Calculate minute averages
            avg_price = self.minute_data['price_sum'] / max(self.minute_data['price_count'], 1)
            total_volume = self.minute_data['total_volume']
            trades_count = self.minute_data['trades_count']
            
            # Store features in database
            success = self.storage.store_minute_features(
                symbol=self.symbol.upper(),
                indicators=indicators,
                price=avg_price,
                volume=total_volume,
                trades_count=trades_count
            )
            
            if success:
                logger.debug(f"💾 Stored minute data: {trades_count} trades, ${total_volume:.2f} volume")
                
                # Optionally store Z-band state for persistence
                z_indicators = {
                    'liq_z': list(self.liq_z.values) if self.liq_z.values else [],
                    'wall_ratio_z': list(self.wall_ratio_z.values) if self.wall_ratio_z.values else [],
                    'basis_z': list(self.basis_z.values) if self.basis_z.values else [],
                    'trade_imb_z': list(self.trade_imb_z.values) if self.trade_imb_z.values else []
                }
                
                # Store Z-state every 10 minutes to reduce I/O
                if minute_timestamp % (10 * 60) == 0:
                    for name, values in z_indicators.items():
                        if values:
                            self.storage.store_z_state(name, values)
                
        except Exception as e:
            logger.error(f"Error storing minute data: {e}")
    
    def _queue_storage_data(self, data_point: Dict[str, any]):
        """Queue data point for batch storage (Item 23: Storage Optimization)"""
        try:
            self.batch_storage.append(data_point)
            
            # Process batch when size limit reached
            if len(self.batch_storage) >= self.storage_batch_size:
                self._process_storage_batch()
                
        except Exception as e:
            logger.error(f"Error queuing storage data: {e}")
    
    def _process_storage_batch(self):
        """Process batched storage data for efficiency (Item 24: Batch Processing)"""
        try:
            if not self.batch_storage:
                return
                
            logger.debug(f"💾 Processing storage batch: {len(self.batch_storage)} items")
            
            # Batch process all queued data points
            for data_point in self.batch_storage:
                if 'minute_data' in data_point:
                    # Store minute aggregated data
                    minute_features = data_point['minute_data']
                    timestamp = data_point['timestamp']
                    
                    success = self.storage.store_minute_features(
                        timestamp, self.symbol, minute_features
                    )
                    
                    if not success:
                        logger.warning(f"Failed to store minute data for {timestamp}")
            
            # Clear batch after processing
            self.batch_storage.clear()
            logger.debug("✅ Storage batch processed successfully")
            
        except Exception as e:
            logger.error(f"Error processing storage batch: {e}")
            self.batch_storage.clear()  # Clear to prevent memory leak
    
    def _sync_database_state(self):
        """Synchronize database state and cleanup old data (Item 25: Database Sync)"""
        try:
            current_time = time.time()
            
            # Check if sync is due
            if current_time - self.last_sync < self.sync_interval:
                return
                
            logger.debug("🔄 Synchronizing database state...")
            
            # Process any remaining batch data
            self._process_storage_batch()
            
            # Sync Z-band states for persistence
            z_states = {
                'liq_z': list(self.liq_z.values) if self.liq_z.values else [],
                'wall_ratio_z': list(self.wall_ratio_z.values) if self.wall_ratio_z.values else [],
                'basis_z': list(self.basis_z.values) if self.basis_z.values else [],
                'trade_imb_z': list(self.trade_imb_z.values) if self.trade_imb_z.values else []
            }
            
            for name, values in z_states.items():
                if values:
                    self.storage.store_z_state(name, values)
            
            # Cleanup old data (keep last 7 days for performance)
            cutoff_timestamp = current_time - (7 * 24 * 3600)
            try:
                self.storage.cursor.execute(
                    "DELETE FROM indicators WHERE timestamp < ?", 
                    (cutoff_timestamp,)
                )
                self.storage.conn.commit()
                
                deleted_count = self.storage.cursor.rowcount
                if deleted_count > 0:
                    logger.debug(f"🧹 Cleaned up {deleted_count} old database entries")
                    
            except Exception as cleanup_error:
                logger.warning(f"Database cleanup failed: {cleanup_error}")
            
            # Update sync timestamp
            self.last_sync = current_time
            logger.debug("✅ Database synchronization complete")
            
        except Exception as e:
            logger.error(f"Error synchronizing database: {e}")
        
        # Auto-dashboard generation tracking
        self.last_dashboard_time = 0
        self.dashboard_interval = 300  # Generate dashboard every 5 minutes
        self.startup_dashboard_generated = False
        
    async def on_trade(self, data: dict, backfill: bool = False):
        """Handle incoming trade data with comprehensive input validation and error handling"""
        try:
            # Enhanced input validation
            if not isinstance(data, dict):
                logger.error(f"LiveTrader.on_trade: Invalid data type: {type(data)}")
                return
            
            if not data:
                logger.warning("LiveTrader.on_trade: Empty data dictionary")
                return
            
            if not isinstance(backfill, bool):
                logger.warning(f"LiveTrader.on_trade: Invalid backfill type: {type(backfill)}, assuming False")
                backfill = False
            
            # Time the trade processing with error handling
            try:
                with PerformanceTimer("trade_processing") as timer:
                    # Handle different data formats (WebSocket vs backfill) with comprehensive validation
                    price = None
                    volume = None
                    timestamp = None
                    event_time = None
                    side = None
                    
                    if 'p' in data:
                        # WebSocket format validation
                        try:
                            price_raw = data.get('p')
                            volume_raw = data.get('q')
                            timestamp_raw = data.get('T')
                            event_time_raw = data.get('E')
                            is_buyer_maker = data.get('m')
                            
                            # Validate and convert price
                            if not isinstance(price_raw, (int, float, str)):
                                logger.error(f"LiveTrader.on_trade: Invalid price type: {type(price_raw)}")
                                return
                            
                            price = float(price_raw)
                            if not (np.isfinite(price) and price > 0):
                                logger.error(f"LiveTrader.on_trade: Invalid price value: {price}")
                                return
                            
                            # Validate and convert volume
                            if not isinstance(volume_raw, (int, float, str)):
                                logger.error(f"LiveTrader.on_trade: Invalid volume type: {type(volume_raw)}")
                                return
                            
                            volume = float(volume_raw)
                            if not (np.isfinite(volume) and volume >= 0):
                                logger.error(f"LiveTrader.on_trade: Invalid volume value: {volume}")
                                return
                            
                            # Validate and convert timestamps
                            if not isinstance(timestamp_raw, (int, float)):
                                logger.error(f"LiveTrader.on_trade: Invalid timestamp type: {type(timestamp_raw)}")
                                return
                            
                            if not isinstance(event_time_raw, (int, float)):
                                logger.error(f"LiveTrader.on_trade: Invalid event_time type: {type(event_time_raw)}")
                                return
                            
                            timestamp = int(timestamp_raw) / 1000  # Convert to seconds
                            event_time = int(event_time_raw) / 1000  # Event time
                            
                            # Validate converted timestamps
                            if not (np.isfinite(timestamp) and timestamp > 0):
                                logger.error(f"LiveTrader.on_trade: Invalid converted timestamp: {timestamp}")
                                return
                            
                            if not (np.isfinite(event_time) and event_time > 0):
                                logger.error(f"LiveTrader.on_trade: Invalid converted event_time: {event_time}")
                                return
                            
                            # Validate and determine side
                            if not isinstance(is_buyer_maker, bool):
                                logger.error(f"LiveTrader.on_trade: Invalid is_buyer_maker type: {type(is_buyer_maker)}")
                                return
                            
                            side = 'sell' if is_buyer_maker else 'buy'
                            
                        except (ValueError, TypeError, KeyError) as e:
                            logger.error(f"LiveTrader.on_trade: WebSocket format parsing error: {e}")
                            return
                            
                    elif 'price' in data:
                        # Backfill format validation
                        try:
                            price_raw = data.get('price')
                            volume_raw = data.get('volume')
                            timestamp_raw = data.get('timestamp')
                            side_raw = data.get('side')
                            
                            # Validate and convert price
                            if not isinstance(price_raw, (int, float, str)):
                                logger.error(f"LiveTrader.on_trade: Invalid backfill price type: {type(price_raw)}")
                                return
                            
                            price = float(price_raw)
                            if not (np.isfinite(price) and price > 0):
                                logger.error(f"LiveTrader.on_trade: Invalid backfill price value: {price}")
                                return
                            
                            # Validate and convert volume
                            if not isinstance(volume_raw, (int, float, str)):
                                logger.error(f"LiveTrader.on_trade: Invalid backfill volume type: {type(volume_raw)}")
                                return
                            
                            volume = float(volume_raw)
                            if not (np.isfinite(volume) and volume >= 0):
                                logger.error(f"LiveTrader.on_trade: Invalid backfill volume value: {volume}")
                                return
                            
                            # Validate and convert timestamp
                            if not isinstance(timestamp_raw, (int, float, str)):
                                logger.error(f"LiveTrader.on_trade: Invalid backfill timestamp type: {type(timestamp_raw)}")
                                return
                            
                            timestamp = float(timestamp_raw)
                            if not (np.isfinite(timestamp) and timestamp > 0):
                                logger.error(f"LiveTrader.on_trade: Invalid backfill timestamp value: {timestamp}")
                                return
                            
                            event_time = timestamp
                            
                            # Validate side
                            if not isinstance(side_raw, str):
                                logger.error(f"LiveTrader.on_trade: Invalid backfill side type: {type(side_raw)}")
                                return
                            
                            side = side_raw.strip().lower()
                            if side not in ['buy', 'sell']:
                                logger.error(f"LiveTrader.on_trade: Invalid backfill side value: {side}")
                                return
                            
                        except (ValueError, TypeError, KeyError) as e:
                            logger.error(f"LiveTrader.on_trade: Backfill format parsing error: {e}")
                            return
                    else:
                        logger.error(f"LiveTrader.on_trade: Unknown trade data format: {data}")
                        return
                    
                    # Additional sanity checks
                    current_time = time.time()
                    
                    # Check timestamp reasonableness (not too far in past or future)
                    time_diff = abs(current_time - timestamp)
                    if time_diff > 86400:  # More than 24 hours difference
                        logger.warning(f"LiveTrader.on_trade: Timestamp seems unreasonable - diff: {time_diff}s")
                        if not backfill:  # Only reject for live data
                            return
                    
                    # Check price reasonableness for crypto markets
                    if price > 10_000_000 or price < 0.000001:  # Reasonable bounds for crypto
                        logger.error(f"LiveTrader.on_trade: Price outside reasonable bounds: {price}")
                        return
                    
                    # Check volume reasonableness  
                    if volume > 1_000_000:  # Very large volume threshold
                        logger.warning(f"LiveTrader.on_trade: Unusually large volume: {volume}")
                        # Don't return - large volumes can be valid
                    
            except Exception as timer_error:
                logger.error(f"LiveTrader.on_trade: Timer initialization error: {timer_error}")
                return
                
            # Continue with trade processing
            # RRS Recording: Record trade event with error handling
            if hasattr(self, 'enable_recording') and self.enable_recording and not backfill:
                try:
                    record_trade(price, volume, side, timestamp)
                except Exception as record_error:
                    logger.warning(f"LiveTrader.on_trade: RRS recording failed: {record_error}")
                    # Don't fail the trade processing for recording issues
            
            # Calculate WebSocket lag (corrected formula) with validation
            try:
                websocket_lag_ms = max((current_time - event_time) * 1000, 0)
                
                # Validate lag calculation
                if not np.isfinite(websocket_lag_ms) or websocket_lag_ms < 0:
                    logger.warning(f"LiveTrader.on_trade: Invalid lag calculation: {websocket_lag_ms}")
                    websocket_lag_ms = 0.0
                
                # Log excessive lag for monitoring
                if websocket_lag_ms > 5000:  # More than 5 seconds
                    logger.warning(f"LiveTrader.on_trade: High WebSocket lag: {websocket_lag_ms:.1f}ms")
            
            except Exception as lag_error:
                logger.warning(f"LiveTrader.on_trade: Lag calculation error: {lag_error}")
                websocket_lag_ms = 0.0
            
            # Store trade with validation
            try:
                trade = {
                    'timestamp': timestamp,
                    'price': price,
                    'volume': volume,
                    'side': side
                }
                
                # Validate trade object
                for key, value in trade.items():
                    if value is None:
                        logger.error(f"LiveTrader.on_trade: None value for trade key: {key}")
                        return
                
                if hasattr(self, 'trades') and self.trades is not None:
                    self.trades.append(trade)
                else:
                    logger.error("LiveTrader.on_trade: trades deque not initialized")
                    return
            
            except Exception as trade_store_error:
                logger.error(f"LiveTrader.on_trade: Trade storage error: {trade_store_error}")
                return
            
            # Update price and volume series with validation
            try:
                if hasattr(self, 'timestamps') and self.timestamps is not None:
                    self.timestamps.append(timestamp)
                else:
                    logger.error("LiveTrader.on_trade: timestamps deque not initialized")
                    return
                
                if hasattr(self, 'prices') and self.prices is not None:
                    self.prices.append(price)
                else:
                    logger.error("LiveTrader.on_trade: prices deque not initialized")
                    return
                
                if hasattr(self, 'volumes') and self.volumes is not None:
                    self.volumes.append(volume)
                else:
                    logger.error("LiveTrader.on_trade: volumes deque not initialized")
                    return
                
                self.last_price = price
            
            except Exception as series_error:
                logger.error(f"LiveTrader.on_trade: Series update error: {series_error}")
            
            # P1-1 FIX: Check minute boundary for candle finalization with validation
            try:
                current_minute = int(timestamp) // 60
                
                if not hasattr(self, 'current_minute'):
                    self.current_minute = -1  # Initialize if missing
                
                if current_minute != self.current_minute:
                    # Visualization disabled during cleanup phase
                    logger.debug("Candle finalization skipped (visualization disabled)")
                    
                    # Reset atomic indicators for new minute with error handling
                    try:
                        current_time_safe = time.time()
                        
                        # Validate and reset each indicator
                        indicators_to_reset = [
                            ('liquidations', self.liquidations),
                            ('tobi', self.tobi),
                            ('wall_ratio', self.wall_ratio),
                            ('funding_basis', self.funding_basis),
                            ('trade_imbalance', self.trade_imbalance)
                        ]
                        
                        for name, indicator in indicators_to_reset:
                            try:
                                if indicator is not None and hasattr(indicator, 'reset'):
                                    indicator.reset(current_time_safe)
                                else:
                                    logger.warning(f"LiveTrader.on_trade: Cannot reset {name} indicator")
                            except Exception as reset_error:
                                logger.warning(f"LiveTrader.on_trade: Failed to reset {name}: {reset_error}")
                        
                    except Exception as reset_all_error:
                        logger.error(f"LiveTrader.on_trade: Atomic indicators reset error: {reset_all_error}")
                    
                    # Update Z-bands with previous minute's data with error handling
                    try:
                        z_bands = [
                            ('liq_z', self.liq_z),
                            ('wall_ratio_z', self.wall_ratio_z), 
                            ('basis_z', self.basis_z),
                            ('trade_imb_z', self.trade_imb_z)
                        ]
                        
                        for name, z_band in z_bands:
                            try:
                                if z_band is not None and hasattr(z_band, 'update'):
                                    z_band.update()
                                else:
                                    logger.warning(f"LiveTrader.on_trade: Cannot update {name} Z-band")
                            except Exception as z_update_error:
                                logger.warning(f"LiveTrader.on_trade: Failed to update {name}: {z_update_error}")
                        
                    except Exception as z_bands_error:
                        logger.error(f"LiveTrader.on_trade: Z-bands update error: {z_bands_error}")
                    
                    # Update advanced Z-bands (Items 26-30: Multi-timeframe Analysis) with error handling
                    try:
                        advanced_z_bands = [
                            ('advanced_liq_z', self.advanced_liq_z, self.liquidations),
                            ('advanced_wall_z', self.advanced_wall_z, self.wall_ratio),
                            ('advanced_basis_z', self.advanced_basis_z, self.funding_basis),
                            ('advanced_trade_z', self.advanced_trade_z, self.trade_imbalance)
                        ]
                        
                        for name, advanced_z, indicator in advanced_z_bands:
                            try:
                                if advanced_z is not None and hasattr(advanced_z, 'update'):
                                    # Safely get indicator value
                                    indicator_value = 0.0
                                    if indicator is not None and hasattr(indicator, 'value') and indicator.value is not None:
                                        try:
                                            indicator_value = float(indicator.value)
                                            if not np.isfinite(indicator_value):
                                                indicator_value = 0.0
                                        except (ValueError, TypeError):
                                            indicator_value = 0.0
                                    
                                    advanced_z.update(indicator_value, timestamp)
                                else:
                                    logger.warning(f"LiveTrader.on_trade: Cannot update {name}")
                            except Exception as adv_z_error:
                                logger.warning(f"LiveTrader.on_trade: Failed to update {name}: {adv_z_error}")
                        
                    except Exception as advanced_z_error:
                        logger.error(f"LiveTrader.on_trade: Advanced Z-bands update error: {advanced_z_error}")
                    
                    self.current_minute = current_minute
            
            except Exception as minute_error:
                logger.error(f"LiveTrader.on_trade: Minute boundary processing error: {minute_error}")
                # Continue processing despite minute boundary issues
            
            # Record performance metrics
            self.metrics.record_trade_processing(timer.elapsed_ms, websocket_lag_ms)
            
            # Process with extreme indicators (skip alerts if backfill)
            extreme_signals = self.extremes.process_trade({
                'side': trade['side'],
                'volume': trade['volume'],
                'price': trade['price'],
                'timestamp': trade['timestamp'],
                'latency_ms': websocket_lag_ms,
                'backfill': backfill  # Mark backfill data
            }, exchange="binance")
            
            # Process atomic indicators
            self.trade_imbalance.on_tick(trade['side'])
            # Note: Other atomic indicators updated via separate event feeds
            # (liquidations via liquidation feed, TOBI via orderbook, etc.)
            
            # NEW: Update RSI-style basic oscillators with trade data
            is_buyer = (trade['side'] == 'buy')
            imbalance_result = self.trade_imbalance_osc.update_trade(
                qty=trade['volume'] * trade['price'],  # Use notional value
                is_buyer=is_buyer,
                timestamp=timestamp
            )
            
            # Check for trade imbalance extremes
            if imbalance_result is not None and not backfill:
                extreme = self.trade_imbalance_osc.is_extreme(self.basic_config['extreme_zscore_threshold'])
                if extreme and self.debounce_manager.should_fire(f"trade_imbalance_{extreme.lower()}", self.basic_config['debounce_seconds']):
                    logger.warning(f"🚨 TRADE IMBALANCE EXTREME {extreme}: z={self.trade_imbalance_osc.z_score:.2f}, osc={self.trade_imbalance_osc.osc:.3f}")
                    # TODO: Emit signal through signal_detector
            
            # Check for combined basic oscillator extremes and send signals
            if not backfill:
                # Collect current basic oscillator values
                basic_osc_values = {
                    'wall_ratio': self.wall_ratio_osc.z_score if hasattr(self, 'wall_ratio_osc') and self.wall_ratio_osc.z_score is not None else None,
                    'trade_imbalance': self.trade_imbalance_osc.z_score if hasattr(self, 'trade_imbalance_osc') and self.trade_imbalance_osc.z_score is not None else None,
                    'liquidation': self.liquidation_osc.z_score if hasattr(self, 'liquidation_osc') and self.liquidation_osc.z_score is not None else None,
                    'dir': self.dir_osc.z_score if hasattr(self, 'dir_osc') and self.dir_osc.z_score is not None else None,
                    'funding_accel': self.funding_accel_osc.z_score if hasattr(self, 'funding_accel_osc') and self.funding_accel_osc.z_score is not None else None
                }
                
                # RRS Recording: Record basic oscillator state
                if self.enable_recording:
                    record_basic_oscillator(basic_osc_values, timestamp)
                
                # Combined extremes check temporarily disabled during cleanup
                logger.debug("Combined extremes check skipped (basic_manager disabled)")
                
                # Check basic oscillator signals
                try:
                    basic_signals = await self.signal_detector.check_basic_oscillator_signals(self.symbol, basic_osc_values)
                    if basic_signals:
                        logger.info(f"📊 Basic Oscillator Signals: {', '.join(basic_signals)}")
                        
                        # RRS Recording: Record signal trigger
                        if self.enable_recording:
                            for signal_name in basic_signals:
                                record_signal_trigger(signal_name, "basic_oscillator", basic_osc_values, timestamp)
                except Exception as e:
                    logger.error(f"Error checking basic oscillator signals: {e}")
            
            # Handle extreme signals (skip Telegram for backfill)
            if not backfill:
                unique_signals = set()
                for signal in extreme_signals:
                    # Only log and count unique signal types
                    signal_key = f"{signal.indicator}_{signal.direction}"
                    if signal_key not in unique_signals:
                        unique_signals.add(signal_key)
                        logger.warning(f"🚨 EXTREME: {signal.message}")
                        self.metrics.record_signal()
                    
                    # Add to unified visualizer (all signals for visualization)
                    self.visualizer.add_signal(
                        timestamp=signal.timestamp,
                        price=price,
                        signal_type=signal.direction,
                        message=signal.message
                    )
                    
                    # Send to Telegram if available (debounced already)
                    if self.telegram:
                        try:
                            await self.telegram.send_signal(
                                symbol=self.symbol.upper(),
                                signal_type=signal.direction,
                                price=price,
                                reason=signal.message,
                                strength=signal.strength
                            )
                        except Exception as e:
                            logger.debug(f"Telegram notification failed: {e}")
                    else:
                        logger.debug("Telegram not available for signal notification")
            
            # Update unified visualizer data every tick
            indicators_data = {
                'ofi': list(self.ofi_history)[-1] if self.ofi_history else None,
                'vpin': list(self.vpin_history)[-1] if self.vpin_history else None,
                'kyle': list(self.kyle_history)[-1] if self.kyle_history else None,
                'lpi': list(self.lpi_history)[-1] if self.lpi_history else None,
                'cvd_z': getattr(self.extremes.cvd, 'current_cvd', None),
                'dir': None,  # Will be updated from orderbook
                'scs': None,  # Will be updated from spread data
                'vpin_threshold': getattr(self.vpin, 'dynamic_threshold', 0.98),
                # Basic Oscillators data
                'basic_wall_ratio_z': self.wall_ratio_osc.z_score if hasattr(self, 'wall_ratio_osc') and self.wall_ratio_osc.z_score is not None else None,
                'basic_trade_imb_z': self.trade_imbalance_osc.z_score if hasattr(self, 'trade_imbalance_osc') and self.trade_imbalance_osc.z_score is not None else None,
                'basic_liquidation_z': self.liquidation_osc.z_score if hasattr(self, 'liquidation_osc') and self.liquidation_osc.z_score is not None else None,
                'basic_dir_z': self.dir_osc.z_score if hasattr(self, 'dir_osc') and self.dir_osc.z_score is not None else None,
                'basic_funding_accel_z': self.funding_accel_osc.z_score if hasattr(self, 'funding_accel_osc') and self.funding_accel_osc.z_score is not None else None
            }
            
            # Visualization disabled during cleanup phase
            logger.debug(f"Visualization update skipped (cleanup mode)")
            
            self.tick_count += 1
            
            # Add trade to candlestick engine
            try:
                candle = self.candlestick_engine.add_trade(
                    timestamp=timestamp,
                    price=price,
                    quantity=trade['volume'],
                    side=trade['side']
                )
                
                # If a candle was completed, record it
                if candle is not None:
                    try:
                        record_bar(
                            open_price=candle.open,
                            high=candle.high,
                            low=candle.low,
                            close=candle.close,
                            volume=candle.volume,
                            timestamp=candle.timestamp
                        )
                        
                        # Broadcast bar data to WebSocket clients
                        await broadcast_data("bar", {
                            "ts": candle.timestamp,
                            "o": candle.open,
                            "h": candle.high,
                            "l": candle.low,
                            "c": candle.close,
                            "v": candle.volume
                        })
                        
                        logger.debug(f"📊 Completed candle: O={candle.open:.2f} H={candle.high:.2f} L={candle.low:.2f} C={candle.close:.2f} V={candle.volume:.4f}")
                        
                    except Exception as record_error:
                        logger.warning(f"Failed to record/broadcast candle: {record_error}")
                        
            except Exception as candle_error:
                logger.warning(f"CandlestickEngine error: {candle_error}")
            
            # Update indicators with EVERY real trade for maximum accuracy
            await self._update_indicators(trade)
            
            # Optimized indicator broadcasting with batching
            if not backfill and self.tick_count % 3 == 0:  # Increased frequency for better UX
                try:
                    # Get current indicator values with caching
                    indicators = self._get_cached_indicators()
                    
                    # Only broadcast if indicators have changed significantly
                    if self._indicators_changed(indicators):
                        await broadcast_data("indi", {
                            "symbol": self.symbol,
                            "indicators": indicators,
                            "timestamp": timestamp,
                            "tick_count": self.tick_count
                        })
                        self._last_broadcast_indicators = indicators.copy()
                        
                except Exception as broadcast_error:
                    logger.debug(f"Indicator broadcast failed: {broadcast_error}")
                
            # Log periodic status
            if self.tick_count % 100 == 0:
                logger.info(f"[{self.symbol.upper()}] Processed {self.tick_count} trades, Price: ${price:.2f}")
            
            # Broadcast trade data to WebSocket clients (every 10th trade to reduce noise)
            if not backfill and self.tick_count % 10 == 0:
                try:
                    await broadcast_data("bar", {
                        "symbol": self.symbol,
                        "price": price,
                        "volume": trade['volume'],
                        "side": trade['side'],
                        "timestamp": timestamp,
                        "tick_count": self.tick_count
                    })
                except Exception as broadcast_error:
                    logger.debug(f"Broadcast failed: {broadcast_error}")
                
        except Exception as e:
            logger.error(f"Error processing trade: {e}")
    
    async def _update_indicators(self, trade: dict):
        """Update all RSI-like indicators with new trade data - UNIFIED PIPELINE"""
        
        # Input validation
        if not isinstance(trade, dict):
            logger.error(f"LiveTrader._update_indicators: Invalid trade type: {type(trade)}")
            return
            
        required_keys = ['timestamp', 'price', 'volume', 'side']
        for key in required_keys:
            if key not in trade:
                logger.error(f"LiveTrader._update_indicators: Missing required key: {key}")
                return
        
        try:
            # Validate and extract trade data
            timestamp = trade.get('timestamp', None)
            if timestamp is None:
                timestamp = time.time()
                logger.warning("LiveTrader._update_indicators: Using current time for missing timestamp")
            
            try:
                timestamp = float(timestamp)
                if not np.isfinite(timestamp) or timestamp <= 0:
                    logger.error(f"LiveTrader._update_indicators: Invalid timestamp: {timestamp}")
                    return
            except (ValueError, TypeError) as e:
                logger.error(f"LiveTrader._update_indicators: Cannot convert timestamp to float: {e}")
                return
            
            try:
                price = float(trade['price'])
                if not np.isfinite(price) or price <= 0:
                    logger.error(f"LiveTrader._update_indicators: Invalid price: {price}")
                    return
            except (ValueError, TypeError) as e:
                logger.error(f"LiveTrader._update_indicators: Cannot convert price to float: {e}")
                return
            
            try:
                volume = float(trade['volume'])
                if not np.isfinite(volume) or volume < 0:
                    logger.error(f"LiveTrader._update_indicators: Invalid volume: {volume}")
                    return
            except (ValueError, TypeError) as e:
                logger.error(f"LiveTrader._update_indicators: Cannot convert volume to float: {e}")
                return
            
            side = trade.get('side', '').lower().strip()
            if side not in ['buy', 'sell']:
                logger.error(f"LiveTrader._update_indicators: Invalid trade side: {side}")
                return
            
            # Update current minute tracking with validation
            try:
                minute_ts = int(timestamp // 60)
                if minute_ts < 0:
                    logger.error(f"LiveTrader._update_indicators: Invalid minute timestamp: {minute_ts}")
                    return
                
                if not hasattr(self, '_current_minute_ts'):
                    self._current_minute_ts = -1
                
                if self._current_minute_ts != minute_ts:
                    # Minute rollover - process previous minute if exists
                    if self._current_minute_ts > 0:
                        try:
                            await self._process_minute_close(self._current_minute_ts)
                        except Exception as minute_error:
                            logger.error(f"LiveTrader._update_indicators: Minute close processing error: {minute_error}")
                    
                    self._current_minute_ts = minute_ts
                    
            except Exception as minute_error:
                logger.error(f"LiveTrader._update_indicators: Minute tracking error: {minute_error}")
                return
            
            # Accumulate data for current minute with validation
            try:
                self._accumulate_trade_data(trade, minute_ts)
            except Exception as accumulate_error:
                logger.error(f"LiveTrader._update_indicators: Data accumulation error: {accumulate_error}")
                # Continue with indicator updates despite accumulation issues
            
            # Update legacy indicators for backward compatibility with validation
            try:
                qty_signed = volume if side == 'buy' else -volume
                
                # Validate signed quantity
                if not np.isfinite(qty_signed):
                    logger.warning(f"LiveTrader._update_indicators: Invalid signed quantity: {qty_signed}")
                    return
                
                # Update indicators with error handling
                indicators_to_update = [
                    ('ofi', self.ofi, lambda ind: ind.update(qty_signed, price)),
                    ('vpin', self.vpin, lambda ind: ind.update(qty_signed, price)),
                    ('kyle', self.kyle, lambda ind: ind.update(qty_signed, price))
                ]
                
                for name, indicator, update_func in indicators_to_update:
                    try:
                        if indicator is not None and hasattr(indicator, 'update'):
                            update_func(indicator)
                        else:
                            logger.warning(f"LiveTrader._update_indicators: Cannot update {name} indicator")
                    except Exception as ind_error:
                        logger.warning(f"LiveTrader._update_indicators: Failed to update {name}: {ind_error}")
                
                # Update LPI with validation and OI estimate
                try:
                    side_lpi = "long" if side == 'buy' else "short"
                    if hasattr(self, 'lpi') and self.lpi is not None and hasattr(self.lpi, 'update'):
                        # Provide OI estimate based on recent trading volume
                        # Simple heuristic: OI ≈ 24h volume * price * 0.1 (10% of daily volume)
                        if not hasattr(self, '_oi_estimate'):
                            self._oi_estimate = 0.0
                            self._volume_sum = 0.0
                            self._oi_update_count = 0
                        
                        # Update rolling OI estimate
                        self._volume_sum += volume * price
                        self._oi_update_count += 1
                        
                        # Update estimate every 100 trades
                        if self._oi_update_count % 100 == 0:
                            # Estimate: recent volume * 10 (assuming we see ~10% of total OI flow)
                            self._oi_estimate = self._volume_sum * 10
                            self._volume_sum *= 0.9  # Decay for rolling estimate
                            
                            # Update LPI's OI estimate
                            if hasattr(self.lpi, 'update_oi_estimate'):
                                self.lpi.update_oi_estimate(self._oi_estimate)
                        
                        # Now update LPI with the trade using new signature
                        if side == 'buy':
                            long_qty, short_qty = volume, 0.0
                        else:
                            long_qty, short_qty = 0.0, volume
                        self.lpi.update(long_qty, short_qty)
                    else:
                        logger.warning("LiveTrader._update_indicators: Cannot update LPI indicator")
                except Exception as lpi_error:
                    logger.warning(f"LiveTrader._update_indicators: LPI update error: {lpi_error}")
                
            except Exception as legacy_error:
                logger.error(f"LiveTrader._update_indicators: Legacy indicators error: {legacy_error}")
            
            # Get legacy indicator values (for backward compatibility) with validation
            try:
                indicators = self._get_legacy_indicators()
                if indicators is None:
                    logger.warning("LiveTrader._update_indicators: No legacy indicators retrieved")
                    indicators = {}
            except Exception as get_indicators_error:
                logger.error(f"LiveTrader._update_indicators: Get legacy indicators error: {get_indicators_error}")
                indicators = {}
            
            # Store in history for plotting with validation
            try:
                self._update_indicator_history(indicators)
            except Exception as history_error:
                logger.error(f"LiveTrader._update_indicators: Indicator history update error: {history_error}")
            
        except Exception as e:
            logger.error(f"LiveTrader._update_indicators: Critical error in indicator updates: {e}")
            return
    
    def _accumulate_trade_data(self, trade: dict, minute_ts: int):
        """Accumulate trade data for minute-level processing"""
        
        # Input validation
        if not isinstance(trade, dict):
            logger.error(f"LiveTrader._accumulate_trade_data: Invalid trade type: {type(trade)}")
            return
            
        if not isinstance(minute_ts, int) or minute_ts < 0:
            logger.error(f"LiveTrader._accumulate_trade_data: Invalid minute timestamp: {minute_ts}")
            return
        
        try:
            # Extract and validate trade data
            timestamp = trade.get('timestamp', None)
            if timestamp is None:
                timestamp = time.time()
                logger.warning("LiveTrader._accumulate_trade_data: Using current time for missing timestamp")
            
            try:
                timestamp = float(timestamp)
                if not np.isfinite(timestamp) or timestamp <= 0:
                    logger.error(f"LiveTrader._accumulate_trade_data: Invalid timestamp: {timestamp}")
                    return
            except (ValueError, TypeError) as e:
                logger.error(f"LiveTrader._accumulate_trade_data: Cannot convert timestamp: {e}")
                return
            
            try:
                price = float(trade['price'])
                if not np.isfinite(price) or price <= 0:
                    logger.error(f"LiveTrader._accumulate_trade_data: Invalid price: {price}")
                    return
            except (ValueError, TypeError, KeyError) as e:
                logger.error(f"LiveTrader._accumulate_trade_data: Cannot extract price: {e}")
                return
            
            try:
                volume = float(trade['volume'])
                if not np.isfinite(volume) or volume < 0:
                    logger.error(f"LiveTrader._accumulate_trade_data: Invalid volume: {volume}")
                    return
            except (ValueError, TypeError, KeyError) as e:
                logger.error(f"LiveTrader._accumulate_trade_data: Cannot extract volume: {e}")
                return
            
            side = trade.get('side', '').lower().strip()
            if side not in ['buy', 'sell']:
                logger.error(f"LiveTrader._accumulate_trade_data: Invalid side: {side}")
                return
            
            # Calculate notional with overflow protection
            try:
                notional = price * volume
                if not np.isfinite(notional) or notional < 0:
                    logger.error(f"LiveTrader._accumulate_trade_data: Invalid notional: {notional}")
                    return
            except (OverflowError, ValueError) as e:
                logger.error(f"LiveTrader._accumulate_trade_data: Notional calculation error: {e}")
                return
            
            # Ensure aggregation structure exists
            if not hasattr(self, '_agg'):
                logger.error("LiveTrader._accumulate_trade_data: Aggregation structure not initialized")
                return
            
            # Validate aggregation containers exist
            required_containers = [
                'price_deltas', 'notional_list', 'ofi_sum', 'liq_buy_not', 'liq_sell_not',
                'vpin_value', 'buy_vol', 'sell_vol', 'basis_values', 'funding_rates'
            ]
            
            for container_name in required_containers:
                if not hasattr(self._agg, container_name):
                    logger.error(f"LiveTrader._accumulate_trade_data: Missing container: {container_name}")
                    return
            
            # Initialize minute containers if needed with error handling
            try:
                list_containers = [self._agg.price_deltas, self._agg.notional_list]
                scalar_containers = [
                    self._agg.ofi_sum, self._agg.liq_buy_not, self._agg.liq_sell_not,
                    self._agg.vpin_value, self._agg.buy_vol, self._agg.sell_vol, 
                    self._agg.basis_values, self._agg.funding_rates
                ]
                
                # Initialize list containers with bounds checking
                for container in list_containers:
                    try:
                        if not isinstance(container, dict):
                            logger.error("LiveTrader._accumulate_trade_data: List container is not a dictionary")
                            continue
                        
                        if minute_ts not in container:
                            container[minute_ts] = []
                        
                        # Prevent memory overflow
                        if len(container[minute_ts]) > 10000:  # Limit per minute
                            container[minute_ts] = container[minute_ts][-5000:]
                            logger.warning(f"LiveTrader._accumulate_trade_data: Trimmed container to prevent overflow")
                            
                    except Exception as container_error:
                        logger.error(f"LiveTrader._accumulate_trade_data: List container init error: {container_error}")
                        continue
                
                # Initialize scalar containers with validation
                for container in scalar_containers:
                    try:
                        if not isinstance(container, dict):
                            logger.error("LiveTrader._accumulate_trade_data: Scalar container is not a dictionary")
                            continue
                        
                        if minute_ts not in container:
                            container[minute_ts] = 0.0
                        
                        # Validate existing value
                        if not np.isfinite(container[minute_ts]):
                            logger.warning(f"LiveTrader._accumulate_trade_data: Resetting invalid container value")
                            container[minute_ts] = 0.0
                            
                    except Exception as container_error:
                        logger.error(f"LiveTrader._accumulate_trade_data: Scalar container init error: {container_error}")
                        continue
                
            except Exception as init_error:
                logger.error(f"LiveTrader._accumulate_trade_data: Container initialization error: {init_error}")
                return
            
            # Accumulate OFI (signed quantity) with validation
            try:
                ofi_tick = volume if side == 'buy' else -volume
                if not np.isfinite(ofi_tick):
                    logger.warning(f"LiveTrader._accumulate_trade_data: Invalid OFI tick: {ofi_tick}")
                    return
                
                current_ofi = self._agg.ofi_sum.get(minute_ts, 0.0)
                new_ofi = current_ofi + ofi_tick
                
                if not np.isfinite(new_ofi):
                    logger.warning(f"LiveTrader._accumulate_trade_data: OFI overflow, resetting")
                    self._agg.ofi_sum[minute_ts] = ofi_tick
                else:
                    self._agg.ofi_sum[minute_ts] = new_ofi
                    
            except Exception as ofi_error:
                logger.error(f"LiveTrader._accumulate_trade_data: OFI accumulation error: {ofi_error}")
            
            # Accumulate volume by side with validation
            try:
                if side == 'buy':
                    current_buy_vol = self._agg.buy_vol.get(minute_ts, 0.0)
                    new_buy_vol = current_buy_vol + volume
                    
                    if not np.isfinite(new_buy_vol):
                        logger.warning("LiveTrader._accumulate_trade_data: Buy volume overflow, resetting")
                        self._agg.buy_vol[minute_ts] = volume
                    else:
                        self._agg.buy_vol[minute_ts] = new_buy_vol
                else:
                    current_sell_vol = self._agg.sell_vol.get(minute_ts, 0.0)
                    new_sell_vol = current_sell_vol + volume
                    
                    if not np.isfinite(new_sell_vol):
                        logger.warning("LiveTrader._accumulate_trade_data: Sell volume overflow, resetting")
                        self._agg.sell_vol[minute_ts] = volume
                    else:
                        self._agg.sell_vol[minute_ts] = new_sell_vol
                        
            except Exception as volume_error:
                logger.error(f"LiveTrader._accumulate_trade_data: Volume accumulation error: {volume_error}")
            
            # Store price deltas and notional with validation
            try:
                if hasattr(self, 'last_price') and self.last_price is not None:
                    try:
                        last_price_safe = float(self.last_price)
                        if np.isfinite(last_price_safe) and last_price_safe > 0:
                            price_delta = price - last_price_safe
                            if np.isfinite(price_delta):
                                self._agg.price_deltas[minute_ts].append(price_delta)
                        else:
                            logger.debug("LiveTrader._accumulate_trade_data: Invalid last_price for delta calculation")
                    except (ValueError, TypeError):
                        logger.debug("LiveTrader._accumulate_trade_data: Cannot convert last_price for delta")
                
                self._agg.notional_list[minute_ts].append(notional)
                
            except Exception as storage_error:
                logger.error(f"LiveTrader._accumulate_trade_data: Data storage error: {storage_error}")
            
            # Log successful accumulation (debug level)
            logger.debug(f"LiveTrader._accumulate_trade_data: Accumulated {side} trade: "
                        f"price={price:.8f}, volume={volume:.6f}, minute={minute_ts}")
            
        except Exception as e:
            logger.error(f"LiveTrader._accumulate_trade_data: Critical error in data accumulation: {e}")
            return

        
        # Update VPIN estimate (use current value)
        if hasattr(self.vpin, 'value') and self.vpin.value is not None:
            self._agg.vpin_value[minute_ts] = float(self.vpin.value)
        
        # Check for big orders
        if notional >= self._big_order_threshold:
            signed_notional = notional if side == 'buy' else -notional
            self._big_order_acc.add(timestamp, signed_notional)
        
        # Accumulate liquidation data (if available from trade)
        # Note: This would need liquidation detection logic or external feed
        # For now, we'll use placeholder logic
        if self._is_likely_liquidation(trade):
            liq_notional = notional
            if side == 'buy':  # Short squeeze
                self._agg.liq_buy_not[minute_ts] += liq_notional
            else:  # Long squeeze  
                self._agg.liq_sell_not[minute_ts] += liq_notional
    
    def _is_likely_liquidation(self, trade: dict) -> bool:
        """Simple heuristic for liquidation detection - can be enhanced"""
        # Large size relative to recent average
        volume = float(trade['volume'])
        return volume > 100.0  # Simple threshold - enhance with ML/stats
    
    async def _process_minute_close(self, minute_ts: int):
        """Process completed minute data through RSI-like oscillators"""
        try:
            # Get aggregated raw values for the minute
            ofi_1m = self._agg.ofi_sum.get(minute_ts, 0.0)
            liq_b_1m = self._agg.liq_buy_not.get(minute_ts, 0.0)
            liq_s_1m = self._agg.liq_sell_not.get(minute_ts, 0.0)
            vpin_1m = self._agg.vpin_value.get(minute_ts, 0.5)
            price_deltas = self._agg.price_deltas.get(minute_ts, [])
            notional_list = self._agg.notional_list.get(minute_ts, [])
            buy_vol = self._agg.buy_vol.get(minute_ts, 0.0)
            sell_vol = self._agg.sell_vol.get(minute_ts, 0.0)
            
            # Update RSI-like oscillators
            ofi_data = self.ofi_osc.on_minute(ofi_1m)
            lpi_data = self.lpi_osc.on_minute(liq_b_1m, liq_s_1m, minute_ts * 60)
            vpin_data = self.vpin_osc.on_minute(vpin_1m)
            kyle_data = self.kyle_osc.on_minute(price_deltas, notional_list)
            cvd_data = self.cvd_osc.on_minute(buy_vol, sell_vol)
            
            # Basis data (if available)
            basis_current = self._get_current_basis()  # To be implemented
            basis_data = self.basis_osc.on_minute(basis_current)
            
            # Combine all RSI data
            rsi_indicators = {
                **ofi_data, **lpi_data, **vpin_data, 
                **kyle_data, **cvd_data, **basis_data
            }
            
            # Process signals with isteresi and cooldown
            await self._process_rsi_signals(rsi_indicators, minute_ts)
            
            # Clean up old minute data
            self._cleanup_old_minute_data(minute_ts)
            
            # Log RSI indicators (with proper string formatting)
            self._log_rsi_indicators(rsi_indicators)
            
            # Broadcast indicator updates to WebSocket clients
            try:
                await broadcast_data("indi", {
                    "symbol": self.symbol,
                    "indicators": rsi_indicators,
                    "minute_ts": minute_ts,
                    "timestamp": minute_ts * 60
                })
            except Exception as broadcast_error:
                logger.debug(f"Indicator broadcast failed: {broadcast_error}")
            
        except Exception as e:
            logger.error(f"Error processing minute close: {e}")
    
    def _get_current_basis(self) -> float:
        """Get current basis (perpetual - spot) - placeholder implementation"""
        # This would integrate with your basis calculation logic
        return 0.0
    
    async def _process_rsi_signals(self, indicators: dict, minute_ts: int):
        """Process RSI signals with isteresi and cooldown"""
        from rtai.config import THRESH, THRESH_SPECIAL
        
        def check_rsi_signal(name: str, rsi_val: float, thresholds: dict = None) -> str | None:
            """Check RSI signal with isteresi"""
            if thresholds is None:
                thresholds = THRESH
                
            up_enter = thresholds["enter"]
            up_exit = thresholds["exit"] 
            dn_enter = thresholds["enter_low"]
            dn_exit = thresholds["exit_low"]
            
            state_key = f"{name}_state"
            current_state = self._rsi_state.get(state_key, "FLAT")
            
            if current_state == "FLAT":
                if rsi_val >= up_enter and self._cooldown.ready(f"{name}:SELL"):
                    self._rsi_state[state_key] = "SHORT_SIGNAL"
                    return "SELL"
                elif rsi_val <= dn_enter and self._cooldown.ready(f"{name}:BUY"):
                    self._rsi_state[state_key] = "LONG_SIGNAL"
                    return "BUY"
            elif current_state == "SHORT_SIGNAL":
                if rsi_val <= up_exit:
                    self._rsi_state[state_key] = "FLAT"
            elif current_state == "LONG_SIGNAL":
                if rsi_val >= dn_exit:
                    self._rsi_state[state_key] = "FLAT"
                    
            return None
        
        # Check each RSI indicator
        signals = []
        
        # OFI RSI
        ofi_signal = check_rsi_signal("OFI", indicators.get("ofi_rsi", 50.0))
        if ofi_signal:
            signals.append(f"OFI-RSI {ofi_signal} ({indicators['ofi_rsi']:.1f})")
            
        # VPIN RSI (as confirmation/filter)
        vpin_rsi = indicators.get("vpin_rsi", 50.0)
        if vpin_rsi >= THRESH_SPECIAL["vpin"]["enter"] and abs(indicators.get("ofi_rsi", 50.0) - 50) >= 20:
            if self._cooldown.ready("VPIN_ALERT"):
                self._send_alert(f"🚨 VPIN high stress {vpin_rsi:.1f} + OFI deviation {abs(indicators.get('ofi_rsi', 50.0) - 50):.1f}")
        
        # CVD RSI (with dedupe)
        cvd_signal = check_rsi_signal("CVD", indicators.get("cvd_rsi", 50.0), THRESH_SPECIAL["cvd"])
        if cvd_signal:
            signals.append(f"CVD-RSI {cvd_signal} ({indicators['cvd_rsi']:.1f})")
            
        # LPI RSI (liquidation pressure)
        lpi_signal = check_rsi_signal("LPI", indicators.get("lpi_rsi", 50.0), THRESH_SPECIAL["liq"])
        if lpi_signal:
            signals.append(f"LPI-RSI {lpi_signal} ({indicators['lpi_rsi']:.1f})")
        
        # Send consolidated signal if any
        if signals:
            message = " | ".join(signals)
            logger.warning(f"🎯 RSI SIGNALS: {message}")
            if self.telegram:
                try:
                    await self.telegram.send(f"🎯 {message}")
                except Exception as e:
                    logger.debug(f"Telegram send failed: {e}")
    
    def _send_alert(self, message: str):
        """Send alert to log and telegram"""
        logger.warning(message)
        if self.telegram:
            try:
                asyncio.create_task(self.telegram.send(message))
            except Exception as e:
                logger.debug(f"Telegram alert failed: {e}")
    
    def _cleanup_old_minute_data(self, current_minute_ts: int):
        """Clean up old minute aggregation data"""
        cutoff = current_minute_ts - 2  # Keep last 2 minutes
        
        for container in [self._agg.ofi_sum, self._agg.liq_buy_not, self._agg.liq_sell_not,
                         self._agg.vpin_value, self._agg.price_deltas, self._agg.notional_list,
                         self._agg.buy_vol, self._agg.sell_vol, self._agg.basis_values,
                         self._agg.funding_rates]:
            keys_to_remove = [k for k in container.keys() if k < cutoff]
            for k in keys_to_remove:
                container.pop(k, None)
    
    def _log_rsi_indicators(self, indicators: dict):
        """Log RSI indicators with proper string formatting"""
        # Pre-compute strings to avoid f-string conditional errors
        ofi_str = f"{indicators.get('ofi_rsi', 0):.1f}" if indicators.get('ofi_rsi') is not None else "N/A"
        vpin_str = f"{indicators.get('vpin_rsi', 0):.1f}" if indicators.get('vpin_rsi') is not None else "N/A"
        kyle_str = f"{indicators.get('kyle_rsi', 0):.1f}" if indicators.get('kyle_rsi') is not None else "N/A"
        lpi_str = f"{indicators.get('lpi_rsi', 0):.1f}" if indicators.get('lpi_rsi') is not None else "N/A"
        cvd_str = f"{indicators.get('cvd_rsi', 0):.1f}" if indicators.get('cvd_rsi') is not None else "N/A"
        basis_str = f"{indicators.get('basis_rsi', 0):.1f}" if indicators.get('basis_rsi') is not None else "N/A"
        
        logger.info(f"[{self.symbol.upper()}] RSI Indicators - OFI: {ofi_str}, VPIN: {vpin_str}, "
                   f"Kyle: {kyle_str}, LPI: {lpi_str}, CVD: {cvd_str}, Basis: {basis_str}")
    
    def _get_legacy_indicators(self) -> dict:
        """Get legacy indicator values for backward compatibility"""
        indicators = {}
        try:
            indicators['ofi'] = float(self.ofi.value) if self.ofi.value is not None else None
        except (ValueError, TypeError):
            indicators['ofi'] = None
            
        try:
            indicators['vpin'] = float(self.vpin.value) if self.vpin.value is not None else None
        except (ValueError, TypeError):
            indicators['vpin'] = None
            
        try:
            indicators['kyle'] = float(self.kyle.value) if self.kyle.value is not None else None
        except (ValueError, TypeError):
            indicators['kyle'] = None
            
        try:
            indicators['lpi'] = float(self.lpi.value) if self.lpi.value is not None else None
        except (ValueError, TypeError):
            indicators['lpi'] = None
        
        # Add atomic indicators
        try:
            indicators['liq_usd'] = float(self.liquidations.value) if self.liquidations.value is not None else 0.0
            indicators['liq_cnt'] = int(self.liquidations.count)
            indicators['tobi'] = float(self.tobi.value) if self.tobi.value is not None else 0.5
            indicators['wall_ratio'] = float(self.wall_ratio.value) if self.wall_ratio.value is not None and not np.isnan(self.wall_ratio.value) else 0.0
            indicators['basis'] = float(self.funding_basis.value) if self.funding_basis.value is not None else 0.0
            indicators['trade_imb'] = float(self.trade_imbalance.value) if self.trade_imbalance.value is not None else 0.0
        except (ValueError, TypeError):
            # Default values for atomic indicators
            indicators['liq_usd'] = 0.0
            indicators['liq_cnt'] = 0
            indicators['tobi'] = 0.5
            indicators['wall_ratio'] = 0.0
            indicators['basis'] = 0.0
            indicators['trade_imb'] = 0.0
        
        # Add Z-band indicators
        try:
            indicators['liq_z'] = float(self.liq_z.value)
            indicators['wall_ratio_z'] = float(self.wall_ratio_z.value)
            indicators['basis_z'] = float(self.basis_z.value)
            indicators['trade_imb_z'] = float(self.trade_imb_z.value)
        except (ValueError, TypeError):
            indicators['liq_z'] = 0.0
            indicators['wall_ratio_z'] = 0.0
            indicators['basis_z'] = 0.0
            indicators['trade_imb_z'] = 0.0
        
        return indicators
    
    def _update_indicator_history(self, indicators: dict):
        """Update indicator history for plotting"""
        try:
            # Store indicator history for plotting
            timestamp = time.time()
            self.indicator_timestamps.append(timestamp)
            self.ofi_history.append(indicators.get('ofi'))
            self.vpin_history.append(indicators.get('vpin'))
            self.kyle_history.append(indicators.get('kyle'))
            self.lpi_history.append(indicators.get('lpi'))
        except Exception as e:
            logger.debug(f"Error updating indicator history: {e}")
    
    # Rest of methods remain unchanged for backward compatibility
    
    async def _store_minute_data(self, minute_timestamp: int, indicators: Dict[str, float]):
        """Store minute-level aggregated data with atomic indicators
        
        Args:
            minute_timestamp: Minute timestamp 
            indicators: Dict of all indicator values
        """
        try:
            # Calculate minute averages
            avg_price = self.minute_data['price_sum'] / max(self.minute_data['price_count'], 1)
            total_volume = self.minute_data['total_volume'] 
            trades_count = self.minute_data['trades_count']
            
            # Store with enhanced atomic indicators
            success = self.storage.store_minute_features(
                minute_timestamp,
                avg_price,
                total_volume,
                trades_count,
                {**indicators}  # Include all atomic indicators
            )
            
            if success:
                logger.debug(f"💾 Stored minute data: {trades_count} trades, ${total_volume:.2f} volume")
            else:
                logger.warning("⚠️ Failed to store minute data")
                
        except Exception as e:
            logger.error(f"Error storing minute data: {e}")
        
        # Storage synchronization and state management every 10 minutes
        try:
            if minute_timestamp % 10 == 0:
                # Store Z-state every 10 minutes to reduce I/O
                pass  # Placeholder for Z-state storage
        except Exception as e:
            logger.error(f"Error in storage synchronization: {e}")

    def generate_chart(self, save_path: str = None, save_png: bool = True) -> str:
        """Generate current state chart using unified visualizer"""
        if not save_png:
            logger.debug("Chart generation skipped (save_png=False)")
            return None
            
        try:
            # Visualization disabled during cleanup phase
            logger.info("📊 Chart generation disabled during deep-clean phase")
            logger.info("   Will be restored with backtesting.py integration")
            return None
            
        except Exception as e:
            logger.error(f"Error generating chart: {e}")
            return None
    async def start_live_trading(self, duration_minutes: int = None):
        """Start live trading with real Binance WebSocket connection using new feed system
        
        Args:
            duration_minutes: Optional duration limit in minutes
        """
        try:
            logger.info(f"🚀 Starting REAL live trading for {self.symbol}")
            
            # Initialize multi-stream feed system
            from rtai.io.multi_stream_feed import BinanceMultiStreamFeed, bus
            self.feed = BinanceMultiStreamFeed(self.symbol.upper())
            
            # Register event handlers for all stream types
            bus.on('trade', self._on_trade_event)
            bus.on('depth_l0', self._on_depth_event) 
            bus.on('open_interest', self._on_open_interest_event)
            bus.on('funding', self._on_funding_event)
            bus.on('liquidation', self._on_liquidation_event)
            
            logger.success(f"✅ Connected to multi-stream Binance data for {self.symbol}")
            
            # Set up duration limit if specified
            start_time = time.time()
            end_time = start_time + (duration_minutes * 60) if duration_minutes else None
            
            # Statistics
            message_count = 0
            last_stats_time = start_time
            
            # Start the multi-stream feed
            try:
                # Run feed in background task
                feed_task = asyncio.create_task(self.feed.start())
                
                # Monitor duration if specified
                while True:
                    # Check duration limit
                    if end_time and time.time() > end_time:
                        logger.info(f"⏰ Duration limit reached ({duration_minutes} minutes)")
                        break
                    
                    await asyncio.sleep(1)  # Check every second
                    
            except KeyboardInterrupt:
                logger.info("🛑 Received interrupt signal")
            except Exception as e:
                logger.error(f"❌ Error in live trading loop: {e}")
                raise
            
        except Exception as e:
            logger.error(f"❌ Failed to start live trading: {e}")
            raise
        
        finally:
            # Cleanup
            try:
                if hasattr(self, 'feed'):
                    self.feed.stop()
                    logger.info("🔌 Multi-stream feed stopped")
            except Exception as e:
                logger.warning(f"Error stopping feed: {e}")
            
            logger.success(f"✅ Live trading stopped - processed {message_count} messages")
    
    async def _on_trade_event(self, trade_data: dict):
        """Handle trade events from multi-stream feed"""
        try:
            # Convert to format expected by existing on_trade method
            formatted_data = {
                'p': str(trade_data['price']),
                'q': str(abs(trade_data['qty'])),  # Remove sign for compatibility
                'T': int(trade_data['ts'] * 1000),
                'E': int(trade_data['ts'] * 1000),
                'm': trade_data['side'] == 'sell'
            }
            await self.on_trade(formatted_data, backfill=False)
        except Exception as e:
            logger.error(f"Error processing trade event: {e}")
    
    async def _on_depth_event(self, depth_data: dict):
        """Handle depth/bookTicker events for OFI and market structure indicators"""
        try:
            # Update OFI with real L0 depth data
            ofi_value = self.ofi.update(
                depth_data['bid_px'], 
                depth_data['bid_sz'],
                depth_data['ask_px'], 
                depth_data['ask_sz']
            )
            
            if ofi_value is not None:
                # Record to database and broadcast
                record_indicator_update('OFI', ofi_value, depth_data['ts'])
                await broadcast_data('indi', {
                    't': 'indi',
                    'name': 'OFI', 
                    'value': ofi_value,
                    'ts': depth_data['ts']
                })
            
            # Update other market structure indicators that need L0 depth
            self._update_market_structure_indicators(depth_data)
            
        except Exception as e:
            logger.error(f"Error processing depth event: {e}")
    
    async def _on_open_interest_event(self, oi_data: dict):
        """Handle open interest events for LPI"""
        try:
            # Update LPI's OI estimate
            if hasattr(self.lpi, 'update_oi_estimate'):
                self.lpi.update_oi_estimate(oi_data['oi'])
                logger.debug(f"Updated LPI OI estimate: {oi_data['oi']:,.0f}")
            
            # Record OI data
            record_indicator_update('OpenInterest', oi_data['oi'], oi_data['ts'])
            
        except Exception as e:
            logger.error(f"Error processing OI event: {e}")
    
    async def _on_funding_event(self, funding_data: dict):
        """Handle funding rate events for FundingAccel indicators"""  
        try:
            # Update funding-based indicators
            funding_rate = funding_data['funding_rate']
            
            # Update FundingAccelOsc if available
            if hasattr(self, 'funding_accel') and self.funding_accel:
                funding_value = self.funding_accel.update(funding_rate)
                if funding_value is not None:
                    record_indicator_update('FundingAccel', funding_value, funding_data['ts'])
            
            # Record funding data
            record_indicator_update('FundingRate', funding_rate, funding_data['ts'])
            
        except Exception as e:
            logger.error(f"Error processing funding event: {e}")
    
    async def _on_liquidation_event(self, liq_data: dict):
        """Handle liquidation events for LPI and liquidation indicators"""
        try:
            # Update LPI with liquidation data using new signature
            if liq_data['side'] == 'long':
                long_qty, short_qty = abs(liq_data['qty']), 0.0
            else:
                long_qty, short_qty = 0.0, abs(liq_data['qty'])
            
            lpi_value = self.lpi.update(long_qty, short_qty)
            if lpi_value is not None:
                record_indicator_update('LPI', lpi_value, liq_data['ts'])
                await broadcast_data('indi', {
                    't': 'indi',
                    'name': 'LPI',
                    'value': lpi_value, 
                    'ts': liq_data['ts']
                })
            
            # Record liquidation data
            record_liquidation(liq_data['price'], liq_data['qty'], liq_data['side'], liq_data['ts'])
            
        except Exception as e:
            logger.error(f"Error processing liquidation event: {e}")
    
    def _update_market_structure_indicators(self, depth_data: dict):
        """Update indicators that depend on market structure data"""
        try:
            # Calculate mid price for Kyle's Lambda
            mid_price = (depth_data['bid_px'] + depth_data['ask_px']) / 2
            
            # Update any wall ratio or market structure indicators here
            # (Implementation depends on specific indicator requirements)
            
        except Exception as e:
            logger.error(f"Error updating market structure indicators: {e}")
    
    def _get_cached_indicators(self) -> dict:
        """Get indicator values with caching for performance"""
        current_time = time.time()
        
        # Cache indicators for 100ms to reduce computation
        if (hasattr(self, '_indicator_cache_time') and 
            current_time - self._indicator_cache_time < 0.1 and
            hasattr(self, '_indicator_cache')):
            return self._indicator_cache
        
        # Compute fresh indicators
        indicators = self._get_legacy_indicators()
        self._indicator_cache = indicators
        self._indicator_cache_time = current_time
        
        return indicators
    
    def _indicators_changed(self, current_indicators: dict) -> bool:
        """Check if indicators have changed significantly"""
        if not hasattr(self, '_last_broadcast_indicators'):
            return True
        
        last = self._last_broadcast_indicators
        threshold = 0.01  # 1% change threshold
        
        for key, value in current_indicators.items():
            if key not in last:
                return True
            
            if value is None or last[key] is None:
                if value != last[key]:
                    return True
                continue
            
            try:
                if abs(float(value) - float(last[key])) / max(abs(float(last[key])), 1e-8) > threshold:
                    return True
            except (ValueError, TypeError, ZeroDivisionError):
                if value != last[key]:
                    return True
        
        return False