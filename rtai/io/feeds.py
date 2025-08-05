"""
RTAI Unified Feed System
========================

Unified interface for both live WebSocket feeds and replay data streams.
Consolidates BinanceWebSocket from utils/environment.py into a single feed interface.
Enhanced with health monitoring and structured logging.
"""

import asyncio
import json
import time
from abc import ABC, abstractmethod
from typing import AsyncIterator, Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path

import websockets
import pandas as pd
from loguru import logger

# Import health monitoring
try:
    from ..utils.health import record_trade, record_error, set_feed_status
    from ..utils.structured_logging import get_structured_logger, correlation_context, log_feed_event
    HEALTH_MONITORING_AVAILABLE = True
except ImportError:
    HEALTH_MONITORING_AVAILABLE = False
    logger.warning("Health monitoring not available - running without metrics")


@dataclass
class Trade:
    """Unified trade data structure"""
    timestamp: float
    price: float
    volume: float
    side: str  # 'buy' or 'sell'
    symbol: str
    source: str = "unknown"


class FeedBase(ABC):
    """Abstract base class for data feeds"""
    
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol.upper()
        self.is_running = False
    
    @abstractmethod
    async def __aiter__(self) -> AsyncIterator[Trade]:
        """Async iterator yielding Trade objects"""
        pass
    
    async def start(self):
        """Start the feed"""
        self.is_running = True
        logger.info(f"Started {self.__class__.__name__} for {self.symbol}")
    
    async def stop(self):
        """Stop the feed"""
        self.is_running = False
        logger.info(f"Stopped {self.__class__.__name__} for {self.symbol}")


class BinanceLiveTrades(FeedBase):
    """Live Binance WebSocket trade feed with health monitoring"""
    
    def __init__(self, symbol: str = "BTCUSDT"):
        super().__init__(symbol)
        self.websocket = None
        self.reconnect_delay = 5.0
        self.max_reconnects = 10
        self._reconnect_count = 0
        self.structured_logger = get_structured_logger("binance_feed") if HEALTH_MONITORING_AVAILABLE else None
    
    async def __aiter__(self) -> AsyncIterator[Trade]:
        """Stream live trades from Binance WebSocket"""
        await self.start()
        
        while self.is_running:
            try:
                await self._connect()
                
                if self.websocket:
                    async for message in self.websocket:
                        if not self.is_running:
                            break
                            
                        try:
                            data = json.loads(message)
                            trade = self._parse_binance_trade(data)
                            if trade:
                                # Record trade for health monitoring
                                if HEALTH_MONITORING_AVAILABLE:
                                    record_trade()
                                    if self.structured_logger:
                                        self.structured_logger.debug(
                                            f"📈 Trade received: {trade.symbol} @ {trade.price}",
                                            symbol=trade.symbol,
                                            price=trade.price,
                                            volume=trade.volume,
                                            side=trade.side
                                        )
                                
                                yield trade
                                
                        except json.JSONDecodeError as e:
                            if HEALTH_MONITORING_AVAILABLE:
                                record_error("json_decode")
                            logger.warning(f"BinanceLiveTrades: Invalid JSON: {e}")
                            continue
                        except Exception as e:
                            if HEALTH_MONITORING_AVAILABLE:
                                record_error("trade_parsing")
                            logger.error(f"BinanceLiveTrades: Trade parsing error: {e}")
                            continue
                            
            except websockets.exceptions.ConnectionClosed:
                if HEALTH_MONITORING_AVAILABLE:
                    set_feed_status(False)
                    record_error("connection_closed")
                logger.warning("BinanceLiveTrades: WebSocket connection closed")
                await self._handle_reconnect()
            except Exception as e:
                if HEALTH_MONITORING_AVAILABLE:
                    set_feed_status(False)
                    record_error("connection_error")
                logger.error(f"BinanceLiveTrades: Connection error: {e}")
                await self._handle_reconnect()
    
    async def _connect(self):
        """Establish WebSocket connection"""
        try:
            uri = f"wss://stream.binance.com:9443/ws/{self.symbol.lower()}@trade"
            logger.info(f"BinanceLiveTrades: Connecting to {uri}")
            
            self.websocket = await websockets.connect(
                uri,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            
            logger.success(f"BinanceLiveTrades: Connected to {self.symbol} trade stream")
            self._reconnect_count = 0  # Reset on successful connection
            
            # Update health status
            if HEALTH_MONITORING_AVAILABLE:
                set_feed_status(True)
                if self.structured_logger:
                    self.structured_logger.info(
                        f"📡 WebSocket connected to {self.symbol}",
                        event_type="connection_established",
                        symbol=self.symbol,
                        uri=uri
                    )
            
        except Exception as e:
            logger.error(f"BinanceLiveTrades: Connection failed: {e}")
            if HEALTH_MONITORING_AVAILABLE:
                set_feed_status(False)
                record_error("connection_failed")
            self.websocket = None
            raise
    
    async def _handle_reconnect(self):
        """Handle reconnection logic"""
        if self._reconnect_count >= self.max_reconnects:
            logger.error(f"BinanceLiveTrades: Max reconnection attempts reached ({self.max_reconnects})")
            self.is_running = False
            return
        
        self._reconnect_count += 1
        logger.info(f"BinanceLiveTrades: Reconnecting in {self.reconnect_delay}s (attempt {self._reconnect_count}/{self.max_reconnects})")
        
        await asyncio.sleep(self.reconnect_delay)
        
        # Exponential backoff
        self.reconnect_delay = min(self.reconnect_delay * 1.5, 60.0)
    
    def _parse_binance_trade(self, data: Dict[str, Any]) -> Optional[Trade]:
        """Parse Binance WebSocket trade message"""
        try:
            # Validate required fields
            required_fields = ['p', 'q', 'T', 'E', 'm', 's']
            for field in required_fields:
                if field not in data:
                    logger.warning(f"BinanceLiveTrades: Missing field '{field}'")
                    return None
            
            # Extract and validate data
            price = float(data['p'])
            volume = float(data['q'])
            timestamp = int(data['T']) / 1000  # Convert to seconds
            event_time = int(data['E']) / 1000
            is_buyer_maker = data['m']
            symbol = data['s']
            
            # Validate numerical values
            if not (price > 0 and volume > 0 and timestamp > 0):
                logger.warning(f"BinanceLiveTrades: Invalid values: price={price}, volume={volume}, timestamp={timestamp}")
                return None
            
            # Determine side
            side = 'sell' if is_buyer_maker else 'buy'
            
            return Trade(
                timestamp=timestamp,
                price=price,
                volume=volume,
                side=side,
                symbol=symbol,
                source="binance_live"
            )
            
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"BinanceLiveTrades: Parse error: {e}")
            return None
    
    async def stop(self):
        """Stop the live feed"""
        await super().stop()
        
        if self.websocket:
            try:
                await self.websocket.close()
            except:
                pass
            self.websocket = None


class ParquetReplayTrades(FeedBase):
    """Replay trades from Parquet files"""
    
    def __init__(self, file_path: Union[str, Path], symbol: str = "BTCUSDT", speed_multiplier: float = 1.0):
        super().__init__(symbol)
        self.file_path = Path(file_path)
        self.speed_multiplier = max(0.1, speed_multiplier)  # Minimum 0.1x speed
        self.df = None
        self._start_time = None
        self._replay_start = None
    
    async def __aiter__(self) -> AsyncIterator[Trade]:
        """Stream trades from Parquet file"""
        await self.start()
        
        if not self._load_data():
            logger.error(f"ParquetReplayTrades: Failed to load data from {self.file_path}")
            return
        
        logger.info(f"ParquetReplayTrades: Starting replay of {len(self.df)} trades at {self.speed_multiplier}x speed")
        
        self._start_time = time.time()
        first_timestamp = self.df.iloc[0]['timestamp']
        self._replay_start = first_timestamp
        
        for _, row in self.df.iterrows():
            if not self.is_running:
                break
            
            # Calculate replay timing
            elapsed_real = time.time() - self._start_time
            elapsed_data = row['timestamp'] - self._replay_start
            target_delay = elapsed_data / self.speed_multiplier
            
            if target_delay > elapsed_real:
                sleep_time = target_delay - elapsed_real
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            # Create Trade object
            trade = Trade(
                timestamp=row['timestamp'],
                price=float(row['price']),
                volume=float(row['volume']),
                side=str(row['side']).lower(),
                symbol=self.symbol,
                source="parquet_replay"
            )
            
            yield trade
        
        logger.info("ParquetReplayTrades: Replay completed")
    
    def _load_data(self) -> bool:
        """Load trade data from Parquet file"""
        try:
            if not self.file_path.exists():
                logger.error(f"ParquetReplayTrades: File not found: {self.file_path}")
                return False
            
            self.df = pd.read_parquet(self.file_path)
            
            # Validate required columns
            required_columns = ['timestamp', 'price', 'volume', 'side']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            
            if missing_columns:
                logger.error(f"ParquetReplayTrades: Missing columns: {missing_columns}")
                return False
            
            # Sort by timestamp
            self.df = self.df.sort_values('timestamp').reset_index(drop=True)
            
            # Filter by symbol if column exists
            if 'symbol' in self.df.columns:
                symbol_matches = self.df['symbol'].str.upper() == self.symbol.upper()
                self.df = self.df[symbol_matches].reset_index(drop=True)
            
            if len(self.df) == 0:
                logger.error(f"ParquetReplayTrades: No data for symbol {self.symbol}")
                return False
            
            logger.info(f"ParquetReplayTrades: Loaded {len(self.df)} trades for {self.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"ParquetReplayTrades: Failed to load data: {e}")
            return False


class CSVReplayTrades(FeedBase):
    """Replay trades from CSV files (legacy format support)"""
    
    def __init__(self, file_path: Union[str, Path], symbol: str = "BTCUSDT", speed_multiplier: float = 1.0):
        super().__init__(symbol)
        self.file_path = Path(file_path)
        self.speed_multiplier = max(0.1, speed_multiplier)
        self.df = None
        self._start_time = None
        self._replay_start = None
    
    async def __aiter__(self) -> AsyncIterator[Trade]:
        """Stream trades from CSV file"""
        await self.start()
        
        if not self._load_data():
            logger.error(f"CSVReplayTrades: Failed to load data from {self.file_path}")
            return
        
        logger.info(f"CSVReplayTrades: Starting replay of {len(self.df)} trades at {self.speed_multiplier}x speed")
        
        self._start_time = time.time()
        first_timestamp = self.df.iloc[0]['timestamp']
        self._replay_start = first_timestamp
        
        for _, row in self.df.iterrows():
            if not self.is_running:
                break
            
            # Calculate replay timing
            elapsed_real = time.time() - self._start_time
            elapsed_data = row['timestamp'] - self._replay_start
            target_delay = elapsed_data / self.speed_multiplier
            
            if target_delay > elapsed_real:
                sleep_time = target_delay - elapsed_real
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            # Create Trade object
            trade = Trade(
                timestamp=row['timestamp'],
                price=float(row['price']),
                volume=float(row['volume']),
                side=str(row['side']).lower(),
                symbol=self.symbol,
                source="csv_replay"
            )
            
            yield trade
        
        logger.info("CSVReplayTrades: Replay completed")
    
    def _load_data(self) -> bool:
        """Load trade data from CSV file"""
        try:
            if not self.file_path.exists():
                logger.error(f"CSVReplayTrades: File not found: {self.file_path}")
                return False
            
            self.df = pd.read_csv(self.file_path)
            
            # Validate required columns
            required_columns = ['timestamp', 'price', 'volume', 'side']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            
            if missing_columns:
                logger.error(f"CSVReplayTrades: Missing columns: {missing_columns}")
                return False
            
            # Sort by timestamp
            self.df = self.df.sort_values('timestamp').reset_index(drop=True)
            
            # Filter by symbol if column exists
            if 'symbol' in self.df.columns:
                symbol_matches = self.df['symbol'].str.upper() == self.symbol.upper()
                self.df = self.df[symbol_matches].reset_index(drop=True)
            
            if len(self.df) == 0:
                logger.error(f"CSVReplayTrades: No data for symbol {self.symbol}")
                return False
            
            logger.info(f"CSVReplayTrades: Loaded {len(self.df)} trades for {self.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"CSVReplayTrades: Failed to load data: {e}")
            return False


# Factory function for creating feeds
def create_feed(feed_type: str, **kwargs) -> FeedBase:
    """
    Factory function to create feed instances
    
    Args:
        feed_type: Type of feed ('live', 'parquet', 'csv')
        **kwargs: Feed-specific parameters
    
    Returns:
        FeedBase instance
    """
    feed_types = {
        'live': BinanceLiveTrades,
        'binance': BinanceLiveTrades,
        'parquet': ParquetReplayTrades,
        'csv': CSVReplayTrades
    }
    
    if feed_type.lower() not in feed_types:
        raise ValueError(f"Unknown feed type: {feed_type}. Available: {list(feed_types.keys())}")
    
    feed_class = feed_types[feed_type.lower()]
    return feed_class(**kwargs)
