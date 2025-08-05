"""Utility functions for RTAI system"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, Optional, Callable

import httpx
import websockets
from loguru import logger

# Logging utilities (using loguru directly)
def configure_logging(log_level: str = "INFO", performance_logging: bool = False):
    """Configure logging with loguru"""
    logger.remove()
    logger.add(
        "logs/rtai_{time:YYYY-MM-DD}.log",
        level=log_level,
        rotation="1 day",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    )
    logger.add(
        lambda msg: print(msg, end=""),
        level=log_level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan> | {message}"
    )

def get_log_level():
    """Get current log level"""
    return "INFO"

from .metrics import PerformanceMetrics, PerformanceTimer
from .backfill import BackfillManager

# Import structured logging functions for easier access
try:
    from .structured_logging import (
        get_structured_logger,
        log_indicator_calculation,
        log_trade_processed,
        log_feed_event,
        log_error_with_context
    )
except ImportError:
    # Fallback stubs if structured logging not available
    def get_structured_logger(name: str):
        return logger
    def log_indicator_calculation(*args, **kwargs):
        pass
    def log_trade_processed(*args, **kwargs):
        pass  
    def log_feed_event(*args, **kwargs):
        pass
    def log_error_with_context(*args, **kwargs):
        pass

# Rate limiting for API calls
_last_api_call = 0
_API_RATE_LIMIT = 0.1  # 100ms between calls


def binance_ts_to_unix(binance_ts: int) -> float:
    """Convert Binance timestamp (milliseconds) to Unix timestamp (seconds)"""
    return binance_ts / 1000


async def http_get(url: str, *, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """HTTP GET with rate limiting and error handling"""
    global _last_api_call
    
    # Rate limiting
    now = time.time()
    if now - _last_api_call < _API_RATE_LIMIT:
        await asyncio.sleep(_API_RATE_LIMIT - (now - _last_api_call))
    _last_api_call = time.time()
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()


class BinanceWebSocket:
    """Real Binance WebSocket client for live trading data"""
    
    def __init__(self, max_reconnect_attempts: int = 10, reconnect_delay: float = 1.0):
        self.websocket = None
        self.max_reconnect_attempts = max_reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.reconnect_count = 0
        self.symbol = None
        
        # Connection state
        self.stop_event = asyncio.Event()
        self.is_connected = False
        
        # Connection health monitoring
        self.last_ping_time = 0.0
        self.last_pong_time = 0.0
        self.connection_start_time = 0.0
        self.total_messages_received = 0
        self.total_bytes_received = 0
        
        # Performance metrics
        self.message_queue_size = 0
        self.max_queue_size = 10000  # Overflow protection
        self.connection_stats = {
            'reconnections': 0,
            'total_uptime': 0.0,
            'avg_latency': 0.0,
            'last_error': None
        }
        
    async def ping(self) -> bool:
        """Send ping and measure response time"""
        if not self.websocket:
            return False
            
        try:
            self.last_ping_time = time.time()
            await self.websocket.ping()
            # Pong handler will update last_pong_time
            return True
        except Exception as e:
            logger.warning(f"WebSocket ping failed: {e}")
            return False
    
    async def handle_pong(self, data: bytes) -> None:
        """Handle pong response and calculate latency"""
        self.last_pong_time = time.time()
        if self.last_ping_time > 0:
            latency = (self.last_pong_time - self.last_ping_time) * 1000  # ms
            self.connection_stats['avg_latency'] = latency
            
    def is_connection_healthy(self) -> bool:
        """Check if connection is healthy based on recent activity"""
        current_time = time.time()
        
        # Connection is unhealthy if no pong for 30+ seconds
        if self.last_pong_time > 0 and current_time - self.last_pong_time > 30:
            return False
            
        # Connection is unhealthy if queue is overflowing
        if self.message_queue_size > self.max_queue_size:
            logger.warning(f"Message queue overflow: {self.message_queue_size}")
            return False
            
        return self.websocket is not None
        
    async def connect_with_backoff(self, uri: str) -> bool:
        """Connect with exponential backoff retry - Capped at 300s per audit"""
        attempt = 0
        
        while attempt < self.max_reconnect_attempts:
            try:
                delay = min(self.reconnect_delay * (2 ** attempt), 300.0)  # Cap at 300s per audit
                if attempt > 0:
                    logger.info(f"Reconnection attempt {attempt + 1}/{self.max_reconnect_attempts} after {delay:.1f}s")
                    await asyncio.sleep(delay)
                
                self.websocket = await websockets.connect(uri, ping_interval=20, ping_timeout=10)
                self.connection_start_time = time.time()
                self.reconnect_count = 0
                self.connection_stats['reconnections'] += 1
                
                logger.info(f"✅ WebSocket connected to {uri}")
                return True
                
            except Exception as e:
                attempt += 1
                self.connection_stats['last_error'] = str(e)
                logger.warning(f"Connection attempt {attempt} failed: {e}")
                
        logger.error(f"❌ Failed to connect after {self.max_reconnect_attempts} attempts")
        return False
        
    def update_connection_stats(self, message_size: int) -> None:
        """Update connection statistics"""
        self.total_messages_received += 1
        self.total_bytes_received += message_size
        
        if self.connection_start_time > 0:
            self.connection_stats['total_uptime'] = time.time() - self.connection_start_time
            
    async def close_gracefully(self) -> None:
        """Gracefully close WebSocket connection"""
        if self.websocket:
            try:
                await self.websocket.close()
                logger.info("📡 WebSocket connection closed gracefully")
            except Exception as e:
                logger.warning(f"Error during WebSocket close: {e}")
            finally:
                self.websocket = None
                
    def get_connection_health_report(self) -> Dict[str, Any]:
        """Get detailed connection health report"""
        current_time = time.time()
        
        return {
            'is_connected': self.websocket is not None,
            'is_healthy': self.is_connection_healthy(),
            'uptime_seconds': current_time - self.connection_start_time if self.connection_start_time > 0 else 0,
            'total_messages': self.total_messages_received,
            'total_bytes': self.total_bytes_received,
            'avg_latency_ms': self.connection_stats['avg_latency'],
            'reconnection_count': self.connection_stats['reconnections'],
            'queue_size': self.message_queue_size,
            'last_ping': current_time - self.last_ping_time if self.last_ping_time > 0 else -1,
            'last_pong': current_time - self.last_pong_time if self.last_pong_time > 0 else -1,
            'last_error': self.connection_stats['last_error']
        }
        
    async def connect(self, symbol: str = "BTCUSDT"):
        """Connect to Binance WebSocket for real-time data"""
        self.symbol = symbol.upper()
        uri = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@trade"
        
        logger.info(f"🔌 Connecting to Binance WebSocket: {uri}")
        
        try:
            self.websocket = await websockets.connect(
                uri,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10
            )
            self.is_connected = True
            self.connection_start_time = time.time()
            logger.success(f"✅ Connected to {symbol.upper()} trade stream")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Binance WebSocket: {e}")
            self.is_connected = False
            return False
    
    async def receive_message(self):
        """Receive and parse message from WebSocket with robust error handling"""
        if not self.websocket or not self.is_connected:
            return None
        
        retry_count = 0
        max_retries = 3
        
        while retry_count < max_retries:
            try:
                # Receive raw message with timeout
                raw_message = await asyncio.wait_for(
                    self.websocket.recv(), 
                    timeout=5.0
                )
                
                # Parse JSON message
                message = json.loads(raw_message)
                
                # Update statistics
                self.update_connection_stats(len(raw_message))
                
                # Convert Binance format to RTAI format with validation
                if 'e' in message and message['e'] == 'trade':
                    try:
                        return {
                            'symbol': str(message['s']),
                            'price': float(message['p']),
                            'volume': float(message['q']),
                            'timestamp': float(message['T']) / 1000,  # Convert to seconds
                            'side': 'buy' if message['m'] == False else 'sell',  # m=false means buyer is market maker
                            'trade_id': int(message['t']),
                            'event_time': float(message['E']) / 1000
                        }
                    except (KeyError, ValueError, TypeError) as e:
                        logger.warning(f"⚠️ Invalid trade message format: {e}")
                        return None
                
                return message
                
            except asyncio.TimeoutError:
                # Timeout is normal, return None
                return None
            except websockets.exceptions.ConnectionClosed:
                logger.warning("🔌 WebSocket connection closed, attempting reconnect...")
                self.is_connected = False
                
                # Attempt reconnection
                if await self._attempt_reconnect():
                    retry_count += 1
                    continue
                else:
                    return None
                    
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ Failed to parse WebSocket message: {e}")
                retry_count += 1
                await asyncio.sleep(0.1)  # Brief pause before retry
                continue
                
            except Exception as e:
                logger.error(f"❌ Error receiving WebSocket message: {e}")
                retry_count += 1
                await asyncio.sleep(0.1)
                continue
        
        logger.error(f"❌ Max retries ({max_retries}) exceeded for WebSocket receive")
        return None
    
    async def _attempt_reconnect(self) -> bool:
        """Attempt to reconnect WebSocket with exponential backoff"""
        if not self.symbol:
            return False
        
        try:
            await asyncio.sleep(min(2 ** self.reconnect_count, 30))  # Exponential backoff, max 30s
            return await self.connect(self.symbol)
        except Exception as e:
            logger.error(f"❌ Reconnection failed: {e}")
            self.reconnect_count += 1
            return False
    
    async def disconnect(self):
        """Disconnect from WebSocket"""
        self.stop_event.set()
        if self.websocket:
            try:
                await self.websocket.close()
                logger.info("🔌 WebSocket disconnected")
            except Exception as e:
                logger.warning(f"Error during disconnect: {e}")
        self.is_connected = False
        self.websocket = None
    
    async def connect_trade_stream(self, symbol: str, callback: Callable):
        """Connect to Binance trade stream with callback (legacy method)"""
        await self.connect(symbol)
        
        while not self.stop_event.is_set() and self.is_connected:
            try:
                message = await self.receive_message()
                if message:
                    await callback(message)
                else:
                    # Small delay to prevent CPU spinning
                    await asyncio.sleep(0.001)
                    
            except Exception as e:
                logger.error(f"Error in trade stream callback: {e}")
                await asyncio.sleep(1)
    
    async def close(self):
        """Close WebSocket connection"""
        logger.info("Closing WebSocket connection...")
        self.stop_event.set()
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False


async def connect_with_backoff(uri: str, *, initial: float = 1.0, maximum: float = 60.0, **ws_kwargs):
    """WebSocket connection with exponential backoff"""
    delay = initial
    while True:
        try:
            async with websockets.connect(uri, **ws_kwargs) as ws:
                delay = initial
                yield ws
        except websockets.exceptions.ConnectionClosed:
            await asyncio.sleep(delay)
            delay = min(delay * 2, maximum)
        except Exception as e:
            print(f"Connection error: {e}")
            await asyncio.sleep(delay)
            delay = min(delay * 2, maximum)
