"""
RTAI Enhanced Multi-Stream Feed System
=====================================

Complete implementation of all Binance streams for comprehensive indicator coverage:
- Trade stream: VPIN, Kyle's Lambda, CVD, TradeImbalance
- Depth stream: OFI, WallRatio, DIR, market structure indicators  
- Open Interest stream: LPI with real OI estimates
- Funding stream: FundingAccel, FundingBasis indicators
- Liquidation stream: LPI liquidation pressure, SCS indicators

Centralized event bus with data cache integration for zero-coupling architecture.
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from pathlib import Path
import websockets
from loguru import logger

# Import centralized cache
from ..state.cache import cache

# Enhanced Event Bus with priority and filtering
class EventBus:
    def __init__(self):
        self._handlers = {}
        self._filters = {}
        self._stats = {}
    
    def on(self, event: str, handler: Callable, priority: int = 0, filter_func: Optional[Callable] = None):
        """Register event handler with optional priority and filtering"""
        if event not in self._handlers:
            self._handlers[event] = []
            self._stats[event] = {'emitted': 0, 'handled': 0, 'errors': 0}
        
        self._handlers[event].append({
            'handler': handler,
            'priority': priority,
            'filter': filter_func
        })
        
        # Sort by priority (higher first)
        self._handlers[event].sort(key=lambda x: x['priority'], reverse=True)
    
    async def emit(self, event: str, data: Dict[str, Any]):
        """Emit event to all registered handlers"""
        self._stats[event]['emitted'] += 1
        
        if event in self._handlers:
            for handler_info in self._handlers[event]:
                try:
                    # Apply filter if present
                    if handler_info['filter'] and not handler_info['filter'](data):
                        continue
                    
                    handler = handler_info['handler']
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                    
                    self._stats[event]['handled'] += 1
                    
                except Exception as e:
                    self._stats[event]['errors'] += 1
                    logger.error(f"Event handler error for {event}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics"""
        return self._stats.copy()

# Global enhanced event bus
bus = EventBus()

@dataclass
class StreamEndpoint:
    """Configuration for a WebSocket stream endpoint"""
    url: str
    stream_type: str  # 'trade', 'depth_l0', 'open_interest', 'funding', 'liquidation'
    topic: str        # Topic name for recording
    symbol: str
    priority: int = 0  # Stream priority for processing order
    
class BinanceMultiStreamFeed:
    """
    Enhanced Multi-Stream Binance Feed
    
    Supports all required streams for complete indicator coverage:
    - Trade stream (100-200 Hz): VPIN, Kyle's Lambda, CVD, TradeImbalance
    - BookTicker stream (10 Hz): OFI, WallRatio, DIR  
    - Open Interest stream (1 Hz): LPI OI estimates
    - Mark Price stream (1 Hz): FundingAccel, FundingBasis
    - Force Order stream (<1 Hz): Liquidation indicators
    """
    
    # Binance endpoint configurations
    ENDPOINTS = {
        'spot': {
            'base_url': 'wss://stream.binance.com:9443/ws',
            'streams': {
                'trade': '{symbol}@trade',
                'depth': '{symbol}@bookTicker'
            }
        },
        'futures': {
            'base_url': 'wss://fstream.binance.com/ws',
            'combined_url': 'wss://fstream.binance.com/stream',
            'streams': {
                'trade': '{symbol}@trade', 
                'depth': '{symbol}@bookTicker',
                'liquidation': '{symbol}@forceOrder',
                'oi_funding': '!openInterest@arr@1s/!markPrice@arr'  # Combined stream
            }
        }
    }
    
    def __init__(self, symbol: str = "BTCUSDT", use_futures: bool = True):
        self.symbol = symbol.upper()
        self.symbol_lower = symbol.lower()
        self.use_futures = use_futures
        self.is_running = False
        self.reconnect_count = 0
        self.max_reconnects = 10
        
        # Setup stream endpoints based on configuration
        self.endpoints = self._create_endpoints()
        
        # Connection tracking
        self.connections = {}
        self.connection_stats = {}
        
    def _create_endpoints(self) -> List[StreamEndpoint]:
        """Create stream endpoint configurations"""
        endpoints = []
        
        if self.use_futures:
            base_url = self.ENDPOINTS['futures']['base_url']
            combined_url = self.ENDPOINTS['futures']['combined_url']
            
            # Individual streams
            endpoints.extend([
                StreamEndpoint(
                    url=f"{base_url}/{self.symbol_lower}@trade",
                    stream_type="trade",
                    topic="trade", 
                    symbol=self.symbol,
                    priority=10  # High priority for trade data
                ),
                StreamEndpoint(
                    url=f"{base_url}/{self.symbol_lower}@bookTicker",
                    stream_type="depth_l0",
                    topic="depth_l0",
                    symbol=self.symbol,
                    priority=9  # High priority for depth
                ),
                StreamEndpoint(
                    url=f"{base_url}/{self.symbol_lower}@forceOrder",
                    stream_type="liquidation", 
                    topic="liquidation",
                    symbol=self.symbol,
                    priority=5  # Medium priority
                )
            ])
            
            # Combined OI + Funding stream
            endpoints.append(
                StreamEndpoint(
                    url=f"{combined_url}?streams=!openInterest@arr@1s/!markPrice@arr",
                    stream_type="oi_funding_combined",
                    topic="oi_funding",
                    symbol=self.symbol,
                    priority=3  # Lower priority
                )
            )
        else:
            # Spot only streams
            base_url = self.ENDPOINTS['spot']['base_url']
            endpoints.extend([
                StreamEndpoint(
                    url=f"{base_url}/{self.symbol_lower}@trade",
                    stream_type="trade",
                    topic="trade",
                    symbol=self.symbol,
                    priority=10
                ),
                StreamEndpoint(
                    url=f"{base_url}/{self.symbol_lower}@bookTicker", 
                    stream_type="depth_l0",
                    topic="depth_l0",
                    symbol=self.symbol,
                    priority=9
                )
            ])
        
        return endpoints
    
    async def _handle_trade_message(self, msg: dict, endpoint: StreamEndpoint):
        """Handle trade stream messages with cache integration"""
        try:
            ts = msg['T'] / 1e3  # Event time in seconds
            price = float(msg['p'])
            volume = float(msg['q'])
            is_buyer_maker = msg['m']
            
            # Calculate signed quantity (positive for buy, negative for sell)
            signed_qty = volume * (-1 if is_buyer_maker else 1)
            side = 'sell' if is_buyer_maker else 'buy'
            
            # Update cache
            cache.update_trade(ts, price, signed_qty, side, self.symbol)
            
            # Emit to event bus
            trade_data = {
                'ts': ts,
                'price': price,
                'qty': signed_qty,  # Signed quantity for indicators
                'volume': volume,   # Absolute volume
                'side': side,
                'symbol': self.symbol,
                'raw_data': msg
            }
            
            await bus.emit('trade', trade_data)
            
        except Exception as e:
            logger.error(f"Error handling trade message: {e}")
    
    async def _handle_depth_message(self, msg: dict, endpoint: StreamEndpoint):
        """Handle bookTicker messages for L0 depth with cache integration"""
        try:
            ts = msg['E'] / 1e3  # Event time in seconds
            bid_px = float(msg['b'])
            bid_sz = float(msg['B'])
            ask_px = float(msg['a']) 
            ask_sz = float(msg['A'])
            
            # Update cache
            cache.update_depth(ts, bid_px, bid_sz, ask_px, ask_sz, self.symbol)
            
            # Emit to event bus
            depth_data = {
                'ts': ts,
                'bid_px': bid_px,
                'bid_sz': bid_sz,
                'ask_px': ask_px,
                'ask_sz': ask_sz,
                'symbol': self.symbol,
                'mid_price': (bid_px + ask_px) / 2,
                'spread': ask_px - bid_px,
                'raw_data': msg
            }
            
            await bus.emit('depth_l0', depth_data)
            
        except Exception as e:
            logger.error(f"Error handling depth message: {e}")
    
    async def _handle_combined_oi_funding_message(self, msg: dict, endpoint: StreamEndpoint):
        """Handle combined OI and funding messages from stream array"""
        try:
            # Handle stream wrapper format
            if 'stream' in msg and 'data' in msg:
                stream_name = msg['stream']
                data = msg['data']
                
                if 'openInterest' in stream_name:
                    await self._process_oi_message(data)
                elif 'markPrice' in stream_name:
                    await self._process_funding_message(data)
            else:
                # Handle direct array format
                if isinstance(msg, list):
                    for item in msg:
                        event_type = item.get('e', '')
                        if event_type == 'openInterest':
                            await self._process_oi_message(item)
                        elif event_type == 'markPriceUpdate':
                            await self._process_funding_message(item)
                            
        except Exception as e:
            logger.error(f"Error handling combined OI/funding message: {e}")
    
    async def _process_oi_message(self, data: dict):
        """Process open interest message"""
        try:
            # Filter for our symbol
            if data.get('s') == self.symbol:
                ts = data['E'] / 1e3
                oi = float(data['oi'])
                
                # Update cache
                cache.update_open_interest(ts, oi, self.symbol)
                
                # Emit to event bus
                oi_data = {
                    'ts': ts,
                    'oi': oi,
                    'symbol': self.symbol,
                    'raw_data': data
                }
                
                await bus.emit('open_interest', oi_data)
                
        except Exception as e:
            logger.error(f"Error processing OI message: {e}")
    
    async def _process_funding_message(self, data: dict):
        """Process funding/mark price message"""
        try:
            # Filter for our symbol
            if data.get('s') == self.symbol:
                ts = data['E'] / 1e3
                funding_rate = float(data.get('r', 0))
                mark_price = float(data.get('p', 0))
                
                # Update cache
                cache.update_funding(ts, funding_rate, mark_price, self.symbol)
                
                # Emit to event bus
                funding_data = {
                    'ts': ts,
                    'funding_rate': funding_rate,
                    'mark_price': mark_price,
                    'symbol': self.symbol,
                    'raw_data': data
                }
                
                await bus.emit('funding', funding_data)
                
        except Exception as e:
            logger.error(f"Error processing funding message: {e}")
    
    async def _handle_liquidation_message(self, msg: dict, endpoint: StreamEndpoint):
        """Handle liquidation messages with cache integration"""
        try:
            order_data = msg.get('o', {})
            ts = order_data.get('T', msg.get('E', time.time() * 1000)) / 1e3
            price = float(order_data.get('p', 0))
            volume = float(order_data.get('q', 0))
            side_raw = order_data.get('S', 'BUY')
            
            # Convert to signed quantity and side
            side = 'long' if side_raw == 'BUY' else 'short'
            signed_qty = volume * (1 if side == 'long' else -1)
            
            # Update cache
            cache.update_liquidation(ts, price, signed_qty, side, self.symbol)
            
            # Emit to event bus
            liq_data = {
                'ts': ts,
                'price': price,
                'qty': signed_qty,
                'volume': volume,
                'side': side,
                'symbol': self.symbol,
                'raw_data': msg
            }
            
            await bus.emit('liquidation', liq_data)
            
        except Exception as e:
            logger.error(f"Error handling liquidation message: {e}")
    
    async def _process_message(self, message: str, endpoint: StreamEndpoint):
        """Process incoming WebSocket message with intelligent routing"""
        try:
            msg = json.loads(message)
            
            # Route message to appropriate handler based on stream type and content
            if endpoint.stream_type == "trade":
                if msg.get('e') == 'trade':
                    await self._handle_trade_message(msg, endpoint)
                    
            elif endpoint.stream_type == "depth_l0":
                if msg.get('e') == 'bookTicker':
                    await self._handle_depth_message(msg, endpoint)
                    
            elif endpoint.stream_type == "liquidation":
                if msg.get('e') == 'forceOrder':
                    await self._handle_liquidation_message(msg, endpoint)
                    
            elif endpoint.stream_type == "oi_funding_combined":
                # Handle combined stream format
                await self._handle_combined_oi_funding_message(msg, endpoint)
                
        except Exception as e:
            logger.error(f"Error processing message from {endpoint.stream_type}: {e}")
    
    async def _run_stream(self, endpoint: StreamEndpoint):
        """Run a single WebSocket stream with enhanced reconnection logic"""
        connection_id = f"{endpoint.stream_type}_{endpoint.symbol}"
        self.connection_stats[connection_id] = {
            'connect_time': None,
            'message_count': 0,
            'error_count': 0,
            'reconnect_count': 0
        }
        
        while self.is_running:
            try:
                logger.info(f"🔗 Connecting to {endpoint.stream_type} stream")
                
                async with websockets.connect(
                    endpoint.url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10,
                    max_size=1024*1024  # 1MB message size limit
                ) as websocket:
                    
                    self.connections[connection_id] = websocket
                    self.connection_stats[connection_id]['connect_time'] = time.time()
                    
                    logger.success(f"✅ Connected to {endpoint.stream_type} stream")
                    
                    async for message in websocket:
                        if not self.is_running:
                            break
                            
                        self.connection_stats[connection_id]['message_count'] += 1
                        await self._process_message(message, endpoint)
                        
            except Exception as e:
                self.connection_stats[connection_id]['error_count'] += 1
                
                if self.is_running:
                    self.connection_stats[connection_id]['reconnect_count'] += 1
                    reconnect_delay = min(5 * self.connection_stats[connection_id]['reconnect_count'], 60)
                    
                    logger.error(f"❌ {endpoint.stream_type} stream error: {e}")
                    logger.info(f"🔄 Reconnecting in {reconnect_delay}s...")
                    
                    await asyncio.sleep(reconnect_delay)
                    
                    # Max reconnect limit
                    if self.connection_stats[connection_id]['reconnect_count'] > self.max_reconnects:
                        logger.error(f"❌ Max reconnects exceeded for {endpoint.stream_type}")
                        break
            finally:
                if connection_id in self.connections:
                    del self.connections[connection_id]
    
    async def start(self):
        """Start all WebSocket streams with prioritized connection order"""
        self.is_running = True
        logger.info(f"🚀 Starting enhanced multi-stream feed for {self.symbol}")
        logger.info(f"📊 Configured {len(self.endpoints)} streams")
        
        # Sort endpoints by priority for connection order
        sorted_endpoints = sorted(self.endpoints, key=lambda x: x.priority, reverse=True)
        
        # Start streams with staggered connection to avoid rate limits
        tasks = []
        for i, endpoint in enumerate(sorted_endpoints):
            # Add small delay between connections to avoid overwhelming the server
            await asyncio.sleep(0.1 * i)
            task = asyncio.create_task(self._run_stream(endpoint))
            tasks.append(task)
        
        # Wait for all streams
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error in multi-stream feed: {e}")
        finally:
            self.is_running = False
    
    def stop(self):
        """Stop all streams gracefully"""
        self.is_running = False
        logger.info(f"🛑 Stopping multi-stream feed for {self.symbol}")
        
        # Close all active connections
        for connection_id, websocket in self.connections.items():
            try:
                asyncio.create_task(websocket.close())
            except Exception as e:
                logger.warning(f"Error closing connection {connection_id}: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive feed statistics"""
        stats = {
            'symbol': self.symbol,
            'is_running': self.is_running,
            'endpoints_configured': len(self.endpoints),
            'active_connections': len(self.connections),
            'connection_stats': self.connection_stats.copy(),
            'cache_stats': cache.get_stats(),
            'bus_stats': bus.get_stats()
        }
        
        # Add stream-specific stats
        for endpoint in self.endpoints:
            connection_id = f"{endpoint.stream_type}_{endpoint.symbol}"
            if connection_id in self.connection_stats:
                stats[f"{endpoint.stream_type}_messages"] = self.connection_stats[connection_id]['message_count']
                stats[f"{endpoint.stream_type}_errors"] = self.connection_stats[connection_id]['error_count']
        
        return stats

# Backward compatibility - enhanced single trade stream
class BinanceLiveTrades:
    """
    Enhanced backward compatible single trade stream
    
    Now integrates with the centralized cache system while maintaining
    the same interface for existing code.
    """
    
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol.upper()
        self.symbol_lower = symbol.lower()
        self.url = f"wss://stream.binance.com:9443/ws/{self.symbol_lower}@trade"
        self.is_running = False
        self.websocket = None

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.websocket or self.websocket.closed:
            self.websocket = await websockets.connect(self.url)
            self.is_running = True
            
        message = await self.websocket.recv()
        trade_data = json.loads(message)
        
        # Update cache (integration with new system)
        ts = trade_data['T'] / 1000
        price = float(trade_data['p'])
        volume = float(trade_data['q'])
        is_buyer_maker = trade_data['m']
        signed_qty = volume * (-1 if is_buyer_maker else 1)
        side = 'sell' if is_buyer_maker else 'buy'
        
        cache.update_trade(ts, price, signed_qty, side, self.symbol)
        
        # Convert to Trade object format expected by LiveTrader
        from .feeds import Trade
        return Trade(
            timestamp=ts,
            price=price,
            volume=volume,
            side=side,
            symbol=self.symbol,
            source='binance'
        )

# Unified Recorder with topic-based format
class UnifiedRecorder:
    """
    Unified recorder that writes all stream data in consistent JSON format
    
    Format: {"topic": "trade|depth_l0|oi|funding|liquidation", "ts": timestamp, ...}
    This enables easy replay and analysis with pandas/DuckDB.
    """
    
    def __init__(self, output_dir: str = "recordings"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.current_file = None
        self.symbol = None
        
    def start_recording(self, symbol: str, session_name: Optional[str] = None):
        """Start recording for a symbol"""
        self.symbol = symbol
        
        if session_name is None:
            session_name = f"{symbol}_{int(time.time())}"
            
        filename = f"{session_name}.rec.gz"
        self.current_file = self.output_dir / filename
        
        logger.info(f"📹 Starting recording: {self.current_file}")
    
    def write_trade(self, data: Dict[str, Any]):
        """Write trade data"""
        self._write_record("trade", data)
    
    def write_depth(self, data: Dict[str, Any]):
        """Write depth data"""  
        self._write_record("depth_l0", data)
    
    def write_oi(self, data: Dict[str, Any]):
        """Write open interest data"""
        self._write_record("open_interest", data)
    
    def write_funding(self, data: Dict[str, Any]):
        """Write funding data"""
        self._write_record("funding", data)
    
    def write_liquidation(self, data: Dict[str, Any]):
        """Write liquidation data"""
        self._write_record("liquidation", data)
    
    def write_indicator(self, name: str, value: float, ts: float):
        """Write indicator value"""
        indicator_data = {
            'ts': ts,
            'name': name,
            'value': value,
            'symbol': self.symbol
        }
        self._write_record("indi", indicator_data)
    
    def _write_record(self, topic: str, data: Dict[str, Any]):
        """Write record to file in unified format"""
        if not self.current_file:
            return
            
        import gzip
        import json
        
        record = {"topic": topic, **data}
        line = json.dumps(record, separators=(',', ':')) + '\n'
        
        try:
            with gzip.open(self.current_file, 'at') as f:
                f.write(line)
        except Exception as e:
            logger.error(f"Error writing to recording file: {e}")

# Global recorder instance
recorder = UnifiedRecorder()

# Backward compatibility - single trade stream
class BinanceLiveTrades:
    """Backward compatible single trade stream (kept for existing code)"""
    
    def __init__(self, symbol: str = "BTCUSDT"):
        self.symbol = symbol.upper()
        self.symbol_lower = symbol.lower()
        self.url = f"wss://stream.binance.com:9443/ws/{self.symbol_lower}@trade"
        self.is_running = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not hasattr(self, 'websocket') or not self.websocket:
            self.websocket = await websockets.connect(self.url)
            self.is_running = True
            
        message = await self.websocket.recv()
        trade_data = json.loads(message)
        
        # Convert to Trade object format expected by LiveTrader
        from .feeds import Trade
        return Trade(
            timestamp=trade_data['T'] / 1000,
            price=float(trade_data['p']),
            volume=float(trade_data['q']),
            side='buy' if not trade_data['m'] else 'sell',
            symbol=self.symbol,
            source='binance'
        )
