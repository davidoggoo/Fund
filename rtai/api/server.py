"""
FastAPI WebSocket Server for RTAI Live Trading Dashboard
Provides real-time data streaming and backtesting integration
"""
import asyncio
import json
from typing import Set, Dict, Any, Optional
from datetime import datetime
from contextlib import asynccontextmanager
import aiohttp
import psutil
import os
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBearer
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn
import time
from collections import defaultdict, deque

from rtai.backtesting_strategy import RTAIStrategy

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Connection Pool Manager
class ConnectionPoolManager:
    """Manages HTTP connection pools for external API calls"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.connector: Optional[aiohttp.TCPConnector] = None
    
    async def initialize(self):
        """Initialize connection pool"""
        self.connector = aiohttp.TCPConnector(
            limit=100,  # Total connection pool size
            limit_per_host=30,  # Per host limit
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=timeout,
            headers={'User-Agent': 'RTAI-Trading-Bot/1.0'}
        )
    
    async def cleanup(self):
        """Cleanup connection pool"""
        if self.session:
            await self.session.close()
        if self.connector:
            await self.connector.close()

# Circular Buffer for Memory Optimization
class CircularBuffer:
    """Memory-efficient circular buffer for streaming data"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.total_added = 0
    
    def add(self, item: Any):
        """Add item to buffer"""
        self.buffer.append(item)
        self.total_added += 1
    
    def get_recent(self, count: int = None) -> list:
        """Get recent items"""
        if count is None:
            return list(self.buffer)
        return list(self.buffer)[-count:] if count <= len(self.buffer) else list(self.buffer)
    
    def clear(self):
        """Clear buffer"""
        self.buffer.clear()
    
    def size(self) -> int:
        """Get current size"""
        return len(self.buffer)
    
    def memory_usage(self) -> dict:
        """Get memory usage stats"""
        return {
            'current_size': len(self.buffer),
            'max_size': self.max_size,
            'total_added': self.total_added,
            'memory_efficiency': f"{(len(self.buffer) / self.max_size) * 100:.1f}%"
        }

# Circuit Breaker Pattern
class CircuitBreaker:
    """Circuit breaker for API calls"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def can_execute(self) -> bool:
        """Check if execution is allowed"""
        if self.state == 'CLOSED':
            return True
        elif self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.timeout:
                self.state = 'HALF_OPEN'
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        """Record successful execution"""
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def record_failure(self):
        """Record failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'

# Global instances
connection_pool = ConnectionPoolManager()
trade_buffer = CircularBuffer(max_size=5000)  # Store last 5000 trades
indicator_buffer = CircularBuffer(max_size=1000)  # Store last 1000 indicator updates
circuit_breaker = CircuitBreaker()

class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware for collecting request metrics"""
    
    async def dispatch(self, request, call_next):
        start_time = time.time()
        
        # Increment request counter
        metrics_storage["requests_total"] += 1
        
        try:
            response = await call_next(request)
            
            # Record response time
            response_time = time.time() - start_time
            metrics_storage["response_times"].append(response_time)
            
            return response
            
        except Exception as e:
            # Increment error counter
            metrics_storage["errors_total"] += 1
            raise

# Rate limiting
rate_limit_storage = defaultdict(list)
security = HTTPBearer(auto_error=False)

# Metrics collection
metrics_storage = {
    "requests_total": 0,
    "websocket_connections": 0,
    "backtest_requests": 0,
    "errors_total": 0,
    "response_times": [],
    "last_reset": time.time()
}


class ConnectionManager:
    """Manages WebSocket connections for real-time data broadcasting with authentication"""
    
    def __init__(self) -> None:
        self.active_connections: Set[WebSocket] = set()
        self.authenticated_connections: Dict[WebSocket, dict] = {}
        self.connection_stats: Dict[str, int] = {
            "total_connected": 0,
            "total_disconnected": 0,
            "messages_sent": 0,
            "broadcast_errors": 0,
            "auth_failures": 0
        }
        self.valid_api_keys = {
            "rtai-demo-key": {"name": "Demo User", "permissions": ["read"]},
            "rtai-admin-key": {"name": "Admin User", "permissions": ["read", "write"]},
            os.getenv("RTAI_API_KEY", "rtai-default-key"): {"name": "Default User", "permissions": ["read", "write"]}
        }
    
    async def connect(self, websocket: WebSocket, api_key: str = None) -> bool:
        """Accept and authenticate WebSocket connection"""
        try:
            await websocket.accept()
            
            # Authenticate if API key provided
            if api_key and api_key in self.valid_api_keys:
                user_info = self.valid_api_keys[api_key]
                self.authenticated_connections[websocket] = {
                    "api_key": api_key,
                    "user": user_info["name"],
                    "permissions": user_info["permissions"],
                    "connected_at": time.time()
                }
                logger.info(f"🔐 Authenticated connection: {user_info['name']}")
            else:
                # Allow unauthenticated connections with limited access
                self.authenticated_connections[websocket] = {
                    "api_key": None,
                    "user": "Anonymous",
                    "permissions": ["read"],
                    "connected_at": time.time()
                }
                if api_key:  # Invalid key provided
                    self.connection_stats["auth_failures"] += 1
                    logger.warning(f"🚫 Invalid API key attempted: {api_key[:8]}...")
            
            self.active_connections.add(websocket)
            self.connection_stats["total_connected"] += 1
            metrics_storage["websocket_connections"] += 1
            logger.info(f"✅ Client connected. Total: {len(self.active_connections)}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Connection failed: {e}")
            return False
    
    def disconnect(self, websocket: WebSocket) -> None:
        """Remove and track disconnected WebSocket"""
        self.active_connections.discard(websocket)
        if websocket in self.authenticated_connections:
            user_info = self.authenticated_connections[websocket]
            logger.info(f"🔓 User disconnected: {user_info['user']}")
            del self.authenticated_connections[websocket]
        
        self.connection_stats["total_disconnected"] += 1
        logger.info(f"❌ Client disconnected. Total: {len(self.active_connections)}")
    
    def get_user_permissions(self, websocket: WebSocket) -> list:
        """Get permissions for a WebSocket connection"""
        if websocket in self.authenticated_connections:
            return self.authenticated_connections[websocket]["permissions"]
        return ["read"]  # Default permissions
    
    def validate_api_key(self, api_key: str) -> bool:
        """Validate API key"""
        return api_key in self.valid_api_keys
    
    async def broadcast(self, message: Dict[str, Any]) -> int:
        """
        Broadcast message to all connected clients
        
        Returns:
            Number of successful broadcasts
        """
        if not self.active_connections:
            return 0
            
        message_str = json.dumps(message, default=str)
        disconnected = set()
        successful_sends = 0
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_str)
                successful_sends += 1
                self.connection_stats["messages_sent"] += 1
            except Exception as e:
                logger.debug(f"Failed to send to client: {e}")
                disconnected.add(connection)
                self.connection_stats["broadcast_errors"] += 1
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.active_connections.discard(conn)
        
        return successful_sends
    
    def get_stats(self) -> Dict[str, int]:
        """Get connection statistics"""
        return {
            **self.connection_stats,
            "active_connections": len(self.active_connections)
        }


# Global connection manager instance
manager = ConnectionManager()

def rate_limit_check(client_ip: str, endpoint: str, max_requests: int = 10, window_seconds: int = 60):
    """Simple rate limiting implementation"""
    current_time = time.time()
    key = f"{client_ip}:{endpoint}"
    
    # Clean old requests
    rate_limit_storage[key] = [
        req_time for req_time in rate_limit_storage[key] 
        if current_time - req_time < window_seconds
    ]
    
    # Check if limit exceeded
    if len(rate_limit_storage[key]) >= max_requests:
        raise HTTPException(
            status_code=429, 
            detail=f"Rate limit exceeded: {max_requests} requests per {window_seconds} seconds"
        )
    
    # Add current request
    rate_limit_storage[key].append(current_time)

# FastAPI app initialization
app = FastAPI(
    title="RTAI Trading Dashboard",
    description="Real-time algorithmic trading indicators with TradingVue.js integration",
    version="2.0.0"
)

# Add metrics middleware
app.add_middleware(MetricsMiddleware)

# CORS middleware with security hardening
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:3000",  # For development
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, api_key: str = None):
    """WebSocket endpoint for real-time data streaming with authentication"""
    # Extract API key from query parameters if provided
    query_params = dict(websocket.query_params)
    api_key = query_params.get("api_key", api_key)
    
    # Connect with authentication
    connected = await manager.connect(websocket, api_key)
    if not connected:
        await websocket.close(code=1008, reason="Connection failed")
        return
    
    try:
        # Send welcome message with user info
        user_info = manager.authenticated_connections.get(websocket, {})
        await websocket.send_text(json.dumps({
            "type": "welcome",
            "data": {
                "user": user_info.get("user", "Unknown"),
                "permissions": user_info.get("permissions", []),
                "server_time": datetime.now().isoformat()
            }
        }))
        
        while True:
            # Keep connection alive and handle client messages
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                message_type = message.get("type", "unknown")
                
                # Handle different message types
                if message_type == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    }))
                elif message_type == "subscribe":
                    # Handle subscription requests (future feature)
                    await websocket.send_text(json.dumps({
                        "type": "subscribed",
                        "data": message.get("data", {})
                    }))
                else:
                    # Echo back for unknown messages
                    await websocket.send_text(json.dumps({
                        "type": "echo",
                        "data": message
                    }))
                    
            except json.JSONDecodeError:
                # Handle plain text messages
                await websocket.send_text(json.dumps({
                    "type": "echo",
                    "data": {"message": data}
                }))
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
        manager.disconnect(websocket)


from fastapi import Request, Header
from pydantic import BaseModel

class BacktestRequest(BaseModel):
    symbol: str = "BTCUSDT"
    start_date: Optional[str] = None
    end_date: Optional[str] = None

# API Key validation endpoint
@app.get("/validate-key")
async def validate_api_key(api_key: str = Header(None, alias="X-API-Key")):
    """Validate API key and return user information"""
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required")
    
    if manager.validate_api_key(api_key):
        user_info = manager.valid_api_keys[api_key]
        return {
            "valid": True,
            "user": user_info["name"],
            "permissions": user_info["permissions"]
        }
    else:
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.post("/backtest")
async def run_backtest(
    backtest_request: BacktestRequest,
    request: Request
):
    """Run REAL backtesting with historical data and return results"""
    try:
        # Request size validation
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > 1024 * 1024:  # 1MB limit
            raise HTTPException(status_code=413, detail="Request too large (max 1MB)")
        
        # Circuit breaker check
        if not circuit_breaker.can_execute():
            raise HTTPException(
                status_code=503, 
                detail="Service temporarily unavailable due to high error rate"
            )
        
        # Rate limiting
        client_ip = request.client.host
        rate_limit_check(client_ip, "backtest", max_requests=5, window_seconds=300)  # 5 requests per 5 minutes
        
        # Input validation with graceful degradation
        symbol = backtest_request.symbol.upper()
        start_date = backtest_request.start_date
        end_date = backtest_request.end_date
        
        # Validate symbol format
        if not symbol or len(symbol) < 3 or len(symbol) > 20:
            raise HTTPException(status_code=400, detail="Invalid symbol format")
        
        # Graceful degradation: limit date range for performance
        if start_date and end_date:
            from datetime import datetime
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            days_diff = (end_dt - start_dt).days
            
            if days_diff > 365:  # Limit to 1 year
                logger.warning(f"Date range too large ({days_diff} days), limiting to 365 days")
                start_dt = end_dt - timedelta(days=365)
                start_date = start_dt.isoformat() + 'Z'
        
        # Sanitize symbol (allow only alphanumeric and common crypto pairs)
        allowed_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        if not all(c in allowed_chars for c in symbol):
            raise HTTPException(status_code=400, detail="Symbol contains invalid characters")
        
        # Validate date formats if provided
        if start_date:
            try:
                from datetime import datetime
                datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start_date format")
        
        if end_date:
            try:
                from datetime import datetime
                datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid end_date format")
        from rtai.io.rec_converter import get_latest_rec_files, rec_to_ohlcv
        from backtesting import Backtest
        from rtai.backtesting_strategy import RTAIStrategy
        import pandas as pd
        
        logger.info(f"🎯 Starting REAL backtest for {symbol}")
        
        # Find latest recording file for the symbol
        rec_files = get_latest_rec_files("recordings", limit=5)
        target_file = None
        
        for file_path in rec_files:
            if symbol.upper() in file_path.upper():
                target_file = file_path
                break
        
        if not target_file:
            # No recording file found - return 404 instead of sample data
            available_files = [f.name for f in rec_files] if rec_files else []
            raise HTTPException(
                status_code=404, 
                detail=f"No recording file found for {symbol}. Available files: {available_files[:5]}"
            )
            
        else:
            # Convert recording file to OHLCV DataFrame
            logger.info(f"Converting {target_file} to OHLCV data")
            ohlcv_data = rec_to_ohlcv(target_file, timeframe='1m')
            
            if not ohlcv_data:
                raise ValueError(f"No OHLCV data could be extracted from {target_file}")
            
            # Convert to DataFrame
            from rtai.io.rec_converter import ohlcv_to_dataframe
            df = ohlcv_to_dataframe(ohlcv_data)
            
            if df.empty:
                raise ValueError("Converted DataFrame is empty")
        
        logger.info(f"📊 Backtesting with {len(df)} data points")
        
        # Run backtesting with RTAI strategy
        bt = Backtest(
            df, 
            RTAIStrategy,
            commission=0.0004,  # 0.04% commission
            trade_on_close=True,
            exclusive_orders=True
        )
        
        # Execute backtest
        stats = bt.run()
        
        # Extract results
        results = {
            "symbol": symbol,
            "total_return": float(stats['Return [%]']) if 'Return [%]' in stats else 0.0,
            "sharpe_ratio": float(stats['Sharpe Ratio']) if 'Sharpe Ratio' in stats else 0.0,
            "max_drawdown": float(stats['Max. Drawdown [%]']) if 'Max. Drawdown [%]' in stats else 0.0,
            "win_rate": float(stats['Win Rate [%]']) if 'Win Rate [%]' in stats else 0.0,
            "total_trades": int(stats['# Trades']) if '# Trades' in stats else 0,
            "start_date": df.index[0].isoformat() if not df.empty else None,
            "end_date": df.index[-1].isoformat() if not df.empty else None,
            "data_points": len(df),
            "trades": []
        }
        
        # Extract trade details if available
        if hasattr(stats, '_trades') and stats._trades is not None:
            trades_df = stats._trades
            results["trades"] = [
                {
                    "entry_time": row['EntryTime'].isoformat() if pd.notna(row['EntryTime']) else None,
                    "exit_time": row['ExitTime'].isoformat() if pd.notna(row['ExitTime']) else None,
                    "entry_price": float(row['EntryPrice']) if pd.notna(row['EntryPrice']) else 0.0,
                    "exit_price": float(row['ExitPrice']) if pd.notna(row['ExitPrice']) else 0.0,
                    "size": float(row['Size']) if pd.notna(row['Size']) else 0.0,
                    "pnl": float(row['PnL']) if pd.notna(row['PnL']) else 0.0,
                    "return_pct": float(row['ReturnPct']) if pd.notna(row['ReturnPct']) else 0.0
                }
                for _, row in trades_df.iterrows()
            ]
        
        logger.success(f"✅ Backtest completed: {results['total_return']:.2f}% return, {results['total_trades']} trades")
        
        # Broadcast results to connected clients
        await manager.broadcast({
            "type": "backtest_results",
            "data": results,
            "timestamp": datetime.now().isoformat()
        })
        
        return {"status": "success", "results": results}
        
    except Exception as e:
        logger.error(f"❌ Backtest failed: {e}")
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "connections": len(manager.active_connections)
    }

@app.get("/metrics")
async def get_metrics():
    """Advanced metrics endpoint for monitoring"""
    current_time = time.time()
    uptime = current_time - metrics_storage["last_reset"]
    
    # Calculate average response time
    response_times = metrics_storage["response_times"]
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    
    # Clean old response times (keep last 1000)
    if len(response_times) > 1000:
        metrics_storage["response_times"] = response_times[-1000:]
    
    return {
        "uptime_seconds": uptime,
        "requests_total": metrics_storage["requests_total"],
        "websocket_connections_current": len(manager.active_connections),
        "websocket_connections_total": metrics_storage["websocket_connections"],
        "backtest_requests_total": metrics_storage["backtest_requests"],
        "errors_total": metrics_storage["errors_total"],
        "avg_response_time_ms": avg_response_time * 1000,
        "requests_per_second": metrics_storage["requests_total"] / max(uptime, 1),
        "error_rate": metrics_storage["errors_total"] / max(metrics_storage["requests_total"], 1),
        "timestamp": datetime.now().isoformat()
    }


# Global broadcast function for LiveTrader integration
async def broadcast_data(message_type: str, data: Dict[str, Any]):
    """Global function to broadcast data from LiveTrader with graceful degradation"""
    try:
        # Store data in circular buffers for memory optimization
        if message_type == "trade":
            trade_buffer.add(data)
        elif message_type == "indi":
            indicator_buffer.add(data)
        
        # Graceful degradation for indicators
        if message_type == "indi" and isinstance(data, dict):
            # Validate indicator data and provide fallbacks
            validated_data = {}
            for key, value in data.items():
                try:
                    if value is not None and not (isinstance(value, (int, float)) and (value != value)):  # Check for NaN
                        validated_data[key] = float(value) if isinstance(value, (int, float)) else value
                    else:
                        # Graceful degradation: use last known value or default
                        if indicator_buffer.size() > 1:
                            recent_indicators = indicator_buffer.get_recent(2)
                            if len(recent_indicators) > 1 and key in recent_indicators[-2]:
                                validated_data[key] = recent_indicators[-2][key]
                                logger.debug(f"🔄 Using fallback value for {key}")
                            else:
                                validated_data[key] = 0.0  # Safe default
                        else:
                            validated_data[key] = 0.0
                except (ValueError, TypeError) as e:
                    logger.warning(f"⚠️ Invalid indicator value for {key}: {value}, using default")
                    validated_data[key] = 0.0
            
            data = validated_data
        
        # Comprehensive logging
        logger.debug(f"📡 Broadcasting {message_type}: {len(str(data))} bytes to {len(manager.active_connections)} clients")
        
        # Circuit breaker for broadcasting
        if circuit_breaker.can_execute():
            await manager.broadcast({
                "type": message_type,
                "data": data,
                "timestamp": datetime.now().isoformat(),
                "buffer_stats": {
                    "trades": trade_buffer.memory_usage(),
                    "indicators": indicator_buffer.memory_usage()
                } if message_type == "indi" else None
            })
            circuit_breaker.record_success()
        else:
            logger.warning("🚫 Broadcasting circuit breaker is OPEN, skipping broadcast")
            
    except Exception as e:
        logger.error(f"❌ Broadcast error for {message_type}: {e}")
        circuit_breaker.record_failure()
        
        # Graceful degradation: try to send minimal data
        try:
            await manager.broadcast({
                "type": "error",
                "data": {"message": "Temporary broadcasting issue", "original_type": message_type},
                "timestamp": datetime.now().isoformat()
            })
        except Exception as fallback_error:
            logger.error(f"❌ Fallback broadcast also failed: {fallback_error}")


# App lifecycle events
@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup"""
    logger.info("🚀 Starting RTAI API Server...")
    await connection_pool.initialize()
    logger.info("✅ Connection pool initialized")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    logger.info("🛑 Shutting down RTAI API Server...")
    await connection_pool.cleanup()
    logger.info("✅ Resources cleaned up")

if __name__ == "__main__":
    uvicorn.run(
        "rtai.api.server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )