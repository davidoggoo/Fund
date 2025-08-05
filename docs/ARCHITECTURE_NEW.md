# RTAI System Architecture (Updated)

## Overview

RTAI (Real-Time AI Trading System) is a unified live trading and backtesting platform that combines:
- **Real-time data ingestion** from Binance WebSocket
- **Advanced indicators** (OFI, VPIN, Kyle's Lambda, LPI)
- **Signal generation** with multiple strategies
- **Live dashboard** with TradingVue.js integration
- **Backtesting engine** using historical data
- **Comprehensive logging** and health monitoring

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        RTAI Trading System                      │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Binance WebSocket │    │  Historical Data  │    │  Frontend (Vue) │
│     (Live Feed)     │    │   (.rec.gz files) │    │   TradingVue.js │
└─────────┬───────────┘    └─────────┬───────┘    └─────────┬───────┘
          │                          │                      │
          ▼                          ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FastAPI Server                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   WebSocket     │  │   Backtest      │  │    Health       │ │
│  │   Broadcast     │  │   Endpoint      │  │   Monitoring    │ │
│  │   /ws           │  │   /backtest     │  │   /health       │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
          │                          │                      │
          ▼                          ▼                      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   LiveTrader    │    │  Backtesting.py │    │  HealthDashboard│
│                 │    │     Engine      │    │                 │
│ ┌─────────────┐ │    │                 │    │ ┌─────────────┐ │
│ │ Feed System │ │    │ ┌─────────────┐ │    │ │  Metrics    │ │
│ │             │ │    │ │RTAI Strategy│ │    │ │ Collection  │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
│ ┌─────────────┐ │    │                 │    │                 │
│ │ Indicators  │ │    │ ┌─────────────┐ │    │ ┌─────────────┐ │
│ │ OFI, VPIN   │ │    │ │  Portfolio  │ │    │ │  Alerts     │ │
│ │ Kyle, LPI   │ │    │ │ Management  │ │    │ │  System     │ │
│ └─────────────┘ │    │ └─────────────┘ │    │ └─────────────┘ │
│ ┌─────────────┐ │    └─────────────────┘    └─────────────────┘
│ │   Signal    │ │              │                      │
│ │ Generation  │ │              ▼                      ▼
│ └─────────────┘ │    ┌─────────────────┐    ┌─────────────────┐
│ ┌─────────────┐ │    │   Trade Results │    │    Log Files    │
│ │  Recorder   │ │    │   Equity Curve  │    │  Health Metrics │
│ │             │ │    │   Performance   │    │   Error Logs    │
│ └─────────────┘ │    └─────────────────┘    └─────────────────┘
└─────────────────┘              │                      │
          │                      ▼                      ▼
          ▼            ┌─────────────────┐    ┌─────────────────┐
┌─────────────────┐    │   SQLite DB     │    │  External       │
│  Recording      │    │                 │    │  Monitoring     │
│  (.rec.gz)      │    │ ┌─────────────┐ │    │  (Prometheus)   │
│                 │    │ │ indicators  │ │    │                 │
│ ┌─────────────┐ │    │ │ equity      │ │    │                 │
│ │Trade History│ │    │ │ trades      │ │    │                 │
│ │Indicators   │ │    │ └─────────────┘ │    │                 │
│ │Signals      │ │    └─────────────────┘    └─────────────────┘
│ └─────────────┘ │              ▲                      ▲
└─────────────────┘              │                      │
          │                      │                      │
          └──────────────────────┘                      │
                                                         │
┌─────────────────────────────────────────────────────────────────┐
│                      Telegram Bot                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │   Signal        │  │   Performance   │  │    Error        │ │
│  │ Notifications   │  │    Reports      │  │   Alerts        │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Live Trading Mode - Enhanced Multi-Stream
1. **Binance Multi-Stream WebSocket** → Real-time trade, depth, OI, funding, liquidation data
2. **LiveTrader Event Bus** → Processes all streams, updates indicators  
3. **Signal Detection** → Generates buy/sell signals from multiple indicators
4. **WebSocket Broadcast** → Pushes data to frontend dashboard
5. **TradingVue Dashboard** → Real-time visualization with multi-indicator overlays
6. **Recording System** → Saves structured data for backtesting
7. **Database Storage** → Stores indicators, signals, equity with ts_ms indexing

### Backtesting Mode  
1. **Frontend Request** → POST /backtest with recording file path
2. **Recording Validation** → Strict file existence checking, no sample data
3. **Data Conversion** → Convert .rec.gz to OHLCV DataFrame using rec_converter
4. **Strategy Execution** → Run RTAI strategy on historical indicator values
5. **Results Analysis** → Return performance metrics and trade markers
6. **Visualization** → Display equity curve and trade signals on TradingVue

## Key Components

### LiveTrader (`rtai/live_trader.py`)
- **Real-time processing** of Binance WebSocket trades
- **Indicator calculation** (OFI, VPIN, Kyle's Lambda, LPI)
- **Signal generation** using multiple strategies
- **Data recording** for backtesting
- **WebSocket broadcasting** for dashboard

### Feed System (`rtai/io/feeds.py`)
- **BinanceLiveTrades** - WebSocket feed for live data
- **Replayer** - Historical data replay for backtesting
- **Health monitoring** and reconnection logic

### Indicators (`rtai/indicators/base.py`)
- **OFI** - Order Flow Imbalance with z-score normalization
- **VPIN** - Volume-Synchronized Probability of Informed Trading
- **KyleLambda** - Market impact coefficient
- **LPI** - Liquidation Pressure Index with multi-venue aggregation

### FastAPI Server (`rtai/api/server.py`)
- **WebSocket endpoint** (`/ws`) for real-time data streaming
- **Backtest endpoint** (`/backtest`) for historical analysis
- **Health monitoring** (`/health`) for system status
- **Rate limiting** and error handling

### Frontend Dashboard (`frontend/`)
- **TradingVue.js** integration for advanced charting
- **Real-time data** via WebSocket connection
- **Interactive backtesting** with strategy tester
- **Performance metrics** and trade visualization

## Database Schema

### `indicators` table
```sql
CREATE TABLE indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    ts_ms INTEGER NOT NULL,           -- TradingVue compatibility
    symbol TEXT DEFAULT 'BTCUSDT',
    indicator_name TEXT NOT NULL,
    value REAL NOT NULL,
    metadata TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### `equity` table
```sql
CREATE TABLE equity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    ts_ms INTEGER NOT NULL,
    eq REAL NOT NULL,
    symbol TEXT DEFAULT 'BTCUSDT',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### `trades` table
```sql
CREATE TABLE trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_time REAL NOT NULL,
    exit_time REAL,
    entry_price REAL NOT NULL,
    exit_price REAL,
    quantity REAL NOT NULL,
    side TEXT NOT NULL CHECK(side IN ('buy', 'sell')),
    pnl REAL,
    symbol TEXT DEFAULT 'BTCUSDT',
    strategy TEXT DEFAULT 'RTAI',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## Configuration

### Environment Variables
```bash
# WebSocket Configuration
BINANCE_WS_URL=wss://stream.binance.com:9443/ws/btcusdt@trade

# API Configuration  
API_PORT=8000
API_HOST=localhost

# Database
DB_PATH=state/indicators.db

# Logging
LOG_LEVEL=INFO
STRUCTURED_LOGGING=true

# Telegram Bot (optional)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### Indicator Parameters
```python
# OFI Configuration
OFI_ALPHA = 0.15              # EWMA alpha for mean
OFI_MAD_ALPHA = 0.3           # EWMA alpha for deviation
OFI_MICRO_THRESHOLD = 0.001   # Filter micro-prints

# VPIN Configuration  
VPIN_BASE_BUCKET = 50.0       # Base bucket size (USDT)
VPIN_WIN_BUCKETS = 12         # Rolling window size

# Kyle's Lambda Configuration
KYLE_WINDOW = 120             # Window size (trades)
KYLE_EWMA_ALPHA = 0.25        # EWMA smoothing factor

# LPI Configuration
LPI_WINDOW_SECONDS = 60       # Time window for decay
LPI_SENSITIVITY = 10.0        # Sensitivity multiplier
```

## Performance Characteristics

### Latency
- **WebSocket processing**: < 5ms per trade
- **Indicator calculation**: < 1ms per update
- **Database writes**: < 2ms per batch
- **Frontend updates**: < 100ms end-to-end

### Throughput
- **Live trades**: 1000+ trades/second sustained
- **Database storage**: 10MB/day typical
- **Memory usage**: < 500MB resident
- **CPU usage**: < 20% on modern hardware

### Reliability
- **Automatic reconnection**: WebSocket with exponential backoff
- **Data validation**: Comprehensive input sanitization
- **Error recovery**: Graceful degradation on component failure
- **Health monitoring**: Continuous system status checks

## Monitoring and Observability

### Structured Logging
```json
{
  "timestamp": "2025-08-05T10:30:45.123Z",
  "level": "INFO", 
  "component": "LiveTrader",
  "message": "Signal generated: BUY",
  "correlation_id": "uuid-1234",
  "symbol": "BTCUSDT",
  "price": 50123.45,
  "signal_strength": 2.34
}
```

### Health Metrics
- **Trade processing rate**: trades/second
- **Indicator calculation time**: milliseconds
- **WebSocket connection status**: connected/disconnected
- **Database performance**: query time, write throughput
- **Memory usage**: heap size, GC frequency

### Alerts
- **Connection failures**: WebSocket disconnections
- **Performance degradation**: High latency warnings
- **Error thresholds**: Exception rate monitoring
- **System resources**: Memory/CPU usage alerts

## Deployment

### Docker Deployment
```bash
docker-compose up -d
```

### Manual Deployment
```bash
# Install dependencies
pip install -r requirements.txt

# Start API server
uvicorn rtai.api.server:app --port 8000 &

# Start frontend
python -m http.server 8080 -d frontend &

# Start live trading
python -m rtai.main --live --symbol BTCUSDT
```

### Production Checklist
- [ ] Environment variables configured
- [ ] Database initialized and migrated
- [ ] SSL certificates installed (if HTTPS)
- [ ] Monitoring dashboards configured
- [ ] Backup procedures in place
- [ ] Log rotation configured
- [ ] Resource limits set

## API Reference

### WebSocket Messages

#### Trade Data
```json
{
  "type": "trade",
  "symbol": "BTCUSDT", 
  "price": 50123.45,
  "volume": 1.234,
  "side": "buy",
  "timestamp": 1628123456.789
}
```

#### Indicator Updates  
```json
{
  "type": "indicator",
  "name": "OFI",
  "value": 1.23,
  "symbol": "BTCUSDT",
  "timestamp": 1628123456.789
}
```

#### Signal Generation
```json
{
  "type": "signal", 
  "action": "BUY",
  "symbol": "BTCUSDT",
  "price": 50123.45,
  "strength": 2.34,
  "timestamp": 1628123456.789,
  "indicators": {
    "ofi": 1.23,
    "vpin": 0.67,
    "kyle": 0.89
  }
}
```

### REST Endpoints

#### POST /backtest
```json
{
  "symbol": "BTCUSDT",
  "file": "recordings/BTCUSDT_latest.rec.gz",
  "start_date": null,
  "end_date": null
}
```

Response:
```json
{
  "status": "success",
  "results": {
    "total_return": 12.34,
    "sharpe_ratio": 1.56,
    "max_drawdown": -5.67,
    "total_trades": 42,
    "win_rate": 65.5
  }
}
```

#### GET /health
```json
{
  "status": "healthy",
  "timestamp": "2025-08-05T10:30:45.123Z",
  "uptime": 3600,
  "components": {
    "websocket": "connected",
    "database": "available", 
    "indicators": "active"
  }
}
```

This architecture provides a robust, scalable foundation for real-time trading with comprehensive monitoring and backtesting capabilities.
