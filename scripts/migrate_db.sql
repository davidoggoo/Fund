-- RTAI Database Migration Script
-- Add ts_ms column for TradingVue compatibility and improved indexing

BEGIN;

-- Add ts_ms column if it doesn't exist
ALTER TABLE indicators ADD COLUMN ts_ms INTEGER;

-- Populate ts_ms from existing timestamp column
UPDATE indicators SET ts_ms = CAST(timestamp * 1000 AS INTEGER) WHERE ts_ms IS NULL;

-- Create index on ts_ms for fast queries
CREATE INDEX IF NOT EXISTS idx_indicators_ts_ms ON indicators(ts_ms);

-- Create index on indicator_name for filtered queries  
CREATE INDEX IF NOT EXISTS idx_indicators_name ON indicators(indicator_name);

-- Create composite index for time-series queries
CREATE INDEX IF NOT EXISTS idx_indicators_name_ts ON indicators(indicator_name, ts_ms);

-- Add equity table for portfolio tracking
CREATE TABLE IF NOT EXISTS equity (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    ts_ms INTEGER NOT NULL,
    eq REAL NOT NULL,
    symbol TEXT DEFAULT 'BTCUSDT',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes on equity table
CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity(ts);
CREATE INDEX IF NOT EXISTS idx_equity_ts_ms ON equity(ts_ms);
CREATE INDEX IF NOT EXISTS idx_equity_symbol ON equity(symbol);

-- Add trade history table for backtest analysis
CREATE TABLE IF NOT EXISTS trades (
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

-- Create indexes on trades table
CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);
CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);

-- Verify migration
SELECT 
    'indicators' as table_name,
    COUNT(*) as row_count,
    MIN(ts_ms) as min_ts_ms,
    MAX(ts_ms) as max_ts_ms
FROM indicators
WHERE ts_ms IS NOT NULL

UNION ALL

SELECT 
    'equity' as table_name,
    COUNT(*) as row_count,
    COALESCE(MIN(ts_ms), 0) as min_ts_ms,
    COALESCE(MAX(ts_ms), 0) as max_ts_ms
FROM equity

UNION ALL

SELECT 
    'trades' as table_name,
    COUNT(*) as row_count,
    0 as min_ts_ms,
    0 as max_ts_ms
FROM trades;

COMMIT;

-- Print success message
.print "✅ Database migration completed successfully"
.print "📊 Tables: indicators, equity, trades"
.print "🔗 All indexes created for optimal query performance"
