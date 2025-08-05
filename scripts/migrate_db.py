#!/usr/bin/env python3
"""
Database migration script for RTAI indicators database
Adds ts_ms column, equity table, and proper indexing
"""

import sqlite3
import os
import sys
from pathlib import Path

def run_migration():
    """Run database migration"""
    
    # Check if database exists
    db_path = Path("state/indicators.db")
    if not db_path.exists():
        print("❌ Database not found at state/indicators.db")
        print("💡 Run LiveTrader first to create the database")
        return False
    
    print(f"🔍 Migrating database: {db_path}")
    
    try:
        # Connect to database
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Check current schema
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        existing_tables = [row[0] for row in cursor.fetchall()]
        print(f"📋 Existing tables: {existing_tables}")
        
        # Check if indicators table has ts_ms column
        cursor.execute("PRAGMA table_info(indicators);")
        columns = [row[1] for row in cursor.fetchall()]
        has_ts_ms = 'ts_ms' in columns
        
        print(f"📊 Indicators table columns: {columns}")
        print(f"🕐 Has ts_ms column: {has_ts_ms}")
        
        # Begin transaction
        cursor.execute("BEGIN;")
        
        # Add ts_ms column if it doesn't exist
        if not has_ts_ms:
            print("➕ Adding ts_ms column...")
            cursor.execute("ALTER TABLE indicators ADD COLUMN ts_ms INTEGER;")
            
            # Populate ts_ms from existing timestamp column
            print("📝 Populating ts_ms values...")
            cursor.execute("UPDATE indicators SET ts_ms = CAST(timestamp * 1000 AS INTEGER) WHERE ts_ms IS NULL;")
            
            updated_rows = cursor.rowcount
            print(f"✅ Updated {updated_rows} rows with ts_ms values")
        else:
            print("✅ ts_ms column already exists")
        
        # Create indexes
        print("🔗 Creating indexes...")
        indexes = [
            ("idx_indicators_ts_ms", "CREATE INDEX IF NOT EXISTS idx_indicators_ts_ms ON indicators(ts_ms);"),
            ("idx_indicators_name", "CREATE INDEX IF NOT EXISTS idx_indicators_name ON indicators(indicator_name);"),
            ("idx_indicators_name_ts", "CREATE INDEX IF NOT EXISTS idx_indicators_name_ts ON indicators(indicator_name, ts_ms);")
        ]
        
        for idx_name, idx_sql in indexes:
            cursor.execute(idx_sql)
            print(f"   ✅ {idx_name}")
        
        # Create equity table
        if 'equity' not in existing_tables:
            print("➕ Creating equity table...")
            cursor.execute("""
                CREATE TABLE equity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    ts_ms INTEGER NOT NULL,
                    eq REAL NOT NULL,
                    symbol TEXT DEFAULT 'BTCUSDT',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            # Create equity indexes
            equity_indexes = [
                "CREATE INDEX idx_equity_ts ON equity(ts);",
                "CREATE INDEX idx_equity_ts_ms ON equity(ts_ms);",
                "CREATE INDEX idx_equity_symbol ON equity(symbol);"
            ]
            
            for idx_sql in equity_indexes:
                cursor.execute(idx_sql)
            
            print("✅ Equity table created with indexes")
        else:
            print("✅ Equity table already exists")
        
        # Create trades table
        if 'trades' not in existing_tables:
            print("➕ Creating trades table...")
            cursor.execute("""
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
            """)
            
            # Create trades indexes
            trades_indexes = [
                "CREATE INDEX idx_trades_entry_time ON trades(entry_time);",
                "CREATE INDEX idx_trades_symbol ON trades(symbol);",
                "CREATE INDEX idx_trades_strategy ON trades(strategy);"
            ]
            
            for idx_sql in trades_indexes:
                cursor.execute(idx_sql)
            
            print("✅ Trades table created with indexes")
        else:
            print("✅ Trades table already exists")
        
        # Commit transaction
        cursor.execute("COMMIT;")
        
        # Verify migration
        print("\n📊 Migration verification:")
        
        # Check indicators count
        cursor.execute("SELECT COUNT(*) FROM indicators WHERE ts_ms IS NOT NULL;")
        indicators_count = cursor.fetchone()[0]
        print(f"   Indicators with ts_ms: {indicators_count}")
        
        # Check equity count
        cursor.execute("SELECT COUNT(*) FROM equity;")
        equity_count = cursor.fetchone()[0]
        print(f"   Equity records: {equity_count}")
        
        # Check trades count
        cursor.execute("SELECT COUNT(*) FROM trades;")
        trades_count = cursor.fetchone()[0]
        print(f"   Trade records: {trades_count}")
        
        # List all indexes
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%';")
        indexes = [row[0] for row in cursor.fetchall()]
        print(f"   Indexes created: {len(indexes)}")
        for idx in indexes:
            print(f"     - {idx}")
        
        conn.close()
        print("\n🎉 Database migration completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Migration error: {e}")
        try:
            cursor.execute("ROLLBACK;")
            conn.close()
        except:
            pass
        return False

if __name__ == "__main__":
    success = run_migration()
    sys.exit(0 if success else 1)
