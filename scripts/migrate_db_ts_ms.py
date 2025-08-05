#!/usr/bin/env python3
"""
Database migration script to add ts_ms column for TradingVue compatibility
"""
import sqlite3
from pathlib import Path
import sys

def migrate_database():
    """Add ts_ms column and create indexes for TradingVue compatibility"""
    
    db_path = Path("state/indicators.db")
    if not db_path.exists():
        print(f"❌ Database not found at {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if ts_ms column already exists
        cursor.execute("PRAGMA table_info(indicators)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'ts_ms' not in columns:
            print("📝 Adding ts_ms column to indicators table...")
            cursor.execute("ALTER TABLE indicators ADD COLUMN ts_ms INTEGER")
            
            # Populate ts_ms from timestamp
            print("📝 Populating ts_ms from timestamp...")
            cursor.execute("UPDATE indicators SET ts_ms = CAST(timestamp * 1000 AS INTEGER) WHERE ts_ms IS NULL")
            
            # Create index on ts_ms
            print("📝 Creating index on ts_ms...")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_indicators_ts_ms ON indicators(ts_ms)")
        else:
            print("✅ ts_ms column already exists")
        
        # Create additional performance indexes
        print("📝 Creating performance indexes...")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_indicators_symbol_ts ON indicators(symbol, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_indicators_name_ts ON indicators(indicator_name, timestamp)")
        
        # Check if equity table exists, create if not
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS equity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                ts_ms INTEGER NOT NULL,
                eq REAL NOT NULL,
                symbol TEXT DEFAULT 'BTCUSDT',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_equity_ts ON equity(ts)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_equity_ts_ms ON equity(ts_ms)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_equity_symbol_ts ON equity(symbol, ts)")
        
        # Check if trades table has proper indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades(symbol, entry_time)")
        
        conn.commit()
        
        # Verify migration
        cursor.execute("SELECT COUNT(*) FROM indicators WHERE ts_ms IS NOT NULL")
        ts_ms_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM indicators")
        total_count = cursor.fetchone()[0]
        
        print(f"✅ Migration completed successfully!")
        print(f"   - Total indicators: {total_count}")
        print(f"   - With ts_ms: {ts_ms_count}")
        print(f"   - Success rate: {ts_ms_count/total_count*100:.1f}%" if total_count > 0 else "   - No data yet")
        
        return True
        
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False
    finally:
        conn.close()

if __name__ == "__main__":
    success = migrate_database()
    sys.exit(0 if success else 1)
