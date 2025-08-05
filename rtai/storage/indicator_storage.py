"""
Indicator Storage for RTAI System
Handles persistence of indicator values and metadata
"""
import json
import sqlite3
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
from loguru import logger


class IndicatorStorage:
    """Simple storage for indicator values and metadata"""
    
    def __init__(self, db_path: str = "state/indicators.db"):
        """Initialize storage with SQLite database"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS indicators (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        symbol TEXT NOT NULL,
                        indicator_name TEXT NOT NULL,
                        value REAL,
                        metadata TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp_symbol 
                    ON indicators(timestamp, symbol)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_indicator_name 
                    ON indicators(indicator_name)
                """)
                
                conn.commit()
                logger.info(f"Initialized indicator storage at {self.db_path}")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
    
    def store_indicator(self, timestamp: float, symbol: str, indicator_name: str, 
                       value: float, metadata: Optional[Dict[str, Any]] = None):
        """Store indicator value"""
        try:
            metadata_json = json.dumps(metadata) if metadata else None
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO indicators (timestamp, symbol, indicator_name, value, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (timestamp, symbol, indicator_name, value, metadata_json))
                conn.commit()
                
        except Exception as e:
            logger.warning(f"Failed to store indicator {indicator_name}: {e}")
    
    def get_indicators(self, symbol: str, indicator_name: Optional[str] = None,
                      start_time: Optional[float] = None, end_time: Optional[float] = None,
                      limit: int = 1000) -> List[Dict[str, Any]]:
        """Retrieve indicator values"""
        try:
            query = "SELECT * FROM indicators WHERE symbol = ?"
            params = [symbol]
            
            if indicator_name:
                query += " AND indicator_name = ?"
                params.append(indicator_name)
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                
                results = []
                for row in cursor.fetchall():
                    metadata = json.loads(row['metadata']) if row['metadata'] else {}
                    results.append({
                        'timestamp': row['timestamp'],
                        'symbol': row['symbol'],
                        'indicator_name': row['indicator_name'],
                        'value': row['value'],
                        'metadata': metadata
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to retrieve indicators: {e}")
            return []
    
    def get_latest_value(self, symbol: str, indicator_name: str) -> Optional[float]:
        """Get latest value for specific indicator"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT value FROM indicators 
                    WHERE symbol = ? AND indicator_name = ?
                    ORDER BY timestamp DESC LIMIT 1
                """, (symbol, indicator_name))
                
                row = cursor.fetchone()
                return row[0] if row else None
                
        except Exception as e:
            logger.warning(f"Failed to get latest value for {indicator_name}: {e}")
            return None
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Remove old indicator data"""
        try:
            cutoff_time = time.time() - (days_to_keep * 24 * 3600)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM indicators WHERE timestamp < ?
                """, (cutoff_time,))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} old indicator records")
                
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
    
    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Total records
                cursor = conn.execute("SELECT COUNT(*) FROM indicators")
                total_records = cursor.fetchone()[0]
                
                # Records by indicator
                cursor = conn.execute("""
                    SELECT indicator_name, COUNT(*) as count 
                    FROM indicators 
                    GROUP BY indicator_name
                """)
                by_indicator = dict(cursor.fetchall())
                
                # Date range
                cursor = conn.execute("""
                    SELECT MIN(timestamp) as min_time, MAX(timestamp) as max_time 
                    FROM indicators
                """)
                time_range = cursor.fetchone()
                
                return {
                    'total_records': total_records,
                    'by_indicator': by_indicator,
                    'time_range': {
                        'start': time_range[0] if time_range[0] else None,
                        'end': time_range[1] if time_range[1] else None
                    },
                    'db_size_mb': self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0
                }
                
        except Exception as e:
            logger.error(f"Failed to get storage stats: {e}")
            return {}