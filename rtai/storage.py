"""Storage module for RTAI indicators data with atomic indicators support"""

import sqlite3
import time
from typing import Dict, Optional
from loguru import logger
import os
from pathlib import Path


class IndicatorStorage:
    """Database storage for indicators with atomic indicators support
    
    Handles both traditional indicators and new atomic indicators:
    - Liquidations (USD notional, count)
    - TopOfBookImbalance (TOBI ratio)
    - WallRatio (large order percentage)
    - FundingBasis (spot-perp spread)
    - TradeImbalance (buy/sell ratio)
    - Z-band indicators (rolling Z-scores)
    """
    
    def __init__(self, db_path: str = "data/rtai_features.db"):
        self.db_path = db_path
        
        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Initialize database with tables for features and Z-state"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Main features table with atomic indicators
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS features_1m (
                        timestamp INTEGER PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        ofi REAL,
                        vpin REAL,
                        kyle REAL,
                        lpi REAL,
                        liq_usd REAL DEFAULT 0,
                        liq_cnt INTEGER DEFAULT 0,
                        wall_ratio REAL DEFAULT 0,
                        basis REAL DEFAULT 0,
                        tobi REAL DEFAULT 0,
                        trade_imb REAL DEFAULT 0,
                        liq_z REAL,
                        wall_ratio_z REAL,
                        basis_z REAL,
                        trade_imb_z REAL,
                        price REAL,
                        volume REAL,
                        trades_count INTEGER DEFAULT 0
                    )
                """)
                
                # Z-band state persistence table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS z_state (
                        indicator_name TEXT,
                        timestamp INTEGER,
                        value REAL,
                        PRIMARY KEY (indicator_name, timestamp)
                    )
                """)
                
                # Create indices for performance
                conn.execute("CREATE INDEX IF NOT EXISTS idx_features_timestamp ON features_1m (timestamp DESC)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_features_symbol ON features_1m (symbol)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_z_state_indicator ON z_state (indicator_name, timestamp DESC)")
                
                conn.commit()
                logger.info(f"Database initialized: {self.db_path}")
                
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            
    def store_minute_features(self, symbol: str, indicators: Dict[str, float], 
                            price: float = None, volume: float = None, trades_count: int = 0) -> bool:
        """Store minute-level features with atomic indicators
        
        Args:
            symbol: Trading symbol
            indicators: Dictionary with all indicator values
            price: Current price
            volume: Minute volume
            trades_count: Number of trades this minute
        """
        try:
            timestamp = int(time.time() // 60) * 60  # Round to minute
            
            with sqlite3.connect(self.db_path) as conn:
                # Insert or replace minute features
                conn.execute("""
                    INSERT OR REPLACE INTO features_1m (
                        timestamp, symbol, ofi, vpin, kyle, lpi,
                        liq_usd, liq_cnt, wall_ratio, basis, tobi, trade_imb,
                        liq_z, wall_ratio_z, basis_z, trade_imb_z,
                        price, volume, trades_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestamp, symbol,
                    indicators.get('ofi'), indicators.get('vpin'), 
                    indicators.get('kyle'), indicators.get('lpi'),
                    indicators.get('liq_usd', 0.0), indicators.get('liq_cnt', 0),
                    indicators.get('wall_ratio', 0.0), indicators.get('basis', 0.0),
                    indicators.get('tobi', 0.5), indicators.get('trade_imb', 0.0),
                    indicators.get('liq_z'), indicators.get('wall_ratio_z'),
                    indicators.get('basis_z'), indicators.get('trade_imb_z'),
                    price, volume, trades_count
                ))
                
                conn.commit()
                logger.debug(f"Stored minute features for {symbol} at {timestamp}")
                return True
                
        except Exception as e:
            logger.error(f"Error storing minute features: {e}")
            return False
            
    def store_z_state(self, indicator_name: str, values: list) -> bool:
        """Store Z-band state for persistence across restarts
        
        Args:
            indicator_name: Name of the indicator (e.g., 'liq_z')
            values: List of recent values for Z-score calculation
        """
        try:
            current_time = int(time.time())
            
            with sqlite3.connect(self.db_path) as conn:
                # Clear old state for this indicator (keep last 120 minutes)
                cutoff_time = current_time - (120 * 60)
                conn.execute("""
                    DELETE FROM z_state 
                    WHERE indicator_name = ? AND timestamp < ?
                """, (indicator_name, cutoff_time))
                
                # Insert new values
                for i, value in enumerate(values[-120:]):  # Keep last 120 values
                    timestamp = current_time - (len(values) - 1 - i) * 60
                    conn.execute("""
                        INSERT OR REPLACE INTO z_state (indicator_name, timestamp, value)
                        VALUES (?, ?, ?)
                    """, (indicator_name, timestamp, float(value)))
                
                conn.commit()
                logger.debug(f"Stored Z-state for {indicator_name}: {len(values)} values")
                return True
                
        except Exception as e:
            logger.error(f"Error storing Z-state: {e}")
            return False
            
    def load_z_state(self, indicator_name: str) -> list:
        """Load Z-band state for indicator persistence
        
        Args:
            indicator_name: Name of the indicator
            
        Returns:
            List of recent values for Z-score calculation
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT value FROM z_state 
                    WHERE indicator_name = ? 
                    ORDER BY timestamp ASC
                """, (indicator_name,))
                
                values = [row[0] for row in cursor.fetchall()]
                logger.debug(f"Loaded Z-state for {indicator_name}: {len(values)} values")
                return values
                
        except Exception as e:
            logger.error(f"Error loading Z-state: {e}")
            return []
            
    def get_recent_features(self, symbol: str, minutes: int = 60) -> list:
        """Get recent features for analysis
        
        Args:
            symbol: Trading symbol
            minutes: Number of recent minutes to retrieve
            
        Returns:
            List of feature dictionaries
        """
        try:
            cutoff_time = int(time.time()) - (minutes * 60)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row  # Enable column access by name
                
                cursor = conn.execute("""
                    SELECT * FROM features_1m 
                    WHERE symbol = ? AND timestamp >= ? 
                    ORDER BY timestamp ASC
                """, (symbol, cutoff_time))
                
                features = [dict(row) for row in cursor.fetchall()]
                logger.debug(f"Retrieved {len(features)} recent features for {symbol}")
                return features
                
        except Exception as e:
            logger.error(f"Error retrieving recent features: {e}")
            return []
            
    def get_statistics(self) -> Dict[str, int]:
        """Get database statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Count total features
                cursor = conn.execute("SELECT COUNT(*) FROM features_1m")
                features_count = cursor.fetchone()[0]
                
                # Count Z-state entries
                cursor = conn.execute("SELECT COUNT(*) FROM z_state")
                z_state_count = cursor.fetchone()[0]
                
                # Get unique symbols
                cursor = conn.execute("SELECT COUNT(DISTINCT symbol) FROM features_1m")
                symbols_count = cursor.fetchone()[0]
                
                return {
                    'total_features': features_count,
                    'z_state_entries': z_state_count,
                    'symbols_tracked': symbols_count,
                    'database_size_mb': os.path.getsize(self.db_path) / (1024 * 1024) if os.path.exists(self.db_path) else 0
                }
                
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
            
    def cleanup_old_data(self, days_to_keep: int = 7):
        """Clean up old data to manage database size
        
        Args:
            days_to_keep: Number of days of data to keep
        """
        try:
            cutoff_time = int(time.time()) - (days_to_keep * 24 * 3600)
            
            with sqlite3.connect(self.db_path) as conn:
                # Clean old features
                cursor = conn.execute("DELETE FROM features_1m WHERE timestamp < ?", (cutoff_time,))
                features_deleted = cursor.rowcount
                
                # Clean old Z-state (keep last 120 minutes only)
                z_cutoff = int(time.time()) - (120 * 60)
                cursor = conn.execute("DELETE FROM z_state WHERE timestamp < ?", (z_cutoff,))
                z_state_deleted = cursor.rowcount
                
                conn.commit()
                
                logger.info(f"Cleaned up database: {features_deleted} old features, {z_state_deleted} old Z-state entries")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
