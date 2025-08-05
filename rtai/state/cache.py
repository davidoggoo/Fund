"""
RTAI Central Data Cache
======================

Centralized in-memory cache for all market data to decouple components
and ensure data consistency across the system.
"""

from typing import Dict, Any, Optional
import time
from dataclasses import dataclass, field
from threading import Lock

@dataclass
class DepthData:
    """Level 0 depth data"""
    ts: float
    bid_px: float
    bid_sz: float
    ask_px: float
    ask_sz: float
    symbol: str = "BTCUSDT"
    
    @property
    def mid_price(self) -> float:
        """Calculate mid price"""
        return (self.bid_px + self.ask_px) / 2
    
    @property
    def spread(self) -> float:
        """Calculate spread"""
        return self.ask_px - self.bid_px
    
    @property
    def spread_bps(self) -> float:
        """Calculate spread in basis points"""
        return (self.spread / self.mid_price) * 10000

@dataclass  
class TradeData:
    """Trade data"""
    ts: float
    price: float
    qty: float  # signed quantity
    side: str
    symbol: str = "BTCUSDT"

@dataclass
class OpenInterestData:
    """Open interest data"""
    ts: float
    oi: float
    symbol: str = "BTCUSDT"

@dataclass
class FundingData:
    """Funding rate data"""
    ts: float
    funding_rate: float
    mark_price: float
    symbol: str = "BTCUSDT"

@dataclass
class LiquidationData:
    """Liquidation data"""
    ts: float
    price: float
    qty: float  # signed quantity
    side: str  # 'long' or 'short'
    symbol: str = "BTCUSDT"

class DataCache:
    """
    Central data cache for all market data
    
    Thread-safe singleton cache that stores the latest market data
    from all streams. Indicators can access this data without coupling
    to the WebSocket feeds directly.
    """
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized'):
            return
        
        self._initialized = True
        self._data_lock = Lock()
        
        # Latest market data
        self.depth: Optional[DepthData] = None
        self.last_trade: Optional[TradeData] = None
        self.open_interest: Optional[OpenInterestData] = None
        self.funding: Optional[FundingData] = None
        self.last_liquidation: Optional[LiquidationData] = None
        
        # Indicator values cache
        self.indicators: Dict[str, float] = {}
        self.indicator_timestamps: Dict[str, float] = {}
        
        # Performance tracking
        self.update_counts: Dict[str, int] = {
            'depth': 0,
            'trade': 0, 
            'oi': 0,
            'funding': 0,
            'liquidation': 0
        }
        
        self.last_update_times: Dict[str, float] = {}
    
    def update_depth(self, ts: float, bid_px: float, bid_sz: float, 
                    ask_px: float, ask_sz: float, symbol: str = "BTCUSDT"):
        """Update depth data"""
        with self._data_lock:
            self.depth = DepthData(ts, bid_px, bid_sz, ask_px, ask_sz, symbol)
            self.update_counts['depth'] += 1
            self.last_update_times['depth'] = time.time()
    
    def update_trade(self, ts: float, price: float, qty: float, 
                    side: str, symbol: str = "BTCUSDT"):
        """Update trade data"""
        with self._data_lock:
            self.last_trade = TradeData(ts, price, qty, side, symbol)
            self.update_counts['trade'] += 1
            self.last_update_times['trade'] = time.time()
    
    def update_open_interest(self, ts: float, oi: float, symbol: str = "BTCUSDT"):
        """Update open interest data"""
        with self._data_lock:
            self.open_interest = OpenInterestData(ts, oi, symbol)
            self.update_counts['oi'] += 1
            self.last_update_times['oi'] = time.time()
    
    def update_funding(self, ts: float, funding_rate: float, mark_price: float, 
                      symbol: str = "BTCUSDT"):
        """Update funding data"""
        with self._data_lock:
            self.funding = FundingData(ts, funding_rate, mark_price, symbol)
            self.update_counts['funding'] += 1
            self.last_update_times['funding'] = time.time()
    
    def update_liquidation(self, ts: float, price: float, qty: float, 
                          side: str, symbol: str = "BTCUSDT"):
        """Update liquidation data"""
        with self._data_lock:
            self.last_liquidation = LiquidationData(ts, price, qty, side, symbol)
            self.update_counts['liquidation'] += 1
            self.last_update_times['liquidation'] = time.time()
    
    def update_indicator(self, name: str, value: float, ts: float):
        """Update indicator value"""
        with self._data_lock:
            self.indicators[name] = value
            self.indicator_timestamps[name] = ts
    
    def get_mid_price(self) -> Optional[float]:
        """Get current mid price from depth data"""
        if self.depth:
            return self.depth.mid_price
        return None
    
    def get_spread_bps(self) -> Optional[float]:
        """Get current spread in basis points"""
        if self.depth:
            return self.depth.spread_bps
        return None
    
    def get_indicator(self, name: str, max_age_seconds: float = 60) -> Optional[float]:
        """Get indicator value if not too old"""
        with self._data_lock:
            if name in self.indicators:
                ts = self.indicator_timestamps.get(name, 0)
                if time.time() - ts <= max_age_seconds:
                    return self.indicators[name]
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._data_lock:
            now = time.time()
            
            # Calculate data freshness
            freshness = {}
            for data_type, last_update in self.last_update_times.items():
                age = now - last_update if last_update else float('inf')
                freshness[f"{data_type}_age_seconds"] = age
            
            return {
                'update_counts': self.update_counts.copy(),
                'freshness': freshness,
                'has_depth': self.depth is not None,
                'has_trade': self.last_trade is not None,
                'has_oi': self.open_interest is not None,
                'has_funding': self.funding is not None,
                'indicator_count': len(self.indicators),
                'mid_price': self.get_mid_price(),
                'spread_bps': self.get_spread_bps()
            }
    
    def reset(self):
        """Reset all cached data (for testing)"""
        with self._data_lock:
            self.depth = None
            self.last_trade = None
            self.open_interest = None
            self.funding = None
            self.last_liquidation = None
            self.indicators.clear()
            self.indicator_timestamps.clear()
            self.update_counts = {k: 0 for k in self.update_counts}
            self.last_update_times.clear()

# Global cache instance
cache = DataCache()
