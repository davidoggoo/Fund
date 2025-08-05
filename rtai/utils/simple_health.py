"""
Simple Windows-Compatible Health Check
=====================================

Lightweight health monitoring that works reliably on Windows PowerShell.
No blocking calls, no complex psutil operations.
"""

import time
import threading
from typing import Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class SimpleHealthStatus:
    """Simplified health status for Windows"""
    timestamp: str
    status: str  # "healthy", "degraded", "unhealthy"
    uptime_seconds: float
    trades_processed: int
    indicators_calculated: int
    feed_connected: bool
    error_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SimpleHealthMonitor:
    """Simplified health monitor for Windows compatibility"""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics_lock = threading.Lock()
        self.metrics = {
            'trades_processed': 0,
            'indicators_calculated': 0,
            'errors_count': 0,
            'feed_connected': False
        }
    
    def record_trade(self) -> None:
        """Record a processed trade"""
        with self.metrics_lock:
            self.metrics['trades_processed'] += 1
    
    def record_indicator_calculation(self) -> None:
        """Record an indicator calculation"""
        with self.metrics_lock:
            self.metrics['indicators_calculated'] += 1
    
    def record_error(self, error_type: str = "general") -> None:
        """Record an error occurrence"""
        with self.metrics_lock:
            self.metrics['errors_count'] += 1
    
    def set_feed_status(self, connected: bool) -> None:
        """Update feed connection status"""
        with self.metrics_lock:
            self.metrics['feed_connected'] = connected
    
    def get_health_status(self) -> SimpleHealthStatus:
        """Get current health status - Windows compatible"""
        
        uptime = time.time() - self.start_time
        
        with self.metrics_lock:
            trades = self.metrics['trades_processed']
            indicators = self.metrics['indicators_calculated']
            errors = self.metrics['errors_count']
            feed_connected = self.metrics['feed_connected']
        
        # Simple health determination
        if not feed_connected and uptime > 60:  # Only check feed after 1 minute
            status = "unhealthy"
        elif errors > 10:
            status = "unhealthy" 
        elif errors > 3:
            status = "degraded"
        else:
            status = "healthy"
        
        return SimpleHealthStatus(
            timestamp=datetime.now().isoformat(),
            status=status,
            uptime_seconds=uptime,
            trades_processed=trades,
            indicators_calculated=indicators,
            feed_connected=feed_connected,
            error_count=errors
        )


# Global simple monitor instance
_simple_monitor = SimpleHealthMonitor()


def record_trade():
    """Record a processed trade"""
    _simple_monitor.record_trade()


def record_indicator_calculation():
    """Record an indicator calculation"""
    _simple_monitor.record_indicator_calculation()


def record_error(error_type: str = "general"):
    """Record an error occurrence"""
    _simple_monitor.record_error(error_type)


def set_feed_status(connected: bool):
    """Update feed connection status"""
    _simple_monitor.set_feed_status(connected)


def get_health_status() -> SimpleHealthStatus:
    """Get current system health status - Windows compatible"""
    return _simple_monitor.get_health_status()


if __name__ == "__main__":
    # Test the simple health monitor
    print("🏥 Testing Simple Health Monitor (Windows Compatible)")
    print("=" * 60)
    
    # Simulate some activity
    record_trade()
    record_indicator_calculation() 
    set_feed_status(True)
    
    # Get status
    status = get_health_status()
    print(f"✅ Status: {status.status}")
    print(f"📊 Uptime: {status.uptime_seconds:.1f}s")
    print(f"📈 Trades: {status.trades_processed}")
    print(f"🔢 Indicators: {status.indicators_calculated}")
    print(f"📡 Feed Connected: {status.feed_connected}")
    print(f"❌ Errors: {status.error_count}")
    
    print("\n🎉 Simple health monitor test completed successfully!")
