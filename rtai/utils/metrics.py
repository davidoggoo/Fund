"""Enhanced performance monitoring and metrics for RTAI with p50/p99 tracking"""

import time
import psutil
import os
from typing import Dict, List, Optional, Deque
from dataclasses import dataclass, field
from statistics import mean, stdev, median
from collections import deque
import numpy as np
from loguru import logger

@dataclass
class MetricSnapshot:
    """Enhanced snapshot of system metrics at a point in time"""
    timestamp: float
    processing_time_ms: float
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    websocket_lag_ms: Optional[float] = None
    trades_processed: int = 0
    signals_generated: int = 0
    queue_size: int = 0
    connection_health: bool = True

@dataclass 
class PerformanceMetrics:
    """Enhanced system performance tracking with percentile metrics"""
    start_time: float = field(default_factory=time.time)
    snapshots: List[MetricSnapshot] = field(default_factory=list)
    total_trades: int = 0
    total_signals: int = 0
    last_trade_time: Optional[float] = None
    
    # Enhanced metrics tracking
    processing_times: Deque[float] = field(default_factory=lambda: deque(maxlen=10000))
    websocket_lags: Deque[float] = field(default_factory=lambda: deque(maxlen=10000))
    memory_usage: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    cpu_usage: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    
    # Process handle for CPU/memory monitoring
    _process: Optional[psutil.Process] = field(default=None, init=False)
    _last_system_check: float = field(default=0.0, init=False)
    
    def __post_init__(self):
        """Initialize process monitoring"""
        try:
            self._process = psutil.Process(os.getpid())
        except psutil.Error:
            logger.warning("Failed to initialize process monitoring")
    
    def record_latency(self, latency_seconds: float):
        """Record latency measurement"""
        self.processing_times.append(latency_seconds * 1000)  # Convert to ms
    
    def record_throughput(self, count: int):
        """Record throughput measurement"""
        self.total_trades += count
    
    def get_latency_stats(self) -> Dict:
        """Get latency statistics"""
        if not self.processing_times:
            return {}
        
        times = list(self.processing_times)
        return {
            'mean_ms': mean(times),
            'median_ms': median(times),
            'p99_ms': np.percentile(times, 99) if len(times) >= 10 else None,
            'count': len(times)
        }
    
    def get_throughput_stats(self) -> Dict:
        """Get throughput statistics"""
        duration = time.time() - self.start_time
        return {
            'total_trades': self.total_trades,
            'trades_per_second': self.total_trades / max(duration, 1),
            'duration_seconds': duration
        }
    
    def _update_system_metrics(self) -> tuple[Optional[float], Optional[float]]:
        """Update system-level metrics (CPU, memory) with rate limiting"""
        current_time = time.time()
        
        # Rate limit system checks to every 1 second
        if current_time - self._last_system_check < 1.0:
            return None, None
            
        self._last_system_check = current_time
        
        try:
            if self._process:
                # Memory usage in MB
                memory_info = self._process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                self.memory_usage.append(memory_mb)
                
                # CPU usage percentage (fix audit: add interval for first call)
                cpu_percent = self._process.cpu_percent(interval=0.1)  # 100ms interval for accuracy
                self.cpu_usage.append(cpu_percent)
                
                return memory_mb, cpu_percent
        except psutil.Error as e:
            logger.debug(f"System metrics collection error: {e}")
            
        return None, None
    
    def record_trade_processing(self, processing_time_ms: float, websocket_lag_ms: Optional[float] = None, 
                              queue_size: int = 0, connection_healthy: bool = True):
        """Enhanced trade processing metrics recording"""
        self.total_trades += 1
        self.last_trade_time = time.time()
        
        # Store for percentile calculations
        self.processing_times.append(processing_time_ms)
        if websocket_lag_ms is not None:
            self.websocket_lags.append(websocket_lag_ms)
        
        # Update system metrics
        memory_mb, cpu_percent = self._update_system_metrics()
        
        snapshot = MetricSnapshot(
            timestamp=time.time(),
            processing_time_ms=processing_time_ms,
            memory_usage_mb=memory_mb,
            cpu_usage_percent=cpu_percent,
            websocket_lag_ms=websocket_lag_ms,
            trades_processed=1,
            queue_size=queue_size,
            connection_health=connection_healthy
        )
        
        self.snapshots.append(snapshot)
        
        # Keep only last 1000 snapshots to prevent memory issues
        if len(self.snapshots) > 1000:
            self.snapshots = self.snapshots[-1000:]
    
    def record_signal(self):
        """Record signal generation"""
        self.total_signals += 1
    
    def get_percentile_stats(self, data: Deque[float], name: str) -> Dict[str, float]:
        """Calculate percentile statistics for a metric"""
        if len(data) < 10:
            return {f"{name}_count": len(data)}
            
        data_array = np.array(data)
        return {
            f"{name}_count": len(data),
            f"{name}_mean": float(np.mean(data_array)),
            f"{name}_median": float(np.median(data_array)),
            f"{name}_p50": float(np.percentile(data_array, 50)),
            f"{name}_p95": float(np.percentile(data_array, 95)),
            f"{name}_p99": float(np.percentile(data_array, 99)),
            f"{name}_min": float(np.min(data_array)),
            f"{name}_max": float(np.max(data_array)),
            f"{name}_std": float(np.std(data_array))
        }
    
    def get_stats(self) -> Dict:
        """Enhanced performance statistics with percentiles"""
        if not self.snapshots:
            return {"status": "no_data"}
            
        runtime_minutes = (time.time() - self.start_time) / 60
        
        # Basic stats
        basic_stats = {
            "runtime_minutes": round(runtime_minutes, 2),
            "total_trades": self.total_trades,
            "total_signals": self.total_signals,
            "trades_per_minute": round(self.total_trades / max(runtime_minutes, 0.01), 2),
        }
        
        # Enhanced percentile statistics
        stats = basic_stats.copy()
        
        # Processing time percentiles
        if self.processing_times:
            stats.update(self.get_percentile_stats(self.processing_times, "processing_time_ms"))
        
        # WebSocket lag percentiles
        if self.websocket_lags:
            stats.update(self.get_percentile_stats(self.websocket_lags, "websocket_lag_ms"))
        
        # Memory usage percentiles
        if self.memory_usage:
            stats.update(self.get_percentile_stats(self.memory_usage, "memory_mb"))
        
        # CPU usage percentiles
        if self.cpu_usage:
            stats.update(self.get_percentile_stats(self.cpu_usage, "cpu_percent"))
        
        # Performance alerts based on percentiles
        alerts = []
        if stats.get("processing_time_ms_p99", 0) > 500:
            alerts.append("HIGH_P99_PROCESSING_TIME")
        if stats.get("websocket_lag_ms_p95", 0) > 2000:
            alerts.append("HIGH_P95_WEBSOCKET_LAG")  
        if stats.get("memory_mb_mean", 0) > 1000:
            alerts.append("HIGH_MEMORY_USAGE")
        if stats.get("cpu_percent_p95", 0) > 80:
            alerts.append("HIGH_CPU_USAGE")
        if stats["trades_per_minute"] < 1:
            alerts.append("LOW_TRADE_RATE")
            
        if alerts:
            stats["alerts"] = alerts
        
        return stats
    
    def log_performance_summary(self):
        """Enhanced performance summary logging"""
        stats = self.get_stats()
        
        if stats.get("status") == "no_data":
            logger.info("📊 No performance data available")
            return
            
        logger.info(f"📊 Enhanced Performance Summary:")
        logger.info(f"   Runtime: {stats['runtime_minutes']} minutes")
        logger.info(f"   Trades processed: {stats['total_trades']} ({stats['trades_per_minute']}/min)")
        logger.info(f"   Signals generated: {stats['total_signals']}")
        
        # Processing time percentiles
        if 'processing_time_ms_p50' in stats:
            logger.info(f"   Processing time: p50={stats['processing_time_ms_p50']:.2f}ms, p95={stats['processing_time_ms_p95']:.2f}ms, p99={stats['processing_time_ms_p99']:.2f}ms")
        
        # WebSocket lag percentiles  
        if 'websocket_lag_ms_p50' in stats and len(self.websocket_lags) >= 100:
            logger.info(f"   WebSocket lag: p50={stats['websocket_lag_ms_p50']:.2f}ms, p95={stats['websocket_lag_ms_p95']:.2f}ms, p99={stats['websocket_lag_ms_p99']:.2f}ms")
        
        # System resources
        if 'memory_mb_mean' in stats:
            logger.info(f"   Memory usage: avg={stats['memory_mb_mean']:.1f}MB, max={stats['memory_mb_max']:.1f}MB")
        if 'cpu_percent_mean' in stats:
            logger.info(f"   CPU usage: avg={stats['cpu_percent_mean']:.1f}%, p95={stats['cpu_percent_p95']:.1f}%")
            
        if "alerts" in stats:
            logger.warning(f"🚨 Performance alerts: {', '.join(stats['alerts'])}")
    
    def get_health_score(self) -> float:
        """Calculate overall system health score (0-1)"""
        stats = self.get_stats()
        if stats.get("status") == "no_data":
            return 0.5  # Neutral score for no data
        
        score = 1.0
        
        # Deduct for high latencies
        if stats.get("processing_time_ms_p99", 0) > 100:
            score -= 0.2
        if stats.get("websocket_lag_ms_p95", 0) > 1000:
            score -= 0.3
        
        # Deduct for resource usage
        if stats.get("memory_mb_mean", 0) > 500:
            score -= 0.1
        if stats.get("cpu_percent_p95", 0) > 60:
            score -= 0.2
        
        # Deduct for low throughput
        if stats.get("trades_per_minute", 0) < 10:
            score -= 0.2
            
        return max(0.0, min(1.0, score))  # Clamp 0-1

class PerformanceTimer:
    """Context manager for timing operations"""
    
    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
    
    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds"""
        if self.end_time is None:
            return (time.perf_counter() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000
