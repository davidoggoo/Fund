"""
Health Monitoring System for RTAI
=================================

Provides system health checks, metrics collection, and monitoring endpoints.
Used for production monitoring and alerting.
"""

import time
import os
import psutil
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from loguru import logger
from pathlib import Path


@dataclass
class HealthStatus:
    """System health status snapshot"""
    timestamp: str
    status: str  # "healthy", "degraded", "unhealthy"
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    active_connections: int
    error_rate_5min: float
    last_trade_timestamp: Optional[str]
    uptime_seconds: float
    indicators_calculated: int
    feed_connected: bool
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HealthMonitor:
    """System health monitoring with metrics collection"""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics_lock = threading.Lock()
        self.metrics = {
            'trades_processed': 0,
            'indicators_calculated': 0,
            'errors_5min': 0,
            'last_trade_time': None,
            'feed_connected': False,
            'connection_errors': 0
        }
        self.error_timestamps = []
        self.health_history = []
        self.max_history_size = 1000
        
    def record_trade(self) -> None:
        """Record a processed trade"""
        with self.metrics_lock:
            self.metrics['trades_processed'] += 1
            self.metrics['last_trade_time'] = datetime.now()
    
    def record_indicator_calculation(self) -> None:
        """Record an indicator calculation"""
        with self.metrics_lock:
            self.metrics['indicators_calculated'] += 1
    
    def record_error(self, error_type: str = "general") -> None:
        """Record an error occurrence"""
        now = datetime.now()
        with self.metrics_lock:
            self.error_timestamps.append(now)
            # Keep only last 5 minutes of errors
            cutoff = now - timedelta(minutes=5)
            self.error_timestamps = [ts for ts in self.error_timestamps if ts > cutoff]
    
    def set_feed_status(self, connected: bool) -> None:
        """Update feed connection status"""
        with self.metrics_lock:
            if not connected:
                self.metrics['connection_errors'] += 1
            self.metrics['feed_connected'] = connected
    
    def get_error_rate_5min(self) -> float:
        """Get error rate in last 5 minutes (errors per minute)"""
        now = datetime.now()
        cutoff = now - timedelta(minutes=5)
        
        with self.metrics_lock:
            recent_errors = [ts for ts in self.error_timestamps if ts > cutoff]
            return len(recent_errors) / 5.0  # errors per minute
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system resource metrics with Windows-compatible timeout protection"""
        
        try:
            # Use non-blocking CPU measurement for Windows compatibility
            cpu_percent = psutil.cpu_percent(interval=None)  # Non-blocking
            if cpu_percent == 0.0:  # First call returns 0.0, get a quick sample
                try:
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                except:
                    cpu_percent = -1  # Fallback if CPU measurement fails
            
            memory = psutil.virtual_memory()
            
            # Get disk usage for current drive (Windows compatible)
            try:
                if hasattr(os, 'name') and os.name == 'nt':
                    # Windows - use current drive
                    disk = psutil.disk_usage('C:')
                else:
                    # Unix-like systems
                    disk = psutil.disk_usage('/')
            except (PermissionError, FileNotFoundError, OSError):
                # Fallback for disk usage
                disk = type('disk', (), {'percent': -1})()
            
            # Network connections (skip on Windows to avoid hanging)
            connections = -1
            if hasattr(os, 'name') and os.name != 'nt':
                # Only try network connections on non-Windows systems
                try:
                    connections = len(psutil.net_connections())
                except (psutil.AccessDenied, psutil.NoSuchProcess, PermissionError):
                    connections = -1
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_usage_percent': disk.percent,
                'active_connections': connections,
                'uptime_seconds': time.time() - self.start_time
            }
        except Exception as e:
            logger.error(f"❌ Failed to get system metrics: {e}")
            return {
                'cpu_percent': -1,
                'memory_percent': -1,
                'disk_usage_percent': -1,
                'active_connections': -1,
                'uptime_seconds': time.time() - self.start_time
            }
    
    def determine_health_status(self, system_metrics: Dict[str, Any]) -> str:
        """Determine overall system health status"""
        
        error_rate = self.get_error_rate_5min()
        cpu_percent = system_metrics['cpu_percent']
        memory_percent = system_metrics['memory_percent']
        
        with self.metrics_lock:
            feed_connected = self.metrics['feed_connected']
            last_trade = self.metrics['last_trade_time']
        
        # Check for unhealthy conditions
        if not feed_connected:
            return "unhealthy"
        
        if error_rate > 10:  # More than 10 errors per minute
            return "unhealthy"
        
        if cpu_percent > 90 or memory_percent > 90:
            return "unhealthy"
        
        # Check for trade freshness (no trades in last 5 minutes might be an issue)
        if last_trade:
            time_since_trade = (datetime.now() - last_trade).total_seconds()
            if time_since_trade > 300:  # 5 minutes
                return "degraded"
        
        # Check for degraded conditions
        if error_rate > 2 or cpu_percent > 70 or memory_percent > 70:
            return "degraded"
        
        return "healthy"
    
    def get_health_status(self) -> HealthStatus:
        """Get current comprehensive health status"""
        
        system_metrics = self.get_system_metrics()
        health_status = self.determine_health_status(system_metrics)
        
        with self.metrics_lock:
            last_trade_str = None
            if self.metrics['last_trade_time']:
                last_trade_str = self.metrics['last_trade_time'].isoformat()
            
            status = HealthStatus(
                timestamp=datetime.now().isoformat(),
                status=health_status,
                cpu_percent=system_metrics['cpu_percent'],
                memory_percent=system_metrics['memory_percent'],
                disk_usage_percent=system_metrics['disk_usage_percent'],
                active_connections=system_metrics['active_connections'],
                error_rate_5min=self.get_error_rate_5min(),
                last_trade_timestamp=last_trade_str,
                uptime_seconds=system_metrics['uptime_seconds'],
                indicators_calculated=self.metrics['indicators_calculated'],
                feed_connected=self.metrics['feed_connected']
            )
        
        # Store in history
        self.health_history.append(status)
        if len(self.health_history) > self.max_history_size:
            self.health_history.pop(0)
        
        return status
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary for the last hour"""
        
        if not self.health_history:
            return {"error": "No health data available"}
        
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_statuses = [
            h for h in self.health_history 
            if datetime.fromisoformat(h.timestamp) > one_hour_ago
        ]
        
        if not recent_statuses:
            recent_statuses = self.health_history[-10:]  # Last 10 if no recent data
        
        healthy_count = len([s for s in recent_statuses if s.status == "healthy"])
        degraded_count = len([s for s in recent_statuses if s.status == "degraded"])
        unhealthy_count = len([s for s in recent_statuses if s.status == "unhealthy"])
        
        avg_cpu = sum(s.cpu_percent for s in recent_statuses if s.cpu_percent >= 0) / len(recent_statuses)
        avg_memory = sum(s.memory_percent for s in recent_statuses if s.memory_percent >= 0) / len(recent_statuses)
        
        return {
            "period": "last_hour",
            "total_checks": len(recent_statuses),
            "healthy_count": healthy_count,
            "degraded_count": degraded_count,
            "unhealthy_count": unhealthy_count,
            "avg_cpu_percent": round(avg_cpu, 2),
            "avg_memory_percent": round(avg_memory, 2),
            "current_status": recent_statuses[-1].status if recent_statuses else "unknown"
        }


# Global health monitor instance
_health_monitor = HealthMonitor()


def get_health_monitor() -> HealthMonitor:
    """Get the global health monitor instance"""
    return _health_monitor


def record_trade():
    """Record a processed trade"""
    _health_monitor.record_trade()


def record_indicator_calculation():
    """Record an indicator calculation"""
    _health_monitor.record_indicator_calculation()


def record_error(error_type: str = "general"):
    """Record an error occurrence"""
    _health_monitor.record_error(error_type)


def set_feed_status(connected: bool):
    """Update feed connection status"""
    _health_monitor.set_feed_status(connected)


def get_health_status() -> HealthStatus:
    """Get current system health status"""
    return _health_monitor.get_health_status()


def get_health_summary() -> Dict[str, Any]:
    """Get health summary"""
    return _health_monitor.get_health_summary()


def log_health_status():
    """Log current health status (structured JSON)"""
    health = get_health_status()
    
    logger.bind(
        component="health_monitor",
        health_status=health.status,
        cpu_percent=health.cpu_percent,
        memory_percent=health.memory_percent,
        error_rate_5min=health.error_rate_5min,
        feed_connected=health.feed_connected,
        uptime_seconds=health.uptime_seconds
    ).info("🏥 System Health Check")
