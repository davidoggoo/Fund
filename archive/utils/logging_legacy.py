"""
Singleton production logging configuration for RTAI system.
Enhanced with structured output, performance metrics, and component-specific loggers.
"""

import os
import sys
import time
import threading
from pathlib import Path
from typing import Dict, Optional, Any
from loguru import logger
from datetime import datetime

# Thread-safe singleton pattern
_logging_lock = threading.Lock()
_logging_configured = False
_component_loggers: Dict[str, Any] = {}


class RTAILogger:
    """Singleton RTAI logger with component-specific configuration"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized') or not self._initialized:
            self.log_level = "INFO"
            self.log_dir = Path("logs")
            self.component_loggers = {}
            self.performance_metrics = {}
            self._initialized = True
    
    def configure(self, log_level: str = "INFO", log_file: bool = True, 
                 performance_logging: bool = True) -> None:
        """Configure singleton logging system"""
        global _logging_configured
        
        with _logging_lock:
            if _logging_configured:
                logger.debug("🔄 RTAI Logging already configured, skipping")
                return
            
            self.log_level = log_level
            self._setup_base_logging(log_level, log_file)
            
            if performance_logging:
                self._setup_performance_logging()
            
            self._setup_component_loggers()
            
            _logging_configured = True
            logger.info("🚀 RTAI Singleton Logging System Initialized")
    
    def _setup_base_logging(self, log_level: str, log_file: bool) -> None:
        """Setup base logging configuration"""
        # Remove default handler to prevent duplicates
        logger.remove()
        
        # Enhanced console handler with component info
        logger.add(
            sys.stderr,
            format="<green>{time:HH:mm:ss.SSS}</green> | <level>{level: <5}</level> | <cyan>{name}</cyan> | <level>{message}</level>",
            level=log_level,
            colorize=True,
            backtrace=False,
            diagnose=False
        )
        
        if log_file:
            self.log_dir.mkdir(exist_ok=True)
            
            # Main application log with rotation
            logger.add(
                self.log_dir / "rtai_{time:YYYY-MM-DD}.log",
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <5} | {name} | {function}:{line} | {message}",
                level=log_level,
                rotation="100 MB",
                retention="7 days",
                compression="gz",
                enqueue=True  # Thread-safe
            )
            
            # Critical errors log
            logger.add(
                self.log_dir / "rtai_errors_{time:YYYY-MM-DD}.log",
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <5} | {name} | {function}:{line} | {message}",
                level="ERROR",
                rotation="1 day",
                retention="90 days",
                compression="gz",
                enqueue=True
            )
    
    def _setup_performance_logging(self) -> None:
        """Setup performance monitoring logging"""
        # Performance metrics log
        logger.add(
            self.log_dir / "rtai_performance_{time:YYYY-MM-DD}.log",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | PERF | {message}",
            level="DEBUG",
            rotation="1 day",
            retention="3 days",
            filter=lambda record: "PERF" in record["message"],
            enqueue=True
        )
    
    def _setup_component_loggers(self) -> None:
        """Setup component-specific loggers"""
        components = [
            "rtai.live_trader",
            "rtai.indicators", 
            "rtai.plotting",
            "rtai.state",
            "rtai.utils",
            "rtai.main"
        ]
        
        for component in components:
            self.component_loggers[component] = logger.bind(component=component)
    
    def get_component_logger(self, component: str):
        """Get component-specific logger"""
        return self.component_loggers.get(component, logger)
    
    def log_performance(self, component: str, operation: str, duration_ms: float, 
                       extra_data: Optional[Dict[str, Any]] = None) -> None:
        """Log performance metrics"""
        perf_data = {
            'component': component,
            'operation': operation,
            'duration_ms': round(duration_ms, 3),
            'timestamp': datetime.now().isoformat()
        }
        
        if extra_data:
            perf_data.update(extra_data)
        
        # Store in memory for aggregation
        key = f"{component}.{operation}"
        if key not in self.performance_metrics:
            self.performance_metrics[key] = []
        
        self.performance_metrics[key].append(duration_ms)
        
        # Keep only recent measurements
        if len(self.performance_metrics[key]) > 1000:
            self.performance_metrics[key] = self.performance_metrics[key][-500:]
        
        # Log to performance file
        logger.debug(f"PERF | {component} | {operation} | {duration_ms:.3f}ms | {extra_data or {}}")
    
    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get aggregated performance statistics"""
        stats = {}
        
        for key, measurements in self.performance_metrics.items():
            if measurements:
                import statistics
                stats[key] = {
                    'count': len(measurements),
                    'avg_ms': round(statistics.mean(measurements), 3),
                    'median_ms': round(statistics.median(measurements), 3),
                    'p95_ms': round(statistics.quantiles(measurements, n=20)[18], 3) if len(measurements) >= 20 else round(max(measurements), 3),
                    'min_ms': round(min(measurements), 3),
                    'max_ms': round(max(measurements), 3)
                }
        
        return stats


# Singleton instance
_rtai_logger = RTAILogger()


def configure_logging(log_level: str = "INFO", log_file: bool = True, 
                     performance_logging: bool = True) -> None:
    """
    Configure singleton RTAI logging system.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Enable file logging
        performance_logging: Enable performance metrics logging
    """
    _rtai_logger.configure(log_level, log_file, performance_logging)


def get_component_logger(component: str):
    """Get component-specific logger"""
    return _rtai_logger.get_component_logger(component)


def log_performance(component: str, operation: str, duration_ms: float, 
                   extra_data: Optional[Dict[str, Any]] = None) -> None:
    """Log performance metrics"""
    _rtai_logger.log_performance(component, operation, duration_ms, extra_data)


def get_performance_stats() -> Dict[str, Dict[str, float]]:
    """Get aggregated performance statistics"""
    return _rtai_logger.get_performance_stats()


def get_log_level() -> str:
    """Get log level from environment or default to INFO"""
    return os.getenv("RTAI_LOG_LEVEL", "INFO").upper()


class PerformanceTimer:
    """Context manager for performance timing"""
    
    def __init__(self, component: str, operation: str, extra_data: Optional[Dict[str, Any]] = None):
        self.component = component
        self.operation = operation
        self.extra_data = extra_data
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration_ms = (time.perf_counter() - self.start_time) * 1000
            log_performance(self.component, self.operation, duration_ms, self.extra_data)
