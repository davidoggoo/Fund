"""
Enhanced JSON Structured Logging for RTAI
==========================================

Provides structured JSON logging with correlation IDs, metrics, and monitoring integration.
Designed for production observability and log aggregation systems.
"""

import json
import uuid
import threading
from typing import Dict, Any, Optional
from datetime import datetime
from loguru import logger
from contextlib import contextmanager

# Export all main functions
__all__ = [
    'JSONFormatter',
    'get_structured_logger', 
    'correlation_context',
    'log_trade_processed',
    'log_indicator_calculation',
    'log_feed_event',
    'log_error_with_context',
    'configure_json_logging'
]


class JSONFormatter:
    """Custom JSON formatter for structured logging"""
    
    def format(self, record):
        """Format log record as structured JSON"""
        
        # Extract custom fields from record.extra
        custom_fields = {}
        if hasattr(record, 'extra'):
            custom_fields = record.extra.copy()
        
        # Base structure
        log_entry = {
            "timestamp": record["time"].isoformat(),
            "level": record["level"].name,
            "component": custom_fields.pop("component", "rtai"),
            "message": record["message"],
            "module": record["name"],
            "function": record["function"],
            "line": record["line"]
        }
        
        # Add correlation ID if present
        if "correlation_id" in custom_fields:
            log_entry["correlation_id"] = custom_fields.pop("correlation_id")
        
        # Add performance metrics if present
        if "duration_ms" in custom_fields:
            log_entry["performance"] = {
                "duration_ms": custom_fields.pop("duration_ms"),
                "operation": custom_fields.pop("operation", "unknown")
            }
        
        # Add health metrics if present
        health_fields = ["health_status", "cpu_percent", "memory_percent", 
                        "error_rate_5min", "feed_connected", "uptime_seconds"]
        health_data = {}
        for field in health_fields:
            if field in custom_fields:
                health_data[field] = custom_fields.pop(field)
        
        if health_data:
            log_entry["health"] = health_data
        
        # Add any remaining custom fields
        if custom_fields:
            log_entry["extra"] = custom_fields
        
        # Add exception info if present
        if record["exception"]:
            log_entry["exception"] = {
                "type": record["exception"].type.__name__ if record["exception"].type else None,
                "message": str(record["exception"].value) if record["exception"].value else None,
                "traceback": record["exception"].traceback if record["exception"].traceback else None
            }
        
        return json.dumps(log_entry, default=str)


class CorrelationContext:
    """Thread-local correlation ID context"""
    
    _local = threading.local()
    
    @classmethod
    def get_correlation_id(cls) -> Optional[str]:
        """Get current correlation ID"""
        return getattr(cls._local, 'correlation_id', None)
    
    @classmethod
    def set_correlation_id(cls, correlation_id: str) -> None:
        """Set correlation ID for current thread"""
        cls._local.correlation_id = correlation_id
    
    @classmethod
    def clear_correlation_id(cls) -> None:
        """Clear correlation ID for current thread"""
        if hasattr(cls._local, 'correlation_id'):
            delattr(cls._local, 'correlation_id')


@contextmanager
def correlation_context(correlation_id: Optional[str] = None):
    """Context manager for correlation ID"""
    
    if correlation_id is None:
        correlation_id = str(uuid.uuid4())
    
    old_id = CorrelationContext.get_correlation_id()
    CorrelationContext.set_correlation_id(correlation_id)
    
    try:
        yield correlation_id
    finally:
        if old_id:
            CorrelationContext.set_correlation_id(old_id)
        else:
            CorrelationContext.clear_correlation_id()


def get_structured_logger(component: str):
    """Get a logger that automatically includes component and correlation ID"""
    
    def log_with_context(level: str, message: str, **kwargs):
        """Log with automatic context injection"""
        
        # Add component
        kwargs["component"] = component
        
        # Add correlation ID if available
        correlation_id = CorrelationContext.get_correlation_id()
        if correlation_id:
            kwargs["correlation_id"] = correlation_id
        
        # Get the appropriate logger method
        log_method = getattr(logger, level.lower())
        log_method(logger.bind(**kwargs), message)
    
    class StructuredLogger:
        def debug(self, message: str, **kwargs):
            log_with_context("DEBUG", message, **kwargs)
        
        def info(self, message: str, **kwargs):
            log_with_context("INFO", message, **kwargs)
        
        def warning(self, message: str, **kwargs):
            log_with_context("WARNING", message, **kwargs)
        
        def error(self, message: str, **kwargs):
            log_with_context("ERROR", message, **kwargs)
        
        def critical(self, message: str, **kwargs):
            log_with_context("CRITICAL", message, **kwargs)
    
    return StructuredLogger()


def configure_json_logging(log_file: str = "logs/rtai_structured.jsonl"):
    """Configure JSON structured logging"""
    
    # Remove default handler
    logger.remove()
    
    # Add console handler with JSON format for development
    logger.add(
        lambda msg: print(msg, end=""),
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}\n",
        level="INFO"
    )
    
    # Add JSON file handler for structured logs
    logger.add(
        log_file,
        format=lambda record: JSONFormatter().format(record) + "\n",
        level="DEBUG",
        rotation="100 MB",
        retention="30 days",
        compression="gzip"
    )


def log_trade_processed(symbol: str, price: float, size: float, trade_id: str, **extra):
    """Log a processed trade with structured data"""
    
    structured_logger = get_structured_logger("trade_processor")
    structured_logger.info(
        f"📈 Trade processed: {symbol} @ {price}",
        trade_id=trade_id,
        symbol=symbol,
        price=price,
        size=size,
        event_type="trade_processed",
        **extra
    )


def log_indicator_calculation(indicator_name: str, symbol: str, value: float, 
                            duration_ms: float, **extra):
    """Log an indicator calculation with performance data"""
    
    structured_logger = get_structured_logger("indicators")
    structured_logger.info(
        f"🔢 {indicator_name} calculated for {symbol}: {value:.6f}",
        indicator_name=indicator_name,
        symbol=symbol,
        value=value,
        duration_ms=duration_ms,
        operation="indicator_calculation",
        event_type="indicator_calculated",
        **extra
    )


def log_feed_event(event_type: str, message: str, **extra):
    """Log a feed-related event"""
    
    structured_logger = get_structured_logger("feed")
    structured_logger.info(
        f"📡 {message}",
        event_type=event_type,
        **extra
    )


def log_error_with_context(component: str, error: Exception, context: Dict[str, Any]):
    """Log an error with full context"""
    
    structured_logger = get_structured_logger(component)
    structured_logger.error(
        f"❌ {str(error)}",
        error_type=type(error).__name__,
        event_type="error",
        **context
    )


if __name__ == "__main__":
    # Test structured logging
    configure_json_logging("test_structured.jsonl")
    
    with correlation_context() as corr_id:
        print(f"Using correlation ID: {corr_id}")
        
        log_trade_processed("BTCUSDT", 45123.45, 0.001, "test_trade_123")
        log_indicator_calculation("OFI", "BTCUSDT", 1.234567, 15.2)
        log_feed_event("connection_established", "Connected to Binance WebSocket")
        
        try:
            raise ValueError("Test error")
        except Exception as e:
            log_error_with_context("test", e, {"test": True, "attempt": 1})
    
    print("✅ Structured logging test completed")
