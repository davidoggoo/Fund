"""
RTAI Record-Replay-Score (RRS) Engine - Recorder Module
========================================================

Professional recording system for all trading events with compression and deterministic ordering.
Drop-in integration with existing LiveTrader pipeline.
"""

import os
import gzip
import json
import time
import atexit
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass
from loguru import logger


@dataclass
class RecordConfig:
    """Configuration for recording system"""
    output_dir: str = "records"
    compression: bool = True
    buffer_size: int = 8192
    auto_flush_interval: float = 10.0  # seconds
    max_file_size_mb: int = 500  # Auto-rotate after this size


class EventRecorder:
    """
    High-performance event recorder for RRS system.
    Records all trading events with deterministic ordering and compression.
    """
    
    def __init__(self, symbol: str, config: RecordConfig = None):
        self.symbol = symbol.upper()
        self.config = config or RecordConfig()
        self.is_recording = False
        self._fh = None
        self._seq = 0  # Sequence number for deterministic ordering
        self._last_flush = time.time()
        self._bytes_written = 0
        self._current_file_path = None
        
        # Ensure output directory exists
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Register cleanup on exit
        atexit.register(self.close)
        
    def start_recording(self, session_id: Optional[str] = None) -> str:
        """Start recording session with unique identifier"""
        if self.is_recording:
            logger.warning("Recording already in progress")
            return self._current_file_path
            
        if session_id is None:
            session_id = str(int(time.time()))
            
        # Create unique filename with compression
        ext = ".rec.gz" if self.config.compression else ".rec"
        filename = f"{self.symbol}_{session_id}{ext}"
        self._current_file_path = self.output_dir / filename
        
        try:
            self._open_file()
            self.is_recording = True
            self._seq = 0
            self._bytes_written = 0
            
            # Write header with metadata
            header = {
                "created": datetime.now(tz=timezone.utc).isoformat(),
                "symbol": self.symbol,
                "version": "1.0",
                "session_id": session_id,
                "compression": self.config.compression
            }
            self._write_line(json.dumps(header))
            
            logger.info(f"🎬 Recording started: {self._current_file_path}")
            return str(self._current_file_path)
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            self.is_recording = False
            raise
    
    def _open_file(self):
        """Open file handle with optional compression"""
        if self.config.compression:
            self._fh = gzip.open(self._current_file_path, 'wt', encoding='utf-8')
        else:
            self._fh = open(self._current_file_path, 'w', encoding='utf-8')
    
    def _write_line(self, line: str):
        """Write line to file with buffering and size tracking"""
        if not self._fh:
            return
        
        line_bytes = line + "\n"
        self._fh.write(line_bytes)
        self._bytes_written += len(line_bytes.encode('utf-8'))
        
        # Auto-flush based on time interval
        now = time.time()
        if now - self._last_flush >= self.config.auto_flush_interval:
            self._fh.flush()
            os.fsync(self._fh.fileno())  # Force to disk
            self._last_flush = now
    
    def record_event(self, event_type: str, data: Dict[str, Any], timestamp: Optional[float] = None):
        """Record an event with deterministic ordering"""
        if not self.is_recording:
            return
            
        if timestamp is None:
            timestamp = time.time()
        
        # Create event record with sequence number for deterministic ordering
        event = {
            "seq": self._seq,
            "timestamp": timestamp,
            "type": event_type,
            "data": data
        }
        
        try:
            self._write_line(json.dumps(event, separators=(',', ':')))  # Compact JSON
            self._seq += 1
            
            # Check for file rotation based on size
            if self._bytes_written > self.config.max_file_size_mb * 1024 * 1024:
                self._rotate_file()
                
        except Exception as e:
            logger.error(f"Failed to record event {event_type}: {e}")
    
    def _rotate_file(self):
        """Rotate to new file when size limit exceeded"""
        if not self.is_recording:
            return
            
        old_path = self._current_file_path
        self.stop_recording()
        
        # Start new file with incremented session ID
        session_id = str(int(time.time()))
        new_path = self.start_recording(session_id)
        
        logger.info(f"📁 File rotated: {old_path} → {new_path}")
    
    def record_trade(self, trade_data: Dict[str, Any]):
        """Record trade event with standardized format"""
        standardized_trade = {
            "symbol": trade_data.get("symbol", self.symbol),
            "price": float(trade_data["price"]),
            "quantity": float(trade_data["quantity"]),
            "side": trade_data["side"],
            "timestamp": trade_data.get("timestamp", time.time()),
            "trade_id": trade_data.get("trade_id"),
            "is_buyer_maker": trade_data.get("is_buyer_maker", False)
        }
        self.record_event("trade", standardized_trade)
    
    def record_depth_snapshot(self, depth_data: Dict[str, Any]):
        """Record orderbook depth snapshot"""
        standardized_depth = {
            "symbol": depth_data.get("symbol", self.symbol),
            "timestamp": depth_data.get("timestamp", time.time()),
            "bids": depth_data["bids"],  # [[price, quantity], ...]
            "asks": depth_data["asks"],  # [[price, quantity], ...]
            "event_time": depth_data.get("event_time")
        }
        self.record_event("depth", standardized_depth)
    
    def record_liquidation(self, liquidation_data: Dict[str, Any]):
        """Record liquidation event"""
        standardized_liq = {
            "symbol": liquidation_data.get("symbol", self.symbol),
            "timestamp": liquidation_data.get("timestamp", time.time()),
            "side": liquidation_data["side"],
            "price": float(liquidation_data["price"]),
            "quantity": float(liquidation_data["quantity"]),
            "time": liquidation_data.get("time")
        }
        self.record_event("liquidation", standardized_liq)
    
    def record_funding_update(self, funding_data: Dict[str, Any]):
        """Record funding rate update"""
        standardized_funding = {
            "symbol": funding_data.get("symbol", self.symbol),
            "timestamp": funding_data.get("timestamp", time.time()),
            "funding_rate": float(funding_data["funding_rate"]),
            "mark_price": float(funding_data.get("mark_price", 0)),
            "next_funding_time": funding_data.get("next_funding_time")
        }
        self.record_event("funding", standardized_funding)
    
    def record_indicator_update(self, indicator_name: str, value: Union[float, Dict[str, Any]], timestamp: Optional[float] = None):
        """Record indicator value update"""
        indicator_data = {
            "name": indicator_name,
            "value": value,
            "timestamp": timestamp or time.time()
        }
        self.record_event("indicator", indicator_data)
    
    def record_signal(self, signal_data: Dict[str, Any]):
        """Record trading signal"""
        standardized_signal = {
            "timestamp": signal_data.get("timestamp", time.time()),
            "signal_type": signal_data["signal_type"],
            "indicator": signal_data["indicator"],
            "value": signal_data["value"],
            "reason": signal_data["reason"],
            "confidence": signal_data.get("confidence", 1.0)
        }
        self.record_event("signal", standardized_signal)
    
    def record_basic_oscillator(self, osc_name: str, osc_data: Dict[str, Any]):
        """Record basic oscillator update (for RSI-style indicators)"""
        oscillator_data = {
            "name": osc_name,
            "raw_value": osc_data.get("raw"),
            "oscillator": osc_data.get("osc"),
            "z_score": osc_data.get("z_score"),
            "is_extreme": osc_data.get("is_extreme", False),
            "timestamp": osc_data.get("timestamp", time.time())
        }
        self.record_event("basic_oscillator", oscillator_data)
    
    def stop_recording(self):
        """Stop recording and close file safely"""
        if not self.is_recording:
            return
        
        try:
            if self._fh:
                # Final flush and sync
                self._fh.flush()
                os.fsync(self._fh.fileno())
                self._fh.close()
                self._fh = None
            
            self.is_recording = False
            
            # Log recording statistics
            if self._current_file_path and os.path.exists(self._current_file_path):
                file_size_mb = os.path.getsize(self._current_file_path) / (1024 * 1024)
                logger.info(f"🎬 Recording stopped: {self._current_file_path}")
                logger.info(f"📊 Events recorded: {self._seq}, Size: {file_size_mb:.2f} MB")
            
        except Exception as e:
            logger.error(f"Error stopping recording: {e}")
    
    def close(self):
        """Alias for stop_recording for atexit compatibility"""
        self.stop_recording()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current recording statistics"""
        stats = {
            "is_recording": self.is_recording,
            "current_file": str(self._current_file_path) if self._current_file_path else None,
            "events_recorded": self._seq,
            "bytes_written": self._bytes_written,
            "session_duration": time.time() - self._last_flush if self.is_recording else 0
        }
        
        if self._current_file_path and self._current_file_path.exists():
            stats["file_size_mb"] = os.path.getsize(self._current_file_path) / (1024 * 1024)
        
        return stats
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures recording stops"""
        self.stop_recording()


# Global recorder instance for drop-in integration
_global_recorder: Optional[EventRecorder] = None


def get_recorder(symbol: str = "BTCUSDT", config: RecordConfig = None) -> EventRecorder:
    """Get or create global recorder instance"""
    global _global_recorder
    
    if _global_recorder is None or _global_recorder.symbol != symbol:
        _global_recorder = EventRecorder(symbol, config)
    
    return _global_recorder


def start_recording(symbol: str = "BTCUSDT", session_id: Optional[str] = None, config: RecordConfig = None) -> str:
    """Convenience function to start recording"""
    recorder = get_recorder(symbol, config)
    return recorder.start_recording(session_id)


def stop_recording():
    """Convenience function to stop recording"""
    global _global_recorder
    if _global_recorder:
        _global_recorder.stop_recording()


def record_event(event_type: str, data: Dict[str, Any], timestamp: Optional[float] = None):
    """Convenience function to record an event"""
    global _global_recorder
    if _global_recorder and _global_recorder.is_recording:
        _global_recorder.record_event(event_type, data, timestamp)


# Global convenience functions for surgical drop-in integration

def record_trade(price: float, volume: float, side: str, timestamp: Optional[float] = None):
    """Record a trade event"""
    data = {
        "price": price,
        "volume": volume,
        "side": side
    }
    record_event("trade", data, timestamp)


def record_depth_snapshot(bids: list, asks: list, timestamp: Optional[float] = None):
    """Record a depth snapshot event"""
    data = {
        "bids": bids,
        "asks": asks
    }
    record_event("depth_snapshot", data, timestamp)


def record_liquidation(symbol: str, size: float, price: float, side: str, timestamp: Optional[float] = None):
    """Record a liquidation event"""
    data = {
        "symbol": symbol,
        "size": size,
        "price": price,
        "side": side
    }
    record_event("liquidation", data, timestamp)


def record_funding_update(symbol: str, funding_rate: float, timestamp: Optional[float] = None):
    """Record a funding rate update"""
    data = {
        "symbol": symbol,
        "funding_rate": funding_rate
    }
    record_event("funding_update", data, timestamp)


def record_indicator_update(indicators: Dict[str, float], timestamp: Optional[float] = None):
    """Record indicator values update"""
    data = {
        "indicators": indicators
    }
    record_event("indicator_update", data, timestamp)


def record_basic_oscillator(oscillators: Dict[str, float], timestamp: Optional[float] = None):
    """Record basic oscillator values"""
    data = {
        "oscillators": oscillators
    }
    record_event("basic_oscillator", data, timestamp)


def record_signal_trigger(signal_name: str, signal_type: str, signal_data: Dict[str, Any], timestamp: Optional[float] = None):
    """Record a signal trigger event"""
    data = {
        "signal_name": signal_name,
        "signal_type": signal_type,
        "signal_data": signal_data
    }
    record_event("signal_trigger", data, timestamp)


def record_bar(open_price: float, high: float, low: float, close: float, volume: float, timestamp: Optional[float] = None):
    """Record OHLCV bar data"""
    data = {
        "o": open_price,
        "h": high,
        "l": low,
        "c": close,
        "v": volume
    }
    record_event("bar", data, timestamp)


def record_signal(side: str, price: float, timestamp: Optional[float] = None):
    """Record trading signal"""
    data = {
        "side": side,
        "price": price
    }
    record_event("signal", data, timestamp)


def record_equity(value: float, timestamp: Optional[float] = None):
    """Record equity/portfolio value"""
    data = {
        "value": value
    }
    record_event("equity", data, timestamp)