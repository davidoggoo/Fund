"""
RTAI Record-Replay-Score (RRS) Engine - Replayer Module
========================================================

High-performance replay system with speed control and drift compensation.
Replays recorded events maintaining exact timing and sequence.
"""

import os
import gzip
import json
import time
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Iterator, Union
from dataclasses import dataclass
from loguru import logger


@dataclass 
class ReplayConfig:
    """Configuration for replay system"""
    speed_multiplier: float = 1.0  # 1.0 = real-time, 10.0 = 10x faster
    max_speed: float = 10000.0  # Maximum replay speed
    start_from_seq: int = 0  # Start from specific sequence number
    end_at_seq: Optional[int] = None  # Stop at specific sequence number
    event_filter: Optional[List[str]] = None  # Only replay specific event types
    time_drift_compensation: bool = True  # Compensate for CPU time drift


class EventReplayer:
    """
    Professional event replay system for RRS engine.
    Maintains exact timing relationships while allowing speed control.
    """
    
    def __init__(self, config: ReplayConfig = None):
        self.config = config or ReplayConfig()
        self.is_replaying = False
        self._start_time = None
        self._base_timestamp = None
        self._events_replayed = 0
        self._current_file = None
        self._event_handlers: Dict[str, List[Callable]] = {}
    
    def register_handler(self, event_type: str, handler: Callable[[Dict[str, Any]], None]):
        """Register event handler for specific event type"""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
        logger.debug(f"Registered handler for {event_type}")
    
    def unregister_handler(self, event_type: str, handler: Callable):
        """Unregister specific event handler"""
        if event_type in self._event_handlers:
            try:
                self._event_handlers[event_type].remove(handler)
            except ValueError:
                pass
    
    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit event to registered handlers"""
        if event_type in self._event_handlers:
            for handler in self._event_handlers[event_type]:
                try:
                    handler(data)
                except Exception as e:
                    logger.error(f"Error in {event_type} handler: {e}")
    
    def _load_events(self, file_path: Union[str, Path]) -> Iterator[Dict[str, Any]]:
        """Load events from recording file with support for compression"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Recording file not found: {file_path}")
        
        # Determine if file is compressed by extension
        is_compressed = file_path.suffix == '.gz'
        
        try:
            if is_compressed:
                fh = gzip.open(file_path, 'rt', encoding='utf-8')
            else:
                fh = open(file_path, 'r', encoding='utf-8')
            
            with fh:
                line_number = 0
                header_processed = False
                
                for line in fh:
                    line_number += 1
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        event = json.loads(line)
                        
                        # First line is header
                        if not header_processed:
                            header_processed = True
                            if 'version' in event:  # This is a header
                                logger.info(f"📼 Replay file: {file_path}")
                                logger.info(f"📅 Created: {event.get('created', 'unknown')}")
                                logger.info(f"🎯 Symbol: {event.get('symbol', 'unknown')}")
                                continue
                        
                        # Apply sequence filtering
                        seq = event.get('seq', 0)
                        if seq < self.config.start_from_seq:
                            continue
                        if self.config.end_at_seq is not None and seq > self.config.end_at_seq:
                            break
                        
                        # Apply event type filtering
                        event_type = event.get('type')
                        if self.config.event_filter and event_type not in self.config.event_filter:
                            continue
                        
                        yield event
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON at line {line_number}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error loading events from {file_path}: {e}")
            raise
    
    def _calculate_sleep_time(self, event_timestamp: float) -> float:
        """Calculate sleep time with drift compensation and speed control"""
        if self._base_timestamp is None:
            self._base_timestamp = event_timestamp
            self._start_time = time.time()
            return 0.0
        
        # Calculate expected elapsed time in recording
        recording_elapsed = event_timestamp - self._base_timestamp
        
        # Calculate actual elapsed time since replay started
        actual_elapsed = time.time() - self._start_time
        
        # Calculate target elapsed time with speed multiplier
        target_elapsed = recording_elapsed / self.config.speed_multiplier
        
        # Calculate sleep time (negative means we're behind)
        sleep_time = target_elapsed - actual_elapsed
        
        # Apply drift compensation and speed limits
        if self.config.time_drift_compensation:
            # Don't sleep for negative times (catch up naturally)
            sleep_time = max(0.0, sleep_time)
            
            # Limit maximum sleep to prevent excessive delays
            max_sleep = 1.0 / max(self.config.speed_multiplier, 1.0)
            sleep_time = min(sleep_time, max_sleep)
        
        return sleep_time
    
    async def _sleep(self, duration: float):
        """Optimized sleep for high-speed replay"""
        if duration <= 0:
            return
        
        # For very high speeds, use shorter sleep intervals to maintain CPU efficiency
        if self.config.speed_multiplier > 100:
            # Use shorter chunks for very fast replay
            chunk_size = min(duration, 0.001)  # 1ms chunks
            while duration > 0:
                await asyncio.sleep(min(chunk_size, duration))
                duration -= chunk_size
        else:
            await asyncio.sleep(duration)
    
    async def replay_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Replay events from recording file.
        Returns statistics about the replay session.
        """
        if self.is_replaying:
            raise RuntimeError("Replay already in progress")
        
        file_path = Path(file_path)
        self._current_file = file_path
        self.is_replaying = True
        self._events_replayed = 0
        self._base_timestamp = None
        self._start_time = None
        
        start_time = time.time()
        
        try:
            logger.info(f"🎬 Starting replay: {file_path}")
            logger.info(f"⚡ Speed: {self.config.speed_multiplier}x")
            
            if self.config.event_filter:
                logger.info(f"🔍 Filtering events: {self.config.event_filter}")
            
            events_by_type = {}
            
            for event in self._load_events(file_path):
                if not self.is_replaying:  # Allow early termination
                    break
                
                event_type = event.get('type')
                event_timestamp = event.get('timestamp', time.time())
                event_data = event.get('data', {})
                
                # Calculate and apply timing
                sleep_time = self._calculate_sleep_time(event_timestamp)
                if sleep_time > 0:
                    await self._sleep(sleep_time)
                
                # Emit event to handlers
                self._emit_event(event_type, event_data)
                
                # Update statistics
                self._events_replayed += 1
                events_by_type[event_type] = events_by_type.get(event_type, 0) + 1
                
                # Log progress for long replays
                if self._events_replayed % 10000 == 0:
                    elapsed = time.time() - start_time
                    rate = self._events_replayed / elapsed
                    logger.info(f"📊 Replayed {self._events_replayed} events ({rate:.0f} events/sec)")
            
            # Final statistics
            total_time = time.time() - start_time
            
            stats = {
                "file_path": str(file_path),
                "events_replayed": self._events_replayed,
                "replay_duration_sec": total_time,
                "events_per_second": self._events_replayed / total_time if total_time > 0 else 0,
                "speed_multiplier": self.config.speed_multiplier,
                "events_by_type": events_by_type
            }
            
            logger.success(f"✅ Replay completed: {self._events_replayed} events in {total_time:.2f}s")
            logger.info(f"📈 Average rate: {stats['events_per_second']:.0f} events/sec")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error during replay: {e}")
            raise
        finally:
            self.is_replaying = False
            self._current_file = None
    
    def stop_replay(self):
        """Stop ongoing replay"""
        if self.is_replaying:
            self.is_replaying = False
            logger.info("⏹️ Replay stopped by user")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current replay statistics"""
        return {
            "is_replaying": self.is_replaying,
            "current_file": str(self._current_file) if self._current_file else None,
            "events_replayed": self._events_replayed,
            "speed_multiplier": self.config.speed_multiplier
        }
    
    async def replay_directory(self, directory_path: Union[str, Path], pattern: str = "*.rec*") -> List[Dict[str, Any]]:
        """Replay all matching files in directory in chronological order"""
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        # Find all matching recording files
        files = list(directory_path.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No recording files found matching {pattern}")
        
        # Sort by modification time (chronological order)
        files.sort(key=lambda f: f.stat().st_mtime)
        
        results = []
        for file_path in files:
            logger.info(f"📁 Replaying file {len(results) + 1}/{len(files)}: {file_path.name}")
            stats = await self.replay_file(file_path)
            results.append(stats)
            
            if not self.is_replaying:  # Check if stopped
                break
        
        return results


# Convenience functions for common replay scenarios

async def replay_file(file_path: Union[str, Path], 
                     speed: float = 1.0,
                     event_handlers: Optional[Dict[str, Callable]] = None) -> Dict[str, Any]:
    """
    Convenience function to replay a single file with optional handlers.
    
    Args:
        file_path: Path to recording file
        speed: Replay speed multiplier
        event_handlers: Dict of {event_type: handler_function}
    
    Returns:
        Replay statistics
    """
    config = ReplayConfig(speed_multiplier=speed)
    replayer = EventReplayer(config)
    
    # Register handlers if provided
    if event_handlers:
        for event_type, handler in event_handlers.items():
            replayer.register_handler(event_type, handler)
    
    return await replayer.replay_file(file_path)


async def replay_latest(records_dir: Union[str, Path] = "records",
                       symbol: str = "BTCUSDT",
                       speed: float = 1.0,
                       event_handlers: Optional[Dict[str, Callable]] = None) -> Dict[str, Any]:
    """
    Convenience function to replay the latest recording for a symbol.
    
    Args:
        records_dir: Directory containing recordings
        symbol: Trading symbol to find recordings for
        speed: Replay speed multiplier
        event_handlers: Dict of {event_type: handler_function}
    
    Returns:
        Replay statistics
    """
    records_dir = Path(records_dir)
    
    # Find latest recording for symbol
    pattern = f"{symbol}_*.rec*"
    files = list(records_dir.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No recordings found for {symbol}")
    
    # Get most recent file
    latest_file = max(files, key=lambda f: f.stat().st_mtime)
    
    logger.info(f"🎯 Replaying latest recording: {latest_file.name}")
    
    return await replay_file(latest_file, speed, event_handlers)


# Event handler helper classes

class TradeHandler:
    """Handler for trade events during replay"""
    
    def __init__(self, callback: Optional[Callable] = None):
        self.callback = callback
        self.trade_count = 0
        self.total_volume = 0.0
    
    def __call__(self, trade_data: Dict[str, Any]):
        self.trade_count += 1
        self.total_volume += trade_data.get('quantity', 0)
        
        if self.callback:
            self.callback(trade_data)
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "trade_count": self.trade_count,
            "total_volume": self.total_volume
        }


class IndicatorHandler:
    """Handler for indicator events during replay"""
    
    def __init__(self, callback: Optional[Callable] = None):
        self.callback = callback
        self.indicators = {}
        self.update_count = 0
    
    def __call__(self, indicator_data: Dict[str, Any]):
        name = indicator_data.get('name')
        value = indicator_data.get('value')
        
        if name:
            self.indicators[name] = value
            self.update_count += 1
        
        if self.callback:
            self.callback(indicator_data)
    
    def get_latest_values(self) -> Dict[str, Any]:
        return self.indicators.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            "update_count": self.update_count,
            "indicators_tracked": len(self.indicators)
        }
