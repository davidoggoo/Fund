"""
Unified state management system for convergence of snapshot and live pipelines.
Enhanced with candlestick engine integration and performance optimization.
"""
import abc
import time
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Protocol, List
from loguru import logger
from .utils.candlestick_engine import CandlestickEngine, OHLCV


class StateStore(Protocol):
    """Enhanced interface for unified state persistence across snapshot and live trading"""
    
    def save_state(self, component: str, state: Dict[str, Any]) -> bool:
        """Save component state"""
        ...
    
    def load_state(self, component: str) -> Optional[Dict[str, Any]]:
        """Load component state"""
        ...
    
    def list_components(self) -> list[str]:
        """List all components with saved state"""
        ...
    
    def clear_state(self, component: str) -> bool:
        """Clear component state"""
        ...
    
    def save_candles(self, symbol: str, candles: List[OHLCV]) -> bool:
        """Save candlestick data"""
        ...
    
    def load_candles(self, symbol: str, limit: int = 1000) -> List[OHLCV]:
        """Load candlestick data"""
        ...


class InMemoryStore:
    """Enhanced in-memory state store for live trading with candlestick support"""
    
    def __init__(self):
        self._state: Dict[str, Dict[str, Any]] = {}
        self._last_update: Dict[str, float] = {}
        self._candles: Dict[str, List[OHLCV]] = {}
        self._candle_engines: Dict[str, CandlestickEngine] = {}
    
    def save_state(self, component: str, state: Dict[str, Any]) -> bool:
        """Save component state in memory"""
        try:
            self._state[component] = state.copy()
            self._last_update[component] = time.time()
            logger.debug(f"💾 InMemory: Saved {component} state")
            return True
        except Exception as e:
            logger.error(f"❌ InMemory: Failed to save {component}: {e}")
            return False
    
    def load_state(self, component: str) -> Optional[Dict[str, Any]]:
        """Load component state from memory"""
        state = self._state.get(component)
        if state:
            logger.debug(f"📂 InMemory: Loaded {component} state")
            return state.copy()
        return None
    
    def list_components(self) -> list[str]:
        """List all components with saved state"""
        return list(self._state.keys())
    
    def clear_state(self, component: str) -> bool:
        """Clear component state"""
        if component in self._state:
            del self._state[component]
            self._last_update.pop(component, None)
            logger.debug(f"🗑️ InMemory: Cleared {component} state")
            return True
        return False
    
    def save_candles(self, symbol: str, candles: List[OHLCV]) -> bool:
        """Save candlestick data in memory"""
        try:
            self._candles[symbol] = candles.copy()
            logger.debug(f"💾 InMemory: Saved {len(candles)} candles for {symbol}")
            return True
        except Exception as e:
            logger.error(f"❌ InMemory: Failed to save candles for {symbol}: {e}")
            return False
    
    def load_candles(self, symbol: str, limit: int = 1000) -> List[OHLCV]:
        """Load candlestick data from memory"""
        candles = self._candles.get(symbol, [])
        return candles[-limit:] if candles else []
    
    def get_candle_engine(self, symbol: str) -> CandlestickEngine:
        """Get or create candlestick engine for symbol"""
        if symbol not in self._candle_engines:
            self._candle_engines[symbol] = CandlestickEngine()
        return self._candle_engines[symbol]


class ParquetStore:
    """Enhanced Parquet-based state store for snapshot pipeline with candlestick support"""
    
    def __init__(self, base_path: str = "state"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.base_path / "components").mkdir(exist_ok=True)
        (self.base_path / "candles").mkdir(exist_ok=True)
        
        # Metadata file for component tracking
        self.metadata_file = self.base_path / "state_metadata.json"
        self._load_metadata()
    
    def _load_metadata(self):
        """Load metadata about stored components"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            else:
                self.metadata = {}
        except Exception as e:
            logger.warning(f"⚠️ ParquetStore: Failed to load metadata: {e}")
            self.metadata = {}
    
    def _save_metadata(self):
        """Save metadata about stored components"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.error(f"❌ ParquetStore: Failed to save metadata: {e}")
    
    def save_state(self, component: str, state: Dict[str, Any]) -> bool:
        """Save component state to parquet"""
        try:
            # Convert state to DataFrame-compatible format
            state_file = self.base_path / f"{component}_state.parquet"
            
            # Convert complex state to simple records
            records = []
            timestamp = time.time()
            
            for key, value in state.items():
                if isinstance(value, (list, tuple)):
                    # Store sequences as separate records
                    for i, item in enumerate(value):
                        records.append({
                            'timestamp': timestamp,
                            'key': f"{key}_{i}",
                            'value': str(item),
                            'type': type(item).__name__
                        })
                else:
                    records.append({
                        'timestamp': timestamp,
                        'key': key,
                        'value': str(value),
                        'type': type(value).__name__
                    })
            
            if records:
                df = pd.DataFrame(records)
                df.to_parquet(state_file, index=False)
                
                # Update metadata
                self.metadata[component] = {
                    'last_update': timestamp,
                    'record_count': len(records),
                    'file': str(state_file)
                }
                self._save_metadata()
                
                logger.debug(f"💾 ParquetStore: Saved {component} state ({len(records)} records)")
                return True
            
        except Exception as e:
            logger.error(f"❌ ParquetStore: Failed to save {component}: {e}")
        
        return False
    
    def load_state(self, component: str) -> Optional[Dict[str, Any]]:
        """Load component state from parquet"""
        try:
            state_file = self.base_path / f"{component}_state.parquet"
            
            if not state_file.exists():
                return None
            
            df = pd.read_parquet(state_file)
            if df.empty:
                return None
            
            # Reconstruct state from records
            state = {}
            
            # Group by key prefix to reconstruct sequences
            key_groups = {}
            for _, row in df.iterrows():
                key = row['key']
                value = row['value']
                value_type = row['type']
                
                # Handle sequence reconstruction
                if '_' in key and key.split('_')[-1].isdigit():
                    base_key = '_'.join(key.split('_')[:-1])
                    index = int(key.split('_')[-1])
                    
                    if base_key not in key_groups:
                        key_groups[base_key] = {}
                    key_groups[base_key][index] = self._convert_value(value, value_type)
                else:
                    state[key] = self._convert_value(value, value_type)
            
            # Reconstruct sequences
            for base_key, indexed_values in key_groups.items():
                max_index = max(indexed_values.keys())
                sequence = [indexed_values.get(i) for i in range(max_index + 1)]
                state[base_key] = sequence
            
            logger.debug(f"📂 ParquetStore: Loaded {component} state")
            return state
            
        except Exception as e:
            logger.error(f"❌ ParquetStore: Failed to load {component}: {e}")
            return None
    
    def save_candles(self, symbol: str, candles: List[OHLCV]) -> bool:
        """Save candlestick data to parquet"""
        try:
            if not candles:
                return False
            
            candle_file = self.base_path / "candles" / f"{symbol}_candles.parquet"
            
            # Convert to DataFrame
            data = []
            for candle in candles:
                data.append({
                    'timestamp': candle.timestamp,
                    'open': candle.open,
                    'high': candle.high,
                    'low': candle.low,
                    'close': candle.close,
                    'volume': candle.volume,
                    'trade_count': candle.trade_count,
                    'buy_volume': candle.buy_volume,
                    'sell_volume': candle.sell_volume,
                    'vwap': candle.vwap
                })
            
            df = pd.DataFrame(data)
            df.to_parquet(candle_file, index=False)
            
            logger.debug(f"💾 ParquetStore: Saved {len(candles)} candles for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"❌ ParquetStore: Failed to save candles for {symbol}: {e}")
            return False
    
    def load_candles(self, symbol: str, limit: int = 1000) -> List[OHLCV]:
        """Load candlestick data from parquet"""
        try:
            candle_file = self.base_path / "candles" / f"{symbol}_candles.parquet"
            
            if not candle_file.exists():
                return []
            
            df = pd.read_parquet(candle_file)
            df = df.tail(limit)  # Get most recent candles
            
            # Convert to OHLCV objects
            candles = []
            for _, row in df.iterrows():
                candle = OHLCV(
                    timestamp=row['timestamp'],
                    open=row['open'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume'],
                    trade_count=row.get('trade_count', 0),
                    buy_volume=row.get('buy_volume', 0.0),
                    sell_volume=row.get('sell_volume', 0.0),
                    vwap=row.get('vwap', row['close'])
                )
                candles.append(candle)
            
            logger.debug(f"📂 ParquetStore: Loaded {len(candles)} candles for {symbol}")
            return candles
            
        except Exception as e:
            logger.error(f"❌ ParquetStore: Failed to load candles for {symbol}: {e}")
            return []
    
    def _convert_value(self, value_str: str, value_type: str):
        """Convert string back to original type"""
        try:
            if value_type == 'int':
                return int(value_str)
            elif value_type == 'float':
                return float(value_str)
            elif value_type == 'bool':
                return value_str.lower() == 'true'
            elif value_type == 'NoneType':
                return None
            else:
                return value_str  # Keep as string
        except (ValueError, TypeError):
            return value_str  # Fallback to string
    
    def list_components(self) -> list[str]:
        """List all components with saved state"""
        return list(self.metadata.keys())
    
    def clear_state(self, component: str) -> bool:
        """Clear component state"""
        try:
            state_file = self.base_path / f"{component}_state.parquet"
            if state_file.exists():
                state_file.unlink()
            
            if component in self.metadata:
                del self.metadata[component]
                self._save_metadata()
            
            logger.debug(f"🗑️ ParquetStore: Cleared {component} state")
            return True
            
        except Exception as e:
            logger.error(f"❌ ParquetStore: Failed to clear {component}: {e}")
            return False


class StateAdapter:
    """Adapter to make indicators work with different state stores"""
    
    def __init__(self, store: StateStore):
        self.store = store
    
    def save_indicator_state(self, indicator, component_name: str) -> bool:
        """Save indicator state using its existing save_state method"""
        try:
            if hasattr(indicator, 'save_state'):
                state = indicator.save_state()
                return self.store.save_state(component_name, state)
            else:
                logger.warning(f"⚠️ {component_name}: No save_state method")
                return False
        except Exception as e:
            logger.error(f"❌ Failed to save {component_name}: {e}")
            return False
    
    def load_indicator_state(self, indicator, component_name: str) -> bool:
        """Load indicator state using its existing load_state method"""
        try:
            if hasattr(indicator, 'load_state'):
                state = self.store.load_state(component_name)
                if state:
                    indicator.load_state(state)
                    return True
            else:
                logger.warning(f"⚠️ {component_name}: No load_state method")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to load {component_name}: {e}")
            return False


# Factory function for choosing store implementation
def create_state_store(store_type: str = "memory", **kwargs) -> StateStore:
    """Factory function to create appropriate state store"""
    if store_type.lower() == "memory":
        return InMemoryStore()
    elif store_type.lower() == "parquet":
        return ParquetStore(**kwargs)
    else:
        raise ValueError(f"Unknown store type: {store_type}")
