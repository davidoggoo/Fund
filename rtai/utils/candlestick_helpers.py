"""
Professional candlestick and OHLC helper functions with pandas integration.
Enhanced for true 1-minute candle generation with proper aggregation.
"""
from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from .candlestick_engine import CandlestickEngine, OHLCV, Trade


def build_ohlc_from_ticks(tick_data: List[Dict[str, Any]], 
                         interval_seconds: int = 60) -> List[Tuple]:
    """Build OHLC candlestick data from tick data using professional engine
    
    Args:
        tick_data: List of tick dictionaries with 'timestamp' and 'price'
        interval_seconds: Candlestick interval in seconds (default: 60 for 1-minute)
    
    Returns:
        List of (timestamp, open, high, low, close, volume) tuples
    """
    if not tick_data:
        return []
    
    # Use professional candlestick engine
    engine = CandlestickEngine(interval_seconds=interval_seconds)
    
    # Process all ticks
    for tick in tick_data:
        timestamp = tick['timestamp']
        price = tick['price']
        volume = tick.get('volume', 1.0)
        side = tick.get('side', 'unknown')
        
        # Add trade to engine
        engine.add_trade(timestamp, price, volume, side)
    
    # Force finalize any remaining candle
    engine.force_finalize_current()
    
    # Convert to legacy tuple format
    result = []
    for candle in engine.get_recent_candles():
        result.append((
            candle.timestamp - interval_seconds,  # Use candle start time
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume
        ))
    
    return result


def ohlc_to_dataframe(ohlc_data: List[Tuple]) -> pd.DataFrame:
    """Convert OHLC tuples to pandas DataFrame with enhanced analytics
    
    Args:
        ohlc_data: List of (timestamp, open, high, low, close, volume) tuples
    
    Returns:
        DataFrame with datetime index and OHLC columns plus analytics
    """
    if not ohlc_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(ohlc_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Enhanced analytics for 1-minute candles
    df['body'] = abs(df['close'] - df['open'])
    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    df['range'] = df['high'] - df['low']
    df['is_bullish'] = df['close'] > df['open']
    df['is_doji'] = df['body'] / df['range'] < 0.1  # Body < 10% of range
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    
    df.set_index('datetime', inplace=True)
    return df


def calculate_candle_metrics(ohlc_data: List[Tuple]) -> Dict[str, Any]:
    """Calculate enhanced candlestick pattern metrics for 1-minute analysis
    
    Args:
        ohlc_data: List of (timestamp, open, high, low, close, volume) tuples
    
    Returns:
        Dictionary with comprehensive pattern statistics
    """
    if not ohlc_data:
        return {}
    
    df = ohlc_to_dataframe(ohlc_data)
    if df.empty:
        return {}
    
    bullish_count = df['is_bullish'].sum()
    bearish_count = (~df['is_bullish']).sum()
    doji_count = df['is_doji'].sum()
    total_volume = df['volume'].sum()
    
    # Enhanced metrics for 1-minute edge
    avg_body_ratio = (df['body'] / df['range']).mean()
    volatility = df['range'].std()
    volume_weighted_price = (df['typical_price'] * df['volume']).sum() / total_volume if total_volume > 0 else 0
    
    price_range = []
    
    for candle in ohlc_data:
        timestamp, open_price, high, low, close, volume = candle
        
        total_volume += volume
        price_range.append(high - low)
        
        # Classify candle
        body_size = abs(close - open_price)
        candle_range = high - low
        
        if candle_range == 0:
            continue
            
        body_ratio = body_size / candle_range
        
        if body_ratio < 0.1:  # Small body relative to range
            doji_count += 1
        elif close > open_price:
            bullish_count += 1
        else:
            bearish_count += 1
    
    return {
        'total_candles': len(ohlc_data),
        'bullish_count': bullish_count,
        'bearish_count': bearish_count,
        'doji_count': doji_count,
        'total_volume': total_volume,
        'avg_range': sum(price_range) / len(price_range) if price_range else 0,
        'max_range': max(price_range) if price_range else 0,
        'min_range': min(price_range) if price_range else 0
    }


# Legacy aliases for backward compatibility
def update_candlestick_data(*args, **kwargs):
    """Legacy alias - use build_ohlc_from_ticks instead"""
    return build_ohlc_from_ticks(*args, **kwargs)


def get_candlestick_metrics(*args, **kwargs):
    """Legacy alias - use calculate_candle_metrics instead"""
    return calculate_candle_metrics(*args, **kwargs)
