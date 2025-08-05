"""
Recording Data Converter for RTAI System
Converts between .rec.gz format and TradingVue.js compatible OHLCV data
"""
import gzip
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from loguru import logger


def rec_to_ohlcv(rec_file_path: str, timeframe: str = '1m') -> List[List[float]]:
    """
    Convert .rec.gz file to OHLCV format compatible with TradingVue.js
    
    Args:
        rec_file_path: Path to .rec.gz file
        timeframe: Timeframe for candles ('1m', '5m', '15m', '1h')
    
    Returns:
        List of [timestamp, open, high, low, close, volume] arrays
    """
    try:
        # Read compressed recording file
        with gzip.open(rec_file_path, 'rt', encoding='utf-8') as f:
            trades = []
            for line in f:
                try:
                    trade_data = json.loads(line.strip())
                    if trade_data.get('type') == 'trade':
                        trades.append(trade_data)
                except json.JSONDecodeError:
                    continue
        
        if not trades:
            logger.warning(f"No trade data found in {rec_file_path}")
            return []
        
        # Convert to DataFrame
        df = pd.DataFrame(trades)
        
        # Ensure required columns exist
        required_cols = ['timestamp', 'price', 'volume']
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                return []
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # Remove invalid data
        df = df.dropna(subset=['price', 'volume'])
        
        if df.empty:
            logger.warning("No valid trade data after cleaning")
            return []
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        # Resample to specified timeframe
        timeframe_map = {
            '1m': '1T',
            '5m': '5T', 
            '15m': '15T',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D'
        }
        
        resample_freq = timeframe_map.get(timeframe, '1T')
        
        # Create OHLCV data
        ohlcv = df['price'].resample(resample_freq).ohlc()
        volume = df['volume'].resample(resample_freq).sum()
        
        # Combine OHLC with volume
        ohlcv['volume'] = volume
        ohlcv = ohlcv.dropna()
        
        # Convert to TradingVue format: [timestamp_ms, open, high, low, close, volume]
        result = []
        for timestamp, row in ohlcv.iterrows():
            result.append([
                int(timestamp.timestamp() * 1000),  # Convert to milliseconds
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row['volume'])
            ])
        
        logger.info(f"Converted {len(trades)} trades to {len(result)} {timeframe} candles")
        return result
        
    except Exception as e:
        logger.error(f"Error converting {rec_file_path}: {e}")
        return []


def ohlcv_to_dataframe(ohlcv_data: List[List[float]]) -> pd.DataFrame:
    """
    Convert OHLCV data to pandas DataFrame for backtesting
    
    Args:
        ohlcv_data: List of [timestamp, open, high, low, close, volume] arrays
    
    Returns:
        DataFrame with OHLCV columns and datetime index
    """
    try:
        if not ohlcv_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        
        # Convert timestamp from milliseconds to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Ensure numeric types
        numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove invalid data
        df = df.dropna()
        
        logger.info(f"Created DataFrame with {len(df)} rows")
        return df
        
    except Exception as e:
        logger.error(f"Error creating DataFrame: {e}")
        return pd.DataFrame()


def get_latest_rec_files(recordings_dir: str = "recordings", limit: int = 10) -> List[str]:
    """
    Get list of latest .rec.gz files
    
    Args:
        recordings_dir: Directory containing recording files
        limit: Maximum number of files to return
    
    Returns:
        List of file paths sorted by modification time (newest first)
    """
    try:
        recordings_path = Path(recordings_dir)
        if not recordings_path.exists():
            logger.warning(f"Recordings directory not found: {recordings_dir}")
            return []
        
        # Find all .rec.gz files
        rec_files = list(recordings_path.glob("*.rec.gz"))
        
        if not rec_files:
            logger.info(f"No .rec.gz files found in {recordings_dir}")
            return []
        
        # Sort by modification time (newest first)
        rec_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Return paths as strings
        result = [str(f) for f in rec_files[:limit]]
        logger.info(f"Found {len(result)} recording files")
        return result
        
    except Exception as e:
        logger.error(f"Error getting rec files: {e}")
        return []


def merge_rec_files(rec_files: List[str], output_file: str) -> bool:
    """
    Merge multiple .rec.gz files into a single file
    
    Args:
        rec_files: List of .rec.gz file paths to merge
        output_file: Output file path
    
    Returns:
        True if successful, False otherwise
    """
    try:
        all_data = []
        
        for rec_file in rec_files:
            logger.info(f"Reading {rec_file}")
            with gzip.open(rec_file, 'rt', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        all_data.append(data)
                    except json.JSONDecodeError:
                        continue
        
        if not all_data:
            logger.warning("No data to merge")
            return False
        
        # Sort by timestamp
        all_data.sort(key=lambda x: x.get('timestamp', 0))
        
        # Write merged data
        with gzip.open(output_file, 'wt', encoding='utf-8') as f:
            for data in all_data:
                f.write(json.dumps(data) + '\n')
        
        logger.info(f"Merged {len(all_data)} records to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error merging files: {e}")
        return False


def extract_indicators_from_rec(rec_file_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract indicator data from .rec.gz file
    
    Args:
        rec_file_path: Path to .rec.gz file
    
    Returns:
        Dictionary with indicator names as keys and list of values as values
    """
    try:
        indicators = {}
        
        with gzip.open(rec_file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    
                    if data.get('type') == 'indicator':
                        indicator_name = data.get('name', 'unknown')
                        
                        if indicator_name not in indicators:
                            indicators[indicator_name] = []
                        
                        indicators[indicator_name].append({
                            'timestamp': data.get('timestamp'),
                            'value': data.get('value'),
                            'metadata': data.get('metadata', {})
                        })
                        
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"Extracted {len(indicators)} indicator types from {rec_file_path}")
        return indicators
        
    except Exception as e:
        logger.error(f"Error extracting indicators: {e}")
        return {}


# Utility functions for data validation
def validate_ohlcv_data(ohlcv_data: List[List[float]]) -> bool:
    """Validate OHLCV data structure"""
    if not ohlcv_data:
        return False
    
    for candle in ohlcv_data:
        if len(candle) != 6:
            return False
        
        timestamp, open_price, high, low, close, volume = candle
        
        # Basic validation
        if not all(isinstance(x, (int, float)) for x in candle):
            return False
        
        if high < max(open_price, close) or low > min(open_price, close):
            return False
        
        if volume < 0:
            return False
    
    return True


def get_data_summary(ohlcv_data: List[List[float]]) -> Dict[str, Any]:
    """Get summary statistics of OHLCV data"""
    if not ohlcv_data:
        return {}
    
    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    return {
        'total_candles': len(df),
        'start_time': pd.to_datetime(df['timestamp'].min(), unit='ms').isoformat(),
        'end_time': pd.to_datetime(df['timestamp'].max(), unit='ms').isoformat(),
        'price_range': {
            'min': float(df['low'].min()),
            'max': float(df['high'].max()),
            'avg': float(df['close'].mean())
        },
        'volume_stats': {
            'total': float(df['volume'].sum()),
            'avg': float(df['volume'].mean()),
            'max': float(df['volume'].max())
        }
    }