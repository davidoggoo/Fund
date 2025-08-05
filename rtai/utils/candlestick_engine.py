"""
Professional 1-minute candlestick engine with pandas-based OHLC aggregation.
Enhanced with proper timestamp handling, gap detection, and volume analytics.
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from collections import defaultdict
import time
import logging

logger = logging.getLogger(__name__)

@dataclass
class Trade:
    """Individual trade data structure"""
    timestamp: float
    price: float
    quantity: float
    side: str  # 'buy' or 'sell'
    
@dataclass
class OHLCV:
    """OHLC with volume data structure"""
    timestamp: float  # Unix timestamp of candle close
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: int
    buy_volume: float
    sell_volume: float
    vwap: float  # Volume-weighted average price

class CandlestickEngine:
    """
    Professional 1-minute candlestick aggregation engine.
    Features:
    - Pandas-based OHLC aggregation
    - Proper timezone handling
    - Gap detection and filling
    - Volume analytics (buy/sell breakdown)
    - VWAP calculation
    - Configurable aggregation intervals
    """
    
    def __init__(self, interval_seconds: int = 60):
        self.interval_seconds = interval_seconds
        self.trades_buffer: List[Trade] = []
        self.completed_candles: List[OHLCV] = []
        self.current_minute_start: Optional[float] = None
        self.last_processed_minute: Optional[float] = None
        
        # Volume analytics
        self.volume_profile: Dict[float, float] = defaultdict(float)
        self.tick_distribution: Dict[str, int] = {'up': 0, 'down': 0, 'flat': 0}
        
    def add_trade(self, timestamp: float, price: float, quantity: float, side: str = 'unknown') -> Optional[OHLCV]:
        """
        Add trade and return completed candle if minute boundary crossed.
        
        Args:
            timestamp: Unix timestamp
            price: Trade price
            quantity: Trade quantity (positive)
            side: Trade side ('buy', 'sell', or 'unknown')
            
        Returns:
            OHLCV candle if minute completed, None otherwise
        """
        try:
            # Create trade object
            trade = Trade(
                timestamp=float(timestamp),
                price=float(price),
                quantity=abs(float(quantity)),  # Ensure positive
                side=side.lower() if isinstance(side, str) else 'unknown'
            )
            
            # Calculate minute boundary (floor to minute)
            minute_boundary = int(trade.timestamp // self.interval_seconds) * self.interval_seconds
            
            # Initialize on first trade
            if self.current_minute_start is None:
                self.current_minute_start = minute_boundary
                logger.info(f"Candlestick engine initialized at minute {self.current_minute_start}")
            
            # Check if we've crossed into a new minute
            if minute_boundary > self.current_minute_start:
                # Finalize the current minute candle
                completed_candle = self._finalize_current_candle()
                
                # Check for gaps and fill if necessary
                self._handle_gaps(self.current_minute_start, minute_boundary)
                
                # Start new minute
                self.current_minute_start = minute_boundary
                self.trades_buffer = [trade]  # Start with current trade
                
                return completed_candle
            else:
                # Add to current minute buffer
                self.trades_buffer.append(trade)
                return None
                
        except (ValueError, TypeError) as e:
            logger.warning(f"Invalid trade data: {e}")
            return None
    
    def _finalize_current_candle(self) -> Optional[OHLCV]:
        """Finalize the current minute's trades into an OHLCV candle"""
        if not self.trades_buffer or self.current_minute_start is None:
            return None
            
        try:
            # Convert trades to DataFrame for efficient aggregation
            df = pd.DataFrame([
                {
                    'timestamp': trade.timestamp,
                    'price': trade.price,
                    'quantity': trade.quantity,
                    'side': trade.side,
                    'notional': trade.price * trade.quantity
                }
                for trade in self.trades_buffer
            ])
            
            # OHLC calculation
            prices = df['price']
            open_price = prices.iloc[0]  # First trade price
            high_price = prices.max()
            low_price = prices.min()
            close_price = prices.iloc[-1]  # Last trade price
            
            # Volume analytics
            total_volume = df['quantity'].sum()
            total_notional = df['notional'].sum()
            trade_count = len(df)
            
            # Buy/Sell volume breakdown
            buy_mask = df['side'] == 'buy'
            sell_mask = df['side'] == 'sell'
            buy_volume = df.loc[buy_mask, 'quantity'].sum() if buy_mask.any() else 0.0
            sell_volume = df.loc[sell_mask, 'quantity'].sum() if sell_mask.any() else 0.0
            
            # VWAP calculation
            vwap = total_notional / total_volume if total_volume > 0 else close_price
            
            # Create OHLCV candle
            candle = OHLCV(
                timestamp=self.current_minute_start + self.interval_seconds,  # Candle close time
                open=float(open_price),
                high=float(high_price),
                low=float(low_price),
                close=float(close_price),
                volume=float(total_volume),
                trade_count=trade_count,
                buy_volume=float(buy_volume),
                sell_volume=float(sell_volume),
                vwap=float(vwap)
            )
            
            # Store completed candle
            self.completed_candles.append(candle)
            self.last_processed_minute = self.current_minute_start
            
            # Update analytics
            self._update_analytics(df)
            
            logger.debug(f"Finalized candle: O={open_price:.2f} H={high_price:.2f} L={low_price:.2f} C={close_price:.2f} V={total_volume:.2f}")
            
            return candle
            
        except Exception as e:
            logger.error(f"Error finalizing candle: {e}")
            return None
    
    def _handle_gaps(self, last_minute: float, current_minute: float) -> None:
        """Handle gaps in data by creating flat candles"""
        gap_minutes = int((current_minute - last_minute) / self.interval_seconds) - 1
        
        if gap_minutes > 0:
            logger.warning(f"Detected {gap_minutes} minute gap in data, filling with flat candles")
            
            # Get last known price for gap filling
            last_price = self.completed_candles[-1].close if self.completed_candles else 50000.0
            
            # Fill gaps with flat candles
            for i in range(1, gap_minutes + 1):
                gap_timestamp = last_minute + (i * self.interval_seconds)
                gap_candle = OHLCV(
                    timestamp=gap_timestamp + self.interval_seconds,
                    open=last_price,
                    high=last_price,
                    low=last_price,
                    close=last_price,
                    volume=0.0,
                    trade_count=0,
                    buy_volume=0.0,
                    sell_volume=0.0,
                    vwap=last_price
                )
                self.completed_candles.append(gap_candle)
    
    def _update_analytics(self, trades_df: pd.DataFrame) -> None:
        """Update volume profile and tick distribution analytics"""
        try:
            # Volume profile (price levels)
            for _, trade in trades_df.iterrows():
                # Round price to nearest dollar for volume profile
                price_level = round(trade['price'])
                self.volume_profile[price_level] += trade['quantity']
            
            # Tick distribution (price movements)
            if len(trades_df) > 1:
                price_changes = trades_df['price'].diff().dropna()
                for change in price_changes:
                    if change > 0:
                        self.tick_distribution['up'] += 1
                    elif change < 0:
                        self.tick_distribution['down'] += 1
                    else:
                        self.tick_distribution['flat'] += 1
                        
        except Exception as e:
            logger.warning(f"Error updating analytics: {e}")
    
    def get_recent_candles(self, count: int = 100) -> List[OHLCV]:
        """Get the most recent completed candles"""
        return self.completed_candles[-count:] if self.completed_candles else []
    
    def force_finalize_current(self) -> Optional[OHLCV]:
        """Force finalization of current minute (useful for shutdown)"""
        if self.trades_buffer and self.current_minute_start is not None:
            return self._finalize_current_candle()
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics"""
        total_candles = len(self.completed_candles)
        total_trades = sum(candle.trade_count for candle in self.completed_candles)
        
        # Calculate average metrics
        if total_candles > 0:
            avg_volume = sum(candle.volume for candle in self.completed_candles) / total_candles
            avg_trades_per_minute = total_trades / total_candles
        else:
            avg_volume = 0.0
            avg_trades_per_minute = 0.0
        
        return {
            'total_candles': total_candles,
            'total_trades': total_trades,
            'avg_volume_per_minute': avg_volume,
            'avg_trades_per_minute': avg_trades_per_minute,
            'current_buffer_size': len(self.trades_buffer),
            'tick_distribution': dict(self.tick_distribution),
            'volume_profile_levels': len(self.volume_profile)
        }
    
    def to_pandas(self) -> pd.DataFrame:
        """Convert completed candles to pandas DataFrame"""
        if not self.completed_candles:
            return pd.DataFrame()
        
        data = []
        for candle in self.completed_candles:
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
                'vwap': candle.vwap,
                'buy_ratio': candle.buy_volume / candle.volume if candle.volume > 0 else 0.5
            })
        
        df = pd.DataFrame(data)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        return df
