"""Backfill utility for warm-up after reconnection"""

import asyncio
import time
import httpx
from typing import List, Dict, Optional
from loguru import logger


class BackfillManager:
    """Manages historical data backfill after reconnection"""
    
    def __init__(self, base_url: str = "https://api.binance.com"):
        self.base_url = base_url
        
    async def request_historical_trades(self, symbol: str, seconds: int = 60) -> List[Dict]:
        """Request historical trades for the last N seconds"""
        try:
            # Calculate timestamp for N seconds ago
            end_time = int(time.time() * 1000)
            start_time = end_time - (seconds * 1000)
            
            url = f"{self.base_url}/api/v3/aggTrades"
            params = {
                'symbol': symbol.upper(),
                'startTime': start_time,
                'endTime': end_time,
                'limit': 1000
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, timeout=10)
                response.raise_for_status()
                
                trades = response.json()
                logger.info(f"🔄 Backfilled {len(trades)} historical trades for {symbol}")
                
                # Convert to standard format
                historical_trades = []
                for trade in trades:
                    historical_trades.append({
                        'price': float(trade['p']),
                        'volume': float(trade['q']),
                        'timestamp': float(trade['T']) / 1000,
                        'side': 'sell' if trade['m'] else 'buy',  # m=true means buyer is maker
                        'backfill': True  # Mark as backfill data
                    })
                
                return historical_trades
                
        except Exception as e:
            logger.error(f"❌ Failed to backfill historical trades: {e}")
            return []
    
    async def warm_up_indicators(self, symbol: str, handler, seconds: int = 60):
        """Warm up indicators with historical data"""
        logger.info(f"🔥 Warming up indicators with {seconds}s of historical data...")
        
        historical_trades = await self.request_historical_trades(symbol, seconds)
        
        if not historical_trades:
            logger.warning("⚠️ No historical data available for warm-up")
            return
        
        # Process historical trades through handler
        backfill_count = 0
        for trade in historical_trades:
            try:
                await handler(trade)
                backfill_count += 1
            except Exception as e:
                logger.debug(f"Error processing backfill trade: {e}")
        
        logger.success(f"✅ Warm-up complete: processed {backfill_count} historical trades")
