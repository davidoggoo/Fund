#!/usr/bin/env python3
"""
RTAI Real-Time Pipeline Test - NO MOCK DATA
Direct connection to Binance WebSocket streams for live data
Testing the complete pipeline with actual market data
"""

import asyncio
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any
import pandas as pd
import numpy as np

# Import our real components - NO MOCKS
from RTAIDataProvider import RTAIDataProvider
from user_data.strategies.lib.rtai_indicators import (
    adaptive_ofi_series,
    microprice_divergence,
    enhanced_vpin_series,
    robust_z_score
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RealTimePipelineTest:
    """
    Test the complete real-time pipeline:
    Binance WebSocket → RTAIDataProvider → Enhanced Indicators → Strategy Signals
    """
    
    def __init__(self):
        self.data_provider = RTAIDataProvider()
        self.historical_data = []
        self.indicator_values = {
            'ofi': [],
            'vpin': [],
            'mpd': [],
            'lpi': [],
            'atr': []
        }
        self.signal_count = 0
        self.start_time = time.time()
        
    async def initialize_provider(self):
        """Initialize data provider with real Binance connection"""
        logger.info("🚀 Initializing Real-Time Data Provider...")
        await self.data_provider.initialize()
        logger.info("✅ RTAIDataProvider initialized with live Binance connection")
        
    async def process_real_data(self, duration_minutes: int = 5):
        """Process real-time data for specified duration"""
        logger.info(f"📊 Starting {duration_minutes}-minute real-time data processing...")
        
        end_time = time.time() + (duration_minutes * 60)
        
        while time.time() < end_time:
            try:
                # Get real-time data from provider
                current_data = await self.data_provider.get_current_data()
                
                if current_data and len(current_data.get('trades', [])) > 0:
                    await self.process_tick_data(current_data)
                    
                # Process every 500ms for high-frequency monitoring
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"❌ Error in real-time processing: {e}")
                await asyncio.sleep(1)
                
        logger.info(f"🏁 Completed {duration_minutes}-minute real-time test")
        
    async def process_tick_data(self, tick_data: Dict[str, Any]):
        """Process individual tick with real indicators"""
        try:
            # Store historical data
            self.historical_data.append({
                'timestamp': tick_data.get('timestamp', time.time()),
                'price': tick_data.get('price', 0),
                'volume': tick_data.get('volume', 0),
                'trades': tick_data.get('trades', []),
                'orderbook': tick_data.get('orderbook', {})
            })
            
            # Keep last 200 ticks for indicator calculation
            if len(self.historical_data) > 200:
                self.historical_data = self.historical_data[-200:]
                
            # Calculate indicators when we have enough data
            if len(self.historical_data) >= 20:
                await self.calculate_real_indicators()
                
        except Exception as e:
            logger.error(f"❌ Error processing tick data: {e}")
            
    async def calculate_real_indicators(self):
        """Calculate all indicators with REAL data - NO PLACEHOLDERS"""
        try:
            # Convert to DataFrame for indicator calculation
            df = pd.DataFrame(self.historical_data)
            
            if len(df) < 20:
                return
                
            # Calculate OFI using real trade data
            if 'trades' in df.columns and len(df) > 10:
                ofi_values = self.calculate_real_ofi(df)
                if len(ofi_values) > 0:
                    self.indicator_values['ofi'].extend(ofi_values)
                    
            # Calculate VPIN using real volume data  
            if 'volume' in df.columns and len(df) > 15:
                vpin_values = self.calculate_real_vpin(df)
                if len(vpin_values) > 0:
                    self.indicator_values['vpin'].extend(vpin_values)
                    
            # Calculate Microprice Divergence
            if 'orderbook' in df.columns:
                mpd_values = self.calculate_real_mpd(df)
                if len(mpd_values) > 0:
                    self.indicator_values['mpd'].extend(mpd_values)
                    
            # Generate strategy signals
            await self.generate_strategy_signals()
            
        except Exception as e:
            logger.error(f"❌ Error calculating indicators: {e}")
            
    def calculate_real_ofi(self, df: pd.DataFrame) -> list:
        """Calculate Order Flow Imbalance using real trade data"""
        try:
            ofi_series = []
            
            for _, row in df.iterrows():
                trades = row.get('trades', [])
                if not trades:
                    continue
                    
                buy_volume = sum(t.get('volume', 0) for t in trades if t.get('side') == 'buy')
                sell_volume = sum(t.get('volume', 0) for t in trades if t.get('side') == 'sell')
                
                if buy_volume + sell_volume > 0:
                    ofi = (buy_volume - sell_volume) / (buy_volume + sell_volume)
                    ofi_series.append(ofi)
                    
            return ofi_series[-5:] if len(ofi_series) >= 5 else []
            
        except Exception as e:
            logger.error(f"❌ Error calculating real OFI: {e}")
            return []
            
    def calculate_real_vpin(self, df: pd.DataFrame) -> list:
        """Calculate VPIN using real volume data"""
        try:
            if len(df) < 10:
                return []
                
            volumes = df['volume'].dropna().tolist()
            if len(volumes) < 10:
                return []
                
            # Use our enhanced VPIN function with real data
            vpin_result = enhanced_vpin_series(
                volumes=np.array(volumes),
                bucket_size=len(volumes) // 5,  # Dynamic bucket sizing
                window=min(10, len(volumes) // 2)
            )
            
            if hasattr(vpin_result, '__iter__'):
                return vpin_result.tolist()[-3:] if len(vpin_result) > 0 else []
            else:
                return [float(vpin_result)] if not np.isnan(vpin_result) else []
                
        except Exception as e:
            logger.error(f"❌ Error calculating real VPIN: {e}")
            return []
            
    def calculate_real_mpd(self, df: pd.DataFrame) -> list:
        """Calculate Microprice Divergence using real orderbook data"""
        try:
            mpd_series = []
            
            for _, row in df.iterrows():
                orderbook = row.get('orderbook', {})
                if not orderbook:
                    continue
                    
                bids = orderbook.get('bids', [])
                asks = orderbook.get('asks', [])
                
                if bids and asks:
                    best_bid = float(bids[0][0]) if bids[0] else 0
                    best_ask = float(asks[0][0]) if asks[0] else 0
                    mid_price = (best_bid + best_ask) / 2
                    
                    # Microprice calculation
                    bid_vol = float(bids[0][1]) if bids[0] else 0
                    ask_vol = float(asks[0][1]) if asks[0] else 0
                    
                    if bid_vol + ask_vol > 0:
                        microprice = (best_bid * ask_vol + best_ask * bid_vol) / (bid_vol + ask_vol)
                        mpd = (microprice - mid_price) / mid_price * 10000  # bps
                        mpd_series.append(mpd)
                        
            return mpd_series[-3:] if len(mpd_series) >= 3 else []
            
        except Exception as e:
            logger.error(f"❌ Error calculating real MPD: {e}")
            return []
            
    async def generate_strategy_signals(self):
        """Generate trading signals using real indicator values"""
        try:
            if (len(self.indicator_values['ofi']) >= 5 and 
                len(self.indicator_values['vpin']) >= 3 and
                len(self.indicator_values['mpd']) >= 3):
                
                # Get latest values
                latest_ofi = self.indicator_values['ofi'][-1]
                latest_vpin = self.indicator_values['vpin'][-1]
                latest_mpd = self.indicator_values['mpd'][-1]
                
                # Calculate Z-scores using robust method
                ofi_z = self.calculate_z_score(self.indicator_values['ofi'][-10:])
                mpd_z = self.calculate_z_score(self.indicator_values['mpd'][-10:])
                
                # Strategy logic: Mean-reversion signals
                signal_strength = 0
                signal_type = "NEUTRAL"
                
                # Entry condition: |OFI_z| > 2.25 AND sign(OFI_z) != sign(MPD_z) AND |MPD_z| > 1.5
                if abs(ofi_z) > 2.25 and abs(mpd_z) > 1.5:
                    if np.sign(ofi_z) != np.sign(mpd_z):
                        signal_strength = abs(ofi_z) * abs(mpd_z) * latest_vpin
                        signal_type = "SELL" if ofi_z > 0 else "BUY"
                        
                        self.signal_count += 1
                        
                        logger.info(f"🎯 SIGNAL #{self.signal_count}: {signal_type}")
                        logger.info(f"   📊 OFI_z: {ofi_z:.3f}, MPD_z: {mpd_z:.3f}, VPIN: {latest_vpin:.4f}")
                        logger.info(f"   💪 Signal Strength: {signal_strength:.4f}")
                        
                # Log periodic status
                elapsed = time.time() - self.start_time
                if int(elapsed) % 30 == 0:  # Every 30 seconds
                    logger.info(f"📈 Status - Elapsed: {elapsed:.1f}s, Signals: {self.signal_count}")
                    logger.info(f"   📊 Latest: OFI={latest_ofi:.4f}, VPIN={latest_vpin:.4f}, MPD={latest_mpd:.2f}")
                    
        except Exception as e:
            logger.error(f"❌ Error generating signals: {e}")
            
    def calculate_z_score(self, values: list) -> float:
        """Calculate robust Z-score using MAD"""
        try:
            if len(values) < 3:
                return 0.0
                
            arr = np.array(values)
            median = np.median(arr)
            mad = np.median(np.abs(arr - median))
            
            if mad == 0:
                return 0.0
                
            # Robust Z-score
            z_score = 0.6745 * (arr[-1] - median) / mad
            return float(z_score)
            
        except Exception as e:
            logger.error(f"❌ Error calculating Z-score: {e}")
            return 0.0
            
    async def run_complete_test(self, duration_minutes: int = 3):
        """Run complete real-time pipeline test"""
        logger.info("🚀 STARTING REAL-TIME PIPELINE TEST")
        logger.info("=" * 60)
        
        try:
            # Initialize provider
            await self.initialize_provider()
            
            # Process real-time data
            await self.process_real_data(duration_minutes)
            
            # Final summary
            await self.print_test_summary()
            
        except Exception as e:
            logger.error(f"❌ Test failed: {e}")
        finally:
            await self.cleanup()
            
    async def print_test_summary(self):
        """Print test results summary"""
        elapsed = time.time() - self.start_time
        
        logger.info("=" * 60)
        logger.info("📊 REAL-TIME PIPELINE TEST SUMMARY")
        logger.info("=" * 60)
        logger.info(f"⏱️  Total Duration: {elapsed:.1f} seconds")
        logger.info(f"📈 Data Points Processed: {len(self.historical_data)}")
        logger.info(f"🎯 Trading Signals Generated: {self.signal_count}")
        logger.info(f"📊 OFI Values: {len(self.indicator_values['ofi'])}")
        logger.info(f"📊 VPIN Values: {len(self.indicator_values['vpin'])}")
        logger.info(f"📊 MPD Values: {len(self.indicator_values['mpd'])}")
        
        if self.signal_count > 0:
            logger.info(f"✅ SUCCESS: Real-time pipeline working with live data!")
        else:
            logger.info(f"⚠️  No signals generated - market may be quiet")
            
        logger.info("=" * 60)
        
    async def cleanup(self):
        """Cleanup resources"""
        try:
            await self.data_provider.cleanup()
            logger.info("🧹 Cleanup completed")
        except Exception as e:
            logger.error(f"❌ Cleanup error: {e}")

async def main():
    """Main test execution"""
    test = RealTimePipelineTest()
    await test.run_complete_test(duration_minutes=2)  # 2-minute test

if __name__ == "__main__":
    print("🚀 Starting RTAI Real-Time Pipeline Test")
    print("📡 Connecting to live Binance data streams...")
    print("💡 NO MOCK DATA - REAL MARKET DATA ONLY")
    print("-" * 50)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
