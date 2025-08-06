#!/usr/bin/env python3
"""
🚀 RTAI PERFORMANCE OPTIMIZATION TESTING
========================================

Performance stress testing to achieve <1 second processing for live trading.
FOCUS: Performance optimization and high-frequency market updates.

VALIDATION TARGETS:
- Sub-second processing for 100+ candles
- Memory efficiency under continuous updates
- Stress testing with high-frequency data
- Real-time performance benchmarks
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
import os
import asyncio
import time
import psutil
import gc

# Add paths  
sys.path.append(r'C:\Users\Davidoggo\Desktop\Fund')
sys.path.append(r'C:\Users\Davidoggo\Desktop\Fund\rtai')
sys.path.append(r'C:\Users\Davidoggo\Desktop\Fund\ft\user_data\strategies\lib')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RTAIPerformanceOptimizer:
    """
    ⚡ RTAI Performance Optimization Suite
    
    Comprehensive performance testing for live trading readiness:
    - Latency optimization (target <1s for 100 candles)
    - Memory efficiency monitoring
    - High-frequency data processing
    - Stress testing under various conditions
    """
    
    def __init__(self):
        self.performance_results = {
            'baseline_performance': {},
            'memory_efficiency': {},
            'high_frequency_test': {},
            'stress_test': {},
            'optimization_achieved': False,
            'benchmarks': {},
            'errors': []
        }
        self.process = psutil.Process()
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def create_high_frequency_data(self, num_candles: int = 200, volatility_factor: float = 1.0) -> pd.DataFrame:
        """Generate high-frequency trading data with realistic microstructure"""
        np.random.seed(42)
        
        base_price = 65000.0
        timestamps = [datetime.now() - timedelta(seconds=num_candles-i) for i in range(num_candles)]
        
        # High-frequency price movements with momentum and mean reversion
        momentum_factor = np.random.randn(num_candles) * 0.7  # Momentum component
        mean_reversion = -0.3 * np.cumsum(np.random.randn(num_candles)) * volatility_factor  # Mean reversion
        noise = np.random.randn(num_candles) * 20 * volatility_factor
        
        price_changes = momentum_factor + mean_reversion + noise
        closes = base_price + np.cumsum(price_changes)
        
        # OHLCV with high-frequency characteristics
        tick_size = 0.5  # BTC tick size
        highs = closes + np.abs(np.random.randn(num_candles)) * 15 * volatility_factor
        lows = closes - np.abs(np.random.randn(num_candles)) * 15 * volatility_factor
        opens = np.concatenate([[base_price], closes[:-1]])
        
        # High-frequency volume patterns
        base_volume = 500
        volume_spikes = np.random.poisson(volatility_factor, num_candles) * 300
        volumes = base_volume + volume_spikes + np.random.randint(50, 400, num_candles)
        
        # Realistic microstructure data
        spreads = np.random.uniform(0.5, 3.0 * volatility_factor, num_candles)
        bids = closes - spreads/2
        asks = closes + spreads/2
        
        # Dynamic bid/ask sizes (higher during volatility)
        volatility = np.abs(np.diff(closes, prepend=closes[0]))
        size_base = 2.0
        bid_sizes = size_base + (volatility / 50) + np.random.uniform(0.5, 4.0, num_candles)
        ask_sizes = size_base + (volatility / 50) + np.random.uniform(0.5, 4.0, num_candles)
        
        # Liquidation data with clustering during high volatility
        vol_percentile = np.percentile(volatility, 80)
        liquidation_intensity = np.where(volatility > vol_percentile, 
                                       np.random.poisson(5 * volatility_factor), 
                                       np.random.poisson(1))
        long_liquidations = np.maximum(liquidation_intensity + np.random.poisson(1), 0)
        short_liquidations = np.maximum(liquidation_intensity + np.random.poisson(1), 0)
        
        df = pd.DataFrame({
            'date': timestamps,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'volume': volumes,
            'bid': bids,
            'ask': asks,
            'bid_size': bid_sizes,
            'ask_size': ask_sizes,
            'long_liquidations': long_liquidations,
            'short_liquidations': short_liquidations
        })
        
        return df
    
    def test_baseline_performance(self) -> bool:
        """Test 1: Baseline Performance Measurement"""
        logger.info("🔥 TEST 1: Baseline Performance Measurement")
        
        try:
            from rtai_indicators import add_all_rtai_indicators
            
            # Test different data sizes for performance profiling
            test_sizes = [50, 100, 200, 500]
            results = {}
            
            for size in test_sizes:
                logger.info(f"  📊 Testing size {size} candles...")
                
                # Generate test data
                test_data = self.create_high_frequency_data(size)
                
                # Memory before processing
                mem_before = self.get_memory_usage()
                
                # Process with timing
                start_time = time.time()
                enhanced_data = add_all_rtai_indicators(test_data)
                processing_time = time.time() - start_time
                
                # Memory after processing
                mem_after = self.get_memory_usage()
                memory_used = mem_after - mem_before
                
                # Performance metrics
                candles_per_second = size / processing_time
                added_columns = enhanced_data.shape[1] - test_data.shape[1]
                
                results[size] = {
                    'processing_time': processing_time,
                    'candles_per_second': candles_per_second,
                    'memory_used_mb': memory_used,
                    'added_columns': added_columns
                }
                
                logger.info(f"    • Size {size}: {processing_time:.3f}s ({candles_per_second:.1f} candles/s, {memory_used:.1f}MB)")
                
                # Force garbage collection
                gc.collect()
            
            # Analyze results
            avg_performance = np.mean([results[size]['candles_per_second'] for size in test_sizes])
            target_size_performance = results[100]['processing_time']  # 100 candles target
            
            logger.info(f"  📊 Baseline Performance Summary:")
            logger.info(f"    • Average processing rate: {avg_performance:.1f} candles/s")
            logger.info(f"    • 100-candle processing: {target_size_performance:.3f}s")
            logger.info(f"    • Memory efficiency: {np.mean([results[size]['memory_used_mb'] for size in test_sizes]):.1f}MB avg")
            
            self.performance_results['baseline_performance'] = {
                'avg_candles_per_second': avg_performance,
                'target_processing_time': target_size_performance,
                'detailed_results': results
            }
            
            # Success criteria: Should process 100 candles in under 2 seconds
            success = target_size_performance < 2.0 and avg_performance > 50
            
            if success:
                logger.info("  ✅ Baseline performance PASSED")
                return True
            else:
                logger.error(f"  ❌ Baseline performance FAILED - Target: <2s, Got: {target_size_performance:.3f}s")
                return False
                
        except Exception as e:
            logger.error(f"  ❌ Baseline performance test failed: {e}")
            self.performance_results['errors'].append(f"Baseline Performance: {e}")
            return False
    
    def test_memory_efficiency(self) -> bool:
        """Test 2: Memory Efficiency Under Load"""
        logger.info("🔥 TEST 2: Memory Efficiency Under Continuous Load")
        
        try:
            from rtai_indicators import add_all_rtai_indicators
            
            # Initialize with base data
            base_data = self.create_high_frequency_data(60)
            
            memory_snapshots = []
            processing_times = []
            iterations = 20  # Simulate 20 continuous updates
            
            initial_memory = self.get_memory_usage()
            logger.info(f"  💾 Initial memory: {initial_memory:.1f}MB")
            
            for i in range(iterations):
                # Simulate new market update
                last_row = base_data.iloc[-1].copy()
                last_row['date'] = datetime.now()
                last_row['close'] += np.random.randn() * 10  # Price movement
                last_row['high'] = max(last_row['close'], last_row['high'])
                last_row['low'] = min(last_row['close'], last_row['low'])
                last_row['volume'] = np.random.randint(400, 1000)
                last_row['bid'] = last_row['close'] - np.random.uniform(1, 3)
                last_row['ask'] = last_row['close'] + np.random.uniform(1, 3)
                last_row['long_liquidations'] = np.random.poisson(2)
                last_row['short_liquidations'] = np.random.poisson(2)
                
                # Update rolling dataset
                base_data = pd.concat([base_data, pd.DataFrame([last_row])], ignore_index=True)
                if len(base_data) > 100:
                    base_data = base_data.iloc[-100:].copy()  # Keep rolling window
                
                # Process indicators
                start_time = time.time()
                enhanced_data = add_all_rtai_indicators(base_data)
                processing_time = time.time() - start_time
                
                # Memory tracking
                current_memory = self.get_memory_usage()
                memory_snapshots.append(current_memory)
                processing_times.append(processing_time)
                
                if (i + 1) % 5 == 0:
                    logger.info(f"    Iteration {i+1}: {processing_time:.3f}s, Memory: {current_memory:.1f}MB")
                
                # Cleanup attempt
                if i % 10 == 0:
                    gc.collect()
            
            # Memory analysis
            final_memory = self.get_memory_usage()
            peak_memory = max(memory_snapshots)
            avg_memory = np.mean(memory_snapshots)
            memory_growth = final_memory - initial_memory
            avg_processing_time = np.mean(processing_times)
            max_processing_time = max(processing_times)
            
            logger.info(f"  📊 Memory Efficiency Results:")
            logger.info(f"    • Initial memory: {initial_memory:.1f}MB")
            logger.info(f"    • Final memory: {final_memory:.1f}MB")
            logger.info(f"    • Peak memory: {peak_memory:.1f}MB")
            logger.info(f"    • Average memory: {avg_memory:.1f}MB")
            logger.info(f"    • Memory growth: {memory_growth:.1f}MB")
            logger.info(f"    • Avg processing time: {avg_processing_time:.3f}s")
            logger.info(f"    • Max processing time: {max_processing_time:.3f}s")
            
            self.performance_results['memory_efficiency'] = {
                'initial_memory_mb': initial_memory,
                'final_memory_mb': final_memory,
                'peak_memory_mb': peak_memory,
                'memory_growth_mb': memory_growth,
                'avg_processing_time': avg_processing_time,
                'max_processing_time': max_processing_time
            }
            
            # Success criteria: Memory growth should be minimal, processing should be consistent
            success = (memory_growth < 50 and  # Less than 50MB growth
                      avg_processing_time < 1.5 and  # Average under 1.5s
                      max_processing_time < 3.0)  # Max under 3s
            
            if success:
                logger.info("  ✅ Memory efficiency test PASSED")
                return True
            else:
                logger.error("  ❌ Memory efficiency test FAILED")
                return False
                
        except Exception as e:
            logger.error(f"  ❌ Memory efficiency test failed: {e}")
            self.performance_results['errors'].append(f"Memory Efficiency: {e}")
            return False
    
    async def test_high_frequency_processing(self, duration_seconds: int = 10) -> bool:
        """Test 3: High-Frequency Data Processing"""
        logger.info(f"🔥 TEST 3: High-Frequency Processing ({duration_seconds}s)")
        
        try:
            from rtai_indicators import add_all_rtai_indicators
            
            # Start with volatile market conditions
            market_data = self.create_high_frequency_data(50, volatility_factor=2.0)
            
            start_time = time.time()
            updates_processed = 0
            processing_times = []
            signal_events = []
            memory_snapshots = []
            
            while time.time() - start_time < duration_seconds:
                # High-frequency market tick (every 200ms)
                last_close = market_data['close'].iloc[-1]
                
                # Realistic high-frequency price movement
                tick_size = 0.5
                price_move = np.random.choice([-tick_size*2, -tick_size, 0, tick_size, tick_size*2], 
                                            p=[0.1, 0.3, 0.2, 0.3, 0.1])  # Mean-reverting bias
                new_price = last_close + price_move + np.random.randn() * 5  # Some noise
                
                # Create high-frequency update
                new_row = pd.DataFrame([{
                    'date': datetime.now(),
                    'open': last_close,
                    'high': max(last_close, new_price) + abs(np.random.randn()) * 3,
                    'low': min(last_close, new_price) - abs(np.random.randn()) * 3,
                    'close': new_price,
                    'volume': np.random.randint(100, 500),  # Smaller size for HFT
                    'bid': new_price - np.random.uniform(0.5, 2.0),
                    'ask': new_price + np.random.uniform(0.5, 2.0),
                    'bid_size': np.random.uniform(0.5, 3.0),
                    'ask_size': np.random.uniform(0.5, 3.0),
                    'long_liquidations': np.random.poisson(0.5),
                    'short_liquidations': np.random.poisson(0.5)
                }])
                
                # Update with rolling window
                market_data = pd.concat([market_data, new_row], ignore_index=True)
                if len(market_data) > 120:  # Larger window for HFT
                    market_data = market_data.iloc[-120:].copy()
                
                # Process indicators with timing
                update_start = time.time()
                enhanced_data = add_all_rtai_indicators(market_data)
                update_time = time.time() - update_start
                
                processing_times.append(update_time)
                memory_snapshots.append(self.get_memory_usage())
                
                # Signal detection
                if 'ofi_z' in enhanced_data.columns and 'mpd_z' in enhanced_data.columns:
                    latest_ofi = enhanced_data['ofi_z'].iloc[-1]
                    latest_mpd = enhanced_data['mpd_z'].iloc[-1]
                    latest_composite = enhanced_data.get('rtai_composite', pd.Series([50])).iloc[-1]
                    
                    # HFT signal conditions (more sensitive)
                    if abs(latest_ofi) > 1.5 or abs(latest_mpd) > 1.0 or latest_composite < 25 or latest_composite > 75:
                        signal_events.append({
                            'time': datetime.now(),
                            'ofi': latest_ofi,
                            'mpd': latest_mpd,
                            'composite': latest_composite,
                            'price': new_price,
                            'processing_time': update_time
                        })
                
                updates_processed += 1
                
                # Progress logging
                if updates_processed % 10 == 0:
                    avg_time = np.mean(processing_times[-10:])
                    logger.info(f"    Update {updates_processed}: Avg time {avg_time:.3f}s, Signals: {len(signal_events)}")
                
                # High-frequency delay
                await asyncio.sleep(0.2)  # 200ms intervals = 5 Hz
            
            # Performance analysis
            total_time = time.time() - start_time
            avg_processing_time = np.mean(processing_times) if processing_times else 0
            max_processing_time = max(processing_times) if processing_times else 0
            percentile_95 = np.percentile(processing_times, 95) if processing_times else 0
            
            throughput = updates_processed / total_time
            signal_rate = len(signal_events) / updates_processed if updates_processed > 0 else 0
            
            logger.info(f"  📊 High-Frequency Processing Results:")
            logger.info(f"    • Updates processed: {updates_processed}")
            logger.info(f"    • Total time: {total_time:.1f}s")
            logger.info(f"    • Throughput: {throughput:.1f} updates/s")
            logger.info(f"    • Avg processing time: {avg_processing_time:.3f}s")
            logger.info(f"    • Max processing time: {max_processing_time:.3f}s")
            logger.info(f"    • 95th percentile: {percentile_95:.3f}s")
            logger.info(f"    • Signals detected: {len(signal_events)} ({signal_rate*100:.1f}% rate)")
            logger.info(f"    • Memory usage: {np.mean(memory_snapshots):.1f}MB avg")
            
            self.performance_results['high_frequency_test'] = {
                'updates_processed': updates_processed,
                'throughput': throughput,
                'avg_processing_time': avg_processing_time,
                'max_processing_time': max_processing_time,
                'percentile_95': percentile_95,
                'signals_detected': len(signal_events),
                'signal_rate': signal_rate
            }
            
            # Success criteria for HFT
            success = (throughput >= 3.0 and  # At least 3 updates/s
                      avg_processing_time < 0.8 and  # Fast average
                      percentile_95 < 2.0)  # 95% under 2s
            
            if success:
                logger.info("  ✅ High-frequency processing test PASSED")
                return True
            else:
                logger.error("  ❌ High-frequency processing test FAILED")
                return False
                
        except Exception as e:
            logger.error(f"  ❌ High-frequency processing test failed: {e}")
            self.performance_results['errors'].append(f"High Frequency: {e}")
            return False
    
    async def test_stress_conditions(self) -> bool:
        """Test 4: Stress Test Under Extreme Conditions"""
        logger.info("🔥 TEST 4: Stress Test - Extreme Market Conditions")
        
        try:
            from rtai_indicators import add_all_rtai_indicators
            
            stress_scenarios = [
                {"name": "Flash Crash", "volatility": 5.0, "duration": 15},
                {"name": "High Volume", "volatility": 3.0, "duration": 10},
                {"name": "Low Liquidity", "volatility": 2.0, "duration": 12}
            ]
            
            stress_results = {}
            
            for scenario in stress_scenarios:
                logger.info(f"  💥 Testing {scenario['name']} scenario...")
                
                # Generate extreme market data
                extreme_data = self.create_high_frequency_data(80, scenario['volatility'])
                
                # Add extreme conditions
                if scenario['name'] == "Flash Crash":
                    # Simulate flash crash - large price drops
                    crash_indices = [20, 21, 22, 23, 24]
                    for idx in crash_indices:
                        if idx < len(extreme_data):
                            extreme_data.loc[idx, 'close'] *= 0.95  # 5% drop each candle
                            extreme_data.loc[idx, 'low'] = extreme_data.loc[idx, 'close'] * 0.98
                            extreme_data.loc[idx, 'long_liquidations'] = 100  # Massive liquidations
                            extreme_data.loc[idx, 'ask'] = extreme_data.loc[idx, 'close'] * 1.02  # Wide spreads
                
                elif scenario['name'] == "High Volume":
                    # Simulate high volume period
                    extreme_data['volume'] *= 5
                    extreme_data['long_liquidations'] += np.random.poisson(10, len(extreme_data))
                    extreme_data['short_liquidations'] += np.random.poisson(10, len(extreme_data))
                
                elif scenario['name'] == "Low Liquidity":
                    # Simulate low liquidity - wide spreads, small sizes
                    extreme_data['bid_size'] *= 0.2
                    extreme_data['ask_size'] *= 0.2
                    spread_multiplier = 5.0
                    mid_price = (extreme_data['bid'] + extreme_data['ask']) / 2
                    extreme_data['bid'] = mid_price - spread_multiplier
                    extreme_data['ask'] = mid_price + spread_multiplier
                
                # Process under stress
                processing_times = []
                memory_usage = []
                
                for iteration in range(scenario['duration']):
                    mem_before = self.get_memory_usage()
                    
                    start_time = time.time()
                    enhanced_data = add_all_rtai_indicators(extreme_data)
                    processing_time = time.time() - start_time
                    
                    processing_times.append(processing_time)
                    memory_usage.append(self.get_memory_usage() - mem_before)
                    
                    # Simulate data evolution
                    if iteration < scenario['duration'] - 1:
                        last_row = extreme_data.iloc[-1].copy()
                        last_row['close'] += np.random.randn() * scenario['volatility'] * 10
                        extreme_data = pd.concat([extreme_data, pd.DataFrame([last_row])], ignore_index=True)
                        if len(extreme_data) > 100:
                            extreme_data = extreme_data.iloc[-100:].copy()
                    
                    await asyncio.sleep(0.1)  # Brief pause
                
                # Analyze stress results
                avg_processing = np.mean(processing_times)
                max_processing = max(processing_times)
                avg_memory = np.mean(memory_usage)
                
                stress_results[scenario['name']] = {
                    'avg_processing_time': avg_processing,
                    'max_processing_time': max_processing,
                    'avg_memory_usage': avg_memory,
                    'stability': max_processing / avg_processing  # Stability ratio
                }
                
                logger.info(f"    • {scenario['name']}: Avg {avg_processing:.3f}s, Max {max_processing:.3f}s, Memory {avg_memory:.1f}MB")
            
            # Overall stress analysis
            overall_avg = np.mean([stress_results[s]['avg_processing_time'] for s in stress_results])
            overall_max = max([stress_results[s]['max_processing_time'] for s in stress_results])
            stability_score = np.mean([stress_results[s]['stability'] for s in stress_results])
            
            logger.info(f"  📊 Stress Test Summary:")
            logger.info(f"    • Overall average: {overall_avg:.3f}s")
            logger.info(f"    • Overall maximum: {overall_max:.3f}s") 
            logger.info(f"    • Stability score: {stability_score:.2f}x")
            
            self.performance_results['stress_test'] = {
                'scenarios': stress_results,
                'overall_avg': overall_avg,
                'overall_max': overall_max,
                'stability_score': stability_score
            }
            
            # Success criteria for stress test
            success = (overall_avg < 1.5 and  # Average under 1.5s even under stress
                      overall_max < 4.0 and     # Maximum under 4s
                      stability_score < 3.0)    # Reasonable stability
            
            if success:
                logger.info("  ✅ Stress test PASSED")
                return True
            else:
                logger.error("  ❌ Stress test FAILED")
                return False
                
        except Exception as e:
            logger.error(f"  ❌ Stress test failed: {e}")
            self.performance_results['errors'].append(f"Stress Test: {e}")
            return False
    
    async def run_complete_optimization_test(self) -> dict:
        """Run complete performance optimization test suite"""
        logger.info("🚀 STARTING RTAI PERFORMANCE OPTIMIZATION TESTING")
        logger.info("=" * 70)
        
        start_time = time.time()
        
        # Execute optimization test sequence
        test1_passed = self.test_baseline_performance()
        test2_passed = self.test_memory_efficiency() if test1_passed else False
        test3_passed = await self.test_high_frequency_processing(8) if test2_passed else False  # 8-second HFT test
        test4_passed = await self.test_stress_conditions() if test3_passed else False
        
        # Performance optimization assessment
        total_time = time.time() - start_time
        all_tests_passed = all([test1_passed, test2_passed, test3_passed, test4_passed])
        
        # Calculate optimization score
        if all_tests_passed:
            baseline_perf = self.performance_results['baseline_performance']['target_processing_time']
            optimization_score = min(2.0 / baseline_perf, 3.0)  # Score out of 3
            self.performance_results['optimization_achieved'] = baseline_perf < 1.0
        else:
            optimization_score = 0
            self.performance_results['optimization_achieved'] = False
        
        logger.info("=" * 70)
        logger.info("📊 RTAI PERFORMANCE OPTIMIZATION RESULTS SUMMARY")
        logger.info(f"🕐 Total testing time: {total_time:.1f}s")
        logger.info(f"1️⃣  Baseline Performance: {'✅ PASS' if test1_passed else '❌ FAIL'}")
        logger.info(f"2️⃣  Memory Efficiency: {'✅ PASS' if test2_passed else '❌ FAIL'}")
        logger.info(f"3️⃣  High-Frequency Processing: {'✅ PASS' if test3_passed else '❌ FAIL'}")
        logger.info(f"4️⃣  Stress Conditions: {'✅ PASS' if test4_passed else '❌ FAIL'}")
        logger.info(f"⚡ Optimization Score: {optimization_score:.1f}/3.0")
        
        if all_tests_passed:
            logger.info("🎉 ALL PERFORMANCE TESTS PASSED")
            if self.performance_results['optimization_achieved']:
                logger.info("⚡ PERFORMANCE TARGET ACHIEVED: <1s processing time")
                logger.info("🚀 RTAI SYSTEM IS OPTIMIZED FOR LIVE TRADING")
            else:
                logger.info("⚠️ PERFORMANCE ACCEPTABLE BUT NOT OPTIMAL")
        else:
            logger.error("💥 PERFORMANCE OPTIMIZATION INCOMPLETE")
            logger.error(f"🔍 Errors: {self.performance_results['errors']}")
        
        self.performance_results['all_passed'] = all_tests_passed
        self.performance_results['optimization_score'] = optimization_score
        self.performance_results['execution_time'] = total_time
        
        return self.performance_results


async def main():
    """Main optimization test execution"""
    optimizer = RTAIPerformanceOptimizer()
    results = await optimizer.run_complete_optimization_test()
    
    exit_code = 0 if results['all_passed'] else 1
    logger.info(f"🏁 Performance optimization testing completed with exit code: {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
