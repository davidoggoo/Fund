#!/usr/bin/env python3
"""
RTAI Signal Generation Test
Test with relaxed conditions to verify signal generation works
"""

import sys
import os
import pandas as pd
import numpy as np

# Add paths
sys.path.append('/mnt/c/Users/Davidoggo/Desktop/Fund/ft')
sys.path.append('C:/Users/Davidoggo/Desktop/Fund/ft')

from user_data.strategies.RTAIStrategy import RTAIStrategy
from user_data.strategies.lib.rtai_indicators import add_all_rtai_indicators

def create_test_data():
    """Create realistic test data"""
    np.random.seed(42)
    n_points = 1000
    
    # Create base OHLCV data
    base_price = 65000
    price_changes = np.random.normal(0, 0.002, n_points)  # 0.2% std
    prices = [base_price]
    
    for change in price_changes:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    prices = prices[1:]  # Remove first element
    
    # Create OHLCV
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_points, freq='1min'),
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(100, 10000, n_points)
    })
    
    df.set_index('timestamp', inplace=True)
    return df

def test_signal_generation():
    """Test signal generation with relaxed conditions"""
    print("🧪 RTAI SIGNAL GENERATION TEST")
    print("=" * 50)
    
    # Create test data
    print("📊 Creating test data...")
    df = create_test_data()
    print(f"   Created {len(df)} data points")
    print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    
    # Add RTAI indicators
    print("🔬 Adding RTAI indicators...")
    df = add_all_rtai_indicators(df)
    
    # Print indicator stats
    rtai_indicators = ['ofi_z', 'mpd_z', 'vpin', 'lpi_z', 'momentum_z', 'volume_ratio']
    for indicator in rtai_indicators:
        if indicator in df.columns:
            values = df[indicator].dropna()
            print(f"   📊 {indicator}: {len(values)} values, range: {values.min():.6f} to {values.max():.6f}")
    
    # Create strategy with relaxed conditions
    print("🎯 Testing strategy with relaxed conditions...")
    
    # Mock config for strategy
    config = {
        'stake_currency': 'USDT',
        'stake_amount': 1000,
        'trading_mode': 'spot',
        'dry_run': True
    }
    
    strategy = RTAIStrategy(config)
    
    # Override hyperopt parameters with relaxed values
    strategy.ofi_threshold.value = 1.0  # Lower from default
    strategy.mpd_threshold.value = 0.001  # Lower from default  
    strategy.vpin_max.value = 0.8  # Higher from default
    strategy.lpi_max.value = 1000  # Higher from default
    strategy.volume_spike_threshold.value = 0.5  # Lower from default
    strategy.volatility_threshold.value = 0.5  # Lower from default
    strategy.trend_filter.value = False  # Disable strict filters
    strategy.market_regime_filter.value = False  # Disable strict filters
    
    print("🔧 Relaxed parameters:")
    print(f"   OFI threshold: {strategy.ofi_threshold.value}")
    print(f"   MPD threshold: {strategy.mpd_threshold.value}")
    print(f"   VPIN max: {strategy.vpin_max.value}")
    print(f"   LPI max: {strategy.lpi_max.value}")
    
    # Run strategy
    df_indicators = strategy.populate_indicators(df.copy(), {'pair': 'BTC/USDT'})
    df_entry = strategy.populate_entry_trend(df_indicators.copy(), {'pair': 'BTC/USDT'})
    df_exit = strategy.populate_exit_trend(df_entry.copy(), {'pair': 'BTC/USDT'})
    
    # Count signals
    long_entries = df_exit['enter_long'].sum() if 'enter_long' in df_exit.columns else 0
    short_entries = df_exit['enter_short'].sum() if 'enter_short' in df_exit.columns else 0
    long_exits = df_exit['exit_long'].sum() if 'exit_long' in df_exit.columns else 0
    short_exits = df_exit['exit_short'].sum() if 'exit_short' in df_exit.columns else 0
    
    print("📈 SIGNAL RESULTS:")
    print(f"   Long entries: {long_entries}")
    print(f"   Short entries: {short_entries}")
    print(f"   Long exits: {long_exits}")
    print(f"   Short exits: {short_exits}")
    print(f"   Total entry signals: {long_entries + short_entries}")
    
    if long_entries + short_entries > 0:
        print("✅ Signal generation working!")
        
        # Show sample of signals
        entry_signals = df_exit[df_exit['enter_long'] == 1].head(3)
        if len(entry_signals) > 0:
            print("\n📊 Sample long entry signals:")
            for idx, row in entry_signals.iterrows():
                print(f"   {idx}: Price=${row['close']:.2f}, OFI_Z={row['ofi_z']:.3f}, VPIN={row['vpin']:.6f}")
        
        entry_signals = df_exit[df_exit['enter_short'] == 1].head(3)
        if len(entry_signals) > 0:
            print("\n📊 Sample short entry signals:")
            for idx, row in entry_signals.iterrows():
                print(f"   {idx}: Price=${row['close']:.2f}, OFI_Z={row['ofi_z']:.3f}, VPIN={row['vpin']:.6f}")
    else:
        print("❌ No signals generated - conditions still too restrictive")
        
        # Debug conditions
        print("\n🔍 DEBUG ANALYSIS:")
        df_debug = df_exit.dropna()
        
        if len(df_debug) > 0:
            ofi_condition = (df_debug['ofi_z'].abs() > strategy.ofi_threshold.value).sum()
            mpd_condition = (df_debug['mpd_z'].abs() > strategy.mpd_threshold.value).sum()
            vpin_condition = (df_debug['vpin'] < strategy.vpin_max.value).sum()
            lpi_condition = (df_debug['lpi_z'].abs() < strategy.lpi_max.value).sum()
            
            print(f"   OFI condition met: {ofi_condition}/{len(df_debug)} ({ofi_condition/len(df_debug)*100:.1f}%)")
            print(f"   MPD condition met: {mpd_condition}/{len(df_debug)} ({mpd_condition/len(df_debug)*100:.1f}%)")
            print(f"   VPIN condition met: {vpin_condition}/{len(df_debug)} ({vpin_condition/len(df_debug)*100:.1f}%)")
            print(f"   LPI condition met: {lpi_condition}/{len(df_debug)} ({lpi_condition/len(df_debug)*100:.1f}%)")

if __name__ == "__main__":
    test_signal_generation()
