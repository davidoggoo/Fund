#!/usr/bin/env python3
"""
Test RTAIStrategy (Production-Grade Unified)
Test the production strategy to verify signal generation
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

def test_production_strategy():
    """Test production strategy signal generation"""
    
    print("🏭 RTAI PRODUCTION STRATEGY TEST")
    print("=" * 50)
    
    # Create realistic test data
    np.random.seed(42)
    n_points = 2000
    base_price = 65000
    
    # More realistic price action with trends and reversals
    price_changes = []
    trend = 0.0001  # Small uptrend
    
    for i in range(n_points):
        # Add trend reversal every 200 candles
        if i % 200 == 0:
            trend = -trend
        
        # Random walk with trend
        change = np.random.normal(trend, 0.003)  # 0.3% volatility
        price_changes.append(change)
    
    prices = [base_price]
    for change in price_changes:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    prices = prices[1:]
    
    # Create OHLCV with realistic intrabar movement
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_points, freq='1min'),
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.0015))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.0015))) for p in prices],
        'close': prices,
        'volume': np.random.lognormal(8, 1, n_points)  # More realistic volume distribution
    })
    
    df.set_index('timestamp', inplace=True)
    
    print(f"📊 Created {len(df)} data points")
    print(f"   Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"   Volume range: {df['volume'].min():.0f} - {df['volume'].max():.0f}")
    
    # Create and test strategy
    config = {
        'stake_currency': 'USDT',
        'stake_amount': 1000,
        'trading_mode': 'spot',
        'dry_run': True
    }
    
    strategy = RTAIStrategy(config)
    
    print(f"🎯 Strategy: {strategy.__class__.__name__}")
    print(f"   Timeframe: {strategy.timeframe}")
    print(f"   Can short: {strategy.can_short}")
    print(f"   ROI: {strategy.minimal_roi}")
    print(f"   Stoploss: {strategy.stoploss}")
    
    # Run full strategy pipeline
    print("\n🔄 Running strategy pipeline...")
    
    try:
        # Populate indicators
        df_indicators = strategy.populate_indicators(df.copy(), {'pair': 'BTC/USDT'})
        print("✅ Indicators populated successfully")
        
        # Count RTAI indicators
        rtai_cols = [col for col in df_indicators.columns if col in [
            'ofi', 'ofi_z', 'vpin', 'mpd', 'mpd_z', 'lpi', 'lpi_z', 
            'momentum', 'momentum_z', 'volume_ratio', 'atr', 'bid', 'ask'
        ]]
        print(f"   RTAI indicators: {len(rtai_cols)} ({', '.join(rtai_cols[:5])}...)")
        
        # Populate entry signals
        df_entry = strategy.populate_entry_trend(df_indicators.copy(), {'pair': 'BTC/USDT'})
        print("✅ Entry signals populated successfully")
        
        # Populate exit signals
        df_final = strategy.populate_exit_trend(df_entry.copy(), {'pair': 'BTC/USDT'})
        print("✅ Exit signals populated successfully")
        
        # Count signals
        long_entries = df_final['enter_long'].sum() if 'enter_long' in df_final.columns else 0
        short_entries = df_final['enter_short'].sum() if 'enter_short' in df_final.columns else 0
        long_exits = df_final['exit_long'].sum() if 'exit_long' in df_final.columns else 0
        short_exits = df_final['exit_short'].sum() if 'exit_short' in df_final.columns else 0
        
        print(f"\n📈 SIGNAL RESULTS:")
        print(f"   Long entries: {long_entries} ({long_entries/len(df_final)*100:.2f}%)")
        print(f"   Short entries: {short_entries} ({short_entries/len(df_final)*100:.2f}%)")
        print(f"   Long exits: {long_exits} ({long_exits/len(df_final)*100:.2f}%)")
        print(f"   Short exits: {short_exits} ({short_exits/len(df_final)*100:.2f}%)")
        print(f"   Total entry signals: {long_entries + short_entries}")
        
        if long_entries + short_entries > 0:
            print("\n✅ PRODUCTION STRATEGY SUCCESSFUL!")
            
            # Show sample signals with context
            entry_samples = df_final[df_final['enter_long'] == 1].head(3)
            if len(entry_samples) > 0:
                print("\n📊 Sample LONG entry signals:")
                for idx, row in entry_samples.iterrows():
                    ofi_z = row.get('ofi_z', 0)
                    vpin = row.get('vpin', 0)
                    rsi = row.get('rsi', 50)
                    volume_ratio = row.get('volume_ratio', 1)
                    print(f"   {idx.strftime('%H:%M')}: Price=${row['close']:.2f}, OFI_Z={ofi_z:.2f}, VPIN={vpin:.4f}, RSI={rsi:.1f}, Vol={volume_ratio:.2f}")
            
            entry_samples = df_final[df_final['enter_short'] == 1].head(3)
            if len(entry_samples) > 0:
                print("\n📊 Sample SHORT entry signals:")
                for idx, row in entry_samples.iterrows():
                    ofi_z = row.get('ofi_z', 0)
                    vpin = row.get('vpin', 0)
                    rsi = row.get('rsi', 50)
                    volume_ratio = row.get('volume_ratio', 1)
                    print(f"   {idx.strftime('%H:%M')}: Price=${row['close']:.2f}, OFI_Z={ofi_z:.2f}, VPIN={vpin:.4f}, RSI={rsi:.1f}, Vol={volume_ratio:.2f}")
            
            # Signal distribution analysis
            print(f"\n📊 SIGNAL DISTRIBUTION:")
            entry_hours = df_final[df_final['enter_long'] == 1].index.hour.value_counts().head(3)
            if len(entry_hours) > 0:
                print(f"   Most active hours: {dict(entry_hours)}")
            
            # Parameter analysis
            print(f"\n🔧 STRATEGY PARAMETERS:")
            print(f"   OFI threshold: {strategy.ofi_threshold.value}")
            print(f"   MPD threshold: {strategy.mpd_threshold.value}")
            print(f"   VPIN max: {strategy.vpin_max.value}")
            print(f"   Volume spike threshold: {strategy.volume_spike_threshold.value}")
            print(f"   Momentum threshold: {strategy.momentum_threshold.value}")
            
        else:
            print("\n❌ No signals generated")
            
            # Debug the conditions
            df_debug = df_final.dropna()
            if len(df_debug) > 0:
                ofi_bearish = (df_debug['ofi_z'] < -strategy.ofi_threshold.value).sum()
                ofi_bullish = (df_debug['ofi_z'] > strategy.ofi_threshold.value).sum()
                vpin_low = (df_debug['vpin'] < strategy.vpin_max.value).sum()
                mpd_active = (df_debug['mpd_z'].abs() > strategy.mpd_threshold.value).sum()
                
                print(f"\n🔍 DEBUG ANALYSIS:")
                print(f"   OFI bearish (< -{strategy.ofi_threshold.value}): {ofi_bearish}/{len(df_debug)} ({ofi_bearish/len(df_debug)*100:.1f}%)")
                print(f"   OFI bullish (> {strategy.ofi_threshold.value}): {ofi_bullish}/{len(df_debug)} ({ofi_bullish/len(df_debug)*100:.1f}%)")
                print(f"   VPIN low (< {strategy.vpin_max.value}): {vpin_low}/{len(df_debug)} ({vpin_low/len(df_debug)*100:.1f}%)")
                print(f"   MPD active (|z| > {strategy.mpd_threshold.value}): {mpd_active}/{len(df_debug)} ({mpd_active/len(df_debug)*100:.1f}%)")
                
    except Exception as e:
        print(f"❌ Strategy testing failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_production_strategy()
