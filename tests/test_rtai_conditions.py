#!/usr/bin/env python3
"""
RTAI Condition Analysis
Analyze each condition individually to find the bottleneck
"""

import sys
import os
import pandas as pd
import numpy as np

# Add paths
sys.path.append('C:/Users/Davidoggo/Desktop/Fund/ft')
sys.path.append('C:/Users/Davidoggo/Desktop/Fund/strategies')
sys.path.append('C:/Users/Davidoggo/Desktop/Fund')

from strategies.RTAIStrategy import RTAIStrategy
from strategies.lib.rtai_indicators import add_all_rtai_indicators

def analyze_conditions():
    """Analyze each condition individually"""
    
    # Create test data (same as before)
    np.random.seed(42)
    n_points = 1000
    base_price = 65000
    price_changes = np.random.normal(0, 0.002, n_points)
    prices = [base_price]
    
    for change in price_changes:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    prices = prices[1:]
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_points, freq='1min'),
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(100, 10000, n_points)
    })
    
    df.set_index('timestamp', inplace=True)
    
    # Add RTAI indicators
    df = add_all_rtai_indicators(df)
    
    # Mock config and strategy
    config = {'stake_currency': 'USDT', 'stake_amount': 1000, 'trading_mode': 'spot', 'dry_run': True}
    strategy = RTAIStrategy(config)
    
    # Use strategy defaults - parameters are optimized
    print("🔧 Using strategy default parameters")
    
    # Add indicators from strategy
    df = strategy.populate_indicators(df.copy(), {'pair': 'BTC/USDT'})
    
    # Clean data
    df_clean = df.dropna()
    total_rows = len(df_clean)
    
    print(f"🔍 CONDITION ANALYSIS ({total_rows} valid rows)")
    print("=" * 60)
    
    # Individual conditions for LONG
    print("📈 LONG CONDITIONS:")
    
    # Core conditions
    ofi_abs = (df_clean['ofi_z'].abs() > strategy.ofi_entry_threshold.value).sum()
    print(f"   OFI |z| > {strategy.ofi_entry_threshold.value}: {ofi_abs}/{total_rows} ({ofi_abs/total_rows*100:.1f}%)")
    
    mpd_abs = (df_clean['mpd_z'].abs() > strategy.mpd_entry_threshold.value).sum()
    print(f"   MPD |z| > {strategy.mpd_entry_threshold.value}: {mpd_abs}/{total_rows} ({mpd_abs/total_rows*100:.1f}%)")
    
    # Mean reversion condition (the killer?)
    sign_diff = (np.sign(df_clean['ofi_z']) != np.sign(df_clean['mpd_z'])).sum()
    print(f"   Mean reversion (signs differ): {sign_diff}/{total_rows} ({sign_diff/total_rows*100:.1f}%)")
    
    # Risk filters - check if columns exist first
    if 'tobi' in df_clean.columns:
        tobi_cond = ((df_clean['tobi'] >= strategy.tobi_min.value) & (df_clean['tobi'] <= strategy.tobi_max.value)).sum()
        print(f"   TOBI in [{strategy.tobi_min.value}, {strategy.tobi_max.value}]: {tobi_cond}/{total_rows} ({tobi_cond/total_rows*100:.1f}%)")
    else:
        print(f"   TOBI: column not available")
    
    if 'vpin' in df_clean.columns:
        # Calculate VPIN percentile threshold
        vpin_threshold = df_clean['vpin'].quantile(strategy.vpin_percentile.value)
        vpin_cond = (df_clean['vpin'] < vpin_threshold).sum()
        print(f"   VPIN < p{strategy.vpin_percentile.value*100:.0f} ({vpin_threshold:.3f}): {vpin_cond}/{total_rows} ({vpin_cond/total_rows*100:.1f}%)")
    else:
        print(f"   VPIN: column not available")
    
    if 'lpi_z' in df_clean.columns:
        lpi_cond = (df_clean['lpi_z'].abs() < strategy.lpi_exit_threshold.value).sum()
        print(f"   LPI |z| < {strategy.lpi_exit_threshold.value}: {lpi_cond}/{total_rows} ({lpi_cond/total_rows*100:.1f}%)")
    else:
        print(f"   LPI: column not available")
    
    ofi_bearish = (df_clean['ofi_z'] < -strategy.ofi_entry_threshold.value).sum()
    print(f"   OFI bearish < -{strategy.ofi_entry_threshold.value}: {ofi_bearish}/{total_rows} ({ofi_bearish/total_rows*100:.1f}%)")
    
    # Combined core condition (simplified to available columns)
    core_long = (
        (df_clean['ofi_z'].abs() > strategy.ofi_entry_threshold.value) &
        (df_clean['mpd_z'].abs() > strategy.mpd_entry_threshold.value) &
        (np.sign(df_clean['ofi_z']) != np.sign(df_clean['mpd_z'])) &
        (df_clean['ofi_z'] < -strategy.ofi_entry_threshold.value)
    ).sum()
    print(f"   ✅ CORE LONG (simplified): {core_long}/{total_rows} ({core_long/total_rows*100:.1f}%)")
    
    # Additional filters
    print("\n🔧 ADDITIONAL FILTERS:")
    if 'volume_ratio' in df_clean.columns:
        volume_ratio_cond = (df_clean['volume_ratio'] > 1.2).sum()  # Use fixed threshold for analysis
        print(f"   Volume spike > 1.2: {volume_ratio_cond}/{total_rows} ({volume_ratio_cond/total_rows*100:.1f}%)")
    else:
        print(f"   Volume ratio: column not available")
    
    # Print distribution analysis
    print("\n📊 DISTRIBUTION ANALYSIS:")
    print(f"   OFI_Z: min={df_clean['ofi_z'].min():.3f}, max={df_clean['ofi_z'].max():.3f}, std={df_clean['ofi_z'].std():.3f}")
    print(f"   MPD_Z: min={df_clean['mpd_z'].min():.6f}, max={df_clean['mpd_z'].max():.6f}, std={df_clean['mpd_z'].std():.6f}")
    
    # Sign analysis
    ofi_positive = (df_clean['ofi_z'] > 0).sum()
    ofi_negative = (df_clean['ofi_z'] < 0).sum()
    mpd_positive = (df_clean['mpd_z'] > 0).sum()
    mpd_negative = (df_clean['mpd_z'] < 0).sum()
    
    print(f"   OFI signs: +{ofi_positive}, -{ofi_negative}")
    print(f"   MPD signs: +{mpd_positive}, -{mpd_negative}")
    
    # Test with even more relaxed mean reversion
    print("\n🚀 TESTING RELAXED MEAN REVERSION:")
    
    # Just require OFI and MPD to have some threshold without sign requirement
    relaxed_core = (
        (df_clean['ofi_z'].abs() > 0.3) &
        (df_clean['mpd_z'].abs() > 0.00001) &
        (df_clean['ofi_z'] < -0.3)  # Just bearish OFI for long
    ).sum()
    print(f"   Relaxed core (no sign requirement): {relaxed_core}/{total_rows} ({relaxed_core/total_rows*100:.1f}%)")
    
    # Even simpler - just OFI based
    simple_long = (df_clean['ofi_z'] < -0.5).sum()
    print(f"   Simple OFI < -0.5: {simple_long}/{total_rows} ({simple_long/total_rows*100:.1f}%)")
    
    simple_short = (df_clean['ofi_z'] > 0.5).sum()
    print(f"   Simple OFI > 0.5: {simple_short}/{total_rows} ({simple_short/total_rows*100:.1f}%)")
    
    # Show available columns
    print(f"\n📋 Available columns: {sorted([col for col in df_clean.columns if 'ofi' in col or 'mpd' in col or 'vpin' in col or 'lpi' in col or 'tobi' in col])}")

if __name__ == "__main__":
    analyze_conditions()
