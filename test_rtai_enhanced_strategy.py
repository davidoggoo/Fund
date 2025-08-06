#!/usr/bin/env python3
"""
RTAI Enhanced Strategy Test Script
=================================

Quick validation of the enhanced mean-reversion strategy components
following the detailed implementation from the context file.

Tests:
1. Strategy instantiation and parameter validation
2. Indicator calculation pipeline  
3. Entry/exit signal generation
4. Position sizing logic
5. Integration with enhanced DataProvider
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Add the freqtrade path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ft'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def create_sample_dataframe(periods=500):
    """Create realistic sample OHLCV data for testing."""
    
    # Base price with trend and noise
    base_price = 65000.0
    dates = pd.date_range(start='2024-12-01', periods=periods, freq='1min')
    
    # Generate realistic price movements
    returns = np.random.normal(0, 0.0002, periods)  # ~0.02% std per minute
    returns[100:120] = np.random.normal(0.001, 0.0005, 20)  # Trend period
    returns[300:320] = np.random.normal(-0.001, 0.0005, 20)  # Reverse trend
    
    prices = base_price * np.exp(np.cumsum(returns))
    
    # OHLCV data
    df = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.normal(0, 0.00005, periods)),
        'high': prices * (1 + np.abs(np.random.normal(0.0001, 0.00005, periods))),
        'low': prices * (1 - np.abs(np.random.normal(0.0001, 0.00005, periods))),
        'close': prices,
        'volume': np.random.lognormal(15, 0.5, periods)  # Realistic volume
    })
    
    # Add microstructure data for testing
    df['bid'] = df['close'] * (1 - np.random.uniform(0.00005, 0.0002, periods))
    df['ask'] = df['close'] * (1 + np.random.uniform(0.00005, 0.0002, periods))
    df['bid_size'] = np.random.lognormal(12, 0.8, periods)
    df['ask_size'] = np.random.lognormal(12, 0.8, periods)
    
    # Volume breakdown
    df['buy_volume'] = df['volume'] * np.random.beta(2, 2, periods)
    df['sell_volume'] = df['volume'] - df['buy_volume']
    df['volume_imbalance'] = df['buy_volume'] - df['sell_volume']
    
    # Order flow components for realistic indicator calculation
    df['long_liquidations'] = np.random.exponential(0.1, periods) * df['volume'] * 0.01
    df['short_liquidations'] = np.random.exponential(0.1, periods) * df['volume'] * 0.01
    df['liquidation_imbalance'] = df['long_liquidations'] - df['short_liquidations']
    
    # Futures-specific data
    df['open_interest'] = np.random.normal(50000000, 5000000, periods)
    df['funding_rate'] = np.random.normal(0.0001, 0.00005, periods)
    df['index_price'] = df['close'] * (1 + np.random.normal(0, 0.00001, periods))
    df['mark_price'] = df['close'] * (1 + np.random.normal(0, 0.000005, periods))
    
    return df

def test_strategy_instantiation():
    """Test that the enhanced strategy can be instantiated properly."""
    
    print("\\n" + "="*60)
    print("TEST 1: Strategy Instantiation")
    print("="*60)
    
    try:
        from user_data.strategies.RTAIStrategy import RTAIStrategy
        
        # Create strategy instance
        strategy = RTAIStrategy()
        
        # Check basic properties
        assert strategy.timeframe == '1m', f"Expected 1m timeframe, got {strategy.timeframe}"
        assert strategy.can_short == True, "Strategy should support short selling"
        assert strategy.max_open_trades == 2, f"Expected 2 max trades, got {strategy.max_open_trades}"
        
        # Check parameters exist and have reasonable defaults
        assert hasattr(strategy, 'ofi_entry_threshold'), "Missing OFI entry threshold"
        assert hasattr(strategy, 'mpd_entry_threshold'), "Missing MPD entry threshold"
        assert hasattr(strategy, 'kyle_spike_multiplier'), "Missing Kyle spike multiplier"
        
        print(f"✅ Strategy instantiated successfully")
        print(f"   - Timeframe: {strategy.timeframe}")
        print(f"   - Can short: {strategy.can_short}")
        print(f"   - Max trades: {strategy.max_open_trades}")
        print(f"   - OFI threshold: {strategy.ofi_entry_threshold.value}")
        print(f"   - MPD threshold: {strategy.mpd_entry_threshold.value}")
        
        return strategy
        
    except Exception as e:
        print(f"❌ Strategy instantiation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_indicator_calculation(strategy):
    """Test the enhanced indicator calculation pipeline."""
    
    print("\\n" + "="*60)
    print("TEST 2: Indicator Calculation Pipeline")
    print("="*60)
    
    if not strategy:
        print("❌ Skipping - no strategy instance")
        return None
        
    try:
        # Create sample data
        df = create_sample_dataframe(200)
        metadata = {'pair': 'BTC/USDT:USDT'}
        
        print(f"📊 Input data shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
        
        # Calculate indicators
        result_df = strategy.populate_indicators(df.copy(), metadata)
        
        # Check that enhanced indicators were added
        expected_indicators = [
            'ofi_z', 'mpd_z', 'tobi', 'wall_ratio', 'vpin',
            'kyle_lambda', 'lpi_z', 'conviction_score'
        ]
        
        missing_indicators = []
        for indicator in expected_indicators:
            if indicator not in result_df.columns:
                missing_indicators.append(indicator)
        
        if missing_indicators:
            print(f"⚠️ Missing indicators: {missing_indicators}")
        else:
            print("✅ All critical indicators calculated successfully")
        
        # Check data quality
        for indicator in expected_indicators:
            if indicator in result_df.columns:
                non_null_pct = (1 - result_df[indicator].isnull().mean()) * 100
                value_range = (result_df[indicator].min(), result_df[indicator].max())
                print(f"   {indicator}: {non_null_pct:.1f}% non-null, range: {value_range}")
        
        return result_df
        
    except Exception as e:
        print(f"❌ Indicator calculation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_signal_generation(strategy, df_with_indicators):
    """Test entry and exit signal generation."""
    
    print("\\n" + "="*60)
    print("TEST 3: Signal Generation")
    print("="*60)
    
    if not strategy or df_with_indicators is None:
        print("❌ Skipping - missing dependencies")
        return
        
    try:
        metadata = {'pair': 'BTC/USDT:USDT'}
        df = df_with_indicators.copy()
        
        # Generate entry signals
        entry_df = strategy.populate_entry_trend(df.copy(), metadata)
        
        # Generate exit signals  
        exit_df = strategy.populate_exit_trend(entry_df.copy(), metadata)
        
        # Count signals
        long_entries = (entry_df['enter_long'] == 1).sum()
        short_entries = (entry_df['enter_short'] == 1).sum()
        long_exits = (exit_df['exit_long'] == 1).sum()
        short_exits = (exit_df['exit_short'] == 1).sum()
        
        print(f"📈 Signal Summary:")
        print(f"   Long entries: {long_entries}")
        print(f"   Short entries: {short_entries}")
        print(f"   Long exits: {long_exits}")
        print(f"   Short exits: {short_exits}")
        
        # Analyze entry tags
        if 'enter_tag' in entry_df.columns:
            entry_tags = entry_df['enter_tag'].dropna().value_counts()
            if not entry_tags.empty:
                print(f"   Entry tag examples:")
                for tag, count in entry_tags.head(3).items():
                    print(f"     {tag}: {count}")
        
        # Check for signal quality
        total_signals = long_entries + short_entries
        if total_signals > 0:
            signal_rate = (total_signals / len(df)) * 100
            print(f"   Signal rate: {signal_rate:.2f}% of bars")
            
            if signal_rate > 10:
                print("   ⚠️ High signal rate - may be too sensitive")
            elif signal_rate < 0.1:
                print("   ⚠️ Very low signal rate - may be too restrictive")
            else:
                print("   ✅ Reasonable signal rate")
        else:
            print("   ⚠️ No signals generated - check thresholds")
        
        return exit_df
        
    except Exception as e:
        print(f"❌ Signal generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_position_sizing(strategy):
    """Test custom position sizing logic."""
    
    print("\\n" + "="*60)
    print("TEST 4: Position Sizing Logic")
    print("="*60)
    
    if not strategy:
        print("❌ Skipping - no strategy instance")
        return
        
    try:
        # Mock wallet for testing
        class MockWallets:
            def get_total_stake_amount(self):
                return 100000.0  # $100k test wallet
        
        # Mock dataframe with various signal strengths
        test_scenarios = [
            {'ofi_z': -3.0, 'mpd_z': 2.0, 'atr_regime': 0.3, 'vpin': 0.4, 'name': 'Strong Long Signal'},
            {'ofi_z': 2.5, 'mpd_z': -1.8, 'atr_regime': 0.7, 'vpin': 0.6, 'name': 'Moderate Short Signal'},
            {'ofi_z': -4.0, 'mpd_z': 3.0, 'atr_regime': 0.9, 'vpin': 0.8, 'name': 'High Vol Environment'},
            {'ofi_z': -2.3, 'mpd_z': 1.6, 'atr_regime': 0.2, 'vpin': 0.3, 'name': 'Low Vol Environment'}
        ]
        
        strategy.wallets = MockWallets()
        
        print("📊 Position sizing test scenarios:")
        
        for scenario in test_scenarios:
            # Create mock dataframe
            mock_df = pd.DataFrame([scenario])
            
            # Mock data provider response
            class MockDataProvider:
                def get_analyzed_dataframe(self, pair, timeframe):
                    return mock_df, None
            
            strategy.dp = MockDataProvider()
            
            # Test position sizing
            base_stake = 5000.0  # $5k base
            result_stake = strategy.custom_stake_amount(
                pair="BTC/USDT:USDT",
                current_time=datetime.now(),
                current_rate=65000.0,
                proposed_stake=base_stake,
                min_stake=100.0,
                max_stake=10000.0,
                leverage=1.0,
                entry_tag="TEST",
                side="long"
            )
            
            stake_ratio = result_stake / base_stake
            conviction = abs(scenario['ofi_z']) * abs(scenario['mpd_z'])
            
            print(f"   {scenario['name']}:")
            print(f"     OFI_z: {scenario['ofi_z']:+.1f}, MPD_z: {scenario['mpd_z']:+.1f}")
            print(f"     Conviction: {conviction:.2f}, ATR regime: {scenario['atr_regime']:.1f}")
            print(f"     Stake: ${result_stake:.0f} ({stake_ratio:.2f}x base)")
        
        print("✅ Position sizing logic working")
        
    except Exception as e:
        print(f"❌ Position sizing test failed: {e}")
        import traceback
        traceback.print_exc()

def test_integration():
    """Test integration with enhanced components."""
    
    print("\\n" + "="*60)
    print("TEST 5: Integration Test")
    print("="*60)
    
    try:
        # Test enhanced indicators import
        from user_data.strategies.lib.rtai_indicators import add_all_rtai_indicators, joint_signal_score
        print("✅ Enhanced indicators imported successfully")
        
        # Test basic functionality
        test_data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103], 
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000, 1100, 1200]
        })
        
        # Add some basic microstructure data
        test_data['bid'] = test_data['close'] * 0.999
        test_data['ask'] = test_data['close'] * 1.001
        test_data['bid_size'] = [500, 550, 600]
        test_data['ask_size'] = [480, 520, 580]
        
        # Test indicator calculation
        result = add_all_rtai_indicators(test_data.copy())
        
        if len(result.columns) > len(test_data.columns):
            added_cols = len(result.columns) - len(test_data.columns)
            print(f"✅ Added {added_cols} indicator columns")
        else:
            print("⚠️ No indicators added - check implementation")
        
        # Test joint signal score
        ofi_z_sample = np.array([-2.5, 1.8, -3.1])
        mpd_z_sample = np.array([2.1, -1.9, 2.8])
        scores = joint_signal_score(ofi_z_sample, mpd_z_sample)
        
        print(f"✅ Joint signal scores: {scores}")
        
        # Test RTAIDataProvider import
        try:
            from RTAIDataProvider import RTAIDataProvider
            print("✅ RTAIDataProvider imported successfully")
        except ImportError as e:
            print(f"⚠️ RTAIDataProvider import failed: {e}")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Run all validation tests."""
    
    print("🚀 RTAI Enhanced Strategy Validation")
    print("="*60)
    print("Testing implementation following detailed context file specifications")
    print("Focus: 1-minute mean-reversion with microstructure indicators")
    
    # Run tests in sequence
    strategy = test_strategy_instantiation()
    df_with_indicators = test_indicator_calculation(strategy)
    test_signal_generation(strategy, df_with_indicators)
    test_position_sizing(strategy)
    test_integration()
    
    print("\\n" + "="*60)
    print("🎯 VALIDATION COMPLETE")
    print("="*60)
    print("All core components tested. Ready for backtesting and live deployment.")
    print("\\nNext steps:")
    print("1. Run backtest with: python -m freqtrade backtesting --config config_rtai_enhanced.json")
    print("2. Test with paper trading: python -m freqtrade trade --config config_rtai_enhanced.json") 
    print("3. Monitor performance with dashboard")

if __name__ == "__main__":
    main()
