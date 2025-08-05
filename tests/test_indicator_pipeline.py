#!/usr/bin/env python3
"""
Test suite for RTAI indicator pipeline
Tests OFI, VPIN, KyleLambda, and LPI indicators with real data conditions
"""

import time
import numpy as np
from rtai.indicators.base import OFI, VPIN, KyleLambda, LPI

class TestIndicatorPipeline:
    """Test the core indicator pipeline"""
    
    def test_ofi_basic_functionality(self):
        """Test OFI indicator basic functionality"""
        ofi = OFI(alpha=0.15, mad_alpha=0.3)
        
        # Generate some test trades (signed quantities)
        test_trades = [
            (1.5, 50000.0),   # Buy 1.5 BTC at 50k
            (-2.0, 50100.0),  # Sell 2.0 BTC at 50.1k  
            (0.8, 50050.0),   # Buy 0.8 BTC at 50.05k
            (-1.2, 49950.0),  # Sell 1.2 BTC at 49.95k
            (2.5, 50200.0),   # Buy 2.5 BTC at 50.2k (large order)
        ]
        
        results = []
        for qty_signed, price in test_trades:
            result = ofi.update(qty_signed, price)
            results.append(result)
            time.sleep(0.01)  # Small delay to avoid timestamp issues
        
        # Check that we get some non-None results (after initialization)
        non_none_results = [r for r in results if r is not None]
        assert len(non_none_results) > 0, "OFI should produce some values"
        
        # Check that values are reasonable z-scores (typically -5 to +5)
        for result in non_none_results:
            assert -10 < result < 10, f"OFI z-score out of range: {result}"
        
        print(f"✅ OFI test passed: {len(non_none_results)} values generated")
    
    def test_vpin_basic_functionality(self):
        """Test VPIN indicator basic functionality"""
        vpin = VPIN(base_bucket_usdt=50.0, win_buckets=12)
        
        # Generate test trades with volume to fill buckets
        test_trades = [
            (1.0, 50000.0),    # Buy 1 BTC
            (-1.5, 50100.0),   # Sell 1.5 BTC  
            (2.0, 50050.0),    # Buy 2 BTC
            (-0.5, 49950.0),   # Sell 0.5 BTC
            (1.8, 50200.0),    # Buy 1.8 BTC
            (-2.2, 50150.0),   # Sell 2.2 BTC
        ]
        
        results = []
        for qty_signed, price in test_trades:
            result = vpin.update(qty_signed, price)
            results.append(result)
        
        # VPIN values should be between 0 and 1
        for result in [r for r in results if r is not None]:
            assert 0 <= result <= 1, f"VPIN out of range [0,1]: {result}"
        
        print(f"✅ VPIN test passed: valid range checks")
    
    def test_kyle_lambda_basic_functionality(self):
        """Test Kyle's Lambda indicator basic functionality"""
        kyle = KyleLambda(window=120)
        
        # Generate test trades with price impact
        base_price = 50000.0
        test_trades = [
            (1.0, base_price),
            (2.0, base_price + 10),     # Price moves up with buy
            (-1.5, base_price + 5),     # Price moves down with sell
            (0.8, base_price + 15),     # Price continues up
            (-2.5, base_price - 20),    # Large sell, price drops
        ]
        
        results = []
        for qty_signed, price in test_trades:
            result = kyle.update(qty_signed, price)
            results.append(result)
            time.sleep(0.01)
        
        # Kyle's lambda should be non-negative
        for result in [r for r in results if r is not None]:
            assert result >= 0, f"Kyle's Lambda should be non-negative: {result}"
        
        print(f"✅ Kyle's Lambda test passed: non-negative values")
    
    def test_lpi_with_oi_estimate(self):
        """Test LPI indicator with OI estimate"""
        lpi = LPI(window_seconds=60)
        
        # CRITICAL: Set OI estimate first (this was the missing piece!)
        lpi.update_oi_estimate(1000000.0)  # 1M USD OI estimate
        
        # Generate liquidation events
        liquidations = [
            (100.0, 0.0),      # 100 long liquidated
            (0.0, 150.0),      # 150 short liquidated
            (50.0, 0.0),       # 50 long liquidated
            (0.0, 200.0),      # 200 short liquidated (net short pressure)
            (300.0, 0.0),      # 300 long liquidated (big long liq)
        ]
        
        results = []
        for long_qty, short_qty in liquidations:
            result = lpi.update(long_qty, short_qty)
            results.append(result)
            time.sleep(0.1)  # Spacing for decay calculation
        
        # LPI should produce some non-None results
        non_none_results = [r for r in results if r is not None]
        assert len(non_none_results) > 0, "LPI should produce values with OI estimate"
        
        # LPI values should be in reasonable range [-2, +2] (tanh bounded)
        for result in non_none_results:
            assert -3 < result < 3, f"LPI out of expected range: {result}"
        
        print(f"✅ LPI test passed: {len(non_none_results)} values with OI estimate")
    
    def test_all_indicators_integration(self):
        """Test all indicators working together"""
        # Initialize all indicators
        ofi = OFI()
        vpin = VPIN()
        kyle = KyleLambda()
        lpi = LPI()
        
        # Set LPI OI estimate
        lpi.update_oi_estimate(500000.0)
        
        # Simulate a trading sequence
        trades = [
            (1.2, 50000.0),
            (-0.8, 50050.0),
            (2.1, 50100.0),
            (-1.5, 50080.0),
            (0.9, 50120.0),
        ]
        
        all_results = []
        
        for i, (qty_signed, price) in enumerate(trades):
            # Update all indicators
            ofi_val = ofi.update(qty_signed, price)
            vpin_val = vpin.update(qty_signed, price)
            kyle_val = kyle.update(qty_signed, price)
            
            # Add some liquidations for LPI
            if i > 0:  # Skip first trade
                liq_long = abs(qty_signed) * 10 if qty_signed < 0 else 0
                liq_short = abs(qty_signed) * 10 if qty_signed > 0 else 0
                lpi_val = lpi.update(liq_long, liq_short)
            else:
                lpi_val = None
            
            all_results.append({
                'ofi': ofi_val,
                'vpin': vpin_val, 
                'kyle': kyle_val,
                'lpi': lpi_val
            })
            
            time.sleep(0.05)
        
        # Check that at least some indicators produced values
        total_values = sum(
            1 for result in all_results 
            for val in result.values() 
            if val is not None
        )
        
        assert total_values > 0, "At least some indicators should produce values"
        print(f"✅ Integration test passed: {total_values} total indicator values")
        
        # Print summary for debugging
        for i, result in enumerate(all_results):
            ofi_str = f"{result['ofi']:.3f}" if result['ofi'] is not None else 'None'
            vpin_str = f"{result['vpin']:.3f}" if result['vpin'] is not None else 'None'
            kyle_str = f"{result['kyle']:.3f}" if result['kyle'] is not None else 'None'
            lpi_str = f"{result['lpi']:.3f}" if result['lpi'] is not None else 'None'
            
            print(f"   Trade {i+1}: OFI={ofi_str} VPIN={vpin_str} Kyle={kyle_str} LPI={lpi_str}")

def test_ofi_vpin_kyle_lpi():
    """Main test function that can be called directly"""
    test_suite = TestIndicatorPipeline()
    
    print("🧪 Running indicator pipeline tests...")
    
    test_suite.test_ofi_basic_functionality()
    test_suite.test_vpin_basic_functionality() 
    test_suite.test_kyle_lambda_basic_functionality()
    test_suite.test_lpi_with_oi_estimate()
    test_suite.test_all_indicators_integration()
    
    print("🎉 All indicator tests passed!")

if __name__ == "__main__":
    test_ofi_vpin_kyle_lpi()
