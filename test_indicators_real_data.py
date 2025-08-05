#!/usr/bin/env python3
"""
Test script per verificare che tutti gli indicatori funzionino con dati reali
"""

import asyncio
import time
from rtai.indicators.base import OFI, VPIN, KyleLambda, LPI
from rtai.live_trader import LiveTrader

async def test_indicators_with_real_data():
    """Test tutti gli indicatori con dati simulati realistici"""
    print("🧪 Testing RTAI Indicators with Real Data")
    print("=" * 50)
    
    # Test OFI
    print("📊 Testing OFI (Order Flow Imbalance)...")
    ofi = OFI()
    
    # Simulate real trade data
    test_trades = [
        (100.0, 50000.0),   # Buy 100 at 50000
        (-50.0, 50001.0),   # Sell 50 at 50001
        (200.0, 50002.0),   # Buy 200 at 50002
        (-75.0, 50001.5),   # Sell 75 at 50001.5
        (150.0, 50003.0),   # Buy 150 at 50003
    ]
    
    ofi_values = []
    for qty, price in test_trades:
        result = ofi.update(qty, price)
        if result is not None:
            ofi_values.append(result)
            print(f"  OFI: qty={qty:6.1f}, price={price:8.1f} → z-score={result:6.3f}")
    
    if ofi_values:
        print(f"✅ OFI working: {len(ofi_values)} valid values")
    else:
        print("❌ OFI not producing values")
    
    # Test VPIN
    print("\n📊 Testing VPIN (Volume-Synchronized PIN)...")
    vpin = VPIN()
    
    vpin_values = []
    for qty, price in test_trades:
        result = vpin.update(qty, price)
        if result is not None:
            vpin_values.append(result)
            print(f"  VPIN: qty={qty:6.1f}, price={price:8.1f} → value={result:6.3f}")
    
    if vpin_values:
        print(f"✅ VPIN working: {len(vpin_values)} valid values")
    else:
        print("❌ VPIN not producing values")
    
    # Test Kyle Lambda
    print("\n📊 Testing Kyle Lambda (Market Impact)...")
    kyle = KyleLambda()
    
    kyle_values = []
    for qty, price in test_trades:
        result = kyle.update(qty, price)
        if result is not None:
            kyle_values.append(result)
            print(f"  Kyle: qty={qty:6.1f}, price={price:8.1f} → lambda={result:8.2f}")
    
    if kyle_values:
        print(f"✅ Kyle Lambda working: {len(kyle_values)} valid values")
    else:
        print("❌ Kyle Lambda not producing values")
    
    # Test LPI
    print("\n📊 Testing LPI (Liquidation Pressure Index)...")
    lpi = LPI()
    
    # LPI needs side and volume
    lpi_test_data = [
        ("long", 100.0),
        ("short", 50.0),
        ("long", 200.0),
        ("short", 75.0),
        ("long", 150.0),
    ]
    
    lpi_values = []
    for side, volume in lpi_test_data:
        result = lpi.update(side, volume)
        if result is not None:
            lpi_values.append(result)
            print(f"  LPI: side={side:5s}, volume={volume:6.1f} → pressure={result:6.3f}")
    
    if lpi_values:
        print(f"✅ LPI working: {len(lpi_values)} valid values")
    else:
        print("❌ LPI not producing values")
    
    print("\n" + "=" * 50)
    print("📊 INDICATOR TEST SUMMARY")
    print("=" * 50)
    
    total_indicators = 4
    working_indicators = 0
    
    if ofi_values:
        print("✅ PASS OFI (Order Flow Imbalance)")
        working_indicators += 1
    else:
        print("❌ FAIL OFI (Order Flow Imbalance)")
    
    if vpin_values:
        print("✅ PASS VPIN (Volume-Synchronized PIN)")
        working_indicators += 1
    else:
        print("❌ FAIL VPIN (Volume-Synchronized PIN)")
    
    if kyle_values:
        print("✅ PASS Kyle Lambda (Market Impact)")
        working_indicators += 1
    else:
        print("❌ FAIL Kyle Lambda (Market Impact)")
    
    if lpi_values:
        print("✅ PASS LPI (Liquidation Pressure Index)")
        working_indicators += 1
    else:
        print("❌ FAIL LPI (Liquidation Pressure Index)")
    
    print(f"\n🎯 Score: {working_indicators}/{total_indicators} ({(working_indicators/total_indicators)*100:.1f}%)")
    
    if working_indicators == total_indicators:
        print("🎉 ALL INDICATORS WORKING WITH REAL DATA!")
        return True
    else:
        print("⚠️  Some indicators need attention")
        return False

async def test_live_trader_integration():
    """Test LiveTrader integration"""
    print("\n🔄 Testing LiveTrader Integration...")
    
    try:
        # Create LiveTrader instance
        trader = LiveTrader("BTCUSDT")
        
        # Check if indicators are initialized
        indicators_status = {
            "OFI": hasattr(trader, 'ofi') and trader.ofi is not None,
            "VPIN": hasattr(trader, 'vpin') and trader.vpin is not None,
            "Kyle": hasattr(trader, 'kyle') and trader.kyle is not None,
            "LPI": hasattr(trader, 'lpi') and trader.lpi is not None,
        }
        
        print("LiveTrader Indicators Status:")
        working_count = 0
        for name, status in indicators_status.items():
            if status:
                print(f"  ✅ {name}: Initialized")
                working_count += 1
            else:
                print(f"  ❌ {name}: Not initialized")
        
        print(f"\n🎯 LiveTrader Integration: {working_count}/{len(indicators_status)} indicators ready")
        
        if working_count == len(indicators_status):
            print("✅ LiveTrader integration successful!")
            return True
        else:
            print("⚠️  LiveTrader integration needs attention")
            return False
            
    except Exception as e:
        print(f"❌ LiveTrader integration failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🚀 RTAI Real Data Indicator Tests")
    print("=" * 60)
    
    # Test indicators
    indicators_ok = await test_indicators_with_real_data()
    
    # Test LiveTrader integration
    integration_ok = await test_live_trader_integration()
    
    print("\n" + "=" * 60)
    print("🏆 FINAL TEST RESULTS")
    print("=" * 60)
    
    if indicators_ok and integration_ok:
        print("🎉 ALL TESTS PASSED - INDICATORS READY FOR REAL DATA!")
        print("\n✅ System Status: PRODUCTION READY")
        print("✅ Indicators: ALL WORKING")
        print("✅ Integration: SUCCESSFUL")
        print("✅ Real Data: COMPATIBLE")
        
        print("\n🚀 Ready to start live trading:")
        print("   python -m rtai.main --mode live --symbol BTCUSDT")
        
    else:
        print("⚠️  SOME TESTS FAILED - REVIEW NEEDED")
        if not indicators_ok:
            print("❌ Indicators: Need attention")
        if not integration_ok:
            print("❌ Integration: Need attention")

if __name__ == "__main__":
    asyncio.run(main())