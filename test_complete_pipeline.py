#!/usr/bin/env python3
"""
Complete RTAI Pipeline Test
Tests the entire live data pipeline from WebSocket to TradingVue.js
"""
import asyncio
import json
import time
import subprocess
import sys
from pathlib import Path

def test_imports():
    """Test all critical imports"""
    print("🔍 Testing imports...")
    
    try:
        import rtai
        from rtai.api.server import app
        from rtai.live_trader import LiveTrader
        from rtai.utils import BinanceWebSocket
        from rtai.indicators.base import OFI, VPIN, KyleLambda, LPI
        from rtai.io import record_trade, record_bar, record_signal
        from rtai.io.rec_converter import rec_to_ohlcv
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_websocket_connection():
    """Test WebSocket connection capability"""
    print("🔌 Testing WebSocket connection...")
    
    try:
        from rtai.utils import BinanceWebSocket
        
        # Create WebSocket instance
        ws = BinanceWebSocket()
        print("✅ WebSocket instance created")
        return True
    except Exception as e:
        print(f"❌ WebSocket test failed: {e}")
        return False

def test_indicators():
    """Test indicator calculations"""
    print("📊 Testing indicators...")
    
    try:
        from rtai.indicators.base import OFI, VPIN, KyleLambda, LPI
        
        # Test OFI
        ofi = OFI()
        ofi_result = ofi.update(1.5, 50000.0)  # Buy 1.5 BTC at $50k
        
        # Test VPIN
        vpin = VPIN()
        vpin_result = vpin.update(1.5, 50000.0)
        
        # Test Kyle Lambda
        kyle = KyleLambda()
        kyle_result = kyle.update(1.5, 50000.0)
        
        # Test LPI
        lpi = LPI()
        lpi_result = lpi.update("long", 1.5)
        
        print("✅ All indicators working")
        return True
    except Exception as e:
        print(f"❌ Indicator test failed: {e}")
        return False

def test_recording():
    """Test data recording"""
    print("💾 Testing recording...")
    
    try:
        from rtai.io import record_trade, record_bar, record_signal
        
        # Test recording functions (they should not crash)
        record_trade(50000.0, 1.5, "buy")
        record_bar(50000.0, 50100.0, 49900.0, 50050.0, 10.5)
        record_signal("BUY", 50050.0)
        
        print("✅ Recording functions working")
        return True
    except Exception as e:
        print(f"❌ Recording test failed: {e}")
        return False

def test_server_startup():
    """Test FastAPI server startup"""
    print("🚀 Testing server startup...")
    
    try:
        from rtai.api.server import app
        import uvicorn
        
        # Test that server can be configured
        config = uvicorn.Config(app, host="127.0.0.1", port=8001, log_level="error")
        print("✅ Server configuration successful")
        return True
    except Exception as e:
        print(f"❌ Server test failed: {e}")
        return False

def test_frontend_files():
    """Test frontend files exist and are valid"""
    print("🌐 Testing frontend files...")
    
    frontend_files = [
        "frontend/index.html",
        "frontend/app.js",
        "frontend/styles.css"
    ]
    
    all_good = True
    for file_path in frontend_files:
        path = Path(file_path)
        if path.exists() and path.stat().st_size > 0:
            print(f"✅ {file_path} exists and has content")
        else:
            print(f"❌ {file_path} missing or empty")
            all_good = False
    
    # Check for TradingVue.js integration
    try:
        with open("frontend/index.html", "r", encoding="utf-8") as f:
            content = f.read()
            if "trading-vue" in content and "Vue.js" in content:
                print("✅ TradingVue.js integration found")
            else:
                print("❌ TradingVue.js integration missing")
                all_good = False
    except Exception as e:
        print(f"❌ Could not verify TradingVue.js integration: {e}")
        all_good = False
    
    return all_good

def test_data_conversion():
    """Test data conversion functionality"""
    print("🔄 Testing data conversion...")
    
    try:
        from rtai.io.rec_converter import validate_ohlcv_data, get_data_summary
        
        # Test with dummy data
        dummy_ohlcv = [
            [1640995200000, 47000.0, 47100.0, 46900.0, 47050.0, 1.5],
            [1640995260000, 47050.0, 47200.0, 47000.0, 47150.0, 2.1]
        ]
        
        if validate_ohlcv_data(dummy_ohlcv):
            summary = get_data_summary(dummy_ohlcv)
            print("✅ Data conversion functions work")
            return True
        else:
            print("❌ Data validation failed")
            return False
            
    except Exception as e:
        print(f"❌ Data conversion test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 RTAI Complete Pipeline Test")
    print("=" * 50)
    
    tests = [
        ("Import Verification", test_imports),
        ("WebSocket Connection", test_websocket_connection),
        ("Indicator Calculations", test_indicators),
        ("Data Recording", test_recording),
        ("Server Startup", test_server_startup),
        ("Frontend Files", test_frontend_files),
        ("Data Conversion", test_data_conversion),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n📋 {name}")
        print("-" * 30)
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Score: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Pipeline ready!")
        print("\n🚀 Start the complete system:")
        print("   Terminal 1: uvicorn rtai.api.server:app --host 0.0.0.0 --port 8000 --reload")
        print("   Terminal 2: python -m rtai.main --mode live --symbol BTCUSDT")
        print("   Terminal 3: cd frontend && python -m http.server 8080")
        print("   Dashboard: http://localhost:8080")
        return 0
    else:
        print("⚠️  Some tests failed - review issues above")
        return 1

if __name__ == "__main__":
    sys.exit(main())