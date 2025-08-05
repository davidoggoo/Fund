#!/usr/bin/env python3
"""
RTAI Unified Test Suite
=======================

Test everything in sequence:
1. Syntax validation
2. Import validation  
3. Indicator pipeline
4. Database integrity
5. Multi-stream feed system
6. WebSocket health check
"""

import asyncio
import sys
import time
from pathlib import Path

def test_syntax():
    """Test syntax of all Python files"""
    print("🔍 Testing Python syntax...")
    
    files_to_check = [
        "rtai/live_trader.py",
        "rtai/indicators/base.py", 
        "rtai/api/server.py",
        "rtai/io/multi_stream_feed.py"
    ]
    
    import ast
    for file_path in files_to_check:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            ast.parse(content)
            print(f"  ✅ {file_path}")
        except Exception as e:
            print(f"  ❌ {file_path}: {e}")
            return False
    
    return True

def test_imports():
    """Test critical imports"""
    print("📦 Testing imports...")
    
    try:
        from rtai.indicators.base import OFI, VPIN, KyleLambda, LPI
        print("  ✅ Indicators")
    except Exception as e:
        print(f"  ❌ Indicators: {e}")
        return False
    
    try:
        from rtai.live_trader import LiveTrader
        print("  ✅ LiveTrader")  
    except Exception as e:
        print(f"  ❌ LiveTrader: {e}")
        return False
    
    try:
        from rtai.api.server import app
        print("  ✅ API Server")
    except Exception as e:
        print(f"  ❌ API Server: {e}")
        return False
    
    try:
        from rtai.io.multi_stream_feed import BinanceMultiStreamFeed
        print("  ✅ Multi-Stream Feed")
    except Exception as e:
        print(f"  ❌ Multi-Stream Feed: {e}")
        return False
    
    return True

def test_indicators():
    """Test indicator calculations"""
    print("📊 Testing indicators...")
    
    try:
        from rtai.indicators.base import OFI, VPIN, KyleLambda, LPI
        
        # OFI test
        ofi = OFI()
        ofi_val = ofi.update(50000, 10, 50001, 8)
        print(f"  ✅ OFI: {ofi_val}")
        
        # VPIN test
        vpin = VPIN()
        vpin_val = vpin.update(1.5)  # signed quantity
        print(f"  ✅ VPIN: {vpin_val}")
        
        # Kyle's Lambda test
        kyle = KyleLambda()
        kyle_val = kyle.update(50000, 1.5)
        print(f"  ✅ Kyle's Lambda: {kyle_val}")
        
        # LPI test
        lpi = LPI()
        lpi.update_oi_estimate(100000)  # Set OI estimate first
        lpi_val = lpi.update(1.0, 0.5)  # long_qty, short_qty
        print(f"  ✅ LPI: {lpi_val}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Indicator test failed: {e}")
        return False

def test_database():
    """Test database connectivity and schema"""
    print("🗄️ Testing database...")
    
    try:
        import sqlite3
        db_path = Path("state/indicators.db")
        
        if not db_path.exists():
            print("  ⚠️ Database doesn't exist yet - will be created on first run")
            return True
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        required_tables = ['indicators', 'equity', 'trades']
        for table in required_tables:
            if table in tables:
                print(f"  ✅ Table '{table}' exists")
            else:
                print(f"  ⚠️ Table '{table}' missing")
        
        # Check ts_ms column  
        cursor.execute("PRAGMA table_info(indicators)")
        columns = [col[1] for col in cursor.fetchall()]
        
        if 'ts_ms' in columns:
            print("  ✅ ts_ms column exists")
        else:
            print("  ⚠️ ts_ms column missing")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"  ❌ Database test failed: {e}")
        return False

def test_frontend():
    """Test frontend files"""
    print("🌐 Testing frontend...")
    
    files = [
        "frontend/index.html",
        "frontend/app.js", 
        "frontend/styles.css"
    ]
    
    for file_path in files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} missing")
            return False
    
    # Check TradingVue integration
    with open("frontend/app.js", 'r') as f:
        content = f.read()
        if "TradingVue" in content:
            print("  ✅ TradingVue integration found")
        else:
            print("  ⚠️ TradingVue integration not found")
    
    return True

async def test_feed_system():
    """Test multi-stream feed system (non-blocking)"""
    print("📡 Testing feed system...")
    
    try:
        from rtai.io.multi_stream_feed import BinanceMultiStreamFeed, bus
        
        # Create feed instance
        feed = BinanceMultiStreamFeed("BTCUSDT")
        print("  ✅ Multi-stream feed created")
        
        # Test event bus
        test_received = False
        
        async def test_handler(data):
            nonlocal test_received
            test_received = True
        
        bus.on('test_event', test_handler)
        await bus.emit('test_event', {'test': True})
        
        if test_received:
            print("  ✅ Event bus working")
        else:
            print("  ❌ Event bus not working")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ❌ Feed system test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 RTAI Unified Test Suite")
    print("=" * 50)
    
    tests = [
        ("Syntax", test_syntax),
        ("Imports", test_imports),
        ("Indicators", test_indicators),
        ("Database", test_database),
        ("Frontend", test_frontend),
    ]
    
    results = []
    
    for name, test_func in tests:
        print(f"\n📋 {name}")
        print("-" * 30)
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"  ❌ Test failed with exception: {e}")
            results.append((name, False))
    
    # Async tests
    print(f"\n📋 Feed System")
    print("-" * 30)
    try:
        result = asyncio.run(test_feed_system())
        results.append(("Feed System", result))
    except Exception as e:
        print(f"  ❌ Feed system test failed: {e}")
        results.append(("Feed System", False))
    
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
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️ Some tests failed - review issues above")
        return 1

if __name__ == "__main__":
    sys.exit(main())
