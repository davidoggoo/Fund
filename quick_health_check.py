#!/usr/bin/env python3
"""
RTAI Quick Health Check
======================

Minimal test to verify core system is working
"""

print("🧪 RTAI Quick Health Check")
print("=" * 40)

# Test 1: Indicator calculations
print("\n📊 Testing core indicators...")
try:
    from rtai.indicators.base import OFI, VPIN, KyleLambda, LPI
    
    # OFI
    ofi = OFI()
    ofi_val = ofi.update(50000, 10, 50001, 8)
    print(f"  ✅ OFI: {ofi_val}")
    
    # VPIN  
    vpin = VPIN()
    for i in range(5):
        vpin_val = vpin.update(1.5 * (1 if i % 2 == 0 else -1))
    print(f"  ✅ VPIN: {vpin_val}")
    
    # Kyle's Lambda
    kyle = KyleLambda()
    kyle_val = kyle.update(50000, 1.5)
    print(f"  ✅ Kyle's Lambda: {kyle_val}")
    
    # LPI
    lpi = LPI()
    lpi.update_oi_estimate(100000)
    lpi_val = lpi.update(1.0, 0.5)
    print(f"  ✅ LPI: {lpi_val}")
    
except Exception as e:
    print(f"  ❌ Indicator test failed: {e}")
    exit(1)

# Test 2: Database
print("\n🗄️ Testing database...")
try:
    import sqlite3
    from pathlib import Path
    
    db_path = Path("state/indicators.db")
    if db_path.exists():
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM indicators")
        count = cursor.fetchone()[0]
        print(f"  ✅ Database accessible, {count} indicators")
        conn.close()
    else:
        print("  ⚠️ Database will be created on first run")
        
except Exception as e:
    print(f"  ❌ Database test failed: {e}")

# Test 3: API Server import
print("\n🌐 Testing API server...")
try:
    from rtai.api.server import app
    print("  ✅ FastAPI server import OK")
except Exception as e:
    print(f"  ❌ API server import failed: {e}")

# Test 4: Frontend files
print("\n📱 Testing frontend...")
try:
    from pathlib import Path
    
    files = ["frontend/index.html", "frontend/app.js", "frontend/styles.css"]
    for file_path in files:
        if Path(file_path).exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path} missing")
            
except Exception as e:
    print(f"  ❌ Frontend test failed: {e}")

print("\n🎉 Core system health check complete!")
print("\nNext steps:")
print("  1. Start API: uvicorn rtai.api.server:app --port 8000")
print("  2. Start frontend: python -m http.server 8080 -d frontend")
print("  3. Start live trading: python -m rtai.main --live --symbol BTCUSDT")
