#!/usr/bin/env python3
"""
RTAI Dashboard Startup Script
Starts both backend FastAPI server and frontend development server
"""
import subprocess
import sys
import time
import signal
import os
from pathlib import Path


def start_backend():
    """Start FastAPI backend server"""
    print("🚀 Starting RTAI FastAPI backend...")
    return subprocess.Popen([
        sys.executable, "-m", "uvicorn", 
        "rtai.api.server:app", 
        "--host", "0.0.0.0", 
        "--port", "8000", 
        "--reload"
    ])


def start_frontend():
    """Start frontend development server"""
    print("🌐 Starting frontend development server...")
    frontend_dir = Path(__file__).parent / "frontend"
    return subprocess.Popen([
        sys.executable, "-m", "http.server", "8080"
    ], cwd=frontend_dir)


def start_live_trader(symbol="BTCUSDT"):
    """Start live trader (optional)"""
    print(f"📈 Starting live trader for {symbol}...")
    return subprocess.Popen([
        sys.executable, "-m", "rtai.main", 
        "--mode", "live",
        "--symbol", symbol
    ])


def main():
    """Main startup function"""
    processes = []
    
    try:
        # Start backend
        backend_process = start_backend()
        processes.append(("Backend", backend_process))
        time.sleep(2)  # Give backend time to start
        
        # Start frontend
        frontend_process = start_frontend()
        processes.append(("Frontend", frontend_process))
        time.sleep(1)
        
        # Optionally start live trader
        if "--with-trader" in sys.argv:
            symbol = "BTCUSDT"
            if "--symbol" in sys.argv:
                symbol_idx = sys.argv.index("--symbol")
                if symbol_idx + 1 < len(sys.argv):
                    symbol = sys.argv[symbol_idx + 1]
            
            trader_process = start_live_trader(symbol)
            processes.append(("Live Trader", trader_process))
        
        print("\n" + "="*60)
        print("🎉 RTAI Dashboard Started Successfully!")
        print("="*60)
        print("📊 Dashboard: http://localhost:8080")
        print("🔧 API Docs:  http://localhost:8000/docs")
        print("💚 Health:    http://localhost:8000/health")
        print("="*60)
        print("Press Ctrl+C to stop all services")
        print("="*60 + "\n")
        
        # Wait for processes
        while True:
            time.sleep(1)
            
            # Check if any process died
            for name, process in processes:
                if process.poll() is not None:
                    print(f"❌ {name} process died with code {process.returncode}")
                    return 1
    
    except KeyboardInterrupt:
        print("\n🛑 Shutting down RTAI Dashboard...")
        
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        return 1
    
    finally:
        # Cleanup processes
        for name, process in processes:
            try:
                print(f"🔄 Stopping {name}...")
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"⚠️  Force killing {name}...")
                process.kill()
            except Exception as e:
                print(f"⚠️  Error stopping {name}: {e}")
    
    print("✅ All services stopped")
    return 0


if __name__ == "__main__":
    # Handle command line arguments
    if "--help" in sys.argv or "-h" in sys.argv:
        print("RTAI Dashboard Startup Script")
        print("\nUsage:")
        print("  python start_rtai_dashboard.py [options]")
        print("\nOptions:")
        print("  --with-trader     Start live trader along with dashboard")
        print("  --symbol SYMBOL   Symbol for live trader (default: BTCUSDT)")
        print("  --help, -h        Show this help message")
        print("\nExamples:")
        print("  python start_rtai_dashboard.py")
        print("  python start_rtai_dashboard.py --with-trader")
        print("  python start_rtai_dashboard.py --with-trader --symbol ETHUSDT")
        sys.exit(0)
    
    sys.exit(main())