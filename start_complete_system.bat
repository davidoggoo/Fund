@echo off
echo 🚀 Starting RTAI Complete Trading System
echo ========================================

echo.
echo 📊 Starting FastAPI Server (Terminal 1)...
start "RTAI API Server" cmd /k "uvicorn rtai.api.server:app --host 0.0.0.0 --port 8000 --reload"

timeout /t 3 /nobreak >nul

echo.
echo 🌐 Starting Frontend Server (Terminal 2)...
start "RTAI Frontend" cmd /k "cd frontend && python -m http.server 8080"

timeout /t 3 /nobreak >nul

echo.
echo 📈 Starting Live Trading (Terminal 3)...
start "RTAI Live Trading" cmd /k "python -m rtai.main --mode live --symbol BTCUSDT"

echo.
echo ✅ System Started Successfully!
echo ========================================
echo 📊 API Server: http://localhost:8000
echo 📋 API Docs: http://localhost:8000/docs
echo 💚 Health Check: http://localhost:8000/health
echo 📈 Metrics: http://localhost:8000/metrics
echo 🔐 Validate API Key: http://localhost:8000/validate-key
echo.
echo 🌐 Frontend Dashboard: http://localhost:8080
echo ========================================
echo.
echo Press any key to continue...
pause >nul