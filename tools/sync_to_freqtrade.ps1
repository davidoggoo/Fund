#!/usr/bin/env powershell
# RTAI Freqtrade Sync Script
# Syncs custom strategies and providers to ft/ for Freqtrade compatibility

Write-Host "🔄 Syncing RTAI files to Freqtrade structure..." -ForegroundColor Cyan

# Strategy files
Copy-Item -Path "strategies\RTAIStrategy.py" -Destination "ft\user_data\strategies\RTAIStrategy.py" -Force
Write-Host "✅ Synced RTAIStrategy.py" -ForegroundColor Green

# Indicator library
New-Item -ItemType Directory -Force -Path "ft\user_data\strategies\lib" | Out-Null
Copy-Item -Path "strategies\lib\rtai_indicators.py" -Destination "ft\user_data\strategies\lib\rtai_indicators.py" -Force
Write-Host "✅ Synced rtai_indicators.py" -ForegroundColor Green

# Data provider (create directory if needed)
New-Item -ItemType Directory -Force -Path "ft\user_data\dataprovider" | Out-Null
Copy-Item -Path "dataproviders\RTAIDataProvider.py" -Destination "ft\user_data\dataprovider\RTAIDataProvider.py" -Force
Write-Host "✅ Synced RTAIDataProvider.py" -ForegroundColor Green

Write-Host "🎯 All RTAI files synced to Freqtrade structure!" -ForegroundColor Green
Write-Host "💡 Run this script whenever you modify files in strategies/ or dataproviders/" -ForegroundColor Yellow
