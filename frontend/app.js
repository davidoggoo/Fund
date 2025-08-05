// RTAI Trading Dashboard - Vue.js Application
new Vue({
    el: '#app',
    data: {
        // WebSocket connection
        ws: null,
        wsConnected: false,
        reconnectAttempts: 0,
        maxReconnectAttempts: 5,
        
        // Authentication
        apiKey: localStorage.getItem('rtai_api_key') || '',
        userInfo: null,
        showApiKeyInput: false,
        
        // Chart configuration
        chartWidth: 800,
        chartHeight: 500,
        selectedSymbol: 'BTCUSDT',
        
        // Chart data structure for TradingVue
        chartData: {
            ohlcv: [],
            onchart: [],
            offchart: []
        },
        
        // TradingVue overlays and extensions
        overlays: [],
        extensions: [],
        colors: {
            colorBack: '#1e1e1e',
            colorGrid: '#333333',
            colorText: '#ffffff',
            colorTextHL: '#00d4ff',
            colorCandleUp: '#00ff88',
            colorCandleDw: '#ff4757',
            colorWickUp: '#00ff88',
            colorWickDw: '#ff4757',
            colorVolUp: '#00ff8844',
            colorVolDw: '#ff475744'
        },
        
        // Live data
        latestIndicators: {},
        backtestResults: null,
        backtestRunning: false,
        
        // Logging
        logs: [],
        maxLogs: 100,
        
        // Data buffers
        priceBuffer: [],
        volumeBuffer: [],
        currentCandle: null,
        candleStartTime: null
    },
    
    mounted() {
        this.initChart();
        this.connectWebSocket();
        this.setupResizeHandler();
        this.setupKeyboardShortcuts();
        this.addLog('info', '🚀 Dashboard initialized with keyboard shortcuts');
    },
    
    beforeDestroy() {
        if (this.ws) {
            this.ws.close();
        }
    },
    
    methods: {
        // WebSocket Management
        connectWebSocket() {
            // Include API key in WebSocket URL if available
            const apiKeyParam = this.apiKey ? `?api_key=${encodeURIComponent(this.apiKey)}` : '';
            const wsUrl = `ws://localhost:8000/ws${apiKeyParam}`;
            this.addLog('info', `Connecting to WebSocket${this.apiKey ? ' (authenticated)' : ' (anonymous)'}...`);
            
            try {
                this.ws = new WebSocket(wsUrl);
                
                this.ws.onopen = () => {
                    this.wsConnected = true;
                    this.reconnectAttempts = 0;
                    this.addLog('success', 'WebSocket connected');
                };
                
                this.ws.onmessage = (event) => {
                    try {
                        const message = JSON.parse(event.data);
                        this.handleWebSocketMessage(message);
                    } catch (error) {
                        console.error('Failed to parse WebSocket message:', error);
                    }
                };
                
                this.ws.onclose = () => {
                    this.wsConnected = false;
                    this.addLog('warning', 'WebSocket disconnected');
                    this.attemptReconnect();
                };
                
                this.ws.onerror = (error) => {
                    this.addLog('error', `WebSocket error: ${error.message || 'Connection failed'}`);
                };
                
            } catch (error) {
                this.addLog('error', `Failed to create WebSocket: ${error.message}`);
                this.attemptReconnect();
            }
        },
        
        attemptReconnect() {
            if (this.reconnectAttempts < this.maxReconnectAttempts) {
                this.reconnectAttempts++;
                const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);
                
                this.addLog('info', `Reconnecting in ${delay/1000}s... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
                
                setTimeout(() => {
                    this.connectWebSocket();
                }, delay);
            } else {
                this.addLog('error', 'Max reconnection attempts reached');
            }
        },
        
        // Message Handling
        handleWebSocketMessage(message) {
            switch (message.t || message.type) {
                case 'welcome':
                    this.handleWelcomeMessage(message.data || message);
                    break;
                case 'bar':
                    this.handleBarData(message.data || message);
                    break;
                case 'trade':
                    this.handleTradeData(message.data || message);
                    break;
                case 'indi':
                    this.handleIndicatorData(message.data || message);
                    break;
                case 'sig':
                case 'signal':
                    this.handleSignalData(message.data || message);
                    break;
                case 'eq':
                case 'equity':
                    this.handleEquityData(message.data || message);
                    break;
                case 'error':
                    this.handleErrorMessage(message.data || message);
                    break;
                case 'backtest_results':
                    this.handleBacktestResults(message.data || message);
                    break;
                default:
                    console.log('Unknown message type:', message.t || message.type, message);
            }
        },
        
        handleBarData(data) {
            // Handle real OHLCV bar data from WebSocket with validation
            try {
                // Data validation
                if (!this.validateOHLCVData(data)) {
                    this.addLog('warning', '⚠️ Invalid OHLCV data received, skipping');
                    return;
                }
                
                const timestamp = data.ts * 1000; // Convert to milliseconds
                const ohlcv = [timestamp, data.o, data.h, data.l, data.c, data.v];
                
                // Anomaly detection
                if (this.detectPriceAnomaly(data)) {
                    this.addLog('warning', `🚨 Price anomaly detected: ${JSON.stringify(data)}`);
                }
                
                // Add to chart
                this.chartData.ohlcv.push(ohlcv);
                if (this.chartData.ohlcv.length > 1000) {
                    this.chartData.ohlcv.shift();
                }
                
                // Force reactivity update
                this.$set(this.chartData, 'ohlcv', [...this.chartData.ohlcv]);
                
                this.addLog('info', `📊 New candle: O=${data.o.toFixed(2)} H=${data.h.toFixed(2)} L=${data.l.toFixed(2)} C=${data.c.toFixed(2)} V=${data.v.toFixed(4)}`);
                
            } catch (error) {
                this.addLog('error', `❌ Error processing bar data: ${error.message}`);
            }
        },
        
        validateOHLCVData(data) {
            // Validate OHLCV data structure and values
            const required = ['ts', 'o', 'h', 'l', 'c', 'v'];
            
            // Check required fields
            for (const field of required) {
                if (!(field in data)) {
                    return false;
                }
            }
            
            // Check numeric values
            const { o, h, l, c, v } = data;
            if (!this.isValidNumber(o) || !this.isValidNumber(h) || 
                !this.isValidNumber(l) || !this.isValidNumber(c) || 
                !this.isValidNumber(v)) {
                return false;
            }
            
            // Check OHLC relationships
            if (h < Math.max(o, c) || l > Math.min(o, c)) {
                return false;
            }
            
            // Check volume is non-negative
            if (v < 0) {
                return false;
            }
            
            return true;
        },
        
        detectPriceAnomaly(data) {
            // Simple anomaly detection for price data
            if (this.chartData.ohlcv.length === 0) {
                return false;
            }
            
            const lastCandle = this.chartData.ohlcv[this.chartData.ohlcv.length - 1];
            const lastClose = lastCandle[4];
            const currentOpen = data.o;
            
            // Check for price gaps > 5%
            const priceChange = Math.abs(currentOpen - lastClose) / lastClose;
            if (priceChange > 0.05) {
                return true;
            }
            
            // Check for extreme price ranges
            const range = (data.h - data.l) / data.l;
            if (range > 0.1) { // 10% range
                return true;
            }
            
            return false;
        },
        
        isValidNumber(value) {
            return typeof value === 'number' && isFinite(value) && value >= 0;
        },
        
        handleTradeData(data) {
            // Handle individual trade data (for tick updates)
            if (data.symbol !== this.selectedSymbol) return;
            
            const timestamp = data.timestamp * 1000; // Convert to milliseconds
            const price = parseFloat(data.price);
            const volume = parseFloat(data.volume);
            
            // Update current candle or create new one (for real-time updates between bars)
            this.updateCandle(timestamp, price, volume);
            
            // Add to price buffer for real-time updates
            this.priceBuffer.push({ time: timestamp, price: price });
            if (this.priceBuffer.length > 1000) {
                this.priceBuffer.shift();
            }
            
            this.addLog('info', `${data.symbol}: $${price.toFixed(2)} (${data.side})`);
        },
        
        handleIndicatorData(data) {
            if (data.symbol !== this.selectedSymbol) return;
            
            this.latestIndicators = { ...data.indicators };
            
            // Add indicators to chart overlays
            this.updateIndicatorOverlays(data.indicators, data.timestamp);
            
            this.addLog('info', `Indicators updated: ${Object.keys(data.indicators).length} values`);
        },
        
        handleSignalData(data) {
            this.addLog('warning', `🚨 SIGNAL: ${data.type} - ${data.message}`);
            
            // Add signal marker to chart
            if (data.price && data.timestamp) {
                this.addSignalMarker(data.timestamp * 1000, data.price, data.type, data.message);
            }
        },
        
        handleEquityData(data) {
            // Handle equity curve updates
            this.addLog('info', `Portfolio: $${data.equity?.toFixed(2) || 'N/A'}`);
        },
        
        handleWelcomeMessage(data) {
            // Handle welcome message with user info
            this.userInfo = data;
            this.addLog('success', `Welcome ${data.user} (${data.permissions.join(', ')})`);
        },
        
        handleErrorMessage(data) {
            // Handle error messages
            this.addLog('error', data.message || 'Unknown error occurred');
        },
        
        handleBacktestResults(data) {
            this.backtestResults = data;
            this.backtestRunning = false;
            this.addLog('success', `Backtest completed: ${data.total_return}% return`);
            
            // Add trade markers to chart
            if (data.trades) {
                this.addTradeMarkers(data.trades);
            }
        },
        
        // Chart Management
        initChart() {
            // Initialize with empty data structure
            this.chartData = {
                ohlcv: [],
                onchart: [
                    {
                        name: 'Signals',
                        type: 'Spline',
                        data: [],
                        settings: { color: '#ff6b6b' }
                    }
                ],
                offchart: [
                    {
                        name: 'OFI',
                        type: 'Spline',
                        data: [],
                        settings: { color: '#00d4ff' }
                    },
                    {
                        name: 'VPIN',
                        type: 'Spline', 
                        data: [],
                        settings: { color: '#00ff88' }
                    },
                    {
                        name: 'Kyle Lambda',
                        type: 'Spline',
                        data: [],
                        settings: { color: '#ffa502' }
                    }
                ]
            };
            
            this.setupCustomOverlays();
        },
        
        updateCandle(timestamp, price, volume) {
            const candleTime = Math.floor(timestamp / 60000) * 60000; // 1-minute candles
            
            if (!this.currentCandle || this.candleStartTime !== candleTime) {
                // Start new candle
                if (this.currentCandle) {
                    // Push completed candle
                    this.chartData.ohlcv.push([...this.currentCandle]);
                    if (this.chartData.ohlcv.length > 1000) {
                        this.chartData.ohlcv.shift();
                    }
                }
                
                this.currentCandle = [candleTime, price, price, price, price, volume];
                this.candleStartTime = candleTime;
            } else {
                // Update existing candle
                this.currentCandle[2] = Math.max(this.currentCandle[2], price); // High
                this.currentCandle[3] = Math.min(this.currentCandle[3], price); // Low
                this.currentCandle[4] = price; // Close
                this.currentCandle[5] += volume; // Volume
            }
            
            // Update chart data reactively
            this.$set(this.chartData, 'ohlcv', [...this.chartData.ohlcv]);
        },
        
        updateIndicatorOverlays(indicators, timestamp) {
            const time = timestamp * 1000;
            
            // Update OFI overlay
            if (indicators.ofi_rsi !== undefined) {
                const ofiOverlay = this.chartData.offchart.find(o => o.name === 'OFI');
                if (ofiOverlay) {
                    ofiOverlay.data.push([time, indicators.ofi_rsi]);
                    if (ofiOverlay.data.length > 500) {
                        ofiOverlay.data.shift();
                    }
                }
            }
            
            // Update VPIN overlay
            if (indicators.vpin_rsi !== undefined) {
                const vpinOverlay = this.chartData.offchart.find(o => o.name === 'VPIN');
                if (vpinOverlay) {
                    vpinOverlay.data.push([time, indicators.vpin_rsi]);
                    if (vpinOverlay.data.length > 500) {
                        vpinOverlay.data.shift();
                    }
                }
            }
            
            // Update Kyle Lambda overlay
            if (indicators.kyle_rsi !== undefined) {
                const kyleOverlay = this.chartData.offchart.find(o => o.name === 'Kyle Lambda');
                if (kyleOverlay) {
                    kyleOverlay.data.push([time, indicators.kyle_rsi]);
                    if (kyleOverlay.data.length > 500) {
                        kyleOverlay.data.shift();
                    }
                }
            }
            
            // Force reactivity update
            this.$set(this.chartData, 'offchart', [...this.chartData.offchart]);
        },
        
        addSignalMarker(timestamp, price, type, message) {
            const signalOverlay = this.chartData.onchart.find(o => o.name === 'Signals');
            if (signalOverlay) {
                signalOverlay.data.push([timestamp, price]);
                if (signalOverlay.data.length > 100) {
                    signalOverlay.data.shift();
                }
                this.$set(this.chartData, 'onchart', [...this.chartData.onchart]);
            }
        },
        
        addTradeMarkers(trades) {
            // Add backtest trade markers to chart
            trades.forEach(trade => {
                const timestamp = new Date(trade.entry_time).getTime();
                this.addSignalMarker(timestamp, trade.entry_price, trade.type, `${trade.type} Trade`);
            });
        },
        
        setupCustomOverlays() {
            // Custom overlay configurations for advanced indicators
            this.overlays = [
                // OFI Z-Score overlay with threshold lines
                {
                    name: 'OFI_Z',
                    type: 'Spline',
                    data: [],
                    settings: {
                        color: '#00d4ff',
                        lineWidth: 2,
                        upper: 2.0,
                        lower: -2.0,
                        showThresholds: true
                    }
                },
                // VPIN overlay with dynamic threshold
                {
                    name: 'VPIN',
                    type: 'Spline',
                    data: [],
                    settings: {
                        color: '#00ff88',
                        lineWidth: 2,
                        threshold: 0.98
                    }
                }
            ];
        },
        
        // API Key Management
        saveApiKey() {
            if (this.apiKey.trim()) {
                localStorage.setItem('rtai_api_key', this.apiKey.trim());
                this.addLog('success', 'API key saved');
                this.showApiKeyInput = false;
                
                // Reconnect WebSocket with new API key
                if (this.ws) {
                    this.ws.close();
                }
                setTimeout(() => this.connectWebSocket(), 1000);
            }
        },
        
        clearApiKey() {
            this.apiKey = '';
            localStorage.removeItem('rtai_api_key');
            this.userInfo = null;
            this.addLog('info', 'API key cleared');
            
            // Reconnect as anonymous
            if (this.ws) {
                this.ws.close();
            }
            setTimeout(() => this.connectWebSocket(), 1000);
        },
        
        toggleApiKeyInput() {
            this.showApiKeyInput = !this.showApiKeyInput;
        },
        
        // Backtest Integration with enhanced UX
        async runBacktest() {
            if (this.backtestRunning) return;
            
            this.backtestRunning = true;
            this.backtestResults = null;
            this.addLog('info', `🎯 Starting REAL backtest for ${this.selectedSymbol}...`);
            
            // Show loading indicator
            const loadingSteps = [
                'Loading historical data...',
                'Calculating indicators...',
                'Running strategy...',
                'Generating results...'
            ];
            
            let stepIndex = 0;
            const loadingInterval = setInterval(() => {
                if (stepIndex < loadingSteps.length) {
                    this.addLog('info', `⏳ ${loadingSteps[stepIndex]}`);
                    stepIndex++;
                }
            }, 1000);
            
            try {
                const startTime = Date.now();
                
                const response = await fetch('http://localhost:8000/backtest', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        symbol: this.selectedSymbol,
                        file: `recordings/${this.selectedSymbol}_latest.rec.gz`,  // Pass recording file path
                        start_date: null, // Use default
                        end_date: null    // Use default
                    })
                });
                
                clearInterval(loadingInterval);
                
                if (!response.ok) {
                    const errorText = await response.text();
                    throw new Error(`HTTP ${response.status}: ${errorText}`);
                }
                
                const result = await response.json();
                const duration = ((Date.now() - startTime) / 1000).toFixed(1);
                
                if (result.status === 'success') {
                    this.addLog('success', `✅ Backtest completed in ${duration}s: ${result.results.total_return.toFixed(2)}% return`);
                    // Results will be received via WebSocket
                } else {
                    throw new Error(result.detail || 'Backtest failed on server');
                }
                
            } catch (error) {
                clearInterval(loadingInterval);
                this.backtestRunning = false;
                this.addLog('error', `❌ Backtest failed: ${error.message}`);
                
                // Show user-friendly error message
                if (error.message.includes('429')) {
                    this.addLog('warning', '⚠️ Rate limit exceeded. Please wait before trying again.');
                } else if (error.message.includes('400')) {
                    this.addLog('warning', '⚠️ Invalid request. Please check your parameters.');
                } else if (error.message.includes('500')) {
                    this.addLog('error', '🔧 Server error. Please try again later.');
                }
            }
        },
        
        // Symbol Management
        changeSymbol() {
            this.addLog('info', `Switching to ${this.selectedSymbol}`);
            // Clear current data
            this.chartData.ohlcv = [];
            this.chartData.onchart.forEach(overlay => overlay.data = []);
            this.chartData.offchart.forEach(overlay => overlay.data = []);
            this.latestIndicators = {};
            this.currentCandle = null;
            this.candleStartTime = null;
        },
        
        // Utility Functions
        formatIndicatorName(key) {
            return key.replace(/_/g, ' ').toUpperCase();
        },
        
        formatIndicatorValue(value) {
            if (value === null || value === undefined) return 'N/A';
            if (typeof value === 'number') {
                return value.toFixed(4);
            }
            return String(value);
        },
        
        getIndicatorClass(key, value) {
            if (value === null || value === undefined) return 'neutral';
            
            // Z-score based classification
            if (key.includes('_z') || key.includes('_rsi')) {
                const numValue = parseFloat(value);
                if (numValue > 2) return 'extreme';
                if (numValue > 0.5) return 'positive';
                if (numValue < -2) return 'extreme';
                if (numValue < -0.5) return 'negative';
            }
            
            return 'neutral';
        },
        
        formatTime(timestamp) {
            return new Date(timestamp).toLocaleTimeString();
        },
        
        addLog(type, message) {
            this.logs.unshift({
                type: type,
                message: message,
                timestamp: Date.now()
            });
            
            if (this.logs.length > this.maxLogs) {
                this.logs.pop();
            }
            
            // Auto-scroll log container
            this.$nextTick(() => {
                if (this.$refs.logContainer) {
                    this.$refs.logContainer.scrollTop = 0;
                }
            });
        },
        
        setupResizeHandler() {
            const updateChartSize = () => {
                const container = document.querySelector('.chart-container');
                if (container) {
                    this.chartWidth = Math.max(container.clientWidth - 32, 400); // Minimum width
                    this.chartHeight = Math.max(container.clientHeight - 32, 300); // Minimum height
                }
            };
            
            window.addEventListener('resize', updateChartSize);
            this.$nextTick(updateChartSize);
        },
        
        setupKeyboardShortcuts() {
            document.addEventListener('keydown', (event) => {
                // Ctrl/Cmd + B: Run Backtest
                if ((event.ctrlKey || event.metaKey) && event.key === 'b') {
                    event.preventDefault();
                    this.runBacktest();
                    this.addLog('info', '⌨️ Backtest triggered via keyboard shortcut');
                }
                
                // Ctrl/Cmd + R: Reconnect WebSocket
                if ((event.ctrlKey || event.metaKey) && event.key === 'r') {
                    event.preventDefault();
                    this.connectWebSocket();
                    this.addLog('info', '⌨️ WebSocket reconnection triggered');
                }
                
                // Ctrl/Cmd + L: Clear logs
                if ((event.ctrlKey || event.metaKey) && event.key === 'l') {
                    event.preventDefault();
                    this.logs = [];
                    this.addLog('info', '⌨️ Logs cleared via keyboard shortcut');
                }
                
                // Escape: Stop backtest (if running)
                if (event.key === 'Escape' && this.backtestRunning) {
                    this.backtestRunning = false;
                    this.addLog('warning', '⌨️ Backtest cancelled via Escape key');
                }
            });
            
            // Show keyboard shortcuts in console
            console.log(`
🎹 RTAI Dashboard Keyboard Shortcuts:
• Ctrl/Cmd + B: Run Backtest
• Ctrl/Cmd + R: Reconnect WebSocket  
• Ctrl/Cmd + L: Clear Logs
• Escape: Cancel Backtest
            `);
        }
    }
});