"""
RTAI Backtesting Strategy
Integrates RTAI indicators with kernc/backtesting.py framework
"""

import numpy as np
import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import warnings
warnings.filterwarnings('ignore')

# Import RTAI components
from .indicators.extremes import ExtremeIndicatorManager, ExtremeSignal
from .indicators.simple import (
    WallRatioOsc, TradeImbalanceOsc, LiquidationOsc, 
    DIROsc, FundingAccelOsc, DebounceManager
)
from .signals import SignalDetector, SignalConfig


class RTAIStrategy(Strategy):
    """
    RTAI Backtesting Strategy using kernc/backtesting framework
    
    Features:
    - Multi-oscillator signal generation
    - Extreme detection with persistence
    - Debounced entry/exit logic
    - Risk management via position sizing
    """
    
    # Strategy parameters (tunable via optimization)
    extreme_threshold = 0.8        # Extreme detection sensitivity
    oscillator_threshold = 0.7     # Multi-oscillator consensus
    debounce_period = 5           # Signal debouncing (bars)
    position_size = 0.95          # Max position size (95%)
    stop_loss = 0.02              # 2% stop loss
    take_profit = 0.04            # 4% take profit
    
    def init(self):
        """Initialize RTAI components and indicators"""
        print("🚀 Initializing RTAI Strategy...")
        
        # Initialize core components
        self.extreme_manager = ExtremeIndicatorManager()
        self.debounce_manager = DebounceManager()  # No parameters needed
        self.signal_detector = SignalDetector(SignalConfig())
        
        # Initialize oscillators
        self.oscillators = {
            'wall_ratio': WallRatioOsc(),
            'trade_imbalance': TradeImbalanceOsc(),
            'liquidation': LiquidationOsc(),
            'DIR': DIROsc(),
            'funding_accel': FundingAccelOsc()
        }
        
        # Strategy state
        self.entry_price = None
        self.extreme_count = 0
        self.signal_history = []
        
        print(f"✅ RTAI Strategy initialized with {len(self.oscillators)} oscillators")
    
    def next(self):
        """Execute strategy logic for each bar"""
        # Get current OHLCV data
        current_data = {
            'open': self.data.Open[-1],
            'high': self.data.High[-1], 
            'low': self.data.Low[-1],
            'close': self.data.Close[-1],
            'volume': self.data.Volume[-1] if hasattr(self.data, 'Volume') else 1000
        }
        
        # Update oscillators with current data
        oscillator_signals = {}
        for name, osc in self.oscillators.items():
            try:
                # Update oscillator (mock data for backtesting)
                osc.update({
                    'price': current_data['close'],
                    'volume': current_data['volume'],
                    'time': len(self.data) * 60  # Mock timestamp
                })
                oscillator_signals[name] = osc.get_signal()
            except Exception as e:
                oscillator_signals[name] = 0.0
        
        # Calculate oscillator consensus
        valid_signals = [sig for sig in oscillator_signals.values() if abs(sig) > 0.1]
        if valid_signals:
            avg_signal = np.mean(valid_signals)
            signal_strength = abs(avg_signal)
        else:
            avg_signal = 0.0
            signal_strength = 0.0
        
        # Update extreme detector
        try:
            # Update extreme manager with mock data
            mock_market_data = {
                'timestamp': len(self.data) * 60,
                'price': current_data['close'],
                'volume': current_data['volume'],
                'bid': current_data['close'] * 0.999,
                'ask': current_data['close'] * 1.001,
                'spread': current_data['close'] * 0.002
            }
            
            extreme_signals = self.extreme_manager.update_all(mock_market_data)
            
            # Get strongest extreme signal
            if extreme_signals:
                strongest_signal = max(extreme_signals, key=lambda s: abs(s.strength))
                extreme_strength = abs(strongest_signal.strength)
                extreme_direction = strongest_signal.direction
            else:
                extreme_strength = 0.0
                extreme_direction = "NEUTRAL"
                
        except Exception as e:
            extreme_strength = 0.0
            extreme_direction = "NEUTRAL"
        
        # Generate combined signal
        combined_strength = max(signal_strength, extreme_strength)
        
        # Determine signal direction
        if extreme_strength > self.extreme_threshold:
            signal_direction = 1 if extreme_direction == "BUY" else -1 if extreme_direction == "SELL" else 0
        elif signal_strength > self.oscillator_threshold:
            signal_direction = 1 if avg_signal > 0 else -1
        else:
            signal_direction = 0
        
        # Apply debouncing
        if signal_direction != 0:
            debounced_signal = self.debounce_manager.should_execute(
                f"signal_{signal_direction}", combined_strength
            )
        else:
            debounced_signal = False
        
        # Execute trading logic
        if not self.position:
            # No position - look for entry
            if debounced_signal and combined_strength > self.oscillator_threshold:
                self._enter_position(signal_direction, current_data['close'], combined_strength)
        
        else:
            # Have position - manage risk
            self._manage_position(current_data['close'])
    
    def _enter_position(self, direction: int, price: float, strength: float):
        """Enter long/short position with risk management"""
        try:
            # Calculate position size based on signal strength
            size = self.position_size * strength
            
            if direction > 0:
                # Long entry
                self.buy(size=size)
                self.entry_price = price
                print(f"📈 LONG entry at {price:.4f}, size: {size:.3f}, strength: {strength:.3f}")
            
            elif direction < 0:
                # Short entry (if supported)
                self.sell(size=size)
                self.entry_price = price
                print(f"📉 SHORT entry at {price:.4f}, size: {size:.3f}, strength: {strength:.3f}")
                
        except Exception as e:
            print(f"❌ Entry error: {e}")
    
    def _manage_position(self, current_price: float):
        """Manage existing position with stop loss and take profit"""
        if not self.entry_price:
            return
            
        try:
            pnl_pct = (current_price - self.entry_price) / self.entry_price
            
            if self.position.is_long:
                # Long position management
                if pnl_pct <= -self.stop_loss:
                    self.position.close()
                    print(f"🛑 LONG stop loss at {current_price:.4f}, PnL: {pnl_pct:.3%}")
                    
                elif pnl_pct >= self.take_profit:
                    self.position.close()
                    print(f"💰 LONG take profit at {current_price:.4f}, PnL: {pnl_pct:.3%}")
            
            elif self.position.is_short:
                # Short position management
                if pnl_pct >= self.stop_loss:
                    self.position.close()
                    print(f"🛑 SHORT stop loss at {current_price:.4f}, PnL: {pnl_pct:.3%}")
                    
                elif pnl_pct <= -self.take_profit:
                    self.position.close()
                    print(f"💰 SHORT take profit at {current_price:.4f}, PnL: {pnl_pct:.3%}")
            
            # Clear entry price when position closed
            if not self.position:
                self.entry_price = None
                
        except Exception as e:
            print(f"❌ Position management error: {e}")


def run_rtai_backtest(data: pd.DataFrame, 
                     cash: float = 10000,
                     commission: float = 0.001,
                     **strategy_params) -> Backtest:
    """
    Run RTAI backtest with given data and parameters
    
    Args:
        data: OHLCV DataFrame with DatetimeIndex
        cash: Starting cash amount
        commission: Commission rate (0.001 = 0.1%)
        **strategy_params: Strategy parameter overrides
    
    Returns:
        Configured Backtest instance (call .run() to execute)
    """
    
    # Validate data format
    required_cols = ['Open', 'High', 'Low', 'Close']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create backtest instance
    bt = Backtest(
        data=data,
        strategy=RTAIStrategy,
        cash=cash,
        commission=commission,
        exclusive_orders=True  # Prevent simultaneous long/short
    )
    
    print(f"🔬 RTAI Backtest configured:")
    print(f"   📊 Data: {len(data)} bars ({data.index[0]} to {data.index[-1]})")
    print(f"   💰 Capital: ${cash:,.0f}")
    print(f"   📈 Commission: {commission:.3%}")
    print(f"   ⚙️  Parameters: {strategy_params}")
    
    return bt


def generate_backtest_report(results, output_path: str = None) -> str:
    """
    Generate comprehensive HTML backtest report
    
    Args:
        results: Backtest results from bt.run() (dict-like or Results object)
        output_path: Optional HTML output path
    
    Returns:
        HTML report string
    """
    
    # Handle both dict and Results object
    if hasattr(results, '__getitem__'):
        # Dict-like access
        def get_metric(key, default=0):
            return results.get(key, default)
    else:
        # Results object access
        def get_metric(key, default=0):
            return getattr(results, key.replace('[%]', '_pct').replace(' ', '_').replace('.', '_'), default)
    
    # Extract key metrics safely
    total_return = get_metric('Return [%]', 0.0)
    num_trades = get_metric('# Trades', 0)
    win_rate = get_metric('Win Rate [%]', 0.0)
    sharpe_ratio = get_metric('Sharpe Ratio', 0.0)
    max_drawdown = get_metric('Max. Drawdown [%]', 0.0)
    avg_trade = get_metric('Avg. Trade [%]', 0.0)
    best_trade = get_metric('Best Trade [%]', 0.0)
    worst_trade = get_metric('Worst Trade [%]', 0.0)
    
    # Get trading period (fallback to estimate)
    try:
        if hasattr(results, '_trades'):
            trading_days = len(results._trades)
        elif '_trades' in results:
            trading_days = len(results['_trades'])
        else:
            trading_days = num_trades * 2  # Rough estimate
    except:
        trading_days = max(100, num_trades * 2)  # Fallback estimate
    
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>RTAI Backtesting Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                      color: white; padding: 20px; border-radius: 10px; text-align: center; }}
            .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
                       gap: 15px; margin: 20px 0; }}
            .metric {{ background: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
            .metric-label {{ color: #6c757d; font-size: 14px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚀 RTAI Backtesting Report</h1>
            <p>Advanced cryptocurrency trading strategy backtesting results</p>
        </div>
        
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{total_return:.2f}%</div>
                <div class="metric-label">Total Return</div>
            </div>
            <div class="metric">
                <div class="metric-value">{num_trades}</div>
                <div class="metric-label">Total Trades</div>
            </div>
            <div class="metric">
                <div class="metric-value">{win_rate:.1f}%</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value">{sharpe_ratio:.2f}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            <div class="metric">
                <div class="metric-value">{max_drawdown:.2f}%</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
        </div>
        
        <h2>📈 Performance Summary</h2>
        <ul>
            <li><strong>Strategy:</strong> RTAI Multi-Oscillator + Extreme Detection</li>
            <li><strong>Period:</strong> {trading_days} trading days</li>
            <li><strong>Average Trade:</strong> {avg_trade:.2f}%</li>
            <li><strong>Best Trade:</strong> {best_trade:.2f}%</li>
            <li><strong>Worst Trade:</strong> {worst_trade:.2f}%</li>
        </ul>
        
        <h2>⚙️ Strategy Configuration</h2>
        <ul>
            <li><strong>Oscillators:</strong> WallRatio, TradeImbalance, Liquidation, DIR, FundingAccel</li>
            <li><strong>Extreme Detection:</strong> Persistence-based with adaptive thresholds</li>
            <li><strong>Risk Management:</strong> Stop loss, take profit, position sizing</li>
            <li><strong>Signal Processing:</strong> Debounced entries, consensus-based decisions</li>
        </ul>
        
        <p style="text-align: center; color: #6c757d; margin-top: 40px;">
            Generated by RTAI Backtesting System • {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        </p>
    </body>
    </html>
    """
    
    # Save to file if path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_template)
        print(f"📄 Report saved to: {output_path}")
    
    return html_template


# Example usage and testing
if __name__ == "__main__":
    print("🧪 RTAI Backtesting Strategy - Test Module")
    
    # Create sample data for testing
    dates = pd.date_range('2024-01-01', periods=1000, freq='1H')
    np.random.seed(42)
    
    # Generate realistic OHLCV data
    close_prices = 45000 + np.cumsum(np.random.normal(0, 100, 1000))
    sample_data = pd.DataFrame({
        'Open': close_prices * (1 + np.random.normal(0, 0.001, 1000)),
        'High': close_prices * (1 + np.random.uniform(0, 0.02, 1000)),
        'Low': close_prices * (1 - np.random.uniform(0, 0.02, 1000)),
        'Close': close_prices,
        'Volume': np.random.uniform(100, 1000, 1000)
    }, index=dates)
    
    try:
        # Test backtest setup
        bt = run_rtai_backtest(sample_data, cash=10000)
        print("✅ Backtest configuration successful")
        
        # Note: Actual .run() would require full RTAI components
        print("⚠️  Full backtest requires complete RTAI indicator ecosystem")
        
    except Exception as e:
        print(f"❌ Test error: {e}")
