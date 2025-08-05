"""
RTAI RRS Engine - Complete Backtesting System
==============================================

Professional end-to-end backtesting system combining all RRS components.
Record -> Replay -> Score workflow with professional trading simulation.
"""

import asyncio
import time
import os
from typing import Dict, Any, Optional, List
from pathlib import Path
from loguru import logger

from rtai.io import EventReplayer
from rtai.sim import SimulationBroker, Portfolio, BacktestScorer, ReplayHandlers, ReplayContext, score_backtest


class RRSBacktester:
    """
    Complete RRS backtesting system.
    Orchestrates replay, simulation, and analysis for professional backtesting.
    """
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        initial_cash: float = 100000.0,
        commission_rate: float = 0.001,
        slippage_bps: float = 1.0
    ):
        self.symbol = symbol
        self.initial_cash = initial_cash
        
        # Initialize components
        self.broker = SimulationBroker(
            initial_cash=initial_cash,
            commission_rate=commission_rate,
            slippage_bps=slippage_bps
        )
        
        self.portfolio = Portfolio(
            initial_cash=initial_cash,
            symbol=symbol
        )
        
        self.scorer = BacktestScorer(symbol=symbol)
        
        # Replay configuration
        self.replay_speed = 10.0  # 10x speed by default
        self.enable_filtering = True
        self.filter_events = ["trade", "depth_snapshot", "signal_trigger"]  # Key events only
        
        logger.info(f"🎯 RRS Backtester initialized for {symbol} with ${initial_cash:,.2f}")
    
    async def run_backtest(
        self,
        recording_file: str,
        output_dir: str = "backtest_results",
        max_duration_hours: Optional[float] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Run complete backtest on recorded data.
        
        Args:
            recording_file: Path to recorded events file
            output_dir: Directory for backtest outputs
            max_duration_hours: Maximum backtest duration (None = no limit)
            progress_callback: Optional callback for progress updates
        
        Returns:
            Backtest results dictionary
        """
        logger.info(f"🚀 Starting RRS backtest: {recording_file}")
        
        # Setup output directory
        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time())
        result_prefix = f"{self.symbol}_{timestamp}"
        
        try:
            # Initialize replay system
            replayer = EventReplayer(
                speed_multiplier=self.replay_speed,
                enable_stats=True
            )
            
            # Setup replay context
            context = ReplayContext(
                symbol=self.symbol,
                broker=self.broker,
                portfolio=self.portfolio,
                scorer=self.scorer,
                current_time=0.0,
                replay_speed=self.replay_speed
            )
            
            # Initialize handlers
            handlers = ReplayHandlers(context)
            
            # Register event handlers
            replayer.register_handler("trade", handlers.handle_trade)
            replayer.register_handler("depth_snapshot", handlers.handle_depth_snapshot)
            replayer.register_handler("liquidation", handlers.handle_liquidation)
            replayer.register_handler("funding_update", handlers.handle_funding_update)
            replayer.register_handler("indicator_update", handlers.handle_indicator_update)
            replayer.register_handler("basic_oscillator", handlers.handle_basic_oscillator)
            replayer.register_handler("signal_trigger", handlers.handle_signal_trigger)
            
            # Configure filtering
            if self.enable_filtering:
                replayer.set_event_filter(self.filter_events)
            
            # Run replay
            start_time = time.time()
            last_progress_time = start_time
            
            async def progress_handler(stats: Dict[str, Any]):
                nonlocal last_progress_time
                current_time = time.time()
                
                # Update scorer with replay speed
                if 'effective_speed' in stats:
                    self.scorer.record_replay_speed(stats['effective_speed'])
                
                # Progress callback
                if progress_callback and current_time - last_progress_time > 5.0:  # Every 5 seconds
                    progress_callback(stats)
                    last_progress_time = current_time
                
                # Duration limit check
                if max_duration_hours:
                    elapsed_hours = (current_time - start_time) / 3600
                    if elapsed_hours > max_duration_hours:
                        logger.warning(f"⏰ Maximum duration reached: {elapsed_hours:.1f} hours")
                        return False  # Stop replay
                
                return True  # Continue replay
            
            # Execute replay
            replay_stats = await replayer.replay_file(
                recording_file,
                progress_handler=progress_handler
            )
            
            # Generate results
            results = score_backtest(
                portfolio=self.portfolio,
                scorer=self.scorer,
                output_path=os.path.join(output_dir, f"{result_prefix}_results.json")
            )
            
            # Export portfolio history
            portfolio_csv = os.path.join(output_dir, f"{result_prefix}_portfolio.csv")
            self.portfolio.export_to_csv(portfolio_csv)
            
            # Generate summary report
            report_path = os.path.join(output_dir, f"{result_prefix}_report.txt")
            self._generate_text_report(results, replay_stats, report_path)
            
            logger.success(f"✅ Backtest completed! Results saved to {output_dir}")
            
            return {
                "results": results,
                "replay_stats": replay_stats,
                "output_files": {
                    "results_json": f"{result_prefix}_results.json",
                    "portfolio_csv": f"{result_prefix}_portfolio.csv",
                    "report_txt": f"{result_prefix}_report.txt"
                },
                "output_dir": output_dir
            }
            
        except Exception as e:
            logger.error(f"❌ Backtest failed: {str(e)}")
            raise
    
    def _generate_text_report(self, results, replay_stats: Dict[str, Any], output_path: str):
        """Generate human-readable text report"""
        with open(output_path, 'w') as f:
            f.write("RTAI RRS ENGINE - BACKTEST REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Basic info
            f.write(f"Symbol: {results.symbol}\n")
            f.write(f"Duration: {results.duration_hours:.2f} hours\n")
            f.write(f"Start: {time.ctime(results.start_time)}\n")
            f.write(f"End: {time.ctime(results.end_time)}\n\n")
            
            # Portfolio performance
            pm = results.portfolio_metrics
            f.write("PORTFOLIO PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            f.write(f"Initial Capital: ${self.initial_cash:,.2f}\n")
            f.write(f"Final Equity: ${self.initial_cash + pm.total_return:,.2f}\n")
            f.write(f"Total Return: ${pm.total_return:,.2f} ({pm.total_return_pct:.2f}%)\n")
            f.write(f"Annualized Return: {pm.annualized_return:.2f}%\n")
            f.write(f"Volatility: {pm.volatility:.2f}%\n")
            f.write(f"Sharpe Ratio: {pm.sharpe_ratio:.2f}\n")
            f.write(f"Max Drawdown: ${pm.max_drawdown:,.2f} ({pm.max_drawdown_pct:.2f}%)\n")
            f.write(f"Win Rate: {pm.win_rate:.1f}%\n")
            f.write(f"Profit Factor: {pm.profit_factor:.2f}\n")
            f.write(f"Total Trades: {pm.total_trades}\n\n")
            
            # Signal performance
            f.write("SIGNAL PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            if results.signal_performance:
                for signal_name, perf in results.signal_performance.items():
                    f.write(f"{signal_name}:\n")
                    f.write(f"  Total Signals: {perf.total_signals}\n")
                    f.write(f"  Win Rate: {perf.win_rate:.1f}%\n")
                    f.write(f"  Average Return: {perf.avg_return:.4f}\n")
                    f.write(f"  Total Return: {perf.total_return:.4f}\n")
                    f.write(f"  Profit Factor: {perf.profit_factor:.2f}\n")
                    f.write(f"  Max Profit: {perf.max_profit:.4f}\n")
                    f.write(f"  Max Loss: {perf.max_loss:.4f}\n\n")
            else:
                f.write("No signals fired during backtest\n\n")
            
            # Market statistics
            f.write("MARKET STATISTICS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Average Price: ${results.avg_price:,.2f}\n")
            f.write(f"Price Volatility: {results.price_volatility:.4f}\n")
            f.write(f"Total Volume: {results.total_volume:,.0f}\n")
            f.write(f"Trades Processed: {results.total_trades_processed:,}\n")
            f.write(f"Depth Updates: {results.total_depth_updates:,}\n")
            f.write(f"Liquidations: {results.total_liquidations:,}\n")
            f.write(f"Funding Updates: {results.total_funding_updates:,}\n")
            f.write(f"Indicator Updates: {results.total_indicator_updates:,}\n")
            f.write(f"Signals Fired: {results.total_signals_fired:,}\n\n")
            
            # Replay performance
            f.write("REPLAY PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            f.write(f"Target Speed: {self.replay_speed:.1f}x\n")
            f.write(f"Average Speed: {results.replay_speed_avg:.1f}x\n")
            f.write(f"Events/Second: {results.events_per_second:.0f}\n")
            f.write(f"Total Events: {replay_stats.get('total_events', 0):,}\n")
            f.write(f"Filtered Events: {replay_stats.get('filtered_events', 0):,}\n")
            f.write(f"Processing Errors: {replay_stats.get('errors', 0):,}\n\n")
            
            # Configuration
            f.write("CONFIGURATION\n")
            f.write("-" * 40 + "\n")
            f.write(f"Commission Rate: {self.broker.commission_rate:.4f}\n")
            f.write(f"Slippage (bps): {self.broker.slippage_bps:.1f}\n")
            f.write(f"Event Filtering: {self.enable_filtering}\n")
            if self.enable_filtering:
                f.write(f"Filtered Events: {', '.join(self.filter_events)}\n")
            f.write(f"Max Position Size: {getattr(self, 'max_position_size', 'N/A')}\n")
            f.write(f"Risk Per Trade: {getattr(self, 'risk_per_trade', 'N/A')}\n")
            
        logger.info(f"📝 Text report generated: {output_path}")
    
    def configure_trading(
        self,
        risk_per_trade: float = 0.02,
        max_position_size: float = 0.5,
        signal_confidence_threshold: float = 0.7
    ):
        """Configure trading parameters"""
        self.risk_per_trade = risk_per_trade
        self.max_position_size = max_position_size
        self.signal_confidence_threshold = signal_confidence_threshold
        
        logger.info(f"Trading config updated: risk={risk_per_trade}, max_pos={max_position_size}, confidence={signal_confidence_threshold}")
    
    def configure_replay(
        self,
        speed_multiplier: float = 10.0,
        enable_filtering: bool = True,
        filter_events: Optional[List[str]] = None
    ):
        """Configure replay parameters"""
        self.replay_speed = speed_multiplier
        self.enable_filtering = enable_filtering
        
        if filter_events is not None:
            self.filter_events = filter_events
        
        logger.info(f"Replay config updated: speed={speed_multiplier}x, filtering={enable_filtering}")


async def run_simple_backtest(
    recording_file: str,
    symbol: str = "BTCUSDT",
    initial_cash: float = 100000.0,
    replay_speed: float = 10.0,
    output_dir: str = "backtest_results"
) -> Dict[str, Any]:
    """
    Convenience function for simple backtesting.
    
    Args:
        recording_file: Path to recorded events file
        symbol: Trading symbol
        initial_cash: Initial portfolio cash
        replay_speed: Replay speed multiplier
        output_dir: Output directory for results
    
    Returns:
        Backtest results dictionary
    """
    backtester = RRSBacktester(
        symbol=symbol,
        initial_cash=initial_cash
    )
    
    backtester.configure_replay(speed_multiplier=replay_speed)
    
    return await backtester.run_backtest(
        recording_file=recording_file,
        output_dir=output_dir
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="RTAI RRS Engine Backtester")
    parser.add_argument("recording_file", help="Path to recording file")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--cash", type=float, default=100000, help="Initial cash")
    parser.add_argument("--speed", type=float, default=10.0, help="Replay speed multiplier")
    parser.add_argument("--output", default="backtest_results", help="Output directory")
    
    args = parser.parse_args()
    
    async def main():
        results = await run_simple_backtest(
            recording_file=args.recording_file,
            symbol=args.symbol,
            initial_cash=args.cash,
            replay_speed=args.speed,
            output_dir=args.output
        )
        
        print(f"\n✅ Backtest completed!")
        print(f"📊 Results saved to: {results['output_dir']}")
        
        # Print key metrics
        perf = results['results'].portfolio_metrics
        print(f"\n💰 Key Results:")
        print(f"   Total Return: ${perf.total_return:,.2f} ({perf.total_return_pct:.2f}%)")
        print(f"   Sharpe Ratio: {perf.sharpe_ratio:.2f}")
        print(f"   Max Drawdown: {perf.max_drawdown_pct:.2f}%")
        print(f"   Win Rate: {perf.win_rate:.1f}%")
        print(f"   Total Trades: {perf.total_trades}")
    
    asyncio.run(main())
