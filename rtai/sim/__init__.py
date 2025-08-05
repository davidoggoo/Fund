"""
RTAI RRS Engine - Simulation Module Init
========================================

Professional RRS simulation components.
"""

from .broker import SimulationBroker, Order, Fill, OrderSide, OrderType
from .portfolio import Portfolio, EquitySnapshot, PerformanceMetrics, calculate_position_size
from .scorer import BacktestScorer, BacktestResults, SignalPerformance, score_backtest
from .handlers import ReplayHandlers, ReplayContext, create_replay_handlers
from .backtester import RRSBacktester, run_simple_backtest

__all__ = [
    "SimulationBroker",
    "Order", 
    "Fill",
    "OrderSide",
    "OrderType",
    "Portfolio",
    "EquitySnapshot", 
    "PerformanceMetrics",
    "calculate_position_size",
    "BacktestScorer",
    "BacktestResults",
    "SignalPerformance",
    "score_backtest",
    "ReplayHandlers",
    "ReplayContext",
    "create_replay_handlers",
    "RRSBacktester",
    "run_simple_backtest"
]
