"""RTAI - Real-Time Algorithmic Indicators

A comprehensive package for cryptocurrency trading indicators.

Main components:
- indicators: Core (OFI, VPIN, KyleLambda, LPI) + Atomic (Liquidations, TOBI, etc.)
- state: State management for pipeline convergence
- utils: Common utilities and helpers
- plotting: Visualization functions
"""

from .indicators import (
    OFI, VPIN, KyleLambda, LPI,
    Liquidations, TopOfBookImbalance, WallRatio, FundingBasis,
    TradeImbalance, ZBand, AdvancedZBandWrapper
)
from .state import StateAdapter, create_state_store

__version__ = "0.9.0"
__all__ = [
    # Core indicators
    "OFI", "VPIN", "KyleLambda", "LPI",
    # Atomic indicators 
    "Liquidations", "TopOfBookImbalance", "WallRatio", "FundingBasis",
    "TradeImbalance", "ZBand", "AdvancedZBandWrapper",
    # State management
    "StateAdapter", "create_state_store"
]
