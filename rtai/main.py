"""Main entry point for RTAI system

Usage:
    python -m rtai.main [options]
    rtai [options]  # if installed

This is the canonical entry point that starts the full pipeline.
Enhanced with health monitoring and production deployment features.
"""

import argparse
import asyncio
import sys
import os
import signal
from pathlib import Path
from loguru import logger

# Add parent directory to path for imports (when running from source)
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from rtai.utils import configure_logging, get_log_level
from rtai.utils.environment import ensure_environment_defaults, validate_environment_security
from rtai.live_trader import LiveTrader
from rtai.state import create_state_store, StateAdapter

# Health monitoring imports
try:
    from rtai.utils.health_dashboard import start_health_dashboard, stop_health_dashboard
    from rtai.utils.structured_logging import configure_json_logging
    HEALTH_MONITORING_AVAILABLE = True
except ImportError:
    HEALTH_MONITORING_AVAILABLE = False
    logger.warning("Health monitoring not available")

# Configure event loop policy (avoid Windows warnings)
if sys.platform != 'win32':
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        pass  # uvloop not available


# Global shutdown handler
_shutdown_event = asyncio.Event()
_health_dashboard = None


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
    _shutdown_event.set()


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown"""
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Main entry point with argument parsing"""
    try:
        # Set secure environment defaults first
        ensure_environment_defaults()
        
        # Configure logging early
        log_level = get_log_level()
        configure_logging(log_level)
        
        # Validate environment security
        env_validation = validate_environment_security()
        if not env_validation['valid']:
            logger.error("❌ Environment validation failed:")
            for issue in env_validation['issues']:
                logger.error(f"  • {issue['message']}")
            return 1
            
        # Enhanced argument parsing with validation
        parser = create_argument_parser()
        args = parse_and_validate_args(parser)
        
        if args is None:
            return 1  # Validation failed
        
        # Execute based on mode
        if args.mode == "live":
            return run_live_mode(args)
        else:
            return run_snapshot_mode(args)
            
    except KeyboardInterrupt:
        logger.info("🛑 Program interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"❌ Critical error in main: {e}")
        return 1

def create_argument_parser():
    """Create and configure argument parser"""
    parser = argparse.ArgumentParser(
        description="RTAI - Real-Time Algorithmic Indicators",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  rtai                                 # Generate snapshot and exit
  rtai --mode live                     # Start live trading with signals
  rtai --mode live --duration 60       # Live trade for 60 minutes
  rtai --symbol ETHUSDT                # Monitor ETHUSDT
  rtai --once                          # Generate one snapshot and exit
  rtai --dry-run 30                    # Run for 30 seconds then exit
        """
    )
    
    parser.add_argument("--mode", choices=["snapshot", "live"], default="snapshot",
                       help="Run mode: snapshot (generate chart) or live (real-time trading)")
    parser.add_argument("--symbol", default="BTCUSDT", 
                       help="Trading symbol (default: BTCUSDT)")
    parser.add_argument("--state-store", choices=["memory", "parquet"], default="memory",
                       help="State persistence mode (default: memory)")
    parser.add_argument("--dry-run", type=int, metavar="SECONDS",
                       help="Run for specified seconds then exit")
    parser.add_argument("--once", action="store_true",
                       help="Run once then exit")
    parser.add_argument("--window", type=int, default=100,
                       help="Indicator window size for live mode (default: 100)")
    parser.add_argument("--duration", type=int, metavar="MINUTES",
                       help="Duration in minutes for live mode (default: indefinite)")
    
    return parser

def parse_and_validate_args(parser):
    """Parse and validate command line arguments"""
    try:
        args = parser.parse_args()
        
        # Enhanced validation
        if not isinstance(args.symbol, str) or not args.symbol.strip():
            logger.error("❌ Invalid symbol: must be non-empty string")
            return None
        
        # Normalize symbol
        args.symbol = args.symbol.strip().upper()
        
        # Validate window size
        if args.window <= 0:
            logger.error("❌ Invalid window size: must be positive")
            return None
        
        if args.window > 10000:
            logger.warning(f"⚠️ Large window size: {args.window}, capping at 10000")
            args.window = 10000
        
        # Validate dry-run duration
        if args.dry_run is not None:
            if args.dry_run <= 0:
                logger.error("❌ Invalid dry-run duration: must be positive")
                return None
            if args.dry_run > 86400:  # 24 hours
                logger.warning(f"⚠️ Long dry-run duration: {args.dry_run}s, capping at 24h")
                args.dry_run = 86400
        
        # Validate duration
        if args.duration is not None:
            if args.duration <= 0:
                logger.error("❌ Invalid duration: must be positive")
                return None
            if args.duration > 1440:  # 24 hours
                logger.warning(f"⚠️ Long duration: {args.duration}min, capping at 24h")
                args.duration = 1440
        
        return args
        
    except SystemExit:
        # argparse calls sys.exit on error
        return None
    except Exception as e:
        logger.error(f"❌ Error parsing arguments: {e}")
        return None

def run_live_mode(args):
    """Run live trading mode with enhanced error handling"""
    try:
        logger.info(f"🚀 Starting live trading mode for {args.symbol}")
        
        async def run_live():
            try:
                trader = LiveTrader(symbol=args.symbol, window_size=args.window)
                
                # Override state store if different from default
                if args.state_store != "memory":
                    try:
                        trader.state_store = create_state_store(args.state_store)
                        trader.state_adapter = StateAdapter(trader.state_store)
                        logger.info(f"📊 Using {args.state_store} state store")
                    except Exception as e:
                        logger.error(f"❌ Failed to create state store: {e}")
                        return 1
                
                # Calculate duration
                duration = None
                if args.duration:
                    duration = args.duration
                elif args.dry_run:
                    duration = args.dry_run / 60
                
                if duration:
                    logger.info(f"⏱️ Running for {duration:.1f} minutes")
                else:
                    logger.info("⏱️ Running indefinitely (Ctrl+C to stop)")
                
                await trader.start_live_trading(duration_minutes=duration)
                return 0
                
            except Exception as e:
                logger.error(f"❌ Error in live trading: {e}")
                return 1
        
        return asyncio.run(run_live())
        
    except KeyboardInterrupt:
        logger.info("🛑 Live trading stopped by user")
        return 0
    except Exception as e:
        logger.error(f"❌ Critical error in live mode: {e}")
        return 1

def run_snapshot_mode(args):
    """Run snapshot mode with enhanced error handling"""
    try:
        logger.info(f"📊 Snapshot mode for {args.symbol}")
        
        # TEMPORARILY DISABLED - will use backtesting.py
        logger.info("📊 Snapshot generation temporarily disabled - use backtesting integration")
        logger.info(f"🔄 Run: python -m rtai.bt.quick_plot --symbol {args.symbol}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("📊 Snapshot generation cancelled")
        return 0
    except Exception as e:
        logger.error(f"❌ Error in snapshot mode: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
