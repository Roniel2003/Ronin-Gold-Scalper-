"""
CLI - Command Line Interface
Entry point for all Ronin bot operations and commands.
"""

import argparse
import sys
import os
from datetime import datetime
from config import get_config, validate_config
import asyncio
from typing import Dict, Any


def cmd_help():
    """Display help information"""
    print("Ronin Bot - 15-Minute Trend-Momentum Trading System")
    print("=" * 50)
    print("\nAvailable Commands:")
    print("  load        Load and preview CSV data")
    print("  indicators  Compute technical indicators")
    print("  signals     Generate trading signals")
    print("  backtest    Run backtesting analysis (Stage 3)")
    print("  monte-carlo Run Monte Carlo robustness analysis (Stage 4)")
    print("  optimize    Run parameter optimization (Stage 4)")
    print("  walk-forward Run walk-forward analysis (Stage 4)")
    print("  selftest    Run system self-tests")
    print("  live        Start live trading (Stage 5)")
    print("  status      Get live trading status (Stage 5)")
    print("  stop        Stop live trading (Stage 5)")
    print("\nUse 'python cli.py <command> --help' for command-specific options")


def cmd_load(args):
    """Load and preview CSV data"""
    try:
        from data_loader import load_csv, filter_sessions, resample_to_15m, validate_data_format, get_data_summary
        
        print(f"📥 Loading CSV data from: {args.csv}")
        
        # Get configuration
        cfg = get_config()
        
        # Load CSV data
        df = load_csv(args.csv, cfg['timezone'])
        
        # Validate data format
        validate_data_format(df)
        
        # Resample to 15m if needed
        df = resample_to_15m(df)
        
        # Filter sessions if requested
        if not args.no_sessions:
            df = filter_sessions(df, cfg)
        
        # Get and display summary
        summary = get_data_summary(df)
        
        print("\n📊 Data Summary:")
        print(f"   Total bars: {summary['total_bars']:,}")
        print(f"   Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        print(f"   Days: {summary['date_range']['days']}")
        print(f"   Price range: ${summary['price_range']['min']:.2f} - ${summary['price_range']['max']:.2f}")
        print(f"   Latest price: ${summary['price_range']['latest']:.2f}")
        print(f"   Timezone: {summary['timezone']}")
        print(f"   Timeframe: {summary['timeframe']}")
        
        # Show preview if requested
        if args.preview:
            print("\n📋 Data Preview (first 5 bars):")
            print(df.head().to_string(index=False))
            
            print("\n📋 Data Preview (last 5 bars):")
            print(df.tail().to_string(index=False))
        
        # Save processed data if output specified
        if args.out:
            df.to_csv(args.out, index=False)
            print(f"\n✅ Processed data saved to: {args.out}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return 1


def cmd_indicators(args):
    """Compute indicators on loaded data"""
    try:
        from data_loader import load_csv, filter_sessions, resample_to_15m
        from ronin_engine import compute_indicators
        from config import get_config
        
        cfg = get_config()
        
        if not args.csv:
            print("❌ Error: --csv parameter required")
            return 1
        
        print(f"🔄 Loading and processing data from {args.csv}")
        
        # Load and process data
        df = load_csv(args.csv, cfg['timezone'])
        
        if not args.no_sessions:
            df = filter_sessions(df, cfg['sessions'], cfg['timezone'])
        
        df = resample_to_15m(df)
        
        # Compute indicators
        df_with_indicators = compute_indicators(df, cfg)
        
        # Show preview if requested
        if args.preview:
            print("\n📊 Data with Indicators Preview:")
            indicator_cols = ['Time', 'Close', 'zscore', 'atr', 'atr_pct', 'ema_fast', 'ema_slow', 'trend_up']
            available_cols = [col for col in indicator_cols if col in df_with_indicators.columns]
            print(df_with_indicators[available_cols].tail(10).to_string(index=False))
        
        # Save output if specified
        if args.out:
            df_with_indicators.to_csv(args.out, index=False)
            print(f"💾 Saved indicators data to {args.out}")
        
        # Show summary
        print(f"\n📈 Indicators Summary:")
        print(f"   Total bars: {len(df_with_indicators):,}")
        print(f"   Z-score range: {df_with_indicators['zscore'].min():.2f} to {df_with_indicators['zscore'].max():.2f}")
        print(f"   ATR average: {df_with_indicators['atr'].mean():.4f}")
        print(f"   Trend up %: {(df_with_indicators['trend_up'].sum() / len(df_with_indicators) * 100):.1f}%")
        
        return 0
        
    except Exception as e:
        print(f"❌ Indicators computation failed: {e}")
        return 1


def cmd_signals(args):
    """Generate trading signals from data with indicators"""
    try:
        from data_loader import load_csv, filter_sessions, resample_to_15m
        from ronin_engine import compute_indicators, generate_signals
        from config import get_config
        
        cfg = get_config()
        
        if not args.csv:
            print("❌ Error: --csv parameter required")
            return 1
        
        print(f"🔄 Loading data and generating signals from {args.csv}")
        
        # Load and process data
        df = load_csv(args.csv, cfg['timezone'])
        
        if not args.no_sessions:
            df = filter_sessions(df, cfg['sessions'], cfg['timezone'])
        
        df = resample_to_15m(df)
        
        # Compute indicators and generate signals
        df_with_indicators = compute_indicators(df, cfg)
        df_with_signals = generate_signals(df_with_indicators, cfg)
        
        # Show preview if requested
        if args.preview:
            print("\n🎯 Signals Preview:")
            signal_cols = ['Time', 'Close', 'zscore', 'signal_long', 'signal_short', 'signal_strength']
            available_cols = [col for col in signal_cols if col in df_with_signals.columns]
            
            # Show recent signals
            signals_df = df_with_signals[
                (df_with_signals['signal_long']) | (df_with_signals['signal_short'])
            ][available_cols]
            
            if len(signals_df) > 0:
                print(signals_df.tail(10).to_string(index=False))
            else:
                print("   No signals found in the data")
        
        # Save output if specified
        if args.out:
            df_with_signals.to_csv(args.out, index=False)
            print(f"💾 Saved signals data to {args.out}")
        
        # Show signal summary
        long_signals = df_with_signals['signal_long'].sum()
        short_signals = df_with_signals['signal_short'].sum()
        total_bars = len(df_with_signals)
        
        print(f"\n🎯 Signals Summary:")
        print(f"   Total bars: {total_bars:,}")
        print(f"   Long signals: {long_signals}")
        print(f"   Short signals: {short_signals}")
        print(f"   Signal frequency: {((long_signals + short_signals) / total_bars * 100):.2f}%")
        
        if long_signals + short_signals > 0:
            avg_strength = df_with_signals[
                (df_with_signals['signal_long']) | (df_with_signals['signal_short'])
            ]['signal_strength'].mean()
            print(f"   Average signal strength: {avg_strength:.2f}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Signal generation failed: {e}")
        return 1


def cmd_drytrade(args):
    """Simulate trade construction"""
    print(f"Dry trading simulation on: {args.csv}")
    print("This functionality will be implemented in Stage 3")
    return 0


def cmd_backtest(args):
    """Run comprehensive backtesting analysis"""
    try:
        from data_loader import load_csv, filter_sessions, resample_to_15m
        from ronin_engine import compute_indicators, generate_signals
        from backtester import run_backtest, save_backtest_results, print_backtest_summary
        from config import get_config
        
        cfg = get_config()
        
        if not args.csv:
            print("❌ Error: --csv parameter required")
            return 1
        
        print(f"🔄 Running backtest on {args.csv}")
        
        # Load and process data
        df = load_csv(args.csv, cfg['timezone'])
        
        if not args.no_sessions:
            df = filter_sessions(df, cfg['sessions'], cfg['timezone'])
        
        df = resample_to_15m(df)
        
        # Compute indicators and generate signals
        df_with_indicators = compute_indicators(df, cfg)
        df_with_signals = generate_signals(df_with_indicators, cfg)
        
        # Run backtest
        start_equity = args.equity if args.equity else 10000.0
        results = run_backtest(df_with_signals, cfg, start_equity)
        
        # Print summary to console
        print_backtest_summary(results)
        
        # Save results if output directory specified
        if args.out:
            save_backtest_results(results, args.out)
        
        # Show key metrics
        metrics = results['metrics']
        if metrics['total_trades'] > 0:
            print(f"\n🎯 Key Results:")
            print(f"   Win Rate: {metrics['win_rate']:.1%}")
            print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"   Total Return: {metrics['total_return_pct']:+.1f}%")
            print(f"   Max Drawdown: {metrics['max_dd_pct']:.1f}%")
        else:
            print("\n⚠️  No trades generated - check signal parameters")
        
        return 0
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_monte_carlo(args):
    """Run Monte Carlo robustness analysis"""
    try:
        from data_loader import load_csv, filter_sessions, resample_to_15m
        from ronin_engine import compute_indicators, generate_signals
        from backtester import run_backtest
        from monte_carlo import run_monte_carlo_analysis, print_monte_carlo_summary, save_monte_carlo_results
        from config import get_config
        
        cfg = get_config()
        
        if not args.csv:
            print("❌ Error: --csv parameter required")
            return 1
        
        print(f"🎲 Running Monte Carlo analysis on {args.csv}")
        
        # Load and process data
        df = load_csv(args.csv, cfg['timezone'])
        
        if not args.no_sessions:
            df = filter_sessions(df, cfg['sessions'], cfg['timezone'])
        
        df = resample_to_15m(df)
        
        # Compute indicators and generate signals
        df_with_indicators = compute_indicators(df, cfg)
        df_with_signals = generate_signals(df_with_indicators, cfg)
        
        # Run initial backtest to get trades
        start_equity = args.equity if args.equity else 10000.0
        backtest_results = run_backtest(df_with_signals, cfg, start_equity)
        
        if not backtest_results['trades']:
            print("⚠️  No trades found - cannot run Monte Carlo analysis")
            return 1
        
        # Run Monte Carlo analysis
        n_simulations = args.simulations if args.simulations else 1000
        mc_results = run_monte_carlo_analysis(
            backtest_results['trades'], 
            start_equity, 
            n_simulations
        )
        
        # Print summary
        print_monte_carlo_summary(mc_results)
        
        # Save results if output directory specified
        if args.out:
            save_monte_carlo_results(mc_results, args.out)
        
        # Show robustness assessment
        stability_ratio = abs(mc_results['total_return']['mean']) / mc_results['total_return']['std'] if mc_results['total_return']['std'] > 0 else 0
        
        print(f"\n🎯 Robustness Assessment:")
        if stability_ratio > 1.0:
            print("🟢 Strategy shows HIGH robustness")
        elif stability_ratio > 0.5:
            print("🟡 Strategy shows MEDIUM robustness")
        else:
            print("🔴 Strategy shows LOW robustness - consider parameter optimization")
        
        return 0
        
    except Exception as e:
        print(f"❌ Monte Carlo analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_optimize(args):
    """Run parameter optimization"""
    try:
        from data_loader import load_csv, filter_sessions, resample_to_15m
        from optimizer import optimize_parameters, print_optimization_summary, save_optimization_results
        from config import get_config
        
        cfg = get_config()
        
        if not args.csv:
            print("❌ Error: --csv parameter required")
            return 1
        
        print(f"🔍 Running parameter optimization on {args.csv}")
        
        # Load and process data
        df = load_csv(args.csv, cfg['timezone'])
        
        if not args.no_sessions:
            df = filter_sessions(df, cfg['sessions'], cfg['timezone'])
        
        df = resample_to_15m(df)
        
        # Define parameter ranges (can be made configurable later)
        param_ranges = {
            'zscore_threshold': [1.5, 2.0, 2.5, 3.0],
            'atr_threshold': [0.5, 1.0, 1.5, 2.0],
            'ema_fast': [10, 15, 20],
            'ema_slow': [30, 40, 50]
        }
        
        # Override with custom ranges if provided
        if args.param_ranges:
            import json
            try:
                custom_ranges = json.loads(args.param_ranges)
                param_ranges.update(custom_ranges)
            except json.JSONDecodeError:
                print("⚠️  Invalid parameter ranges JSON, using defaults")
        
        # Run optimization
        start_equity = args.equity if args.equity else 10000.0
        optimization_metric = args.metric if args.metric else 'sharpe_ratio'
        
        opt_results = optimize_parameters(
            df, param_ranges, cfg, start_equity, 
            optimization_metric, min_trades=args.min_trades if args.min_trades else 10
        )
        
        # Print summary
        print_optimization_summary(opt_results)
        
        # Save results if output directory specified
        if args.out:
            save_optimization_results(opt_results, args.out)
        
        return 0
        
    except Exception as e:
        print(f"❌ Parameter optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_walk_forward(args):
    """Run walk-forward analysis"""
    try:
        from data_loader import load_csv, filter_sessions, resample_to_15m
        from optimizer import walk_forward_analysis, save_optimization_results
        from config import get_config
        
        cfg = get_config()
        
        if not args.csv:
            print("❌ Error: --csv parameter required")
            return 1
        
        print(f"🚶 Running walk-forward analysis on {args.csv}")
        
        # Load and process data
        df = load_csv(args.csv, cfg['timezone'])
        
        if not args.no_sessions:
            df = filter_sessions(df, cfg['sessions'], cfg['timezone'])
        
        df = resample_to_15m(df)
        
        # Define parameter ranges
        param_ranges = {
            'zscore_threshold': [1.5, 2.0, 2.5, 3.0],
            'atr_threshold': [0.5, 1.0, 1.5, 2.0],
            'ema_fast': [10, 15, 20],
            'ema_slow': [30, 40, 50]
        }
        
        # Run walk-forward analysis
        start_equity = args.equity if args.equity else 10000.0
        window_months = args.window if args.window else 6
        step_months = args.step if args.step else 1
        optimization_metric = args.metric if args.metric else 'sharpe_ratio'
        
        wf_results = walk_forward_analysis(
            df, param_ranges, cfg, window_months, step_months, 
            start_equity, optimization_metric
        )
        
        # Print summary
        print(f"\n🎯 Walk-Forward Analysis Summary")
        print("=" * 40)
        
        agg = wf_results['aggregate_stats']
        print(f"Total Periods: {agg['total_periods']}")
        print(f"Successful Periods: {agg['successful_periods']}")
        print(f"Average Return: {agg['avg_return']:+.2f}%")
        print(f"Return Std Dev: {agg['std_return']:.2f}%")
        print(f"Period Win Rate: {agg['win_rate_periods']:.1%}")
        print(f"Sharpe Ratio: {agg['sharpe_ratio']:.3f}")
        print(f"Total Trades: {agg['total_trades']}")
        
        # Save results if output directory specified
        if args.out:
            save_optimization_results(wf_results, args.out)
        
        return 0
        
    except Exception as e:
        print(f"❌ Walk-forward analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_selftest(args):
    """Run system self-tests"""
    print("Running Ronin system self-tests...")
    
    try:
        # Test config loading
        config = get_config()
        validate_config(config)
        print("✅ Configuration validation passed")
        
        # Test module imports
        import ronin_engine
        import risk_manager
        import backtester
        import data_loader
        import logger
        print("✅ All modules imported successfully")
        
        # Test basic functionality
        from risk_manager import FtmoGuard
        guard = FtmoGuard(10000.0, config)
        print("✅ FTMO guard initialization passed")
        
        # Test data loader functionality
        from data_loader import get_data_summary
        import pandas as pd
        import pytz
        
        # Create test data
        ny_tz = pytz.timezone('America/New_York')
        test_data = {
            'Time': [ny_tz.localize(datetime(2024, 1, 15, 10, 30))],
            'Open': [100.0],
            'High': [105.0],
            'Low': [99.0],
            'Close': [103.0],
            'Volume': [1000]
        }
        test_df = pd.DataFrame(test_data)
        summary = get_data_summary(test_df)
        print("✅ Data loader functionality passed")
        
        print("\n🎉 All self-tests passed!")
        return 0
        
    except Exception as e:
        print(f"❌ Self-test failed: {e}")
        return 1


def cmd_live(args):
    """Start live trading"""
    try:
        import asyncio
        from live_signal_processor import LiveTradingIntegrator
        from data_feeds import DataFeedManager, create_data_feed, DataFeedConfig, DataProvider
        from live_trader import SimulatedBroker
        from mt5_broker import MT5Broker
        import json
        
        print("🚀 Starting Ronin Live Trading System")
        print("=" * 50)
        
        # Get configuration
        cfg = get_config()
        
        # Determine broker type
        broker_type = args.broker or cfg.get('live_trading', {}).get('broker_type', 'simulated')
        
        if broker_type == 'simulated':
            print("📊 Using simulated broker for testing")
            broker = SimulatedBroker(initial_balance=args.equity)
        elif broker_type == 'mt5':
            print("🔗 Connecting to MetaTrader 5")
            broker = MT5Broker(cfg)
        else:
            raise ValueError(f"Unsupported broker type: {broker_type}")
        
        # Create data feed
        data_provider = args.data_provider or cfg.get('live_trading', {}).get('data_provider', 'simulated')
        symbols = args.symbols or cfg.get('symbols', ['NVDA', 'TSLA', 'AAPL'])
        
        if data_provider == 'simulated':
            print(f"📈 Using simulated data feed for symbols: {symbols}")
            feed_config = DataFeedConfig(
                provider=DataProvider.SIMULATED,
                symbols=symbols,
                update_interval=args.update_interval
            )
            feed = create_data_feed(feed_config)
        elif data_provider == 'alpaca':
            print(f"📡 Connecting to Alpaca data feed for symbols: {symbols}")
            feed_config = DataFeedConfig(
                provider=DataProvider.ALPACA,
                api_key=cfg.get('alpaca', {}).get('api_key'),
                api_secret=cfg.get('alpaca', {}).get('api_secret'),
                base_url=cfg.get('alpaca', {}).get('base_url', 'https://paper-api.alpaca.markets'),
                symbols=symbols,
                update_interval=args.update_interval
            )
            feed = create_data_feed(feed_config)
        elif data_provider == 'mt5':
            print(f"📈 Using MetaTrader 5 data feed for symbols: {symbols}")
            feed_config = DataFeedConfig(
                provider=DataProvider.MT5,
                symbols=symbols,
                update_interval=args.update_interval
            )
            feed = create_data_feed(feed_config)
        else:
            raise ValueError(f"Unsupported data provider: {data_provider}")
        
        # Create data feed manager
        data_manager = DataFeedManager()
        data_manager.add_feed(data_provider, feed)
        
        # Create and start live trading system
        async def run_live_trading():
            integrator = LiveTradingIntegrator()
            await integrator.initialize(broker, data_manager)
            
            try:
                await integrator.start()
                print(f"✅ {args.strategy.upper()} Live trading started (PID: {os.getpid()})")
                print(f"🔍 Scanning market for {', '.join(symbols)} signals...")
                
                # Show initial status once
                status = integrator.get_status()
                active_processors = len([p for p in status.get('processors', {}).values() 
                                       if p.get('is_running', False)])
                
                print(f"📊 {args.strategy.upper()} Ready: {active_processors} processors active, Strategy: {status.get('strategy_version', 'Unknown')}")
                print("⏳ Waiting for trading signals...")
                
                # Keep process alive without status printing
                while True:
                    await asyncio.sleep(10)
                    
            except KeyboardInterrupt:
                print("\n🛑 Stopping live trading...")
                await integrator.stop()
                
                # Clean up PID file
                if os.path.exists('.ronin_live_pid'):
                    os.remove('.ronin_live_pid')
                
                print("✅ Live trading stopped")
        
        asyncio.run(run_live_trading())
        
    except Exception as e:
        print(f"❌ Error starting live trading: {e}")
        import traceback
        traceback.print_exc()


def cmd_status(args):
    """Get live trading status"""
    try:
        import os
        import psutil
        import json
        import requests
        
        # Check if live trading is running
        if not os.path.exists('.ronin_live_pid'):
            print("❌ Live trading is not running")
            return
        
        with open('.ronin_live_pid', 'r') as f:
            pid = int(f.read().strip())
        
        if not psutil.pid_exists(pid):
            print("❌ Live trading process not found")
            os.remove('.ronin_live_pid')
            return
        
        print("✅ Live trading is running")
        print(f"📊 Process ID: {pid}")
        
        # Try to get detailed status if available
        # This would require implementing a status endpoint
        print("💡 Use 'python cli.py live --status-interval 30' for real-time updates")
        
    except Exception as e:
        print(f"❌ Error checking status: {e}")


def cmd_stop(args):
    """Stop live trading"""
    try:
        import os
        import psutil
        import signal
        
        if not os.path.exists('.ronin_live_pid'):
            print("❌ Live trading is not running")
            return
        
        with open('.ronin_live_pid', 'r') as f:
            pid = int(f.read().strip())
        
        if not psutil.pid_exists(pid):
            print("❌ Live trading process not found")
            os.remove('.ronin_live_pid')
            return
        
        print(f"🛑 Stopping live trading (PID: {pid})...")
        
        # Send SIGINT (Ctrl+C) to gracefully stop
        os.kill(pid, signal.SIGINT)
        
        # Wait for process to stop
        import time
        for _ in range(10):
            if not psutil.pid_exists(pid):
                break
            time.sleep(1)
        
        if psutil.pid_exists(pid):
            print("⚠️  Process still running, forcing termination...")
            os.kill(pid, signal.SIGTERM)
        
        # Clean up PID file
        if os.path.exists('.ronin_live_pid'):
            os.remove('.ronin_live_pid')
        
        print("✅ Live trading stopped")
        
    except Exception as e:
        print(f"❌ Error stopping live trading: {e}")


def add_backtest_args(parser):
    """Add backtest-specific arguments"""
    parser.add_argument('--symbol', type=str, default='NVDA',
                       help='Symbol to backtest (default: NVDA)')
    parser.add_argument('--start-date', type=str, default='2024-01-01',
                       help='Start date (YYYY-MM-DD, default: 2024-01-01)')
    parser.add_argument('--end-date', type=str, default='2024-12-31',
                       help='End date (YYYY-MM-DD, default: 2024-12-31)')
    parser.add_argument('--equity', type=float, default=10000.0,
                       help='Starting equity (default: 10000)')
    parser.add_argument('--strategy', type=str, choices=['v1', 'v2', 'v3'], default='v2',
                       help='Strategy version (default: v2)')
    parser.add_argument('--config', type=str, 
                       help='Path to custom config file')
    parser.add_argument('--save-results', action='store_true',
                       help='Save backtest results to file')
    parser.add_argument('--plot', action='store_true',
                       help='Generate performance plots')


def add_live_args(parser):
    """Add live trading arguments"""
    parser.add_argument('--broker', type=str, choices=['simulated', 'mt5'], 
                       default='simulated', help='Broker type (default: simulated)')
    parser.add_argument('--data-provider', type=str, choices=['simulated', 'mt5', 'alpaca'], 
                       default='simulated', help='Data provider (default: simulated)')
    parser.add_argument('--symbols', type=str, nargs='+', 
                       default=['NVDA', 'TSLA', 'AAPL'], 
                       help='Symbols to trade (default: NVDA TSLA AAPL)')
    parser.add_argument('--equity', type=float, default=10000.0,
                       help='Starting equity (default: 10000)')
    parser.add_argument('--strategy', type=str, choices=['v1', 'v2', 'v3'], default='v2',
                       help='Strategy version (default: v2)')
    parser.add_argument('--config', type=str,
                       help='Path to custom config file')
    parser.add_argument('--update-interval', type=int, default=1,
                       help='Status update interval in seconds (default: 1)')


async def run_backtest(args):
    """Run backtesting with specified strategy version"""
    from config import get_config, validate_config
    
    try:
        # Load configuration
        cfg = get_config(args.config)
        validate_config(cfg)
        
        print(f"🔄 Starting {args.strategy.upper()} backtest for {args.symbol}")
        print(f"   Period: {args.start_date} to {args.end_date}")
        print(f"   Starting equity: ${args.equity:,.2f}")
        
        # Load data based on strategy version
        if args.strategy == 'v2':
            # V2.0 uses 1-minute data
            cfg['timeframe'] = '1min'
            from data_loader_v2 import load_data_v2
            from backtester_v2 import run_backtest_v2, print_backtest_summary_v2
            
            df = load_data_v2(args.symbol, args.start_date, args.end_date, cfg)
            results = run_backtest_v2(df, cfg, args.equity)
            print_backtest_summary_v2(results)
            
        elif args.strategy == 'v3':
            # V3.0 strategy - XAUUSD 1-minute scalping
            cfg['timeframe'] = '1min'
            cfg['asset'] = 'XAUUSD'
            
            # V3.0 specific parameters
            cfg['rsi_period'] = 14
            cfg['zscore_period'] = 21
            cfg['z_threshold_v3'] = 1.1
            cfg['atr_period'] = 14
            
            from data_loader_v3 import load_data_v3
            from backtester_v3 import run_backtest_v3, print_backtest_summary_v3
            
            df = load_data_v3(args.symbol, args.start_date, args.end_date, cfg)
            results = run_backtest_v3(df, cfg, args.equity)
            print_backtest_summary_v3(results)
            
        else:
            # V1 strategy (legacy)
            cfg['timeframe'] = '15m'
            from data_loader import load_csv, filter_sessions, resample_to_15m
            from backtester import run_backtest, print_backtest_summary
            
            # For V1, we need to implement load_data or use existing functions
            print("❌ V1 strategy support needs to be implemented")
            return
        
        # Save results if requested
        if args.save_results:
            import json
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_{args.strategy}_{args.symbol}_{timestamp}.json"
            
            # Convert results to JSON-serializable format
            json_results = {
                'strategy_version': args.strategy.upper(),
                'symbol': args.symbol,
                'start_date': args.start_date,
                'end_date': args.end_date,
                'starting_equity': args.equity,
                'final_equity': results['final_equity'],
                'total_pnl': results['total_pnl'],
                'metrics': results['metrics'],
                'total_trades': len(results['trades']),
                'timestamp': timestamp
            }
            
            with open(filename, 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
            
            print(f"💾 Results saved to {filename}")
        
        # Generate plots if requested
        if args.plot:
            try:
                import matplotlib.pyplot as plt
                
                # Plot equity curve
                equity_curve = results['equity_curve']
                
                plt.figure(figsize=(12, 8))
                
                # Main equity plot
                plt.subplot(2, 1, 1)
                plt.plot(equity_curve['Time'], equity_curve['Equity'], 'b-', linewidth=2)
                plt.title(f'{args.strategy.upper()} Strategy - Equity Curve ({args.symbol})')
                plt.ylabel('Equity ($)')
                plt.grid(True, alpha=0.3)
                
                # Daily P&L plot
                plt.subplot(2, 1, 2)
                plt.plot(equity_curve['Time'], equity_curve['Daily_PnL'], 'g-', alpha=0.7)
                plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
                plt.title('Daily P&L')
                plt.ylabel('Daily P&L ($)')
                plt.xlabel('Time')
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                plot_filename = f"backtest_{args.strategy}_{args.symbol}_{timestamp}.png"
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                plt.show()
                
                print(f"📊 Plot saved to {plot_filename}")
                
            except ImportError:
                print("⚠️  Matplotlib not available for plotting")
            except Exception as e:
                print(f"⚠️  Error generating plots: {e}")
        
        return results
        
    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        return None


async def start_live_trading(args):
    """Start live trading with specified strategy version"""
    from config import get_config, validate_config
    
    try:
        # Load configuration
        cfg = get_config(args.config)
        validate_config(cfg)
        
        print(f"🚀 Starting {args.strategy.upper()} live trading")
        print(f"   Broker: {args.broker}")
        print(f"   Data Provider: {args.data_provider}")
        print(f"   Symbols: {args.symbols}")
        print(f"   Starting Equity: ${args.equity:,.2f}")
        
        # Import appropriate modules based on strategy version
        if args.strategy == 'v3':
            # Ronin Gold V3.0 strategy - XAUUSD 1-minute scalping
            cfg['timeframe'] = '1min'
            cfg['asset'] = 'XAUUSD'
            
            # V3.0 specific parameters
            cfg['rsi_period'] = 14
            cfg['zscore_period'] = 21
            cfg['z_threshold_v3'] = 1.1
            cfg['atr_period'] = 14
            
            from live_signal_processor_v3 import LiveTradingIntegratorV3
            
            # Initialize components
            broker = await get_broker(args.broker, cfg)
            data_feed = await get_data_feed(args.data_provider, cfg)
            trading_engine = await create_trading_engine_with_broker(broker, cfg)
            
            # Create V3.0 integrator
            integrator = LiveTradingIntegratorV3(broker, data_feed, trading_engine, cfg)
            
        elif args.strategy == 'v2':
            # V2.0 strategy
            cfg['timeframe'] = '1min'
            from live_signal_processor_v2 import LiveTradingIntegratorV2
            
            # Initialize components
            broker = await get_broker(args.broker, cfg)
            data_feed = await get_data_feed(args.data_provider, cfg)
            trading_engine = await create_trading_engine_with_broker(broker, cfg)
            
            # Create V2.0 integrator
            integrator = LiveTradingIntegratorV2(broker, data_feed, trading_engine, cfg)
            
        else:
            # V1 strategy (legacy)
            cfg['timeframe'] = '15m'
            from live_signal_processor import LiveTradingIntegrator
            
            # Initialize components
            broker = await get_broker(args.broker, cfg)
            data_feed = await get_data_feed(args.data_provider, cfg)
            trading_engine = await create_trading_engine_with_broker(broker, cfg)
            
            # Create V1 integrator
            integrator = LiveTradingIntegrator(broker, data_feed, trading_engine, cfg)
        
        # Start live trading
        await integrator.start(args.symbols, args.equity)
        
        # Save PID for process management
        pid = os.getpid()
        with open('.ronin_live_pid', 'w') as f:
            f.write(str(pid))
        
        print(f"✅ {args.strategy.upper()} Live trading started (PID: {pid})")
        print(f"🔍 Scanning market for {', '.join(args.symbols)} signals...")
        
        # Show initial status once
        status = integrator.get_status()
        active_processors = len([p for p in status.get('processors', {}).values() 
                               if p.get('is_running', False)])
        
        print(f"📊 {args.strategy.upper()} Ready: {active_processors} processors active, Strategy: {status.get('strategy_version', 'Unknown')}")
        print("⏳ Waiting for trading signals...")
        
        # Keep process alive without printing
        try:
            while True:
                await asyncio.sleep(10)
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping live trading...")
            await integrator.stop()
            
            # Remove PID file
            if os.path.exists('.ronin_live_pid'):
                os.remove('.ronin_live_pid')
            
            print("✅ Live trading stopped successfully")
        
    except Exception as e:
        print(f"❌ Live trading failed: {e}")
        
        # Clean up PID file on error
        if os.path.exists('.ronin_live_pid'):
            os.remove('.ronin_live_pid')


async def get_broker(broker_type: str, cfg: Dict[str, Any]):
    """Initialize broker based on type"""
    if broker_type == 'simulated':
        from live_trader import SimulatedBroker
        return SimulatedBroker()
    
    elif broker_type == 'mt5':
        from mt5_broker import MT5Broker
        # Pass the full config to MT5Broker
        broker = MT5Broker(cfg)
        await broker.connect()
        return broker
    
    elif broker_type == 'alpaca':
        from live_trader import AlpacaBroker
        return AlpacaBroker(
            api_key=cfg.get('alpaca_api_key'),
            secret_key=cfg.get('alpaca_secret_key'),
            base_url=cfg.get('alpaca_base_url', 'https://paper-api.alpaca.markets')
        )
    
    else:
        raise ValueError(f"Unknown broker type: {broker_type}")


async def get_data_feed(provider_type: str, cfg: Dict[str, Any]):
    """Initialize data feed based on provider type"""
    if provider_type == 'simulated':
        from data_feeds import SimulatedDataFeed, DataFeedConfig, DataProvider
        
        # Create proper config object
        feed_config = DataFeedConfig(
            provider=DataProvider.SIMULATED,
            symbols=cfg.get('symbols', ['NVDA', 'TSLA', 'AAPL']),
            update_interval=cfg.get('update_interval', 1)
        )
        data_feed = SimulatedDataFeed(feed_config)
        return data_feed
        
    elif provider_type == 'alpaca':
        from data_feeds import AlpacaDataFeed, DataFeedConfig, DataProvider
        
        # Create proper config object for Alpaca
        feed_config = DataFeedConfig(
            provider=DataProvider.ALPACA,
            api_key=os.getenv('ALPACA_API_KEY'),
            api_secret=os.getenv('ALPACA_SECRET_KEY'),
            base_url=os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets'),
            symbols=cfg.get('symbols', ['NVDA', 'TSLA', 'AAPL']),
            update_interval=cfg.get('update_interval', 1)
        )
        data_feed = AlpacaDataFeed(feed_config)
        return data_feed
        
    elif provider_type == 'mt5':
        from data_feeds import MT5DataFeed, DataFeedConfig, DataProvider
        
        # Create proper config object for MT5
        feed_config = DataFeedConfig(
            provider=DataProvider.MT5,
            symbols=cfg.get('symbols', ['NVDA', 'TSLA', 'AAPL']),
            update_interval=cfg.get('update_interval', 1)
        )
        data_feed = MT5DataFeed(feed_config)
        return data_feed
        
    else:
        raise ValueError(f"Unknown data provider: {provider_type}")


async def create_trading_engine_with_broker(broker, cfg: Dict[str, Any]):
    """Create trading engine with broker and config"""
    from live_trader import LiveTradingEngine
    
    # Create trading engine with proper constructor signature
    engine = LiveTradingEngine(broker, cfg)
    
    return engine


async def run_optimization(args):
    """Run parameter optimization for specified strategy"""
    print(f"🔧 Parameter optimization for {args.strategy.upper()} strategy")
    print("This feature is coming soon...")
    
    # For now, suggest using the parameter_tweaker.py directly
    print("\nTo run optimization manually:")
    print(f"python parameter_tweaker.py")


async def check_live_status():
    """Check live trading status"""
    try:
        import os
        import psutil
        
        # Check if live trading is running
        if not os.path.exists('.ronin_live_pid'):
            print("❌ Live trading is not running")
            return
        
        with open('.ronin_live_pid', 'r') as f:
            pid = int(f.read().strip())
        
        if not psutil.pid_exists(pid):
            print("❌ Live trading process not found")
            os.remove('.ronin_live_pid')
            return
        
        print("✅ Live trading is running")
        print(f"📊 Process ID: {pid}")
        
        # Try to get detailed status if available
        print("💡 Use 'python cli.py live --status-interval 30' for real-time updates")
        
    except Exception as e:
        print(f"❌ Error checking status: {e}")


async def stop_live_trading():
    """Stop live trading"""
    try:
        import os
        import psutil
        import signal
        import time
        
        if not os.path.exists('.ronin_live_pid'):
            print("❌ Live trading is not running")
            return
        
        with open('.ronin_live_pid', 'r') as f:
            pid = int(f.read().strip())
        
        if not psutil.pid_exists(pid):
            print("❌ Live trading process not found")
            os.remove('.ronin_live_pid')
            return
        
        print(f"🛑 Stopping live trading (PID: {pid})...")
        
        # Send SIGINT (Ctrl+C) to gracefully stop
        os.kill(pid, signal.SIGINT)
        
        # Wait for process to stop
        for _ in range(10):
            if not psutil.pid_exists(pid):
                break
            time.sleep(1)
        
        if psutil.pid_exists(pid):
            print("⚠️  Process still running, forcing termination...")
            os.kill(pid, signal.SIGTERM)
        
        # Clean up PID file
        if os.path.exists('.ronin_live_pid'):
            os.remove('.ronin_live_pid')
        
        print("✅ Live trading stopped")
        
    except Exception as e:
        print(f"❌ Error stopping live trading: {e}")


def main():
    """Main CLI entry point with V2.0 support"""
    parser = argparse.ArgumentParser(
        description='Ronin Trading Bot - Multi-Strategy Quantitative Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run V2.0 backtest (1-minute strategy)
  python cli.py backtest --symbol NVDA --strategy v2 --start-date 2024-01-01 --end-date 2024-03-31
  
  # Run V1 backtest (15-minute legacy strategy)
  python cli.py backtest --symbol NVDA --strategy v1 --start-date 2024-01-01 --end-date 2024-03-31
  
  # Start V2.0 live trading
  python cli.py live --strategy v2 --symbols NVDA TSLA AAPL --broker simulated
  
  # Start V1 live trading
  python cli.py live --strategy v1 --symbols NVDA TSLA AAPL --broker simulated
  
  # Run parameter optimization
  python cli.py optimize --symbol NVDA --strategy v2
  
  # Check live trading status
  python cli.py status
  
  # Stop live trading
  python cli.py stop
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtesting')
    add_backtest_args(backtest_parser)
    
    # Live trading command
    live_parser = subparsers.add_parser('live', help='Start live trading')
    add_live_args(live_parser)
    
    # Optimization command
    optimize_parser = subparsers.add_parser('optimize', help='Run parameter optimization')
    optimize_parser.add_argument('--symbol', type=str, default='NVDA',
                                help='Symbol to optimize (default: NVDA)')
    optimize_parser.add_argument('--strategy', type=str, choices=['v1', 'v2', 'v3'], default='v2',
                                help='Strategy version (default: v2)')
    optimize_parser.add_argument('--config', type=str,
                                help='Path to custom config file')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check live trading status')
    
    # Stop command
    stop_parser = subparsers.add_parser('stop', help='Stop live trading')
    
    # Config command
    config_parser = subparsers.add_parser('config', help='Show current configuration')
    config_parser.add_argument('--strategy', type=str, choices=['v1', 'v2', 'v3'], default='v2',
                              help='Strategy version (default: v2)')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Handle commands
    if args.command == 'backtest':
        asyncio.run(run_backtest(args))
    elif args.command == 'live':
        asyncio.run(start_live_trading(args))
    elif args.command == 'optimize':
        asyncio.run(run_optimization(args))
    elif args.command == 'status':
        asyncio.run(check_live_status())
    elif args.command == 'stop':
        asyncio.run(stop_live_trading())
    elif args.command == 'config':
        show_config(args)

def show_config(args):
    """Show current configuration for specified strategy"""
    from config import get_config
    
    cfg = get_config()
    
    print(f"\n📋 Ronin {args.strategy.upper()} Configuration")
    print("=" * 50)
    
    if args.strategy == 'v3':
        print(f"Strategy Version: {cfg.get('strategy_version', 'V3.0')}")
        print(f"Timeframe: {cfg.get('timeframe', '1min')}")
        print(f"Asset: {cfg.get('asset', 'XAUUSD')}")
        print(f"RSI Period: {cfg.get('rsi_period', 14)}")
        print(f"Z-Score Period: {cfg.get('zscore_period', 21)}")
        print(f"Z-Score Threshold: ±{cfg.get('z_threshold_v3', 1.1)}")
        print(f"ATR Period: {cfg.get('atr_period', 14)}")
        print(f"Session 1 (Asian): {cfg.get('session_start_hour', 19)}:{cfg.get('session_start_minute', 0):02d} - "
              f"{cfg.get('session_end_hour', 1)}:{cfg.get('session_end_minute', 0):02d} {cfg.get('timezone', 'US/Eastern')}")
        print(f"Session 2 (London/NY): {cfg.get('session_start_hour_2', 8)}:{cfg.get('session_start_minute_2', 0):02d} - "
              f"{cfg.get('session_end_hour_2', 12)}:{cfg.get('session_end_minute_2', 0):02d} {cfg.get('timezone', 'US/Eastern')}")
        print(f"Daily Profit Cap: ${cfg.get('daily_profit_cap', 500):,.2f}")
        print(f"Max Daily Loss: ${cfg.get('max_daily_loss', 500):,.2f}")
        print(f"Max Total Loss: ${cfg.get('max_total_loss', 1000):,.2f}")
        
    elif args.strategy == 'v2':
        print(f"Strategy Version: {cfg.get('strategy_version', 'V2.0')}")
        print(f"Timeframe: {cfg.get('timeframe', '1min')}")
        print(f"EMA Periods: {cfg.get('ema_periods', [200, 50, 21])}")
        print(f"Z-Score Period: {cfg.get('z_period_v2', 50)}")
        print(f"Z-Score Threshold: ±{cfg.get('z_threshold_v2', 1.75)}")
        print(f"ATR Period: {cfg.get('atr_period_v2', 14)}")
        print(f"ATR Multiplier: {cfg.get('atr_multiplier_v2', 1.0)}")
        print(f"Risk per Trade: {cfg.get('risk_per_trade_v2', 0.015):.1%}")
        print(f"Risk-Reward Ratio: 1:{cfg.get('risk_reward_v2', 1.25)}")
        print(f"Session: {cfg.get('session_start_hour', 9)}:{cfg.get('session_start_minute', 30):02d} - "
              f"{cfg.get('session_end_hour', 11)}:{cfg.get('session_end_minute', 0):02d} {cfg.get('timezone', 'US/Eastern')}")
        print(f"Daily Profit Cap: ${cfg.get('daily_profit_cap', 500):,.2f}")
        print(f"Max Daily Loss: ${cfg.get('max_daily_loss', 500):,.2f}")
        print(f"Max Total Loss: ${cfg.get('max_total_loss', 1000):,.2f}")
        
    else:
        print(f"Strategy Version: V1 (Legacy)")
        print(f"Timeframe: {cfg.get('timeframe', '15m')}")
        print(f"Z-Score Period: {cfg.get('z_period', 20)}")
        print(f"Z-Score Threshold: ±{cfg.get('z_thresh', 2.0)}")
        print(f"ATR Period: {cfg.get('atr_period', 14)}")
        print(f"ATR Multiplier: {cfg.get('atr_mult', 2.0)}")
        print(f"Risk Base: {cfg.get('risk_base_pct', 0.01):.1%}")
        print(f"Risk-Reward Ratio: 1:{cfg.get('rr', 2.0)}")
        print(f"Max Daily Loss: ${cfg.get('max_daily_loss', 500):,.2f}")
        print(f"Max Total Loss: ${cfg.get('max_total_loss', 1000):,.2f}")
    
    print("=" * 50)

if __name__ == "__main__":
    sys.exit(main())
