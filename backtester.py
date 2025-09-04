"""
Backtester - Event-Driven Backtesting Engine
Handles backtesting with first-touch exits and performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from datetime import datetime
import os


def run_backtest(df: pd.DataFrame, cfg: Dict[str, Any], start_equity: float = 10000.0, 
                 seed: int = 42) -> Dict[str, Any]:
    """
    Run event-driven backtest across entire dataset.
    
    Args:
        df: DataFrame with OHLCV data and indicators/signals
        cfg: Configuration dictionary
        start_equity: Starting account equity
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with backtest results and metrics
    """
    from ronin_engine import build_order, walk_exits
    from risk_manager import check_ftmo_limits, update_daily_pnl
    
    np.random.seed(seed)
    
    print(f"🔄 Starting backtest with ${start_equity:,.2f} starting equity")
    
    # Initialize backtest state
    equity = start_equity
    daily_pnl = 0.0
    total_pnl = 0.0
    open_orders = []
    completed_trades = []
    equity_curve = []
    daily_stats = {}
    current_date = None
    
    # Track FTMO limits
    max_daily_loss = cfg['max_daily_loss']
    max_total_loss = cfg['max_total_loss']
    profit_target = cfg['profit_target']
    
    # Process each bar
    for i in range(len(df)):
        current_bar = df.iloc[i]
        bar_date = current_bar['Time'].date()
        
        # Reset daily P&L on new day
        if current_date != bar_date:
            if current_date is not None:
                daily_stats[current_date] = {
                    'pnl': daily_pnl,
                    'equity': equity,
                    'trades': len([t for t in completed_trades if t['exit_time'].date() == current_date])
                }
            current_date = bar_date
            daily_pnl = 0.0
        
        # Check for exits on existing orders
        if open_orders:
            day_trades = walk_exits(df.iloc[i:i+1], open_orders, cfg)
            
            for trade in day_trades:
                completed_trades.append(trade)
                equity += trade['pnl']
                daily_pnl += trade['pnl']
                total_pnl += trade['pnl']
                
                print(f"📈 Trade closed: {trade['direction']} {trade['pnl']:+.2f} | Equity: ${equity:,.2f}")
        
        # Check FTMO limits before new trades
        ftmo_check = check_ftmo_limits(equity, daily_pnl, total_pnl, start_equity, cfg)
        if not ftmo_check['can_trade']:
            print(f"🚫 FTMO limit hit: {ftmo_check['reason']}")
            break
        
        # Look for new signals (only if no open positions for single-position mode)
        if cfg['one_position_at_a_time'] and len(open_orders) > 0:
            continue
            
        if current_bar.get('signal_long', False) or current_bar.get('signal_short', False):
            # Need next bar for entry
            if i + 1 >= len(df):
                continue
                
            entry_bar = df.iloc[i + 1]
            
            # Build order
            order = build_order(current_bar, entry_bar, equity, daily_pnl, cfg)
            
            if order is not None:
                open_orders.append(order)
                print(f"🎯 New {order['direction']} signal: Entry ${order['entry_price']:.2f}, SL ${order['stop_loss']:.2f}, TP ${order['take_profit']:.2f}")
        
        # Record equity curve point
        equity_curve.append({
            'Time': current_bar['Time'],
            'Equity': equity,
            'Daily_PnL': daily_pnl,
            'Total_PnL': total_pnl,
            'Open_Positions': len(open_orders)
        })
    
    # Close final day stats
    if current_date is not None:
        daily_stats[current_date] = {
            'pnl': daily_pnl,
            'equity': equity,
            'trades': len([t for t in completed_trades if t['exit_time'].date() == current_date])
        }
    
    # Calculate metrics
    metrics = calculate_metrics(completed_trades, start_equity, equity)
    
    print(f"✅ Backtest complete: {len(completed_trades)} trades, Final equity: ${equity:,.2f}")
    
    results = {
        'trades': completed_trades,
        'equity_curve': pd.DataFrame(equity_curve),
        'daily_stats': daily_stats,
        'metrics': metrics,
        'final_equity': equity,
        'total_pnl': total_pnl,
        'open_orders': open_orders
    }
    
    return results


def calculate_metrics(trades: List[Dict[str, Any]], start_equity: float, final_equity: float) -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics from trade list.
    
    Args:
        trades: List of trade dictionaries
        start_equity: Starting account equity
        final_equity: Final account equity
        
    Returns:
        Dictionary with calculated metrics
    """
    if not trades:
        return {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'expectancy': 0.0,
            'avg_hold_bars': 0.0,
            'max_dd_pct': 0.0,
            'max_dd_usd': 0.0,
            'total_return_pct': 0.0,
            'sharpe_ratio': 0.0
        }
    
    # Basic trade statistics
    total_trades = len(trades)
    pnls = [trade['pnl'] for trade in trades]
    wins = [pnl for pnl in pnls if pnl > 0]
    losses = [pnl for pnl in pnls if pnl < 0]
    
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = win_count / total_trades if total_trades > 0 else 0.0
    
    # Profit factor
    gross_profit = sum(wins) if wins else 0.0
    gross_loss = abs(sum(losses)) if losses else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
    
    # Expectancy
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0
    expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
    # Hold time
    hold_times = [trade.get('duration_bars', 0) for trade in trades]
    avg_hold_bars = np.mean(hold_times) if hold_times else 0.0
    
    # Drawdown calculation
    equity_curve = []
    running_equity = start_equity
    peak_equity = start_equity
    max_dd_usd = 0.0
    max_dd_pct = 0.0
    
    for trade in trades:
        running_equity += trade['pnl']
        equity_curve.append(running_equity)
        
        if running_equity > peak_equity:
            peak_equity = running_equity
        
        drawdown_usd = peak_equity - running_equity
        drawdown_pct = (drawdown_usd / peak_equity) * 100 if peak_equity > 0 else 0.0
        
        if drawdown_usd > max_dd_usd:
            max_dd_usd = drawdown_usd
        if drawdown_pct > max_dd_pct:
            max_dd_pct = drawdown_pct
    
    # Total return
    total_return_pct = ((final_equity - start_equity) / start_equity) * 100
    
    # Sharpe ratio (simplified - assumes daily returns)
    if len(pnls) > 1:
        returns_std = np.std(pnls)
        sharpe_ratio = (np.mean(pnls) / returns_std) * np.sqrt(252) if returns_std > 0 else 0.0
    else:
        sharpe_ratio = 0.0
    
    return {
        'total_trades': total_trades,
        'wins': win_count,
        'losses': loss_count,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'expectancy': expectancy,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'avg_hold_bars': avg_hold_bars,
        'max_dd_pct': max_dd_pct,
        'max_dd_usd': max_dd_usd,
        'total_return_pct': total_return_pct,
        'sharpe_ratio': sharpe_ratio,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss
    }


def generate_equity_curve(trades: List[Dict[str, Any]], start_equity: float) -> pd.DataFrame:
    """
    Generate detailed equity curve from trade results.
    
    Args:
        trades: List of completed trades
        start_equity: Starting account equity
        
    Returns:
        DataFrame with equity curve data
    """
    if not trades:
        return pd.DataFrame(columns=['Time', 'Equity', 'Drawdown_Pct', 'Drawdown_USD'])
    
    equity_data = []
    running_equity = start_equity
    peak_equity = start_equity
    
    for trade in trades:
        running_equity += trade['pnl']
        
        if running_equity > peak_equity:
            peak_equity = running_equity
        
        drawdown_usd = peak_equity - running_equity
        drawdown_pct = (drawdown_usd / peak_equity) * 100 if peak_equity > 0 else 0.0
        
        equity_data.append({
            'Time': trade['exit_time'],
            'Equity': running_equity,
            'Drawdown_Pct': drawdown_pct,
            'Drawdown_USD': drawdown_usd,
            'Trade_PnL': trade['pnl'],
            'Peak_Equity': peak_equity
        })
    
    return pd.DataFrame(equity_data)


def save_backtest_results(results: Dict[str, Any], output_dir: str) -> None:
    """
    Save backtest results to files.
    
    Args:
        results: Backtest results dictionary
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save trades
    if results['trades']:
        trades_df = pd.DataFrame(results['trades'])
        trades_df.to_csv(os.path.join(output_dir, 'trades.csv'), index=False)
        print(f"💾 Saved {len(results['trades'])} trades to {output_dir}/trades.csv")
    
    # Save equity curve
    if not results['equity_curve'].empty:
        results['equity_curve'].to_csv(os.path.join(output_dir, 'equity_curve.csv'), index=False)
        print(f"💾 Saved equity curve to {output_dir}/equity_curve.csv")
    
    # Save metrics
    metrics_df = pd.DataFrame([results['metrics']])
    metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
    print(f"💾 Saved metrics to {output_dir}/metrics.csv")
    
    # Save summary report
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write("RONIN BOT BACKTEST SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total Trades: {results['metrics']['total_trades']}\n")
        f.write(f"Win Rate: {results['metrics']['win_rate']:.1%}\n")
        f.write(f"Profit Factor: {results['metrics']['profit_factor']:.2f}\n")
        f.write(f"Expectancy: ${results['metrics']['expectancy']:.2f}\n")
        f.write(f"Total Return: {results['metrics']['total_return_pct']:.1f}%\n")
        f.write(f"Max Drawdown: {results['metrics']['max_dd_pct']:.1f}% (${results['metrics']['max_dd_usd']:.2f})\n")
        f.write(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}\n")
        f.write(f"Average Hold: {results['metrics']['avg_hold_bars']:.1f} bars\n")
        
        f.write(f"\nFinal Equity: ${results['final_equity']:,.2f}\n")
        f.write(f"Total P&L: ${results['total_pnl']:+,.2f}\n")
    
    print(f"💾 Saved summary report to {output_dir}/summary.txt")


def print_backtest_summary(results: Dict[str, Any]) -> None:
    """
    Print formatted backtest summary to console.
    
    Args:
        results: Backtest results dictionary
    """
    metrics = results['metrics']
    
    print("\n" + "=" * 60)
    print("📊 RONIN BOT BACKTEST RESULTS")
    print("=" * 60)
    
    print(f"📈 Performance Metrics:")
    print(f"   Total Trades: {metrics['total_trades']}")
    print(f"   Win Rate: {metrics['win_rate']:.1%} ({metrics['wins']}W / {metrics['losses']}L)")
    print(f"   Profit Factor: {metrics['profit_factor']:.2f}")
    print(f"   Expectancy: ${metrics['expectancy']:+.2f}")
    print(f"   Total Return: {metrics['total_return_pct']:+.1f}%")
    
    print(f"\n💰 P&L Analysis:")
    print(f"   Gross Profit: ${metrics['gross_profit']:+,.2f}")
    print(f"   Gross Loss: ${metrics['gross_loss']:+,.2f}")
    print(f"   Average Win: ${metrics['avg_win']:+.2f}")
    print(f"   Average Loss: ${metrics['avg_loss']:+.2f}")
    
    print(f"\n📉 Risk Metrics:")
    print(f"   Max Drawdown: {metrics['max_dd_pct']:.1f}% (${metrics['max_dd_usd']:.2f})")
    print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"   Average Hold: {metrics['avg_hold_bars']:.1f} bars")
    
    print(f"\n💼 Final Results:")
    print(f"   Final Equity: ${results['final_equity']:,.2f}")
    print(f"   Total P&L: ${results['total_pnl']:+,.2f}")
    print(f"   Open Orders: {len(results['open_orders'])}")
    
    print("=" * 60)


if __name__ == "__main__":
    from config import get_config
    
    cfg = get_config()
    print("✅ Backtester module loaded successfully")
    print(f"Ready for event-driven backtesting with {cfg['timeframe']} timeframe")
    print(f"One position at a time: {cfg['one_position_at_a_time']}")
    print(f"Confirm next bar: {cfg['confirm_next_bar']}")
