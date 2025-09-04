"""
Backtester V2.0 - Enhanced Event-Driven Backtesting for Ronin V2.0
Includes daily profit cap, session filtering, and enhanced logging.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
from datetime import datetime, date
import os

from ronin_engine_v2 import build_order_v2


def run_backtest_v2(df: pd.DataFrame, cfg: Dict[str, Any], start_equity: float = 10000.0, 
                    seed: int = 42) -> Dict[str, Any]:
    """
    Run event-driven backtest for Ronin V2.0 strategy.
    
    Args:
        df: DataFrame with OHLCV data and indicators/signals (1-minute)
        cfg: Configuration dictionary
        start_equity: Starting account equity
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary with backtest results and metrics
    """
    np.random.seed(seed)
    
    print(f"🔄 Starting Ronin V2.0 backtest with ${start_equity:,.2f} starting equity")
    
    # Initialize backtest state
    equity = start_equity
    daily_pnl = 0.0
    total_pnl = 0.0
    open_orders = []
    completed_trades = []
    equity_curve = []
    daily_stats = {}
    current_date = None
    
    # V2.0 Risk Controls
    max_daily_loss = cfg['max_daily_loss']
    max_total_loss = cfg['max_total_loss']
    daily_profit_cap = cfg['daily_profit_cap']
    
    # Trading state flags
    trading_halted = False
    daily_profit_cap_reached = False
    
    # Process each bar
    for i in range(len(df)):
        current_bar = df.iloc[i]
        bar_date = current_bar['Time'].date()
        
        # Reset daily tracking on new day
        if current_date != bar_date:
            if current_date is not None:
                daily_stats[current_date] = {
                    'pnl': daily_pnl,
                    'equity': equity,
                    'trades': len([t for t in completed_trades if t['exit_time'].date() == current_date]),
                    'profit_cap_reached': daily_profit_cap_reached,
                    'trading_halted': trading_halted
                }
            
            current_date = bar_date
            daily_pnl = 0.0
            daily_profit_cap_reached = False
            trading_halted = False  # Reset daily halt (but not total loss halt)
        
        # Check for exits on existing orders
        if open_orders:
            day_trades = walk_exits_v2(df.iloc[i:i+1], open_orders, cfg)
            
            for trade in day_trades:
                completed_trades.append(trade)
                equity += trade['pnl']
                daily_pnl += trade['pnl']
                total_pnl += trade['pnl']
                
                print(f"📈 Trade closed: {trade['signal_type']} {trade['direction']} "
                      f"{trade['pnl']:+.2f} | Equity: ${equity:,.2f}")
        
        # Check V2.0 risk controls
        risk_check = check_risk_controls_v2(equity, daily_pnl, total_pnl, start_equity, cfg)
        if not risk_check['can_trade']:
            if 'daily_profit_cap' in risk_check['reason']:
                daily_profit_cap_reached = True
                print(f"🎯 Daily profit cap reached: ${daily_pnl:+.2f}")
            else:
                trading_halted = True
                print(f"🚫 Trading halted: {risk_check['reason']}")
            
            if 'total_loss' in risk_check['reason']:
                break  # Stop entire backtest
        
        # Only trade during session and if not halted
        if not current_bar.get('in_session', False) or trading_halted or daily_profit_cap_reached:
            continue
        
        # Look for new signals (only if no open positions for single-position mode)
        if cfg['one_position_at_a_time'] and len(open_orders) > 0:
            continue
            
        if current_bar.get('signal_long', False) or current_bar.get('signal_short', False):
            # Need next bar for entry
            if i + 1 >= len(df):
                continue
                
            entry_bar = df.iloc[i + 1]
            
            # Build order using V2.0 logic
            order = build_order_v2(current_bar, entry_bar, equity, daily_pnl, cfg)
            
            if order is not None:
                open_orders.append(order)
                print(f"🎯 New {order['signal_type']} {order['direction']} signal: "
                      f"Entry ${order['entry_price']:.2f}, SL ${order['stop_loss']:.2f}, "
                      f"TP ${order['take_profit']:.2f}")
        
        # Record equity curve point
        equity_curve.append({
            'Time': current_bar['Time'],
            'Equity': equity,
            'Daily_PnL': daily_pnl,
            'Total_PnL': total_pnl,
            'Open_Positions': len(open_orders),
            'In_Session': current_bar.get('in_session', False),
            'Trading_Halted': trading_halted,
            'Profit_Cap_Reached': daily_profit_cap_reached
        })
    
    # Close final day stats
    if current_date is not None:
        daily_stats[current_date] = {
            'pnl': daily_pnl,
            'equity': equity,
            'trades': len([t for t in completed_trades if t['exit_time'].date() == current_date]),
            'profit_cap_reached': daily_profit_cap_reached,
            'trading_halted': trading_halted
        }
    
    # Calculate metrics
    metrics = calculate_metrics_v2(completed_trades, start_equity, equity, daily_stats)
    
    print(f"✅ Ronin V2.0 backtest complete: {len(completed_trades)} trades, "
          f"Final equity: ${equity:,.2f}")
    
    results = {
        'trades': completed_trades,
        'equity_curve': pd.DataFrame(equity_curve),
        'daily_stats': daily_stats,
        'metrics': metrics,
        'final_equity': equity,
        'total_pnl': total_pnl,
        'open_orders': open_orders,
        'strategy_version': 'V2.0'
    }
    
    return results


def check_risk_controls_v2(equity: float, daily_pnl: float, total_pnl: float, 
                          start_equity: float, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check V2.0 risk controls including daily profit cap.
    
    Args:
        equity: Current equity
        daily_pnl: Daily P&L
        total_pnl: Total P&L
        start_equity: Starting equity
        cfg: Configuration
        
    Returns:
        Dictionary with can_trade flag and reason
    """
    # Check daily loss limit
    if daily_pnl <= -cfg['max_daily_loss']:
        return {
            'can_trade': False,
            'reason': f"daily_loss_limit: ${daily_pnl:.2f} <= -${cfg['max_daily_loss']:.2f}"
        }
    
    # Check total loss limit
    if total_pnl <= -cfg['max_total_loss']:
        return {
            'can_trade': False,
            'reason': f"total_loss_limit: ${total_pnl:.2f} <= -${cfg['max_total_loss']:.2f}"
        }
    
    # Check daily profit cap
    if daily_pnl >= cfg['daily_profit_cap']:
        return {
            'can_trade': False,
            'reason': f"daily_profit_cap: ${daily_pnl:.2f} >= ${cfg['daily_profit_cap']:.2f}"
        }
    
    return {'can_trade': True, 'reason': 'all_checks_passed'}


def walk_exits_v2(df: pd.DataFrame, orders: list, cfg: Dict[str, Any]) -> list:
    """
    Walk through bars and check for exit conditions on open orders (V2.0).
    
    Args:
        df: DataFrame with OHLCV data
        orders: List of open orders
        cfg: Configuration dictionary
        
    Returns:
        List of completed trades
    """
    completed_trades = []
    
    for i, row in df.iterrows():
        current_time = row['Time']
        high = row['High']
        low = row['Low']
        close = row['Close']
        
        # Check each open order for exit conditions
        orders_to_remove = []
        
        for j, order in enumerate(orders):
            if order is None:
                continue
                
            entry_time = order['entry_time']
            if current_time <= entry_time:
                continue  # Order not yet active
            
            direction = order['direction']
            entry_price = order['entry_price']
            stop_loss = order['stop_loss']
            take_profit = order['take_profit']
            position_size = order['position_size']
            
            exit_price = None
            exit_reason = None
            
            # Check for stop loss or take profit hits (first-touch basis)
            if direction == 'LONG':
                if low <= stop_loss:
                    exit_price = stop_loss
                    exit_reason = 'STOP_LOSS'
                elif high >= take_profit:
                    exit_price = take_profit
                    exit_reason = 'TAKE_PROFIT'
            else:  # SHORT
                if high >= stop_loss:
                    exit_price = stop_loss
                    exit_reason = 'STOP_LOSS'
                elif low <= take_profit:
                    exit_price = take_profit
                    exit_reason = 'TAKE_PROFIT'
            
            # If exit triggered, complete the trade
            if exit_price is not None:
                if direction == 'LONG':
                    pnl = (exit_price - entry_price) * position_size
                else:
                    pnl = (entry_price - exit_price) * position_size
                
                # Calculate duration
                entry_idx = df[df['Time'] == entry_time].index
                if len(entry_idx) > 0:
                    duration_bars = i - entry_idx[0]
                else:
                    duration_bars = 0
                
                trade = {
                    **order,
                    'exit_price': exit_price,
                    'exit_time': current_time,
                    'exit_reason': exit_reason,
                    'pnl': pnl,
                    'duration_bars': duration_bars,
                    'win': pnl > 0
                }
                
                completed_trades.append(trade)
                orders_to_remove.append(j)
        
        # Remove completed orders
        for j in reversed(orders_to_remove):
            orders.pop(j)
    
    return completed_trades


def calculate_metrics_v2(trades: List[Dict[str, Any]], start_equity: float, 
                        final_equity: float, daily_stats: Dict) -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics for V2.0 strategy.
    
    Args:
        trades: List of trade dictionaries
        start_equity: Starting account equity
        final_equity: Final account equity
        daily_stats: Daily statistics dictionary
        
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
            'sharpe_ratio': 0.0,
            'avg_daily_pnl': 0.0,
            'profitable_days': 0,
            'losing_days': 0,
            'profit_cap_days': 0
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
    
    # Sharpe ratio (simplified)
    if len(pnls) > 1:
        returns_std = np.std(pnls)
        sharpe_ratio = (np.mean(pnls) / returns_std) * np.sqrt(252) if returns_std > 0 else 0.0
    else:
        sharpe_ratio = 0.0
    
    # Daily statistics
    daily_pnls = [stats['pnl'] for stats in daily_stats.values()]
    avg_daily_pnl = np.mean(daily_pnls) if daily_pnls else 0.0
    profitable_days = len([pnl for pnl in daily_pnls if pnl > 0])
    losing_days = len([pnl for pnl in daily_pnls if pnl < 0])
    profit_cap_days = len([stats for stats in daily_stats.values() if stats.get('profit_cap_reached', False)])
    
    # Signal type breakdown
    bullish_trades = [t for t in trades if t.get('signal_type') == 'Bullish']
    bearish_trades = [t for t in trades if t.get('signal_type') == 'Bearish']
    
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
        'gross_loss': gross_loss,
        'avg_daily_pnl': avg_daily_pnl,
        'profitable_days': profitable_days,
        'losing_days': losing_days,
        'profit_cap_days': profit_cap_days,
        'bullish_trades': len(bullish_trades),
        'bearish_trades': len(bearish_trades),
        'bullish_win_rate': len([t for t in bullish_trades if t['win']]) / len(bullish_trades) if bullish_trades else 0.0,
        'bearish_win_rate': len([t for t in bearish_trades if t['win']]) / len(bearish_trades) if bearish_trades else 0.0
    }


def print_backtest_summary_v2(results: Dict[str, Any]) -> None:
    """
    Print formatted backtest summary for V2.0 strategy.
    
    Args:
        results: Backtest results dictionary
    """
    metrics = results['metrics']
    
    print("\n" + "=" * 70)
    print("📊 RONIN V2.0 BACKTEST RESULTS")
    print("=" * 70)
    
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
    print(f"   Average Daily P&L: ${metrics['avg_daily_pnl']:+.2f}")
    
    print(f"\n📉 Risk Metrics:")
    print(f"   Max Drawdown: {metrics['max_dd_pct']:.1f}% (${metrics['max_dd_usd']:.2f})")
    print(f"   Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"   Average Hold: {metrics['avg_hold_bars']:.1f} bars")
    
    print(f"\n🎯 V2.0 Strategy Breakdown:")
    print(f"   Bullish Trades: {metrics['bullish_trades']} (Win Rate: {metrics['bullish_win_rate']:.1%})")
    print(f"   Bearish Trades: {metrics['bearish_trades']} (Win Rate: {metrics['bearish_win_rate']:.1%})")
    print(f"   Profitable Days: {metrics['profitable_days']}")
    print(f"   Losing Days: {metrics['losing_days']}")
    print(f"   Profit Cap Days: {metrics['profit_cap_days']}")
    
    print(f"\n💼 Final Results:")
    print(f"   Final Equity: ${results['final_equity']:,.2f}")
    print(f"   Total P&L: ${results['total_pnl']:+,.2f}")
    print(f"   Open Orders: {len(results['open_orders'])}")
    
    print("=" * 70)


if __name__ == "__main__":
    from config import get_config
    
    cfg = get_config()
    print("✅ Ronin V2.0 Backtester loaded successfully")
    print(f"Strategy: {cfg.get('strategy_version', 'V2.0')}")
    print(f"Timeframe: {cfg.get('timeframe', '1min')}")
    print(f"Risk per trade: {cfg.get('risk_per_trade_v2', 0.015):.1%}")
    print(f"Daily profit cap: ${cfg.get('daily_profit_cap', 500):.2f}")
