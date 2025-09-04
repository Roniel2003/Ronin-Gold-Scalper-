"""
Monte Carlo Simulation Engine - Stage 4
Provides robustness testing and statistical analysis of trading strategies.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple, Any
import json
import os


def bootstrap_trades(trades: List[Dict], n_simulations: int = 1000, 
                    block_size: int = 5) -> List[List[Dict]]:
    """
    Bootstrap trade sequences using block bootstrap to preserve serial correlation.
    
    Args:
        trades: List of trade dictionaries with 'pnl' and other fields
        n_simulations: Number of bootstrap simulations to run
        block_size: Size of blocks to preserve correlation structure
        
    Returns:
        List of bootstrapped trade sequences
    """
    if len(trades) == 0:
        return [[] for _ in range(n_simulations)]
    
    simulations = []
    n_trades = len(trades)
    
    for _ in range(n_simulations):
        bootstrapped_trades = []
        
        # Create blocks
        blocks = []
        for i in range(0, n_trades, block_size):
            block = trades[i:i + block_size]
            if block:  # Only add non-empty blocks
                blocks.append(block)
        
        if not blocks:
            simulations.append([])
            continue
        
        # Sample blocks with replacement
        target_trades = n_trades
        while len(bootstrapped_trades) < target_trades:
            block = random.choice(blocks)
            bootstrapped_trades.extend(block)
        
        # Trim to exact length
        bootstrapped_trades = bootstrapped_trades[:target_trades]
        simulations.append(bootstrapped_trades)
    
    return simulations


def run_monte_carlo_analysis(trades: List[Dict], start_equity: float = 10000.0,
                           n_simulations: int = 1000, 
                           confidence_levels: List[float] = [0.05, 0.95]) -> Dict:
    """
    Run comprehensive Monte Carlo analysis on trade results.
    
    Args:
        trades: List of historical trades
        start_equity: Starting equity for simulations
        n_simulations: Number of Monte Carlo simulations
        confidence_levels: Confidence intervals to calculate
        
    Returns:
        Dictionary with Monte Carlo results and statistics
    """
    if len(trades) == 0:
        return {
            'simulations': 0,
            'final_equity': {'mean': start_equity, 'std': 0.0, 'percentiles': {}},
            'total_return': {'mean': 0.0, 'std': 0.0, 'percentiles': {}},
            'max_drawdown': {'mean': 0.0, 'std': 0.0, 'percentiles': {}},
            'win_rate': {'mean': 0.0, 'std': 0.0, 'percentiles': {}},
            'profit_factor': {'mean': 0.0, 'std': 0.0, 'percentiles': {}},
            'risk_of_ruin': 0.0,
            'var_95': 0.0,
            'expected_shortfall': 0.0
        }
    
    # Generate bootstrapped trade sequences
    bootstrapped_sequences = bootstrap_trades(trades, n_simulations)
    
    # Run simulations
    results = {
        'final_equity': [],
        'total_return': [],
        'max_drawdown': [],
        'win_rate': [],
        'profit_factor': [],
        'consecutive_losses': []
    }
    
    for sequence in bootstrapped_sequences:
        if not sequence:
            continue
            
        # Calculate equity curve for this simulation
        equity = start_equity
        equity_curve = [equity]
        peak_equity = equity
        max_dd = 0.0
        wins = 0
        losses = 0
        gross_profit = 0.0
        gross_loss = 0.0
        consecutive_loss_count = 0
        max_consecutive_losses = 0
        
        for trade in sequence:
            pnl = trade.get('pnl', 0.0)
            equity += pnl
            equity_curve.append(equity)
            
            # Track drawdown
            if equity > peak_equity:
                peak_equity = equity
            else:
                drawdown = (peak_equity - equity) / peak_equity * 100
                max_dd = max(max_dd, drawdown)
            
            # Track win/loss statistics
            if pnl > 0:
                wins += 1
                gross_profit += pnl
                consecutive_loss_count = 0
            elif pnl < 0:
                losses += 1
                gross_loss += abs(pnl)
                consecutive_loss_count += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_loss_count)
        
        # Calculate metrics for this simulation
        total_trades = wins + losses
        win_rate = wins / total_trades if total_trades > 0 else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
        total_return = (equity - start_equity) / start_equity * 100
        
        results['final_equity'].append(equity)
        results['total_return'].append(total_return)
        results['max_drawdown'].append(max_dd)
        results['win_rate'].append(win_rate)
        results['profit_factor'].append(profit_factor if profit_factor != float('inf') else 10.0)  # Cap for stats
        results['consecutive_losses'].append(max_consecutive_losses)
    
    # Calculate statistics
    def calc_stats(values: List[float]) -> Dict:
        if not values:
            return {'mean': 0.0, 'std': 0.0, 'percentiles': {}}
        
        values_array = np.array(values)
        percentiles = {}
        for cl in confidence_levels:
            percentiles[f'p{int(cl*100)}'] = np.percentile(values_array, cl * 100)
        
        return {
            'mean': float(np.mean(values_array)),
            'std': float(np.std(values_array)),
            'percentiles': percentiles
        }
    
    # Risk of ruin (probability of losing more than 50% of capital)
    ruin_threshold = start_equity * 0.5
    risk_of_ruin = sum(1 for eq in results['final_equity'] if eq < ruin_threshold) / len(results['final_equity'])
    
    # Value at Risk (95% confidence level)
    returns = results['total_return']
    var_95 = np.percentile(returns, 5) if returns else 0.0
    
    # Expected Shortfall (average of worst 5% outcomes)
    sorted_returns = sorted(returns)
    worst_5pct = sorted_returns[:max(1, len(sorted_returns) // 20)]
    expected_shortfall = np.mean(worst_5pct) if worst_5pct else 0.0
    
    return {
        'simulations': len(bootstrapped_sequences),
        'final_equity': calc_stats(results['final_equity']),
        'total_return': calc_stats(results['total_return']),
        'max_drawdown': calc_stats(results['max_drawdown']),
        'win_rate': calc_stats(results['win_rate']),
        'profit_factor': calc_stats(results['profit_factor']),
        'consecutive_losses': calc_stats(results['consecutive_losses']),
        'risk_of_ruin': risk_of_ruin,
        'var_95': var_95,
        'expected_shortfall': expected_shortfall,
        'raw_results': results  # For detailed analysis
    }


def generate_monte_carlo_report(mc_results: Dict, output_path: str = None) -> str:
    """
    Generate a comprehensive Monte Carlo analysis report.
    
    Args:
        mc_results: Results from run_monte_carlo_analysis
        output_path: Optional path to save report
        
    Returns:
        Report text
    """
    report_lines = [
        "=" * 60,
        "MONTE CARLO ROBUSTNESS ANALYSIS",
        "=" * 60,
        "",
        f"Simulations Run: {mc_results['simulations']:,}",
        "",
        "PERFORMANCE DISTRIBUTION",
        "-" * 30,
        f"Final Equity:",
        f"  Mean: ${mc_results['final_equity']['mean']:,.2f}",
        f"  Std Dev: ${mc_results['final_equity']['std']:,.2f}",
        "",
        f"Total Return (%):",
        f"  Mean: {mc_results['total_return']['mean']:+.2f}%",
        f"  Std Dev: {mc_results['total_return']['std']:.2f}%",
        "",
        "CONFIDENCE INTERVALS",
        "-" * 30
    ]
    
    # Add percentile information
    for metric_name, metric_data in mc_results.items():
        if isinstance(metric_data, dict) and 'percentiles' in metric_data:
            if metric_data['percentiles']:
                report_lines.append(f"{metric_name.replace('_', ' ').title()}:")
                for pct_name, pct_value in metric_data['percentiles'].items():
                    if 'return' in metric_name or 'drawdown' in metric_name:
                        report_lines.append(f"  {pct_name}: {pct_value:+.2f}%")
                    elif 'equity' in metric_name:
                        report_lines.append(f"  {pct_name}: ${pct_value:,.2f}")
                    else:
                        report_lines.append(f"  {pct_name}: {pct_value:.3f}")
                report_lines.append("")
    
    # Risk metrics
    report_lines.extend([
        "RISK METRICS",
        "-" * 30,
        f"Risk of Ruin (>50% loss): {mc_results['risk_of_ruin']:.2%}",
        f"Value at Risk (95%): {mc_results['var_95']:+.2f}%",
        f"Expected Shortfall: {mc_results['expected_shortfall']:+.2f}%",
        "",
        f"Max Consecutive Losses:",
        f"  Mean: {mc_results['consecutive_losses']['mean']:.1f}",
        f"  Worst Case: {max(mc_results['raw_results']['consecutive_losses']) if mc_results['raw_results']['consecutive_losses'] else 0:.0f}",
        "",
        "ROBUSTNESS ASSESSMENT",
        "-" * 30
    ])
    
    # Robustness assessment
    total_return_mean = mc_results['total_return']['mean']
    total_return_std = mc_results['total_return']['std']
    
    if total_return_std > 0:
        stability_ratio = abs(total_return_mean) / total_return_std
        if stability_ratio > 1.0:
            stability_assessment = "HIGH - Strategy shows consistent performance"
        elif stability_ratio > 0.5:
            stability_assessment = "MEDIUM - Moderate performance variability"
        else:
            stability_assessment = "LOW - High performance variability"
    else:
        stability_assessment = "UNDEFINED - Insufficient variability"
    
    report_lines.extend([
        f"Performance Stability: {stability_assessment}",
        f"Stability Ratio: {stability_ratio:.2f}" if total_return_std > 0 else "Stability Ratio: N/A",
        ""
    ])
    
    # Risk assessment
    if mc_results['risk_of_ruin'] < 0.01:
        risk_assessment = "LOW - Very low probability of significant loss"
    elif mc_results['risk_of_ruin'] < 0.05:
        risk_assessment = "MODERATE - Acceptable risk level"
    else:
        risk_assessment = "HIGH - Significant risk of large losses"
    
    report_lines.extend([
        f"Risk Assessment: {risk_assessment}",
        "",
        "=" * 60
    ])
    
    report_text = "\n".join(report_lines)
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
    
    return report_text


def save_monte_carlo_results(mc_results: Dict, output_dir: str):
    """
    Save Monte Carlo results to files.
    
    Args:
        mc_results: Results from run_monte_carlo_analysis
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary statistics
    summary_stats = {k: v for k, v in mc_results.items() if k != 'raw_results'}
    with open(os.path.join(output_dir, 'mc_summary.json'), 'w') as f:
        json.dump(summary_stats, f, indent=2)
    
    # Save raw simulation results
    raw_results = mc_results['raw_results']
    results_df = pd.DataFrame(raw_results)
    results_df.to_csv(os.path.join(output_dir, 'mc_simulations.csv'), index=False)
    
    # Generate and save report
    report_text = generate_monte_carlo_report(mc_results)
    with open(os.path.join(output_dir, 'mc_report.txt'), 'w') as f:
        f.write(report_text)
    
    print(f"✅ Monte Carlo results saved to {output_dir}")


def print_monte_carlo_summary(mc_results: Dict):
    """Print a concise Monte Carlo summary to console."""
    print("\n🎲 Monte Carlo Analysis Summary")
    print("=" * 40)
    print(f"Simulations: {mc_results['simulations']:,}")
    print(f"Mean Return: {mc_results['total_return']['mean']:+.2f}%")
    print(f"Return Std Dev: {mc_results['total_return']['std']:.2f}%")
    print(f"Risk of Ruin: {mc_results['risk_of_ruin']:.2%}")
    print(f"VaR (95%): {mc_results['var_95']:+.2f}%")
    
    # Quick robustness assessment
    stability_ratio = abs(mc_results['total_return']['mean']) / mc_results['total_return']['std'] if mc_results['total_return']['std'] > 0 else 0
    print(f"Stability Ratio: {stability_ratio:.2f}")
    
    if stability_ratio > 1.0:
        print("🟢 Strategy shows HIGH robustness")
    elif stability_ratio > 0.5:
        print("🟡 Strategy shows MEDIUM robustness")
    else:
        print("🔴 Strategy shows LOW robustness")


if __name__ == "__main__":
    # Example usage
    sample_trades = [
        {'pnl': 100.0}, {'pnl': -50.0}, {'pnl': 150.0}, {'pnl': -75.0},
        {'pnl': 200.0}, {'pnl': -100.0}, {'pnl': 80.0}, {'pnl': -40.0}
    ]
    
    results = run_monte_carlo_analysis(sample_trades, 10000.0, 1000)
    print_monte_carlo_summary(results)
    print("\nFull Report:")
    print(generate_monte_carlo_report(results))
