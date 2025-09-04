"""
Parameter Optimization Framework - Stage 4
Provides grid search optimization and walk-forward analysis for strategy parameters.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


def generate_parameter_grid(param_ranges: Dict[str, List]) -> List[Dict]:
    """
    Generate all parameter combinations from ranges.
    
    Args:
        param_ranges: Dictionary with parameter names and their value lists
        
    Returns:
        List of parameter dictionaries
    """
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    
    combinations = list(itertools.product(*param_values))
    
    return [dict(zip(param_names, combo)) for combo in combinations]


def run_single_optimization(data: pd.DataFrame, params: Dict, 
                          base_config: Dict, start_equity: float = 10000.0) -> Dict:
    """
    Run backtest with specific parameter set.
    
    Args:
        data: OHLCV data for backtesting
        params: Parameter set to test
        base_config: Base configuration to modify
        start_equity: Starting equity
        
    Returns:
        Results dictionary with metrics and parameters
    """
    try:
        # Import here to avoid circular imports
        from data_loader import resample_to_15min
        from ronin_engine import compute_indicators, generate_signals
        from backtester import run_backtest
        
        # Create modified config
        test_config = base_config.copy()
        test_config.update(params)
        
        # Run the pipeline
        df_indicators = compute_indicators(data, test_config)
        df_signals = generate_signals(df_indicators, test_config)
        
        # Run backtest
        results = run_backtest(df_signals, test_config, start_equity)
        
        # Extract key metrics
        metrics = results['metrics']
        
        return {
            'parameters': params,
            'total_trades': metrics.get('total_trades', 0),
            'win_rate': metrics.get('win_rate', 0.0),
            'profit_factor': metrics.get('profit_factor', 0.0),
            'total_return_pct': metrics.get('total_return_pct', 0.0),
            'max_dd_pct': metrics.get('max_dd_pct', 0.0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
            'expectancy': metrics.get('expectancy', 0.0),
            'final_equity': results.get('final_equity', start_equity),
            'success': True
        }
        
    except Exception as e:
        return {
            'parameters': params,
            'error': str(e),
            'success': False
        }


def optimize_parameters(data: pd.DataFrame, param_ranges: Dict[str, List],
                       base_config: Dict, start_equity: float = 10000.0,
                       optimization_metric: str = 'sharpe_ratio',
                       min_trades: int = 10, max_workers: int = None) -> Dict:
    """
    Run parameter optimization using grid search.
    
    Args:
        data: OHLCV data for optimization
        param_ranges: Parameter ranges to test
        base_config: Base configuration
        start_equity: Starting equity
        optimization_metric: Metric to optimize for
        min_trades: Minimum trades required for valid result
        max_workers: Maximum parallel workers (None = auto)
        
    Returns:
        Optimization results dictionary
    """
    print(f"🔍 Starting parameter optimization...")
    
    # Generate parameter grid
    param_grid = generate_parameter_grid(param_ranges)
    total_combinations = len(param_grid)
    
    print(f"Testing {total_combinations:,} parameter combinations")
    print(f"Optimizing for: {optimization_metric}")
    
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), 8)
    
    # Run optimization
    results = []
    completed = 0
    
    # Sequential execution for now (can be parallelized later)
    for params in param_grid:
        result = run_single_optimization(data, params, base_config, start_equity)
        results.append(result)
        
        completed += 1
        if completed % max(1, total_combinations // 20) == 0:
            progress = completed / total_combinations * 100
            print(f"Progress: {progress:.1f}% ({completed}/{total_combinations})")
    
    # Filter successful results with minimum trades
    valid_results = [
        r for r in results 
        if r.get('success', False) and r.get('total_trades', 0) >= min_trades
    ]
    
    if not valid_results:
        return {
            'best_params': None,
            'best_result': None,
            'all_results': results,
            'valid_results': 0,
            'total_tested': total_combinations,
            'optimization_metric': optimization_metric
        }
    
    # Find best result
    if optimization_metric == 'sharpe_ratio':
        best_result = max(valid_results, key=lambda x: x.get('sharpe_ratio', -999))
    elif optimization_metric == 'total_return_pct':
        best_result = max(valid_results, key=lambda x: x.get('total_return_pct', -999))
    elif optimization_metric == 'profit_factor':
        best_result = max(valid_results, key=lambda x: x.get('profit_factor', 0))
    elif optimization_metric == 'win_rate':
        best_result = max(valid_results, key=lambda x: x.get('win_rate', 0))
    else:
        # Default to Sharpe ratio
        best_result = max(valid_results, key=lambda x: x.get('sharpe_ratio', -999))
    
    return {
        'best_params': best_result['parameters'],
        'best_result': best_result,
        'all_results': results,
        'valid_results': len(valid_results),
        'total_tested': total_combinations,
        'optimization_metric': optimization_metric,
        'param_ranges': param_ranges
    }


def walk_forward_analysis(data: pd.DataFrame, param_ranges: Dict[str, List],
                         base_config: Dict, window_months: int = 6,
                         step_months: int = 1, start_equity: float = 10000.0,
                         optimization_metric: str = 'sharpe_ratio') -> Dict:
    """
    Perform walk-forward analysis with parameter optimization.
    
    Args:
        data: Full dataset for analysis
        param_ranges: Parameter ranges for optimization
        base_config: Base configuration
        window_months: Optimization window in months
        step_months: Step size in months
        start_equity: Starting equity
        optimization_metric: Metric to optimize for
        
    Returns:
        Walk-forward analysis results
    """
    print(f"🚶 Starting walk-forward analysis...")
    print(f"Window: {window_months} months, Step: {step_months} months")
    
    # Ensure data is sorted by time
    data = data.sort_values('Time').reset_index(drop=True)
    
    # Calculate date ranges
    start_date = data['Time'].min()
    end_date = data['Time'].max()
    
    walk_results = []
    current_date = start_date
    
    while current_date + timedelta(days=window_months*30) <= end_date:
        # Define optimization window
        opt_start = current_date
        opt_end = current_date + timedelta(days=window_months*30)
        
        # Define out-of-sample test window
        test_start = opt_end
        test_end = opt_end + timedelta(days=step_months*30)
        
        if test_end > end_date:
            test_end = end_date
        
        print(f"Optimizing: {opt_start.date()} to {opt_end.date()}")
        print(f"Testing: {test_start.date()} to {test_end.date()}")
        
        # Get optimization data
        opt_data = data[(data['Time'] >= opt_start) & (data['Time'] < opt_end)].copy()
        
        # Get test data
        test_data = data[(data['Time'] >= test_start) & (data['Time'] < test_end)].copy()
        
        if len(opt_data) < 100 or len(test_data) < 20:
            print("Insufficient data for this window, skipping...")
            current_date += timedelta(days=step_months*30)
            continue
        
        # Run optimization on in-sample data
        opt_results = optimize_parameters(
            opt_data, param_ranges, base_config, start_equity, 
            optimization_metric, min_trades=5
        )
        
        if opt_results['best_params'] is None:
            print("No valid parameters found, using defaults...")
            best_params = {}
        else:
            best_params = opt_results['best_params']
        
        # Test on out-of-sample data
        test_result = run_single_optimization(test_data, best_params, base_config, start_equity)
        
        walk_results.append({
            'opt_period': {'start': opt_start, 'end': opt_end},
            'test_period': {'start': test_start, 'end': test_end},
            'best_params': best_params,
            'opt_result': opt_results['best_result'] if opt_results['best_result'] else {},
            'test_result': test_result,
            'valid_opt_results': opt_results['valid_results']
        })
        
        current_date += timedelta(days=step_months*30)
    
    # Calculate aggregate statistics
    test_returns = [r['test_result'].get('total_return_pct', 0) for r in walk_results if r['test_result'].get('success', False)]
    test_trades = [r['test_result'].get('total_trades', 0) for r in walk_results if r['test_result'].get('success', False)]
    
    aggregate_stats = {
        'total_periods': len(walk_results),
        'successful_periods': len(test_returns),
        'avg_return': np.mean(test_returns) if test_returns else 0.0,
        'std_return': np.std(test_returns) if test_returns else 0.0,
        'total_trades': sum(test_trades),
        'avg_trades_per_period': np.mean(test_trades) if test_trades else 0.0,
        'positive_periods': sum(1 for r in test_returns if r > 0),
        'negative_periods': sum(1 for r in test_returns if r < 0)
    }
    
    if len(test_returns) > 0:
        aggregate_stats['win_rate_periods'] = aggregate_stats['positive_periods'] / len(test_returns)
        if aggregate_stats['std_return'] > 0:
            aggregate_stats['sharpe_ratio'] = aggregate_stats['avg_return'] / aggregate_stats['std_return']
        else:
            aggregate_stats['sharpe_ratio'] = 0.0
    else:
        aggregate_stats['win_rate_periods'] = 0.0
        aggregate_stats['sharpe_ratio'] = 0.0
    
    return {
        'walk_results': walk_results,
        'aggregate_stats': aggregate_stats,
        'config': {
            'window_months': window_months,
            'step_months': step_months,
            'optimization_metric': optimization_metric,
            'param_ranges': param_ranges
        }
    }


def sensitivity_analysis(data: pd.DataFrame, base_params: Dict, 
                        base_config: Dict, sensitivity_ranges: Dict[str, List],
                        start_equity: float = 10000.0) -> Dict:
    """
    Perform sensitivity analysis around best parameters.
    
    Args:
        data: OHLCV data
        base_params: Best parameters from optimization
        base_config: Base configuration
        sensitivity_ranges: Parameter variations to test
        start_equity: Starting equity
        
    Returns:
        Sensitivity analysis results
    """
    print(f"📊 Running sensitivity analysis...")
    
    results = {}
    
    for param_name, param_values in sensitivity_ranges.items():
        param_results = []
        
        for value in param_values:
            # Create test parameters
            test_params = base_params.copy()
            test_params[param_name] = value
            
            # Run backtest
            result = run_single_optimization(data, test_params, base_config, start_equity)
            result['param_value'] = value
            param_results.append(result)
        
        results[param_name] = param_results
    
    return results


def save_optimization_results(results: Dict, output_dir: str):
    """Save optimization results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save main results
    with open(os.path.join(output_dir, 'optimization_results.json'), 'w') as f:
        # Convert datetime objects for JSON serialization
        json_results = results.copy()
        if 'walk_results' in json_results:
            for wr in json_results['walk_results']:
                for period_key in ['opt_period', 'test_period']:
                    if period_key in wr:
                        for date_key in ['start', 'end']:
                            if date_key in wr[period_key]:
                                wr[period_key][date_key] = wr[period_key][date_key].isoformat()
        
        json.dump(json_results, f, indent=2, default=str)
    
    # Save parameter grid results if available
    if 'all_results' in results:
        results_df = pd.DataFrame(results['all_results'])
        results_df.to_csv(os.path.join(output_dir, 'parameter_grid_results.csv'), index=False)
    
    print(f"✅ Optimization results saved to {output_dir}")


def print_optimization_summary(results: Dict):
    """Print optimization summary to console."""
    print("\n🎯 Optimization Summary")
    print("=" * 40)
    
    if results.get('best_params'):
        print(f"Best Parameters:")
        for param, value in results['best_params'].items():
            print(f"  {param}: {value}")
        
        best = results['best_result']
        print(f"\nBest Result ({results['optimization_metric']}):")
        print(f"  Total Return: {best.get('total_return_pct', 0):+.2f}%")
        print(f"  Sharpe Ratio: {best.get('sharpe_ratio', 0):.3f}")
        print(f"  Win Rate: {best.get('win_rate', 0):.1%}")
        print(f"  Max Drawdown: {best.get('max_dd_pct', 0):.1f}%")
        print(f"  Total Trades: {best.get('total_trades', 0)}")
    else:
        print("No valid parameters found!")
    
    print(f"\nTesting Summary:")
    print(f"  Valid Results: {results.get('valid_results', 0)}")
    print(f"  Total Tested: {results.get('total_tested', 0)}")


if __name__ == "__main__":
    # Example parameter ranges
    param_ranges = {
        'zscore_threshold': [1.5, 2.0, 2.5, 3.0],
        'atr_threshold': [0.5, 1.0, 1.5, 2.0],
        'ema_fast': [10, 15, 20],
        'ema_slow': [30, 40, 50]
    }
    
    print(f"Example parameter grid: {len(generate_parameter_grid(param_ranges))} combinations")
