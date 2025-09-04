"""
Logger - Trade and Equity Logging
Handles logging of trades, equity curve, and performance data to CSV files.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import os
from datetime import datetime


def write_trades(trades: List[Dict[str, Any]], out_dir: str) -> None:
    """
    Write trade log to CSV file.
    
    Args:
        trades: List of trade dictionaries
        out_dir: Output directory path
    """
    # Placeholder implementation - will be built in Stage 4
    os.makedirs(out_dir, exist_ok=True)
    
    if not trades:
        # Create empty trades file with headers
        columns = [
            'signal_time', 'entry_time', 'exit_time', 'side', 'entry', 'sl', 'tp',
            'exit_price', 'result', 'realized_R', 'z_at_signal', 'atr_at_signal',
            'risk_pct', 'qty', 'session'
        ]
        empty_df = pd.DataFrame(columns=columns)
        empty_df.to_csv(os.path.join(out_dir, 'trades.csv'), index=False)
        return
    
    # Convert trades to DataFrame and save
    df = pd.DataFrame(trades)
    df.to_csv(os.path.join(out_dir, 'trades.csv'), index=False)


def write_equity(equity_df: pd.DataFrame, out_dir: str) -> None:
    """
    Write equity curve to CSV file.
    
    Args:
        equity_df: DataFrame with equity curve data
        out_dir: Output directory path
    """
    # Placeholder implementation - will be built in Stage 4
    os.makedirs(out_dir, exist_ok=True)
    equity_df.to_csv(os.path.join(out_dir, 'equity_curve.csv'), index=False)


def write_metrics(metrics: Dict[str, Any], out_dir: str) -> None:
    """
    Write performance metrics to JSON and CSV files.
    
    Args:
        metrics: Dictionary with performance metrics
        out_dir: Output directory path
    """
    import json
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Write to JSON
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    # Write to CSV for easy viewing
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(out_dir, 'metrics.csv'), index=False)


def create_run_directory(base_dir: str = "runs") -> str:
    """
    Create timestamped run directory for output files.
    
    Args:
        base_dir: Base directory for runs
        
    Returns:
        Path to created run directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def log_trade_entry(trade_data: Dict[str, Any], log_file: str = None) -> None:
    """
    Log trade entry in real-time (for live trading).
    
    Args:
        trade_data: Dictionary with trade entry data
        log_file: Optional log file path
    """
    if log_file is None:
        print(f"ENTRY: {trade_data}")
    else:
        with open(log_file, 'a') as f:
            f.write(f"{datetime.now()}: ENTRY - {trade_data}\n")


def log_trade_exit(trade_data: Dict[str, Any], log_file: str = None) -> None:
    """
    Log trade exit in real-time (for live trading).
    
    Args:
        trade_data: Dictionary with trade exit data
        log_file: Optional log file path
    """
    if log_file is None:
        print(f"EXIT: {trade_data}")
    else:
        with open(log_file, 'a') as f:
            f.write(f"{datetime.now()}: EXIT - {trade_data}\n")


if __name__ == "__main__":
    print("✅ Logger module loaded successfully")
    print("Ready for trade and equity logging")
    
    # Test run directory creation
    test_dir = create_run_directory("test_runs")
    print(f"Test run directory created: {test_dir}")
    
    # Clean up test directory
    import shutil
    shutil.rmtree("test_runs", ignore_errors=True)
