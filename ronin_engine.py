"""
Ronin Engine - Signal Generation and Order Construction
Pure logic for trend-momentum signals and trade construction.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


def compute_indicators(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute all required indicators for signal generation.
    
    Args:
        df: DataFrame with OHLCV data
        cfg: Configuration dictionary
        
    Returns:
        DataFrame with added indicator columns
    """
    df = df.copy()
    
    # Z-score momentum indicator
    df = _compute_zscore(df, cfg)
    
    # ATR volatility filter
    df = _compute_atr(df, cfg)
    
    # EMA trend filters
    df = _compute_emas(df, cfg)
    
    # Volume filter
    df = _compute_volume_filter(df, cfg)
    
    print(f"✅ Computed indicators for {len(df)} bars")
    return df


def _compute_zscore(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Compute Z-score momentum indicator"""
    period = cfg['z_period']
    
    # Price change (close-to-close returns)
    df['price_change'] = df['Close'].pct_change()
    
    # Rolling mean and std for Z-score
    rolling_mean = df['price_change'].rolling(window=period).mean()
    rolling_std = df['price_change'].rolling(window=period).std()
    
    # Z-score calculation
    df['zscore'] = (df['price_change'] - rolling_mean) / rolling_std
    
    # Handle division by zero
    df['zscore'] = df['zscore'].fillna(0)
    
    return df


def _compute_atr(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Compute Average True Range for volatility filtering"""
    period = cfg['atr_period']
    
    # True Range calculation
    df['prev_close'] = df['Close'].shift(1)
    df['tr1'] = df['High'] - df['Low']
    df['tr2'] = abs(df['High'] - df['prev_close'])
    df['tr3'] = abs(df['Low'] - df['prev_close'])
    
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # ATR as simple moving average of True Range
    df['atr'] = df['true_range'].rolling(window=period).mean()
    
    # ATR as percentage of price for normalization
    df['atr_pct'] = (df['atr'] / df['Close']) * 100
    
    # Clean up intermediate columns
    df = df.drop(['prev_close', 'tr1', 'tr2', 'tr3'], axis=1)
    
    return df


def _compute_emas(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Compute EMA trend filters"""
    fast_period = cfg['ema_fast']
    slow_period = cfg['ema_slow']
    
    # Fast and slow EMAs
    df['ema_fast'] = df['Close'].ewm(span=fast_period).mean()
    df['ema_slow'] = df['Close'].ewm(span=slow_period).mean()
    
    # Trend direction
    df['trend_up'] = df['ema_fast'] > df['ema_slow']
    df['trend_dn'] = df['ema_fast'] < df['ema_slow']
    
    # Price above/below EMAs
    df['price_above_fast'] = df['Close'] > df['ema_fast']
    df['price_above_slow'] = df['Close'] > df['ema_slow']
    
    return df


def _compute_volume_filter(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Compute volume-based filters"""
    vol_period = cfg['vol_period']
    
    # Average volume
    df['vol_avg'] = df['Volume'].rolling(window=vol_period).mean()
    
    # Volume above average (confirmation filter)
    df['vol_above_avg'] = df['Volume'] > df['vol_avg']
    
    return df


def generate_signals(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Generate long/short signals based on trend-momentum conditions.
    
    Args:
        df: DataFrame with indicators
        cfg: Configuration dictionary
        
    Returns:
        DataFrame with signal columns added
    """
    df = df.copy()
    
    # Initialize signal columns
    df['signal_long'] = False
    df['signal_short'] = False
    df['signal_strength'] = 0.0
    
    # Z-score thresholds
    z_thresh = cfg['z_thresh']
    atr_min = cfg['atr_min']
    
    # Long signal conditions
    long_conditions = (
        (df['zscore'] > z_thresh) &                    # Strong positive momentum
        (df['trend_up']) &                             # Uptrend (EMA fast > slow)
        (df['price_above_fast']) &                     # Price above fast EMA
        (df['atr_pct'] > atr_min) &                   # Sufficient volatility
        (df['vol_above_avg'])                          # Volume confirmation
    )
    
    # Short signal conditions  
    short_conditions = (
        (df['zscore'] < -z_thresh) &                   # Strong negative momentum
        (df['trend_dn']) &                             # Downtrend (EMA fast < slow)
        (~df['price_above_fast']) &                    # Price below fast EMA
        (df['atr_pct'] > atr_min) &                   # Sufficient volatility
        (df['vol_above_avg'])                          # Volume confirmation
    )
    
    # Apply signal conditions
    df.loc[long_conditions, 'signal_long'] = True
    df.loc[short_conditions, 'signal_short'] = True
    
    # Signal strength based on Z-score magnitude
    df.loc[df['signal_long'], 'signal_strength'] = df.loc[df['signal_long'], 'zscore']
    df.loc[df['signal_short'], 'signal_strength'] = abs(df.loc[df['signal_short'], 'zscore'])
    
    # Count signals
    long_signals = df['signal_long'].sum()
    short_signals = df['signal_short'].sum()
    
    print(f"✅ Generated {long_signals} long and {short_signals} short signals")
    
    return df


def build_order(row_signal: pd.Series, row_entry: pd.Series, equity: float, 
                intraday_pl: float, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Construct order details including entry, SL, TP, and position sizing.
    
    Args:
        row_signal: Signal bar data
        row_entry: Entry bar data (i+1 from signal)
        equity: Current account equity
        intraday_pl: Intraday realized P&L
        cfg: Configuration dictionary
        
    Returns:
        Dictionary with order details
    """
    # Determine signal direction
    is_long = row_signal['signal_long']
    is_short = row_signal['signal_short']
    
    if not (is_long or is_short):
        return None
    
    # Entry price (open of next bar)
    entry_price = row_entry['Open']
    
    # ATR for stop/target calculation
    atr = row_signal['atr']
    atr_mult = cfg['atr_mult']
    rr_ratio = cfg['rr']
    
    # Calculate stop loss and take profit
    if is_long:
        stop_loss = entry_price - (atr * atr_mult)
        take_profit = entry_price + (atr * atr_mult * rr_ratio)
        direction = 'LONG'
    else:
        stop_loss = entry_price + (atr * atr_mult)
        take_profit = entry_price - (atr * atr_mult * rr_ratio)
        direction = 'SHORT'
    
    # Position sizing based on risk management
    risk_amount = _calculate_risk_amount(equity, intraday_pl, cfg)
    
    # Position size based on stop loss distance
    stop_distance = abs(entry_price - stop_loss)
    position_size = risk_amount / stop_distance if stop_distance > 0 else 0
    
    # Order details
    order = {
        'direction': direction,
        'entry_price': entry_price,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'position_size': position_size,
        'risk_amount': risk_amount,
        'signal_strength': row_signal['signal_strength'],
        'atr': atr,
        'entry_time': row_entry['Time'],
        'signal_time': row_signal['Time']
    }
    
    return order


def _calculate_risk_amount(equity: float, intraday_pl: float, cfg: Dict[str, Any]) -> float:
    """Calculate risk amount based on current equity and P&L"""
    base_risk_pct = cfg['risk_base_pct']
    red_risk_pct = cfg['risk_red_pct']
    up_risk_pct = cfg['risk_up_pct']
    
    # Adjust risk based on intraday P&L
    if intraday_pl < 0:
        # Reduce risk when losing
        risk_pct = red_risk_pct
    elif intraday_pl > 500:  # After $500 profit
        # Increase risk when winning
        risk_pct = up_risk_pct
    else:
        # Base risk
        risk_pct = base_risk_pct
    
    return equity * risk_pct


def walk_exits(df: pd.DataFrame, orders: list, cfg: Dict[str, Any]) -> list:
    """
    Walk through bars and check for exit conditions on open orders.
    
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
            
            # Check for stop loss or take profit hits
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
                
                trade = {
                    **order,
                    'exit_price': exit_price,
                    'exit_time': current_time,
                    'exit_reason': exit_reason,
                    'pnl': pnl,
                    'duration_bars': i - df[df['Time'] == entry_time].index[0]
                }
                
                completed_trades.append(trade)
                orders_to_remove.append(j)
        
        # Remove completed orders
        for j in reversed(orders_to_remove):
            orders.pop(j)
    
    return completed_trades


if __name__ == "__main__":
    print("✅ Ronin Engine module loaded successfully")
    print("Signal generation and order construction logic ready for implementation")
