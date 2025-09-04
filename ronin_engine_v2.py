"""
Ronin Engine V2.0 - 1-Minute Dual-Directional Momentum Strategy
Advanced order block and fair value gap detection with dynamic risk controls.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime, time
import logging

# Set up logging
logging.basicConfig(filename='live_trading.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def calculate_indicators_v2(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Calculate all indicators for Ronin V2.0 strategy.
    
    Args:
        df: DataFrame with OHLCV data (1-minute timeframe)
        cfg: Configuration dictionary
        
    Returns:
        DataFrame with added indicator columns
    """
    logger.info(f"🔧 Starting V2.0 indicator calculation")
    logger.info(f"   Input DataFrame shape: {df.shape}")
    logger.info(f"   Input columns: {list(df.columns)}")
    
    if len(df) == 0:
        logger.error("❌ CRITICAL: Empty DataFrame received!")
        return df
    
    # Check for required OHLCV columns
    required_ohlcv = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_ohlcv = [col for col in required_ohlcv if col not in df.columns]
    if missing_ohlcv:
        logger.error(f"❌ CRITICAL: Missing OHLCV columns: {missing_ohlcv}")
        # Try alternative column names
        alt_mapping = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
        for alt_name, std_name in alt_mapping.items():
            if alt_name in df.columns and std_name not in df.columns:
                df[std_name] = df[alt_name]
                logger.info(f"   Mapped '{alt_name}' to '{std_name}'")
    
    df = df.copy()
    
    # EMA indicators (200, 50, 21)
    df = _compute_emas_v2(df, cfg)
    
    # Z-score momentum (50-period baseline)
    df = _compute_zscore_v2(df, cfg)
    
    # ATR for volatility-based SL/TP
    df = _compute_atr_v2(df, cfg)
    # Session filtering (configurable)
    df = _apply_session_filter(df, cfg)
    # Order blocks detection
    df = _detect_order_blocks(df, cfg)
    
    # Fair value gaps detection
    df = _detect_fair_value_gaps(df, cfg)
    
    logger.info(f"✅ Calculated V2.0 indicators for {len(df)} 1-minute bars")
    return df


def _compute_emas_v2(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Compute EMA 200, 50, and 21 for trend alignment"""
    # EMA periods for V2.0
    ema_periods = [200, 50, 21]
    
    logger.info(f"🔧 Computing EMAs for {len(df)} bars")
    logger.info(f"   Available columns: {list(df.columns)}")
    logger.info(f"   Close column exists: {'Close' in df.columns}")
    
    if 'Close' in df.columns:
        logger.info(f"   Close price range: {df['Close'].min():.2f} - {df['Close'].max():.2f}")
        logger.info(f"   Close data type: {df['Close'].dtype}")
        logger.info(f"   Non-null Close values: {df['Close'].notna().sum()}/{len(df)}")
    else:
        logger.error("❌ CRITICAL: 'Close' column missing from data!")
        return df
    
    for period in ema_periods:
        df[f'ema_{period}'] = df['Close'].ewm(span=period).mean()
        # Also create uppercase version for signal generation
        df[f'EMA_{period}'] = df[f'ema_{period}']
        
        # Debug EMA calculation
        valid_ema_count = df[f'EMA_{period}'].notna().sum()
        if valid_ema_count > 0:
            ema_range = f"{df[f'EMA_{period}'].min():.2f} - {df[f'EMA_{period}'].max():.2f}"
            logger.info(f"   EMA_{period}: {valid_ema_count}/{len(df)} valid values, range: {ema_range}")
        else:
            logger.error(f"❌ EMA_{period}: All values are NaN!")
    
    # Trend alignment conditions
    df['bullish_alignment'] = (
        (df['Close'] > df['ema_200']) & 
        (df['Close'] > df['ema_50']) & 
        (df['Close'] > df['ema_21']) &
        (df['ema_21'] > df['ema_50']) &
        (df['ema_50'] > df['ema_200'])
    )
    
    df['bearish_alignment'] = (
        (df['Close'] < df['ema_200']) & 
        (df['Close'] < df['ema_50']) & 
        (df['Close'] < df['ema_21']) &
        (df['ema_21'] < df['ema_50']) &
        (df['ema_50'] < df['ema_200'])
    )
    
    return df


def _compute_zscore_v2(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Compute Z-score with 50-period baseline for momentum filtering"""
    period = cfg.get('z_period_v2', 50)
    
    logger.info(f"🔧 Computing Z-score with period: {period}")
    logger.info(f"   Input data shape: {df.shape}")
    logger.info(f"   Close price range: {df['Close'].min():.2f} - {df['Close'].max():.2f}")
    
    # Price returns
    df['returns'] = df['Close'].pct_change()
    
    # Rolling statistics for Z-score
    rolling_mean = df['returns'].rolling(window=period).mean()
    rolling_std = df['returns'].rolling(window=period).std()
    
    # Z-score calculation
    df['zscore'] = (df['returns'] - rolling_mean) / rolling_std
    df['zscore'] = df['zscore'].fillna(0)
    
    # Debug Z-score calculation
    valid_zscore_count = (df['zscore'] != 0).sum()
    logger.info(f"   Valid Z-score values: {valid_zscore_count}/{len(df)}")
    if valid_zscore_count > 0:
        logger.info(f"   Z-score range: {df['zscore'].min():.4f} - {df['zscore'].max():.4f}")
    
    # Momentum conditions (not overheated/oversold)
    z_threshold = cfg.get('z_threshold', 1.0)  # Use consistent config key
    df['momentum_ok_long'] = df['zscore'] < z_threshold
    df['momentum_ok_short'] = df['zscore'] > -z_threshold
    # Also create uppercase version for signal generation
    df['Z_Score'] = df['zscore']
    return df


def _compute_atr_v2(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Compute ATR for volatility-based position sizing"""
    period = cfg.get('atr_period_v2', 14)
    
    logger.info(f"🔧 Computing ATR with period: {period}")
    logger.info(f"   Required columns: High, Low, Close")
    logger.info(f"   High exists: {'High' in df.columns}")
    logger.info(f"   Low exists: {'Low' in df.columns}")
    logger.info(f"   Close exists: {'Close' in df.columns}")
    
    if not all(col in df.columns for col in ['High', 'Low', 'Close']):
        logger.error("❌ CRITICAL: Missing OHLC columns for ATR calculation!")
        df['ATR'] = 0.0
        return df
    
    # True Range calculation
    df['prev_close'] = df['Close'].shift(1)
    df['tr1'] = df['High'] - df['Low']
    df['tr2'] = abs(df['High'] - df['prev_close'])
    df['tr3'] = abs(df['Low'] - df['prev_close'])
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # ATR calculation
    df['ATR'] = df['true_range'].rolling(window=period).mean()
    df['ATR'] = df['ATR'].fillna(0)
    
    # Debug ATR calculation
    valid_atr_count = (df['ATR'] > 0).sum()
    if valid_atr_count > 0:
        logger.info(f"   Valid ATR values: {valid_atr_count}/{len(df)}")
        logger.info(f"   ATR range: {df['ATR'].min():.4f} - {df['ATR'].max():.4f}")
    else:
        logger.error("❌ ATR: All values are 0 or NaN!")
    
    # Clean up temporary columns
    df.drop(['prev_close', 'tr1', 'tr2', 'tr3', 'true_range'], axis=1, inplace=True)
    
    return df


def _detect_order_blocks(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Detect order blocks (significant engulfing candles or supply/demand zones).
    Order blocks are areas where institutional orders create significant price reactions.
    """
    # Initialize order block columns
    df['bullish_ob'] = False
    df['bearish_ob'] = False
    df['ob_high'] = np.nan
    df['ob_low'] = np.nan
    
    # Parameters for order block detection
    min_body_size = cfg.get('ob_min_body_pct', 0.002)  # 0.2% minimum body size
    lookback = cfg.get('ob_lookback', 5)  # Bars to look back for validation
    
    for i in range(lookback, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        
        # Calculate candle body sizes
        current_body = abs(current['Close'] - current['Open'])
        prev_body = abs(prev['Close'] - prev['Open'])
        
        # Safety check to prevent division by zero
        if current['Close'] == 0 or pd.isna(current['Close']):
            continue
            
        current_body_pct = current_body / current['Close']
        
        # Bullish Order Block Detection
        # Strong bullish candle that engulfs previous bearish movement
        if (current['Close'] > current['Open'] and  # Bullish candle
            current_body_pct > min_body_size and    # Significant body
            current['Close'] > prev['High'] and     # Breaks previous high
            current['Open'] < prev['Low']):         # Opens below previous low
            
            df.iloc[i, df.columns.get_loc('bullish_ob')] = True
            df.iloc[i, df.columns.get_loc('ob_low')] = current['Open']
            df.iloc[i, df.columns.get_loc('ob_high')] = current['Close']
        
        # Bearish Order Block Detection
        # Strong bearish candle that engulfs previous bullish movement
        elif (current['Close'] < current['Open'] and  # Bearish candle
              current_body_pct > min_body_size and    # Significant body
              current['Close'] < prev['Low'] and      # Breaks previous low
              current['Open'] > prev['High']):        # Opens above previous high
            
            df.iloc[i, df.columns.get_loc('bearish_ob')] = True
            df.iloc[i, df.columns.get_loc('ob_low')] = current['Close']
            df.iloc[i, df.columns.get_loc('ob_high')] = current['Open']
    
    # Forward-fill order block levels for reference
    df['active_bull_ob_low'] = df['ob_low'].where(df['bullish_ob']).ffill()
    df['active_bull_ob_high'] = df['ob_high'].where(df['bullish_ob']).ffill()
    df['active_bear_ob_low'] = df['ob_low'].where(df['bearish_ob']).ffill()
    df['active_bear_ob_high'] = df['ob_high'].where(df['bearish_ob']).ffill()
    
    return df


def _detect_fair_value_gaps(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Detect fair value gaps (imbalance zones where price gaps exist).
    FVGs occur when there's a gap between consecutive candles.
    """
    # Initialize FVG columns
    df['bullish_fvg'] = False
    df['bearish_fvg'] = False
    df['fvg_top'] = np.nan
    df['fvg_bottom'] = np.nan
    
    # Parameters for FVG detection
    min_gap_pct = cfg.get('fvg_min_gap_pct', 0.001)  # 0.1% minimum gap
    
    for i in range(2, len(df)):
        prev2 = df.iloc[i-2]  # Two bars ago
        prev1 = df.iloc[i-1]  # Previous bar
        current = df.iloc[i]  # Current bar
        
        # Bullish FVG: Gap between prev2 high and current low
        # (Previous bar doesn't fill the gap)
        if (prev2['High'] < current['Low'] and
            prev1['High'] < current['Low'] and
            prev1['Low'] > prev2['High']):
            
            gap_size = (current['Low'] - prev2['High']) / current['Close']
            if gap_size > min_gap_pct:
                df.iloc[i, df.columns.get_loc('bullish_fvg')] = True
                df.iloc[i, df.columns.get_loc('fvg_bottom')] = prev2['High']
                df.iloc[i, df.columns.get_loc('fvg_top')] = current['Low']
        
        # Bearish FVG: Gap between prev2 low and current high
        elif (prev2['Low'] > current['High'] and
              prev1['Low'] > current['High'] and
              prev1['High'] < prev2['Low']):
            
            gap_size = (prev2['Low'] - current['High']) / current['Close']
            if gap_size > min_gap_pct:
                df.iloc[i, df.columns.get_loc('bearish_fvg')] = True
                df.iloc[i, df.columns.get_loc('fvg_bottom')] = current['High']
                df.iloc[i, df.columns.get_loc('fvg_top')] = prev2['Low']
    
    # Track active FVG levels
    df['active_bull_fvg_bottom'] = df['fvg_bottom'].where(df['bullish_fvg']).ffill()
    df['active_bull_fvg_top'] = df['fvg_top'].where(df['bullish_fvg']).ffill()
    df['active_bear_fvg_bottom'] = df['fvg_bottom'].where(df['bearish_fvg']).ffill()
    df['active_bear_fvg_top'] = df['fvg_top'].where(df['bearish_fvg']).ffill()
    
    return df


def _apply_session_filter(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Apply session filtering (configurable)"""
    # Convert to EST timezone if needed
    if df['Time'].dt.tz is None:
        df['Time'] = pd.to_datetime(df['Time']).dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
    elif df['Time'].dt.tz != 'US/Eastern':
        df['Time'] = df['Time'].dt.tz_convert('US/Eastern')
    
    # Extract time component
    df['time_only'] = df['Time'].dt.time
    
    # Session window (configurable)
    session_start_hour = cfg.get('session_start_hour', 0)
    session_start_minute = cfg.get('session_start_minute', 0)
    session_end_hour = cfg.get('session_end_hour', 23)
    session_end_minute = cfg.get('session_end_minute', 59)
    
    session_start = time(session_start_hour, session_start_minute)
    session_end = time(session_end_hour, session_end_minute)
    
    df['in_session'] = (
        (df['time_only'] >= session_start) & 
        (df['time_only'] <= session_end)
    )
    
    return df


def generate_signals_v2(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Generate V2.0 trading signals based on indicators.
    
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
    df['signal_type'] = ''
    df['signal_strength'] = 0.0
    
    # V2.0 Parameters
    z_threshold = cfg.get('z_threshold', 1.0)
    
    logger.info(f"🔧 Signal generation debug:")
    logger.info(f"   Z-threshold: ±{z_threshold}")
    logger.info(f"   Config keys available: {list(cfg.keys())}")
    
    # Ensure we have required columns
    required_cols = ['EMA_200', 'EMA_50', 'EMA_21', 'Z_Score', 'ATR', 'in_session']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"⚠️  Missing columns for signal generation: {missing_cols}")
        return df
    
    # Debug: Check session filtering
    session_bars = df['in_session'].sum()
    logger.info(f"🔍 Debug: {session_bars} bars in session out of {len(df)} total bars")
    
    # Debug: Check latest values
    if len(df) > 0:
        latest = df.iloc[-1]
        logger.info(f"🔍 Latest bar debug:")
        logger.info(f"   Time: {latest.get('Time', 'N/A')}")
        logger.info(f"   In Session: {latest.get('in_session', False)}")
        logger.info(f"   EMA 21/50/200: {latest.get('EMA_21', 0):.2f} / {latest.get('EMA_50', 0):.2f} / {latest.get('EMA_200', 0):.2f}")
        logger.info(f"   Z-Score: {latest.get('Z_Score', 0):.2f} (threshold: ±{z_threshold})")
        logger.info(f"   ATR: {latest.get('ATR', 0):.2f}")
        
        # Check if indicators are actually calculated
        if latest.get('EMA_21', 0) == 0 and latest.get('EMA_50', 0) == 0:
            logger.error("❌ PROBLEM: All EMA values are 0 - indicators not calculating properly!")
        if latest.get('Z_Score', 0) == 0:
            logger.warning("❌ PROBLEM: Z-Score is 0 - momentum not calculating!")
        if latest.get('ATR', 0) == 0:
            logger.warning("❌ PROBLEM: ATR is 0 - volatility not calculating!")
    
    # Generate signals for each row
    signals_checked = 0
    for i in range(len(df)):
        row = df.iloc[i]
        
        # Only generate signals during session
        if not row.get('in_session', False):
            continue
            
        signals_checked += 1
        
        # Skip if insufficient data
        if pd.isna(row['EMA_200']) or pd.isna(row['Z_Score']) or pd.isna(row['ATR']):
            continue
        
        # EMA trend alignment
        ema_21 = row['EMA_21']
        ema_50 = row['EMA_50']
        ema_200 = row['EMA_200']
        z_score = row['Z_Score']
        
        # Bullish signal conditions (relaxed EMA requirements)
        bullish_trend = ema_21 > ema_50  # Only require 21 > 50 (removed 200 requirement)
        bullish_momentum = z_score > z_threshold
        bullish_fvg = row.get('fvg_signal', False) and row.get('fvg_direction', '') == 'bullish'
        bullish_ob = row.get('order_block', False) and row.get('ob_direction', '') == 'bullish'
        
        # Bearish signal conditions (relaxed EMA requirements)
        bearish_trend = ema_21 < ema_50  # Only require 21 < 50 (removed 200 requirement)
        bearish_momentum = z_score < -z_threshold
        bearish_fvg = row.get('fvg_signal', False) and row.get('fvg_direction', '') == 'bearish'
        bearish_ob = row.get('order_block', False) and row.get('ob_direction', '') == 'bearish'
        
        # Debug: Log conditions for latest bar
        if i == len(df) - 1:  # Latest bar
            logger.info(f"🔍 Signal conditions for latest bar:")
            logger.info(f"   Bullish trend (21>50): {bullish_trend}")
            logger.info(f"   Bullish momentum (Z>{z_threshold}): {bullish_momentum}")
            logger.info(f"   Bearish trend (21<50): {bearish_trend}")
            logger.info(f"   Bearish momentum (Z<-{z_threshold}): {bearish_momentum}")
        
        # Generate bullish signal (relaxed requirements)
        if bullish_trend and bullish_momentum:
            df.at[i, 'signal_long'] = True
            df.at[i, 'signal_type'] = 'Bullish'
            
            # Calculate signal strength
            strength = 0.0
            if bullish_trend: strength += 0.5
            if bullish_momentum: strength += 0.5
            if bullish_fvg: strength += 0.1  # Bonus if present
            if bullish_ob: strength += 0.1   # Bonus if present
            df.at[i, 'signal_strength'] = strength
        
        # Generate bearish signal (relaxed requirements)
        elif bearish_trend and bearish_momentum:
            df.at[i, 'signal_short'] = True
            df.at[i, 'signal_type'] = 'Bearish'
            
            # Calculate signal strength
            strength = 0.0
            if bearish_trend: strength += 0.5
            if bearish_momentum: strength += 0.5
            if bearish_fvg: strength += 0.1  # Bonus if present
            if bearish_ob: strength += 0.1   # Bonus if present
            df.at[i, 'signal_strength'] = strength
    
    # Signal summary
    total_signals = (df['signal_long'] | df['signal_short']).sum()
    long_signals = df['signal_long'].sum()
    short_signals = df['signal_short'].sum()
    
    logger.info(f"🎯 Generated {total_signals} V2.0 signals ({long_signals} long, {short_signals} short)")
    logger.info(f"🔍 Debug: Checked {signals_checked} bars during session")
    
    return df


def is_in_session_v2(timestamp: datetime, cfg: Dict[str, Any]) -> bool:
    """
    Check if timestamp is within V2.0 trading session.
    
    Args:
        timestamp: Datetime to check
        cfg: Configuration dictionary
        
    Returns:
        True if in session, False otherwise
    """
    import pytz
    
    # Convert to Eastern time if needed
    if timestamp.tzinfo is None:
        # Assume UTC if no timezone
        timestamp = timestamp.replace(tzinfo=pytz.UTC)
    
    # Convert to Eastern time
    eastern = pytz.timezone('US/Eastern')
    et_time = timestamp.astimezone(eastern)
    
    # Session parameters from config (24-hour session: 0:00-23:59)
    session_start_hour = cfg.get('session_start_hour', 0)
    session_start_minute = cfg.get('session_start_minute', 0)
    session_end_hour = cfg.get('session_end_hour', 23)
    session_end_minute = cfg.get('session_end_minute', 59)
    
    # Create session times
    session_start = time(session_start_hour, session_start_minute)
    session_end = time(session_end_hour, session_end_minute)
    
    # Check if current time is in session
    current_time = et_time.time()
    
    # Handle session that crosses midnight (not applicable for 0:00-23:59 but good practice)
    if session_start <= session_end:
        return session_start <= current_time <= session_end
    else:
        return current_time >= session_start or current_time <= session_end


# Alias for backward compatibility
compute_indicators_v2 = calculate_indicators_v2


def _check_fvg_retracement_long(df: pd.DataFrame, i: int) -> bool:
    """Check if price is retracing into a bullish FVG"""
    if i < 1:
        return False
        
    row = df.iloc[i]
    
    # Check if we have an active bullish FVG
    if pd.isna(row['active_bull_fvg_bottom']) or pd.isna(row['active_bull_fvg_top']):
        return False
    
    # Check if current price is within the FVG zone
    fvg_bottom = row['active_bull_fvg_bottom']
    fvg_top = row['active_bull_fvg_top']
    
    return (row['Low'] <= fvg_top and row['High'] >= fvg_bottom)


def _check_fvg_retracement_short(df: pd.DataFrame, i: int) -> bool:
    """Check if price is retracing into a bearish FVG"""
    if i < 1:
        return False
        
    row = df.iloc[i]
    
    # Check if we have an active bearish FVG
    if pd.isna(row['active_bear_fvg_bottom']) or pd.isna(row['active_bear_fvg_top']):
        return False
    
    # Check if current price is within the FVG zone
    fvg_bottom = row['active_bear_fvg_bottom']
    fvg_top = row['active_bear_fvg_top']
    
    return (row['Low'] <= fvg_top and row['High'] >= fvg_bottom)


def _check_order_block_confirmation_long(df: pd.DataFrame, i: int) -> bool:
    """Check if price reacts positively to a bullish order block"""
    if i < 2:
        return False
        
    row = df.iloc[i]
    prev_row = df.iloc[i-1]
    
    # Check if we have an active bullish order block
    if pd.isna(row['active_bull_ob_low']) or pd.isna(row['active_bull_ob_high']):
        return False
    
    ob_low = row['active_bull_ob_low']
    ob_high = row['active_bull_ob_high']
    
    # Check if price touched the OB zone and reacted positively
    touched_ob = (prev_row['Low'] <= ob_high and prev_row['High'] >= ob_low)
    positive_reaction = (row['Close'] > prev_row['Close'])  # Price moving up from OB
    
    return touched_ob and positive_reaction


def _check_order_block_confirmation_short(df: pd.DataFrame, i: int) -> bool:
    """Check if price rejects a bearish order block"""
    if i < 2:
        return False
        
    row = df.iloc[i]
    prev_row = df.iloc[i-1]
    
    # Check if we have an active bearish order block
    if pd.isna(row['active_bear_ob_low']) or pd.isna(row['active_bear_ob_high']):
        return False
    
    ob_low = row['active_bear_ob_low']
    ob_high = row['active_bear_ob_high']
    
    # Check if price touched the OB zone and was rejected
    touched_ob = (prev_row['Low'] <= ob_high and prev_row['High'] >= ob_low)
    rejection = (row['Close'] < prev_row['Close'])  # Price moving down from OB
    
    return touched_ob and rejection


def build_order_v2(row_signal: pd.Series, row_entry: pd.Series, equity: float, 
                   daily_pnl: float, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Construct order details for Ronin V2.0 strategy.
    
    Args:
        row_signal: Signal bar data
        row_entry: Entry bar data (next bar after signal)
        equity: Current account equity
        daily_pnl: Current daily P&L
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
    
    # V2.0 Risk Management Parameters
    risk_per_trade_pct = cfg.get('risk_per_trade_v2', 0.015)  # 1.5%
    atr_multiplier = cfg.get('atr_multiplier_v2', 1.0)        # 1.0 × ATR for SL
    risk_reward_ratio = cfg.get('risk_reward_v2', 1.25)       # 1:1.25 RR
    
    # Calculate stop loss and take profit
    if is_long:
        stop_loss = entry_price - (atr * atr_multiplier)
        take_profit = entry_price + (atr * atr_multiplier * risk_reward_ratio)
        direction = 'LONG'
    else:
        stop_loss = entry_price + (atr * atr_multiplier)
        take_profit = entry_price - (atr * atr_multiplier * risk_reward_ratio)
        direction = 'SHORT'
    
    # Position sizing based on 1.5% risk
    risk_amount = equity * risk_per_trade_pct
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
        'signal_type': row_signal['signal_type'],
        'atr': atr,
        'entry_time': row_entry['Time'],
        'signal_time': row_signal['Time'],
        'strategy_version': 'V2.0'
    }
    
    return order


if __name__ == "__main__":
    logger.info("✅ Ronin Engine V2.0 loaded successfully")
    logger.info("1-minute dual-directional momentum strategy with order blocks and FVGs")
