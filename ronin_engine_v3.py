"""
Ronin Gold V3.0 - XAUUSD 1-Minute Scalping Strategy
Advanced trend-following + statistical arbitrage + bell curve theory
Optimized for real-time signal generation with minimal lag
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from datetime import datetime, time
import logging
import json
import pytz

logger = logging.getLogger(__name__)

def calculate_indicators_v3(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Calculate all V3.0 indicators: EMA 21/50, RSI(14), Z-Score, ATR(14).
    
    Args:
        df: DataFrame with OHLCV data
        cfg: Configuration dictionary
        
    Returns:
        DataFrame with V3.0 indicators added
    """
    logger.info(f"[START] Starting Ronin Gold V3.0 indicator calculation")
    logger.info(f"   Input DataFrame shape: {df.shape}")
    logger.info(f"   Asset: XAUUSD, Timeframe: 1-minute")
    
    # Validate required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"[ERROR] CRITICAL: Missing required columns: {missing_cols}")
        return df
    
    logger.info(f"   Price range: {df['Close'].min():.2f} - {df['Close'].max():.2f}")
    
    # Calculate each V3.0 indicator
    df = _compute_emas_v3(df, cfg)
    df = _compute_rsi_v3(df, cfg)
    df = _compute_zscore_v3(df, cfg)
    df = _compute_atr_v3(df, cfg)
    df = _add_previous_candle_levels(df)
    df = _apply_session_filter_v3(df, cfg)
    
    logger.info(f"[SUCCESS] Ronin Gold V3.0 indicators calculated successfully")
    return df


def _compute_emas_v3(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Compute EMA 21 and 50 for trend filtering"""
    logger.info(f"[EMA] Computing EMAs for trend filter (21, 50)")
    
    # EMA periods for V3.0 trend filter
    ema_periods = [21, 50]
    
    for period in ema_periods:
        col_name = f'EMA_{period}'
        df[col_name] = df['Close'].ewm(span=period).mean()
        df[col_name] = df[col_name].fillna(0)
        
        # Debug EMA calculation
        valid_ema_count = (df[col_name] > 0).sum()
        if valid_ema_count > 0:
            logger.info(f"   EMA-{period}: {valid_ema_count}/{len(df)} valid values")
        else:
            logger.warning(f"[WARNING] EMA-{period}: All values are 0!")
    
    # Trend direction
    df['trend_bullish'] = df['EMA_21'] > df['EMA_50']
    df['trend_bearish'] = df['EMA_21'] < df['EMA_50']
    
    return df


def _compute_rsi_v3(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Compute RSI(14) for momentum filtering"""
    period = cfg.get('rsi_period', 14)
    logger.info(f"[RSI] Computing RSI({period}) for momentum filter")
    
    # Calculate price changes
    delta = df['Close'].diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gains and losses
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Calculate RSI
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50)  # Neutral RSI for initial values
    
      # RSI momentum conditions for V3.0 - Scalping optimized thresholds
    df['rsi_bullish'] = df['RSI'] > 52  # Scalping long bias
    df['rsi_bearish'] = df['RSI'] < 48  # Scalping short bias
    
    # Debug RSI calculation
    valid_rsi_count = df['RSI'].notna().sum()
    if valid_rsi_count > 0:
        logger.info(f"   RSI: {valid_rsi_count}/{len(df)} valid values, range: {df['RSI'].min():.1f} - {df['RSI'].max():.1f}")
    else:
        logger.warning("[WARNING] RSI: All values are NaN!")
    
    return df


def _compute_zscore_v3(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Compute Z-Score for bell curve distribution filtering"""
    period = cfg.get('zscore_period', 21)
    logger.info(f"[ZSCORE] Computing Z-Score({period}) for bell curve filter")
    
    # Rolling mean and standard deviation
    rolling_mean = df['Close'].rolling(window=period).mean()
    rolling_std = df['Close'].rolling(window=period).std()
    
    # Z-Score calculation
    df['Z_Score'] = (df['Close'] - rolling_mean) / rolling_std
    df['Z_Score'] = df['Z_Score'].fillna(0)
    
    # Scalping Z-Score conditions (no bell curve restriction)
    df['z_long_signal'] = df['Z_Score'] > 0.5   # Long bias for scalping
    df['z_short_signal'] = df['Z_Score'] < -0.5  # Short bias for scalping
    
    # Debug Z-Score calculation
    valid_z_count = (df['Z_Score'] != 0).sum()
    valid_filter_count = df['z_long_signal'].sum() + df['z_short_signal'].sum()
    
    if valid_z_count > 0:
        logger.info(f"   Z-Score: {valid_z_count}/{len(df)} valid values")
        logger.info(f"   Scalping Z-Score filter: {valid_filter_count}/{len(df)} bars with long/short bias")
    else:
        logger.warning("[WARNING] Z-Score: All values are 0!")

    
    return df


def _compute_atr_v3(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Compute ATR(14) for risk management"""
    period = cfg.get('atr_period', 14)
    logger.info(f"[ATR] Computing ATR({period}) for risk management")
    
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
        logger.info(f"   ATR: {valid_atr_count}/{len(df)} valid values, avg: {df['ATR'].mean():.2f}")
    else:
        logger.warning("[WARNING] ATR: All values are 0!")
    
    # Clean up temporary columns
    df.drop(['prev_close', 'tr1', 'tr2', 'tr3', 'true_range'], axis=1, inplace=True)
    
    return df


def _add_previous_candle_levels(df: pd.DataFrame) -> pd.DataFrame:
    """Add previous candle high/low for breakout detection"""
    logger.info(f"[LEVELS] Adding previous candle levels for breakout detection")
    
    df['prev_high'] = df['High'].shift(1)
    df['prev_low'] = df['Low'].shift(1)
    
    return df


def _apply_session_filter_v3(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Apply session filtering for high-liquidity periods"""
    # Convert to EST timezone if needed
    if df['Time'].dt.tz is None:
        df['Time'] = pd.to_datetime(df['Time']).dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
    elif df['Time'].dt.tz != 'US/Eastern':
        df['Time'] = df['Time'].dt.tz_convert('US/Eastern')
    
    # Extract time component
    df['time_only'] = df['Time'].dt.time
    
    # High-liquidity sessions (Asian session and London/NY overlap)
    df['in_session'] = df['Time'].apply(lambda x: is_in_session_v3(x, cfg))
    
    session_bars = df['in_session'].sum()
    logger.info(f"[SESSION] Session filter: {session_bars}/{len(df)} bars in high-liquidity session")
    
    return df


def is_in_session_v3(timestamp: pd.Timestamp, cfg: dict) -> bool:
    """
    Check if timestamp is within V3.0 trading sessions.
    Supports dual session windows for 24/5 gold market
    
    Args:
        timestamp: Timestamp to check
        cfg: Configuration dictionary
        
    Returns:
        True if in session, False otherwise
    """
    # Convert to Eastern Time
    eastern = pytz.timezone('US/Eastern')
    
    # Handle timezone conversion properly
    if timestamp.tzinfo is None:
        # Assume UTC if no timezone
        timestamp = timestamp.replace(tzinfo=pytz.UTC)
    
    et_time = timestamp.astimezone(eastern)
    
    # Get session configuration from sessions block
    sessions_config = cfg.get('sessions', {})
    
    # Session 1: Cross-week session (Sunday 17:00 to Friday 17:00)
    session_start_hour = sessions_config.get('session_start_hour', 17)
    session_start_minute = sessions_config.get('session_start_minute', 0)
    session_end_hour = sessions_config.get('session_end_hour', 17)
    session_end_minute = sessions_config.get('session_end_minute', 0)
    
    # Session 2: Daily coverage (00:00-23:59)
    session_start_hour_2 = sessions_config.get('session_start_hour_2', 0)
    session_start_minute_2 = sessions_config.get('session_start_minute_2', 0)
    session_end_hour_2 = sessions_config.get('session_end_hour_2', 23)
    session_end_minute_2 = sessions_config.get('session_end_minute_2', 59)
    
    current_time = et_time.time()
    weekday = et_time.weekday()  # 0=Monday, 6=Sunday
    
    # Gold market is closed on Saturday and Sunday before 17:00 EST
    if weekday == 5:  # Saturday
        in_session_1 = False
    elif weekday == 6:  # Sunday
        # Market opens at 17:00 EST on Sunday
        sunday_open = time(session_start_hour, session_start_minute)
        in_session_1 = current_time >= sunday_open
    else:  # Monday-Friday
        # Market is open unless it's Friday after 17:00 EST
        if weekday == 4:  # Friday
            friday_close = time(session_end_hour, session_end_minute)
            in_session_1 = current_time < friday_close
        else:  # Monday-Thursday
            in_session_1 = True
    
    # Session 2: Daily 24-hour coverage during weekdays
    session_start_2 = time(session_start_hour_2, session_start_minute_2)
    session_end_2 = time(session_end_hour_2, session_end_minute_2)
    in_session_2 = session_start_2 <= current_time <= session_end_2
    
    # Only log session detection for recent timestamps (avoid spam from historical data)
    now = pd.Timestamp.now(tz=eastern)
    time_diff = abs((et_time - now).total_seconds())
    
    # Only log if timestamp is within last 5 minutes (300 seconds)
    if time_diff <= 300:
        logger.info(f"[SESSION] Time check: {et_time} (local: {current_time}, weekday: {weekday})")
        logger.info(f"[SESSION] Gold market session: {in_session_1}")
        logger.info(f"[SESSION] Daily coverage (00:00-23:59): {in_session_2}")
        logger.info(f"[SESSION] Final result - In any session: {in_session_1 or in_session_2}")
    
    return in_session_1 or in_session_2


def trend_filter(row: pd.Series) -> Tuple[bool, bool]:
    """
    V3.0 Trend Filter: EMA 21 vs EMA 50
    
    Returns:
        (bullish_trend, bearish_trend)
    """
    bullish = row.get('trend_bullish', False)
    bearish = row.get('trend_bearish', False)
    
    return bullish, bearish


def momentum_filter(row: pd.Series) -> Tuple[bool, bool]:
    """
    V3.0 Momentum Filter: RSI(14) thresholds
    
    Returns:
        (bullish_momentum, bearish_momentum)
    """
    rsi = row.get('RSI', 50)
    
    # V3.0 momentum conditions (scalping optimized)
    bullish = rsi > 52  # Scalping long bias
    bearish = rsi < 48  # Scalping short bias
    
    return bullish, bearish


def entry_signal(row: pd.Series, prev_row: pd.Series) -> Tuple[bool, bool, str]:
    """
    V3.0 Scalping Entry Signal: Optimized for 1-minute gold scalping
    
    Scalping Formula:
    Long: RSI > 52 AND Z-Score > 0.5 AND EMA-21 > EMA-50 by >= 0.1
    Short: RSI < 48 AND Z-Score < -0.5 AND EMA-21 < EMA-50 by >= 0.1
    
    Returns:
        (long_signal, short_signal, reason)
    """
    current_price = row['Close']
    rsi = row.get('RSI', 50)
    z_score = row.get('Z_Score', 0)
    ema_21 = row.get('EMA_21', 0)
    ema_50 = row.get('EMA_50', 0)
    in_session = row.get('in_session', False)
    
    # Calculate EMA difference for micro-crossover detection
    ema_diff = ema_21 - ema_50
    
    long_signal = False
    short_signal = False
    reason = ""
    
    # Scalping Long Signal: RSI > 52 AND Z-Score > 0.5 AND EMA-21 > EMA-50 by >= 0.1
    if (rsi > 52 and z_score > 0.5 and ema_diff >= 0.1 and in_session):
        long_signal = True
        reason = f"SCALP LONG: RSI={rsi:.1f}>52, Z={z_score:.2f}>0.5, EMA diff={ema_diff:.2f}>=0.1"
    
    # Scalping Short Signal: RSI < 48 AND Z-Score < -0.5 AND EMA-21 < EMA-50 by >= 0.1
    elif (rsi < 48 and z_score < -0.5 and ema_diff <= -0.1 and in_session):
        short_signal = True
        reason = f"SCALP SHORT: RSI={rsi:.1f}<48, Z={z_score:.2f}<-0.5, EMA diff={ema_diff:.2f}<=-0.1"
    
    return long_signal, short_signal, reason


def risk_manager(entry_price: float, atr: float, direction: str) -> Tuple[float, float]:
    """
    V3.0 Risk Manager: ATR-based SL and TP
    
    Args:
        entry_price: Entry price
        atr: Current ATR value
        direction: 'LONG' or 'SHORT'
        
    Returns:
        (stop_loss, take_profit)
    """
    sl_multiplier = 1.5  # 1.5x ATR for stop loss
    tp_multiplier = 3.0  # 3.0x ATR for take profit
    
    if direction == 'LONG':
        stop_loss = entry_price - (atr * sl_multiplier)
        take_profit = entry_price + (atr * tp_multiplier)
    else:  # SHORT
        stop_loss = entry_price + (atr * sl_multiplier)
        take_profit = entry_price - (atr * tp_multiplier)
    
    return stop_loss, take_profit


def generate_signals_v3(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Generate Ronin Gold V3.0 trading signals with complete quant formula.
    
    Args:
        df: DataFrame with V3.0 indicators
        cfg: Configuration dictionary
        
    Returns:
        DataFrame with V3.0 signals added
    """
    df = df.copy()
    
    # Initialize signal columns
    df['signal_long'] = False
    df['signal_short'] = False
    df['signal_type'] = ''
    df['signal_reason'] = ''
    df['entry_price'] = 0.0
    df['stop_loss'] = 0.0
    df['take_profit'] = 0.0
    df['signal_json'] = ''
    
    logger.info(f"[SIGNALS] Generating Ronin Gold V3.0 signals...")
    
    # Ensure we have required columns
    required_cols = ['EMA_21', 'EMA_50', 'RSI', 'Z_Score', 'ATR', 'in_session', 'prev_high', 'prev_low']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"[WARNING] Missing columns for V3.0 signal generation: {missing_cols}")
        return df
    
    signals_generated = 0
    
    # Generate signals for each row (skip first row due to prev_candle dependency)
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        
        # Generate entry signal
        long_signal, short_signal, reason = entry_signal(row, prev_row)
        
        if long_signal or short_signal:
            signals_generated += 1
            entry_price = row['Close']
            atr = row['ATR']
            
            if long_signal:
                direction = 'LONG'
                df.at[i, 'signal_long'] = True
                df.at[i, 'signal_type'] = 'BUY'
            else:
                direction = 'SHORT'
                df.at[i, 'signal_short'] = True
                df.at[i, 'signal_type'] = 'SELL'
            
            # Calculate risk management levels
            stop_loss, take_profit = risk_manager(entry_price, atr, direction)
            
            # Store signal data
            df.at[i, 'signal_reason'] = reason
            df.at[i, 'entry_price'] = entry_price
            df.at[i, 'stop_loss'] = stop_loss
            df.at[i, 'take_profit'] = take_profit
            
            # Create JSON signal output
            signal_json = {
                "timestamp": row['Time'].isoformat() if pd.notna(row['Time']) else datetime.now().isoformat(),
                "signal": df.at[i, 'signal_type'],
                "entry": round(entry_price, 2),
                "stop_loss": round(stop_loss, 2),
                "take_profit": round(take_profit, 2),
                "reason": reason,
                "rsi": round(row['RSI'], 1),
                "z_score": round(row['Z_Score'], 2),
                "atr": round(atr, 2)
            }
            
            df.at[i, 'signal_json'] = json.dumps(signal_json)
            
            # Log the signal
            logger.info(f"[SIGNAL] RONIN GOLD V3.0 SIGNAL: {df.at[i, 'signal_type']} XAUUSD @ {entry_price:.2f}")
            logger.info(f"   SL: {stop_loss:.2f} | TP: {take_profit:.2f} | Reason: {reason}")
            logger.info(f"   JSON: {json.dumps(signal_json)}")
    
    logger.info(f"[SIGNALS] Generated {signals_generated} Ronin Gold V3.0 signals")
    
    return df


if __name__ == "__main__":
    logger.info("Ronin Gold V3.0 Engine loaded successfully")
    logger.info("XAUUSD 1-minute scalping strategy with trend-following + stat-arb + bell curve theory")
