"""
Data Loader - CSV Loading and Session Filtering
Handles data loading, timezone conversion, and session filtering.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import pytz
from datetime import datetime, time
import warnings
import os


def load_csv(path: str, tz: str) -> pd.DataFrame:
    """
    Load CSV data and ensure proper timezone-aware timestamps.
    
    Args:
        path: Path to CSV file
        tz: Timezone string (e.g., 'America/New_York')
        
    Returns:
        DataFrame with columns: Time (tz-aware), Open, High, Low, Close, Volume
    """
    try:
        # Check if file exists
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found: {path}")
        
        print(f"📁 Loading CSV: {path}")
        
        # Load CSV with flexible parsing
        df = pd.read_csv(path)
        
        print(f"📊 Raw data shape: {df.shape}")
        print(f"📋 Columns found: {list(df.columns)}")
        
        # Map column names to standard format
        column_mapping = {}
        
        # Find time column
        time_cols = [col for col in df.columns if col.lower() in ['time', 'timestamp', 'datetime', 'date']]
        if time_cols:
            column_mapping[time_cols[0]] = 'Time'
        else:
            raise ValueError("No time column found. Expected 'time', 'timestamp', 'datetime', or 'date'")
        
        # Find OHLC columns
        ohlc_mapping = {
            'open': ['open', 'Open', 'OPEN', 'o'],
            'high': ['high', 'High', 'HIGH', 'h'], 
            'low': ['low', 'Low', 'LOW', 'l'],
            'close': ['close', 'Close', 'CLOSE', 'c']
        }
        
        for target_col, possible_names in ohlc_mapping.items():
            found_col = None
            for col in df.columns:
                if col in possible_names:
                    found_col = col
                    break
            
            if found_col:
                column_mapping[found_col] = target_col.capitalize()
            else:
                raise ValueError(f"No {target_col} column found. Expected one of: {possible_names}")
        
        # Find volume column (optional)
        volume_cols = [col for col in df.columns if col.lower() in ['volume', 'vol', 'v']]
        if volume_cols:
            column_mapping[volume_cols[0]] = 'Volume'
        else:
            print("⚠️  No volume column found, using default volume of 1000")
        
        # Rename columns
        df = df.rename(columns=column_mapping)
        
        # Select only the columns we need
        required_cols = ['Time', 'Open', 'High', 'Low', 'Close']
        available_cols = [col for col in required_cols if col in df.columns]
        
        if 'Volume' in df.columns:
            available_cols.append('Volume')
        
        df = df[available_cols].copy()
        
        # Add volume if missing
        if 'Volume' not in df.columns:
            df['Volume'] = 1000
        
        print(f"📈 Mapped columns: {list(df.columns)}")
        
        # Parse timestamps
        print("🕒 Parsing timestamps...")
        
        # Handle different timestamp formats
        if df['Time'].dtype == 'object':
            # Try parsing as ISO format first (like your NVIDIA data)
            try:
                df['Time'] = pd.to_datetime(df['Time'])
            except:
                # Fallback to general parsing
                df['Time'] = pd.to_datetime(df['Time'], infer_datetime_format=True)
        else:
            df['Time'] = pd.to_datetime(df['Time'])
        
        # Convert to target timezone if not already timezone-aware
        target_tz = pytz.timezone(tz)
        
        if df['Time'].dt.tz is None:
            # Assume data is in target timezone if no timezone info
            df['Time'] = df['Time'].dt.tz_localize(target_tz)
            print(f"🌍 Localized timestamps to {tz}")
        else:
            # Convert to target timezone
            df['Time'] = df['Time'].dt.tz_convert(target_tz)
            print(f"🌍 Converted timestamps to {tz}")
        
        # Validate OHLC data
        print("✅ Validating OHLC data...")
        
        # Check for NaN values
        nan_count = df[['Open', 'High', 'Low', 'Close']].isna().sum().sum()
        if nan_count > 0:
            print(f"⚠️  Found {nan_count} NaN values, dropping rows...")
            df = df.dropna(subset=['Open', 'High', 'Low', 'Close'])
        
        # Validate OHLC relationships
        invalid_ohlc = (
            (df['High'] < df['Open']) | 
            (df['High'] < df['Close']) | 
            (df['Low'] > df['Open']) | 
            (df['Low'] > df['Close']) |
            (df['High'] < df['Low'])
        )
        
        invalid_count = invalid_ohlc.sum()
        if invalid_count > 0:
            print(f"⚠️  Found {invalid_count} invalid OHLC relationships, fixing...")
            # Fix invalid relationships
            df.loc[invalid_ohlc, 'High'] = df.loc[invalid_ohlc, ['Open', 'Close']].max(axis=1)
            df.loc[invalid_ohlc, 'Low'] = df.loc[invalid_ohlc, ['Open', 'Close']].min(axis=1)
        
        # Remove duplicates
        initial_len = len(df)
        df = df.drop_duplicates(subset=['Time']).sort_values('Time').reset_index(drop=True)
        
        if len(df) < initial_len:
            print(f"🔄 Removed {initial_len - len(df)} duplicate timestamps")
        
        # Final validation
        if len(df) == 0:
            raise ValueError("No valid data remaining after processing")
        
        print(f"✅ Successfully loaded {len(df)} rows")
        print(f"📅 Date range: {df['Time'].min()} to {df['Time'].max()}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        raise


def filter_sessions(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Filter DataFrame to only include bars within defined trading sessions.
    
    Args:
        df: DataFrame with timezone-aware Time column
        cfg: Configuration dictionary with sessions
        
    Returns:
        Filtered DataFrame containing only session bars
    """
    if df.empty:
        return df
    
    # Get session configuration
    sessions = cfg.get('sessions', [])
    source_tz = cfg.get('timezone', 'America/New_York')
    
    if not sessions:
        print("No sessions defined, returning all data")
        return df
    
    # Convert to UTC for filtering (authoritative checks)
    df_utc = df.copy()
    df_utc['Time'] = df_utc['Time'].dt.tz_convert('UTC')
    
    # Create session mask
    session_mask = pd.Series(False, index=df_utc.index)
    
    # Process each trading day
    source_timezone = pytz.timezone(source_tz)
    utc_timezone = pytz.UTC
    
    for date in df_utc['Time'].dt.date.unique():
        day_mask = df_utc['Time'].dt.date == date
        day_bars = df_utc[day_mask].copy()
        
        if day_bars.empty:
            continue
        
        # Convert session times to UTC for this specific date
        for session_start, session_end in sessions:
            # Parse session times
            start_time = datetime.strptime(session_start, "%H:%M").time()
            end_time = datetime.strptime(session_end, "%H:%M").time()
            
            # Create datetime objects for this date in source timezone
            start_dt = source_timezone.localize(datetime.combine(date, start_time))
            end_dt = source_timezone.localize(datetime.combine(date, end_time))
            
            # Convert to UTC
            start_utc = start_dt.astimezone(utc_timezone)
            end_utc = end_dt.astimezone(utc_timezone)
            
            # Filter bars within this session (using bar close time)
            session_bars = (day_bars['Time'] >= start_utc) & (day_bars['Time'] <= end_utc)
            session_mask[day_mask] |= session_bars
    
    # Apply session filter
    filtered_df = df[session_mask].reset_index(drop=True)
    
    print(f"✅ Session filtering: {len(df):,} → {len(filtered_df):,} bars")
    print(f"   Sessions: {sessions} ({source_tz})")
    print(f"   Filtered out: {len(df) - len(filtered_df):,} bars ({(len(df) - len(filtered_df))/len(df)*100:.1f}%)")
    
    return filtered_df


def resample_to_15m(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample data to 15-minute bars if not already in that timeframe.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame resampled to 15-minute bars
    """
    if df.empty:
        return df
    
    # Check current timeframe
    time_diffs = df['Time'].diff().dropna()
    most_common_interval = time_diffs.mode().iloc[0] if not time_diffs.empty else pd.Timedelta(minutes=15)
    
    print(f"Detected timeframe: {most_common_interval}")
    
    # If already 15-minute bars, return as-is
    if most_common_interval == pd.Timedelta(minutes=15):
        print("✅ Data already in 15-minute timeframe")
        return df
    
    print(f"Resampling from {most_common_interval} to 15-minute bars...")
    
    # Set Time as index for resampling
    df_resampled = df.set_index('Time')
    
    # Define resampling rules
    agg_rules = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    
    # Resample to 15-minute bars
    df_15m = df_resampled.resample('15T').agg(agg_rules)
    
    # Remove bars with NaN values (gaps in data)
    df_15m = df_15m.dropna()
    
    # Reset index to get Time column back
    df_15m = df_15m.reset_index()
    
    print(f"✅ Resampling: {len(df):,} → {len(df_15m):,} bars")
    
    return df_15m


def validate_data_format(df: pd.DataFrame) -> bool:
    """
    Validate that DataFrame has required columns and proper format.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        True if valid, raises exception if invalid
    """
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    required_columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Check for timezone-aware timestamps
    if not hasattr(df['Time'].dtype, 'tz') or df['Time'].dtype.tz is None:
        raise ValueError("Time column must be timezone-aware")
    
    # Check for numeric columns
    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column {col} must be numeric")
    
    # Check for negative prices
    price_cols = ['Open', 'High', 'Low', 'Close']
    for col in price_cols:
        if (df[col] <= 0).any():
            raise ValueError(f"Column {col} contains non-positive values")
    
    # Check for negative volume
    if (df['Volume'] < 0).any():
        raise ValueError("Volume column contains negative values")
    
    print("✅ Data format validation passed")
    return True


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get summary statistics for the loaded data.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        Dictionary with summary statistics
    """
    if df.empty:
        return {}
    
    summary = {
        'total_bars': len(df),
        'date_range': {
            'start': df['Time'].min(),
            'end': df['Time'].max(),
            'days': (df['Time'].max() - df['Time'].min()).days
        },
        'price_range': {
            'min': df['Low'].min(),
            'max': df['High'].max(),
            'latest': df['Close'].iloc[-1]
        },
        'volume': {
            'total': df['Volume'].sum(),
            'avg_daily': df['Volume'].mean(),
            'max_daily': df['Volume'].max()
        },
        'timezone': str(df['Time'].dt.tz),
        'timeframe': df['Time'].diff().mode().iloc[0] if len(df) > 1 else None
    }
    
    return summary


if __name__ == "__main__":
    from config import get_config
    
    print("✅ Data Loader module loaded successfully")
    print("Testing data loading functionality...")
    
    # Test configuration
    cfg = get_config()
    print(f"Sessions: {cfg['sessions']}")
    print(f"Timezone: {cfg['timezone']}")
    
    # Test timezone handling
    ny_tz = pytz.timezone('America/New_York')
    utc_tz = pytz.UTC
    
    test_time = ny_tz.localize(datetime(2024, 1, 15, 10, 30))  # 10:30 AM ET
    test_time_utc = test_time.astimezone(utc_tz)
    
    print(f"Test time conversion:")
    print(f"  NY: {test_time}")
    print(f"  UTC: {test_time_utc}")
    
    print("\n🎯 Ready for CSV loading and session filtering")
