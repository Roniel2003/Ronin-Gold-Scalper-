"""
Data Loader V2.0 - Load and preprocess 1-minute historical data for Ronin V2.0 Strategy
Includes session filtering, indicator calculation, and signal generation.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import pytz
import warnings
warnings.filterwarnings('ignore')

from ronin_engine_v2 import calculate_indicators_v2, generate_signals_v2, is_in_session_v2


def load_data_v2(symbol: str, start_date: str, end_date: str, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Load and preprocess 1-minute historical data for V2.0 strategy.
    
    Args:
        symbol: Trading symbol (e.g., 'NVDA')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        cfg: Configuration dictionary
        
    Returns:
        DataFrame with OHLCV data, indicators, and signals
    """
    print(f"📥 Loading 1-minute data for {symbol}: {start_date} to {end_date}")
    
    try:
        # Download 1-minute data from Yahoo Finance
        ticker = yf.Ticker(symbol)
        
        # Convert dates to datetime objects
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)  # Include end date
        
        # Download data with 1-minute interval
        df = ticker.history(
            start=start_dt,
            end=end_dt,
            interval='1m',
            auto_adjust=True,
            prepost=True,
            actions=False
        )
        
        if df.empty:
            raise ValueError(f"No data found for {symbol} in the specified date range")
        
        # Reset index to get Time column
        df = df.reset_index()
        df = df.rename(columns={'Datetime': 'Time'})
        
        # Ensure timezone awareness (convert to Eastern time)
        timezone = pytz.timezone(cfg.get('timezone', 'US/Eastern'))
        if df['Time'].dt.tz is None:
            df['Time'] = df['Time'].dt.tz_localize('UTC')
        df['Time'] = df['Time'].dt.tz_convert(timezone)
        
        print(f"✅ Downloaded {len(df):,} 1-minute bars")
        print(f"   Period: {df['Time'].iloc[0]} to {df['Time'].iloc[-1]}")
        print(f"   Price range: ${df['Low'].min():.2f} - ${df['High'].max():.2f}")
        
        # Clean data
        df = clean_data_v2(df)
        
        # Add session information
        df['in_session'] = df['Time'].apply(lambda x: is_in_session_v2(x, cfg))
        
        # Filter to session hours only if configured
        if not cfg.get('skip_sessions', False):
            session_bars = df['in_session'].sum()
            print(f"📊 Session filtering: {session_bars:,} bars in session ({session_bars/len(df):.1%})")
            
            # Keep all data but mark session status
            # This allows for indicator calculation on full dataset
        
        # Calculate V2.0 indicators
        print("🔧 Calculating V2.0 indicators...")
        df = calculate_indicators_v2(df, cfg)
        
        # Generate V2.0 signals
        print("🎯 Generating V2.0 signals...")
        df = generate_signals_v2(df, cfg)
        
        # Summary statistics
        total_signals = (df['signal_long'] | df['signal_short']).sum()
        long_signals = df['signal_long'].sum()
        short_signals = df['signal_short'].sum()
        
        print(f"📈 Signal Summary:")
        print(f"   Total signals: {total_signals}")
        print(f"   Long signals: {long_signals}")
        print(f"   Short signals: {short_signals}")
        print(f"   Signal frequency: {total_signals/len(df)*100:.3f}% of bars")
        
        # Data quality check
        check_data_quality_v2(df, symbol)
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading data for {symbol}: {e}")
        raise


def clean_data_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate 1-minute data.
    
    Args:
        df: Raw OHLCV DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    initial_length = len(df)
    
    # Remove rows with missing OHLCV data
    df = df.dropna(subset=['Open', 'High', 'Low', 'Close', 'Volume'])
    
    # Remove rows with zero or negative prices
    price_mask = (df['Open'] > 0) & (df['High'] > 0) & (df['Low'] > 0) & (df['Close'] > 0)
    df = df[price_mask]
    
    # Remove rows with invalid OHLC relationships
    ohlc_mask = (df['High'] >= df['Open']) & (df['High'] >= df['Close']) & \
                (df['Low'] <= df['Open']) & (df['Low'] <= df['Close']) & \
                (df['High'] >= df['Low'])
    df = df[ohlc_mask]
    
    # Remove extreme outliers (price changes > 50% in 1 minute)
    df['price_change'] = df['Close'].pct_change().abs()
    outlier_mask = df['price_change'] < 0.5  # 50% change threshold
    df = df[outlier_mask]
    df = df.drop('price_change', axis=1)
    
    # Reset index
    df = df.reset_index(drop=True)
    
    cleaned_length = len(df)
    removed_count = initial_length - cleaned_length
    
    if removed_count > 0:
        print(f"🧹 Data cleaning: Removed {removed_count:,} invalid bars ({removed_count/initial_length:.2%})")
    
    return df


def check_data_quality_v2(df: pd.DataFrame, symbol: str) -> None:
    """
    Check data quality and print diagnostics.
    
    Args:
        df: Processed DataFrame
        symbol: Trading symbol
    """
    print(f"\n📊 Data Quality Report for {symbol}:")
    print(f"   Total bars: {len(df):,}")
    print(f"   Date range: {df['Time'].iloc[0].date()} to {df['Time'].iloc[-1].date()}")
    
    # Check for missing data
    missing_data = df.isnull().sum()
    if missing_data.any():
        print(f"   Missing data: {missing_data[missing_data > 0].to_dict()}")
    else:
        print(f"   Missing data: None")
    
    # Check indicator coverage
    required_indicators = ['EMA_200', 'EMA_50', 'EMA_21', 'Z_Score', 'ATR']
    for indicator in required_indicators:
        if indicator in df.columns:
            valid_count = df[indicator].notna().sum()
            coverage = valid_count / len(df)
            print(f"   {indicator} coverage: {coverage:.1%} ({valid_count:,} bars)")
    
    # Check session data
    if 'in_session' in df.columns:
        session_bars = df['in_session'].sum()
        session_pct = session_bars / len(df)
        print(f"   Session bars: {session_bars:,} ({session_pct:.1%})")
    
    # Check signal distribution
    if 'signal_long' in df.columns and 'signal_short' in df.columns:
        long_signals = df['signal_long'].sum()
        short_signals = df['signal_short'].sum()
        total_signals = long_signals + short_signals
        signal_rate = total_signals / len(df)
        print(f"   Signal rate: {signal_rate:.3%} ({total_signals:,} total)")
        print(f"   Long/Short ratio: {long_signals}:{short_signals}")


def load_multiple_symbols_v2(symbols: list, start_date: str, end_date: str, 
                            cfg: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Load data for multiple symbols.
    
    Args:
        symbols: List of trading symbols
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        cfg: Configuration dictionary
        
    Returns:
        Dictionary mapping symbols to DataFrames
    """
    print(f"📥 Loading data for {len(symbols)} symbols: {symbols}")
    
    data = {}
    failed_symbols = []
    
    for symbol in symbols:
        try:
            df = load_data_v2(symbol, start_date, end_date, cfg)
            data[symbol] = df
            print(f"✅ {symbol}: {len(df):,} bars loaded")
        except Exception as e:
            print(f"❌ {symbol}: Failed to load - {e}")
            failed_symbols.append(symbol)
    
    print(f"\n📊 Loading Summary:")
    print(f"   Successful: {len(data)} symbols")
    print(f"   Failed: {len(failed_symbols)} symbols")
    
    if failed_symbols:
        print(f"   Failed symbols: {failed_symbols}")
    
    return data


def save_data_v2(df: pd.DataFrame, symbol: str, output_path: Optional[str] = None) -> str:
    """
    Save processed data to CSV file.
    
    Args:
        df: Processed DataFrame
        symbol: Trading symbol
        output_path: Optional output file path
        
    Returns:
        Path to saved file
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{symbol}_1min_v2_{timestamp}.csv"
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"💾 Data saved to {output_path}")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")
    print(f"   File size: {pd.io.common.get_filepath_or_buffer(output_path)[0]}")
    
    return output_path


def preview_data_v2(df: pd.DataFrame, symbol: str, n_rows: int = 10) -> None:
    """
    Preview loaded data.
    
    Args:
        df: DataFrame to preview
        symbol: Trading symbol
        n_rows: Number of rows to display
    """
    print(f"\n📋 Data Preview for {symbol} (First {n_rows} rows):")
    print("=" * 100)
    
    # Select key columns for preview
    preview_cols = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    # Add indicator columns if they exist
    indicator_cols = ['EMA_200', 'EMA_50', 'EMA_21', 'Z_Score', 'ATR']
    for col in indicator_cols:
        if col in df.columns:
            preview_cols.append(col)
    
    # Add signal columns if they exist
    signal_cols = ['in_session', 'signal_long', 'signal_short', 'signal_type']
    for col in signal_cols:
        if col in df.columns:
            preview_cols.append(col)
    
    # Display preview
    preview_df = df[preview_cols].head(n_rows)
    
    # Format numeric columns
    for col in preview_df.columns:
        if col != 'Time' and preview_df[col].dtype in ['float64', 'int64']:
            if col in ['Open', 'High', 'Low', 'Close']:
                preview_df[col] = preview_df[col].round(2)
            elif col in ['EMA_200', 'EMA_50', 'EMA_21']:
                preview_df[col] = preview_df[col].round(2)
            elif col in ['Z_Score']:
                preview_df[col] = preview_df[col].round(3)
            elif col in ['ATR']:
                preview_df[col] = preview_df[col].round(4)
    
    print(preview_df.to_string(index=False))
    print("=" * 100)


# Example usage and testing
def test_data_loader_v2():
    """Test the V2.0 data loader."""
    from config import get_config
    
    cfg = get_config()
    
    # Test single symbol
    symbol = 'NVDA'
    start_date = '2025-08-01'
    end_date = '2025-08-31'  # One month of data
    
    print("🔄 Testing V2.0 Data Loader...")
    
    try:
        # Load data
        df = load_data_v2(symbol, start_date, end_date, cfg)
        
        # Preview data
        preview_data_v2(df, symbol)
        
        # Test saving
        output_file = save_data_v2(df, symbol)
        
        print(f"✅ V2.0 Data Loader test completed successfully")
        print(f"   Output file: {output_file}")
        
        return df
        
    except Exception as e:
        print(f"❌ V2.0 Data Loader test failed: {e}")
        return None


if __name__ == "__main__":
    test_data_loader_v2()
