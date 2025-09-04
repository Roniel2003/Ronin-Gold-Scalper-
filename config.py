"""
Ronin Bot Configuration
Single source of truth for all parameters and settings.
No secrets in code - all tunables externalized here.
"""

import os
import json
from typing import Dict, Any

def get_config(config_path: str = None) -> Dict[str, Any]:
    """
    Get configuration dictionary with all parameters.
    
    Args:
        config_path: Optional path to custom config file
        
    Returns:
        Configuration dictionary
    """
    # Base configuration for Ronin V2.0
    config = {
        # Strategy Version
        "strategy_version": "V2.0",
        "timeframe": "1min",
        
        # V2.0 EMA Parameters
        "ema_periods": [200, 50, 21],
        
        # V2.0 Z-Score Parameters
        "z_period_v2": 50,           # 50-period baseline for Z-score
        "z_threshold_v2": 1.0,       # ±1.0 threshold for momentum filtering (reduced from 1.75)
        
        # V2.0 ATR Parameters
        "atr_period_v2": 14,         # ATR(14) for volatility
        "atr_multiplier_v2": 1.0,    # 1.0 × ATR for stop loss
        
        # V2.0 Risk Management
        "risk_per_trade_v2": 0.015,  # 1.5% risk per trade
        "risk_reward_v2": 1.25,      # 1:1.25 risk-reward ratio
        
        # Order Block Detection
        "ob_min_body_pct": 0.002,    # 0.2% minimum body size for OB
        "ob_lookback": 5,            # Bars to look back for OB validation
        
        # Fair Value Gap Detection
        "fvg_min_gap_pct": 0.001,    # 0.1% minimum gap size for FVG
        
        # Session Controls - Asian Session + London/NY Overlap for XAUUSD
        # Asian Session: 19:00-01:00 EST (Tokyo open to close)
        # London/NY Overlap: 08:00-12:00 EST (highest volatility)
        "session_start_hour": 19,        # 7:00 PM EST (Asian session start)
        "session_start_minute": 0,
        "session_end_hour": 1,           # 1:00 AM EST (Asian session end)
        "session_end_minute": 0,
        "session_start_hour_2": 8,       # 8:00 AM EST (London/NY overlap start)
        "session_start_minute_2": 0,
        "session_end_hour_2": 12,        # 12:00 PM EST (London/NY overlap end)
        "session_end_minute_2": 0,
        "timezone": "US/Eastern",
        
        # V2.0 Risk Controls
        "max_daily_loss": 500.0,     # $500 max daily loss
        "max_total_loss": 1000.0,    # $1,000 max account drawdown
        "daily_profit_cap": 500.0,   # $500 daily profit cap
        
        # Position Management
        "one_position_at_a_time": True,
        "confirm_next_bar": True,
        
        # FTMO Compliance (legacy - kept for compatibility)
        "profit_target": 10000.0,
        
        # Legacy parameters (kept for backward compatibility)
        "z_period": 20,
        "z_thresh": 2.0,
        "atr_period": 14,
        "atr_mult": 2.0,
        "atr_min": 0.1,
        "ema_fast": 12,
        "ema_slow": 26,
        "vol_period": 20,
        "rr": 2.0,
        "risk_base_pct": 0.01,
        "risk_red_pct": 0.005,
        "risk_up_pct": 0.015,
        
        # Data Processing
        "skip_sessions": False,
        
        # Live Trading
        "symbols": ["NVDA", "TSLA", "AAPL", "MSFT", "GOOGL"]
    }
    
    # Load custom config if provided
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
            config.update(custom_config)
            print(f"✅ Loaded custom config from {config_path}")
        except Exception as e:
            print(f"⚠️  Failed to load custom config: {e}")
    
    return config

def validate_config(cfg: Dict[str, Any] = None) -> bool:
    """
    Validate configuration parameters
    
    Args:
        cfg: Configuration dictionary (optional)
    
    Returns:
        True if configuration is valid, False otherwise
    """
    if cfg is None:
        cfg = get_config()
    
    # Basic validation
    assert cfg["z_period_v2"] > 0, "z_period_v2 must be positive"
    assert cfg["z_threshold_v2"] > 0, "z_threshold_v2 must be positive"
    assert cfg["atr_period_v2"] > 0, "atr_period_v2 must be positive"
    assert cfg["atr_multiplier_v2"] > 0, "atr_multiplier_v2 must be positive"
    assert 0 < cfg["risk_per_trade_v2"] < 1, "risk_per_trade_v2 must be between 0 and 1"
    assert cfg["risk_reward_v2"] > 0, "risk_reward_v2 must be positive"
    assert cfg["max_daily_loss"] > 0, "max_daily_loss must be positive"
    assert cfg["max_total_loss"] > 0, "max_total_loss must be positive"
    assert cfg["daily_profit_cap"] > 0, "daily_profit_cap must be positive"
    assert cfg["session_start_hour"] >= 0 and cfg["session_start_hour"] < 24, "session_start_hour must be between 0 and 23"
    assert cfg["session_start_minute"] >= 0 and cfg["session_start_minute"] < 60, "session_start_minute must be between 0 and 59"
    assert cfg["session_end_hour"] >= 0 and cfg["session_end_hour"] < 24, "session_end_hour must be between 0 and 23"
    assert cfg["session_end_minute"] >= 0 and cfg["session_end_minute"] < 60, "session_end_minute must be between 0 and 59"
    assert cfg["session_start_hour_2"] >= 0 and cfg["session_start_hour_2"] < 24, "session_start_hour_2 must be between 0 and 23"
    assert cfg["session_start_minute_2"] >= 0 and cfg["session_start_minute_2"] < 60, "session_start_minute_2 must be between 0 and 59"
    assert cfg["session_end_hour_2"] >= 0 and cfg["session_end_hour_2"] < 24, "session_end_hour_2 must be between 0 and 23"
    assert cfg["session_end_minute_2"] >= 0 and cfg["session_end_minute_2"] < 60, "session_end_minute_2 must be between 0 and 59"
    
    return True

if __name__ == "__main__":
    # Test config loading
    config = get_config()
    validate_config(config)
    print("✅ Configuration loaded and validated successfully")
    print(f"Timezone: {config['timezone']}")
    print(f"Session Start: {config['session_start_hour']}:{config['session_start_minute']}")
    print(f"Session End: {config['session_end_hour']}:{config['session_end_minute']}")
    print(f"Session Start 2: {config['session_start_hour_2']}:{config['session_start_minute_2']}")
    print(f"Session End 2: {config['session_end_hour_2']}:{config['session_end_minute_2']}")
    print(f"Risk per trade: {config['risk_per_trade_v2']:.1%}")
