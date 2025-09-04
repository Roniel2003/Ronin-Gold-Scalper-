# Ronin Gold V3.0 - XAUUSD Scalping Trading System

**Advanced 1-minute scalping strategy for XAUUSD (Gold) with MT5 integration**

## Overview

Ronin Gold V3.0 is a sophisticated algorithmic trading system designed for high-frequency scalping of XAUUSD (Gold) on 1-minute timeframes. The system combines momentum, mean reversion, and trend confirmation indicators to generate precise entry signals during optimal market conditions.

## Key Features

- **Real-time Signal Generation**: Live 1-minute bar processing with V3.0 scalping formula
- **MT5 Integration**: Direct broker connectivity with proper position sizing
- **24/5 Session Management**: Automated gold market hours detection
- **Risk Management**: Built-in stop loss, take profit, and position sizing
- **Live Data Feeds**: Real-time tick data processing and bar formation
- **Comprehensive Logging**: Detailed signal and trade execution logs

## Trading Strategy

### Ronin Gold V3.0 Scalping Formula

**LONG Signal:**
```
RSI(14) > 52 AND Z-Score(21) > 0.5 AND EMA_Diff >= 0.1 AND In_Session = True
```

**SHORT Signal:**
```
RSI(14) < 48 AND Z-Score(21) < -0.5 AND EMA_Diff <= -0.1 AND In_Session = True
```

### Indicators

- **RSI(14)**: Momentum filter with scalping-optimized thresholds
- **Z-Score(21)**: Mean reversion signal for price extremes
- **EMA Difference**: EMA(21) - EMA(50) for trend confirmation
- **ATR(14)**: Volatility-based risk management

### Risk Management

- **Stop Loss**: Entry ± (1.5 × ATR)
- **Take Profit**: Entry ± (3.0 × ATR)
- **Risk Per Trade**: 0.1% of account balance
- **Position Size**: 0.01 lots minimum (MT5 compliant)

## Installation & Setup

### Prerequisites

```bash
pip install pandas numpy yfinance MetaTrader5 pytz requests asyncio
```

### MT5 Configuration

1. Install MetaTrader 5 terminal
2. Create demo account or use live account
3. Enable algorithmic trading in MT5 settings
4. Update `config/mt5_trading.json` with your credentials:

```json
{
  "broker": {
    "type": "mt5",
    "server": "YOUR-BROKER-SERVER",
    "login": YOUR_LOGIN_ID,
    "password": "YOUR_PASSWORD"
  }
}
```

## Usage

### Start Live Trading

```bash
python cli.py live --strategy v3 --symbols XAUUSD --broker mt5 --data-provider mt5 --config config/mt5_trading.json
```

### Command Line Options

- `--strategy v3`: Use Ronin Gold V3.0 scalping strategy
- `--symbols XAUUSD`: Trade gold (required for V3.0)
- `--broker mt5`: Use MetaTrader 5 broker
- `--data-provider mt5`: Real-time MT5 data feed
- `--config`: Path to configuration file

## Project Structure

```
Ronin/
├── cli.py                      # Command line interface
├── ronin_engine_v3.py          # V3.0 indicator calculations and signals
├── live_signal_processor_v3.py # Real-time signal processing
├── live_trader.py              # Trading engine and order management
├── mt5_broker.py               # MT5 broker API integration
├── data_feeds.py               # Real-time data feed management
├── config/
│   ├── mt5_trading.json        # MT5 trading configuration
│   └── live_trading.json       # General trading settings
├── tests/                      # Unit tests
└── README.md                   # This file
```

## Configuration

### Trading Parameters (`config/mt5_trading.json`)

```json
{
  "trading": {
    "risk_per_trade": 0.001,      # 0.1% risk per trade
    "max_positions": 3,           # Maximum concurrent positions
    "symbols": ["XAUUSD"]         # Trading symbols
  },
  "signals": {
    "z_score_threshold": 0.5,     # Z-Score entry threshold
    "rsi_long_threshold": 52,     # RSI long entry threshold
    "rsi_short_threshold": 48,    # RSI short entry threshold
    "ema_crossover_threshold": 0.1 # EMA difference threshold
  },
  "sessions": {
    "timezone": "US/Eastern",     # Session timezone
    "session_start_hour": 17,     # Sunday market open
    "session_end_hour": 17        # Friday market close
  }
}
```

## Session Management

**Gold Market Hours (24/5):**
- **Open**: Sunday 17:00 EST
- **Active**: Monday 00:00 - Friday 17:00 EST
- **Closed**: Friday 17:00 - Sunday 17:00 EST
- **High Liquidity**: 08:00 - 11:00 EST (optimal trading window)

## Risk Management

### FTMO Compliance

- Maximum daily loss: $500
- Maximum total loss: $1,000
- Position sizing based on account balance
- Automatic risk calculations

### Position Sizing

```python
# For XAUUSD with $200,000 account
risk_amount = account_balance * 0.001  # $200
position_size = 0.01 lots  # Minimum MT5 lot size
```

## Performance Monitoring

### Log Files

- `mt5_trading.log`: Trading execution logs
- Console output: Real-time signal generation
- MT5 terminal: Position and order status

### Key Metrics

- Signal frequency: ~2-5 signals per hour (high volatility periods)
- Win rate: Target 60%+ with 3:1 risk-reward
- Average trade duration: 5-30 minutes (scalping)

## Troubleshooting

### Common Issues

**"No money" Error:**
- Check MT5 account balance
- Verify position sizing (should be 0.01 lots minimum)
- Ensure sufficient margin for XAUUSD

**No Signals Generated:**
- Verify session detection (should show "IN SESSION")
- Check indicator calculations (RSI, Z-Score, EMA)
- Confirm XAUUSD data feed is active

**Connection Issues:**
- Verify MT5 credentials in config file
- Check MT5 terminal is running and logged in
- Ensure algorithmic trading is enabled

## Important Notes

- **Demo Trading Recommended**: Test thoroughly before live trading
- **Market Hours**: System only trades during gold market hours
- **Risk Warning**: Scalping involves high frequency trading with inherent risks
- **Broker Requirements**: Ensure your broker supports automated trading

## Support

For issues or questions:
1. Check log files for error messages
2. Verify configuration settings
3. Test with demo account first
4. Monitor MT5 terminal for trade execution

## License

This project is for educational and research purposes. Use at your own risk.

---

**Ronin Gold V3.0 - Precision Scalping for the Modern Trader**
