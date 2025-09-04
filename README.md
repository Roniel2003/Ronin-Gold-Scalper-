# Ronin Bot - 15-Minute Trend-Momentum Trading System

**Version:** 2.2 (Staged Development)  
**Stage:** 4 - Monte Carlo & Optimization 

## Overview

Ronin is a standalone, FTMO-aware trend-momentum trading bot designed for 15-minute intraday trading. Built with a staged development approach, it combines Z-score momentum detection, ATR volatility filtering, and EMA trend analysis to generate high-probability trading signals.

**Current Status:** Stage 4 Complete - Monte Carlo and optimization features implemented

## Features

### Stage 1: Data Loading & Session Filtering
- **CSV Data Loading** with timezone-aware parsing
- **Session Filtering** (America/New_York → UTC conversion)
- **15-Minute Resampling** with proper OHLC aggregation
- **Data Validation** and error handling
- **CLI Integration** with preview and output options

### Stage 2: Indicators & Signals
- **Z-Score Momentum Indicator** - Detects price momentum anomalies
- **ATR Volatility Filter** - Ensures sufficient market volatility
- **EMA Trend Filters** - Fast/slow EMA crossover for trend direction
- **Volume Confirmation** - Above-average volume requirement
- **Signal Generation Logic** - Multi-condition signal validation
- **Order Construction** - Entry, stop-loss, take-profit calculation
- **Risk Management** - Dynamic position sizing based on P&L

### Stage 3: Backtesting & Risk Management
- **Event-driven backtesting engine** - Realistic trade execution simulation
- **FTMO compliance checks** - Daily/total loss limits and active day tracking
- **Performance metrics calculation** - Win rate, profit factor, expectancy, Sharpe ratio
- **Equity curve generation** - Drawdown analysis and peak-to-trough tracking
- **Trade logging and reporting** - Detailed trade logs and summary statistics

### Stage 4: Monte Carlo & Optimization
- **Bootstrap Monte Carlo simulation** - Preserves trade correlation
- **Robustness testing** - Risk of ruin, VaR, and expected shortfall metrics
- **Parameter optimization** - Grid search with multiple objective functions
- **Walk-forward analysis** - Out-of-sample parameter validation
- **Sensitivity analysis** - Around optimal parameters
- **Stability ratio assessment** - Strategy robustness evaluation

## Installation

```bash
git clone <repository-url>
cd Ronin
pip install -r requirements.txt
```

## Configuration

All parameters are centralized in `config.py`:

```python
# Indicator Parameters
"z_period": 20,        # Z-score lookback period
"z_thresh": 2.0,       # Z-score signal threshold
"atr_period": 14,      # ATR calculation period
"atr_mult": 2.0,       # ATR multiplier for stops
"atr_min": 0.1,        # Minimum ATR % for volatility filter
"ema_fast": 12,        # Fast EMA period
"ema_slow": 26,        # Slow EMA period
"vol_period": 20,      # Volume average period
"rr": 2.0,            # Risk-reward ratio

# Risk Management
"risk_base_pct": 0.01,    # 1% base risk per trade
"risk_red_pct": 0.005,    # 0.5% risk when losing
"risk_up_pct": 0.015,     # 1.5% risk when winning
```

## Usage

### Data Loading & Processing
```bash
# Load and preview CSV data
python cli.py load --csv data.csv --preview

# Load with session filtering
python cli.py load --csv data.csv --out processed_data.csv

# Skip session filtering
python cli.py load --csv data.csv --no-sessions --preview
```

### Indicators Computation
```bash
# Compute technical indicators
python cli.py indicators --csv data.csv --preview

# Save indicators to file
python cli.py indicators --csv data.csv --out indicators_data.csv
```

### Signal Generation
```bash
# Generate trading signals
python cli.py signals --csv data.csv --preview

# Save signals to file
python cli.py signals --csv data.csv --out signals_data.csv
```

### Backtesting
```bash
# Run comprehensive backtest
python cli.py backtest --csv data.csv --out results/

# Custom starting equity
python cli.py backtest --csv data.csv --equity 25000 --out results/

# Skip session filtering
python cli.py backtest --csv data.csv --no-sessions --out results/
```

### Monte Carlo Analysis
```bash
# Run Monte Carlo robustness analysis
python cli.py monte-carlo --csv data.csv --out mc_results/

# Custom simulation count
python cli.py monte-carlo --csv data.csv --simulations 5000 --out mc_results/
```

### Parameter Optimization
```bash
# Run parameter optimization
python cli.py optimize --csv data.csv --out opt_results/

# Optimize for specific metric
python cli.py optimize --csv data.csv --metric profit_factor --out opt_results/

# Custom parameter ranges (JSON format)
python cli.py optimize --csv data.csv --param-ranges '{"zscore_threshold": [1.5, 2.0, 2.5]}' --out opt_results/
```

### Walk-Forward Analysis
```bash
# Run walk-forward analysis
python cli.py walk-forward --csv data.csv --out wf_results/

# Custom window and step sizes
python cli.py walk-forward --csv data.csv --window 3 --step 1 --out wf_results/
```

## Signal Logic

Ronin generates signals based on multiple confluent conditions:

### Long Signal Conditions
1. **Z-score > threshold** (strong positive momentum)
2. **EMA Fast > EMA Slow** (uptrend confirmation)
3. **Price > EMA Fast** (price above trend)
4. **ATR % > minimum** (sufficient volatility)
5. **Volume > average** (volume confirmation)

### Short Signal Conditions
1. **Z-score < -threshold** (strong negative momentum)
2. **EMA Fast < EMA Slow** (downtrend confirmation)
3. **Price < EMA Fast** (price below trend)
4. **ATR % > minimum** (sufficient volatility)
5. **Volume > average** (volume confirmation)

## Order Construction

For each signal, Ronin constructs complete order details:

- **Entry Price**: Open of next bar after signal
- **Stop Loss**: Entry ± (ATR × multiplier)
- **Take Profit**: Entry ± (ATR × multiplier × risk-reward ratio)
- **Position Size**: Risk amount ÷ stop distance
- **Risk Amount**: Dynamic based on equity and current P&L

## File Structure

```
Ronin/
├── config.py              # Configuration parameters
├── data_loader.py          # CSV loading and session filtering
├── ronin_engine.py         # Indicators and signal generation
├── risk_manager.py         # FTMO risk management
├── backtester.py          # Backtesting engine
├── logger.py              # Trade and equity logging
├── cli.py                 # Command-line interface
├── requirements.txt       # Python dependencies
├── tests/
│   ├── test_data_loader.py    # Data loader tests
│   └── test_ronin_engine.py   # Indicators & signals tests
└── README.md
```

## Development Stages

### Stage 0: Repository Setup
- [x] Project structure and configuration
- [x] CLI framework and help system
- [x] Placeholder modules with interfaces

### Stage 1: Data Loader + Session Gate
- [x] CSV loading with timezone handling
- [x] Session filtering (NY → UTC conversion)
- [x] 15-minute resampling
- [x] Data validation and unit tests

### Stage 2: Indicators & Signals
- [x] Z-score momentum indicator
- [x] ATR volatility filter
- [x] EMA trend filters
- [x] Signal generation logic
- [x] Order construction
- [x] Unit tests and CLI integration

### Stage 3: Backtesting & Risk Management
- [x] Event-driven backtesting engine
- [x] FTMO compliance checks
- [x] Performance metrics calculation
- [x] Equity curve generation
- [x] Trade logging and reporting

### Stage 4: Monte Carlo & Optimization
- [x] Bootstrap Monte Carlo simulation
- [x] Robustness testing
- [x] Parameter optimization
- [x] Walk-forward analysis
- [x] Sensitivity analysis
- [x] Stability ratio assessment

### Stage 5: Live Trading Integration
- [ ] Real-time data feeds and broker API integration
- [ ] Order management system with execution tracking
- [ ] Live risk monitoring and position management
- [ ] Performance tracking and alerts
- [ ] Production deployment and monitoring

## Testing

### Unit Tests
```bash
# Test data loading functionality
python tests/test_data_loader.py

# Test indicators and signals
python tests/test_ronin_engine.py

# Run all tests
python -m unittest discover tests/
```

### Integration Testing
```bash
# Test full pipeline with sample data
python cli.py selftest

# Test with real data
python cli.py signals --csv your_data.csv --preview
```

## Example Output

### Indicators Summary
```
 Indicators Summary:
   Total bars: 1,440
   Z-score range: -3.45 to 4.12
   ATR average: 0.0234
   Trend up %: 67.3%
```

### Signals Summary
```
 Signals Summary:
   Total bars: 1,440
   Long signals: 23
   Short signals: 18
   Signal frequency: 2.85%
   Average signal strength: 2.34
```

## Contributing

1. Follow the staged development approach
2. Maintain comprehensive unit tests
3. Update documentation for new features
4. Ensure FTMO compliance in risk management

## License

[Your License Here]

---

**Stage 4 Complete** - Ready for Stage 5 Live Trading Integration Development
