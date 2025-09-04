"""
Ronin Gold V3.0 Live Signal Processor - Real-time XAUUSD 1-minute scalping
Processes live tick data and generates V3.0 signals with JSON output and webhook support
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timezone, timedelta
import pytz
from collections import deque
import logging
import json
import yfinance as yf
import requests

from ronin_engine_v3 import (
    calculate_indicators_v3, 
    generate_signals_v3,
    is_in_session_v3
)


class RealTimeSignalProcessorV3:
    """
    Real-time signal processor for Ronin Gold V3.0 strategy.
    Converts tick data to 1-minute bars and generates V3.0 XAUUSD signals.
    """
    
    def __init__(self, symbol: str, cfg: Dict[str, Any], 
                 signal_callback: Optional[Callable] = None,
                 webhook_url: Optional[str] = None):
        """
        Initialize the V3.0 signal processor.
        
        Args:
            symbol: Trading symbol (should be 'XAUUSD')
            cfg: Configuration dictionary with V3.0 parameters
            signal_callback: Callback function for generated signals
            webhook_url: Optional webhook URL for signal notifications
        """
        self.symbol = symbol
        self.cfg = cfg
        self.signal_callback = signal_callback
        self.webhook_url = webhook_url
        
        # V3.0 Parameters - optimized for XAUUSD 1-minute scalping
        self.timeframe = '1min'
        self.max_periods = 100  # Sufficient for RSI(14), EMA(50), Z-Score(21), ATR(14)
        
        # Data storage (1-minute bars)
        self.bars = deque(maxlen=self.max_periods)
        self.current_bar = None
        self.last_bar_time = None
        
        # Timezone handling (EST for session filtering)
        self.timezone = pytz.timezone('US/Eastern')
        
        # State tracking
        self.is_running = False
        self.last_signal_time = None
        self.signals_today = 0
        
        # Setup logging
        self.logger = logging.getLogger(f"RoninGoldV3_{symbol}")
        
        self.logger.info(f"Ronin Gold V3.0 Signal Processor initialized for {symbol}")
        self.logger.info(f"   Timeframe: {self.timeframe}")
        self.logger.info(f"   Max periods: {self.max_periods}")
        self.logger.info(f"   High-liquidity session: 8:00-11:00 AM EST")
        self.logger.info(f"   Webhook: {'Enabled' if webhook_url else 'Disabled'}")
        
        # Invalid price tracking
        self.invalid_price_count = 0
        self.last_invalid_warning = 0
    
    async def start(self):
        """Start the V3.0 signal processor."""
        self.is_running = True
        self.logger.info(f"Ronin Gold V3.0 processor started for {self.symbol}")
        
        # Load historical data to bootstrap indicators
        await self._load_historical_data()
    
    async def stop(self):
        """Stop the V3.0 signal processor."""
        self.is_running = False
        self.logger.info(f"Ronin Gold V3.0 processor stopped for {self.symbol}")
        self.logger.info(f"   Signals generated today: {self.signals_today}")
    
    async def _load_historical_data(self):
        """Load historical XAUUSD data to bootstrap indicator calculations."""
        try:
            self.logger.info(f"Loading historical XAUUSD data for V3.0 indicators...")
            
            # Get sufficient historical data for all indicators
            min_required = self.max_periods
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)  # 7 days should be enough for 1-minute data
            
            # Use Gold futures (GC=F) as proxy for XAUUSD
            yahoo_symbol = 'GC=F'
            
            self.logger.info(f"   Fetching {min_required} bars from Yahoo Finance: {yahoo_symbol}")
            
            # Download historical data
            ticker = yf.Ticker(yahoo_symbol)
            hist_data = ticker.history(
                start=start_date,
                end=end_date,
                interval='1m',
                auto_adjust=True,
                prepost=True
            )
            
            if hist_data.empty:
                self.logger.warning(f"No historical data available for {yahoo_symbol}")
                return
            
            # Convert to our bar format
            historical_bars = []
            for timestamp, row in hist_data.iterrows():
                # Ensure timezone awareness
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=pytz.UTC)
                
                # Convert to Eastern time
                et_timestamp = timestamp.astimezone(self.timezone)
                
                bar = {
                    'Time': et_timestamp,
                    'Open': float(row['Open']),
                    'High': float(row['High']),
                    'Low': float(row['Low']),
                    'Close': float(row['Close']),
                    'Volume': int(row['Volume']) if not pd.isna(row['Volume']) else 0
                }
                historical_bars.append(bar)
            
            # Add historical bars to our deque
            for bar in historical_bars[-min_required:]:
                self.bars.append(bar)
            
            self.logger.info(f"Loaded {len(self.bars)} historical bars for V3.0 indicators")
            
            # Process indicators on historical data
            if len(self.bars) >= 50:  # Minimum for meaningful indicators
                await self._process_historical_indicators()
            
        except Exception as e:
            self.logger.error(f"Error loading historical data: {e}")
            self.logger.info("   Continuing with live data only")
    
    async def _process_historical_indicators(self):
        """Process V3.0 indicators on historical data to establish baseline."""
        try:
            self.logger.info(f"Processing V3.0 indicators on {len(self.bars)} historical bars...")
            
            # Convert to DataFrame
            df = pd.DataFrame(list(self.bars))
            
            # Calculate V3.0 indicators
            df = calculate_indicators_v3(df, self.cfg)
            
            # Log the latest calculated values
            if len(df) > 0:
                latest = df.iloc[-1]
                self.logger.info(f"[V3] V3.0 Historical indicators established:")
                self.logger.info(f"   EMA 21/50: {latest.get('EMA_21', 0):.2f} / {latest.get('EMA_50', 0):.2f}")
                self.logger.info(f"   RSI: {latest.get('RSI', 0):.1f}")
                self.logger.info(f"   Z-Score: {latest.get('Z_Score', 0):.2f}")
                self.logger.info(f"   ATR: {latest.get('ATR', 0):.2f}")
            
        except Exception as e:
            self.logger.error(f"Error processing historical indicators: {e}")
    
    async def process_tick(self, tick_data: Dict[str, Any]):
        """
        Process incoming tick data and update 1-minute bars.
        
        Args:
            tick_data: Dictionary with tick information
                      {'symbol': str, 'price': float, 'volume': int, 'timestamp': datetime}
        """
        if not self.is_running:
            return
        
        try:
            timestamp = tick_data['timestamp']
            price = float(tick_data['price'])
            volume = int(tick_data.get('volume', 0))
            
            # Validate price data - filter out zero/invalid prices
            if price <= 0 or price > 10000:  # XAUUSD should be between 0-10000
                self.invalid_price_count += 1
                # Only log every 1000th invalid price to reduce spam during market closure
                if self.invalid_price_count % 1000 == 0:
                    self.logger.warning(f"Filtered {self.invalid_price_count} invalid prices - likely market closed (latest: {price:.2f} at {timestamp})")
                return
            
            self.logger.debug(f"Processing XAUUSD tick: {price:.2f} at {timestamp}")
            
            # Convert to 1-minute bar timestamp
            bar_time = self._get_bar_time(timestamp)
            
            # Create or update current bar
            if self.current_bar is None or bar_time != self.current_bar['Time']:
                # Finalize previous bar if exists
                if self.current_bar is not None:
                    self.logger.debug(f"Finalizing V3.0 bar for {self.current_bar['Time']}")
                    await self._finalize_bar()
                
                # Start new bar
                self.current_bar = {
                    'Time': bar_time,
                    'Open': price,
                    'High': price,
                    'Low': price,
                    'Close': price,
                    'Volume': volume
                }
                self.logger.debug(f"Started new V3.0 bar for {bar_time}")
            else:
                # Update current bar with valid price
                if price > 0:  # Additional safety check
                    self.current_bar['High'] = max(self.current_bar['High'], price)
                    self.current_bar['Low'] = min(self.current_bar['Low'], price)
                    self.current_bar['Close'] = price
                    self.current_bar['Volume'] += volume
        
        except Exception as e:
            self.logger.error(f"Error processing tick: {e}")
    
    def _get_bar_time(self, timestamp: datetime) -> datetime:
        """Convert timestamp to 1-minute bar time."""
        # Ensure timezone awareness
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        
        # Convert to Eastern time for session logic
        et_time = timestamp.astimezone(self.timezone)
        
        # Truncate to minute boundary
        bar_time = et_time.replace(second=0, microsecond=0)
        
        return bar_time
    
    async def _finalize_bar(self):
        """Finalize the current bar and process for V3.0 signals."""
        if self.current_bar is None:
            return
        
        # Add bar to history
        self.bars.append(self.current_bar.copy())
        self.last_bar_time = self.current_bar['Time']
        
        # Debug: Log bar accumulation
        self.logger.info(f"V3.0 Bar finalized: {self.current_bar['Time']} | Bars: {len(self.bars)}/{self.max_periods}")
        self.logger.info(f"   OHLC: {self.current_bar['Open']:.2f}/{self.current_bar['High']:.2f}/{self.current_bar['Low']:.2f}/{self.current_bar['Close']:.2f}")
        
        # Check if we have enough data for V3.0 indicators
        min_required = 50  # Minimum for RSI(14), EMA(50), etc.
        if len(self.bars) < min_required:
            self.logger.warning(f"Insufficient data: {len(self.bars)} bars, need {min_required} for V3.0 indicators")
            return
        
        self.logger.info(f"Sufficient data for V3.0 processing: {len(self.bars)} bars")
        
        # Convert to DataFrame for processing
        df = pd.DataFrame(list(self.bars))
        
        # Calculate V3.0 indicators first
        self.logger.info(f"[V3] Starting Ronin Gold V3.0 indicator calculations...")
        df = calculate_indicators_v3(df, self.cfg)
        
        # Add session information (only for latest bar to avoid performance issues)
        latest_time = df['Time'].iloc[-1]
        latest_in_session = is_in_session_v3(latest_time, self.cfg)
        df['in_session'] = False  # Initialize all as False
        df.loc[df.index[-1], 'in_session'] = latest_in_session  # Set only latest bar
        
        session_status = "IN SESSION" if latest_in_session else "OUT OF SESSION"
        self.logger.info(f"[V3] Current session status: {session_status} at {latest_time}")
        
        # Debug: Check latest indicators
        latest = df.iloc[-1]
        self.logger.info(f"[V3] Latest V3.0 indicators:")
        self.logger.info(f"   EMA 21/50: {latest.get('EMA_21', 0):.2f} / {latest.get('EMA_50', 0):.2f}")
        self.logger.info(f"   RSI: {latest.get('RSI', 0):.1f}")
        self.logger.info(f"   Z-Score: {latest.get('Z_Score', 0):.2f}")
        self.logger.info(f"   ATR: {latest.get('ATR', 0):.2f}")
        
        # Generate V3.0 signals
        self.logger.info(f"[V3] Starting Ronin Gold V3.0 signal generation...")
        df = generate_signals_v3(df, self.cfg)
        
        # Check for new signals on the latest bar
        latest_bar = df.iloc[-1]
        
        # Debug: Log signal conditions
        self.logger.info(f"[V3] Signal check: {latest_bar.get('signal_type', '') if latest_bar.get('signal_type', '') != 'HOLD' else 'No signal'}")
        self.logger.info(f"   In session: {latest_bar.get('in_session', False)}")
        self.logger.info(f"   Signal long: {latest_bar.get('signal_long', False)}")
        self.logger.info(f"   Signal short: {latest_bar.get('signal_short', False)}")
        
        if latest_bar.get('signal_long', False) or latest_bar.get('signal_short', False):
            self.logger.info(f"[SIGNAL] RONIN GOLD V3.0 SIGNAL GENERATED!")
            await self._handle_signal(latest_bar)
        else:
            self.logger.debug(f"No V3.0 signals generated for current bar")
    
    async def _handle_signal(self, signal_bar: pd.Series):
        """
        Handle a new Ronin Gold V3.0 signal.
        
        Args:
            signal_bar: Pandas Series with V3.0 signal data
        """
        try:
            signal_time = signal_bar['Time']
            
            # Prevent duplicate signals
            if self.last_signal_time == signal_time:
                return
            
            self.last_signal_time = signal_time
            self.signals_today += 1
            
            # Extract signal data
            signal_type = signal_bar.get('signal_type', '')
            signal_reason = signal_bar.get('signal_reason', '')
            entry_price = signal_bar.get('entry_price', 0)
            stop_loss = signal_bar.get('stop_loss', 0)
            take_profit = signal_bar.get('take_profit', 0)
            signal_json_str = signal_bar.get('signal_json', '{}')
            
            # Parse JSON signal
            try:
                signal_json = json.loads(signal_json_str)
            except:
                signal_json = {}
            
            # Create comprehensive signal dictionary
            signal = {
                'symbol': self.symbol,
                'timestamp': signal_time.isoformat() if pd.notna(signal_time) else datetime.now().isoformat(),
                'signal_type': signal_type,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'reason': signal_reason,
                'rsi': signal_bar.get('RSI', 0),
                'z_score': signal_bar.get('Z_Score', 0),
                'atr': signal_bar.get('ATR', 0),
                'ema_21': signal_bar.get('EMA_21', 0),
                'ema_50': signal_bar.get('EMA_50', 0),
                'strategy_version': 'V3.0',
                'signal_number': self.signals_today
            }
            
            # Log the signal
            self.logger.info(f"[SIGNAL] RONIN GOLD V3.0 SIGNAL #{self.signals_today}: {signal_type} {self.symbol}")
            self.logger.info(f"   Entry: {entry_price:.2f} | SL: {stop_loss:.2f} | TP: {take_profit:.2f}")
            self.logger.info(f"   Reason: {signal_reason}")
            self.logger.info(f"   RSI: {signal_bar.get('RSI', 0):.1f} | Z-Score: {signal_bar.get('Z_Score', 0):.2f}")
            
            # Send to webhook if configured
            if self.webhook_url:
                await self._send_webhook(signal)
            
            # Send signal to callback
            if self.signal_callback:
                await self.signal_callback(signal)
        
        except Exception as e:
            self.logger.error(f"Error handling V3.0 signal: {e}")
    
    async def _send_webhook(self, signal: Dict[str, Any]):
        """Send signal to webhook (e.g., Telegram bot)."""
        try:
            # Format message for Telegram
            message = f"""
[V3] **RONIN GOLD V3.0 SIGNAL**

[V3] **{signal['signal_type']} {signal['symbol']}**
[V3] Entry: {signal['entry_price']:.2f}
[V3] Stop Loss: {signal['stop_loss']:.2f}
[V3] Take Profit: {signal['take_profit']:.2f}

[V3] **Indicators:**
[V3] • RSI: {signal['rsi']:.1f}
[V3] • Z-Score: {signal['z_score']:.2f}
[V3] • ATR: {signal['atr']:.2f}

[V3] **Reason:** {signal['reason']}
[V3] **Time:** {signal['timestamp']}
[V3] **Signal #:** {signal['signal_number']}
            """.strip()
            
            payload = {
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            # Send webhook request
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            
            if response.status_code == 200:
                self.logger.info(f"Webhook sent successfully")
            else:
                self.logger.warning(f"Webhook failed: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error sending webhook: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current V3.0 processor status."""
        return {
            'symbol': self.symbol,
            'strategy_version': 'V3.0',
            'is_running': self.is_running,
            'bars_count': len(self.bars),
            'last_bar_time': self.last_bar_time,
            'last_signal_time': self.last_signal_time,
            'signals_today': self.signals_today,
            'current_bar': self.current_bar,
            'timeframe': self.timeframe,
            'webhook_enabled': bool(self.webhook_url)
        }


class LiveTradingIntegratorV3:
    """
    Integrates Ronin Gold V3.0 signal processor with live trading engine.
    Optimized for XAUUSD 1-minute scalping.
    """
    
    def __init__(self, broker, data_feed, trading_engine, cfg: Dict[str, Any]):
        """Initialize the V3.0 live trading integrator."""
        self.broker = broker
        self.data_feed = data_feed
        self.trading_engine = trading_engine
        self.cfg = cfg
        
        # Signal processors for each symbol (should be XAUUSD for V3.0)
        self.signal_processors = {}
        
        # Setup logging
        self.logger = logging.getLogger("LiveTradingIntegratorV3")
        
        self.logger.info("Ronin Gold V3.0 Live Trading Integrator initialized")
        self.logger.info("   Optimized for XAUUSD 1-minute scalping")
    
    async def start(self, symbols: List[str], equity: float = 10000.0):
        """Start live trading with Ronin Gold V3.0 strategy."""
        try:
            self.logger.info(f"Starting Ronin Gold V3.0 live trading for {symbols}")
            
            # Validate symbols (should be XAUUSD)
            if 'XAUUSD' not in symbols:
                self.logger.warning("V3.0 is optimized for XAUUSD - other symbols may not perform optimally")
            
            # Initialize V3.0 signal processors
            webhook_url = self.cfg.get('webhook_url')  # Optional Telegram webhook
            
            for symbol in symbols:
                processor = RealTimeSignalProcessorV3(
                    symbol=symbol,
                    cfg=self.cfg,
                    signal_callback=self._handle_signal,
                    webhook_url=webhook_url
                )
                self.signal_processors[symbol] = processor
                await processor.start()
            
            # Connect to data feed
            await self.data_feed.connect()
            
            # Subscribe to symbols
            for symbol in symbols:
                await self.data_feed.subscribe([symbol])
            
            # Add callback for tick data
            self.data_feed.add_callback(self._on_tick_data)
            
            # Start trading engine
            await self.trading_engine.start()
            
            self.logger.info(f"Ronin Gold V3.0 live trading started for {symbols}")
        
        except Exception as e:
            self.logger.error(f"Error starting V3.0 live trading: {e}")
            raise
    
    async def stop(self):
        """Stop V3.0 live trading."""
        try:
            self.logger.info("Stopping Ronin Gold V3.0 live trading...")
            
            # Stop signal processors
            for processor in self.signal_processors.values():
                await processor.stop()
            
            # Stop trading engine
            await self.trading_engine.stop()
            
            # Disconnect from data feed
            await self.data_feed.disconnect()
            
            self.logger.info("Ronin Gold V3.0 live trading stopped")
        
        except Exception as e:
            self.logger.error(f"Error stopping V3.0 live trading: {e}")
    
    def _on_tick_data(self, market_data):
        """Handle incoming tick data from data feed."""
        self.logger.debug(f"Received XAUUSD tick: {market_data.symbol} @ {market_data.last:.2f}")
        
        # Convert MarketData to tick_data format
        tick_data = {
            'symbol': market_data.symbol,
            'price': market_data.last,
            'volume': market_data.volume,
            'timestamp': market_data.timestamp
        }
        
        # Send to appropriate V3.0 signal processor
        symbol = market_data.symbol
        if symbol in self.signal_processors:
            self.logger.debug(f"Forwarding tick to V3.0 processor for {symbol}")
            asyncio.create_task(
                self.signal_processors[symbol].process_tick(tick_data)
            )
        else:
            self.logger.warning(f"No V3.0 processor found for symbol {symbol}")
    
    async def _handle_signal(self, signal: Dict[str, Any]):
        """Handle signals from V3.0 processors."""
        try:
            # Send signal to trading engine
            await self.trading_engine.process_signal(signal)
            
            self.logger.info(f"V3.0 Signal processed: {signal['signal_type']} {signal['symbol']}")
        
        except Exception as e:
            self.logger.error(f"Error processing V3.0 signal: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get overall V3.0 system status."""
        processor_status = {}
        total_signals = 0
        
        for symbol, processor in self.signal_processors.items():
            status = processor.get_status()
            processor_status[symbol] = status
            total_signals += status.get('signals_today', 0)
        
        return {
            'strategy_version': 'V3.0',
            'strategy_name': 'Ronin Gold',
            'processors': processor_status,
            'total_signals_today': total_signals,
            'trading_engine_status': self.trading_engine.get_status() if hasattr(self.trading_engine, 'get_status') else {},
            'data_feed_status': self.data_feed.get_status() if hasattr(self.data_feed, 'get_status') else {}
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Ronin Gold V3.0 Live Signal Processor loaded")
    logger.info("XAUUSD 1-minute scalping with trend-following + stat-arb + bell curve theory")
