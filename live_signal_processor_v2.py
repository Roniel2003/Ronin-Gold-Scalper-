"""
Live Signal Processor V2.0 - Real-time signal processing for Ronin V2.0 Strategy
Processes 1-minute tick data and generates V2.0 signals with EMA, Z-score, ATR, OB, and FVG.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timezone, timedelta
import pytz
from collections import deque
import logging

from ronin_engine_v2 import (
    calculate_indicators_v2, 
    generate_signals_v2, 
    build_order_v2,
    is_in_session_v2
)


class RealTimeSignalProcessorV2:
    """
    Real-time signal processor for Ronin V2.0 strategy.
    Converts tick data to 1-minute bars and generates V2.0 signals.
    """
    
    def __init__(self, symbol: str, cfg: Dict[str, Any], 
                 signal_callback: Optional[Callable] = None):
        """
        Initialize the V2.0 signal processor.
        
        Args:
            symbol: Trading symbol (e.g., 'NVDA')
            cfg: Configuration dictionary with V2.0 parameters
            signal_callback: Callback function for generated signals
        """
        self.symbol = symbol
        self.cfg = cfg
        self.signal_callback = signal_callback
        
        # V2.0 Parameters
        self.timeframe = cfg.get('timeframe', '1min')
        self.max_periods = max(cfg.get('ema_periods', [200, 50, 21])) + 50  # Buffer for indicators
        
        # Data storage (1-minute bars)
        self.bars = deque(maxlen=self.max_periods)
        self.current_bar = None
        self.last_bar_time = None
        
        # Timezone handling
        self.timezone = pytz.timezone(cfg.get('timezone', 'US/Eastern'))
        
        # State tracking
        self.is_running = False
        self.last_signal_time = None
        
        # Setup logging
        self.logger = logging.getLogger(f"SignalProcessorV2_{symbol}")
        
        self.logger.info(f"✅ V2.0 Signal Processor initialized for {symbol}")
        self.logger.info(f"   Timeframe: {self.timeframe}")
        self.logger.info(f"   Max periods: {self.max_periods}")
        self.logger.info(f"   Session: {cfg.get('session_start_hour', 9)}:{cfg.get('session_start_minute', 30):02d} - "
              f"{cfg.get('session_end_hour', 11)}:{cfg.get('session_end_minute', 0):02d} {cfg.get('timezone', 'US/Eastern')}")
    
    async def start(self):
        """Start the signal processor."""
        self.is_running = True
        self.logger.info(f"V2.0 Signal processor started for {self.symbol}")
    
    async def stop(self):
        """Stop the signal processor."""
        self.is_running = False
        self.logger.info(f"V2.0 Signal processor stopped for {self.symbol}")
    
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
            
            self.logger.debug(f"🔄 Processing tick: {self.symbol} @ {price:.2f} at {timestamp}")
            
            # Convert to 1-minute bar timestamp
            bar_time = self._get_bar_time(timestamp)
            
            # Create or update current bar
            if self.current_bar is None or bar_time != self.current_bar['Time']:
                # Finalize previous bar if exists
                if self.current_bar is not None:
                    self.logger.debug(f"📊 Finalizing bar for {self.current_bar['Time']}")
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
                self.logger.debug(f"🆕 Started new bar for {bar_time}")
            else:
                # Update current bar
                self.current_bar['High'] = max(self.current_bar['High'], price)
                self.current_bar['Low'] = min(self.current_bar['Low'], price)
                self.current_bar['Close'] = price
                self.current_bar['Volume'] += volume
                self.logger.debug(f"📈 Updated current bar: OHLC {self.current_bar['Open']:.2f}/{self.current_bar['High']:.2f}/{self.current_bar['Low']:.2f}/{self.current_bar['Close']:.2f}")
        
        except Exception as e:
            self.logger.error(f"❌ Error processing tick: {e}")
            self.logger.error(f"   Tick data: {tick_data}")
    
    def _get_bar_time(self, timestamp: datetime) -> datetime:
        """
        Convert timestamp to 1-minute bar time.
        
        Args:
            timestamp: Raw timestamp
            
        Returns:
            1-minute bar timestamp (truncated to minute)
        """
        # Ensure timezone awareness
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        
        # Convert to Eastern time for session logic
        et_time = timestamp.astimezone(self.timezone)
        
        # Truncate to minute boundary
        bar_time = et_time.replace(second=0, microsecond=0)
        
        return bar_time
    
    async def _finalize_bar(self):
        """Finalize the current bar and process for signals."""
        if self.current_bar is None:
            return
        
        # Add bar to history
        self.bars.append(self.current_bar.copy())
        self.last_bar_time = self.current_bar['Time']
        
        # Debug: Log bar accumulation
        self.logger.info(f"📊 Bar finalized: {self.current_bar['Time']} | Bars accumulated: {len(self.bars)}/{self.max_periods}")
        self.logger.info(f"   OHLCV: O:{self.current_bar['Open']:.2f} H:{self.current_bar['High']:.2f} L:{self.current_bar['Low']:.2f} C:{self.current_bar['Close']:.2f} V:{self.current_bar['Volume']}")
        
        # Check if we have enough data for indicators
        min_required = max(self.cfg.get('ema_periods', [200, 50, 21]))
        if len(self.bars) < min_required:
            self.logger.warning(f"⚠️ Insufficient data: {len(self.bars)} bars, need {min_required} for EMA_200")
            return
        
        self.logger.info(f"✅ Sufficient data available: {len(self.bars)} bars >= {min_required} required")
        
        # Convert to DataFrame for processing
        df = pd.DataFrame(list(self.bars))
        
        # Debug: Log DataFrame structure
        self.logger.info(f"🔧 DataFrame created: {df.shape} with columns {list(df.columns)}")
        self.logger.info(f"   Price range: {df['Close'].min():.2f} - {df['Close'].max():.2f}")
        
        # Add session information
        df['in_session'] = df['Time'].apply(lambda x: is_in_session_v2(x, self.cfg))
        session_bars = df['in_session'].sum()
        self.logger.info(f"🕐 Session filtering: {session_bars}/{len(df)} bars in session")
        
        # Calculate V2.0 indicators
        self.logger.info(f"🔧 Starting V2.0 indicator calculations...")
        df = calculate_indicators_v2(df, self.cfg)
        
        # Debug: Check if indicators calculated
        latest = df.iloc[-1]
        self.logger.info(f"📈 Latest indicators calculated:")
        self.logger.info(f"   EMA 21/50/200: {latest.get('EMA_21', 0):.2f} / {latest.get('EMA_50', 0):.2f} / {latest.get('EMA_200', 0):.2f}")
        self.logger.info(f"   Z-Score: {latest.get('Z_Score', 0):.2f}")
        self.logger.info(f"   ATR: {latest.get('ATR', 0):.2f}")
        
        # Validate indicators
        if latest.get('EMA_21', 0) == 0 and latest.get('EMA_50', 0) == 0:
            self.logger.error("❌ CRITICAL: All EMA values are 0 - indicators failed to calculate!")
        if latest.get('Z_Score', 0) == 0:
            self.logger.warning("⚠️ Z-Score is 0 - momentum calculation issue")
        if latest.get('ATR', 0) == 0:
            self.logger.warning("⚠️ ATR is 0 - volatility calculation issue")
        
        # Generate V2.0 signals
        self.logger.info(f"🎯 Starting V2.0 signal generation...")
        df = generate_signals_v2(df, self.cfg)
        
        # Check for new signals on the latest bar
        latest_bar = df.iloc[-1]
        
        # Debug: Log signal conditions
        self.logger.info(f"🔍 Signal check for latest bar:")
        self.logger.info(f"   In session: {latest_bar.get('in_session', False)}")
        self.logger.info(f"   Signal long: {latest_bar.get('signal_long', False)}")
        self.logger.info(f"   Signal short: {latest_bar.get('signal_short', False)}")
        
        if latest_bar.get('signal_long', False) or latest_bar.get('signal_short', False):
            self.logger.info(f"🚨 SIGNAL DETECTED! Processing...")
            await self._handle_signal(latest_bar)
        else:
            self.logger.debug(f"📊 No signals generated for current bar")
        
        # Log bar completion
        self.logger.debug(f"Bar finalized: {self.current_bar['Time']} "
                         f"OHLCV: {self.current_bar['Open']:.2f}/{self.current_bar['High']:.2f}/"
                         f"{self.current_bar['Low']:.2f}/{self.current_bar['Close']:.2f}/{self.current_bar['Volume']}")
    
    async def _handle_signal(self, signal_bar: pd.Series):
        """
        Handle a new V2.0 signal.
        
        Args:
            signal_bar: Pandas Series with signal data
        """
        try:
            signal_time = signal_bar['Time']
            
            # Prevent duplicate signals
            if self.last_signal_time == signal_time:
                return
            
            self.last_signal_time = signal_time
            
            # Determine signal type and direction
            signal_type = None
            direction = None
            
            if signal_bar.get('signal_long', False):
                signal_type = signal_bar.get('signal_type', 'Bullish')
                direction = 'LONG'
            elif signal_bar.get('signal_short', False):
                signal_type = signal_bar.get('signal_type', 'Bearish')
                direction = 'SHORT'
            
            if signal_type is None:
                return
            
            # Create signal dictionary
            signal = {
                'symbol': self.symbol,
                'timestamp': signal_time,
                'signal_type': signal_type,
                'direction': direction,
                'entry_price': signal_bar['Close'],
                'ema_200': signal_bar.get('EMA_200', 0),
                'ema_50': signal_bar.get('EMA_50', 0),
                'ema_21': signal_bar.get('EMA_21', 0),
                'z_score': signal_bar.get('Z_Score', 0),
                'atr': signal_bar.get('ATR', 0),
                'in_session': signal_bar.get('in_session', False),
                'order_block': signal_bar.get('order_block', False),
                'fvg_signal': signal_bar.get('fvg_signal', False),
                'strategy_version': 'V2.0'
            }
            
            # Log signal
            self.logger.info(f"V2.0 Signal generated: {signal_type} {direction} at {signal_time}")
            self.logger.info(f"   Z-Score: {signal_bar.get('Z_Score', 0):.2f}, ATR: ${signal_bar.get('ATR', 0):.2f}")
            self.logger.info(f"   EMA: {signal_bar.get('EMA_21', 0):.2f} / {signal_bar.get('EMA_50', 0):.2f} / {signal_bar.get('EMA_200', 0):.2f}")
            
            # Send signal to callback
            if self.signal_callback:
                await self.signal_callback(signal)
        
        except Exception as e:
            self.logger.error(f"Error handling signal: {e}")
    
    def get_latest_data(self) -> Optional[pd.DataFrame]:
        """
        Get the latest processed data with indicators.
        
        Returns:
            DataFrame with latest bars and indicators, or None if insufficient data
        """
        if len(self.bars) < 50:  # Minimum for meaningful analysis
            return None
        
        df = pd.DataFrame(list(self.bars))
        df['in_session'] = df['Time'].apply(lambda x: is_in_session_v2(x, self.cfg))
        df = calculate_indicators_v2(df, self.cfg)
        df = generate_signals_v2(df, self.cfg)
        
        return df
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current processor status.
        
        Returns:
            Status dictionary
        """
        return {
            'symbol': self.symbol,
            'is_running': self.is_running,
            'bars_count': len(self.bars),
            'last_bar_time': self.last_bar_time,
            'last_signal_time': self.last_signal_time,
            'current_bar': self.current_bar,
            'strategy_version': 'V2.0',
            'timeframe': self.timeframe
        }


class LiveTradingIntegratorV2:
    """
    Integrates V2.0 signal processor with live trading engine.
    """
    
    def __init__(self, broker, data_feed, trading_engine, cfg: Dict[str, Any]):
        """
        Initialize the V2.0 live trading integrator.
        
        Args:
            broker: Broker interface
            data_feed: Data feed interface
            trading_engine: Live trading engine
            cfg: Configuration dictionary
        """
        self.broker = broker
        self.data_feed = data_feed
        self.trading_engine = trading_engine
        self.cfg = cfg
        
        # Signal processors for each symbol
        self.signal_processors = {}
        
        # Setup logging
        self.logger = logging.getLogger("LiveTradingIntegratorV2")
        
        self.logger.info("✅ V2.0 Live Trading Integrator initialized")
    
    async def start(self, symbols: List[str], equity: float = 10000.0):
        """
        Start live trading with V2.0 strategy.
        
        Args:
            symbols: List of symbols to trade
            equity: Starting equity
        """
        try:
            self.logger.info(f"🚀 Starting V2.0 live trading for {symbols}")
            
            # Initialize signal processors
            for symbol in symbols:
                processor = RealTimeSignalProcessorV2(
                    symbol=symbol,
                    cfg=self.cfg,
                    signal_callback=self._handle_signal
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
            
            self.logger.info(f"V2.0 Live trading started for {symbols}")
        
        except Exception as e:
            self.logger.error(f"Error starting V2.0 live trading: {e}")
            raise
    
    async def stop(self):
        """Stop live trading."""
        try:
            self.logger.info("🛑 Stopping V2.0 live trading...")
            
            # Stop signal processors
            for processor in self.signal_processors.values():
                await processor.stop()
            
            # Stop trading engine
            await self.trading_engine.stop()
            
            # Disconnect from data feed
            await self.data_feed.disconnect()
            
            self.logger.info("V2.0 Live trading stopped")
        
        except Exception as e:
            self.logger.error(f"Error stopping V2.0 live trading: {e}")
    
    def _on_tick_data(self, market_data):
        """
        Handle incoming tick data from data feed.
        
        Args:
            market_data: MarketData object from data feed
        """
        # Convert MarketData to tick_data format expected by processors
        tick_data = {
            'symbol': market_data.symbol,
            'price': market_data.last,
            'volume': market_data.volume,
            'timestamp': market_data.timestamp
        }
        
        # Send to appropriate signal processor
        symbol = market_data.symbol
        if symbol in self.signal_processors:
            # Run async processor in event loop
            asyncio.create_task(
                self.signal_processors[symbol].process_tick(tick_data)
            )
    
    async def _handle_signal(self, signal: Dict[str, Any]):
        """
        Handle signals from V2.0 processors.
        
        Args:
            signal: Signal dictionary
        """
        try:
            # Send signal to trading engine
            await self.trading_engine.process_signal(signal)
            
            self.logger.info(f"V2.0 Signal processed: {signal['signal_type']} {signal['direction']} "
                           f"for {signal['symbol']}")
        
        except Exception as e:
            self.logger.error(f"Error processing V2.0 signal: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get overall system status.
        
        Returns:
            Status dictionary
        """
        processor_status = {}
        for symbol, processor in self.signal_processors.items():
            processor_status[symbol] = processor.get_status()
        
        return {
            'strategy_version': 'V2.0',
            'processors': processor_status,
            'trading_engine_status': self.trading_engine.get_status() if hasattr(self.trading_engine, 'get_status') else {},
            'data_feed_status': self.data_feed.get_status() if hasattr(self.data_feed, 'get_status') else {}
        }


# Example usage and testing
async def test_signal_processor_v2():
    """Test the V2.0 signal processor with simulated data."""
    from config import get_config
    
    cfg = get_config()
    
    # Create processor
    processor = RealTimeSignalProcessorV2('NVDA', cfg)
    await processor.start()
    
    # Simulate some tick data
    base_time = datetime.now(tz=pytz.timezone('US/Eastern'))
    base_price = 100.0
    
    for i in range(300):  # 5 minutes of ticks
        tick_time = base_time + timedelta(seconds=i)
        price = base_price + np.random.normal(0, 0.1)
        
        tick_data = {
            'symbol': 'NVDA',
            'price': price,
            'volume': 100,
            'timestamp': tick_time
        }
        
        await processor.process_tick(tick_data)
        
        # Small delay to simulate real-time
        await asyncio.sleep(0.01)
    
    # Get status
    status = processor.get_status()
    self.logger.info(f"Final status: {status}")
    
    await processor.stop()


if __name__ == "__main__":
    logging.basicConfig(filename='live_trading.log', level=logging.INFO)
    print("🔄 Testing V2.0 Signal Processor...")
    asyncio.run(test_signal_processor_v2())
    print("✅ V2.0 Signal Processor test complete")
