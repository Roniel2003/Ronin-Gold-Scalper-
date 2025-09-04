"""
Real-time Signal Processor - Stage 5
Bridges real-time market data with Ronin signal generation logic.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Callable, Any
from collections import deque
import asyncio
import threading
import queue

from live_trader import MarketData
from ronin_engine import compute_indicators, generate_signals, build_order
from config import get_config

logger = logging.getLogger(__name__)


class RealTimeSignalProcessor:
    """Real-time signal processor for live trading"""
    
    def __init__(self, config: Dict, signal_callback: Callable = None):
        self.config = config
        self.signal_callback = signal_callback
        
        # Data storage for each symbol
        self.symbol_data = {}  # symbol -> deque of OHLCV bars
        self.current_bars = {}  # symbol -> current incomplete bar
        self.last_signal_time = {}  # symbol -> last signal timestamp
        
        # Configuration
        self.timeframe_minutes = 15  # 15-minute bars
        self.lookback_bars = max(config['z_period'], config['atr_period'], config['vol_period']) + 10
        self.min_bars_for_signals = self.lookback_bars
        
        # Threading
        self.processing_queue = queue.Queue()
        self.is_running = False
        self.processor_thread = None
        
    def start(self):
        """Start the signal processor"""
        self.is_running = True
        self.processor_thread = threading.Thread(target=self._process_signals, daemon=True)
        self.processor_thread.start()
        logger.info("Real-time signal processor started")
    
    def stop(self):
        """Stop the signal processor"""
        self.is_running = False
        if self.processor_thread:
            self.processor_thread.join(timeout=5.0)
        logger.info("Real-time signal processor stopped")
    
    def on_market_data(self, market_data: MarketData):
        """Handle incoming market data"""
        try:
            symbol = market_data.symbol
            timestamp = market_data.timestamp
            
            # Initialize symbol data if needed
            if symbol not in self.symbol_data:
                self.symbol_data[symbol] = deque(maxlen=self.lookback_bars * 2)
                self.current_bars[symbol] = None
                self.last_signal_time[symbol] = None
            
            # Update current bar or create new one
            self._update_bar(symbol, market_data)
            
            # Queue for processing
            self.processing_queue.put((symbol, timestamp))
            
        except Exception as e:
            logger.error(f"Error processing market data for {market_data.symbol}: {e}")
    
    def _update_bar(self, symbol: str, market_data: MarketData):
        """Update or create OHLCV bar from tick data"""
        timestamp = market_data.timestamp
        price = market_data.last
        volume = market_data.volume
        
        # Determine bar timestamp (round down to 15-minute intervals)
        bar_time = self._round_to_timeframe(timestamp)
        
        current_bar = self.current_bars[symbol]
        
        # Check if we need a new bar
        if current_bar is None or current_bar['Time'] != bar_time:
            # Complete previous bar if it exists
            if current_bar is not None:
                self.symbol_data[symbol].append(current_bar)
            
            # Start new bar
            self.current_bars[symbol] = {
                'Time': bar_time,
                'Open': price,
                'High': price,
                'Low': price,
                'Close': price,
                'Volume': volume,
                'tick_count': 1
            }
        else:
            # Update existing bar
            current_bar['High'] = max(current_bar['High'], price)
            current_bar['Low'] = min(current_bar['Low'], price)
            current_bar['Close'] = price
            current_bar['Volume'] += volume
            current_bar['tick_count'] += 1
    
    def _round_to_timeframe(self, timestamp: datetime) -> datetime:
        """Round timestamp down to timeframe boundary"""
        minutes = timestamp.minute
        rounded_minutes = (minutes // self.timeframe_minutes) * self.timeframe_minutes
        
        return timestamp.replace(
            minute=rounded_minutes,
            second=0,
            microsecond=0
        )
    
    def _process_signals(self):
        """Background thread to process signals"""
        while self.is_running:
            try:
                if not self.processing_queue.empty():
                    symbol, timestamp = self.processing_queue.get(timeout=1.0)
                    self._check_for_signals(symbol, timestamp)
                else:
                    # Check all symbols periodically for bar completion
                    for symbol in list(self.symbol_data.keys()):
                        self._check_for_signals(symbol, datetime.now())
                    
                    # Sleep briefly to avoid busy waiting
                    threading.Event().wait(1.0)
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in signal processing thread: {e}")
    
    def _check_for_signals(self, symbol: str, current_time: datetime):
        """Check if we should generate signals for a symbol"""
        try:
            # Need enough historical data
            if len(self.symbol_data[symbol]) < self.min_bars_for_signals:
                return
            
            # Check if current bar is complete (new bar started or sufficient time passed)
            current_bar = self.current_bars[symbol]
            if current_bar is None:
                return
            
            # Only process signals on bar completion or every few minutes
            bar_age = current_time - current_bar['Time']
            if bar_age < timedelta(minutes=self.timeframe_minutes - 1):
                return
            
            # Avoid duplicate signals
            last_signal = self.last_signal_time.get(symbol)
            if last_signal and (current_time - last_signal) < timedelta(minutes=self.timeframe_minutes):
                return
            
            # Create DataFrame from historical data
            historical_data = list(self.symbol_data[symbol])
            if current_bar['tick_count'] > 5:  # Ensure bar has some data
                historical_data.append(current_bar.copy())
            
            if len(historical_data) < self.min_bars_for_signals:
                return
            
            df = pd.DataFrame(historical_data)
            
            # Compute indicators and generate signals
            df = compute_indicators(df, self.config)
            df = generate_signals(df, self.config)
            
            # Check for new signals in the latest bar
            latest_bar = df.iloc[-1]
            
            if latest_bar['signal_long'] or latest_bar['signal_short']:
                # Create signal for live trading engine
                signal = {
                    'symbol': symbol,
                    'timestamp': latest_bar['Time'],
                    'direction': 'LONG' if latest_bar['signal_long'] else 'SHORT',
                    'strength': latest_bar['signal_strength'],
                    'entry_price': latest_bar['Close'],  # Will be refined by live trader
                    'atr': latest_bar['atr'],
                    'zscore': latest_bar['zscore'],
                    'indicators': {
                        'ema_fast': latest_bar['ema_fast'],
                        'ema_slow': latest_bar['ema_slow'],
                        'atr_pct': latest_bar['atr_pct'],
                        'vol_above_avg': latest_bar['vol_above_avg']
                    }
                }
                
                # Send signal to callback
                if self.signal_callback:
                    self.signal_callback(signal)
                
                self.last_signal_time[symbol] = current_time
                
                logger.info(f"🎯 Generated {signal['direction']} signal for {symbol} "
                           f"(strength: {signal['strength']:.2f}, zscore: {signal['zscore']:.2f})")
                
        except Exception as e:
            logger.error(f"Error checking signals for {symbol}: {e}")
    
    def get_latest_data(self, symbol: str, bars: int = 50) -> Optional[pd.DataFrame]:
        """Get latest OHLCV data for a symbol"""
        if symbol not in self.symbol_data:
            return None
        
        data = list(self.symbol_data[symbol])[-bars:]
        if len(data) == 0:
            return None
        
        return pd.DataFrame(data)
    
    def get_status(self) -> Dict:
        """Get processor status"""
        return {
            'is_running': self.is_running,
            'symbols_tracked': len(self.symbol_data),
            'queue_size': self.processing_queue.qsize(),
            'symbols': {
                symbol: {
                    'bars': len(data),
                    'current_bar_ticks': self.current_bars[symbol]['tick_count'] if self.current_bars[symbol] else 0,
                    'last_signal': self.last_signal_time.get(symbol)
                }
                for symbol, data in self.symbol_data.items()
            }
        }


class LiveTradingIntegrator:
    """Integrates all live trading components"""
    
    def __init__(self, config_path: str = None):
        self.config = get_config(config_path) if config_path else get_config()
        
        # Components
        self.signal_processor = None
        self.data_feed_manager = None
        self.trading_engine = None
        self.broker = None
        
        # State
        self.is_running = False
        self.symbols = self.config.get('symbols', ['NVDA', 'TSLA', 'AAPL'])
        
    async def initialize(self, broker, data_feed_manager):
        """Initialize all components"""
        self.broker = broker
        self.data_feed_manager = data_feed_manager
        
        # Create signal processor with callback to trading engine
        self.signal_processor = RealTimeSignalProcessor(
            self.config,
            signal_callback=self._on_signal_generated
        )
        
        # Create trading engine
        from live_trader import LiveTradingEngine
        self.trading_engine = LiveTradingEngine(broker, self.config)
        
        # Connect data feed to signal processor
        self.data_feed_manager.add_callback(self.signal_processor.on_market_data)
        
        logger.info("Live trading integrator initialized")
    
    async def start(self):
        """Start live trading system"""
        if not all([self.broker, self.data_feed_manager, self.signal_processor, self.trading_engine]):
            raise Exception("Components not initialized")
        
        logger.info("Starting live trading system...")
        
        # Start components in order
        await self.data_feed_manager.start()
        await self.data_feed_manager.subscribe_all(self.symbols)
        
        self.signal_processor.start()
        await self.trading_engine.start()
        
        self.is_running = True
        logger.info("🚀 Live trading system started successfully")
    
    async def stop(self):
        """Stop live trading system"""
        logger.info("Stopping live trading system...")
        
        self.is_running = False
        
        # Stop components in reverse order
        if self.trading_engine:
            await self.trading_engine.stop()
        
        if self.signal_processor:
            self.signal_processor.stop()
        
        if self.data_feed_manager:
            await self.data_feed_manager.stop()
        
        logger.info("Live trading system stopped")
    
    def _on_signal_generated(self, signal: Dict):
        """Handle signals from signal processor"""
        try:
            # Add signal to trading engine queue
            if self.trading_engine:
                self.trading_engine.signal_queue.put(signal)
                logger.info(f"Signal queued for execution: {signal['direction']} {signal['symbol']}")
        except Exception as e:
            logger.error(f"Error handling signal: {e}")
    
    def get_status(self) -> Dict:
        """Get system status"""
        status = {
            'is_running': self.is_running,
            'symbols': self.symbols
        }
        
        if self.signal_processor:
            status['signal_processor'] = self.signal_processor.get_status()
        
        if self.trading_engine:
            status['trading_engine'] = self.trading_engine.get_status()
        
        return status


if __name__ == "__main__":
    # Example usage
    async def main():
        from data_feeds import DataFeedManager, SimulatedDataFeed, DataFeedConfig, DataProvider
        from live_trader import SimulatedBroker
        
        # Create components
        config = get_config()
        
        # Simulated broker
        broker = SimulatedBroker(initial_balance=100000.0)
        
        # Simulated data feed
        feed_config = DataFeedConfig(
            provider=DataProvider.SIMULATED,
            symbols=['NVDA', 'TSLA', 'AAPL'],
            update_interval=1
        )
        
        from data_feeds import create_data_feed
        feed = create_data_feed(feed_config)
        
        data_manager = DataFeedManager()
        data_manager.add_feed("simulated", feed)
        
        # Create integrator
        integrator = LiveTradingIntegrator()
        await integrator.initialize(broker, data_manager)
        
        # Start system
        await integrator.start()
        
        # Run for demo
        await asyncio.sleep(30)
        
        # Show status
        status = integrator.get_status()
        print(f"System Status: {status}")
        
        # Stop system
        await integrator.stop()
    
    asyncio.run(main())
