"""
Real-time Data Feeds - Stage 5
Multi-provider real-time market data integration with WebSocket and REST API support.
"""

import asyncio
import websockets
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import threading
import queue
import time
from enum import Enum

from live_trader import MarketData

logger = logging.getLogger(__name__)


class DataProvider(Enum):
    ALPACA = "alpaca"
    POLYGON = "polygon"
    FINNHUB = "finnhub"
    YAHOO = "yahoo"
    SIMULATED = "simulated"
    MT5 = "mt5"


@dataclass
class DataFeedConfig:
    """Configuration for data feeds"""
    provider: DataProvider
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    base_url: Optional[str] = None
    websocket_url: Optional[str] = None
    symbols: List[str] = None
    update_interval: int = 1  # seconds


class DataFeed(ABC):
    """Abstract base class for data feeds"""
    
    def __init__(self, config: DataFeedConfig):
        self.config = config
        self.is_connected = False
        self.callbacks = []
        self.data_queue = queue.Queue()
        
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to data provider"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from data provider"""
        pass
    
    @abstractmethod
    async def subscribe(self, symbols: List[str]):
        """Subscribe to real-time data for symbols"""
        pass
    
    @abstractmethod
    async def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from symbols"""
        pass
    
    def add_callback(self, callback: Callable[[MarketData], None]):
        """Add callback for market data updates"""
        self.callbacks.append(callback)
    
    def _notify_callbacks(self, market_data: MarketData):
        """Notify all callbacks of new market data"""
        for callback in self.callbacks:
            try:
                callback(market_data)
            except Exception as e:
                logger.error(f"Error in callback: {e}")


class AlpacaDataFeed(DataFeed):
    """Alpaca real-time data feed"""
    
    def __init__(self, config: DataFeedConfig):
        super().__init__(config)
        self.websocket = None
        self.auth_token = None
        
    async def connect(self) -> bool:
        """Connect to Alpaca data feed"""
        try:
            # Authenticate with Alpaca API
            auth_url = f"{self.config.base_url or 'https://paper-api.alpaca.markets'}/v2/account"
            headers = {
                'APCA-API-KEY-ID': self.config.api_key,
                'APCA-API-SECRET-KEY': self.config.api_secret
            }
            
            response = requests.get(auth_url, headers=headers)
            if response.status_code != 200:
                logger.error(f"Alpaca authentication failed: {response.text}")
                return False
            
            # Connect to WebSocket
            ws_url = self.config.websocket_url or "wss://stream.data.alpaca.markets/v2/iex"
            self.websocket = await websockets.connect(ws_url)
            
            # Authenticate WebSocket
            auth_msg = {
                "action": "auth",
                "key": self.config.api_key,
                "secret": self.config.api_secret
            }
            await self.websocket.send(json.dumps(auth_msg))
            
            response = await self.websocket.recv()
            auth_response = json.loads(response)
            
            if auth_response.get("T") == "success":
                self.is_connected = True
                logger.info("Connected to Alpaca data feed")
                
                # Start message handler
                asyncio.create_task(self._message_handler())
                return True
            else:
                logger.error(f"Alpaca WebSocket auth failed: {auth_response}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to Alpaca: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from Alpaca"""
        if self.websocket:
            await self.websocket.close()
        self.is_connected = False
        logger.info("Disconnected from Alpaca data feed")
    
    async def subscribe(self, symbols: List[str]):
        """Subscribe to Alpaca real-time data"""
        if not self.is_connected:
            raise Exception("Not connected to Alpaca")
        
        subscribe_msg = {
            "action": "subscribe",
            "quotes": symbols,
            "trades": symbols
        }
        await self.websocket.send(json.dumps(subscribe_msg))
        logger.info(f"Subscribed to Alpaca data for {symbols}")
    
    async def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from symbols"""
        if not self.is_connected:
            return
        
        unsubscribe_msg = {
            "action": "unsubscribe",
            "quotes": symbols,
            "trades": symbols
        }
        await self.websocket.send(json.dumps(unsubscribe_msg))
        logger.info(f"Unsubscribed from Alpaca data for {symbols}")
    
    async def _message_handler(self):
        """Handle incoming WebSocket messages"""
        try:
            while self.is_connected:
                message = await self.websocket.recv()
                data = json.loads(message)
                
                for item in data:
                    if item.get("T") == "q":  # Quote
                        market_data = self._parse_quote(item)
                        if market_data:
                            self._notify_callbacks(market_data)
                    elif item.get("T") == "t":  # Trade
                        market_data = self._parse_trade(item)
                        if market_data:
                            self._notify_callbacks(market_data)
                            
        except Exception as e:
            logger.error(f"Error in Alpaca message handler: {e}")
            self.is_connected = False
    
    def _parse_quote(self, quote_data: Dict) -> Optional[MarketData]:
        """Parse Alpaca quote data"""
        try:
            return MarketData(
                symbol=quote_data["S"],
                timestamp=datetime.fromisoformat(quote_data["t"].replace("Z", "+00:00")),
                bid=float(quote_data["bp"]),
                ask=float(quote_data["ap"]),
                last=float(quote_data.get("ap", quote_data["bp"])),  # Use ask as last if no trade
                volume=0.0  # Quote doesn't have volume
            )
        except Exception as e:
            logger.error(f"Error parsing Alpaca quote: {e}")
            return None
    
    def _parse_trade(self, trade_data: Dict) -> Optional[MarketData]:
        """Parse Alpaca trade data"""
        try:
            return MarketData(
                symbol=trade_data["S"],
                timestamp=datetime.fromisoformat(trade_data["t"].replace("Z", "+00:00")),
                bid=0.0,  # Trade doesn't have bid/ask
                ask=0.0,
                last=float(trade_data["p"]),
                volume=float(trade_data["s"])
            )
        except Exception as e:
            logger.error(f"Error parsing Alpaca trade: {e}")
            return None


class SimulatedDataFeed(DataFeed):
    """Simulated data feed for testing"""
    
    def __init__(self, config: DataFeedConfig):
        super().__init__(config)
        self.symbols = config.symbols or ["NVDA", "TSLA", "AAPL"]
        self.prices = {symbol: 100.0 for symbol in self.symbols}
        self.task = None
        
    async def connect(self) -> bool:
        """Connect to simulated feed"""
        self.is_connected = True
        logger.info("Connected to simulated data feed")
        return True
    
    async def disconnect(self):
        """Disconnect from simulated feed"""
        if self.task:
            self.task.cancel()
        self.is_connected = False
        logger.info("Disconnected from simulated data feed")
    
    async def subscribe(self, symbols: List[str]):
        """Subscribe to simulated data"""
        self.symbols.extend([s for s in symbols if s not in self.symbols])
        for symbol in symbols:
            if symbol not in self.prices:
                self.prices[symbol] = 100.0
        
        # Start data generation
        if not self.task:
            self.task = asyncio.create_task(self._generate_data())
        
        logger.info(f"Subscribed to simulated data for {symbols}")
    
    async def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from symbols"""
        for symbol in symbols:
            if symbol in self.symbols:
                self.symbols.remove(symbol)
            if symbol in self.prices:
                del self.prices[symbol]
        
        logger.info(f"Unsubscribed from simulated data for {symbols}")
    
    async def _generate_data(self):
        """Generate simulated market data"""
        while self.is_connected:
            try:
                for symbol in self.symbols:
                    # Generate random price movement
                    change_pct = np.random.normal(0, 0.001)  # 0.1% volatility
                    self.prices[symbol] *= (1 + change_pct)
                    
                    # Create market data
                    price = self.prices[symbol]
                    spread = price * 0.001  # 0.1% spread
                    
                    market_data = MarketData(
                        symbol=symbol,
                        timestamp=datetime.now(),
                        bid=price - spread/2,
                        ask=price + spread/2,
                        last=price,
                        volume=np.random.randint(100, 1000)
                    )
                    
                    self._notify_callbacks(market_data)
                
                await asyncio.sleep(self.config.update_interval)
                
            except Exception as e:
                logger.error(f"Error generating simulated data: {e}")
                break


class MT5DataFeed(DataFeed):
    """MT5 real-time data feed using MetaTrader 5 terminal"""
    
    def __init__(self, config: DataFeedConfig):
        super().__init__(config)
        self.mt5_initialized = False
        self.task = None
        
    async def connect(self) -> bool:
        """Connect to MT5 terminal"""
        try:
            import MetaTrader5 as mt5
            
            # Initialize MT5 connection
            if not mt5.initialize():
                logger.error("Failed to initialize MT5 terminal")
                return False
            
            # Check if MT5 is connected
            if not mt5.terminal_info():
                logger.error("MT5 terminal not connected")
                mt5.shutdown()
                return False
                
            self.mt5_initialized = True
            self.is_connected = True
            logger.info("Connected to MT5 data feed")
            return True
            
        except ImportError:
            logger.error("MetaTrader5 package not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to MT5: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from MT5"""
        try:
            if self.task:
                self.task.cancel()
            
            if self.mt5_initialized:
                import MetaTrader5 as mt5
                mt5.shutdown()
                
            self.is_connected = False
            self.mt5_initialized = False
            logger.info("Disconnected from MT5 data feed")
            
        except Exception as e:
            logger.error(f"Error disconnecting from MT5: {e}")
    
    async def subscribe(self, symbols: List[str]):
        """Subscribe to MT5 real-time data"""
        if not self.is_connected:
            raise Exception("Not connected to MT5")
        
        try:
            import MetaTrader5 as mt5
            
            # Verify symbols exist
            for symbol in symbols:
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    logger.warning(f"Symbol {symbol} not found in MT5")
                else:
                    # Enable symbol in Market Watch
                    if not mt5.symbol_select(symbol, True):
                        logger.warning(f"Failed to select symbol {symbol}")
            
            # Start data generation task
            if not self.task:
                self.task = asyncio.create_task(self._generate_data(symbols))
            
            logger.info(f"Subscribed to MT5 data for {symbols}")
            
        except Exception as e:
            logger.error(f"Error subscribing to MT5 data: {e}")
            raise
    
    async def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from symbols"""
        try:
            import MetaTrader5 as mt5
            
            for symbol in symbols:
                mt5.symbol_select(symbol, False)
            
            logger.info(f"Unsubscribed from MT5 data for {symbols}")
            
        except Exception as e:
            logger.error(f"Error unsubscribing from MT5: {e}")
    
    async def _generate_data(self, symbols: List[str]):
        """Generate real-time data from MT5"""
        import MetaTrader5 as mt5
        
        logger.info(f"[MT5] Starting MT5 data generation for {symbols}")
        logger.info(f"   Update interval: {self.config.update_interval} seconds")
        logger.info(f"   Registered callbacks: {len(self.callbacks)}")
        
        tick_count = 0
        last_status_log = 0
        
        while self.is_connected:
            try:
                for symbol in symbols:
                    # Get current tick data
                    tick = mt5.symbol_info_tick(symbol)
                    if tick is not None:
                        tick_count += 1
                        
                        # For XAUUSD, use bid price if last price is 0 or invalid
                        price = tick.last
                        if price <= 0 and symbol == 'XAUUSD':
                            price = tick.bid  # Use bid price for gold
                        
                        # Create market data
                        market_data = MarketData(
                            symbol=symbol,
                            timestamp=datetime.fromtimestamp(tick.time),
                            bid=tick.bid,
                            ask=tick.ask,
                            last=price,  # Use corrected price
                            volume=tick.volume if hasattr(tick, 'volume') else 0
                        )
                        
                        # Log first few ticks for debugging
                        if tick_count <= 5:
                            logger.info(f"[TICK] Tick #{tick_count} for {symbol}: bid={tick.bid}, ask={tick.ask}, last={price} (corrected)")
                        
                        # Notify callbacks
                        if self.callbacks:
                            self._notify_callbacks(market_data)
                            if tick_count <= 5:
                                logger.info(f"[NOTIFY] Notified {len(self.callbacks)} callbacks for {symbol}")
                        else:
                            logger.warning(f"[WARNING] No callbacks registered for market data!")
                    else:
                        logger.warning(f"[WARNING] No tick data available for {symbol}")
                
                # Periodic status logging (every 60 seconds)
                if tick_count - last_status_log >= 60:
                    logger.info(f"[STATUS] MT5 Data Status: {tick_count} ticks processed, {len(self.callbacks)} callbacks active")
                    last_status_log = tick_count
                
                # Update every second
                await asyncio.sleep(self.config.update_interval)
                
            except Exception as e:
                logger.error(f"[ERROR] Error generating MT5 data: {e}")
                await asyncio.sleep(1)


class DataFeedManager:
    """Manages multiple data feeds"""
    
    def __init__(self):
        self.feeds = {}
        self.callbacks = []
        self.is_running = False
        
    def add_feed(self, name: str, feed: DataFeed):
        """Add a data feed"""
        self.feeds[name] = feed
        feed.add_callback(self._on_market_data)
        logger.info(f"Added data feed: {name}")
    
    def remove_feed(self, name: str):
        """Remove a data feed"""
        if name in self.feeds:
            del self.feeds[name]
            logger.info(f"Removed data feed: {name}")
    
    def add_callback(self, callback: Callable[[MarketData], None]):
        """Add callback for market data"""
        self.callbacks.append(callback)
    
    async def start(self):
        """Start all data feeds"""
        logger.info("Starting data feed manager...")
        
        for name, feed in self.feeds.items():
            try:
                if await feed.connect():
                    logger.info(f"Started feed: {name}")
                else:
                    logger.error(f"Failed to start feed: {name}")
            except Exception as e:
                logger.error(f"Error starting feed {name}: {e}")
        
        self.is_running = True
        logger.info("Data feed manager started")
    
    async def stop(self):
        """Stop all data feeds"""
        logger.info("Stopping data feed manager...")
        
        for name, feed in self.feeds.items():
            try:
                await feed.disconnect()
                logger.info(f"Stopped feed: {name}")
            except Exception as e:
                logger.error(f"Error stopping feed {name}: {e}")
        
        self.is_running = False
        logger.info("Data feed manager stopped")
    
    async def subscribe_all(self, symbols: List[str]):
        """Subscribe all feeds to symbols"""
        for name, feed in self.feeds.items():
            try:
                await feed.subscribe(symbols)
            except Exception as e:
                logger.error(f"Error subscribing feed {name}: {e}")
    
    async def unsubscribe_all(self, symbols: List[str]):
        """Unsubscribe all feeds from symbols"""
        for name, feed in self.feeds.items():
            try:
                await feed.unsubscribe(symbols)
            except Exception as e:
                logger.error(f"Error unsubscribing feed {name}: {e}")
    
    def _on_market_data(self, market_data: MarketData):
        """Handle market data from feeds"""
        for callback in self.callbacks:
            try:
                callback(market_data)
            except Exception as e:
                logger.error(f"Error in market data callback: {e}")


def create_data_feed(config: DataFeedConfig) -> DataFeed:
    """Factory function to create data feeds"""
    if config.provider == DataProvider.ALPACA:
        return AlpacaDataFeed(config)
    elif config.provider == DataProvider.SIMULATED:
        return SimulatedDataFeed(config)
    elif config.provider == DataProvider.MT5:
        return MT5DataFeed(config)
    else:
        raise ValueError(f"Unsupported data provider: {config.provider}")


if __name__ == "__main__":
    # Example usage
    async def on_market_data(data: MarketData):
        print(f"{data.symbol}: ${data.last:.2f} @ {data.timestamp}")
    
    async def main():
        # Create simulated data feed
        config = DataFeedConfig(
            provider=DataProvider.SIMULATED,
            symbols=["NVDA", "TSLA", "AAPL"],
            update_interval=1
        )
        
        feed = create_data_feed(config)
        manager = DataFeedManager()
        
        manager.add_feed("simulated", feed)
        manager.add_callback(on_market_data)
        
        await manager.start()
        await manager.subscribe_all(["NVDA", "TSLA", "AAPL"])
        
        # Run for 10 seconds
        await asyncio.sleep(10)
        
        await manager.stop()
