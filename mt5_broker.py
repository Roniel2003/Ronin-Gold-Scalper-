"""
MetaTrader 5 Broker Integration
Real-time trading and data feed integration with MT5 platform.
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Any
import asyncio
import threading
import time

from live_trader import BrokerAPI, Order, Position, MarketData, OrderStatus, OrderType, OrderSide
from data_feeds import DataFeed, DataFeedConfig

logger = logging.getLogger(__name__)


class MT5Broker(BrokerAPI):
    """MetaTrader 5 broker implementation"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.connected = False
        self.account_info = {}
        self.positions_cache = {}
        self.orders_cache = {}
        
        # MT5 specific settings from nested config
        mt5_config = config.get('broker', {}).get('mt5', {})
        self.server = mt5_config.get('server', '')
        self.login = mt5_config.get('login', 0)
        self.password = mt5_config.get('password', '')
        self.path = mt5_config.get('path', '')  # MT5 installation path
        self.timeout = mt5_config.get('timeout', 60000) # Default to 60s
        
    async def connect(self) -> bool:
        """Connect to MetaTrader 5"""
        logger.info("Attempting to initialize and login to MT5...")

        # Initialize connection to the MetaTrader 5 terminal
        # Build initialization parameters dynamically
        init_params = {
            'login': self.login,
            'password': self.password,
            'server': self.server,
            'timeout': self.timeout
        }
        
        # Only add path if it's not empty
        if self.path:
            init_params['path'] = self.path
        
        if not mt5.initialize(**init_params):
            logger.error(f"MT5 initialize() or login() failed, error code = {mt5.last_error()}")
            mt5.shutdown()
            return False

        logger.info("MT5 initialized and logged in successfully.")

        # Log terminal info after login
        terminal_info = mt5.terminal_info()
        if terminal_info:
            logger.info(f"Terminal info: Name='{terminal_info.name}', Company='{terminal_info.company}', AlgoTrading='{terminal_info.trade_allowed}'")
            if not terminal_info.trade_allowed:
                logger.warning("Algo trading is disabled. Please check your MT5 terminal settings.")
        else:
            logger.warning("Could not retrieve terminal info.")

        account_info = mt5.account_info()
        if account_info:
            logger.info(f"Account Info: Balance={account_info.balance}, Equity={account_info.equity}, Profit={account_info.profit}")
            self.account_info = account_info._asdict()
            self.connected = True
            
            logger.info(f"Connected to MT5 - Account: {self.account_info.get('login', 'Unknown')}")
            logger.info(f"Balance: ${self.account_info.get('balance', 0):,.2f}")
            logger.info(f"Equity: ${self.account_info.get('equity', 0):,.2f}")
            
            return True
        else:
            logger.error(f"Failed to get account info after login: {mt5.last_error()}")
            mt5.shutdown()
            return False

    async def disconnect(self):
        """Disconnect from MetaTrader 5"""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("Disconnected from MT5")
    
    async def submit_order(self, order: Order) -> str:
        """Submit order to MT5"""
        if not self.connected:
            raise Exception("Not connected to MT5")
        
        try:
            # Convert Ronin order to MT5 request
            symbol = order.symbol
            
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                raise Exception(f"Symbol {symbol} not found")
            
            if not symbol_info.visible:
                if not mt5.symbol_select(symbol, True):
                    raise Exception(f"Failed to select symbol {symbol}")
            
            # Determine order type
            if order.side == OrderSide.BUY:
                order_type = mt5.ORDER_TYPE_BUY if order.order_type == OrderType.MARKET else mt5.ORDER_TYPE_BUY_LIMIT
            else:
                order_type = mt5.ORDER_TYPE_SELL if order.order_type == OrderType.MARKET else mt5.ORDER_TYPE_SELL_LIMIT
            
            # Create request
            request = {
                "action": mt5.TRADE_ACTION_DEAL if order.order_type == OrderType.MARKET else mt5.TRADE_ACTION_PENDING,
                "symbol": symbol,
                "volume": float(order.quantity),
                "type": order_type,
                "magic": 234000,  # Magic number for Ronin orders
                "comment": "Ronin Bot",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            # Add price for limit orders
            if order.order_type == OrderType.LIMIT and order.price:
                request["price"] = float(order.price)
            
            # Add stop loss and take profit if specified
            if order.stop_price:
                if order.side == OrderSide.BUY:
                    request["sl"] = float(order.stop_price)
                else:
                    request["tp"] = float(order.stop_price)
            
            # Send order
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                raise Exception(f"Order failed: {result.comment} (Code: {result.retcode})")
            
            # Update order with MT5 info
            order.broker_order_id = str(result.order)
            order.order_id = str(result.order)
            order.status = OrderStatus.FILLED if result.retcode == mt5.TRADE_RETCODE_DONE else OrderStatus.SUBMITTED
            order.timestamp = datetime.now()
            
            if hasattr(result, 'price'):
                order.avg_fill_price = result.price
                order.filled_quantity = order.quantity
                order.fill_timestamp = datetime.now()
            
            logger.info(f"MT5 order submitted: {order.order_id} - {order.side.value} {order.quantity} {symbol}")
            
            return order.order_id
            
        except Exception as e:
            logger.error(f"Failed to submit MT5 order: {e}")
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order in MT5"""
        try:
            # Get order info
            orders = mt5.orders_get(ticket=int(order_id))
            if not orders:
                logger.warning(f"Order {order_id} not found for cancellation")
                return False
            
            order = orders[0]
            
            # Create cancel request
            request = {
                "action": mt5.TRADE_ACTION_REMOVE,
                "order": int(order_id),
            }
            
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(f"MT5 order cancelled: {order_id}")
                return True
            else:
                logger.error(f"Failed to cancel order {order_id}: {result.comment}")
                return False
                
        except Exception as e:
            logger.error(f"Error cancelling MT5 order {order_id}: {e}")
            return False
    
    async def get_positions(self) -> List[Position]:
        """Get current positions from MT5"""
        try:
            positions = mt5.positions_get()
            if positions is None:
                return []
            
            result = []
            for pos in positions:
                position = Position(
                    symbol=pos.symbol,
                    quantity=pos.volume if pos.type == mt5.POSITION_TYPE_BUY else -pos.volume,
                    avg_price=pos.price_open,
                    market_value=pos.volume * pos.price_current,
                    unrealized_pnl=pos.profit,
                    timestamp=datetime.fromtimestamp(pos.time)
                )
                result.append(position)
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting MT5 positions: {e}")
            return []
    
    async def get_account_info(self) -> Dict:
        """Get account information from MT5"""
        try:
            account = mt5.account_info()
            if account is None:
                return {}
            
            return {
                'balance': account.balance,
                'equity': account.equity,
                'margin': account.margin,
                'free_margin': account.margin_free,
                'margin_level': account.margin_level,
                'total_value': account.equity,
                'buying_power': account.margin_free,
                'pnl': account.profit
            }
            
        except Exception as e:
            logger.error(f"Error getting MT5 account info: {e}")
            return {}
    
    async def subscribe_market_data(self, symbols: List[str], callback):
        """Subscribe to MT5 market data (placeholder - MT5 uses polling)"""
        # MT5 doesn't have real-time streaming, so we'll use the MT5DataFeed class
        logger.info(f"MT5 market data subscription handled by MT5DataFeed")


class MT5DataFeed(DataFeed):
    """MetaTrader 5 data feed implementation"""
    
    def __init__(self, config: DataFeedConfig):
        super().__init__(config)
        self.symbols = config.symbols or []
        self.update_interval = config.update_interval or 1
        self.task = None
        self.last_prices = {}
        
    async def connect(self) -> bool:
        """Connect to MT5 for data feed"""
        try:
            if not mt5.initialize():
                logger.error(f"MT5 data feed initialize failed: {mt5.last_error()}")
                return False
            
            # Ensure all symbols are selected
            for symbol in self.symbols:
                symbol_info = mt5.symbol_info(symbol)
                if symbol_info is None:
                    logger.warning(f"Symbol {symbol} not found")
                    continue
                
                if not symbol_info.visible:
                    if not mt5.symbol_select(symbol, True):
                        logger.warning(f"Failed to select symbol {symbol}")
            
            self.is_connected = True
            logger.info("Connected to MT5 data feed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect MT5 data feed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from MT5 data feed"""
        if self.task:
            self.task.cancel()
        
        if self.is_connected:
            mt5.shutdown()
        
        self.is_connected = False
        logger.info("Disconnected from MT5 data feed")
    
    async def subscribe(self, symbols: List[str]):
        """Subscribe to MT5 symbols"""
        self.symbols.extend([s for s in symbols if s not in self.symbols])
        
        # Start data polling if not already running
        if not self.task and self.is_connected:
            self.task = asyncio.create_task(self._poll_data())
        
        logger.info(f"Subscribed to MT5 data for {symbols}")
    
    async def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from symbols"""
        for symbol in symbols:
            if symbol in self.symbols:
                self.symbols.remove(symbol)
        
        logger.info(f"Unsubscribed from MT5 data for {symbols}")
    
    async def _poll_data(self):
        """Poll MT5 for market data"""
        while self.is_connected:
            try:
                for symbol in self.symbols:
                    # Get current tick
                    tick = mt5.symbol_info_tick(symbol)
                    if tick is None:
                        continue
                    
                    # Create market data
                    market_data = MarketData(
                        symbol=symbol,
                        timestamp=datetime.fromtimestamp(tick.time),
                        bid=tick.bid,
                        ask=tick.ask,
                        last=tick.last,
                        volume=tick.volume_real if hasattr(tick, 'volume_real') else 0.0
                    )
                    
                    # Only notify if price changed
                    last_price = self.last_prices.get(symbol, 0)
                    if tick.last != last_price:
                        self.last_prices[symbol] = tick.last
                        self._notify_callbacks(market_data)
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"Error polling MT5 data: {e}")
                await asyncio.sleep(5)  # Wait before retrying


def create_mt5_config_template() -> Dict:
    """Create MT5 configuration template"""
    return {
        "broker": {
            "type": "mt5",
            "mt5": {
                "server": "YourBroker-Server",
                "login": 12345678,
                "password": "your_password",
                "path": "C:\\Program Files\\MetaTrader 5\\terminal64.exe",  # Optional
                "timeout": 60000  # Optional
            }
        },
        "symbols": ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"],
        "data_feed": {
            "update_interval": 1,
            "reconnect_attempts": 5
        }
    }


if __name__ == "__main__":
    # Example usage
    import json
    
    # Create sample config
    config = create_mt5_config_template()
    print("MT5 Configuration Template:")
    print(json.dumps(config, indent=2))
    
    async def test_mt5():
        # Test MT5 connection (requires actual MT5 setup)
        broker_config = config["broker"]["mt5"]
        broker = MT5Broker(broker_config)
        
        if await broker.connect():
            account_info = await broker.get_account_info()
            print(f"Account Info: {account_info}")
            
            positions = await broker.get_positions()
            print(f"Positions: {len(positions)}")
            
            await broker.disconnect()
        else:
            print("Failed to connect to MT5")
    
    # Uncomment to test (requires MT5 installation and configuration)
    # asyncio.run(test_mt5())
