"""
Live Trading Engine - Stage 5
Real-time trading execution with broker API integration and risk monitoring.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import threading
import queue
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import websockets
import requests
from abc import ABC, abstractmethod


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('live_trading.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Represents a trading order"""
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # Good Till Cancelled
    order_id: Optional[str] = None
    broker_order_id: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    timestamp: Optional[datetime] = None
    fill_timestamp: Optional[datetime] = None


@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    quantity: float
    avg_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float = 0.0
    timestamp: Optional[datetime] = None


@dataclass
class MarketData:
    """Represents real-time market data"""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    volume: float
    open: float = None
    high: float = None
    low: float = None
    close: float = None


class BrokerAPI(ABC):
    """Abstract base class for broker API implementations"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to broker API"""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Disconnect from broker API"""
        pass
    
    @abstractmethod
    async def submit_order(self, order: Order) -> str:
        """Submit order to broker"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get current positions"""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Dict:
        """Get account information"""
        pass
    
    @abstractmethod
    async def subscribe_market_data(self, symbols: List[str], callback: Callable):
        """Subscribe to real-time market data"""
        pass


class SimulatedBroker(BrokerAPI):
    """Simulated broker for testing and development"""
    
    def __init__(self, initial_balance: float = 100000.0):
        self.balance = initial_balance
        self.initial_balance = initial_balance
        self.positions = {}
        self.orders = {}
        self.order_counter = 0
        self.connected = False
        self.market_data_callbacks = []
        
    async def connect(self) -> bool:
        """Connect to simulated broker"""
        self.connected = True
        logger.info("Connected to simulated broker")
        return True
    
    async def disconnect(self):
        """Disconnect from simulated broker"""
        self.connected = False
        logger.info("Disconnected from simulated broker")
    
    async def submit_order(self, order: Order) -> str:
        """Submit order to simulated broker"""
        if not self.connected:
            raise Exception("Not connected to broker")
        
        self.order_counter += 1
        order_id = f"SIM_{self.order_counter:06d}"
        order.order_id = order_id
        order.broker_order_id = order_id
        order.status = OrderStatus.SUBMITTED
        order.timestamp = datetime.now()
        
        self.orders[order_id] = order
        
        # Simulate immediate fill for market orders
        if order.order_type == OrderType.MARKET:
            await self._fill_order(order_id, order.quantity, order.price or 100.0)
        
        logger.info(f"Submitted order {order_id}: {order.side.value} {order.quantity} {order.symbol}")
        return order_id
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        if order_id in self.orders:
            order = self.orders[order_id]
            if order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
                order.status = OrderStatus.CANCELLED
                logger.info(f"Cancelled order {order_id}")
                return True
        return False
    
    async def _fill_order(self, order_id: str, quantity: float, price: float):
        """Simulate order fill"""
        if order_id not in self.orders:
            return
        
        order = self.orders[order_id]
        order.status = OrderStatus.FILLED
        order.filled_quantity = quantity
        order.avg_fill_price = price
        order.fill_timestamp = datetime.now()
        
        # Update positions
        symbol = order.symbol
        if symbol not in self.positions:
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=0.0,
                avg_price=0.0,
                market_value=0.0,
                unrealized_pnl=0.0
            )
        
        position = self.positions[symbol]
        
        if order.side == OrderSide.BUY:
            new_quantity = position.quantity + quantity
            if new_quantity != 0:
                position.avg_price = ((position.quantity * position.avg_price) + (quantity * price)) / new_quantity
            position.quantity = new_quantity
            self.balance -= quantity * price
        else:  # SELL
            position.quantity -= quantity
            self.balance += quantity * price
            if position.quantity == 0:
                position.avg_price = 0.0
        
        position.market_value = position.quantity * price
        position.timestamp = datetime.now()
        
        logger.info(f"Filled order {order_id}: {quantity} @ {price}")
    
    async def get_positions(self) -> List[Position]:
        """Get current positions"""
        return list(self.positions.values())
    
    async def get_account_info(self) -> Dict:
        """Get account information"""
        total_value = self.balance
        for position in self.positions.values():
            total_value += position.market_value
        
        return {
            'balance': self.balance,
            'total_value': total_value,
            'buying_power': self.balance,
            'pnl': total_value - self.initial_balance
        }
    
    async def subscribe_market_data(self, symbols: List[str], callback: Callable):
        """Subscribe to simulated market data"""
        self.market_data_callbacks.append(callback)
        logger.info(f"Subscribed to market data for {symbols}")


class LiveTradingEngine:
    """Main live trading engine"""
    
    def __init__(self, broker: BrokerAPI, config: Dict):
        self.broker = broker
        self.config = config
        self.is_running = False
        self.market_data_queue = queue.Queue()
        self.signal_queue = queue.Queue()
        self.order_queue = queue.Queue()
        
        # Trading state
        self.current_positions = {}
        self.active_orders = {}
        self.market_data = {}
        self.equity_history = []
        
        # Risk management
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.max_daily_loss = config.get('max_daily_loss', 500.0)
        self.max_total_loss = config.get('max_total_loss', 1000.0)
        self.trading_halted = False
        
        # Threading
        self.threads = []
        self.shutdown_event = threading.Event()
    
    async def start(self):
        """Start the live trading engine"""
        logger.info("Starting live trading engine...")
        
        # Connect to broker
        if not await self.broker.connect():
            raise Exception("Failed to connect to broker")
        
        self.is_running = True
        
        # Start worker threads
        self.threads = [
            threading.Thread(target=self._market_data_worker, daemon=True),
            threading.Thread(target=self._signal_processor, daemon=True),
            threading.Thread(target=self._order_manager, daemon=True),
            threading.Thread(target=self._risk_monitor, daemon=True)
        ]
        
        for thread in self.threads:
            thread.start()
        
        logger.info("Live trading engine started")
    
    async def stop(self):
        """Stop the live trading engine"""
        logger.info("Stopping live trading engine...")
        
        self.is_running = False
        self.shutdown_event.set()
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=5.0)
        
        # Disconnect from broker
        await self.broker.disconnect()
        
        logger.info("Live trading engine stopped")
    
    async def process_signal(self, signal: Dict[str, Any]):
        """
        Public async method to process signals from V3.0 signal processors.
        
        Args:
            signal: Signal dictionary from V3.0 processor
        """
        try:
            # Convert V3.0 signal format to internal format
            signal_type = signal.get('signal_type', '')
            
            # Determine direction from signal_type
            if signal_type == 'BUY':
                direction = 'LONG'
            elif signal_type == 'SELL':
                direction = 'SHORT'
            else:
                direction = None
            
            internal_signal = {
                'symbol': signal.get('symbol', 'XAUUSD'),
                'direction': direction,
                'strength': 1.0,  # V3.0 uses binary signals
                'entry_price': signal.get('entry_price', 0.0),
                'stop_loss': signal.get('stop_loss', 0.0),
                'take_profit': signal.get('take_profit', 0.0),
                'signal_type': signal_type,
                'reason': signal.get('reason', '')
            }
            
            if internal_signal['direction']:
                # Add to signal queue for processing
                self.signal_queue.put(internal_signal)
                logger.info(f"V3.0 Signal queued: {internal_signal['direction']} {internal_signal['symbol']} @ {internal_signal['entry_price']}")
                logger.info(f"   SL: {internal_signal['stop_loss']} | TP: {internal_signal['take_profit']}")
            else:
                logger.warning(f"Invalid V3.0 signal format - unknown signal_type: {signal_type}")
                
        except Exception as e:
            logger.error(f"Error processing V3.0 signal: {e}")
    
    def _market_data_worker(self):
        """Process incoming market data"""
        while self.is_running and not self.shutdown_event.is_set():
            try:
                if not self.market_data_queue.empty():
                    market_data = self.market_data_queue.get(timeout=1.0)
                    self.market_data[market_data.symbol] = market_data
                    
                    # Generate signals based on market data
                    self._generate_signals(market_data)
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in market data worker: {e}")
    
    def _signal_processor(self):
        """Process trading signals"""
        while self.is_running and not self.shutdown_event.is_set():
            try:
                if not self.signal_queue.empty():
                    signal = self.signal_queue.get(timeout=1.0)
                    
                    if not self.trading_halted:
                        self._process_signal(signal)
                    else:
                        logger.warning("Trading halted - ignoring signal")
                        
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in signal processor: {e}")
    
    def _order_manager(self):
        """Manage order execution"""
        while self.is_running and not self.shutdown_event.is_set():
            try:
                if not self.order_queue.empty():
                    order = self.order_queue.get(timeout=1.0)
                    asyncio.run(self._execute_order(order))
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in order manager: {e}")
    
    def _risk_monitor(self):
        """Monitor risk limits"""
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Update P&L
                account_info = asyncio.run(self.broker.get_account_info())
                current_pnl = account_info.get('pnl', 0.0)
                
                # Check risk limits
                if abs(current_pnl) > self.max_total_loss:
                    logger.critical(f"Total loss limit exceeded: {current_pnl}")
                    self.trading_halted = True
                
                # Update equity history
                self.equity_history.append({
                    'timestamp': datetime.now(),
                    'equity': account_info.get('total_value', 0.0),
                    'pnl': current_pnl
                })
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in risk monitor: {e}")
    
    def _generate_signals(self, market_data: MarketData):
        """Generate trading signals from market data"""
        # This would integrate with the existing ronin_engine signal generation
        # For now, placeholder logic
        pass
    
    def _process_signal(self, signal: Dict):
        """Process a trading signal"""
        try:
            symbol = signal.get('symbol')
            direction = signal.get('direction')  # 'LONG' or 'SHORT'
            strength = signal.get('strength', 1.0)
            
            if direction == 'LONG':
                order = Order(
                    symbol=symbol,
                    side=OrderSide.BUY,
                    quantity=self._calculate_position_size(symbol, strength),
                    order_type=OrderType.MARKET
                )
            elif direction == 'SHORT':
                order = Order(
                    symbol=symbol,
                    side=OrderSide.SELL,
                    quantity=self._calculate_position_size(symbol, strength),
                    order_type=OrderType.MARKET
                )
            else:
                return
            
            self.order_queue.put(order)
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
    
    def _calculate_position_size(self, symbol: str, strength: float) -> float:
        """Calculate position size based on risk management for MT5"""
        try:
            # Get risk per trade from config
            risk_per_trade = self.config.get('trading', {}).get('risk_per_trade', 0.001)
            
            # Get current account balance (assume $200,000 for demo)
            account_balance = 200000.0
            
            # Calculate risk amount in dollars
            risk_amount = account_balance * risk_per_trade
            
            # For XAUUSD, use minimum lot size with proper risk management
            if symbol == 'XAUUSD':
                # XAUUSD: 1 lot = 100 oz, minimum = 0.01 lot
                # With current price ~$3,552, 0.01 lot = $35.52 per pip
                # Use very small position size for scalping
                base_lot_size = 0.01  # Minimum lot size
                
                # Scale based on risk amount (max 0.1 lots for safety)
                max_lots = min(0.1, risk_amount / 1000)  # Conservative scaling
                position_size = base_lot_size * strength
                position_size = min(position_size, max_lots)
                
                return round(position_size, 2)  # Round to 2 decimals for MT5
            else:
                # For other symbols, use conservative sizing
                return 0.01 * strength
                
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.01  # Safe fallback
    
    async def _execute_order(self, order: Order):
        """Execute an order through the broker"""
        try:
            order_id = await self.broker.submit_order(order)
            self.active_orders[order_id] = order
            logger.info(f"Executed order: {order_id}")
            
        except Exception as e:
            logger.error(f"Failed to execute order: {e}")
    
    def get_status(self) -> Dict:
        """Get current trading status"""
        return {
            'is_running': self.is_running,
            'trading_halted': self.trading_halted,
            'active_orders': len(self.active_orders),
            'positions': len(self.current_positions),
            'daily_pnl': self.daily_pnl,
            'total_pnl': self.total_pnl
        }


if __name__ == "__main__":
    # Example usage
    config = {
        'max_daily_loss': 500.0,
        'max_total_loss': 1000.0,
        'symbols': ['NVDA', 'TSLA', 'AAPL']
    }
    
    # Use simulated broker for testing
    broker = SimulatedBroker(initial_balance=100000.0)
    engine = LiveTradingEngine(broker, config)
    
    async def main():
        await engine.start()
        
        # Run for a short time for demonstration
        await asyncio.sleep(5)
        
        status = engine.get_status()
        print(f"Trading Status: {status}")
        
        await engine.stop()
    
    asyncio.run(main())
