"""
Risk Manager - FTMO Guards and Position Sizing
Handles risk management, FTMO compliance, and position sizing logic.
"""

import pandas as pd
from typing import Tuple, Dict, Any
from datetime import datetime


class FtmoGuard:
    """
    FTMO-style risk management with daily and total loss limits.
    Tracks equity, active days, and enforces trading halts.
    """
    
    def __init__(self, start_equity: float, cfg: Dict[str, Any]):
        """
        Initialize FTMO guard with starting equity and configuration.
        
        Args:
            start_equity: Starting account equity
            cfg: Configuration dictionary with FTMO parameters
        """
        self.start_equity = start_equity
        self.cfg = cfg
        self.day_start_equity = start_equity
        self.current_date = None
        self.active_days = 0
        self.is_halted = False
        self.halt_reason = ""
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        
    def pre_trade_ok(self, equity: float, date_utc: pd.Timestamp) -> Tuple[bool, str]:
        """
        Check if trading is allowed before placing a new trade.
        
        Args:
            equity: Current account equity
            date_utc: Current UTC timestamp
            
        Returns:
            Tuple of (is_allowed, reason_if_not)
        """
        # Check if already halted
        if self.is_halted:
            return False, self.halt_reason
        
        # Update daily tracking
        self._update_daily_tracking(equity, date_utc)
        
        # Check daily loss limit
        if self.daily_pnl < -self.cfg['max_daily_loss']:
            self.is_halted = True
            self.halt_reason = f"Daily loss limit exceeded: ${self.daily_pnl:.2f}"
            return False, self.halt_reason
        
        # Check total loss limit
        if self.total_pnl < -self.cfg['max_total_loss']:
            self.is_halted = True
            self.halt_reason = f"Total loss limit exceeded: ${self.total_pnl:.2f}"
            return False, self.halt_reason
        
        # Check if profit target reached (optional halt)
        if self.total_pnl >= self.cfg['profit_target']:
            # Don't halt, but could implement conservative mode
            pass
        
        # Check minimum active days (for challenges)
        min_days = self.cfg.get('min_active_days', 0)
        if min_days > 0 and self.active_days < min_days:
            # Still need to trade more days - allow trading
            pass
        
        return True, ""
        
    def post_fill_update(self, equity: float, date_utc: pd.Timestamp) -> None:
        """
        Update internal state after a trade fill.
        
        Args:
            equity: Updated account equity after fill
            date_utc: UTC timestamp of fill
        """
        self._update_daily_tracking(equity, date_utc)
        
    def _update_daily_tracking(self, equity: float, date_utc: pd.Timestamp) -> None:
        """Update daily P&L tracking and active days"""
        current_date = date_utc.date()
        
        # New trading day
        if self.current_date != current_date:
            if self.current_date is not None:
                # Previous day had activity, count as active
                if self.daily_pnl != 0:
                    self.active_days += 1
            
            self.current_date = current_date
            self.day_start_equity = equity
            self.daily_pnl = 0.0
        
        # Update P&L
        self.daily_pnl = equity - self.day_start_equity
        self.total_pnl = equity - self.start_equity


def check_ftmo_limits(equity: float, daily_pnl: float, total_pnl: float, 
                     start_equity: float, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check FTMO compliance limits.
    
    Args:
        equity: Current account equity
        daily_pnl: Current day P&L
        total_pnl: Total P&L since start
        start_equity: Starting account equity
        cfg: Configuration dictionary
        
    Returns:
        Dictionary with compliance status
    """
    can_trade = True
    reason = ""
    warnings = []
    
    # Daily loss limit
    max_daily_loss = cfg['max_daily_loss']
    if daily_pnl <= -max_daily_loss:
        can_trade = False
        reason = f"Daily loss limit hit: ${daily_pnl:.2f} <= -${max_daily_loss:.2f}"
    elif daily_pnl <= -max_daily_loss * 0.8:
        warnings.append(f"Daily loss warning: ${daily_pnl:.2f} (80% of limit)")
    
    # Total loss limit
    max_total_loss = cfg['max_total_loss']
    if total_pnl <= -max_total_loss:
        can_trade = False
        reason = f"Total loss limit hit: ${total_pnl:.2f} <= -${max_total_loss:.2f}"
    elif total_pnl <= -max_total_loss * 0.8:
        warnings.append(f"Total loss warning: ${total_pnl:.2f} (80% of limit)")
    
    # Profit target (informational)
    profit_target = cfg['profit_target']
    if total_pnl >= profit_target:
        warnings.append(f"Profit target reached: ${total_pnl:.2f} >= ${profit_target:.2f}")
    
    return {
        'can_trade': can_trade,
        'reason': reason,
        'warnings': warnings,
        'daily_pnl': daily_pnl,
        'total_pnl': total_pnl,
        'daily_limit_used_pct': abs(daily_pnl) / max_daily_loss * 100,
        'total_limit_used_pct': abs(total_pnl) / max_total_loss * 100
    }


def update_daily_pnl(current_pnl: float, trade_pnl: float) -> float:
    """
    Update daily P&L with new trade result.
    
    Args:
        current_pnl: Current daily P&L
        trade_pnl: P&L from completed trade
        
    Returns:
        Updated daily P&L
    """
    return current_pnl + trade_pnl


def calculate_position_size(equity: float, intraday_pl: float, risk_distance: float, 
                          cfg: Dict[str, Any]) -> int:
    """
    Calculate position size based on risk percentage and current equity state.
    
    Args:
        equity: Current account equity
        intraday_pl: Intraday realized P&L
        risk_distance: Distance from entry to stop loss
        cfg: Configuration dictionary
        
    Returns:
        Position size (number of shares/units)
    """
    if risk_distance <= 0:
        return 0
    
    # Determine risk percentage based on current state
    base_risk_pct = cfg['risk_base_pct']
    red_risk_pct = cfg['risk_red_pct']
    up_risk_pct = cfg['risk_up_pct']
    
    if intraday_pl < 0:
        # Reduce risk when losing
        risk_pct = red_risk_pct
    elif intraday_pl > 500:  # After $500 profit
        # Increase risk when winning
        risk_pct = up_risk_pct
    else:
        # Base risk
        risk_pct = base_risk_pct
    
    # Calculate risk amount
    risk_amount = equity * risk_pct
    
    # Calculate position size
    position_size = int(risk_amount / risk_distance)
    
    return max(0, position_size)


def validate_trade_size(position_size: int, equity: float, cfg: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate that trade size is within acceptable limits.
    
    Args:
        position_size: Proposed position size
        equity: Current account equity
        cfg: Configuration dictionary
        
    Returns:
        Tuple of (is_valid, reason_if_not)
    """
    if position_size <= 0:
        return False, "Position size must be positive"
    
    # Check maximum position size (e.g., no more than 10% of equity per trade)
    max_position_value = equity * 0.1  # 10% max
    
    # This is a simplified check - in real implementation would need price
    # For now, assume reasonable position sizes
    if position_size > 10000:  # Arbitrary large number check
        return False, f"Position size too large: {position_size}"
    
    return True, ""


def calculate_risk_metrics(trades: list, start_equity: float) -> Dict[str, float]:
    """
    Calculate risk-related metrics from trade history.
    
    Args:
        trades: List of completed trades
        start_equity: Starting account equity
        
    Returns:
        Dictionary with risk metrics
    """
    if not trades:
        return {
            'max_risk_per_trade': 0.0,
            'avg_risk_per_trade': 0.0,
            'risk_adjusted_return': 0.0,
            'consecutive_losses': 0,
            'max_consecutive_losses': 0
        }
    
    # Calculate risk per trade
    risk_amounts = [abs(trade.get('risk_amount', 0)) for trade in trades]
    max_risk = max(risk_amounts) if risk_amounts else 0.0
    avg_risk = sum(risk_amounts) / len(risk_amounts) if risk_amounts else 0.0
    
    # Calculate consecutive losses
    consecutive_losses = 0
    max_consecutive_losses = 0
    
    for trade in trades:
        if trade.get('pnl', 0) < 0:
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        else:
            consecutive_losses = 0
    
    # Risk-adjusted return (simplified)
    total_pnl = sum(trade.get('pnl', 0) for trade in trades)
    risk_adjusted_return = total_pnl / avg_risk if avg_risk > 0 else 0.0
    
    return {
        'max_risk_per_trade': max_risk,
        'avg_risk_per_trade': avg_risk,
        'risk_adjusted_return': risk_adjusted_return,
        'consecutive_losses': consecutive_losses,
        'max_consecutive_losses': max_consecutive_losses
    }


if __name__ == "__main__":
    from config import get_config
    
    cfg = get_config()
    guard = FtmoGuard(10000.0, cfg)
    
    print("✅ Risk Manager module loaded successfully")
    print(f"FTMO guard initialized with ${guard.start_equity:,.2f} starting equity")
    print(f"Max daily loss: ${cfg['max_daily_loss']:,.2f}")
    print(f"Max total loss: ${cfg['max_total_loss']:,.2f}")
    print(f"Profit target: ${cfg['profit_target']:,.2f}")
    
    # Test FTMO limits
    test_check = check_ftmo_limits(10000, 0, 0, 10000, cfg)
    print(f"Initial FTMO check: {test_check['can_trade']}")
    
    # Test position sizing
    test_size = calculate_position_size(10000, 0, 1.0, cfg)
    print(f"Test position size: {test_size}")
