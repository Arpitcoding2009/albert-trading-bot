from typing import Dict, List, Optional
import numpy as np
from datetime import datetime, timedelta
import logging
from ..utils.config import Settings
from .strategy import Signal

class RiskManager:
    def __init__(self):
        self.settings = Settings()
        self.logger = logging.getLogger(__name__)
        
        # Default risk parameters
        self.max_position_size = 0.1  # 10% of portfolio
        self.stop_loss_pct = 0.02     # 2% stop loss
        self.take_profit_pct = 0.04   # 4% take profit
        self.max_daily_trades = 10
        self.min_confidence = 0.7
        self.max_drawdown = 0.15      # 15% maximum drawdown
        
        # Risk metrics
        self.daily_trades = []
        self.current_drawdown = 0
        self.volatility_multiplier = 1.0

    def configure(self, config: Dict):
        """Configure risk parameters"""
        self.max_position_size = config.get('max_position_size', 0.1)
        self.stop_loss_pct = config.get('stop_loss_percentage', 0.02)
        self.take_profit_pct = config.get('take_profit_percentage', 0.04)
        self.max_daily_trades = config.get('max_daily_trades', 10)
        self.min_confidence = config.get('min_confidence', 0.7)
        self.max_drawdown = config.get('max_drawdown', 0.15)

    def filter_signals(self, signals: List[Signal], portfolio_status: Dict) -> List[Signal]:
        """Filter trading signals based on risk parameters"""
        filtered_signals = []
        
        # Update daily trades list
        self._update_daily_trades()
        
        # Check if maximum daily trades reached
        if len(self.daily_trades) >= self.max_daily_trades:
            self.logger.info("Maximum daily trades reached, no new trades allowed")
            return []
        
        # Check current drawdown
        if self.current_drawdown >= self.max_drawdown:
            self.logger.warning(f"Maximum drawdown reached: {self.current_drawdown:.2%}")
            return []
        
        for signal in signals:
            # Check signal confidence
            if signal.confidence < self.min_confidence:
                continue
                
            # Adjust signal based on volatility
            adjusted_signal = self._adjust_for_volatility(signal)
            
            # Validate position size
            if self.validate_position_size(adjusted_signal, portfolio_status):
                filtered_signals.append(adjusted_signal)
                
        return filtered_signals

    def validate_signal(self, signal: Signal) -> bool:
        """Validate individual trading signal"""
        # Check confidence threshold
        if signal.confidence < self.min_confidence:
            self.logger.info(f"Signal rejected: confidence {signal.confidence} below threshold {self.min_confidence}")
            return False
            
        # Check if signal is too old
        signal_age = datetime.now() - signal.timestamp
        if signal_age > timedelta(minutes=5):
            self.logger.info(f"Signal rejected: age {signal_age.seconds}s exceeds maximum")
            return False
            
        return True

    def calculate_position_size(self, signal: Signal, portfolio_value: float) -> float:
        """Calculate position size based on risk parameters"""
        # Base position size
        base_size = portfolio_value * self.max_position_size
        
        # Adjust based on confidence
        confidence_multiplier = signal.confidence
        
        # Adjust based on volatility
        volatility_adjusted_size = base_size * confidence_multiplier * self.volatility_multiplier
        
        # Ensure minimum position size
        min_position = portfolio_value * 0.01  # 1% minimum
        position_size = max(volatility_adjusted_size, min_position)
        
        # Ensure maximum position size
        max_position = portfolio_value * self.max_position_size
        position_size = min(position_size, max_position)
        
        return position_size

    def validate_position_size(self, signal: Signal, portfolio_status: Dict) -> bool:
        """Validate if position size meets risk requirements"""
        total_position_value = 0
        
        # Calculate total position value
        for position in portfolio_status.get('positions', {}).values():
            total_position_value += position.amount * position.entry_price
            
        portfolio_value = portfolio_status.get('total_value', 0)
        
        # Check if new position would exceed maximum allocation
        if total_position_value / portfolio_value >= self.max_position_size:
            self.logger.info("Position size would exceed maximum allocation")
            return False
            
        return True

    def _update_daily_trades(self):
        """Update daily trades list"""
        current_time = datetime.now()
        
        # Remove trades older than 24 hours
        self.daily_trades = [
            trade for trade in self.daily_trades 
            if (current_time - trade) <= timedelta(hours=24)
        ]

    def _adjust_for_volatility(self, signal: Signal) -> Signal:
        """Adjust signal parameters based on market volatility"""
        # Adjust stop loss and take profit based on volatility
        adjusted_stop_loss = self.stop_loss_pct * self.volatility_multiplier
        adjusted_take_profit = self.take_profit_pct * self.volatility_multiplier
        
        # Create new signal with adjusted parameters
        adjusted_signal = Signal(
            type=signal.type,
            side=signal.side,
            price=signal.price,
            confidence=signal.confidence,
            timestamp=signal.timestamp,
            strategy=signal.strategy,
            metadata={
                **signal.metadata,
                'adjusted_stop_loss': adjusted_stop_loss,
                'adjusted_take_profit': adjusted_take_profit,
                'volatility_multiplier': self.volatility_multiplier
            }
        )
        
        return adjusted_signal

    def update_volatility_multiplier(self, market_data: Dict):
        """Update volatility multiplier based on market conditions"""
        try:
            # Calculate historical volatility
            returns = np.diff(np.log(market_data['close']))
            volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
            
            # Adjust multiplier based on volatility
            base_volatility = 0.5  # Base volatility threshold
            self.volatility_multiplier = base_volatility / volatility if volatility > 0 else 1.0
            
            # Ensure multiplier stays within reasonable bounds
            self.volatility_multiplier = max(0.5, min(2.0, self.volatility_multiplier))
            
        except Exception as e:
            self.logger.error(f"Error updating volatility multiplier: {str(e)}")
            self.volatility_multiplier = 1.0

    def update_drawdown(self, portfolio_value: float, initial_value: float):
        """Update current drawdown"""
        self.current_drawdown = (initial_value - portfolio_value) / initial_value

    def get_current_metrics(self) -> Dict:
        """Get current risk metrics"""
        return {
            'daily_trades': len(self.daily_trades),
            'current_drawdown': self.current_drawdown,
            'volatility_multiplier': self.volatility_multiplier,
            'max_position_size': self.max_position_size,
            'stop_loss_percentage': self.stop_loss_pct,
            'take_profit_percentage': self.take_profit_pct
        }

# Initialize risk manager
risk_manager = RiskManager()
