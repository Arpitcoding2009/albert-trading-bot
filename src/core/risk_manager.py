import numpy as np
from typing import Dict, Optional
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class RiskParameters:
    position_size: float
    stop_loss: float
    take_profit: float
    max_drawdown: float
    risk_reward_ratio: float
    confidence_threshold: float

class RiskManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Default risk parameters
        self.max_position_size = 0.1  # 10% of portfolio
        self.base_stop_loss = 0.02    # 2% stop loss
        self.base_take_profit = 0.04  # 4% take profit
        self.min_confidence = 0.7     # Minimum confidence threshold
        
        # Risk tracking
        self.current_drawdown = 0.0
        self.max_daily_loss = 0.0
        self.daily_trades = 0
        self.max_daily_trades = 10
        
        # Performance tracking
        self.trade_history = []
        self.daily_pnl = 0.0
        self.last_reset = datetime.now()

    def calculate_risk_parameters(self, signal_confidence: float, 
                                market_volatility: float) -> Dict:
        """
        Calculate risk parameters based on signal confidence and market conditions
        Returns position size, stop loss, and take profit levels
        """
        try:
            # Reset daily metrics if needed
            self._check_daily_reset()
            
            # Base calculations
            position_size = self._calculate_position_size(signal_confidence, market_volatility)
            stop_loss = self._calculate_stop_loss(market_volatility)
            take_profit = self._calculate_take_profit(stop_loss, signal_confidence)
            
            # Adjust for market conditions
            risk_params = self._adjust_for_market_conditions(
                position_size, stop_loss, take_profit,
                signal_confidence, market_volatility
            )
            
            # Validate parameters
            self._validate_risk_parameters(risk_params)
            
            return {
                'position_size': risk_params.position_size,
                'stop_loss': risk_params.stop_loss,
                'take_profit': risk_params.take_profit,
                'confidence_threshold': risk_params.confidence_threshold
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating risk parameters: {str(e)}")
            return self._get_default_parameters()

    def _calculate_position_size(self, confidence: float, volatility: float) -> float:
        """
        Calculate optimal position size based on confidence and volatility
        """
        try:
            # Base position size from confidence
            base_size = self.max_position_size * confidence
            
            # Adjust for volatility
            volatility_factor = 1 - (volatility / 2)  # Reduce size in high volatility
            
            # Adjust for daily performance
            performance_factor = 1 - (abs(self.daily_pnl) / 0.2)  # Reduce size after large gains/losses
            
            # Adjust for number of trades
            trade_factor = 1 - (self.daily_trades / self.max_daily_trades)
            
            position_size = base_size * volatility_factor * performance_factor * trade_factor
            
            # Ensure within limits
            return min(max(position_size, 0.01), self.max_position_size)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {str(e)}")
            return self.max_position_size * 0.5

    def _calculate_stop_loss(self, volatility: float) -> float:
        """
        Calculate dynamic stop loss based on market volatility
        """
        try:
            # Adjust base stop loss for volatility
            volatility_multiplier = 1 + (volatility * 2)  # Wider stops in high volatility
            stop_loss = self.base_stop_loss * volatility_multiplier
            
            # Ensure minimum stop loss
            return max(stop_loss, self.base_stop_loss)
            
        except Exception as e:
            self.logger.error(f"Error calculating stop loss: {str(e)}")
            return self.base_stop_loss

    def _calculate_take_profit(self, stop_loss: float, confidence: float) -> float:
        """
        Calculate take profit based on stop loss and confidence
        """
        try:
            # Base risk-reward ratio from confidence
            risk_reward_ratio = 1.5 + (confidence * 1.5)  # 1.5-3.0 range
            
            take_profit = stop_loss * risk_reward_ratio
            
            # Ensure minimum take profit
            return max(take_profit, self.base_take_profit)
            
        except Exception as e:
            self.logger.error(f"Error calculating take profit: {str(e)}")
            return self.base_take_profit

    def _adjust_for_market_conditions(self, position_size: float, 
                                   stop_loss: float, 
                                   take_profit: float,
                                   confidence: float,
                                   volatility: float) -> RiskParameters:
        """
        Adjust risk parameters based on current market conditions
        """
        try:
            # Market condition factors
            market_stress = self._calculate_market_stress(volatility)
            trend_strength = self._calculate_trend_strength()
            liquidity_factor = self._calculate_liquidity_factor()
            
            # Adjust parameters
            adjusted_size = position_size * (1 - market_stress)
            adjusted_stop = stop_loss * (1 + market_stress)
            adjusted_profit = take_profit * (1 + (trend_strength * 0.5))
            
            # Calculate risk-reward ratio
            risk_reward_ratio = adjusted_profit / adjusted_stop
            
            # Calculate max drawdown
            max_drawdown = self._calculate_max_drawdown(adjusted_size)
            
            # Adjust confidence threshold
            confidence_threshold = self.min_confidence * (1 + market_stress)
            
            return RiskParameters(
                position_size=adjusted_size,
                stop_loss=adjusted_stop,
                take_profit=adjusted_profit,
                max_drawdown=max_drawdown,
                risk_reward_ratio=risk_reward_ratio,
                confidence_threshold=confidence_threshold
            )
            
        except Exception as e:
            self.logger.error(f"Error adjusting for market conditions: {str(e)}")
            return self._get_default_risk_parameters()

    def update_trade_metrics(self, trade_result: Dict):
        """
        Update risk metrics after a trade
        """
        try:
            # Update daily metrics
            self.daily_trades += 1
            self.daily_pnl += trade_result['pnl']
            
            # Update drawdown
            if trade_result['pnl'] < 0:
                self.current_drawdown += abs(trade_result['pnl'])
                self.max_daily_loss = min(self.max_daily_loss, trade_result['pnl'])
            else:
                self.current_drawdown = 0
            
            # Add to trade history
            self.trade_history.append({
                'timestamp': datetime.now(),
                'pnl': trade_result['pnl'],
                'position_size': trade_result['position_size'],
                'success': trade_result['pnl'] > 0
            })
            
            # Trim trade history
            if len(self.trade_history) > 1000:
                self.trade_history = self.trade_history[-1000:]
                
        except Exception as e:
            self.logger.error(f"Error updating trade metrics: {str(e)}")

    def _check_daily_reset(self):
        """Reset daily metrics if new day"""
        current_time = datetime.now()
        if current_time.date() > self.last_reset.date():
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.max_daily_loss = 0.0
            self.last_reset = current_time

    def _calculate_market_stress(self, volatility: float) -> float:
        """Calculate market stress level"""
        try:
            # Combine various stress indicators
            volatility_stress = min(volatility / 0.5, 1.0)  # Cap at 1.0
            drawdown_stress = min(self.current_drawdown / 0.1, 1.0)
            loss_stress = min(abs(self.max_daily_loss) / 0.1, 1.0)
            
            # Weighted average of stress factors
            return (volatility_stress * 0.4 + 
                   drawdown_stress * 0.3 + 
                   loss_stress * 0.3)
            
        except Exception as e:
            self.logger.error(f"Error calculating market stress: {str(e)}")
            return 0.5

    def _calculate_trend_strength(self) -> float:
        """Calculate current trend strength"""
        # Implementation details...
        return 0.7  # Example return

    def _calculate_liquidity_factor(self) -> float:
        """Calculate market liquidity factor"""
        # Implementation details...
        return 0.8  # Example return

    def _calculate_max_drawdown(self, position_size: float) -> float:
        """Calculate maximum allowable drawdown"""
        # Implementation details...
        return 0.1  # Example return

    def _validate_risk_parameters(self, params: RiskParameters):
        """Validate risk parameters are within acceptable ranges"""
        try:
            assert 0 < params.position_size <= self.max_position_size
            assert 0 < params.stop_loss <= 0.1  # Max 10% stop loss
            assert params.take_profit >= params.stop_loss
            assert params.risk_reward_ratio >= 1.0
            assert 0 < params.confidence_threshold <= 1.0
        except AssertionError as e:
            self.logger.error(f"Invalid risk parameters: {str(e)}")
            raise

    def _get_default_parameters(self) -> Dict:
        """Return default risk parameters"""
        return {
            'position_size': self.max_position_size * 0.5,
            'stop_loss': self.base_stop_loss,
            'take_profit': self.base_take_profit,
            'confidence_threshold': self.min_confidence
        }

    def _get_default_risk_parameters(self) -> RiskParameters:
        """Return default RiskParameters object"""
        return RiskParameters(
            position_size=self.max_position_size * 0.5,
            stop_loss=self.base_stop_loss,
            take_profit=self.base_take_profit,
            max_drawdown=0.1,
            risk_reward_ratio=2.0,
            confidence_threshold=self.min_confidence
        )

class AdvancedRiskManager:
    def __init__(self, 
                 initial_capital: float = 1000, 
                 max_risk_percentage: float = 0.02,
                 daily_profit_target: float = 0.15):
        """
        Advanced Risk Management System
        
        Args:
            initial_capital (float): Starting trading capital
            max_risk_percentage (float): Maximum risk per trade (default 2%)
            daily_profit_target (float): Daily profit goal (default 15%)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_risk_percentage = max_risk_percentage
        self.daily_profit_target = daily_profit_target
        
        # Risk tracking metrics
        self.total_trades = 0
        self.profitable_trades = 0
        self.total_profit = 0
        self.max_drawdown = 0
        
        # Advanced risk parameters
        self.risk_levels = {
            'conservative': 0.01,   # 1% risk
            'balanced': 0.02,       # 2% risk
            'aggressive': 0.05      # 5% risk
        }
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> float:
        """
        Calculate optimal position size based on risk tolerance
        
        Args:
            entry_price (float): Price at which trade is entered
            stop_loss (float): Stop loss price
        
        Returns:
            float: Recommended position size
        """
        risk_amount = self.current_capital * self.max_risk_percentage
        price_difference = abs(entry_price - stop_loss)
        
        if price_difference == 0:
            return 0
        
        position_size = risk_amount / price_difference
        return min(position_size, self.current_capital * 0.1)  # Cap at 10% of capital
    
    def update_trade_metrics(self, trade_profit: float):
        """
        Update trading metrics after each trade
        
        Args:
            trade_profit (float): Profit/loss from the trade
        """
        self.total_trades += 1
        self.total_profit += trade_profit
        
        if trade_profit > 0:
            self.profitable_trades += 1
        
        # Track max drawdown
        self.current_capital += trade_profit
        self.max_drawdown = min(self.max_drawdown, self.current_capital - self.initial_capital)
    
    def should_continue_trading(self) -> bool:
        """
        Determine if trading should continue based on daily profit target
        
        Returns:
            bool: Whether to continue trading
        """
        profit_percentage = (self.total_profit / self.initial_capital) * 100
        return profit_percentage < (self.daily_profit_target * 100)
    
    def get_risk_metrics(self) -> dict:
        """
        Generate comprehensive risk metrics
        
        Returns:
            dict: Detailed risk and performance metrics
        """
        return {
            'total_trades': self.total_trades,
            'profitable_trades': self.profitable_trades,
            'win_rate': self.profitable_trades / self.total_trades if self.total_trades > 0 else 0,
            'total_profit': self.total_profit,
            'current_capital': self.current_capital,
            'max_drawdown': self.max_drawdown,
            'profit_percentage': (self.total_profit / self.initial_capital) * 100
        }
    
    def kill_switch(self):
        """
        Emergency stop mechanism to halt all trading
        """
        self.max_risk_percentage = 0
        self.daily_profit_target = 0
        return {
            'status': 'TRADING_HALTED',
            'message': 'Emergency kill switch activated. All trading suspended.'
        }
