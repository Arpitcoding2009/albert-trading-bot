import os
import sys
import logging
from typing import Dict, Any, Optional

# Core Trading Components
from src.core.risk_manager import AdvancedRiskManager
from src.sentiment_analyzer import MarketSentimentAnalyzer
from src.strategies.base_strategy import BaseStrategy
from src.ml.predictor import MarketPredictor

class TradingEngine:
    def __init__(
        self, 
        initial_capital: float = 10000, 
        exchange_config: Dict[str, Any] = None
    ):
        """
        Comprehensive Trading Engine
        
        Args:
            initial_capital (float): Starting trading capital
            exchange_config (dict): Exchange-specific configuration
        """
        # Core Components
        self.risk_manager = AdvancedRiskManager(initial_capital)
        self.sentiment_analyzer = MarketSentimentAnalyzer()
        self.market_predictor = MarketPredictor()
        
        # Trading Configuration
        self.exchange_config = exchange_config or {}
        self.active_strategies = []
        
        # Logging Setup
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
    
    def add_strategy(self, strategy: BaseStrategy):
        """
        Add a trading strategy to the engine
        
        Args:
            strategy (BaseStrategy): Trading strategy to add
        """
        self.active_strategies.append(strategy)
        self.logger.info(f"Added strategy: {strategy.__class__.__name__}")
    
    def execute_trading_cycle(self):
        """
        Execute a complete trading cycle
        """
        try:
            # Market Analysis
            market_sentiment = self.sentiment_analyzer.get_current_sentiment()
            market_prediction = self.market_predictor.predict_market_movement()
            
            # Risk Assessment
            risk_metrics = self.risk_manager.get_risk_metrics()
            
            # Strategy Execution
            for strategy in self.active_strategies:
                trade_signal = strategy.generate_signal(
                    sentiment=market_sentiment,
                    prediction=market_prediction,
                    risk_metrics=risk_metrics
                )
                
                if trade_signal.should_trade:
                    position_size = self.risk_manager.calculate_position_size(
                        trade_signal.entry_price, 
                        trade_signal.stop_loss
                    )
                    
                    # Execute Trade
                    trade_result = strategy.execute_trade(
                        signal=trade_signal, 
                        position_size=position_size
                    )
                    
                    # Update Risk Metrics
                    self.risk_manager.update_trade_metrics(trade_result.profit)
            
            # Check Trading Continuation
            if not self.risk_manager.should_continue_trading():
                self.logger.warning("Daily profit target reached. Halting trading.")
                return self.risk_manager.kill_switch()
        
        except Exception as e:
            self.logger.error(f"Trading cycle error: {e}")
            return {
                'status': 'ERROR',
                'message': str(e)
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Comprehensive system status report
        
        Returns:
            dict: Detailed system status
        """
        return {
            'risk_metrics': self.risk_manager.get_risk_metrics(),
            'market_sentiment': self.sentiment_analyzer.get_current_sentiment(),
            'active_strategies': [
                strategy.__class__.__name__ for strategy in self.active_strategies
            ],
            'system_health': 'OPERATIONAL'
        }

# Singleton Trading Engine Instance
trading_engine = TradingEngine()
