from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class TradeSignal:
    """
    Standardized trade signal representation
    """
    should_trade: bool = False
    entry_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    trade_type: str = 'NONE'  # BUY/SELL/NONE

@dataclass
class TradeResult:
    """
    Standardized trade result representation
    """
    success: bool = False
    profit: float = 0.0
    fees: float = 0.0
    timestamp: Optional[str] = None

class BaseStrategy:
    """
    Abstract base class for trading strategies
    """
    def __init__(self, name: str = "BaseStrategy"):
        """
        Initialize base strategy
        
        Args:
            name (str): Strategy identifier
        """
        self.name = name
    
    def generate_signal(
        self, 
        sentiment: Dict[str, float], 
        prediction: Dict[str, float], 
        risk_metrics: Dict[str, Any]
    ) -> TradeSignal:
        """
        Generate trading signal based on market conditions
        
        Args:
            sentiment (dict): Market sentiment indicators
            prediction (dict): Market movement predictions
            risk_metrics (dict): Current risk management metrics
        
        Returns:
            TradeSignal: Recommended trading action
        """
        raise NotImplementedError("Subclasses must implement signal generation")
    
    def execute_trade(
        self, 
        signal: TradeSignal, 
        position_size: float
    ) -> TradeResult:
        """
        Execute a trade based on generated signal
        
        Args:
            signal (TradeSignal): Trading signal to execute
            position_size (float): Size of the trading position
        
        Returns:
            TradeResult: Outcome of the trade
        """
        raise NotImplementedError("Subclasses must implement trade execution")
    
    def __str__(self):
        return f"Strategy: {self.name}"
