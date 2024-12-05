from typing import Dict, List, Optional
import numpy as np
from pydantic import BaseModel
import pandas as pd
from dataclasses import dataclass
import logging

@dataclass
class RiskMetrics:
    var_95: float  # Value at Risk (95% confidence)
    var_99: float  # Value at Risk (99% confidence)
    expected_shortfall: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    beta: float
    alpha: float
    correlation_matrix: np.ndarray

class AdvancedRiskManager:
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        self.max_position_size = config.get('max_position_size', 0.2)
        self.max_portfolio_var = config.get('max_portfolio_var', 0.1)
        self.min_sharpe_ratio = config.get('min_sharpe_ratio', 1.5)

    def calculate_var(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """Calculate Value at Risk using historical simulation"""
        return np.percentile(returns, (1 - confidence) * 100)

    def calculate_expected_shortfall(self, returns: np.ndarray, var: float) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        return np.mean(returns[returns <= var])

    def calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate annualized Sharpe Ratio"""
        excess_returns = returns - self.risk_free_rate / 252  # Daily adjustment
        return np.sqrt(252) * np.mean(excess_returns) / np.std(returns)

    def calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino Ratio using downside deviation"""
        excess_returns = returns - self.risk_free_rate / 252
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else np.std(returns)
        return np.sqrt(252) * np.mean(excess_returns) / downside_std

    def calculate_max_drawdown(self, prices: np.ndarray) -> float:
        """Calculate Maximum Drawdown"""
        peak = np.maximum.accumulate(prices)
        drawdown = (prices - peak) / peak
        return np.min(drawdown)

    def calculate_beta(self, returns: np.ndarray, market_returns: np.ndarray) -> float:
        """Calculate portfolio beta"""
        covariance = np.cov(returns, market_returns)[0][1]
        market_variance = np.var(market_returns)
        return covariance / market_variance

    def calculate_alpha(self, returns: np.ndarray, market_returns: np.ndarray, beta: float) -> float:
        """Calculate portfolio alpha"""
        return np.mean(returns) - (self.risk_free_rate / 252 + beta * (np.mean(market_returns) - self.risk_free_rate / 252))

    def get_position_size(self, current_price: float, volatility: float, available_capital: float) -> float:
        """Calculate optimal position size using Kelly Criterion with safety factor"""
        kelly_fraction = 0.5  # Half-Kelly for safety
        win_rate = 0.55  # Estimated from historical data
        risk_reward = 2.0  # Target risk-reward ratio
        
        kelly_size = kelly_fraction * ((win_rate * risk_reward - (1 - win_rate)) / risk_reward)
        position_size = min(kelly_size * available_capital, self.max_position_size * available_capital)
        
        # Adjust for volatility
        volatility_factor = np.exp(-volatility)  # Reduce size for higher volatility
        return position_size * volatility_factor

    def assess_portfolio_risk(self, portfolio: Dict[str, float], historical_data: pd.DataFrame) -> RiskMetrics:
        """Comprehensive portfolio risk assessment"""
        try:
            returns = historical_data.pct_change().dropna()
            portfolio_returns = returns.dot(pd.Series(portfolio))
            market_returns = historical_data.mean(axis=1)  # Simple market proxy

            var_95 = self.calculate_var(portfolio_returns.values, 0.95)
            var_99 = self.calculate_var(portfolio_returns.values, 0.99)
            es = self.calculate_expected_shortfall(portfolio_returns.values, var_95)
            sharpe = self.calculate_sharpe_ratio(portfolio_returns.values)
            sortino = self.calculate_sortino_ratio(portfolio_returns.values)
            mdd = self.calculate_max_drawdown(portfolio_returns.cumsum().values)
            beta = self.calculate_beta(portfolio_returns.values, market_returns.values)
            alpha = self.calculate_alpha(portfolio_returns.values, market_returns.values, beta)
            corr_matrix = returns.corr().values

            return RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                expected_shortfall=es,
                sharpe_ratio=sharpe,
                sortino_ratio=sortino,
                max_drawdown=mdd,
                beta=beta,
                alpha=alpha,
                correlation_matrix=corr_matrix
            )

        except Exception as e:
            self.logger.error(f"Error in risk assessment: {str(e)}")
            raise

    def validate_trade(self, trade_size: float, current_portfolio: Dict[str, float], 
                      risk_metrics: RiskMetrics) -> bool:
        """Validate if a trade meets risk management criteria"""
        # Check position size limits
        if trade_size > self.max_position_size * sum(current_portfolio.values()):
            return False

        # Check portfolio VaR limit
        if abs(risk_metrics.var_95) > self.max_portfolio_var:
            return False

        # Check Sharpe Ratio threshold
        if risk_metrics.sharpe_ratio < self.min_sharpe_ratio:
            return False

        return True
