from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dataclasses import dataclass
import logging

@dataclass
class PortfolioAllocation:
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    efficient_frontier: List[Tuple[float, float]]

class AdvancedPortfolioOptimizer:
    def __init__(self, risk_free_rate: float = 0.02):
        self.logger = logging.getLogger(__name__)
        self.risk_free_rate = risk_free_rate
        self.min_weight = 0.01  # Minimum weight for any asset
        self.max_weight = 0.4   # Maximum weight for any asset

    def calculate_portfolio_metrics(self, weights: np.ndarray, returns: np.ndarray, 
                                 cov_matrix: np.ndarray) -> Tuple[float, float, float]:
        """Calculate portfolio return, volatility, and Sharpe ratio"""
        portfolio_return = np.sum(returns.mean() * weights) * 252  # Annualized return
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol
        return portfolio_return, portfolio_vol, sharpe_ratio

    def negative_sharpe_ratio(self, weights: np.ndarray, returns: np.ndarray, 
                            cov_matrix: np.ndarray) -> float:
        """Objective function for optimization"""
        portfolio_return, portfolio_vol, sharpe_ratio = self.calculate_portfolio_metrics(
            weights, returns, cov_matrix
        )
        return -sharpe_ratio

    def optimize_portfolio(self, historical_data: pd.DataFrame) -> PortfolioAllocation:
        """Optimize portfolio using Modern Portfolio Theory with constraints"""
        try:
            returns = historical_data.pct_change().dropna()
            mean_returns = returns.mean()
            cov_matrix = returns.cov()
            num_assets = len(historical_data.columns)

            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
            ]
            bounds = tuple((self.min_weight, self.max_weight) for _ in range(num_assets))

            # Initial guess (equal weights)
            initial_weights = np.array([1/num_assets] * num_assets)

            # Optimize for maximum Sharpe ratio
            result = minimize(
                self.negative_sharpe_ratio,
                initial_weights,
                args=(returns, cov_matrix),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )

            optimal_weights = result.x
            expected_return, volatility, sharpe_ratio = self.calculate_portfolio_metrics(
                optimal_weights, returns, cov_matrix
            )

            # Generate efficient frontier
            efficient_frontier = self.generate_efficient_frontier(returns, cov_matrix)

            return PortfolioAllocation(
                weights=optimal_weights,
                expected_return=expected_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                efficient_frontier=efficient_frontier
            )

        except Exception as e:
            self.logger.error(f"Portfolio optimization error: {str(e)}")
            raise

    def generate_efficient_frontier(self, returns: pd.DataFrame, 
                                 cov_matrix: np.ndarray) -> List[Tuple[float, float]]:
        """Generate efficient frontier points"""
        num_portfolios = 100
        target_returns = np.linspace(returns.mean().min(), returns.mean().max(), num_portfolios)
        efficient_frontier = []

        for target_return in target_returns:
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.sum(returns.mean() * x) - target_return}
            ]
            bounds = tuple((self.min_weight, self.max_weight) for _ in range(len(returns.columns)))
            
            result = minimize(
                lambda x: np.sqrt(np.dot(x.T, np.dot(cov_matrix, x))),
                np.array([1/len(returns.columns)] * len(returns.columns)),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                portfolio_vol = np.sqrt(np.dot(result.x.T, np.dot(cov_matrix, result.x)))
                efficient_frontier.append((portfolio_vol, target_return))

        return sorted(efficient_frontier)

    def rebalance_portfolio(self, current_allocation: Dict[str, float], 
                          optimal_allocation: Dict[str, float], 
                          threshold: float = 0.05) -> Dict[str, float]:
        """Calculate required trades for portfolio rebalancing"""
        rebalancing_trades = {}
        
        for asset, optimal_weight in optimal_allocation.items():
            current_weight = current_allocation.get(asset, 0)
            weight_diff = optimal_weight - current_weight
            
            if abs(weight_diff) > threshold:
                rebalancing_trades[asset] = weight_diff

        return rebalancing_trades
