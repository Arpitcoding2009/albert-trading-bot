import gym
import numpy as np
from gym import spaces
from typing import Dict, List, Tuple
import pandas as pd
from ...core.risk_manager import RiskManager
from ...core.position_manager import PositionManager
from ...utils.market_data import MarketData

class TradingEnvironment(gym.Env):
    """Custom Trading Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, market_data: MarketData, risk_manager: RiskManager, 
                 position_manager: PositionManager, window_size: int = 100):
        super(TradingEnvironment, self).__init__()

        self.market_data = market_data
        self.risk_manager = risk_manager
        self.position_manager = position_manager
        self.window_size = window_size
        
        # Define action and observation space
        self.action_space = spaces.Discrete(3)  # Buy, Sell, Hold
        
        # Observation space: OHLCV data + technical indicators + position info
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(25,), dtype=np.float32
        )
        
        self.reset()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one time step within the environment"""
        self.current_step += 1
        
        # Get current market state
        current_price = self.market_data.get_current_price()
        position = self.position_manager.get_current_position()
        
        # Execute action
        reward = self._execute_action(action, current_price, position)
        
        # Update state
        next_state = self._get_observation()
        
        # Check if episode is done
        done = self.current_step >= len(self.market_data) - 1
        
        info = {
            'current_price': current_price,
            'position': position,
            'portfolio_value': self.position_manager.get_portfolio_value()
        }
        
        return next_state, reward, done, info

    def reset(self):
        """Reset the state of the environment to an initial state"""
        self.current_step = 0
        self.position_manager.reset()
        return self._get_observation()

    def render(self, mode='human'):
        """Render the environment to the screen"""
        if mode == 'human':
            print(f'Step: {self.current_step}')
            print(f'Portfolio Value: {self.position_manager.get_portfolio_value()}')
            print(f'Current Position: {self.position_manager.get_current_position()}')

    def _get_observation(self) -> np.ndarray:
        """Get the current market observation"""
        # Get market data features
        market_features = self.market_data.get_features(self.current_step, self.window_size)
        
        # Get position features
        position_features = self.position_manager.get_features()
        
        # Combine all features
        observation = np.concatenate([market_features, position_features])
        
        return observation.astype(np.float32)

    def _execute_action(self, action: int, current_price: float, 
                       current_position: float) -> float:
        """Execute trading action and return reward"""
        if action == 0:  # Buy
            if current_position <= 0:
                position_size = self.risk_manager.calculate_position_size(
                    confidence=0.8,  # Can be dynamic based on state
                    max_position=1.0
                )
                self.position_manager.open_long(current_price, position_size)
                
        elif action == 1:  # Sell
            if current_position >= 0:
                position_size = self.risk_manager.calculate_position_size(
                    confidence=0.8,
                    max_position=1.0
                )
                self.position_manager.open_short(current_price, position_size)
        
        # Calculate reward based on PnL and risk-adjusted metrics
        reward = self._calculate_reward(current_price, current_position)
        
        return reward

    def _calculate_reward(self, current_price: float, position: float) -> float:
        """Calculate the reward signal"""
        # Get portfolio metrics
        portfolio_value = self.position_manager.get_portfolio_value()
        previous_value = self.position_manager.get_previous_portfolio_value()
        
        # Calculate returns
        returns = (portfolio_value - previous_value) / previous_value
        
        # Calculate Sharpe ratio component (if enough data)
        if len(self.position_manager.portfolio_history) > 30:
            returns_history = pd.Series(self.position_manager.portfolio_history)
            sharpe = returns_history.mean() / (returns_history.std() + 1e-6)
            reward = returns + 0.1 * sharpe
        else:
            reward = returns
        
        # Penalize for excessive risk
        max_drawdown = self.position_manager.get_max_drawdown()
        if max_drawdown > 0.1:  # 10% drawdown threshold
            reward -= 0.1 * max_drawdown
        
        return reward
