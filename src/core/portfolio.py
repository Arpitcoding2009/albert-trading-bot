from typing import Dict, List, Optional
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
import logging
from ..api.exchange import exchange_manager
from ..utils.config import Settings

@dataclass
class Position:
    symbol: str
    side: str
    entry_price: float
    amount: float
    timestamp: datetime
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict = None

@dataclass
class Trade:
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    amount: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_percentage: float
    metadata: Dict = None

class PortfolioManager:
    def __init__(self):
        self.settings = Settings()
        self.logger = logging.getLogger(__name__)
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.initial_balance = 0
        self.config = {}

    async def initialize(self, config: Dict):
        """Initialize portfolio manager with configuration"""
        self.config = config
        
        # Get initial balance
        try:
            balance = await exchange_manager.get_balance(
                self.config['primary_exchange']
            )
            self.initial_balance = float(balance['total']['USDT'])
            self.logger.info(f"Initial balance: {self.initial_balance} USDT")
            
        except Exception as e:
            self.logger.error(f"Error getting initial balance: {str(e)}")
            raise

    async def update_position(self, order: Dict):
        """Update portfolio positions based on executed order"""
        try:
            symbol = order['symbol']
            side = order['side']
            price = float(order['price'])
            amount = float(order['amount'])
            
            if side == 'buy':
                if symbol in self.positions:
                    # Average down existing position
                    pos = self.positions[symbol]
                    total_amount = pos.amount + amount
                    avg_price = (pos.entry_price * pos.amount + price * amount) / total_amount
                    
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        side=side,
                        entry_price=avg_price,
                        amount=total_amount,
                        timestamp=datetime.now(),
                        stop_loss=self._calculate_stop_loss(avg_price, side),
                        take_profit=self._calculate_take_profit(avg_price, side)
                    )
                else:
                    # Create new position
                    self.positions[symbol] = Position(
                        symbol=symbol,
                        side=side,
                        entry_price=price,
                        amount=amount,
                        timestamp=datetime.now(),
                        stop_loss=self._calculate_stop_loss(price, side),
                        take_profit=self._calculate_take_profit(price, side)
                    )
                    
            elif side == 'sell':
                if symbol in self.positions:
                    pos = self.positions[symbol]
                    if amount >= pos.amount:
                        # Close position
                        trade = Trade(
                            symbol=symbol,
                            side=side,
                            entry_price=pos.entry_price,
                            exit_price=price,
                            amount=pos.amount,
                            entry_time=pos.timestamp,
                            exit_time=datetime.now(),
                            pnl=self._calculate_pnl(pos, price),
                            pnl_percentage=self._calculate_pnl_percentage(pos, price)
                        )
                        self.trades.append(trade)
                        del self.positions[symbol]
                    else:
                        # Partial close
                        remaining_amount = pos.amount - amount
                        self.positions[symbol] = Position(
                            symbol=symbol,
                            side=pos.side,
                            entry_price=pos.entry_price,
                            amount=remaining_amount,
                            timestamp=pos.timestamp,
                            stop_loss=pos.stop_loss,
                            take_profit=pos.take_profit
                        )
                        
                        # Record partial trade
                        trade = Trade(
                            symbol=symbol,
                            side=side,
                            entry_price=pos.entry_price,
                            exit_price=price,
                            amount=amount,
                            entry_time=pos.timestamp,
                            exit_time=datetime.now(),
                            pnl=self._calculate_pnl(pos, price, amount),
                            pnl_percentage=self._calculate_pnl_percentage(pos, price)
                        )
                        self.trades.append(trade)
                        
            self.logger.info(f"Updated position for {symbol}: {self.positions.get(symbol)}")
            
        except Exception as e:
            self.logger.error(f"Error updating position: {str(e)}")
            raise

    def _calculate_stop_loss(self, entry_price: float, side: str) -> float:
        """Calculate stop loss price based on risk parameters"""
        risk_percentage = self.config.get('risk', {}).get('stop_loss_percentage', 0.02)
        
        if side == 'buy':
            return entry_price * (1 - risk_percentage)
        else:
            return entry_price * (1 + risk_percentage)

    def _calculate_take_profit(self, entry_price: float, side: str) -> float:
        """Calculate take profit price based on risk parameters"""
        profit_percentage = self.config.get('risk', {}).get('take_profit_percentage', 0.04)
        
        if side == 'buy':
            return entry_price * (1 + profit_percentage)
        else:
            return entry_price * (1 - profit_percentage)

    def _calculate_pnl(self, position: Position, exit_price: float, amount: float = None) -> float:
        """Calculate PnL for a trade"""
        trade_amount = amount if amount else position.amount
        
        if position.side == 'buy':
            return (exit_price - position.entry_price) * trade_amount
        else:
            return (position.entry_price - exit_price) * trade_amount

    def _calculate_pnl_percentage(self, position: Position, exit_price: float) -> float:
        """Calculate percentage PnL for a trade"""
        if position.side == 'buy':
            return ((exit_price - position.entry_price) / position.entry_price) * 100
        else:
            return ((position.entry_price - exit_price) / position.entry_price) * 100

    async def close_all_positions(self):
        """Close all open positions"""
        for symbol, position in list(self.positions.items()):
            try:
                # Get current market price
                ticker = await exchange_manager.get_ticker(
                    self.config['primary_exchange'],
                    symbol
                )
                price = float(ticker['last'])
                
                # Place market order to close position
                order = await exchange_manager.place_order(
                    self.config['primary_exchange'],
                    symbol,
                    'market',
                    'sell' if position.side == 'buy' else 'buy',
                    position.amount
                )
                
                # Update position
                await self.update_position(order)
                
            except Exception as e:
                self.logger.error(f"Error closing position for {symbol}: {str(e)}")

    def get_portfolio_status(self) -> Dict:
        """Get current portfolio status"""
        total_pnl = sum(trade.pnl for trade in self.trades)
        total_pnl_percentage = (total_pnl / self.initial_balance) * 100 if self.initial_balance > 0 else 0
        
        return {
            'positions': self.positions,
            'total_pnl': total_pnl,
            'total_pnl_percentage': total_pnl_percentage,
            'trade_count': len(self.trades),
            'win_rate': self._calculate_win_rate(),
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'max_drawdown': self._calculate_max_drawdown()
        }

    def get_trade_history(self, limit: int = 50) -> List[Trade]:
        """Get recent trade history"""
        return sorted(self.trades, key=lambda x: x.exit_time, reverse=True)[:limit]

    def get_performance_metrics(self, timeframe: str = "24h") -> Dict:
        """Get trading performance metrics for specified timeframe"""
        now = datetime.now()
        if timeframe == "24h":
            start_time = now - timedelta(hours=24)
        elif timeframe == "7d":
            start_time = now - timedelta(days=7)
        elif timeframe == "30d":
            start_time = now - timedelta(days=30)
        else:
            start_time = now - timedelta(hours=24)

        # Filter trades within timeframe
        period_trades = [t for t in self.trades if t.exit_time >= start_time]
        
        if not period_trades:
            return {
                'pnl': 0,
                'pnl_percentage': 0,
                'trade_count': 0,
                'win_rate': 0,
                'avg_trade_duration': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }

        pnl = sum(t.pnl for t in period_trades)
        win_trades = [t for t in period_trades if t.pnl > 0]
        
        return {
            'pnl': pnl,
            'pnl_percentage': (pnl / self.initial_balance) * 100 if self.initial_balance > 0 else 0,
            'trade_count': len(period_trades),
            'win_rate': len(win_trades) / len(period_trades) * 100,
            'avg_trade_duration': self._calculate_avg_trade_duration(period_trades),
            'largest_win': max((t.pnl for t in period_trades), default=0),
            'largest_loss': min((t.pnl for t in period_trades), default=0),
            'sharpe_ratio': self._calculate_sharpe_ratio(period_trades),
            'max_drawdown': self._calculate_max_drawdown(period_trades)
        }

    def _calculate_win_rate(self) -> float:
        """Calculate win rate percentage"""
        if not self.trades:
            return 0
        win_trades = len([t for t in self.trades if t.pnl > 0])
        return (win_trades / len(self.trades)) * 100

    def _calculate_sharpe_ratio(self, trades: List[Trade] = None) -> float:
        """Calculate Sharpe ratio"""
        trades = trades or self.trades
        if not trades:
            return 0
            
        returns = [t.pnl_percentage for t in trades]
        if not returns:
            return 0
            
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
            
        # Assume risk-free rate of 2%
        risk_free_rate = 0.02
        
        # Annualize Sharpe ratio
        sharpe_ratio = (avg_return - risk_free_rate) / std_return * np.sqrt(252)
        return sharpe_ratio

    def _calculate_max_drawdown(self, trades: List[Trade] = None) -> float:
        """Calculate maximum drawdown percentage"""
        trades = trades or self.trades
        if not trades:
            return 0
            
        # Calculate cumulative returns
        cumulative_returns = np.cumsum([t.pnl_percentage for t in trades])
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(cumulative_returns)
        
        # Calculate drawdowns
        drawdowns = (running_max - cumulative_returns) / running_max * 100
        
        return np.max(drawdowns) if len(drawdowns) > 0 else 0

    def _calculate_avg_trade_duration(self, trades: List[Trade]) -> float:
        """Calculate average trade duration in hours"""
        if not trades:
            return 0
            
        durations = [(t.exit_time - t.entry_time).total_seconds() / 3600 for t in trades]
        return np.mean(durations)

# Initialize portfolio manager
portfolio_manager = PortfolioManager()
