import asyncio
import websockets
import json
import ccxt
import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Dict, Any, List

from config import Config
from src.logger import logger, log_trade_event, log_error
from src.training import AdvancedTradingStrategy

class UltraAdvancedTradingBot:
    def __init__(self, 
                 exchange_name: str = 'coindcx', 
                 trading_pair: str = 'BTC/USDT'):
        """
        Initializes an ultra-advanced cryptocurrency trading bot
        with multi-exchange, multi-strategy support
        """
        self.exchanges = {
            'coindcx': ccxt.coindcx(),
        }
        
        self.current_exchange = self.exchanges.get(exchange_name, self.exchanges['coindcx'])
        self.trading_pair = trading_pair
        
        # Advanced strategy components
        self.trading_strategy = AdvancedTradingStrategy(
            exchange_name=exchange_name, 
            trading_pair=trading_pair
        )
        
        # Real-time data storage
        self.market_data: Dict[str, Any] = {
            'price_history': [],
            'volume_history': [],
            'indicators': {}
        }
        
        # Trading parameters
        self.max_trade_amount = Config.MAX_TRADE_AMOUNT
        self.risk_tolerance = Config.RISK_TOLERANCE
        
        # Performance tracking
        self.performance_metrics = {
            'total_trades': 0,
            'successful_trades': 0,
            'total_profit': 0.0
        }

    async def fetch_live_market_data(self):
        """
        Fetch real-time market data using WebSocket
        Supports multiple exchanges with fallback mechanism
        """
        exchanges_to_try = list(self.exchanges.keys())
        
        for exchange_name in exchanges_to_try:
            try:
                exchange = self.exchanges[exchange_name]
                ws_endpoint = exchange.fetch_ticker(self.trading_pair)
                
                async with websockets.connect(ws_endpoint) as websocket:
                    while True:
                        data = await websocket.recv()
                        processed_data = self._process_market_data(data)
                        
                        if processed_data:
                            await self.analyze_and_trade(processed_data)
                        
            except Exception as e:
                log_error(f"WebSocket error with {exchange_name}: {e}")
                continue

    def _process_market_data(self, raw_data: str) -> Dict[str, float]:
        """
        Process and normalize market data from different exchanges
        
        Args:
            raw_data (str): Raw WebSocket market data
        
        Returns:
            Dict[str, float]: Processed market data
        """
        try:
            data = json.loads(raw_data)
            return {
                'price': float(data.get('last', 0)),
                'volume': float(data.get('volume', 0)),
                'timestamp': pd.Timestamp.now()
            }
        except Exception as e:
            log_error(f"Data processing error: {e}")
            return {}

    async def analyze_and_trade(self, market_data: Dict[str, float]):
        """
        Comprehensive market analysis and automated trading
        
        Args:
            market_data (Dict[str, float]): Processed market data
        """
        try:
            # Update market data history
            self.market_data['price_history'].append(market_data['price'])
            self.market_data['volume_history'].append(market_data['volume'])
            
            # Keep only recent data
            self.market_data['price_history'] = self.market_data['price_history'][-100:]
            self.market_data['volume_history'] = self.market_data['volume_history'][-100:]
            
            # Calculate advanced indicators
            df = pd.DataFrame({
                'close': self.market_data['price_history'],
                'volume': self.market_data['volume_history']
            })
            
            processed_data = self.trading_strategy.calculate_features(df)
            
            # Get trading signal
            current_data = processed_data.iloc[-1:][['close', 'volume', 'rsi', 'macd', 'bb_upper', 'bb_lower', 'atr']]
            trade_signal = self.trading_strategy.predict_trade_signal(current_data)
            
            # Risk management
            risk_details = self.trading_strategy.risk_management(trade_signal, market_data['price'])
            
            # Execute trade
            if trade_signal != 0:
                await self.execute_trade(risk_details)
            
        except Exception as e:
            log_error("Trading analysis error", e)

    async def execute_trade(self, trade_details: Dict[str, Any]):
        """
        Execute trades with advanced risk management
        
        Args:
            trade_details (Dict[str, Any]): Trade configuration
        """
        try:
            # Validate trade
            if not Config.TRADING_ENABLED:
                log_trade_event("TRADE_SIMULATION", trade_details)
                return
            
            # Execute trade via exchange
            order = await self.current_exchange.create_order(
                symbol=self.trading_pair,
                type='MARKET' if trade_details['signal'] != 0 else 'LIMIT',
                side='buy' if trade_details['signal'] > 0 else 'sell',
                amount=trade_details['trade_size'],
                price=trade_details.get('stop_loss', None)
            )
            
            # Update performance metrics
            self.performance_metrics['total_trades'] += 1
            
            log_trade_event("TRADE_EXECUTED", {
                "order_id": order.get('id', 'N/A'),
                **trade_details
            })
            
        except Exception as e:
            log_error("Trade execution error", e)

    async def monitor_performance(self):
        """
        Continuously monitor and log trading performance
        """
        while True:
            logger.info(f"Performance Metrics: {self.performance_metrics}")
            await asyncio.sleep(3600)  # Log every hour

    async def start(self):
        """
        Start the ultra-advanced trading bot
        """
        logger.info("🚀 Albert Trading Bot Initializing...")
        
        tasks = [
            asyncio.create_task(self.fetch_live_market_data()),
            asyncio.create_task(self.monitor_performance())
        ]
        
        await asyncio.gather(*tasks)

def main():
    bot = UltraAdvancedTradingBot()
    asyncio.run(bot.start())

if __name__ == "__main__":
    main()
