import ccxt
from typing import Dict, List, Optional, Union
import pandas as pd
from datetime import datetime, timedelta
import logging
from ..utils.config import Settings
from ..core.exceptions import ExchangeError
from .coindcx_manager import CoinDCXManager

class ExchangeManager:
    def __init__(self):
        self.settings = Settings()
        self.logger = logging.getLogger(__name__)
        self.exchanges = {}
        self.supported_exchanges = {
            'coindcx': CoinDCXManager,
            'binance': ccxt.binance,
            'kucoin': ccxt.kucoin,
            'huobi': ccxt.huobi
        }

    def initialize_exchange(self, exchange_id: str, credentials: Dict):
        """Initialize exchange with API credentials"""
        try:
            if exchange_id not in self.supported_exchanges:
                raise ExchangeError(f"Exchange {exchange_id} not supported")

            exchange_class = self.supported_exchanges[exchange_id]
            if exchange_id == 'coindcx':
                exchange = exchange_class(credentials)
            else:
                exchange = exchange_class({
                    'apiKey': credentials.get('api_key'),
                    'secret': credentials.get('api_secret'),
                    'enableRateLimit': True,
                    'options': {'defaultType': 'spot'}
                })

            # Test connection
            if exchange_id == 'coindcx':
                exchange.test_connection()
            else:
                exchange.load_markets()
            self.exchanges[exchange_id] = exchange
            self.logger.info(f"Successfully initialized {exchange_id} exchange")
            
            return True
        except Exception as e:
            self.logger.error(f"Error initializing {exchange_id}: {str(e)}")
            raise ExchangeError(f"Failed to initialize {exchange_id}: {str(e)}")

    def fetch_ohlcv(self, 
                         exchange_id: str, 
                         symbol: str, 
                         timeframe: str = '1m', 
                         limit: int = 1000) -> pd.DataFrame:
        """Fetch OHLCV data from exchange"""
        try:
            exchange = self.exchanges.get(exchange_id)
            if not exchange:
                raise ExchangeError(f"Exchange {exchange_id} not initialized")

            if exchange_id == 'coindcx':
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            else:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df.set_index('timestamp')

        except Exception as e:
            self.logger.error(f"Error fetching OHLCV data: {str(e)}")
            raise ExchangeError(f"Failed to fetch OHLCV data: {str(e)}")

    def place_order(self, 
                         exchange_id: str, 
                         symbol: str, 
                         order_type: str, 
                         side: str, 
                         amount: float, 
                         price: Optional[float] = None) -> Dict:
        """Place order on exchange"""
        try:
            exchange = self.exchanges.get(exchange_id)
            if not exchange:
                raise ExchangeError(f"Exchange {exchange_id} not initialized")

            params = {}
            if price:
                params['price'] = price

            if exchange_id == 'coindcx':
                order = exchange.place_order(symbol, order_type, side, amount, price)
            else:
                order = exchange.create_order(
                    symbol=symbol,
                    type=order_type,
                    side=side,
                    amount=amount,
                    params=params
                )

            self.logger.info(f"Order placed successfully: {order['id']}")
            return order

        except Exception as e:
            self.logger.error(f"Error placing order: {str(e)}")
            raise ExchangeError(f"Failed to place order: {str(e)}")

    def get_balance(self, exchange_id: str) -> Dict:
        """Get account balance"""
        try:
            exchange = self.exchanges.get(exchange_id)
            if not exchange:
                raise ExchangeError(f"Exchange {exchange_id} not initialized")

            if exchange_id == 'coindcx':
                balance = exchange.get_balance()
            else:
                balance = exchange.fetch_balance()
            return {
                'total': balance['total'],
                'free': balance['free'],
                'used': balance['used']
            }

        except Exception as e:
            self.logger.error(f"Error fetching balance: {str(e)}")
            raise ExchangeError(f"Failed to fetch balance: {str(e)}")

    def get_order_book(self, 
                           exchange_id: str, 
                           symbol: str, 
                           limit: int = 20) -> Dict:
        """Get order book data"""
        try:
            exchange = self.exchanges.get(exchange_id)
            if not exchange:
                raise ExchangeError(f"Exchange {exchange_id} not initialized")

            if exchange_id == 'coindcx':
                order_book = exchange.get_order_book(symbol, limit)
            else:
                order_book = exchange.fetch_order_book(symbol, limit)
            return {
                'bids': order_book['bids'],
                'asks': order_book['asks'],
                'timestamp': order_book['timestamp'],
                'datetime': order_book['datetime']
            }

        except Exception as e:
            self.logger.error(f"Error fetching order book: {str(e)}")
            raise ExchangeError(f"Failed to fetch order book: {str(e)}")

    def get_ticker(self, exchange_id: str, symbol: str) -> Dict:
        """Get current ticker data"""
        try:
            exchange = self.exchanges.get(exchange_id)
            if not exchange:
                raise ExchangeError(f"Exchange {exchange_id} not initialized")

            if exchange_id == 'coindcx':
                ticker = exchange.get_ticker(symbol)
            else:
                ticker = exchange.fetch_ticker(symbol)
            return ticker

        except Exception as e:
            self.logger.error(f"Error fetching ticker: {str(e)}")
            raise ExchangeError(f"Failed to fetch ticker: {str(e)}")

    def cancel_order(self, 
                         exchange_id: str, 
                         order_id: str, 
                         symbol: str) -> Dict:
        """Cancel existing order"""
        try:
            exchange = self.exchanges.get(exchange_id)
            if not exchange:
                raise ExchangeError(f"Exchange {exchange_id} not initialized")

            if exchange_id == 'coindcx':
                result = exchange.cancel_order(order_id, symbol)
            else:
                result = exchange.cancel_order(order_id, symbol)
            self.logger.info(f"Order cancelled successfully: {order_id}")
            return result

        except Exception as e:
            self.logger.error(f"Error cancelling order: {str(e)}")
            raise ExchangeError(f"Failed to cancel order: {str(e)}")

    def get_trades(self, 
                        exchange_id: str, 
                        symbol: str, 
                        limit: int = 100) -> List[Dict]:
        """Get recent trades"""
        try:
            exchange = self.exchanges.get(exchange_id)
            if not exchange:
                raise ExchangeError(f"Exchange {exchange_id} not initialized")

            if exchange_id == 'coindcx':
                trades = exchange.get_trades(symbol, limit)
            else:
                trades = exchange.fetch_trades(symbol, limit=limit)
            return trades

        except Exception as e:
            self.logger.error(f"Error fetching trades: {str(e)}")
            raise ExchangeError(f"Failed to fetch trades: {str(e)}")

    def close_all(self):
        """Close all exchange connections"""
        for exchange_id, exchange in self.exchanges.items():
            try:
                if exchange_id == 'coindcx':
                    exchange.close_connection()
                else:
                    exchange.close()
                self.logger.info(f"Closed connection to {exchange_id}")
            except Exception as e:
                self.logger.error(f"Error closing {exchange_id} connection: {str(e)}")

# Initialize exchange manager
exchange_manager = ExchangeManager()
