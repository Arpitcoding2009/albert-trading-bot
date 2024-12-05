from fastapi import WebSocket
from typing import Dict, List, Set
import asyncio
import json
import logging
from datetime import datetime

class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        self.logger = logging.getLogger(__name__)

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        self.logger.info(f"New client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        self.logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: Dict):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return
        
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                self.logger.error(f"Error broadcasting message: {str(e)}")
                await self.disconnect(connection)

    async def send_personal_message(self, message: Dict, websocket: WebSocket):
        """Send message to specific client"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            self.logger.error(f"Error sending personal message: {str(e)}")
            await self.disconnect(websocket)

class MarketDataManager:
    def __init__(self):
        self.connection_manager = ConnectionManager()
        self.logger = logging.getLogger(__name__)
        self.is_broadcasting = False
        self.supported_channels = {
            'market_data': self._get_market_data,
            'trades': self._get_trades,
            'signals': self._get_signals,
            'portfolio': self._get_portfolio
        }

    async def start_broadcasting(self):
        """Start broadcasting market data to all connected clients"""
        if self.is_broadcasting:
            return
        
        self.is_broadcasting = True
        while self.is_broadcasting:
            try:
                # Gather data from all channels
                data = {}
                for channel, func in self.supported_channels.items():
                    data[channel] = await func()

                # Add timestamp
                data['timestamp'] = datetime.now().isoformat()

                # Broadcast to all clients
                await self.connection_manager.broadcast(data)
                
                # Sleep to control broadcast rate
                await asyncio.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error in market data broadcast: {str(e)}")
                await asyncio.sleep(5)  # Wait before retrying

    def stop_broadcasting(self):
        """Stop broadcasting market data"""
        self.is_broadcasting = False

    async def _get_market_data(self) -> Dict:
        """Get current market data"""
        # Implement market data fetching logic
        return {
            "price": 0.0,
            "volume": 0.0,
            "high": 0.0,
            "low": 0.0
        }

    async def _get_trades(self) -> List[Dict]:
        """Get recent trades"""
        # Implement trade data fetching logic
        return []

    async def _get_signals(self) -> List[Dict]:
        """Get trading signals"""
        # Implement signal generation logic
        return []

    async def _get_portfolio(self) -> Dict:
        """Get portfolio status"""
        # Implement portfolio status fetching logic
        return {
            "total_value": 0.0,
            "positions": []
        }

class SignalManager:
    def __init__(self):
        self.connection_manager = ConnectionManager()
        self.logger = logging.getLogger(__name__)

    async def broadcast_signal(self, signal: Dict):
        """Broadcast trading signal to all connected clients"""
        try:
            enriched_signal = {
                **signal,
                'timestamp': datetime.now().isoformat(),
                'type': 'trading_signal'
            }
            await self.connection_manager.broadcast(enriched_signal)
            self.logger.info(f"Trading signal broadcasted: {signal['action']}")
        except Exception as e:
            self.logger.error(f"Error broadcasting signal: {str(e)}")

class AlertManager:
    def __init__(self):
        self.connection_manager = ConnectionManager()
        self.logger = logging.getLogger(__name__)

    async def send_alert(self, alert_type: str, message: str, severity: str = "info"):
        """Send alert to all connected clients"""
        try:
            alert = {
                'type': 'alert',
                'alert_type': alert_type,
                'message': message,
                'severity': severity,
                'timestamp': datetime.now().isoformat()
            }
            await self.connection_manager.broadcast(alert)
            self.logger.info(f"Alert sent: {alert_type} - {message}")
        except Exception as e:
            self.logger.error(f"Error sending alert: {str(e)}")

# Initialize managers
market_data_manager = MarketDataManager()
signal_manager = SignalManager()
alert_manager = AlertManager()
