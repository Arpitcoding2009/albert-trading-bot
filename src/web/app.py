import os
import asyncio
import json
from typing import List, Dict

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import ccxt

from src.training import EnhancedTradingBotTrainer
from src.architecture import SecurityManager
from src.utils.config import load_config

class AlbertTradingWebApp:
    def __init__(self):
        self.app = FastAPI(title="Albert Trading Bot", description="AI-Powered Cryptocurrency Trading Platform")
        self.setup_middleware()
        self.setup_routes()
        self.setup_websockets()
        
        self.trainer = EnhancedTradingBotTrainer()
        self.security_manager = SecurityManager()
        self.config = load_config()
        
        # Initialize exchange
        self.exchange = ccxt.coindcx({
            'apiKey': os.getenv('COINDCX_API_KEY', ''),
            'secret': os.getenv('COINDCX_SECRET_KEY', ''),
        })

    def setup_middleware(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Mount static files and templates
        self.app.mount("/static", StaticFiles(directory="src/web/static"), name="static")
        self.templates = Jinja2Templates(directory="src/web/templates")

    def setup_routes(self):
        @self.app.get("/")
        async def index(request: Request):
            return self.templates.TemplateResponse("index.html", {"request": request})
        
        @self.app.get("/api/trading-metrics")
        async def get_trading_metrics():
            metrics = self.trainer.get_trading_metrics()
            return {"metrics": metrics}
        
        @self.app.get("/api/portfolio")
        async def get_portfolio():
            portfolio = self.exchange.fetch_balance()
            return {"portfolio": portfolio}

        @self.app.get("/api/dashboard/summary")
        async def get_dashboard_summary():
            """Get dashboard summary data"""
            try:
                portfolio_status = await self.trainer.get_portfolio_status()
                performance_metrics = await self.trainer.get_performance_metrics()
                trading_status = self.trainer.get_status()
                
                return {
                    "portfolio": portfolio_status,
                    "performance": performance_metrics,
                    "trading_status": trading_status
                }
            except Exception as e:
                print(f"Error getting dashboard summary: {str(e)}")
                raise Exception(status_code=500, detail=str(e))

        @self.app.get("/api/dashboard/trades")
        async def get_recent_trades(limit: int = 50):
            """Get recent trades"""
            try:
                trades = await self.trainer.get_trade_history(limit)
                return trades
            except Exception as e:
                print(f"Error getting recent trades: {str(e)}")
                raise Exception(status_code=500, detail=str(e))

        @self.app.post("/api/trading/start")
        async def start_trading(config: Dict):
            """Start automated trading"""
            try:
                await self.trainer.start_trading(config)
                return {"message": "Trading started successfully"}
            except Exception as e:
                print(f"Error starting trading: {str(e)}")
                raise Exception(status_code=500, detail=str(e))

        @self.app.post("/api/trading/stop")
        async def stop_trading():
            """Stop automated trading"""
            try:
                await self.trainer.stop_trading()
                return {"message": "Trading stopped successfully"}
            except Exception as e:
                print(f"Error stopping trading: {str(e)}")
                raise Exception(status_code=500, detail=str(e))

        @self.app.get("/api/settings")
        async def get_settings():
            """Get current settings"""
            try:
                return self.config.dict()
            except Exception as e:
                print(f"Error getting settings: {str(e)}")
                raise Exception(status_code=500, detail=str(e))

        @self.app.post("/api/settings")
        async def update_settings(new_settings: Dict):
            """Update settings"""
            try:
                self.config.update(new_settings)
                return {"message": "Settings updated successfully"}
            except Exception as e:
                print(f"Error updating settings: {str(e)}")
                raise Exception(status_code=500, detail=str(e))

        @self.app.get("/homepage", response_class=HTMLResponse)
        async def homepage(request: Request):
            return self.templates.TemplateResponse("homepage.html", {"request": request})

        @self.app.get("/control-center", response_class=HTMLResponse)
        async def control_center(request: Request):
            return self.templates.TemplateResponse("control_center.html", {"request": request})

        @self.app.get("/training-zone", response_class=HTMLResponse)
        async def training_zone(request: Request):
            return self.templates.TemplateResponse("training_zone.html", {"request": request})

        @self.app.get("/live-market-hub", response_class=HTMLResponse)
        async def live_market_hub(request: Request):
            return self.templates.TemplateResponse("live_market_hub.html", {"request": request})

        @self.app.get("/analytics-dashboard", response_class=HTMLResponse)
        async def analytics_dashboard(request: Request):
            return self.templates.TemplateResponse("analytics_dashboard.html", {"request": request})

        @self.app.get("/security-settings", response_class=HTMLResponse)
        async def security_settings(request: Request):
            return self.templates.TemplateResponse("security_settings.html", {"request": request})

        @self.app.get("/api/live-stats")
        async def live_stats():
            """API endpoint for live market stats"""
            # Placeholder for real-time data
            data = {'profitability': 0.0, 'trades_executed': 0, 'roi': 0.0}
            return data

        @self.app.get("/api/trading-metrics")
        async def trading_metrics():
            """API endpoint for real-time trading metrics"""
            # Placeholder for real-time trading metrics
            data = {'trades_per_second': 0, 'profits': 0.0, 'loss_mitigation': 0.0}
            return data

    def setup_websockets(self):
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await websocket.accept()
            
            try:
                # Add connection to managers
                # await market_data_manager.connection_manager.connect(websocket)
                # await signal_manager.connection_manager.connect(websocket)
                # await alert_manager.connection_manager.connect(websocket)
                
                # Start broadcasting market data
                # await market_data_manager.start_broadcasting()
                
                while True:
                    # Wait for client messages
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    # Handle client messages
                    if message.get('type') == 'ping':
                        await websocket.send_json({'type': 'pong'})
                        
            except Exception as e:
                print(f"WebSocket error: {str(e)}")
            finally:
                # Remove connection from managers
                # market_data_manager.connection_manager.disconnect(websocket)
                # signal_manager.connection_manager.disconnect(websocket)
                # alert_manager.connection_manager.disconnect(websocket)

        @self.app.websocket("/ws/trading-signals")
        async def trading_signals_websocket(websocket: WebSocket):
            await websocket.accept()
            try:
                while True:
                    # Simulate real-time trading signals
                    signals = await self.generate_trading_signals()
                    await websocket.send_json(signals)
                    await asyncio.sleep(5)  # Update every 5 seconds
            except WebSocketDisconnect:
                print("Client disconnected")

    async def generate_trading_signals(self) -> Dict:
        # Simulate trading signals generation
        try:
            # Fetch latest price data
            ticker = self.exchange.fetch_ticker('BTC/USDT')
            
            # Calculate advanced features
            price_data = [ticker['last']]  # Replace with actual historical data
            performance_metrics = self.trainer.optimize_performance(price_data)
            
            return {
                "price": ticker['last'],
                "moving_average": performance_metrics['moving_average'],
                "rsi": performance_metrics['rsi'],
                "recommendation": self.generate_trade_recommendation(performance_metrics)
            }
        except Exception as e:
            return {"error": str(e)}

    def generate_trade_recommendation(self, metrics):
        # Simple trading recommendation logic
        rsi = metrics['rsi']
        if rsi < 30:
            return "BUY"
        elif rsi > 70:
            return "SELL"
        else:
            return "HOLD"

# Create FastAPI app instance
albert_app = AlbertTradingWebApp()
app = albert_app.app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
