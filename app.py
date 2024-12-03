import os
import sys
import logging
from logging.handlers import RotatingFileHandler
import traceback
import asyncio
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
from typing import Dict, Any

# Core Web Framework
from src.core.risk_manager import AdvancedRiskManager
from src.sentiment_analyzer import MarketSentimentAnalyzer

# Configure comprehensive logging
def setup_logging():
    log_dir = os.path.join(os.getcwd(), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'albert_quantum_platform.log')
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),  # Console output
            RotatingFileHandler(
                log_file, 
                maxBytes=10*1024*1024,  # 10 MB
                backupCount=5
            )
        ]
    )

    # Create a logger for the application
    logger = logging.getLogger('AlbertQuantumPlatform')
    return logger

# Global logger
logger = setup_logging()

class AlbertTradingPlatform:
    def __init__(self):
        """
        Initialize the comprehensive trading platform
        """
        self.risk_manager = AdvancedRiskManager()
        self.sentiment_analyzer = MarketSentimentAnalyzer()
        self.app = self._create_app()

    def _create_app(self) -> FastAPI:
        """
        Create FastAPI application with comprehensive configuration
        """
        app = FastAPI(
            title="Albert AI Trading Platform",
            description="Advanced Cryptocurrency Trading Ecosystem",
            version="1.2.0"
        )

        # Global Exception Handlers
        @app.exception_handler(Exception)
        async def global_exception_handler(request, exc):
            logger.error(f"Unhandled Exception: {exc}")
            logger.error(traceback.format_exc())
            return JSONResponse(
                status_code=500,
                content={
                    "status": "critical_error",
                    "message": "Unexpected system error occurred",
                    "details": str(exc)
                }
            )

        # Middleware Configuration
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        # Static Files
        base_dir = os.path.dirname(os.path.abspath(__file__))
        app.mount("/static", StaticFiles(directory=os.path.join(base_dir, 'static')), name="static")

        # Health Check Endpoint
        @app.get("/health", status_code=200)
        async def health_check() -> Dict[str, Any]:
            try:
                system_health = {
                    "status": "operational",
                    "risk_management": self.risk_manager.get_risk_status(),
                    "sentiment_score": self.sentiment_analyzer.get_current_sentiment(),
                    "trading_enabled": True
                }
                return system_health
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        # Real-time WebSocket Trading Updates
        @app.websocket("/ws/trading-updates")
        async def websocket_trading_updates(websocket: WebSocket):
            await websocket.accept()
            try:
                while True:
                    trading_data = {
                        "current_price": self.risk_manager.get_current_price(),
                        "portfolio_value": self.risk_manager.calculate_portfolio_value(),
                        "sentiment_score": self.sentiment_analyzer.get_current_sentiment(),
                        "risk_level": self.risk_manager.get_current_risk_level()
                    }
                    await websocket.send_json(trading_data)
                    await asyncio.sleep(5)  # Update interval
            except WebSocketDisconnect:
                logger.info("WebSocket client disconnected")
            except Exception as e:
                logger.error(f"WebSocket Error: {e}")

        return app

    def run(self, host: str = '0.0.0.0', port: int = None):
        """
        Run the trading platform
        """
        port = port or int(os.getenv('PORT', 8000))
        uvicorn.run(
            self.app, 
            host=host, 
            port=port,
            log_level="info",
            workers=4
        )

# Error handler for global exceptions
def global_exception_handler(exc_type, exc_value, exc_traceback):
    """
    Global exception handler for unhandled exceptions
    """
    logger.critical(
        "Uncaught exception",
        exc_info=(exc_type, exc_value, exc_traceback)
    )

# Set global exception handler
sys.excepthook = global_exception_handler

# Main Execution
if __name__ == "__main__":
    trading_platform = AlbertTradingPlatform()
    trading_platform.run()
