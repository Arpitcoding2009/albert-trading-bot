from fastapi import FastAPI, Request, WebSocket, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import logging
import json
from typing import Dict
from ..core.trader import trader
from ..api.websocket import market_data_manager, signal_manager, alert_manager
from ..utils.config import Settings

# Initialize FastAPI app
app = FastAPI(title="Albert Trading Bot", version="2.0")

# Setup static files and templates
static_path = Path(__file__).parent.parent.parent / "static"
templates_path = Path(__file__).parent.parent.parent / "templates"

app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
templates = Jinja2Templates(directory=str(templates_path))

# Initialize settings and logger
settings = Settings()
logger = logging.getLogger(__name__)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render main dashboard"""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": "Albert Trading Bot",
            "version": "2.0"
        }
    )

@app.get("/api/dashboard/summary")
async def get_dashboard_summary():
    """Get dashboard summary data"""
    try:
        portfolio_status = await trader.get_portfolio_status()
        performance_metrics = await trader.get_performance_metrics()
        trading_status = trader.get_status()
        
        return {
            "portfolio": portfolio_status,
            "performance": performance_metrics,
            "trading_status": trading_status
        }
    except Exception as e:
        logger.error(f"Error getting dashboard summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dashboard/trades")
async def get_recent_trades(limit: int = 50):
    """Get recent trades"""
    try:
        trades = await trader.get_trade_history(limit)
        return trades
    except Exception as e:
        logger.error(f"Error getting recent trades: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    
    try:
        # Add connection to managers
        await market_data_manager.connection_manager.connect(websocket)
        await signal_manager.connection_manager.connect(websocket)
        await alert_manager.connection_manager.connect(websocket)
        
        # Start broadcasting market data
        await market_data_manager.start_broadcasting()
        
        while True:
            # Wait for client messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle client messages
            if message.get('type') == 'ping':
                await websocket.send_json({'type': 'pong'})
                
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
    finally:
        # Remove connection from managers
        market_data_manager.connection_manager.disconnect(websocket)
        signal_manager.connection_manager.disconnect(websocket)
        alert_manager.connection_manager.disconnect(websocket)

@app.post("/api/trading/start")
async def start_trading(config: Dict):
    """Start automated trading"""
    try:
        await trader.start_trading(config)
        return {"message": "Trading started successfully"}
    except Exception as e:
        logger.error(f"Error starting trading: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/trading/stop")
async def stop_trading():
    """Stop automated trading"""
    try:
        await trader.stop_trading()
        return {"message": "Trading stopped successfully"}
    except Exception as e:
        logger.error(f"Error stopping trading: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/settings")
async def get_settings():
    """Get current settings"""
    try:
        return settings.dict()
    except Exception as e:
        logger.error(f"Error getting settings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/settings")
async def update_settings(new_settings: Dict):
    """Update settings"""
    try:
        settings.update(new_settings)
        return {"message": "Settings updated successfully"}
    except Exception as e:
        logger.error(f"Error updating settings: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/homepage", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("homepage.html", {"request": request})

@app.get("/control-center", response_class=HTMLResponse)
async def control_center(request: Request):
    return templates.TemplateResponse("control_center.html", {"request": request})

@app.get("/training-zone", response_class=HTMLResponse)
async def training_zone(request: Request):
    return templates.TemplateResponse("training_zone.html", {"request": request})

@app.get("/live-market-hub", response_class=HTMLResponse)
async def live_market_hub(request: Request):
    return templates.TemplateResponse("live_market_hub.html", {"request": request})

@app.get("/analytics-dashboard", response_class=HTMLResponse)
async def analytics_dashboard(request: Request):
    return templates.TemplateResponse("analytics_dashboard.html", {"request": request})

@app.get("/security-settings", response_class=HTMLResponse)
async def security_settings(request: Request):
    return templates.TemplateResponse("security_settings.html", {"request": request})

@app.get("/api/live-stats")
async def live_stats():
    """API endpoint for live market stats"""
    # Placeholder for real-time data
    data = {'profitability': 0.0, 'trades_executed': 0, 'roi': 0.0}
    return data

@app.get("/api/trading-metrics")
async def trading_metrics():
    """API endpoint for real-time trading metrics"""
    # Placeholder for real-time trading metrics
    data = {'trades_per_second': 0, 'profits': 0.0, 'loss_mitigation': 0.0}
    return data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
