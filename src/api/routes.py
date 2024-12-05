from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional
import json
import asyncio
from datetime import datetime
import logging
from ..core.trader import AlbertTrader
from ..ml.models import ModelManager
from ..security.auth import verify_credentials
from ..utils.config import Settings

app = FastAPI(title="Albert Trading Bot API", version="2.0")
security = HTTPBasic()
settings = Settings()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
trader = AlbertTrader()
model_manager = ModelManager()

@app.post("/api/login")
async def login(credentials: HTTPBasicCredentials = Depends(security)):
    if not verify_credentials(credentials):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {"message": "Login successful"}

@app.get("/api/status")
async def get_status(credentials: HTTPBasicCredentials = Depends(security)):
    if not verify_credentials(credentials):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    return {
        "status": "active",
        "current_model": model_manager.get_active_model_info(),
        "trading_status": trader.get_status(),
        "last_update": datetime.now().isoformat()
    }

@app.post("/api/train")
async def train_model(
    params: Dict,
    credentials: HTTPBasicCredentials = Depends(security)
):
    if not verify_credentials(credentials):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    try:
        training_task = await model_manager.train_model(params)
        return {"message": "Training started", "task_id": training_task.id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/training/status/{task_id}")
async def get_training_status(
    task_id: str,
    credentials: HTTPBasicCredentials = Depends(security)
):
    if not verify_credentials(credentials):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    status = model_manager.get_training_status(task_id)
    return status

@app.post("/api/trade/start")
async def start_trading(
    config: Dict,
    credentials: HTTPBasicCredentials = Depends(security)
):
    if not verify_credentials(credentials):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    try:
        await trader.start_trading(config)
        return {"message": "Trading started successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/trade/stop")
async def stop_trading(credentials: HTTPBasicCredentials = Depends(security)):
    if not verify_credentials(credentials):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    await trader.stop_trading()
    return {"message": "Trading stopped successfully"}

@app.get("/api/portfolio")
async def get_portfolio(credentials: HTTPBasicCredentials = Depends(security)):
    if not verify_credentials(credentials):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    return await trader.get_portfolio_status()

@app.get("/api/trades/history")
async def get_trade_history(
    limit: int = 50,
    credentials: HTTPBasicCredentials = Depends(security)
):
    if not verify_credentials(credentials):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    return await trader.get_trade_history(limit)

@app.websocket("/ws/market")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            market_data = await trader.get_market_data()
            await websocket.send_json(market_data)
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        await websocket.close()

@app.get("/api/performance")
async def get_performance_metrics(
    timeframe: str = "24h",
    credentials: HTTPBasicCredentials = Depends(security)
):
    if not verify_credentials(credentials):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    return await trader.get_performance_metrics(timeframe)

@app.post("/api/settings/update")
async def update_settings(
    new_settings: Dict,
    credentials: HTTPBasicCredentials = Depends(security)
):
    if not verify_credentials(credentials):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    try:
        await trader.update_settings(new_settings)
        return {"message": "Settings updated successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/exchanges")
async def get_supported_exchanges(
    credentials: HTTPBasicCredentials = Depends(security)
):
    if not verify_credentials(credentials):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    return trader.get_supported_exchanges()

@app.get("/api/models/list")
async def get_available_models(
    credentials: HTTPBasicCredentials = Depends(security)
):
    if not verify_credentials(credentials):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    return model_manager.list_available_models()

@app.post("/api/backtest")
async def run_backtest(
    config: Dict,
    credentials: HTTPBasicCredentials = Depends(security)
):
    if not verify_credentials(credentials):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    try:
        results = await trader.run_backtest(config)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
