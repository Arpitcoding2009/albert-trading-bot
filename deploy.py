import os
import sys
import logging
from typing import List, Dict
from datetime import datetime

import uvicorn
from fastapi import FastAPI, BackgroundTasks
import psutil
import platform

# Performance Profiling
import tracemalloc
import time

# Async Performance
import asyncio

# Ensure compatibility with Render's environment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('deployment.log')
    ]
)
logger = logging.getLogger('deployment_manager')

class PerformanceMonitor:
    @staticmethod
    def start_memory_tracking():
        """Start memory tracking."""
        tracemalloc.start()

    @staticmethod
    def stop_memory_tracking():
        """Stop memory tracking and log memory snapshot."""
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        logger.info("[ Memory Usage Top 10 Lines ]")
        for stat in top_stats[:10]:
            logger.info(stat)
        
        tracemalloc.stop()

    @staticmethod
    def log_system_performance():
        """Log comprehensive system performance metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_info = psutil.virtual_memory()
            disk_usage = psutil.disk_usage('/')

            logger.info(f"🖥️ System Performance Snapshot:")
            logger.info(f"CPU Usage: {cpu_percent}%")
            logger.info(f"Memory Usage: {memory_info.percent}%")
            logger.info(f"Total Memory: {memory_info.total / (1024**3):.2f} GB")
            logger.info(f"Available Memory: {memory_info.available / (1024**3):.2f} GB")
            logger.info(f"Disk Usage: {disk_usage.percent}%")
        except Exception as e:
            logger.error(f"Performance logging error: {e}")

class AlbertDeploymentManager:
    def __init__(self):
        self.config = get_config()
        self.performance_monitor = PerformanceMonitor()

    def deploy(self):
        """
        Comprehensive deployment workflow with performance tracking
        """
        try:
            # Start memory tracking
            self.performance_monitor.start_memory_tracking()
            
            logger.info("🚀 Initializing Albert Trading Bot Deployment")
            
            # Performance logging
            start_time = time.time()
            
            # Validate configuration
            self._validate_config()
            
            # Perform pre-deployment checks
            self._pre_deployment_checks()
            
            # Log deployment time
            deployment_time = time.time() - start_time
            logger.info(f"🕒 Deployment Time: {deployment_time:.2f} seconds")
            
            # Log system performance
            self.performance_monitor.log_system_performance()
            
            # Stop memory tracking
            self.performance_monitor.stop_memory_tracking()
            
            logger.info("✅ Deployment Preparation Complete")
        except Exception as e:
            logger.error(f"Deployment Failed: {e}")
            sys.exit(1)

    def _validate_config(self):
        """
        Validate critical configuration parameters
        """
        critical_checks = {
            'MAX_TRADE_AMOUNT': lambda x: x > 0,
            'RISK_TOLERANCE': lambda x: 0 < x < 1
        }

        for key, validator in critical_checks.items():
            value = getattr(self.config, key, None)
            if not validator(value):
                raise ValueError(f"Invalid configuration for {key}: {value}")

    def _pre_deployment_checks(self):
        """
        Perform pre-deployment system checks
        """
        logger.info("🔍 Performing Pre-Deployment System Checks")
        
        # Check Python version
        python_version = platform.python_version()
        logger.info(f"Python Version: {python_version}")
        
        # Check system resources
        logger.info(f"CPU Cores: {psutil.cpu_count()}")
        logger.info(f"Total Memory: {psutil.virtual_memory().total / (1024**3):.2f} GB")

def create_app():
    """
    Create and configure FastAPI application
    """
    app = FastAPI(title="Albert Trading Bot")

    @app.get("/health")
    async def health_check(background_tasks: BackgroundTasks):
        """
        Comprehensive health check endpoint
        """
        config = get_config()
        health_status = {
            "status": "healthy",
            "trading_enabled": config.TRADING_ENABLED,
            "max_trade_amount": config.MAX_TRADE_AMOUNT,
            "risk_tolerance": config.RISK_TOLERANCE,
            "deployment_timestamp": datetime.now().isoformat()
        }
        
        return health_status

    @app.get("/metrics")
    async def system_metrics():
        """
        Provide system and deployment metrics
        """
        return {
            "python_version": platform.python_version(),
            "os": platform.system(),
            "architecture": platform.machine(),
            "cpu_cores": psutil.cpu_count(),
            "total_memory_gb": psutil.virtual_memory().total / (1024**3)
        }

    return app

# Create FastAPI app
app = create_app()

def main():
    # Initialize deployment manager
    deployment_manager = AlbertDeploymentManager()
    
    # Run deployment workflow
    deployment_manager.deploy()

    # Uvicorn configuration
    uvicorn_config = uvicorn.Config(
        app, 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 8000))
    )
    server = uvicorn.Server(uvicorn_config)
    
    # Run server
    server.run()

if __name__ == "__main__":
    if os.environ.get('RENDER') == 'true':
        server = uvicorn.Server(uvicorn_config)
        server.run()
    else:
        main()
