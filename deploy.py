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

# Enhanced Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Console output
        logging.FileHandler('/tmp/albert_trading_bot.log')  # Log file
    ]
)
logger = logging.getLogger(__name__)

# Add global exception handler
def global_exception_handler(exc_type, exc_value, exc_traceback):
    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = global_exception_handler

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
            self._pre.deployment_checks()
            
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
        Enhanced configuration validation with more comprehensive checks
        """
        critical_checks = {
            'MAX_TRADE_AMOUNT': lambda x: x > 0 and x <= 10000,
            'RISK_TOLERANCE': lambda x: 0 < x < 0.2,  # More conservative risk range
            'TRADING_ENABLED': lambda x: isinstance(x, bool)
        }

        for key, validator in critical_checks.items():
            value = getattr(self.config, key, None)
            if value is None:
                raise ValueError(f"Missing critical configuration: {key}")
            if not validator(value):
                raise ValueError(f"Invalid configuration for {key}: {value}")
        
        # Additional security checks
        if not os.getenv('COINDCX_API_KEY') or not os.getenv('COINDCX_SECRET_KEY'):
            logger.warning("⚠️ Exchange API credentials not fully configured!")

    def _pre_deployment_checks(self):
        """
        Enhanced pre-deployment system checks with more detailed logging
        """
        logger.info("🔍 Performing Comprehensive Pre-Deployment System Checks")
        
        # Detailed system information
        system_info = {
            "Python Version": platform.python_version(),
            "OS": platform.system(),
            "OS Version": platform.version(),
            "Machine Architecture": platform.machine(),
            "Processor": platform.processor(),
            "CPU Cores": psutil.cpu_count(),
            "Total Memory (GB)": f"{psutil.virtual_memory().total / (1024**3):.2f}",
            "Available Memory (GB)": f"{psutil.virtual_memory().available / (1024**3):.2f}"
        }
        
        for key, value in system_info.items():
            logger.info(f"📊 {key}: {value}")
        
        # Resource utilization warnings
        if psutil.virtual_memory().percent > 80:
            logger.warning("⚠️ High memory utilization detected! Consider scaling resources.")
        
        if psutil.cpu_percent() > 70:
            logger.warning("⚠️ High CPU utilization detected! Performance might be impacted.")

def create_app():
    """
    Create and configure FastAPI application
    """
    app = FastAPI(title="Albert Trading Bot")

    @app.get("/health")
    async def health_check():
        """
        Comprehensive health check endpoint with system diagnostics
        """
        try:
            # Gather system health information
            health_status = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "system": {
                    "python_version": platform.python_version(),
                    "os": platform.system(),
                    "cpu_cores": psutil.cpu_count(),
                    "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                    "memory_available_gb": round(psutil.virtual_memory().available / (1024**3), 2),
                    "memory_usage_percent": psutil.virtual_memory().percent
                },
                "trading_config": {
                    "trading_enabled": get_config().TRADING_ENABLED,
                    "max_trade_amount": get_config().MAX_TRADE_AMOUNT,
                    "risk_tolerance": get_config().RISK_TOLERANCE
                }
            }
            return health_status
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "error", "message": str(e)}

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

try:
    # Create FastAPI app
    app = create_app()
except Exception as e:
    logger.error(f"Failed to create application: {e}")
    logger.error(traceback.format_exc())
    raise

def main():
    # Initialize deployment manager
    deployment_manager = AlbertDeploymentManager()
    
    # Run deployment workflow
    deployment_manager.deploy()

    # Get port from environment, default to 10000
    port = int(os.getenv("PORT", 10000))
    
    # Uvicorn configuration
    uvicorn_config = uvicorn.Config(
        app, 
        host="0.0.0.0", 
        port=port
    )
    server = uvicorn.Server(uvicorn_config)
    
    # Run server
    server.run()

if __name__ == "__main__":
    # Ensure proper handling for both local and Render environments
    if os.environ.get('RENDER') == 'true':
        # Specific configuration for Render
        port = int(os.getenv("PORT", 10000))
        uvicorn.run(
            "deploy:app", 
            host="0.0.0.0", 
            port=port, 
            reload=False
        )
    else:
        # Local development
        main()
