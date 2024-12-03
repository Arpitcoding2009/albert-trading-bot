import os
import sys
import logging
from logging.handlers import RotatingFileHandler
import time
import asyncio
import traceback

# Configure logging
def setup_logging():
    log_dir = os.path.join(os.getcwd(), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, 'background_worker.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            RotatingFileHandler(
                log_file, 
                maxBytes=10*1024*1024,  # 10 MB
                backupCount=5
            )
        ]
    )
    
    return logging.getLogger('AlbertBackgroundWorker')

# Global logger
logger = setup_logging()

class QuantumBackgroundWorker:
    def __init__(self):
        self.running = False
        self.tasks = []
    
    async def initialize(self):
        """
        Initialize background worker components
        """
        try:
            # Import quantum intelligence modules
            from src.core.albert_core_intelligence import AlbertQuantumIntelligence
            from src.trading.advanced_trading_engine import QuantumTradingEngine
            
            self.quantum_intelligence = AlbertQuantumIntelligence()
            self.trading_engine = QuantumTradingEngine()
            
            logger.info("Background worker components initialized successfully")
        except Exception as e:
            logger.critical(f"Initialization failed: {e}")
            logger.critical(traceback.format_exc())
            raise
    
    async def background_task_manager(self):
        """
        Manage and coordinate background tasks
        """
        while self.running:
            try:
                # Perform quantum intelligence background tasks
                quantum_tasks = [
                    self.quantum_intelligence.update_global_market_model(),
                    self.quantum_intelligence.train_adaptive_models(),
                    self.trading_engine.analyze_market_trends()
                ]
                
                # Run tasks concurrently
                results = await asyncio.gather(*quantum_tasks, return_exceptions=True)
                
                # Log task results
                for idx, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.error(f"Task {idx} failed: {result}")
                    else:
                        logger.info(f"Task {idx} completed successfully")
                
                # Wait before next iteration
                await asyncio.sleep(300)  # 5-minute interval
            
            except Exception as e:
                logger.error(f"Background task manager error: {e}")
                logger.error(traceback.format_exc())
                await asyncio.sleep(60)  # Wait before retry
    
    async def start(self):
        """
        Start the background worker
        """
        self.running = True
        
        try:
            await self.initialize()
            
            # Start background task manager
            background_task = asyncio.create_task(self.background_task_manager())
            self.tasks.append(background_task)
            
            logger.info("Background worker started successfully")
            
            # Wait for tasks to complete
            await asyncio.gather(*self.tasks)
        
        except Exception as e:
            logger.critical(f"Background worker startup failed: {e}")
            logger.critical(traceback.format_exc())
            self.running = False
    
    async def stop(self):
        """
        Gracefully stop the background worker
        """
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        await asyncio.gather(*self.tasks, return_exceptions=True)
        logger.info("Background worker stopped gracefully")

def main():
    """
    Main entry point for background worker
    """
    worker = QuantumBackgroundWorker()
    
    try:
        asyncio.run(worker.start())
    except KeyboardInterrupt:
        logger.info("Received interrupt signal. Stopping worker...")
        asyncio.run(worker.stop())
    except Exception as e:
        logger.critical(f"Unhandled exception: {e}")
        logger.critical(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
