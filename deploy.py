import os
import sys
import subprocess
import platform
import logging
from typing import List, Dict
import uvicorn
from fastapi import FastAPI, BackgroundTasks
from config import get_config
from datetime import datetime
import psutil

class AlbertDeploymentManager:
    def __init__(self):
        self.logger = logging.getLogger('deployment_manager')
        self.logger.setLevel(logging.INFO)
        
        # Configure logging
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def check_system_requirements(self) -> bool:
        """
        Comprehensive system requirements check
        
        Returns:
            bool: Whether system meets deployment requirements
        """
        self.logger.info("🔍 Checking System Requirements...")
        
        # Python version check
        python_version = platform.python_version()
        if not (3.8 <= float('.'.join(python_version.split('.')[:2])) <= 3.10):
            self.logger.error(f"Unsupported Python version: {python_version}")
            return False
        
        # OS compatibility
        current_os = platform.system()
        supported_os = ['Windows', 'Linux', 'Darwin']
        if current_os not in supported_os:
            self.logger.error(f"Unsupported OS: {current_os}")
            return False
        
        # Check for required system dependencies
        dependencies = {
            'Windows': ['Microsoft Visual C++ Build Tools'],
            'Linux': ['build-essential', 'cmake'],
            'Darwin': ['xcode-select']
        }
        
        # Perform OS-specific checks
        if not self._check_os_dependencies(dependencies.get(current_os, [])):
            return False
        
        return True

    def _check_os_dependencies(self, dependencies: List[str]) -> bool:
        """
        Check for OS-specific build dependencies
        
        Args:
            dependencies (List[str]): List of required dependencies
        
        Returns:
            bool: Whether all dependencies are satisfied
        """
        for dependency in dependencies:
            try:
                subprocess.run(['which' if platform.system() != 'Windows' else 'where', dependency], 
                               capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError:
                self.logger.error(f"Missing dependency: {dependency}")
                return False
        return True

    def create_virtual_environment(self) -> bool:
        """
        Create and set up a virtual environment
        
        Returns:
            bool: Whether virtual environment was successfully created
        """
        self.logger.info("🌐 Creating Virtual Environment...")
        
        try:
            subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
            self.logger.info("✅ Virtual Environment Created")
            return True
        except Exception as e:
            self.logger.error(f"Virtual Environment Creation Failed: {e}")
            return False

    def install_dependencies(self) -> bool:
        """
        Install project dependencies
        
        Returns:
            bool: Whether dependencies were successfully installed
        """
        self.logger.info("📦 Installing Dependencies...")
        
        try:
            # Activate virtual environment and install dependencies
            pip_command = os.path.join('venv', 'Scripts' if platform.system() == 'Windows' else 'bin', 'pip')
            subprocess.run([pip_command, 'install', '--upgrade', 'pip'], check=True)
            subprocess.run([pip_command, 'install', '-r', 'requirements.txt'], check=True)
            
            # Compile C++ modules
            subprocess.run([sys.executable, 'src/cpp/setup.py', 'build_ext', '--inplace'], check=True)
            
            self.logger.info("✅ Dependencies Installed Successfully")
            return True
        except Exception as e:
            self.logger.error(f"Dependency Installation Failed: {e}")
            return False

    def run_tests(self) -> bool:
        """
        Run comprehensive test suite
        
        Returns:
            bool: Whether all tests passed
        """
        self.logger.info("🧪 Running Test Suite...")
        
        try:
            subprocess.run(['pytest', '-v', 'tests/'], check=True)
            self.logger.info("✅ All Tests Passed")
            return True
        except subprocess.CalledProcessError:
            self.logger.error("❌ Test Suite Failed")
            return False

    def start_trading_bot(self):
        """
        Launch the trading bot with comprehensive configuration
        """
        self.logger.info("🚀 Launching Albert Trading Bot...")
        
        try:
            subprocess.Popen([sys.executable, 'src/websocket_trader.py'])
            self.logger.info("✅ Trading Bot Started Successfully")
        except Exception as e:
            self.logger.error(f"Bot Launch Failed: {e}")

    def start_web_service(self):
        """
        Start web service for deployment monitoring and health checks
        """
        app = FastAPI()

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
            
            # Optional: Background task for additional monitoring
            background_tasks.add_task(self.monitor_system_resources)
            
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
                "processor": platform.processor()
            }

        uvicorn_config = uvicorn.Config(
            app, 
            host="0.0.0.0", 
            port=int(os.getenv("PORT", 8000))
        )
        server = uvicorn.Server(uvicorn_config)
        
        self.logger.info("🌐 Web Service Initialized")
        return server

    def monitor_system_resources(self):
        """
        Background monitoring of system resources
        """
        try:
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            
            self.logger.info(f"System Resources: CPU {cpu_usage}%, Memory {memory_usage}%")
        except ImportError:
            self.logger.warning("psutil not available for resource monitoring")

    def deploy(self):
        """
        Comprehensive deployment workflow
        """
        deployment_steps = [
            self.check_system_requirements,
            self.create_virtual_environment,
            self.install_dependencies,
            self.run_tests
        ]
        
        for step in deployment_steps:
            if not step():
                self.logger.critical("Deployment Halted Due to Failure")
                sys.exit(1)
        
        # Start trading bot
        self.start_trading_bot()

def main():
    deployment_manager = AlbertDeploymentManager()
    
    # Start deployment workflow
    deployment_manager.deploy()
    
    # Start web service
    web_server = deployment_manager.start_web_service()
    
    # Run server
    web_server.run()

if __name__ == "__main__":
    main()
