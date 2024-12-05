import psutil
import GPUtil
import cpuinfo
import asyncio
import logging
import json
import os
import time
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuantumHardwareMonitor:
    def __init__(self, 
                 monitoring_interval: int = 60,
                 resource_threshold_percent: float = 80.0):
        self.monitoring_interval = monitoring_interval
        self.resource_threshold_percent = resource_threshold_percent
        self.log_directory = "/opt/albert_data/hardware_logs"
        os.makedirs(self.log_directory, exist_ok=True)
    
    def get_cpu_info(self) -> Dict[str, Any]:
        """Retrieve comprehensive CPU information"""
        cpu_info = cpuinfo.get_cpu_info()
        return {
            "brand": cpu_info.get('brand_raw', 'Unknown'),
            "cores": psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(logical=True),
            "frequency": psutil.cpu_freq().current,
            "usage_percent": psutil.cpu_percent(interval=1)
        }
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Retrieve memory usage statistics"""
        memory = psutil.virtual_memory()
        return {
            "total_gb": round(memory.total / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "used_percent": memory.percent
        }
    
    def get_gpu_info(self) -> List[Dict[str, Any]]:
        """Retrieve GPU information"""
        try:
            gpus = GPUtil.getGPUs()
            return [{
                "name": gpu.name,
                "memory_total_mb": gpu.memoryTotal,
                "memory_used_mb": gpu.memoryUsed,
                "memory_free_mb": gpu.memoryFree,
                "gpu_load_percent": gpu.load * 100
            } for gpu in gpus]
        except Exception as e:
            logger.warning(f"GPU monitoring failed: {e}")
            return []
    
    def get_disk_info(self) -> Dict[str, Any]:
        """Retrieve disk usage information"""
        disk = psutil.disk_usage('/')
        return {
            "total_gb": round(disk.total / (1024**3), 2),
            "used_gb": round(disk.used / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2),
            "used_percent": disk.percent
        }
    
    def log_hardware_metrics(self, metrics: Dict[str, Any]):
        """Log hardware metrics to file"""
        timestamp = int(time.time())
        log_file = os.path.join(
            self.log_directory, 
            f"hardware_metrics_{timestamp}.json"
        )
        
        with open(log_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def check_resource_thresholds(self, metrics: Dict[str, Any]) -> bool:
        """Check if any resource has exceeded threshold"""
        critical_resources = [
            metrics['cpu']['usage_percent'],
            metrics['memory']['used_percent']
        ]
        
        gpu_loads = [gpu['gpu_load_percent'] for gpu in metrics.get('gpus', [])]
        critical_resources.extend(gpu_loads)
        
        return any(
            resource > self.resource_threshold_percent 
            for resource in critical_resources
        )
    
    async def monitor_system(self):
        """Continuously monitor system resources"""
        while True:
            try:
                hardware_metrics = {
                    "timestamp": int(time.time()),
                    "cpu": self.get_cpu_info(),
                    "memory": self.get_memory_info(),
                    "gpus": self.get_gpu_info(),
                    "disk": self.get_disk_info()
                }
                
                self.log_hardware_metrics(hardware_metrics)
                
                if self.check_resource_thresholds(hardware_metrics):
                    logger.warning("Resource threshold exceeded!")
                    # Implement adaptive scaling or notification logic
                
                await asyncio.sleep(self.monitoring_interval)
            
            except Exception as e:
                logger.error(f"System monitoring error: {e}")
                await asyncio.sleep(self.monitoring_interval)

async def main():
    monitor = QuantumHardwareMonitor()
    await monitor.monitor_system()

if __name__ == "__main__":
    asyncio.run(main())
