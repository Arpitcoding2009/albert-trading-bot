import os
import sys
import subprocess
import platform
import json
import logging

class RenderDeploymentTroubleshooter:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s: %(message)s')
        
        # Critical configuration checks
        self.critical_checks = [
            self.check_python_version,
            self.check_dependencies,
            self.check_environment_variables,
            self.validate_render_configuration
        ]
    
    def run_diagnostic(self):
        """
        Run comprehensive deployment diagnostic
        """
        diagnostic_report = {
            "system_info": self.get_system_info(),
            "checks": {}
        }
        
        # Run all critical checks
        for check in self.critical_checks:
            check_name = check.__name__
            try:
                result = check()
                diagnostic_report["checks"][check_name] = result
            except Exception as e:
                diagnostic_report["checks"][check_name] = {
                    "status": "error",
                    "message": str(e)
                }
        
        # Generate detailed report
        self.generate_diagnostic_report(diagnostic_report)
        return diagnostic_report
    
    def get_system_info(self):
        """
        Collect comprehensive system information
        """
        return {
            "python_version": platform.python_version(),
            "os": platform.platform(),
            "architecture": platform.machine(),
            "processor": platform.processor()
        }
    
    def check_python_version(self):
        """
        Validate Python version compatibility
        """
        current_version = platform.python_version()
        required_version = "3.10.11"
        
        return {
            "status": "pass" if current_version.startswith("3.10") else "fail",
            "current_version": current_version,
            "required_version": required_version,
            "recommendation": "Ensure Python 3.10.x is used in deployment environment"
        }
    
    def check_dependencies(self):
        """
        Check project dependencies and potential conflicts
        """
        try:
            # Run pip list to get installed packages
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list"], 
                capture_output=True, 
                text=True
            )
            
            # Parse dependencies
            dependencies = {}
            for line in result.stdout.split('\n')[2:]:
                if line.strip():
                    package, version = line.split()[:2]
                    dependencies[package] = version
            
            return {
                "status": "pass",
                "total_packages": len(dependencies),
                "critical_packages": [
                    "torch", "tensorflow", "qiskit", "pennylane", 
                    "transformers", "scikit-learn"
                ]
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }
    
    def check_environment_variables(self):
        """
        Validate critical environment variables
        """
        critical_vars = [
            "QUANTUM_INTELLIGENCE_LEVEL",
            "DYNAMIC_RESOURCE_ALLOCATION",
            "MAX_TRADE_AMOUNT",
            "RISK_TOLERANCE",
            "QUANTUM_COMPUTING_ENABLED"
        ]
        
        var_status = {}
        for var in critical_vars:
            var_status[var] = {
                "exists": var in os.environ,
                "value": os.environ.get(var, "NOT SET")
            }
        
        return {
            "status": "pass" if all(var_status.values()) else "warning",
            "variables": var_status
        }
    
    def validate_render_configuration(self):
        """
        Check Render-specific configuration files
        """
        config_files = {
            "runtime.txt": os.path.exists("runtime.txt"),
            "render.yaml": os.path.exists("render.yaml"),
            "Procfile": os.path.exists("Procfile"),
            "requirements.txt": os.path.exists("requirements.txt")
        }
        
        return {
            "status": "pass" if all(config_files.values()) else "warning",
            "files": config_files
        }
    
    def generate_diagnostic_report(self, report):
        """
        Generate and save diagnostic report
        """
        report_path = "deployment_diagnostic_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Diagnostic report saved to {report_path}")
        
        # Print summary
        print("\n🔍 Deployment Diagnostic Report")
        print("=" * 40)
        for check, result in report["checks"].items():
            status = result.get("status", "UNKNOWN")
            print(f"{check}: {status.upper()}")
        print("=" * 40)

def main():
    troubleshooter = RenderDeploymentTroubleshooter()
    diagnostic_report = troubleshooter.run_diagnostic()

if __name__ == "__main__":
    main()
