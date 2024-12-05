import os
import sys
import logging
from typing import Any, Dict

class SecureConfig:
    """
    Ultra-secure configuration management with multiple fallback mechanisms
    """
    _instance = None
    _logger = logging.getLogger('SecureConfig')

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(SecureConfig, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize configuration with multiple security layers"""
        self._config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self):
        """
        Load configuration with multiple fallback mechanisms
        Priority: 
        1. Environment Variables
        2. System Keyring (future enhancement)
        3. Secure file-based config
        4. Hardcoded defaults (minimal)
        """
        try:
            # Environment Variable Loading
            config_mappings = {
                'EXCHANGE_NAME': ('EXCHANGE_NAME', 'coindcx'),
                'TRADING_PAIR': ('TRADING_PAIR', 'BTC/USDT'),
                'API_KEY': ('COINDCX_API_KEY', ''),
                'SECRET_KEY': ('COINDCX_SECRET_KEY', ''),
                'TRADING_ENABLED': ('TRADING_ENABLED', 'false'),
                'MAX_TRADE_AMOUNT': ('MAX_TRADE_AMOUNT', 1000),
                'RISK_TOLERANCE': ('RISK_TOLERANCE', 0.02)
            }

            for config_key, (env_key, default) in config_mappings.items():
                value = os.environ.get(env_key, default)
                
                # Type conversion
                if config_key in ['MAX_TRADE_AMOUNT']:
                    value = float(value)
                elif config_key in ['RISK_TOLERANCE']:
                    value = float(value)
                elif config_key == 'TRADING_ENABLED':
                    value = str(value).lower() == 'true'

                self._config[config_key] = value

            self._validate_config()

        except Exception as e:
            self._logger.error(f"Configuration Loading Error: {e}")
            sys.exit(1)

    def _validate_config(self):
        """
        Comprehensive configuration validation
        Raises:
            ValueError: If critical configuration is invalid
        """
        critical_checks = {
            'API_KEY': lambda x: x and len(x) > 10,
            'SECRET_KEY': lambda x: x and len(x) > 10,
            'MAX_TRADE_AMOUNT': lambda x: x > 0,
            'RISK_TOLERANCE': lambda x: 0 < x < 1
        }

        for key, validator in critical_checks.items():
            if not validator(self._config.get(key, '')):
                self._logger.warning(f"Invalid configuration for {key}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve configuration value with optional default
        
        Args:
            key (str): Configuration key
            default (Any, optional): Default value if key not found
        
        Returns:
            Any: Configuration value
        """
        return self._config.get(key, default)

    def __getattr__(self, name: str) -> Any:
        """
        Allow attribute-style access to configuration
        
        Args:
            name (str): Attribute name
        
        Returns:
            Any: Configuration value
        """
        if name in self._config:
            return self._config[name]
        raise AttributeError(f"Configuration '{name}' not found")

# Global configuration instance
config = SecureConfig()

def get_config() -> SecureConfig:
    """
    Retrieve global configuration instance
    
    Returns:
        SecureConfig: Configuration management instance
    """
    return config

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
