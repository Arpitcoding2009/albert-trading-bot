import requests
import logging

class CoinDCXManager:
    def __init__(self, api_key: str, api_secret: str):
        self.base_url = "https://api.coindcx.com"
        self.api_key = api_key
        self.api_secret = api_secret
        self.logger = logging.getLogger(__name__)

    def get_market_data(self):
        url = f"{self.base_url}/exchange/ticker"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching market data: {str(e)}")
            raise

    def get_account_balance(self):
        url = f"{self.base_url}/exchange/v1/users/balances"
        headers = {
            'X-AUTH-APIKEY': self.api_key,
            'X-AUTH-SIGNATURE': self.api_secret
        }
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching account balance: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    api_key = "your_api_key"
    api_secret = "your_api_secret"
    coindcx_manager = CoinDCXManager(api_key, api_secret)
    print(coindcx_manager.get_market_data())
    print(coindcx_manager.get_account_balance())
