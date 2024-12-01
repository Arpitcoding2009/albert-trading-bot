import unittest
from src.training import TradingBotTrainer

class TestTradingBotTrainer(unittest.TestCase):
    def setUp(self):
        self.trainer = TradingBotTrainer()

    def test_fetch_tradingview_data(self):
        # Test fetching data with default parameters
        df = self.trainer.fetch_tradingview_data()
        self.assertIsNotNone(df, "DataFrame should not be None")
        self.assertFalse(df.empty, "DataFrame should not be empty")

    def test_fetch_cryptohopper_data(self):
        # Test fetching data with default parameters
        df = self.trainer.fetch_cryptohopper_data()
        self.assertIsNotNone(df, "DataFrame should not be None")
        self.assertFalse(df.empty, "DataFrame should not be empty")

if __name__ == '__main__':
    unittest.main()
