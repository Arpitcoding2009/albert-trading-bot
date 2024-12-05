import numpy as np
import pandas as pd
import numpy_financial as npf
import ccxt
import sklearn
import pybind11

def test_technical_analysis():
    # Create sample price data
    prices = [100, 102, 104, 103, 105, 107, 106, 108, 110, 112]
    df = pd.DataFrame({'close': prices})
    
    # Test Technical Indicators
    df['rsi'] = (100 - (100 / (1 + (np.mean(df['close'].diff()[1:]) / np.abs(df['close'].diff()[1:]).mean()))))
    df['sma'] = df['close'].rolling(window=5).mean()
    
    print("Technical Analysis Test:")
    print(df)
    
    # Test Financial Calculations
    npv = npf.npv(0.1, prices)
    print(f"\nNet Present Value: {npv}")

def test_exchange_connection():
    try:
        exchanges = [ccxt.coindcx]
        for ExchangeClass in exchanges:
            try:
                exchange = ExchangeClass()
                markets = exchange.load_markets()
                print(f"\n{ExchangeClass.__name__} Connection Test:")
                print(f"Total Markets: {len(markets)}")
                break
            except Exception as e:
                print(f"{ExchangeClass.__name__} Connection Failed: {e}")
    except Exception as e:
        print(f"No exchanges could connect: {e}")

def main():
    print("Albert Trading Bot - Deployment Test")
    test_technical_analysis()
    test_exchange_connection()

if __name__ == "__main__":
    main()
