import requests
from transformers import pipeline
import pandas as pd
import numpy as np

class MarketSentimentAnalyzer:
    def __init__(self):
        """
        Initialize sentiment analysis pipeline and data sources
        """
        self.sentiment_model = pipeline("sentiment-analysis")
        self.news_sources = [
            'https://newsapi.org/v2/everything?q=cryptocurrency&apiKey=YOUR_NEWS_API_KEY',
            'https://cryptopanic.com/api/v1/posts/?auth_token=YOUR_CRYPTOPANIC_TOKEN'
        ]
    
    def fetch_crypto_news(self):
        """
        Fetch latest cryptocurrency news from multiple sources
        """
        news_data = []
        for source in self.news_sources:
            try:
                response = requests.get(source)
                news_data.extend(response.json().get('articles', []))
            except Exception as e:
                print(f"Error fetching news from {source}: {e}")
        
        return news_data
    
    def analyze_sentiment(self, texts):
        """
        Analyze sentiment of multiple texts
        
        Args:
            texts (list): List of text to analyze
        
        Returns:
            dict: Sentiment analysis results
        """
        results = self.sentiment_model(texts)
        
        sentiments = {
            'positive': sum(1 for r in results if r['label'] == 'POSITIVE'),
            'negative': sum(1 for r in results if r['label'] == 'NEGATIVE'),
            'neutral': len(texts) - sum(1 for r in results if r['label'] in ['POSITIVE', 'NEGATIVE'])
        }
        
        return {
            'counts': sentiments,
            'sentiment_score': (sentiments['positive'] - sentiments['negative']) / len(texts)
        }
    
    def generate_market_sentiment_report(self):
        """
        Generate comprehensive market sentiment report
        
        Returns:
            dict: Detailed market sentiment analysis
        """
        news_articles = self.fetch_crypto_news()
        headlines = [article['title'] for article in news_articles]
        
        sentiment_analysis = self.analyze_sentiment(headlines)
        
        return {
            'total_articles': len(headlines),
            'sentiment': sentiment_analysis,
            'top_positive_headlines': [
                h for h, r in zip(headlines, self.sentiment_model(headlines)) 
                if r['label'] == 'POSITIVE'
            ][:5],
            'top_negative_headlines': [
                h for h, r in zip(headlines, self.sentiment_model(headlines)) 
                if r['label'] == 'NEGATIVE'
            ][:5]
        }

# Example usage
if __name__ == "__main__":
    analyzer = MarketSentimentAnalyzer()
    report = analyzer.generate_market_sentiment_report()
    print(report)
