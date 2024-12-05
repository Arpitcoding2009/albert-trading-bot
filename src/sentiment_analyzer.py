import requests
import numpy as np
import logging
from typing import Dict, Any, List
from transformers import pipeline

class MarketSentimentAnalyzer:
    def __init__(self):
        """
        Advanced Market Sentiment Analysis
        """
        self.sentiment_model = pipeline("sentiment-analysis")
        self.logger = logging.getLogger(__name__)
        
        # Configurable News Sources
        self.news_sources = [
            'https://newsapi.org/v2/everything?q=cryptocurrency',
            'https://cryptopanic.com/api/v1/posts/'
        ]
        
        # Sentiment Weights
        self.sentiment_weights = {
            'POSITIVE': 1.0,
            'NEGATIVE': -1.0,
            'NEUTRAL': 0.0
        }
    
    def fetch_crypto_news(self) -> List[Dict[str, Any]]:
        """
        Fetch cryptocurrency news from multiple sources
        
        Returns:
            List of news articles
        """
        news_data = []
        for source in self.news_sources:
            try:
                response = requests.get(source)
                news_data.extend(response.json().get('articles', []))
            except Exception as e:
                self.logger.error(f"News fetch error from {source}: {e}")
        
        return news_data
    
    def analyze_sentiment(self, texts: List[str]) -> Dict[str, Any]:
        """
        Analyze sentiment of multiple texts
        
        Args:
            texts (List[str]): Texts to analyze
        
        Returns:
            Comprehensive sentiment analysis
        """
        results = self.sentiment_model(texts)
        
        sentiments = {
            'POSITIVE': sum(1 for r in results if r['label'] == 'POSITIVE'),
            'NEGATIVE': sum(1 for r in results if r['label'] == 'NEGATIVE'),
            'NEUTRAL': len(texts) - sum(1 for r in results if r['label'] in ['POSITIVE', 'NEGATIVE'])
        }
        
        # Calculate weighted sentiment score
        sentiment_score = sum(
            self.sentiment_weights[r['label']] for r in results
        ) / len(results)
        
        return {
            'counts': sentiments,
            'sentiment_score': sentiment_score,
            'market_mood': self._interpret_sentiment(sentiment_score)
        }
    
    def _interpret_sentiment(self, score: float) -> str:
        """
        Interpret sentiment score
        
        Args:
            score (float): Sentiment score
        
        Returns:
            Market mood interpretation
        """
        if score > 0.5:
            return 'VERY_BULLISH'
        elif score > 0:
            return 'BULLISH'
        elif score == 0:
            return 'NEUTRAL'
        elif score > -0.5:
            return 'BEARISH'
        else:
            return 'VERY_BEARISH'
    
    def get_current_sentiment(self) -> Dict[str, Any]:
        """
        Generate comprehensive market sentiment report
        
        Returns:
            Detailed market sentiment analysis
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

# Singleton Sentiment Analyzer
sentiment_analyzer = MarketSentimentAnalyzer()

# Example usage
if __name__ == "__main__":
    report = sentiment_analyzer.get_current_sentiment()
    print(report)
