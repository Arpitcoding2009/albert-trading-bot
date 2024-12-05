from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from textblob import TextBlob
import tweepy
import praw
import logging
from datetime import datetime, timedelta
import asyncio
import aiohttp
from bs4 import BeautifulSoup

class AdvancedSentimentAnalyzer:
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Initialize sentiment models
        self.finbert = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
        self.tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
        self.sentiment_pipeline = pipeline("sentiment-analysis", model=self.finbert, tokenizer=self.tokenizer)
        
        # Social media API clients
        self.twitter_api = self._init_twitter_api()
        self.reddit_api = self._init_reddit_api()
        
        # News API configuration
        self.news_api_key = config.get('news_api_key')
        self.news_sources = [
            'reuters', 'bloomberg', 'coindesk', 'cointelegraph'
        ]
        
        # Sentiment thresholds
        self.sentiment_threshold = 0.6
        self.confidence_threshold = 0.8
        
    def _init_twitter_api(self) -> tweepy.API:
        """Initialize Twitter API client"""
        auth = tweepy.OAuthHandler(
            self.config['twitter_api_key'],
            self.config['twitter_api_secret']
        )
        auth.set_access_token(
            self.config['twitter_access_token'],
            self.config['twitter_access_token_secret']
        )
        return tweepy.API(auth)
    
    def _init_reddit_api(self) -> praw.Reddit:
        """Initialize Reddit API client"""
        return praw.Reddit(
            client_id=self.config['reddit_client_id'],
            client_secret=self.config['reddit_client_secret'],
            user_agent=self.config['reddit_user_agent']
        )

    async def fetch_news_articles(self, symbol: str) -> List[Dict]:
        """Fetch relevant news articles"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for source in self.news_sources:
                url = f"https://newsapi.org/v2/everything?q={symbol}&sources={source}&apiKey={self.news_api_key}"
                tasks.append(self.fetch_source_articles(session, url))
            
            articles = await asyncio.gather(*tasks)
            return [item for sublist in articles for item in sublist]  # Flatten list

    async def fetch_source_articles(self, session: aiohttp.ClientSession, url: str) -> List[Dict]:
        """Fetch articles from a specific source"""
        async with session.get(url) as response:
            data = await response.json()
            return data.get('articles', [])

    async def analyze_social_sentiment(self, symbol: str) -> Dict:
        """Analyze sentiment from social media"""
        # Twitter sentiment
        tweets = self.twitter_api.search_tweets(q=symbol, lang="en", count=100)
        tweet_sentiments = [TextBlob(tweet.text).sentiment.polarity for tweet in tweets]
        
        # Reddit sentiment
        subreddit = self.reddit_api.subreddit('cryptocurrency')
        posts = subreddit.search(symbol, limit=100)
        reddit_sentiments = [TextBlob(post.title + " " + post.selftext).sentiment.polarity 
                           for post in posts]
        
        return {
            'twitter_sentiment': np.mean(tweet_sentiments),
            'reddit_sentiment': np.mean(reddit_sentiments),
            'twitter_volume': len(tweet_sentiments),
            'reddit_volume': len(reddit_sentiments)
        }

    def analyze_news_sentiment(self, articles: List[Dict]) -> Dict:
        """Analyze sentiment from news articles"""
        if not articles:
            return {'compound': 0, 'positive': 0, 'negative': 0, 'neutral': 0}
        
        sentiments = []
        for article in articles:
            text = f"{article['title']} {article['description']}"
            sentiment = self.sentiment_pipeline(text)[0]
            sentiments.append(sentiment)
        
        sentiment_scores = pd.DataFrame(sentiments)
        return {
            'compound': sentiment_scores['score'].mean(),
            'positive': len(sentiment_scores[sentiment_scores['label'] == 'positive']) / len(sentiments),
            'negative': len(sentiment_scores[sentiment_scores['label'] == 'negative']) / len(sentiments),
            'neutral': len(sentiment_scores[sentiment_scores['label'] == 'neutral']) / len(sentiments)
        }

    async def get_comprehensive_sentiment(self, symbol: str) -> Dict:
        """Get comprehensive sentiment analysis"""
        try:
            # Fetch data concurrently
            news_articles = await self.fetch_news_articles(symbol)
            social_sentiment = await self.analyze_social_sentiment(symbol)
            news_sentiment = self.analyze_news_sentiment(news_articles)
            
            # Combine sentiment scores
            combined_sentiment = (
                0.4 * news_sentiment['compound'] +
                0.3 * social_sentiment['twitter_sentiment'] +
                0.3 * social_sentiment['reddit_sentiment']
            )
            
            # Calculate confidence based on volume and consistency
            confidence = min(
                1.0,
                (social_sentiment['twitter_volume'] + social_sentiment['reddit_volume']) / 1000
            )
            
            return {
                'overall_sentiment': combined_sentiment,
                'confidence': confidence,
                'news_sentiment': news_sentiment,
                'social_sentiment': social_sentiment,
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol
            }
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis error: {str(e)}")
            raise

    def get_trading_signal(self, sentiment_data: Dict) -> Dict:
        """Generate trading signal from sentiment analysis"""
        sentiment = sentiment_data['overall_sentiment']
        confidence = sentiment_data['confidence']
        
        if confidence < self.confidence_threshold:
            return {'action': 'HOLD', 'confidence': confidence}
        
        if sentiment > self.sentiment_threshold:
            return {'action': 'BUY', 'confidence': confidence}
        elif sentiment < -self.sentiment_threshold:
            return {'action': 'SELL', 'confidence': confidence}
        else:
            return {'action': 'HOLD', 'confidence': confidence}
