from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from textblob import TextBlob
import tweepy
import praw
import requests
from datetime import datetime, timedelta
import asyncio
import aiohttp
from bs4 import BeautifulSoup
import json

class SentimentAnalyzer:
    """Advanced sentiment analysis for cryptocurrency markets"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.initialize_models()
        self.initialize_apis()

    def initialize_models(self):
        """Initialize all NLP models"""
        # FinBERT for financial sentiment
        self.finbert = pipeline("sentiment-analysis",
                              model="ProsusAI/finbert",
                              tokenizer="ProsusAI/finbert")
        
        # RoBERTa for general sentiment
        self.roberta = pipeline("sentiment-analysis",
                              model="cardiffnlp/twitter-roberta-base-sentiment")
        
        # Crypto-specific sentiment model
        self.crypto_sentiment = AutoModelForSequenceClassification.from_pretrained(
            "ElKulako/cryptobert")
        self.crypto_tokenizer = AutoTokenizer.from_pretrained("ElKulako/cryptobert")

    def initialize_apis(self):
        """Initialize API clients"""
        # Twitter API
        self.twitter = tweepy.Client(
            bearer_token=self.config['twitter_bearer_token'],
            consumer_key=self.config['twitter_api_key'],
            consumer_secret=self.config['twitter_api_secret'],
            access_token=self.config['twitter_access_token'],
            access_token_secret=self.config['twitter_access_secret']
        )
        
        # Reddit API
        self.reddit = praw.Reddit(
            client_id=self.config['reddit_client_id'],
            client_secret=self.config['reddit_client_secret'],
            user_agent=self.config['reddit_user_agent']
        )
        
        # Initialize async session
        self.session = aiohttp.ClientSession()

    async def get_comprehensive_sentiment(self, symbol: str) -> Dict:
        """Get comprehensive sentiment analysis from multiple sources"""
        tasks = [
            self.get_twitter_sentiment(symbol),
            self.get_reddit_sentiment(symbol),
            self.get_news_sentiment(symbol),
            self.get_social_metrics(symbol)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Combine results with weighted average
        weights = {
            'twitter': 0.3,
            'reddit': 0.2,
            'news': 0.3,
            'social': 0.2
        }
        
        combined_sentiment = sum(result['sentiment'] * weights[source] 
                               for result, source in zip(results, weights.keys()))
        
        return {
            'overall_sentiment': combined_sentiment,
            'twitter_data': results[0],
            'reddit_data': results[1],
            'news_data': results[2],
            'social_data': results[3],
            'timestamp': datetime.now().isoformat()
        }

    async def get_twitter_sentiment(self, symbol: str) -> Dict:
        """Analyze Twitter sentiment"""
        tweets = self.twitter.search_recent_tweets(
            query=f"#{symbol} OR ${symbol}",
            max_results=100,
            tweet_fields=['created_at', 'public_metrics']
        )
        
        sentiments = []
        impact_scores = []
        
        for tweet in tweets.data:
            # Analyze sentiment using multiple models
            finbert_sentiment = self.finbert(tweet.text)[0]
            roberta_sentiment = self.roberta(tweet.text)[0]
            
            # Calculate impact score based on engagement
            impact = (tweet.public_metrics['retweet_count'] * 2 +
                     tweet.public_metrics['like_count'] +
                     tweet.public_metrics['reply_count'] * 1.5)
            
            # Combine sentiment scores
            sentiment_score = (
                float(finbert_sentiment['score']) if finbert_sentiment['label'] == 'positive'
                else -float(finbert_sentiment['score']) if finbert_sentiment['label'] == 'negative'
                else 0
            )
            
            sentiments.append(sentiment_score)
            impact_scores.append(impact)
        
        # Calculate weighted average sentiment
        if sentiments:
            weighted_sentiment = np.average(sentiments, weights=impact_scores)
        else:
            weighted_sentiment = 0
        
        return {
            'sentiment': weighted_sentiment,
            'tweet_count': len(tweets.data) if tweets.data else 0,
            'impact_score': sum(impact_scores)
        }

    async def get_reddit_sentiment(self, symbol: str) -> Dict:
        """Analyze Reddit sentiment"""
        subreddits = ['cryptocurrency', 'cryptomarkets', f'{symbol.lower()}']
        posts = []
        
        for subreddit in subreddits:
            try:
                subreddit_posts = self.reddit.subreddit(subreddit).search(
                    symbol,
                    time_filter='day',
                    limit=50
                )
                posts.extend(subreddit_posts)
            except Exception as e:
                print(f"Error fetching from r/{subreddit}: {str(e)}")
        
        sentiments = []
        impact_scores = []
        
        for post in posts:
            # Analyze post title and body
            text = f"{post.title} {post.selftext}"
            sentiment = self.crypto_sentiment(text)[0]
            
            # Calculate impact score
            impact = (post.score * 2 + post.num_comments * 1.5)
            
            sentiments.append(sentiment['score'])
            impact_scores.append(impact)
        
        weighted_sentiment = np.average(sentiments, weights=impact_scores) if sentiments else 0
        
        return {
            'sentiment': weighted_sentiment,
            'post_count': len(posts),
            'total_impact': sum(impact_scores)
        }

    async def get_news_sentiment(self, symbol: str) -> Dict:
        """Analyze news sentiment"""
        # Fetch news from multiple sources
        news_apis = [
            f"https://newsapi.org/v2/everything?q={symbol}&apiKey={self.config['news_api_key']}",
            f"https://cryptopanic.com/api/v1/posts/?auth_token={self.config['cryptopanic_api_key']}&currencies={symbol}"
        ]
        
        all_articles = []
        
        async with self.session as session:
            tasks = [self.fetch_news(session, api_url) for api_url in news_apis]
            results = await asyncio.gather(*tasks)
            
            for articles in results:
                all_articles.extend(articles)
        
        sentiments = []
        impact_scores = []
        
        for article in all_articles:
            # Analyze title and description
            text = f"{article['title']} {article.get('description', '')}"
            finbert_sentiment = self.finbert(text)[0]
            
            # Calculate impact based on source reliability and freshness
            impact = self.calculate_news_impact(article)
            
            sentiments.append(float(finbert_sentiment['score']))
            impact_scores.append(impact)
        
        weighted_sentiment = np.average(sentiments, weights=impact_scores) if sentiments else 0
        
        return {
            'sentiment': weighted_sentiment,
            'article_count': len(all_articles),
            'source_diversity': len(set(a['source']['name'] for a in all_articles))
        }

    async def get_social_metrics(self, symbol: str) -> Dict:
        """Analyze social media metrics and trends"""
        # Fetch data from various social platforms
        social_metrics = {
            'telegram_members': await self.get_telegram_metrics(symbol),
            'discord_activity': await self.get_discord_metrics(symbol),
            'github_activity': await self.get_github_metrics(symbol)
        }
        
        # Calculate normalized social score
        max_values = {
            'telegram': 1000000,
            'discord': 100000,
            'github': 1000
        }
        
        normalized_scores = {
            platform: min(value / max_values[platform.split('_')[0]], 1)
            for platform, value in social_metrics.items()
        }
        
        return {
            'sentiment': np.mean(list(normalized_scores.values())),
            'metrics': social_metrics,
            'normalized_scores': normalized_scores
        }

    async def fetch_news(self, session: aiohttp.ClientSession, 
                        api_url: str) -> List[Dict]:
        """Fetch news articles from an API"""
        try:
            async with session.get(api_url) as response:
                data = await response.json()
                return data.get('articles', [])
        except Exception as e:
            print(f"Error fetching news: {str(e)}")
            return []

    def calculate_news_impact(self, article: Dict) -> float:
        """Calculate impact score for a news article"""
        # Base impact score
        impact = 1.0
        
        # Adjust for source reliability
        reliable_sources = ['reuters', 'bloomberg', 'coindesk', 'cointelegraph']
        if any(source in article['source']['name'].lower() 
               for source in reliable_sources):
            impact *= 1.5
        
        # Adjust for freshness
        published = datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00'))
        hours_old = (datetime.now() - published).total_seconds() / 3600
        freshness_factor = max(0.5, 1 - (hours_old / 24))
        impact *= freshness_factor
        
        return impact

    async def get_telegram_metrics(self, symbol: str) -> int:
        """Get Telegram group metrics"""
        # Implement Telegram metrics collection
        return 50000  # Placeholder

    async def get_discord_metrics(self, symbol: str) -> int:
        """Get Discord activity metrics"""
        # Implement Discord metrics collection
        return 25000  # Placeholder

    async def get_github_metrics(self, symbol: str) -> int:
        """Get GitHub activity metrics"""
        # Implement GitHub metrics collection
        return 500  # Placeholder

    async def close(self):
        """Clean up resources"""
        await self.session.close()
