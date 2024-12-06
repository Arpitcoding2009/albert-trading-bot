package com.albert.trading.bot.sentiment;

import com.albert.trading.bot.ml.NeuralPredictor;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import java.util.*;
import java.util.concurrent.*;
import java.util.logging.Logger;
import java.time.Instant;

public class SentimentAnalyzer {
    private static final Logger LOGGER = Logger.getLogger(SentimentAnalyzer.class.getName());
    
    private final NeuralPredictor neuralPredictor;
    private final Map<String, NewsSource> newsSources;
    private final Map<String, SocialMediaSource> socialSources;
    private final ExecutorService analysisExecutor;
    private final Map<String, Double> sentimentCache;
    private final TokenizerFactory tokenizerFactory;
    
    private static final int CACHE_SIZE = 10000;
    private static final long CACHE_EXPIRY = 300000; // 5 minutes
    private static final double SOCIAL_WEIGHT = 0.4;
    private static final double NEWS_WEIGHT = 0.6;
    
    public SentimentAnalyzer() {
        this.neuralPredictor = new NeuralPredictor();
        this.newsSources = new ConcurrentHashMap<>();
        this.socialSources = new ConcurrentHashMap<>();
        this.analysisExecutor = Executors.newWorkStealingPool();
        this.sentimentCache = new ConcurrentHashMap<>();
        this.tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
        
        initializeSources();
        startCacheCleanup();
    }
    
    private void initializeSources() {
        // Initialize news sources
        newsSources.put("reuters", new NewsSource("reuters", "API_KEY"));
        newsSources.put("bloomberg", new NewsSource("bloomberg", "API_KEY"));
        newsSources.put("cryptonews", new NewsSource("cryptonews", "API_KEY"));
        
        // Initialize social media sources
        socialSources.put("twitter", new TwitterSource("API_KEY"));
        socialSources.put("reddit", new RedditSource("API_KEY"));
        socialSources.put("telegram", new TelegramSource("API_KEY"));
        
        LOGGER.info("Initialized " + newsSources.size() + " news sources and " +
                   socialSources.size() + " social media sources");
    }
    
    public CompletableFuture<SentimentResult> analyzeSentiment(String ticker) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                Double cachedSentiment = sentimentCache.get(ticker);
                if (cachedSentiment != null) {
                    return new SentimentResult(ticker, cachedSentiment, true);
                }
                
                CompletableFuture<Double> newsSentiment = analyzeNewsSentiment(ticker);
                CompletableFuture<Double> socialSentiment = analyzeSocialSentiment(ticker);
                
                CompletableFuture<Double> combinedSentiment = newsSentiment
                    .thenCombine(socialSentiment, (news, social) -> 
                        (news * NEWS_WEIGHT + social * SOCIAL_WEIGHT));
                
                double sentiment = combinedSentiment.get();
                sentimentCache.put(ticker, sentiment);
                
                return new SentimentResult(ticker, sentiment, false);
            } catch (Exception e) {
                LOGGER.severe("Sentiment analysis failed for " + ticker + ": " + e.getMessage());
                return new SentimentResult(ticker, 0.0, false);
            }
        }, analysisExecutor);
    }
    
    private CompletableFuture<Double> analyzeNewsSentiment(String ticker) {
        List<CompletableFuture<List<NewsArticle>>> articleFutures = new ArrayList<>();
        
        for (NewsSource source : newsSources.values()) {
            articleFutures.add(source.getArticles(ticker));
        }
        
        return CompletableFuture.allOf(articleFutures.toArray(new CompletableFuture[0]))
            .thenApply(v -> articleFutures.stream()
                .flatMap(future -> future.join().stream())
                .collect(Collectors.toList()))
            .thenApply(this::calculateNewsSentiment);
    }
    
    private CompletableFuture<Double> analyzeSocialSentiment(String ticker) {
        List<CompletableFuture<List<SocialPost>>> postFutures = new ArrayList<>();
        
        for (SocialMediaSource source : socialSources.values()) {
            postFutures.add(source.getPosts(ticker));
        }
        
        return CompletableFuture.allOf(postFutures.toArray(new CompletableFuture[0]))
            .thenApply(v -> postFutures.stream()
                .flatMap(future -> future.join().stream())
                .collect(Collectors.toList()))
            .thenApply(this::calculateSocialSentiment);
    }
    
    private double calculateNewsSentiment(List<NewsArticle> articles) {
        if (articles.isEmpty()) return 0.0;
        
        return articles.stream()
            .mapToDouble(article -> {
                double titleSentiment = analyzeSentimentText(article.getTitle());
                double contentSentiment = analyzeSentimentText(article.getContent());
                double sourceTrust = article.getSourceTrust();
                
                return (titleSentiment * 0.3 + contentSentiment * 0.7) * sourceTrust;
            })
            .average()
            .orElse(0.0);
    }
    
    private double calculateSocialSentiment(List<SocialPost> posts) {
        if (posts.isEmpty()) return 0.0;
        
        return posts.stream()
            .mapToDouble(post -> {
                double textSentiment = analyzeSentimentText(post.getText());
                double authorInfluence = post.getAuthorInfluence();
                double engagement = post.getEngagementScore();
                
                return textSentiment * authorInfluence * engagement;
            })
            .average()
            .orElse(0.0);
    }
    
    private double analyzeSentimentText(String text) {
        try {
            // Preprocess text
            String cleanText = preprocessText(text);
            
            // Convert to features
            double[] features = textToFeatures(cleanText);
            
            // Use neural network for prediction
            MarketData data = new MarketData();
            data.setTextFeatures(features);
            
            PredictionResult result = neuralPredictor.predict(data).get();
            return result.getConfidence() * (result.getSignal() == TradingSignal.BUY ? 1 : -1);
        } catch (Exception e) {
            LOGGER.warning("Text sentiment analysis failed: " + e.getMessage());
            return 0.0;
        }
    }
    
    private String preprocessText(String text) {
        // Remove URLs
        text = text.replaceAll("https?://\\S+\\s?", "");
        
        // Remove special characters
        text = text.replaceAll("[^a-zA-Z0-9\\s]", " ");
        
        // Convert to lowercase
        text = text.toLowerCase();
        
        // Remove extra whitespace
        text = text.replaceAll("\\s+", " ").trim();
        
        return text;
    }
    
    private double[] textToFeatures(String text) {
        // Use tokenizer to split text
        List<String> tokens = tokenizerFactory.create(text).getTokens();
        
        // Convert tokens to numerical features
        // This is a simplified version - in practice, you'd use word embeddings
        double[] features = new double[1000]; // Match INPUT_SIZE in NeuralPredictor
        for (String token : tokens) {
            int hash = Math.abs(token.hashCode() % features.length);
            features[hash] += 1.0;
        }
        
        // Normalize features
        double sum = Arrays.stream(features).sum();
        if (sum > 0) {
            for (int i = 0; i < features.length; i++) {
                features[i] /= sum;
            }
        }
        
        return features;
    }
    
    private void startCacheCleanup() {
        ScheduledExecutorService scheduler = Executors.newSingleThreadScheduledExecutor();
        scheduler.scheduleAtFixedRate(() -> {
            try {
                Instant cutoff = Instant.now().minusMillis(CACHE_EXPIRY);
                sentimentCache.clear(); // Simple cache invalidation
            } catch (Exception e) {
                LOGGER.warning("Cache cleanup failed: " + e.getMessage());
            }
        }, CACHE_EXPIRY, CACHE_EXPIRY, TimeUnit.MILLISECONDS);
    }
    
    public Map<String, Double> getSocialMetrics(String ticker) {
        Map<String, Double> metrics = new HashMap<>();
        metrics.put("twitter_sentiment", 
            socialSources.get("twitter").getAverageSentiment(ticker));
        metrics.put("reddit_sentiment", 
            socialSources.get("reddit").getAverageSentiment(ticker));
        metrics.put("telegram_sentiment", 
            socialSources.get("telegram").getAverageSentiment(ticker));
        return metrics;
    }
    
    public void shutdown() {
        analysisExecutor.shutdown();
        try {
            if (!analysisExecutor.awaitTermination(60, TimeUnit.SECONDS)) {
                analysisExecutor.shutdownNow();
            }
        } catch (InterruptedException e) {
            analysisExecutor.shutdownNow();
            Thread.currentThread().interrupt();
        }
        
        neuralPredictor.shutdown();
    }
}
