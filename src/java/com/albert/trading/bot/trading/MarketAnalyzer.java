package com.albert.trading.bot.trading;

import java.time.Instant;
import java.time.temporal.ChronoUnit;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.albert.trading.bot.ml.DeepLearningModel;
import com.albert.trading.bot.model.MarketPrediction;
import com.albert.trading.bot.model.TradingSignal;
import com.albert.trading.bot.sentiment.SentimentAnalyzer;

public class MarketAnalyzer {
    private static final Logger LOGGER = Logger.getLogger(MarketAnalyzer.class.getName());
    private final SentimentAnalyzer sentimentAnalyzer;
    private final DeepLearningModel deepLearningModel;
    public MarketAnalyzer(DeepLearningModel deepLearningModel) {
        this.sentimentAnalyzer = new SentimentAnalyzer();
        this.deepLearningModel = deepLearningModel;
        this.analysisExecutor = Executors.newFixedThreadPool(10);
        this.marketSentiment = new ConcurrentHashMap<>();
        this.priceHistory = new ConcurrentHashMap<>();
        this.scheduler = Executors.newScheduledThreadPool(2);
        
        initializeAnalyzer();
    }

    public DeepLearningModel getDeepLearningModel() {
        return deepLearningModel;
    }

    private final ExecutorService analysisExecutor;
    private final Map<String, Double> marketSentiment;
    private final Map<String, List<Double>> priceHistory;
    private final ScheduledExecutorService scheduler;
    
    private static final int SENTIMENT_WINDOW = 1000;
    public static int getSentimentWindow() {
        return SENTIMENT_WINDOW;
    }

    private static final double ACCURACY_THRESHOLD = 0.995;
    public static double getAccuracyThreshold() {
        return ACCURACY_THRESHOLD;
    }

    private static final int PRICE_HISTORY_SIZE = 1000000;
    
    public static int getPriceHistorySize() {
        return PRICE_HISTORY_SIZE;
    }

    private void initializeAnalyzer() {
        LOGGER.info("Initializing Market Analyzer with high-precision models");
        deepLearningModel.loadPreTrainedModel();
        setupRealTimeAnalysis();
    }
    
    public java.util.concurrent.Future<MarketPrediction> analyzeMarket(String ticker) {
        return analysisExecutor.submit(() -> {
            double sentiment = analyzeSentiment(ticker);
            double[] technicalIndicators = analyzeTechnicals(ticker);
            double[] socialMetrics = analyzeSocialMetrics(ticker);
            
            return generatePrediction(ticker, sentiment, technicalIndicators, socialMetrics);
        });
    }
    
    private double analyzeSentiment(String ticker) {
        try {
            CompletableFuture<Double> newsSentiment = sentimentAnalyzer.analyzeNewsSentiment(ticker);
            CompletableFuture<Double> socialSentiment = sentimentAnalyzer.analyzeSocialMedia(ticker);
            
            // Weight factors for different sentiment sources
            final double NEWS_WEIGHT = 0.6;
            final double SOCIAL_WEIGHT = 0.4;
            
            return CompletableFuture.allOf(newsSentiment, socialSentiment)
                .thenApply(v -> {
                    try {
                        double newsScore = newsSentiment.join();
                        double socialScore = socialSentiment.join();
                        
                        // Get additional social metrics for volume/engagement weighting
                        double[] socialMetrics = sentimentAnalyzer.getSocialMetrics(ticker);
                        double volumeMultiplier = calculateVolumeMultiplier(socialMetrics);
                        
                        // Apply weighted scoring
                        return (newsScore * NEWS_WEIGHT + socialScore * SOCIAL_WEIGHT) * volumeMultiplier;
                    } catch (Exception e) {
                        LOGGER.warning("Error calculating sentiment for " + ticker + ": " + e.getMessage());
                        return 0.0;
                    }
                })
                .join();
        } catch (Exception e) {
            LOGGER.severe("Critical error in sentiment analysis for " + ticker + ": " + e.getMessage());
            return 0.0;
        }
    }
    
    private double calculateVolumeMultiplier(double[] socialMetrics) {
        if (socialMetrics == null || socialMetrics.length < 2) {
            return 1.0;
        }
        
        // Assuming metrics array contains [sentiment, volume] pairs
        double totalVolume = 0;
        for (int i = 1; i < socialMetrics.length; i += 2) {
            totalVolume += socialMetrics[i];
        }
        
        // Normalize volume multiplier between 0.5 and 1.5
        return 0.5 + Math.min(totalVolume / 1000.0, 1.0);
    }
    
    private double[] analyzeTechnicals(String ticker) {
        List<Double> history = priceHistory.getOrDefault(ticker, new ArrayList<>());
        if (history.size() < 100) {
            return new double[]{0.0, 0.0, 0.0}; // RSI, MACD, BB
        }
        
        double rsi = calculateRSI(history);
        double macd = calculateMACD(history);
        double bollingerBands = calculateBollingerBands(history);
        
        return new double[]{rsi, macd, bollingerBands};
    }
    
    private double[] analyzeSocialMetrics(String ticker) {
        return (double[]) sentimentAnalyzer.getSocialMetrics(ticker);
    }
    
    private MarketPrediction generatePrediction(String ticker, double sentiment, 
                                              double[] technicals, double[] socialMetrics) {
        INDArray input = Nd4j.create(concatenateArrays(sentiment, technicals, socialMetrics));
        try (INDArray output = deepLearningModel.predict(input)) {
            double confidence = output.getDouble(0);
            TradingSignal signal = confidence > 0.5 ? TradingSignal.BUY : TradingSignal.SELL;
            
            return new MarketPrediction(ticker, signal, confidence, Instant.now());
        } catch (Exception e) {
            LOGGER.severe("Failed to generate prediction for " + ticker + ": " + e.getMessage());
            return new MarketPrediction(ticker, TradingSignal.HOLD, 0.0, Instant.now());
        } finally {
            input.close();
        }
    }
    
    private void setupRealTimeAnalysis() {
        scheduler.scheduleAtFixedRate(this::updateMarketSentiment, 0, 1, TimeUnit.SECONDS);
        scheduler.scheduleAtFixedRate(this::cleanupOldData, 1, 1, TimeUnit.HOURS);
    }
    
    private void updateMarketSentiment() {
        marketSentiment.keySet().parallelStream().forEach(ticker -> {
            try {
                double newSentiment = analyzeSentiment(ticker);
                marketSentiment.put(ticker, newSentiment);
            } catch (Exception e) {
                LOGGER.warning("Failed to update sentiment for " + ticker + ": " + e.getMessage());
            }
        });
    }
    
    private void cleanupOldData() {
        Instant cutoff = Instant.now().minus(24, ChronoUnit.HOURS);
        
        // Remove old sentiment data
        marketSentiment.entrySet().removeIf(entry -> {
            String key = entry.getKey();
            return !hasRecentData(key, cutoff);
        });

        // Remove old price history
        priceHistory.entrySet().removeIf(entry -> {
            String key = entry.getKey();
            return !hasRecentData(key, cutoff);
        });
    }

    private boolean hasRecentData(String ticker, Instant cutoff) {
        // Check if we have any data points after the cutoff
        return getLastUpdateTime(ticker).isAfter(cutoff);
    }
    
    private Instant getLastUpdateTime(String ticker) {
        // This method should return the last update time for the given ticker
        // For demonstration purposes, it returns the current time
        return Instant.now();
    }
    
    private double calculateRSI(List<Double> prices) {
        if (prices.size() < 14) return 50.0;
        
        double gains = 0.0, losses = 0.0;
        for (int i = 1; i < 14; i++) {
            double diff = prices.get(i) - prices.get(i - 1);
            if (diff > 0) gains += diff;
            else losses -= diff;
        }
        
        if (losses == 0) return 100.0;
        double rs = gains / losses;
        return 100.0 - (100.0 / (1.0 + rs));
    }
    
    private double calculateMACD(List<Double> prices) {
        double ema12 = calculateEMA(prices, 12);
        double ema26 = calculateEMA(prices, 26);
        return ema12 - ema26;
    }
    
    private double calculateEMA(List<Double> prices, int period) {
        double multiplier = 2.0 / (period + 1.0);
        double ema = prices.get(0);
        
        for (int i = 1; i < prices.size(); i++) {
            ema = (prices.get(i) - ema) * multiplier + ema;
        }
        
        return ema;
    }
    
    private double calculateBollingerBands(List<Double> prices) {
        double sma = prices.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double variance = prices.stream()
            .mapToDouble(price -> Math.pow(price - sma, 2))
            .average()
            .orElse(0.0);
        
        double stdDev = Math.sqrt(variance);
        double upperBand = sma + (2 * stdDev);
        double lowerBand = sma - (2 * stdDev);
        
        double lastPrice = prices.get(prices.size() - 1);
        return (lastPrice - lowerBand) / (upperBand - lowerBand); // 0-1 normalized position
    }
    
    private double[] concatenateArrays(double sentiment, double[] technicals, double[] socialMetrics) {
        double[] result = new double[1 + technicals.length + socialMetrics.length];
        result[0] = sentiment;
        System.arraycopy(technicals, 0, result, 1, technicals.length);
        System.arraycopy(socialMetrics, 0, result, 1 + technicals.length, socialMetrics.length);
        return result;
    }
    
    public void shutdown() {
        analysisExecutor.shutdown();
        scheduler.shutdown();
        try {
            if (!analysisExecutor.awaitTermination(60, TimeUnit.SECONDS)) {
                analysisExecutor.shutdownNow();
            }
            if (!scheduler.awaitTermination(60, TimeUnit.SECONDS)) {
                scheduler.shutdownNow();
            }
        } catch (InterruptedException e) {
            analysisExecutor.shutdownNow();
            scheduler.shutdownNow();
            Thread.currentThread().interrupt();
        }
        
        deepLearningModel.shutdown();
        sentimentAnalyzer.shutdown();
    }
}
