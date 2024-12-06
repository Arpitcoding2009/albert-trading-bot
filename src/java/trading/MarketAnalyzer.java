package com.albert.trading.bot.trading;

import com.albert.trading.bot.ml.DeepLearningModel;
import com.albert.trading.bot.sentiment.SentimentAnalyzer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import java.util.*;
import java.util.concurrent.*;
import java.time.Instant;
import java.util.logging.Logger;

public class MarketAnalyzer {
    private static final Logger LOGGER = Logger.getLogger(MarketAnalyzer.class.getName());
    private final DeepLearningModel deepLearningModel;
    private final SentimentAnalyzer sentimentAnalyzer;
    private final ExecutorService analysisExecutor;
    private final Map<String, Double> marketSentiment = new ConcurrentHashMap<>();
    private final Map<String, List<Double>> priceHistory = new ConcurrentHashMap<>();
    
    private static final int SENTIMENT_WINDOW = 1000;
    private static final double ACCURACY_THRESHOLD = 0.995;
    private static final int PRICE_HISTORY_SIZE = 1000000;
    
    public MarketAnalyzer() {
        this.deepLearningModel = new DeepLearningModel();
        this.sentimentAnalyzer = new SentimentAnalyzer();
        this.analysisExecutor = Executors.newWorkStealingPool();
        initializeAnalyzer();
    }
    
    private void initializeAnalyzer() {
        LOGGER.info("Initializing Market Analyzer with high-precision models");
        deepLearningModel.loadPreTrainedModel();
        setupRealTimeAnalysis();
    }
    
    public Future<MarketPrediction> analyzeTicker(String ticker) {
        return analysisExecutor.submit(() -> {
            double sentiment = analyzeSentiment(ticker);
            double[] technicalIndicators = analyzeTechnicals(ticker);
            double[] socialMetrics = analyzeSocialMetrics(ticker);
            
            return generatePrediction(ticker, sentiment, technicalIndicators, socialMetrics);
        });
    }
    
    private double analyzeSentiment(String ticker) {
        CompletableFuture<Double> newsSentiment = sentimentAnalyzer.analyzeNews(ticker);
        CompletableFuture<Double> socialSentiment = sentimentAnalyzer.analyzeSocialMedia(ticker);
        CompletableFuture<Double> redditSentiment = sentimentAnalyzer.analyzeReddit(ticker);
        
        return CompletableFuture.allOf(newsSentiment, socialSentiment, redditSentiment)
            .thenApply(v -> (newsSentiment.join() + socialSentiment.join() + redditSentiment.join()) / 3.0)
            .join();
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
        return sentimentAnalyzer.getSocialMetrics(ticker);
    }
    
    private MarketPrediction generatePrediction(String ticker, double sentiment, 
                                              double[] technicals, double[] socialMetrics) {
        INDArray input = Nd4j.create(concatenateArrays(sentiment, technicals, socialMetrics));
        INDArray output = deepLearningModel.predict(input);
        
        double confidence = output.getDouble(0);
        TradingSignal signal = confidence > 0.5 ? TradingSignal.BUY : TradingSignal.SELL;
        
        return new MarketPrediction(ticker, signal, confidence, Instant.now());
    }
    
    private void setupRealTimeAnalysis() {
        ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(2);
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
        Instant cutoff = Instant.now().minusSeconds(3600); // 1 hour
        priceHistory.values().forEach(history -> {
            while (history.size() > PRICE_HISTORY_SIZE) {
                history.remove(0);
            }
        });
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
        try {
            if (!analysisExecutor.awaitTermination(60, TimeUnit.SECONDS)) {
                analysisExecutor.shutdownNow();
            }
        } catch (InterruptedException e) {
            analysisExecutor.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }
}
