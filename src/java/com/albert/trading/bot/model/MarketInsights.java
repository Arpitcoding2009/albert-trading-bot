package com.albert.trading.bot.model;

import java.time.Instant;
import java.util.HashMap;
import java.util.Map;

public class MarketInsights {
    private final String symbol;
    private final double currentPrice;
    private final double predictedPrice;
    private final double confidence;
    private final TradingSignal recommendedAction;
    private final Map<String, Double> technicalIndicators;
    private final Map<String, Double> sentimentMetrics;
    private final Map<String, Double> riskMetrics;
    private final Instant timestamp;
    
    private MarketInsights(Builder builder) {
        this.symbol = builder.symbol;
        this.currentPrice = builder.currentPrice;
        this.predictedPrice = builder.predictedPrice;
        this.confidence = builder.confidence;
        this.recommendedAction = builder.recommendedAction;
        this.technicalIndicators = new HashMap<>(builder.technicalIndicators);
        this.sentimentMetrics = new HashMap<>(builder.sentimentMetrics);
        this.riskMetrics = new HashMap<>(builder.riskMetrics);
        this.timestamp = builder.timestamp;
    }
    
    public String getSymbol() {
        return symbol;
    }
    
    public double getCurrentPrice() {
        return currentPrice;
    }
    
    public double getPredictedPrice() {
        return predictedPrice;
    }
    
    public double getConfidence() {
        return confidence;
    }
    
    public TradingSignal getRecommendedAction() {
        return recommendedAction;
    }
    
    public Map<String, Double> getTechnicalIndicators() {
        return new HashMap<>(technicalIndicators);
    }
    
    public Map<String, Double> getSentimentMetrics() {
        return new HashMap<>(sentimentMetrics);
    }
    
    public Map<String, Double> getRiskMetrics() {
        return new HashMap<>(riskMetrics);
    }
    
    public Instant getTimestamp() {
        return timestamp;
    }
    
    public static class Builder {
        private String symbol;
        private double currentPrice;
        private double predictedPrice;
        private double confidence;
        private TradingSignal recommendedAction;
        private final Map<String, Double> technicalIndicators = new HashMap<>();
        private final Map<String, Double> sentimentMetrics = new HashMap<>();
        private final Map<String, Double> riskMetrics = new HashMap<>();
        private Instant timestamp = Instant.now();
        
        public Builder(String symbol) {
            this.symbol = symbol;
        }
        
        public Builder currentPrice(double currentPrice) {
            this.currentPrice = currentPrice;
            return this;
        }
        
        public Builder predictedPrice(double predictedPrice) {
            this.predictedPrice = predictedPrice;
            return this;
        }
        
        public Builder confidence(double confidence) {
            this.confidence = confidence;
            return this;
        }
        
        public Builder recommendedAction(TradingSignal recommendedAction) {
            this.recommendedAction = recommendedAction;
            return this;
        }
        
        public Builder addTechnicalIndicator(String name, double value) {
            this.technicalIndicators.put(name, value);
            return this;
        }
        
        public Builder addSentimentMetric(String name, double value) {
            this.sentimentMetrics.put(name, value);
            return this;
        }
        
        public Builder addRiskMetric(String name, double value) {
            this.riskMetrics.put(name, value);
            return this;
        }
        
        public Builder timestamp(Instant timestamp) {
            this.timestamp = timestamp;
            return this;
        }
        
        public MarketInsights build() {
            return new MarketInsights(this);
        }
    }
}
