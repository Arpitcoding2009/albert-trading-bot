package com.albert.trading.bot.model;

import java.util.Arrays;
import java.util.Map;
import java.util.HashMap;

public class MarketData {
    private double[] prices;
    private double[] volumes;
    private Map<String, Double> technicalIndicators;
    private double sentiment;
    private double[] textFeatures;
    private double confidence;
    
    public MarketData() {
        this.technicalIndicators = new HashMap<>();
    }
    
    public double[] getPrices() {
        return prices;
    }
    
    public void setPrices(double[] prices) {
        this.prices = Arrays.copyOf(prices, prices.length);
    }
    
    public double[] getVolumes() {
        return volumes;
    }
    
    public void setVolumes(double[] volumes) {
        this.volumes = Arrays.copyOf(volumes, volumes.length);
    }
    
    public Map<String, Double> getTechnicalIndicators() {
        return new HashMap<>(technicalIndicators);
    }
    
    public void setTechnicalIndicators(Map<String, Double> indicators) {
        this.technicalIndicators = new HashMap<>(indicators);
    }
    
    public double getSentiment() {
        return sentiment;
    }
    
    public void setSentiment(double sentiment) {
        this.sentiment = sentiment;
    }
    
    public double[] getTextFeatures() {
        return textFeatures;
    }
    
    public void setTextFeatures(double[] features) {
        this.textFeatures = Arrays.copyOf(features, features.length);
    }
    
    public double getConfidence() {
        return confidence;
    }
    
    public void setConfidence(double confidence) {
        this.confidence = confidence;
    }
}
