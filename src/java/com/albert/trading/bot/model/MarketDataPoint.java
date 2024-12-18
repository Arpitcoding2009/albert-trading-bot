package com.albert.trading.bot.model;

import java.time.Instant;
import java.util.HashMap;
import java.util.Map;

public class MarketDataPoint {
    private final String symbol;
    private final double price;
    private final double volume;
    private final Instant timestamp;
    private final Map<String, Double> technicalIndicators;
    private final Map<String, Double> marketMetrics;
    
    public MarketDataPoint(String symbol, double price, double volume, Instant timestamp) {
        this.symbol = symbol;
        this.price = price;
        this.volume = volume;
        this.timestamp = timestamp;
        this.technicalIndicators = new HashMap<>();
        this.marketMetrics = new HashMap<>();
    }
    
    public String getSymbol() {
        return symbol;
    }
    
    public double getPrice() {
        return price;
    }
    
    public double getVolume() {
        return volume;
    }
    
    public Instant getTimestamp() {
        return timestamp;
    }
    
    public void addTechnicalIndicator(String name, double value) {
        technicalIndicators.put(name, value);
    }
    
    public void addMarketMetric(String name, double value) {
        marketMetrics.put(name, value);
    }
    
    public Double getTechnicalIndicator(String name) {
        return technicalIndicators.get(name);
    }
    
    public Double getMarketMetric(String name) {
        return marketMetrics.get(name);
    }
    
    public Map<String, Double> getTechnicalIndicators() {
        return new HashMap<>(technicalIndicators);
    }
    
    public Map<String, Double> getMarketMetrics() {
        return new HashMap<>(marketMetrics);
    }
    
    @Override
    public String toString() {
        return String.format("MarketDataPoint[symbol=%s, price=%.2f, volume=%.2f, timestamp=%s]",
            symbol, price, volume, timestamp);
    }
}
