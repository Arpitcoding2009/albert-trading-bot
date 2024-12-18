package com.albert.trading.bot.model;

import java.util.HashMap;
import java.util.Map;

public class StrategyConfig {
    private final String strategyName;
    private final Map<String, Double> parameters;
    private final Map<String, Boolean> flags;
    private final Map<String, String> settings;
    private final double riskLevel;
    private final double minConfidence;
    private final int timeframe;
    
    private StrategyConfig(Builder builder) {
        this.strategyName = builder.strategyName;
        this.parameters = new HashMap<>(builder.parameters);
        this.flags = new HashMap<>(builder.flags);
        this.settings = new HashMap<>(builder.settings);
        this.riskLevel = builder.riskLevel;
        this.minConfidence = builder.minConfidence;
        this.timeframe = builder.timeframe;
    }
    
    public String getStrategyName() {
        return strategyName;
    }
    
    public Double getParameter(String name) {
        return parameters.get(name);
    }
    
    public Boolean getFlag(String name) {
        return flags.get(name);
    }
    
    public String getSetting(String name) {
        return settings.get(name);
    }
    
    public Map<String, Double> getAllParameters() {
        return new HashMap<>(parameters);
    }
    
    public Map<String, Boolean> getAllFlags() {
        return new HashMap<>(flags);
    }
    
    public Map<String, String> getAllSettings() {
        return new HashMap<>(settings);
    }
    
    public double getRiskLevel() {
        return riskLevel;
    }
    
    public double getMinConfidence() {
        return minConfidence;
    }
    
    public int getTimeframe() {
        return timeframe;
    }
    
    public static class Builder {
        private String strategyName;
        private final Map<String, Double> parameters = new HashMap<>();
        private final Map<String, Boolean> flags = new HashMap<>();
        private final Map<String, String> settings = new HashMap<>();
        private double riskLevel = 0.5;
        private double minConfidence = 0.7;
        private int timeframe = 60; // Default 1 hour
        
        public Builder(String strategyName) {
            this.strategyName = strategyName;
        }
        
        public Builder addParameter(String name, double value) {
            this.parameters.put(name, value);
            return this;
        }
        
        public Builder addFlag(String name, boolean value) {
            this.flags.put(name, value);
            return this;
        }
        
        public Builder addSetting(String name, String value) {
            this.settings.put(name, value);
            return this;
        }
        
        public Builder riskLevel(double riskLevel) {
            this.riskLevel = riskLevel;
            return this;
        }
        
        public Builder minConfidence(double minConfidence) {
            this.minConfidence = minConfidence;
            return this;
        }
        
        public Builder timeframe(int timeframe) {
            this.timeframe = timeframe;
            return this;
        }
        
        public StrategyConfig build() {
            return new StrategyConfig(this);
        }
    }
    
    @Override
    public String toString() {
        return String.format("StrategyConfig[name=%s, risk=%.2f, confidence=%.2f, timeframe=%d]",
            strategyName, riskLevel, minConfidence, timeframe);
    }
}
