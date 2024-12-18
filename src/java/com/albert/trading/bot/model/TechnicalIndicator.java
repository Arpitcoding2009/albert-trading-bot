package com.albert.trading.bot.model;

import java.time.Instant;
import java.util.HashMap;
import java.util.Map;

public class TechnicalIndicator {
    private final String symbol;
    private final Instant timestamp;
    private final Map<String, Double> indicators;
    
    private TechnicalIndicator(Builder builder) {
        this.symbol = builder.symbol;
        this.timestamp = builder.timestamp;
        this.indicators = new HashMap<>(builder.indicators);
    }
    
    public String getSymbol() {
        return symbol;
    }
    
    public Instant getTimestamp() {
        return timestamp;
    }
    
    public Double getIndicator(String name) {
        return indicators.get(name);
    }
    
    public Map<String, Double> getAllIndicators() {
        return new HashMap<>(indicators);
    }
    
    public static class Builder {
        private String symbol;
        private Instant timestamp = Instant.now();
        private final Map<String, Double> indicators = new HashMap<>();
        
        public Builder(String symbol) {
            this.symbol = symbol;
        }
        
        public Builder timestamp(Instant timestamp) {
            this.timestamp = timestamp;
            return this;
        }
        
        public Builder addIndicator(String name, double value) {
            this.indicators.put(name, value);
            return this;
        }
        
        public Builder addRSI(double value) {
            return addIndicator("RSI", value);
        }
        
        public Builder addMACD(double value) {
            return addIndicator("MACD", value);
        }
        
        public Builder addBollingerBands(double upper, double middle, double lower) {
            addIndicator("BB_UPPER", upper);
            addIndicator("BB_MIDDLE", middle);
            addIndicator("BB_LOWER", lower);
            return this;
        }
        
        public Builder addEMA(int period, double value) {
            return addIndicator("EMA_" + period, value);
        }
        
        public Builder addSMA(int period, double value) {
            return addIndicator("SMA_" + period, value);
        }
        
        public Builder addStochastic(double k, double d) {
            addIndicator("STOCH_K", k);
            addIndicator("STOCH_D", d);
            return this;
        }
        
        public Builder addATR(double value) {
            return addIndicator("ATR", value);
        }
        
        public Builder addOBV(double value) {
            return addIndicator("OBV", value);
        }
        
        public TechnicalIndicator build() {
            return new TechnicalIndicator(this);
        }
    }
    
    @Override
    public String toString() {
        return String.format("TechnicalIndicator[symbol=%s, timestamp=%s, indicators=%s]",
            symbol, timestamp, indicators);
    }
}
