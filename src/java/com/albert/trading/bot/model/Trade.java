package com.albert.trading.bot.model;

import java.time.Instant;
import java.util.UUID;

public class Trade {
    private final UUID id;
    private final String symbol;
    private final TradingSignal signal;
    private final double entryPrice;
    private final double exitPrice;
    private final double quantity;
    private final double profitLoss;
    private final Instant entryTime;
    private final Instant exitTime;
    private final String strategy;
    private final double confidence;
    
    private Trade(Builder builder) {
        this.id = builder.id;
        this.symbol = builder.symbol;
        this.signal = builder.signal;
        this.entryPrice = builder.entryPrice;
        this.exitPrice = builder.exitPrice;
        this.quantity = builder.quantity;
        this.profitLoss = builder.profitLoss;
        this.entryTime = builder.entryTime;
        this.exitTime = builder.exitTime;
        this.strategy = builder.strategy;
        this.confidence = builder.confidence;
    }
    
    public UUID getId() {
        return id;
    }
    
    public String getSymbol() {
        return symbol;
    }
    
    public TradingSignal getSignal() {
        return signal;
    }
    
    public double getEntryPrice() {
        return entryPrice;
    }
    
    public double getExitPrice() {
        return exitPrice;
    }
    
    public double getQuantity() {
        return quantity;
    }
    
    public double getProfitLoss() {
        return profitLoss;
    }
    
    public Instant getEntryTime() {
        return entryTime;
    }
    
    public Instant getExitTime() {
        return exitTime;
    }
    
    public String getStrategy() {
        return strategy;
    }
    
    public double getConfidence() {
        return confidence;
    }
    
    public static class Builder {
        private UUID id = UUID.randomUUID();
        private String symbol;
        private TradingSignal signal;
        private double entryPrice;
        private double exitPrice;
        private double quantity;
        private double profitLoss;
        private Instant entryTime;
        private Instant exitTime;
        private String strategy;
        private double confidence;
        
        public Builder(String symbol, TradingSignal signal) {
            this.symbol = symbol;
            this.signal = signal;
        }
        
        public Builder entryPrice(double entryPrice) {
            this.entryPrice = entryPrice;
            return this;
        }
        
        public Builder exitPrice(double exitPrice) {
            this.exitPrice = exitPrice;
            return this;
        }
        
        public Builder quantity(double quantity) {
            this.quantity = quantity;
            return this;
        }
        
        public Builder profitLoss(double profitLoss) {
            this.profitLoss = profitLoss;
            return this;
        }
        
        public Builder entryTime(Instant entryTime) {
            this.entryTime = entryTime;
            return this;
        }
        
        public Builder exitTime(Instant exitTime) {
            this.exitTime = exitTime;
            return this;
        }
        
        public Builder strategy(String strategy) {
            this.strategy = strategy;
            return this;
        }
        
        public Builder confidence(double confidence) {
            this.confidence = confidence;
            return this;
        }
        
        public Trade build() {
            return new Trade(this);
        }
    }
    
    @Override
    public String toString() {
        return String.format("Trade[id=%s, symbol=%s, signal=%s, entry=%.2f, exit=%.2f, pl=%.2f]",
            id, symbol, signal, entryPrice, exitPrice, profitLoss);
    }
}
