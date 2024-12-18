package com.albert.trading.bot.model;

import java.time.Instant;

public class MarketPrediction {
    private final String ticker;
    private final TradingSignal signal;
    private final double confidence;
    private final Instant timestamp;
    
    public MarketPrediction(String ticker, TradingSignal signal, double confidence, Instant timestamp) {
        this.ticker = ticker;
        this.signal = signal;
        this.confidence = confidence;
        this.timestamp = timestamp;
    }
    
    public String getTicker() {
        return ticker;
    }
    
    public TradingSignal getSignal() {
        return signal;
    }
    
    public double getConfidence() {
        return confidence;
    }
    
    public Instant getTimestamp() {
        return timestamp;
    }
}
