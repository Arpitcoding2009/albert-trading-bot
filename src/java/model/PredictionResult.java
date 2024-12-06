package com.albert.trading.bot.model;

public class PredictionResult {
    private final TradingSignal signal;
    private final double confidence;
    
    public PredictionResult(TradingSignal signal, double confidence) {
        this.signal = signal;
        this.confidence = confidence;
    }
    
    public TradingSignal getSignal() {
        return signal;
    }
    
    public double getConfidence() {
        return confidence;
    }
}
