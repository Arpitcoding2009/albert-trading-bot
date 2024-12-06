package com.albert.trading.bot.model;

public class TrainingData {
    private final MarketData marketData;
    private final TradingSignal actualOutcome;
    
    public TrainingData(MarketData marketData, TradingSignal actualOutcome) {
        this.marketData = marketData;
        this.actualOutcome = actualOutcome;
    }
    
    public MarketData getMarketData() {
        return marketData;
    }
    
    public TradingSignal getActualOutcome() {
        return actualOutcome;
    }
}
