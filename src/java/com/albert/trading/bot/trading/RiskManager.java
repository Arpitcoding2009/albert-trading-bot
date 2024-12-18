package com.albert.trading.bot.trading;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Logger;
import com.albert.trading.bot.model.TradingSignal;
import com.albert.trading.bot.model.MarketPrediction;

public class RiskManager {
    private static final Logger LOGGER = Logger.getLogger(RiskManager.class.getName());
    private final Map<String, Double> positionSizes;
    private final Map<String, Double> stopLossLevels;
    private final Map<String, Double> riskLevels;
    
    private static final double MAX_POSITION_SIZE = 0.1; // 10% of portfolio
    private static final double DEFAULT_STOP_LOSS = 0.02; // 2% stop loss
    private static final double MAX_RISK_PER_TRADE = 0.01; // 1% risk per trade
    
    public RiskManager() {
        this.positionSizes = new ConcurrentHashMap<>();
        this.stopLossLevels = new ConcurrentHashMap<>();
        this.riskLevels = new ConcurrentHashMap<>();
    }
    
    public double calculatePositionSize(String ticker, double currentPrice, double portfolioValue) {
        double riskLevel = getRiskLevel(ticker);
        double stopLoss = getStopLossLevel(ticker, currentPrice);
        double riskAmount = portfolioValue * MAX_RISK_PER_TRADE * riskLevel;
        double positionSize = riskAmount / (currentPrice * stopLoss);
        return Math.min(positionSize, portfolioValue * MAX_POSITION_SIZE);
    }
    
    public boolean validateTrade(TradingSignal signal, MarketPrediction prediction) {
        if (prediction.getConfidence() < 0.7) {
            LOGGER.warning("Trade rejected: Low confidence prediction");
            return false;
        }
        
        String ticker = signal.getTicker();
        double currentExposure = positionSizes.getOrDefault(ticker, 0.0);
        if (currentExposure >= MAX_POSITION_SIZE) {
            LOGGER.warning("Trade rejected: Maximum position size reached");
            return false;
        }
        
        return true;
    }
    
    public double getStopLossLevel(String ticker, double currentPrice) {
        return stopLossLevels.computeIfAbsent(ticker, k -> DEFAULT_STOP_LOSS);
    }
    
    public void updateStopLoss(String ticker, double newLevel) {
        if (newLevel > 0 && newLevel < 0.1) { // Max 10% stop loss
            stopLossLevels.put(ticker, newLevel);
            LOGGER.info("Updated stop loss for " + ticker + " to " + newLevel);
        }
    }
    
    private double getRiskLevel(String ticker) {
        return riskLevels.computeIfAbsent(ticker, k -> 1.0);
    }
    
    public void updateRiskLevel(String ticker, double volatility, double marketSentiment) {
        double newRiskLevel = calculateDynamicRisk(volatility, marketSentiment);
        riskLevels.put(ticker, newRiskLevel);
        LOGGER.info("Updated risk level for " + ticker + " to " + newRiskLevel);
    }
    
    private double calculateDynamicRisk(double volatility, double marketSentiment) {
        // Normalize volatility to 0-1 range (assuming volatility is in percentage)
        double normalizedVolatility = Math.min(volatility / 100.0, 1.0);
        // Normalize sentiment to 0-1 range
        double normalizedSentiment = (marketSentiment + 1.0) / 2.0;
        
        // Higher volatility = lower risk tolerance
        // Higher positive sentiment = higher risk tolerance
        return (1.0 - normalizedVolatility * 0.5) * (0.5 + normalizedSentiment * 0.5);
    }
    
    public void clearPositions() {
        positionSizes.clear();
        LOGGER.info("Cleared all position sizes");
    }
    
    public void updatePositionSize(String ticker, double size) {
        positionSizes.put(ticker, size);
    }
}
