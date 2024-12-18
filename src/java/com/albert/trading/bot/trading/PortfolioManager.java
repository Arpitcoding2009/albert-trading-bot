package com.albert.trading.bot.trading;

import com.albert.trading.bot.model.Trade;
import com.albert.trading.bot.model.PerformanceMetrics;

import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class PortfolioManager {
    private final Map<String, Position> positions;
    private final List<Trade> tradeHistory;
    private double initialBalance;
    private double currentBalance;
    private final RiskManager riskManager;
    
    public RiskManager getRiskManager() {
        return riskManager;
    }

    public PortfolioManager(double initialBalance, RiskManager riskManager) {
        this.positions = new ConcurrentHashMap<>();
        this.tradeHistory = new ArrayList<>();
        this.initialBalance = initialBalance;
        this.currentBalance = initialBalance;
        this.riskManager = riskManager;
    }
    
    public synchronized void updatePosition(Trade trade) {
        Position position = positions.computeIfAbsent(trade.getSymbol(), 
            k -> new Position(trade.getSymbol()));
        position.updatePosition(trade);
        tradeHistory.add(trade);
        
        if (trade.getExitTime() != null) {
            currentBalance += trade.getProfitLoss();
        }
    }
    
    public synchronized PerformanceMetrics calculatePerformanceMetrics() {
        double totalReturn = (currentBalance - initialBalance) / initialBalance;
        int winningTrades = 0;
        int losingTrades = 0;
        double totalProfit = 0;
        double totalLoss = 0;
        
        for (Trade trade : tradeHistory) {
            if (trade.getExitTime() != null) {
                if (trade.getProfitLoss() > 0) {
                    winningTrades++;
                    totalProfit += trade.getProfitLoss();
                } else {
                    losingTrades++;
                    totalLoss += Math.abs(trade.getProfitLoss());
                }
            }
        }
        
        double winRate = tradeHistory.isEmpty() ? 0 : 
            (double) winningTrades / tradeHistory.size();
        double averageWin = winningTrades == 0 ? 0 : totalProfit / winningTrades;
        double averageLoss = losingTrades == 0 ? 0 : totalLoss / losingTrades;
        double profitFactor = totalLoss == 0 ? Double.POSITIVE_INFINITY : totalProfit / totalLoss;
        
        return new PerformanceMetrics.Builder()
            .totalReturn(totalReturn)
            .totalTrades(tradeHistory.size())
            .winningTrades(winningTrades)
            .losingTrades(losingTrades)
            .winRate(winRate)
            .averageWin(averageWin)
            .averageLoss(averageLoss)
            .profitFactor(profitFactor)
            .lastUpdated(Instant.now())
            .build();
    }
    
    public double getCurrentBalance() {
        return currentBalance;
    }
    
    public Map<String, Position> getPositions() {
        return new HashMap<>(positions);
    }
    
    public Position getPosition(String symbol) {
        return positions.get(symbol);
    }
    
    public List<Trade> getTradeHistory() {
        return new ArrayList<>(tradeHistory);
    }
    
    public static class Position {
        private final String symbol;
        private double quantity;
        private double averageEntryPrice;
        private double unrealizedPnL;
        private Instant lastUpdated;
        
        public Position(String symbol) {
            this.symbol = symbol;
            this.quantity = 0;
            this.averageEntryPrice = 0;
            this.unrealizedPnL = 0;
            this.lastUpdated = Instant.now();
        }
        
        public void updatePosition(Trade trade) {
            if (trade.getExitTime() == null) {
                // Opening trade
                double newQuantity = quantity + trade.getQuantity();
                averageEntryPrice = (quantity * averageEntryPrice + 
                    trade.getQuantity() * trade.getEntryPrice()) / newQuantity;
                quantity = newQuantity;
            } else {
                // Closing trade
                quantity -= trade.getQuantity();
                if (quantity == 0) {
                    averageEntryPrice = 0;
                }
            }
            lastUpdated = Instant.now();
        }
        
        public void updateUnrealizedPnL(double currentPrice) {
            unrealizedPnL = (currentPrice - averageEntryPrice) * quantity;
            lastUpdated = Instant.now();
        }
        
        public String getSymbol() {
            return symbol;
        }
        
        public double getQuantity() {
            return quantity;
        }
        
        public double getAverageEntryPrice() {
            return averageEntryPrice;
        }
        
        public double getUnrealizedPnL() {
            return unrealizedPnL;
        }
        
        public Instant getLastUpdated() {
            return lastUpdated;
        }
    }
}
