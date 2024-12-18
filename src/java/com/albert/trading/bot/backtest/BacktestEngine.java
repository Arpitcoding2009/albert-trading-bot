package com.albert.trading.bot.backtest;

import com.albert.trading.bot.model.*;
import com.albert.trading.bot.trading.RiskManager;
import java.time.Instant;
import java.util.*;

public class BacktestEngine {
    private final List<MarketDataPoint> historicalData;
    private final StrategyConfig strategyConfig;
    private final RiskManager riskManager;
    private double initialBalance;
    private double currentBalance;
    private final List<Trade> trades;
    private final Map<String, BacktestPosition> positions;
    
    public BacktestEngine(List<MarketDataPoint> historicalData, 
            StrategyConfig strategyConfig, double initialBalance) {
        this.historicalData = new ArrayList<>(historicalData);
        this.strategyConfig = strategyConfig;
        this.riskManager = new RiskManager();
        this.initialBalance = initialBalance;
        this.currentBalance = initialBalance;
        this.trades = new ArrayList<>();
        this.positions = new HashMap<>();
    }
    
    public BacktestResult runBacktest() {
        for (MarketDataPoint dataPoint : historicalData) {
            processDataPoint(dataPoint);
        }
        return generateBacktestResult();
    }
    
    private void processDataPoint(MarketDataPoint dataPoint) {
        String symbol = dataPoint.getSymbol();
        double price = dataPoint.getPrice();
        
        // Update positions with current price
        updatePositions(symbol, price, dataPoint.getTimestamp());
        
        // Generate trading signal
        TradingSignal signal = generateSignal(dataPoint);
        if (signal != TradingSignal.HOLD) {
            executeTrade(symbol, signal, price, dataPoint.getTimestamp());
        }
    }
    
    private void updatePositions(String symbol, double price, Instant timestamp) {
        BacktestPosition position = positions.get(symbol);
        if (position != null) {
            position.updateUnrealizedPnL(price);
            
            // Check stop loss and take profit
            if (shouldClosePosition(position)) {
                closePosition(symbol, price, timestamp);
            }
        }
    }
    
    private boolean shouldClosePosition(BacktestPosition position) {
        double unrealizedPnL = position.getUnrealizedPnL();
        double entryValue = position.getQuantity() * position.getEntryPrice();
        
        // Check stop loss
        if (unrealizedPnL < -entryValue * strategyConfig.getRiskLevel()) {
            return true;
        }
        
        // Check take profit
        return unrealizedPnL > entryValue * strategyConfig.getRiskLevel() * 2;
    }
    
    private TradingSignal generateSignal(MarketDataPoint dataPoint) {
        Map<String, Double> indicators = dataPoint.getTechnicalIndicators();
        
        // Simple example strategy using RSI
        double rsi = indicators.getOrDefault("RSI", 50.0);
        if (rsi < 30) {
            return TradingSignal.BUY;
        } else if (rsi > 70) {
            return TradingSignal.SELL;
        }
        return TradingSignal.HOLD;
    }
    
    private void executeTrade(String symbol, TradingSignal signal, double price, Instant timestamp) {
        double quantity = riskManager.calculatePositionSize(symbol, price, strategyConfig.getRiskLevel());
        
        if (signal == TradingSignal.BUY && !positions.containsKey(symbol)) {
            openPosition(symbol, quantity, price, timestamp);
        } else if (signal == TradingSignal.SELL && positions.containsKey(symbol)) {
            closePosition(symbol, price, timestamp);
        }
    }
    
    private void openPosition(String symbol, double quantity, double price, Instant timestamp) {
        double cost = quantity * price;
        if (cost <= currentBalance) {
            Trade trade = new Trade.Builder(symbol, TradingSignal.BUY)
                .entryPrice(price)
                .quantity(quantity)
                .entryTime(timestamp)
                .strategy(strategyConfig.getStrategyName())
                .confidence(1.0)
                .build();
                
            trades.add(trade);
            currentBalance -= cost;
            positions.put(symbol, new BacktestPosition(symbol, quantity, price));
        }
    }
    
    private void closePosition(String symbol, double price, Instant timestamp) {
        BacktestPosition position = positions.get(symbol);
        if (position != null) {
            double pnl = position.getUnrealizedPnL();
            currentBalance += (position.getQuantity() * price);
            
            Trade trade = new Trade.Builder(symbol, TradingSignal.SELL)
                .entryPrice(position.getEntryPrice())
                .exitPrice(price)
                .quantity(position.getQuantity())
                .entryTime(position.getEntryTime())
                .exitTime(timestamp)
                .strategy(strategyConfig.getStrategyName())
                .confidence(1.0)
                .profitLoss(pnl)
                .build();
                
            trades.add(trade);
            positions.remove(symbol);
        }
    }
    
    private BacktestResult generateBacktestResult() {
        double totalReturn = (currentBalance - initialBalance) / initialBalance;
        int winningTrades = 0;
        double totalProfit = 0;
        double maxDrawdown = 0;
        double peak = initialBalance;
        double currentDrawdown;
        
        for (Trade trade : trades) {
            if (trade.getProfitLoss() > 0) {
                winningTrades++;
                totalProfit += trade.getProfitLoss();
            }
            
            double balance = initialBalance;
            for (Trade t : trades.subList(0, trades.indexOf(trade) + 1)) {
                balance += t.getProfitLoss();
            }
            
            if (balance > peak) {
                peak = balance;
            }
            
            currentDrawdown = (peak - balance) / peak;
            if (currentDrawdown > maxDrawdown) {
                maxDrawdown = currentDrawdown;
            }
        }
        
        return new BacktestResult.Builder()
            .totalReturn(totalReturn)
            .totalTrades(trades.size())
            .winningTrades(winningTrades)
            .profitFactor(totalProfit / Math.abs(totalProfit - (currentBalance - initialBalance)))
            .maxDrawdown(maxDrawdown)
            .trades(trades)
            .build();
    }
    
    private static class BacktestPosition {
        private final String symbol;
        private final double quantity;
        private final double entryPrice;
        private final Instant entryTime;
        private double unrealizedPnL;
        
        public BacktestPosition(String symbol, double quantity, double entryPrice) {
            this.symbol = symbol;
            this.quantity = quantity;
            this.entryPrice = entryPrice;
            this.entryTime = Instant.now();
            this.unrealizedPnL = 0;
        }
        
        public void updateUnrealizedPnL(double currentPrice) {
            unrealizedPnL = (currentPrice - entryPrice) * quantity;
        }
        
        public String getSymbol() {
            return symbol;
        }
        
        public double getQuantity() {
            return quantity;
        }
        
        public double getEntryPrice() {
            return entryPrice;
        }
        
        public Instant getEntryTime() {
            return entryTime;
        }
        
        public double getUnrealizedPnL() {
            return unrealizedPnL;
        }
    }
}
