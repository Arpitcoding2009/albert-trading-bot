package com.albert.trading.bot.trading;

import com.albert.trading.bot.model.Trade;
import com.albert.trading.bot.model.TradingSignal;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;

public class OrderManager {
    private final Map<String, List<Trade>> activeTrades;
    private final RiskManager riskManager;
    private final TradeExecutor tradeExecutor;
    
    public OrderManager(RiskManager riskManager, TradeExecutor tradeExecutor) {
        this.activeTrades = new ConcurrentHashMap<>();
        this.riskManager = riskManager;
        this.tradeExecutor = tradeExecutor;
    }
    
    public CompletableFuture<Trade> placeOrder(String symbol, TradingSignal signal, double price, 
            double quantity, String strategy, double confidence) {
        // Validate trade against risk parameters
        if (!riskManager.validateTrade(symbol, quantity, price)) {
            return CompletableFuture.failedFuture(
                new IllegalStateException("Trade validation failed for " + symbol));
        }
        
        // Create and execute the trade
        Trade trade = new Trade.Builder(symbol, signal)
            .entryPrice(price)
            .quantity(quantity)
            .entryTime(Instant.now())
            .strategy(strategy)
            .confidence(confidence)
            .build();
            
        return tradeExecutor.executeTrade(trade)
            .thenApply(executedTrade -> {
                addToActiveTrades(executedTrade);
                return executedTrade;
            });
    }
    
    public CompletableFuture<Trade> closePosition(String symbol, double price) {
        List<Trade> trades = activeTrades.getOrDefault(symbol, new ArrayList<>());
        if (trades.isEmpty()) {
            return CompletableFuture.failedFuture(
                new IllegalStateException("No active trades found for " + symbol));
        }
        
        Trade lastTrade = trades.get(trades.size() - 1);
        Trade closingTrade = new Trade.Builder(symbol, getOppositeSignal(lastTrade.getSignal()))
            .entryPrice(lastTrade.getEntryPrice())
            .exitPrice(price)
            .quantity(lastTrade.getQuantity())
            .entryTime(lastTrade.getEntryTime())
            .exitTime(Instant.now())
            .strategy(lastTrade.getStrategy())
            .confidence(lastTrade.getConfidence())
            .profitLoss(calculateProfitLoss(lastTrade, price))
            .build();
            
        return tradeExecutor.executeTrade(closingTrade)
            .thenApply(executedTrade -> {
                removeFromActiveTrades(symbol, lastTrade);
                return executedTrade;
            });
    }
    
    public List<Trade> getActiveTrades(String symbol) {
        return new ArrayList<>(activeTrades.getOrDefault(symbol, new ArrayList<>()));
    }
    
    public boolean hasActiveTradesForSymbol(String symbol) {
        return activeTrades.containsKey(symbol) && !activeTrades.get(symbol).isEmpty();
    }
    
    private void addToActiveTrades(Trade trade) {
        activeTrades.computeIfAbsent(trade.getSymbol(), k -> new ArrayList<>()).add(trade);
    }
    
    private void removeFromActiveTrades(String symbol, Trade trade) {
        List<Trade> trades = activeTrades.get(symbol);
        if (trades != null) {
            trades.remove(trade);
            if (trades.isEmpty()) {
                activeTrades.remove(symbol);
            }
        }
    }
    
    private TradingSignal getOppositeSignal(TradingSignal signal) {
        return signal == TradingSignal.BUY ? TradingSignal.SELL : TradingSignal.BUY;
    }
    
    private double calculateProfitLoss(Trade trade, double exitPrice) {
        double priceChange = exitPrice - trade.getEntryPrice();
        double multiplier = trade.getSignal() == TradingSignal.BUY ? 1 : -1;
        return priceChange * trade.getQuantity() * multiplier;
    }
}
