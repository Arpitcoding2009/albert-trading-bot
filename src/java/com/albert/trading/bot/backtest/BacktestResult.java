package com.albert.trading.bot.backtest;

import com.albert.trading.bot.model.Trade;
import java.util.ArrayList;
import java.util.List;

public class BacktestResult {
    private final double totalReturn;
    private final int totalTrades;
    private final int winningTrades;
    private final double profitFactor;
    private final double maxDrawdown;
    private final List<Trade> trades;
    
    private BacktestResult(Builder builder) {
        this.totalReturn = builder.totalReturn;
        this.totalTrades = builder.totalTrades;
        this.winningTrades = builder.winningTrades;
        this.profitFactor = builder.profitFactor;
        this.maxDrawdown = builder.maxDrawdown;
        this.trades = new ArrayList<>(builder.trades);
    }
    
    public double getTotalReturn() {
        return totalReturn;
    }
    
    public int getTotalTrades() {
        return totalTrades;
    }
    
    public int getWinningTrades() {
        return winningTrades;
    }
    
    public double getProfitFactor() {
        return profitFactor;
    }
    
    public double getMaxDrawdown() {
        return maxDrawdown;
    }
    
    public List<Trade> getTrades() {
        return new ArrayList<>(trades);
    }
    
    public double getWinRate() {
        return totalTrades == 0 ? 0 : (double) winningTrades / totalTrades;
    }
    
    public double getSharpeRatio() {
        if (trades.isEmpty()) {
            return 0;
        }
        
        double meanReturn = totalReturn / trades.size();
        double variance = trades.stream()
            .mapToDouble(Trade::getProfitLoss)
            .map(r -> Math.pow(r - meanReturn, 2))
            .sum() / trades.size();
            
        return meanReturn / Math.sqrt(variance);
    }
    
    @Override
    public String toString() {
        return String.format("""
            Backtest Results:
            Total Return: %.2f%%
            Total Trades: %d
            Winning Trades: %d
            Win Rate: %.2f%%
            Profit Factor: %.2f
            Max Drawdown: %.2f%%
            Sharpe Ratio: %.2f
            """,
            totalReturn * 100,
            totalTrades,
            winningTrades,
            getWinRate() * 100,
            profitFactor,
            maxDrawdown * 100,
            getSharpeRatio());
    }
    
    public static class Builder {
        private double totalReturn;
        private int totalTrades;
        private int winningTrades;
        private double profitFactor;
        private double maxDrawdown;
        private final List<Trade> trades = new ArrayList<>();
        
        public Builder totalReturn(double totalReturn) {
            this.totalReturn = totalReturn;
            return this;
        }
        
        public Builder totalTrades(int totalTrades) {
            this.totalTrades = totalTrades;
            return this;
        }
        
        public Builder winningTrades(int winningTrades) {
            this.winningTrades = winningTrades;
            return this;
        }
        
        public Builder profitFactor(double profitFactor) {
            this.profitFactor = profitFactor;
            return this;
        }
        
        public Builder maxDrawdown(double maxDrawdown) {
            this.maxDrawdown = maxDrawdown;
            return this;
        }
        
        public Builder trades(List<Trade> trades) {
            this.trades.addAll(trades);
            return this;
        }
        
        public BacktestResult build() {
            return new BacktestResult(this);
        }
    }
}
