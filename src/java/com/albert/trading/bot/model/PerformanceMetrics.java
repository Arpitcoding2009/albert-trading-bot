package com.albert.trading.bot.model;

import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

public class PerformanceMetrics {
    private final double totalReturn;
    private final double sharpeRatio;
    private final double maxDrawdown;
    private final int totalTrades;
    private final int winningTrades;
    private final int losingTrades;
    private final double winRate;
    private final double averageWin;
    private final double averageLoss;
    private final double profitFactor;
    private final List<Trade> recentTrades;
    private final Instant lastUpdated;

    private PerformanceMetrics(Builder builder) {
        this.totalReturn = builder.totalReturn;
        this.sharpeRatio = builder.sharpeRatio;
        this.maxDrawdown = builder.maxDrawdown;
        this.totalTrades = builder.totalTrades;
        this.winningTrades = builder.winningTrades;
        this.losingTrades = builder.losingTrades;
        this.winRate = builder.winRate;
        this.averageWin = builder.averageWin;
        this.averageLoss = builder.averageLoss;
        this.profitFactor = builder.profitFactor;
        this.recentTrades = new ArrayList<>(builder.recentTrades);
        this.lastUpdated = builder.lastUpdated;
    }

    public double getTotalReturn() {
        return totalReturn;
    }

    public double getSharpeRatio() {
        return sharpeRatio;
    }

    public double getMaxDrawdown() {
        return maxDrawdown;
    }

    public int getTotalTrades() {
        return totalTrades;
    }

    public int getWinningTrades() {
        return winningTrades;
    }

    public int getLosingTrades() {
        return losingTrades;
    }

    public double getWinRate() {
        return winRate;
    }

    public double getAverageWin() {
        return averageWin;
    }

    public double getAverageLoss() {
        return averageLoss;
    }

    public double getProfitFactor() {
        return profitFactor;
    }

    public List<Trade> getRecentTrades() {
        return new ArrayList<>(recentTrades);
    }

    public Instant getLastUpdated() {
        return lastUpdated;
    }

    public static class Builder {
        private double totalReturn;
        private double sharpeRatio;
        private double maxDrawdown;
        private int totalTrades;
        private int winningTrades;
        private int losingTrades;
        private double winRate;
        private double averageWin;
        private double averageLoss;
        private double profitFactor;
        private final List<Trade> recentTrades = new ArrayList<>();
        private Instant lastUpdated = Instant.now();

        public Builder totalReturn(double totalReturn) {
            this.totalReturn = totalReturn;
            return this;
        }

        public Builder sharpeRatio(double sharpeRatio) {
            this.sharpeRatio = sharpeRatio;
            return this;
        }

        public Builder maxDrawdown(double maxDrawdown) {
            this.maxDrawdown = maxDrawdown;
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

        public Builder losingTrades(int losingTrades) {
            this.losingTrades = losingTrades;
            return this;
        }

        public Builder winRate(double winRate) {
            this.winRate = winRate;
            return this;
        }

        public Builder averageWin(double averageWin) {
            this.averageWin = averageWin;
            return this;
        }

        public Builder averageLoss(double averageLoss) {
            this.averageLoss = averageLoss;
            return this;
        }

        public Builder profitFactor(double profitFactor) {
            this.profitFactor = profitFactor;
            return this;
        }

        public Builder addRecentTrade(Trade trade) {
            this.recentTrades.add(trade);
            return this;
        }

        public Builder lastUpdated(Instant lastUpdated) {
            this.lastUpdated = lastUpdated;
            return this;
        }

        public PerformanceMetrics build() {
            return new PerformanceMetrics(this);
        }
    }
}
