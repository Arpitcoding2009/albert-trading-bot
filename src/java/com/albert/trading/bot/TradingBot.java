package com.albert.trading.bot;

import com.albert.trading.bot.ai.EvolutionaryOptimizer;
import com.albert.trading.bot.ai.TradingStrategy;
import com.albert.trading.bot.model.*;
import com.albert.trading.bot.ml.DeepLearningModel;
import com.albert.trading.bot.trading.*;
import com.albert.trading.bot.sentiment.SentimentAnalyzer;

import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

public class TradingBot {
    private final String name;
    private final MarketAnalyzer marketAnalyzer;
    private final OrderManager orderManager;
    private final PortfolioManager portfolioManager;
    private final RiskManager riskManager;
    private final DeepLearningModel deepLearningModel;
    private final SentimentAnalyzer sentimentAnalyzer;
    private final EvolutionaryOptimizer evolutionaryOptimizer;
    private final PerformanceOptimizer performanceOptimizer;
    private final Map<String, StrategyConfig> strategyConfigs;
    private final ScheduledExecutorService scheduler;
    private volatile boolean isRunning;
    
    public TradingBot(String name, double initialBalance, Map<String, StrategyConfig> strategyConfigs) {
        this.name = name;
        this.strategyConfigs = strategyConfigs;
        this.scheduler = Executors.newScheduledThreadPool(4);
        
        // Initialize components
        this.riskManager = new RiskManager();
        this.deepLearningModel = new DeepLearningModel();
        this.sentimentAnalyzer = new SentimentAnalyzer();
        this.evolutionaryOptimizer = new EvolutionaryOptimizer();
        this.performanceOptimizer = new PerformanceOptimizer();
        
        TradeExecutor tradeExecutor = new TradeExecutor(riskManager);
        this.orderManager = new OrderManager(riskManager, tradeExecutor);
        this.portfolioManager = new PortfolioManager(initialBalance, riskManager);
        this.marketAnalyzer = new MarketAnalyzer(deepLearningModel, sentimentAnalyzer);
    }
    
    public void start() {
        if (isRunning) {
            return;
        }
        isRunning = true;
        
        // Schedule market analysis and trading tasks
        for (Map.Entry<String, StrategyConfig> entry : strategyConfigs.entrySet()) {
            String symbol = entry.getKey();
            StrategyConfig config = entry.getValue();
            
            // Schedule market analysis
            scheduler.scheduleAtFixedRate(() -> analyzeMarket(symbol, config),
                0, config.getTimeframe(), TimeUnit.SECONDS);
                
            // Schedule portfolio optimization
            scheduler.scheduleAtFixedRate(() -> optimizePortfolio(symbol),
                60, 3600, TimeUnit.SECONDS);
        }
        
        // Schedule performance tracking
        scheduler.scheduleAtFixedRate(this::trackPerformance,
            300, 300, TimeUnit.SECONDS);
    }
    
    public void stop() {
        isRunning = false;
        scheduler.shutdown();
        try {
            if (!scheduler.awaitTermination(60, TimeUnit.SECONDS)) {
                scheduler.shutdownNow();
            }
        } catch (InterruptedException e) {
            scheduler.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }
    
    private void analyzeMarket(String symbol, StrategyConfig config) {
        try {
            CompletableFuture.allOf(
                marketAnalyzer.analyzeTechnicals(symbol),
                marketAnalyzer.analyzeSentiment(symbol)
            ).thenAccept(v -> {
                MarketInsights insights = marketAnalyzer.getMarketInsights(symbol);
                if (shouldTrade(insights, config)) {
                    executeTrade(symbol, insights, config);
                }
            }).exceptionally(e -> {
                System.err.println("Error analyzing market for " + symbol + ": " + e.getMessage());
                return null;
            });
        } catch (Exception e) {
            System.err.println("Error in market analysis for " + symbol + ": " + e.getMessage());
        }
    }
    
    private boolean shouldTrade(MarketInsights insights, StrategyConfig config) {
        return insights.getConfidence() >= config.getMinConfidence() &&
            !orderManager.hasActiveTradesForSymbol(insights.getSymbol());
    }
    
    private void executeTrade(String symbol, MarketInsights insights, StrategyConfig config) {
        try {
            double quantity = riskManager.calculatePositionSize(symbol, 
                insights.getCurrentPrice(), config.getRiskLevel());
                
            orderManager.placeOrder(symbol, insights.getRecommendedAction(),
                insights.getCurrentPrice(), quantity, config.getStrategyName(),
                insights.getConfidence())
                .thenAccept(portfolioManager::updatePosition)
                .exceptionally(e -> {
                    System.err.println("Error executing trade for " + symbol + ": " + e.getMessage());
                    return null;
                });
        } catch (Exception e) {
            System.err.println("Error in trade execution for " + symbol + ": " + e.getMessage());
        }
    }
    
    private void optimizePortfolio(String symbol) {
        try {
            TradingStrategy strategy = evolutionaryOptimizer.optimizeStrategy(
                symbol, portfolioManager.getTradeHistory());
            performanceOptimizer.updateStrategy(symbol, strategy);
        } catch (Exception e) {
            System.err.println("Error optimizing portfolio for " + symbol + ": " + e.getMessage());
        }
    }
    
    private void trackPerformance() {
        try {
            PerformanceMetrics metrics = portfolioManager.calculatePerformanceMetrics();
            System.out.printf("Performance Update for %s:%n", name);
            System.out.printf("Total Return: %.2f%%%n", metrics.getTotalReturn() * 100);
            System.out.printf("Win Rate: %.2f%%%n", metrics.getWinRate() * 100);
            System.out.printf("Profit Factor: %.2f%n", metrics.getProfitFactor());
        } catch (Exception e) {
            System.err.println("Error tracking performance: " + e.getMessage());
        }
    }
    
    public String getName() {
        return name;
    }
    
    public PortfolioManager getPortfolioManager() {
        return portfolioManager;
    }
    
    public boolean isRunning() {
        return isRunning;
    }
}
