package com.albert.trading.bot;

import com.albert.trading.bot.model.StrategyConfig;
import java.util.HashMap;
import java.util.Map;

public class Main {
    public static void main(String[] args) {
        // Create strategy configurations
        Map<String, StrategyConfig> strategyConfigs = new HashMap<>();
        
        // Configure BTC/USD trading strategy
        strategyConfigs.put("BTC/USD", new StrategyConfig.Builder("ML_STRATEGY")
            .riskLevel(0.02)  // 2% risk per trade
            .minConfidence(0.75)
            .timeframe(300)   // 5 minutes
            .addParameter("stopLoss", 0.02)
            .addParameter("takeProfit", 0.04)
            .addFlag("useML", true)
            .addFlag("useSentiment", true)
            .addSetting("exchange", "binance")
            .build());
            
        // Configure ETH/USD trading strategy
        strategyConfigs.put("ETH/USD", new StrategyConfig.Builder("TREND_FOLLOWING")
            .riskLevel(0.015) // 1.5% risk per trade
            .minConfidence(0.8)
            .timeframe(600)   // 10 minutes
            .addParameter("stopLoss", 0.025)
            .addParameter("takeProfit", 0.05)
            .addFlag("useML", true)
            .addFlag("useSentiment", true)
            .addSetting("exchange", "binance")
            .build());
            
        // Create and start the trading bot
        TradingBot bot = new TradingBot("Albert", 10000.0, strategyConfigs);
        
        try {
            System.out.println("Starting trading bot...");
            bot.start();
            
            // Add shutdown hook for graceful shutdown
            Runtime.getRuntime().addShutdownHook(new Thread(() -> {
                System.out.println("Shutting down trading bot...");
                bot.stop();
            }));
            
            // Keep the main thread alive
            while (bot.isRunning()) {
                Thread.sleep(1000);
            }
        } catch (Exception e) {
            System.err.println("Error running trading bot: " + e.getMessage());
            bot.stop();
        }
    }
}
