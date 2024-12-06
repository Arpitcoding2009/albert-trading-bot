package com.albert.trading.bot.risk;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicDouble;
import java.util.logging.Logger;
import java.time.Instant;

public class RiskManager {
    private static final Logger LOGGER = Logger.getLogger(RiskManager.class.getName());
    
    private final Map<String, AtomicDouble> positionSizes = new ConcurrentHashMap<>();
    private final Map<String, List<Double>> profitHistory = new ConcurrentHashMap<>();
    private final Map<String, Double> volatilityMetrics = new ConcurrentHashMap<>();
    
    private volatile double riskMultiplier = 1.0;
    private volatile double maxDrawdown = 0.10; // 10%
    private volatile double baseTradeAmount = 1000.0; // Base amount in USD
    private volatile double maxLeverage = 3.0;
    
    private static final int PROFIT_HISTORY_SIZE = 1000;
    private static final double RISK_FREE_RATE = 0.02; // 2% annual risk-free rate
    private static final double MIN_SHARPE_RATIO = 2.0;
    
    public RiskManager() {
        initializeRiskParameters();
    }
    
    private void initializeRiskParameters() {
        LOGGER.info("Initializing risk parameters with conservative settings");
        updateVolatilityMetrics();
        startRiskMonitoring();
    }
    
    public boolean validateTrade(TradeSignal signal) {
        if (!checkPositionLimits(signal)) return false;
        if (!checkDrawdownLimits(signal)) return false;
        if (!checkVolatilityLimits(signal)) return false;
        if (!checkLeverageLimits(signal)) return false;
        
        return true;
    }
    
    private boolean checkPositionLimits(TradeSignal signal) {
        AtomicDouble currentPosition = positionSizes.computeIfAbsent(
            signal.getTicker(), k -> new AtomicDouble(0.0)
        );
        
        double newPosition = currentPosition.get() + signal.getAmount();
        double maxPosition = calculateMaxPosition(signal.getTicker());
        
        if (newPosition > maxPosition) {
            LOGGER.warning("Position limit exceeded for " + signal.getTicker());
            return false;
        }
        
        return true;
    }
    
    private double calculateMaxPosition(String ticker) {
        double volatility = volatilityMetrics.getOrDefault(ticker, 1.0);
        return baseTradeAmount * riskMultiplier / volatility;
    }
    
    private boolean checkDrawdownLimits(TradeSignal signal) {
        List<Double> history = profitHistory.getOrDefault(signal.getTicker(), new ArrayList<>());
        if (history.isEmpty()) return true;
        
        double currentDrawdown = calculateDrawdown(history);
        if (currentDrawdown > maxDrawdown) {
            LOGGER.warning("Drawdown limit exceeded for " + signal.getTicker());
            return false;
        }
        
        return true;
    }
    
    private double calculateDrawdown(List<Double> history) {
        if (history.isEmpty()) return 0.0;
        
        double peak = history.stream().mapToDouble(d -> d).max().orElse(0.0);
        double current = history.get(history.size() - 1);
        
        return peak > 0 ? (peak - current) / peak : 0.0;
    }
    
    private boolean checkVolatilityLimits(TradeSignal signal) {
        double volatility = volatilityMetrics.getOrDefault(signal.getTicker(), 1.0);
        double volAdjustedAmount = signal.getAmount() * volatility;
        
        if (volAdjustedAmount > baseTradeAmount * riskMultiplier) {
            LOGGER.warning("Volatility limit exceeded for " + signal.getTicker());
            return false;
        }
        
        return true;
    }
    
    private boolean checkLeverageLimits(TradeSignal signal) {
        double currentLeverage = calculateCurrentLeverage(signal.getTicker());
        if (currentLeverage > maxLeverage) {
            LOGGER.warning("Leverage limit exceeded for " + signal.getTicker());
            return false;
        }
        
        return true;
    }
    
    private double calculateCurrentLeverage(String ticker) {
        AtomicDouble position = positionSizes.get(ticker);
        return position != null ? position.get() / baseTradeAmount : 0.0;
    }
    
    public void updateTradeHistory(String ticker, double profit) {
        List<Double> history = profitHistory.computeIfAbsent(
            ticker, k -> new ArrayList<>()
        );
        
        history.add(profit);
        while (history.size() > PROFIT_HISTORY_SIZE) {
            history.remove(0);
        }
        
        updateVolatilityMetrics();
    }
    
    private void updateVolatilityMetrics() {
        profitHistory.forEach((ticker, history) -> {
            if (history.size() >= 2) {
                double volatility = calculateVolatility(history);
                volatilityMetrics.put(ticker, volatility);
            }
        });
    }
    
    private double calculateVolatility(List<Double> history) {
        if (history.size() < 2) return 1.0;
        
        double mean = history.stream().mapToDouble(d -> d).average().orElse(0.0);
        double variance = history.stream()
            .mapToDouble(d -> Math.pow(d - mean, 2))
            .average()
            .orElse(0.0);
            
        return Math.sqrt(variance);
    }
    
    private void startRiskMonitoring() {
        Timer timer = new Timer(true);
        timer.scheduleAtFixedRate(new TimerTask() {
            @Override
            public void run() {
                try {
                    adjustRiskParameters();
                    cleanupOldData();
                } catch (Exception e) {
                    LOGGER.severe("Error in risk monitoring: " + e.getMessage());
                }
            }
        }, 0, 60000); // Every minute
    }
    
    private void adjustRiskParameters() {
        double overallSharpeRatio = calculateOverallSharpeRatio();
        
        if (overallSharpeRatio > MIN_SHARPE_RATIO) {
            riskMultiplier = Math.min(riskMultiplier * 1.1, 2.0);
            LOGGER.info("Increased risk multiplier to " + riskMultiplier);
        } else {
            riskMultiplier = Math.max(riskMultiplier * 0.9, 0.5);
            LOGGER.info("Decreased risk multiplier to " + riskMultiplier);
        }
    }
    
    private double calculateOverallSharpeRatio() {
        double totalReturn = profitHistory.values().stream()
            .flatMap(List::stream)
            .mapToDouble(d -> d)
            .average()
            .orElse(0.0);
            
        double volatility = volatilityMetrics.values().stream()
            .mapToDouble(d -> d)
            .average()
            .orElse(1.0);
            
        return volatility > 0 ? 
            (totalReturn - RISK_FREE_RATE) / volatility : 0.0;
    }
    
    private void cleanupOldData() {
        Instant cutoff = Instant.now().minusSeconds(86400); // 24 hours
        
        profitHistory.values().forEach(history -> {
            while (history.size() > PROFIT_HISTORY_SIZE) {
                history.remove(0);
            }
        });
    }
    
    public void setRiskMultiplier(double multiplier) {
        this.riskMultiplier = multiplier;
    }
    
    public double getBaseTradeAmount() {
        return baseTradeAmount * riskMultiplier;
    }
    
    public void setMaxDrawdown(double maxDrawdown) {
        this.maxDrawdown = maxDrawdown;
    }
    
    public void setMaxLeverage(double maxLeverage) {
        this.maxLeverage = maxLeverage;
    }
    
    public Map<String, Double> getRiskMetrics() {
        Map<String, Double> metrics = new HashMap<>();
        metrics.put("risk_multiplier", riskMultiplier);
        metrics.put("max_drawdown", maxDrawdown);
        metrics.put("max_leverage", maxLeverage);
        metrics.put("sharpe_ratio", calculateOverallSharpeRatio());
        return metrics;
    }
}
