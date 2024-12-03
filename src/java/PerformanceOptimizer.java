package com.albert.trading.bot;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.logging.Logger;
import java.util.logging.Level;

/**
 * Advanced Performance Optimizer for Quantum Trading Intelligence
 * Provides high-performance statistical and technical analysis methods
 */
public class PerformanceOptimizer {
    
    // Logger for tracking and debugging
    private static final Logger LOGGER = Logger.getLogger(PerformanceOptimizer.class.getName());
    
    // Prevent instantiation
    private PerformanceOptimizer() {
        throw new AssertionError("Cannot instantiate utility class");
    }
    
    /**
     * Calculate Moving Average with robust error handling and quantum optimization
     * @param data Input price data
     * @param period Moving average period
     * @return List of moving average values
     * @throws IllegalArgumentException for invalid inputs
     */
    public static List<Double> calculateQuantumMovingAverage(List<Double> data, int period) {
        try {
            validateInputs(data, period);
            
            List<Double> result = new ArrayList<>(data.size() - period + 1);
            double[] window = new double[period];
            double sum = 0.0;
            
            // Preload initial window
            for (int i = 0; i < period; i++) {
                window[i] = data.get(i);
                sum += data.get(i);
            }
            result.add(sum / period);
            
            // Sliding window with minimal computation
            for (int i = period; i < data.size(); i++) {
                sum = sum - window[(i - period) % period] + data.get(i);
                window[i % period] = data.get(i);
                result.add(sum / period);
            }
            
            return result;
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "Error in quantum moving average calculation", e);
            throw e;
        }
    }
    
    /**
     * Advanced Relative Strength Index (RSI) with quantum-inspired calculation
     * @param prices Historical price data
     * @param period RSI calculation period
     * @return Normalized RSI value with quantum smoothing
     */
    public static double calculateQuantumRSI(List<Double> prices, int period) {
        try {
            validateInputs(prices, period);
            
            double[] gains = new double[prices.size()];
            double[] losses = new double[prices.size()];
            
            // Vectorized change calculation
            for (int i = 1; i < prices.size(); i++) {
                double change = prices.get(i) - prices.get(i - 1);
                gains[i] = Math.max(change, 0);
                losses[i] = Math.max(-change, 0);
            }
            
            // Quantum-smoothed exponential moving averages
            double avgGain = calculateSmoothedAverage(gains, period);
            double avgLoss = calculateSmoothedAverage(losses, period);
            
            // Advanced RSI normalization
            return avgLoss == 0 ? 100.0 : 100.0 - (100.0 / (1.0 + (avgGain / avgLoss)));
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "Error in quantum RSI calculation", e);
            throw e;
        }
    }
    
    /**
     * Quantum-smoothed exponential moving average
     * @param data Input data array
     * @param period Smoothing period
     * @return Smoothed average value
     */
    private static double calculateSmoothedAverage(double[] data, int period) {
        try {
            double smoothingFactor = 2.0 / (period + 1.0);
            double smoothedValue = 0.0;
            
            for (int i = 1; i < data.length; i++) {
                if (i < period) {
                    smoothedValue += data[i];
                } else {
                    smoothedValue = (data[i] - smoothedValue) * smoothingFactor + smoothedValue;
                }
            }
            
            return smoothedValue / period;
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "Error in quantum smoothed average calculation", e);
            throw e;
        }
    }
    
    /**
     * Validate input data for statistical calculations with quantum-level checks
     * @param data Input data list
     * @param period Calculation period
     * @throws IllegalArgumentException for invalid inputs
     */
    private static void validateInputs(List<Double> data, int period) {
        try {
            Objects.requireNonNull(data, "Quantum input data cannot be null");
            if (data.isEmpty()) {
                throw new IllegalArgumentException("Quantum input data cannot be empty");
            }
            if (period <= 0 || period > data.size()) {
                throw new IllegalArgumentException("Invalid quantum period: " + period);
            }
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "Error in input validation", e);
            throw e;
        }
    }
    
    /**
     * Quantum volatility calculation
     * @param prices Historical price data
     * @return Volatility metric
     */
    public static double calculateQuantumVolatility(List<Double> prices) {
        try {
            validateInputs(prices, 1);
            
            double mean = prices.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
            double variance = prices.stream()
                .mapToDouble(price -> Math.pow(price - mean, 2))
                .average()
                .orElse(0.0);
            
            return Math.sqrt(variance);
        } catch (Exception e) {
            LOGGER.log(Level.SEVERE, "Error in quantum volatility calculation", e);
            throw e;
        }
    }
}
