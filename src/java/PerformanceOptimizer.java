package com.albert.trading;

import java.util.ArrayList;
import java.util.List;

public class PerformanceOptimizer {
    public static List<Double> calculateMovingAverage(List<Double> data, int period) {
        List<Double> result = new ArrayList<>();
        if (period <= 0 || data.size() < period) {
            return result;
        }

        double sum = 0.0;
        // Calculate initial sum for the first window
        for (int i = 0; i < period; i++) {
            sum += data.get(i);
        }

        // First moving average
        result.add(sum / period);

        // Slide the window
        for (int i = period; i < data.size(); i++) {
            sum = sum - data.get(i - period) + data.get(i);
            result.add(sum / period);
        }

        return result;
    }

    public static double calculateRSI(List<Double> prices, int period) {
        List<Double> gains = new ArrayList<>();
        List<Double> losses = new ArrayList<>();

        for (int i = 1; i < prices.size(); i++) {
            double change = prices.get(i) - prices.get(i - 1);
            if (change >= 0) {
                gains.add(change);
                losses.add(0.0);
            } else {
                gains.add(0.0);
                losses.add(Math.abs(change));
            }
        }

        double avgGain = calculateAverage(gains.subList(0, period));
        double avgLoss = calculateAverage(losses.subList(0, period));

        return 100.0 - (100.0 / (1.0 + (avgGain / avgLoss)));
    }

    private static double calculateAverage(List<Double> values) {
        return values.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
    }
}
