package com.albert.trading.bot.ai;

import com.albert.trading.bot.PerformanceOptimizer.MarketData;
import com.albert.trading.bot.model.TradingSignal;

public class TradingStrategy implements Cloneable {
    private static final int GENE_LENGTH = 100;
    private final double[] genes;
    private double fitness;
    
    public TradingStrategy() {
        this.genes = new double[GENE_LENGTH];
        this.fitness = 0.0;
    }
    
    public TradingSignal predict(model.MarketData data) {
        double buySignal = 0.0;
        double sellSignal = 0.0;
        
        // Process price data
        double[] prices = data.getPrices();
        for (int i = 0; i < Math.min(prices.length, GENE_LENGTH/2); i++) {
            buySignal += prices[i] * genes[i];
            sellSignal += prices[i] * genes[i + GENE_LENGTH/2];
        }
        
        // Add volume influence
        double[] volumes = data.getVolumes();
        for (int i = 0; i < Math.min(volumes.length, GENE_LENGTH/4); i++) {
            buySignal += volumes[i] * genes[i + GENE_LENGTH/2];
            sellSignal += volumes[i] * genes[i + 3*GENE_LENGTH/4];
        }
        
        // Decision threshold
        double threshold = 0.1;
        if (buySignal > threshold && buySignal > sellSignal) {
            return TradingSignal.BUY;
        } else if (sellSignal > threshold && sellSignal > buySignal) {
            return TradingSignal.SELL;
        }
        
        return TradingSignal.HOLD;
    }
    
    public static int getGeneLength() {
        return GENE_LENGTH;
    }

    public double[] getGenes() {
        return genes;
    }
    
    public double getFitness() {
        return fitness;
    }
    
    public void setFitness(double fitness) {
        this.fitness = fitness;
    }
    
    @Override
    public TradingStrategy clone() {
        try {
            TradingStrategy clone = (TradingStrategy) super.clone();
            System.arraycopy(genes, 0, clone.genes, 0, genes.length);
            clone.fitness = this.fitness;
            return clone;
        } catch (CloneNotSupportedException e) {
            throw new RuntimeException("Clone failed", e);
        }
    }

    public TradingSignal predict(MarketData data) {
        throw new UnsupportedOperationException("Unimplemented method 'predict'");
    }
}
