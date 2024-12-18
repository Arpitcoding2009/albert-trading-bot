package com.albert.trading.bot.ai;

import com.albert.trading.bot.model.MarketData;
import com.albert.trading.bot.model.TradingSignal;
import java.util.Arrays;
import java.util.Random;

public class TradingStrategy implements Cloneable {
    private static final int GENE_LENGTH = 100;
    private final double[] genes;
    private double fitness;
    private static final Random random = new Random();
    
    public TradingStrategy() {
        this.genes = new double[GENE_LENGTH];
        randomizeGenes();
        this.fitness = 0.0;
    }
    
    private void randomizeGenes() {
        for (int i = 0; i < GENE_LENGTH; i++) {
            genes[i] = random.nextGaussian();
        }
    }
    
    public TradingSignal predict(MarketData data) {
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
        
        // Add technical indicators
        double[] technicals = data.getTechnicalIndicators();
        if (technicals != null && technicals.length > 0) {
            int offset = GENE_LENGTH/2;
            for (int i = 0; i < Math.min(technicals.length, GENE_LENGTH/4); i++) {
                buySignal += technicals[i] * genes[i + offset];
                sellSignal += technicals[i] * genes[i + offset + GENE_LENGTH/4];
            }
        }
        
        // Decision threshold with adaptive sensitivity
        double threshold = 0.1 * Math.sqrt(getFitness() + 1.0);
        if (buySignal > threshold && buySignal > sellSignal * 1.1) {
            return TradingSignal.BUY;
        } else if (sellSignal > threshold && sellSignal > buySignal * 1.1) {
            return TradingSignal.SELL;
        }
        
        return TradingSignal.HOLD;
    }
    
    public void mutate(double mutationRate) {
        for (int i = 0; i < GENE_LENGTH; i++) {
            if (random.nextDouble() < mutationRate) {
                genes[i] += random.nextGaussian() * 0.1;
            }
        }
    }
    
    public TradingStrategy crossover(TradingStrategy other) {
        TradingStrategy child = new TradingStrategy();
        for (int i = 0; i < GENE_LENGTH; i++) {
            child.genes[i] = random.nextBoolean() ? this.genes[i] : other.genes[i];
        }
        return child;
    }
    
    public static int getGeneLength() {
        return GENE_LENGTH;
    }

    public double[] getGenes() {
        return Arrays.copyOf(genes, genes.length);
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
}
