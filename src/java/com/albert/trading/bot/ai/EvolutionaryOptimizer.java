package com.albert.trading.bot.ai;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.logging.Logger;

import com.albert.trading.bot.model.MarketData;
import com.albert.trading.bot.model.TradingSignal;

public class EvolutionaryOptimizer {
    private static final Logger LOGGER = Logger.getLogger(EvolutionaryOptimizer.class.getName());
    
    private final ExecutorService evolutionExecutor;
    private final List<TradingStrategy> population;
    private final Random random;
    
    private static final int POPULATION_SIZE = 1000;
    private static final int GENERATIONS = 100;
    private static final double MUTATION_RATE = 0.01;
    private static final double CROSSOVER_RATE = 0.7;
    private static final double ELITE_RATIO = 0.1;
    
    public EvolutionaryOptimizer() {
        this.evolutionExecutor = Executors.newWorkStealingPool();
        this.population = new CopyOnWriteArrayList<>();
        this.random = new Random();
        initializePopulation();
    }
    
    private void initializePopulation() {
        for (int i = 0; i < POPULATION_SIZE; i++) {
            population.add(new TradingStrategy());
        }
        LOGGER.info("Initialized population with " + POPULATION_SIZE + " strategies");
    }
    
    public CompletableFuture<TradingStrategy> evolveStrategies(List<MarketData> historicalData) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                for (int gen = 0; gen < GENERATIONS; gen++) {
                    evaluatePopulation(historicalData);
                    List<TradingStrategy> newPopulation = new ArrayList<>();
                    
                    // Elitism - keep best performers
                    int eliteCount = (int)(POPULATION_SIZE * ELITE_RATIO);
                    population.sort(Comparator.comparingDouble(TradingStrategy::getFitness).reversed());
                    newPopulation.addAll(population.subList(0, eliteCount));
                    
                    // Generate rest of population through selection and crossover
                    while (newPopulation.size() < POPULATION_SIZE) {
                        TradingStrategy parent1 = selectParent();
                        TradingStrategy parent2 = selectParent();
                        
                        if (random.nextDouble() < CROSSOVER_RATE) {
                            TradingStrategy child = parent1.crossover(parent2);
                            child.mutate(MUTATION_RATE);
                            newPopulation.add(child);
                        } else {
                            newPopulation.add(parent1.clone());
                        }
                    }
                    
                    population.clear();
                    population.addAll(newPopulation);
                    
                    LOGGER.info(String.format("Generation %d complete. Best fitness: %.4f", 
                        gen, population.get(0).getFitness()));
                }
                
                return population.get(0); // Return best strategy
            } catch (Exception e) {
                LOGGER.severe("Evolution failed: " + e.getMessage());
                throw e;
            }
        }, evolutionExecutor);
    }
    
    private void evaluatePopulation(List<MarketData> historicalData) {
        population.parallelStream().forEach(strategy -> {
            double fitness = evaluateStrategy(strategy, historicalData);
            strategy.setFitness(fitness);
        });
    }
    
    private double evaluateStrategy(TradingStrategy strategy, List<MarketData> historicalData) {
        double totalReturn = 0.0;
        double position = 0.0;
        
        for (int i = 0; i < historicalData.size() - 1; i++) {
            MarketData currentData = historicalData.get(i);
            double currentPrice = currentData.getLatestPrice();
            
            TradingSignal signal = strategy.predict(currentData);
            
            // Simple trading simulation
            if (signal.isBuySignal() && position == 0.0) {
                position = 1.0; // Buy
            } else if (signal.isSellSignal() && position > 0.0) {
                double nextPrice = historicalData.get(i + 1).getLatestPrice();
                totalReturn += (nextPrice - currentPrice) * position;
                position = 0.0; // Sell
            }
        }
        
        // Add penalty for excessive trading
        int numTrades = countTrades(strategy, historicalData);
        double tradingPenalty = numTrades * 0.001; // 0.1% per trade
        
        return totalReturn - tradingPenalty;
    }
    
    private int countTrades(TradingStrategy strategy, List<MarketData> historicalData) {
        int trades = 0;
        boolean inPosition = false;
        
        for (MarketData data : historicalData) {
            TradingSignal signal = strategy.predict(data);
            if (signal.isBuySignal() && !inPosition) {
                trades++;
                inPosition = true;
            } else if (signal.isSellSignal() && inPosition) {
                trades++;
                inPosition = false;
            }
        }
        
        return trades;
    }
    
    private TradingStrategy selectParent() {
        // Tournament selection
        int tournamentSize = 5;
        TradingStrategy best = population.get(random.nextInt(population.size()));
        
        for (int i = 1; i < tournamentSize; i++) {
            TradingStrategy challenger = population.get(random.nextInt(population.size()));
            if (challenger.getFitness() > best.getFitness()) {
                best = challenger;
            }
        }
        
        return best;
    }
    
    public void shutdown() {
        evolutionExecutor.shutdown();
        try {
            if (!evolutionExecutor.awaitTermination(10, TimeUnit.SECONDS)) {
                evolutionExecutor.shutdownNow();
            }
        } catch (InterruptedException e) {
            evolutionExecutor.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }
}
