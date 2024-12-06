package com.albert.trading.bot.ai;

import com.albert.trading.bot.model.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.logging.Logger;

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
            population.add(createRandomStrategy());
        }
        LOGGER.info("Initialized population with " + POPULATION_SIZE + " strategies");
    }
    
    public CompletableFuture<TradingStrategy> evolveStrategies(List<MarketData> historicalData) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                for (int gen = 0; gen < GENERATIONS; gen++) {
                    evaluatePopulation(historicalData);
                    List<TradingStrategy> newPopulation = new ArrayList<>();
                    
                    // Elitism
                    int eliteCount = (int) (POPULATION_SIZE * ELITE_RATIO);
                    population.stream()
                        .sorted((s1, s2) -> Double.compare(s2.getFitness(), s1.getFitness()))
                        .limit(eliteCount)
                        .forEach(newPopulation::add);
                    
                    // Generate new strategies
                    while (newPopulation.size() < POPULATION_SIZE) {
                        TradingStrategy parent1 = selectParent();
                        TradingStrategy parent2 = selectParent();
                        
                        if (random.nextDouble() < CROSSOVER_RATE) {
                            TradingStrategy child = crossover(parent1, parent2);
                            if (random.nextDouble() < MUTATION_RATE) {
                                mutate(child);
                            }
                            newPopulation.add(child);
                        } else {
                            newPopulation.add(parent1.clone());
                        }
                    }
                    
                    population.clear();
                    population.addAll(newPopulation);
                    
                    LOGGER.info("Generation " + gen + " complete. Best fitness: " + 
                              getBestStrategy().getFitness());
                }
                
                return getBestStrategy();
            } catch (Exception e) {
                LOGGER.severe("Evolution failed: " + e.getMessage());
                return createRandomStrategy();
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
        double initialBalance = 10000.0;
        double balance = initialBalance;
        int wins = 0;
        int trades = 0;
        
        for (int i = 0; i < historicalData.size() - 1; i++) {
            TradingSignal signal = strategy.predict(historicalData.get(i));
            if (signal != TradingSignal.HOLD) {
                trades++;
                double nextPrice = historicalData.get(i + 1).getPrices()[0];
                double currentPrice = historicalData.get(i).getPrices()[0];
                
                double profit = signal == TradingSignal.BUY ? 
                    (nextPrice - currentPrice) : (currentPrice - nextPrice);
                
                if (profit > 0) wins++;
                balance += profit;
            }
        }
        
        double returnRatio = (balance - initialBalance) / initialBalance;
        double winRate = trades > 0 ? (double) wins / trades : 0;
        double sharpeRatio = calculateSharpeRatio(strategy, historicalData);
        
        return returnRatio * 0.4 + winRate * 0.3 + sharpeRatio * 0.3;
    }
    
    private double calculateSharpeRatio(TradingStrategy strategy, List<MarketData> historicalData) {
        List<Double> returns = new ArrayList<>();
        
        for (int i = 0; i < historicalData.size() - 1; i++) {
            TradingSignal signal = strategy.predict(historicalData.get(i));
            if (signal != TradingSignal.HOLD) {
                double nextPrice = historicalData.get(i + 1).getPrices()[0];
                double currentPrice = historicalData.get(i).getPrices()[0];
                double return_ = (nextPrice - currentPrice) / currentPrice;
                returns.add(return_);
            }
        }
        
        if (returns.isEmpty()) return 0.0;
        
        double meanReturn = returns.stream()
            .mapToDouble(Double::doubleValue)
            .average()
            .orElse(0.0);
            
        double stdDev = Math.sqrt(returns.stream()
            .mapToDouble(r -> Math.pow(r - meanReturn, 2))
            .average()
            .orElse(0.0));
            
        return stdDev > 0 ? meanReturn / stdDev : 0.0;
    }
    
    private TradingStrategy selectParent() {
        // Tournament selection
        int tournamentSize = 5;
        TradingStrategy best = null;
        double bestFitness = Double.NEGATIVE_INFINITY;
        
        for (int i = 0; i < tournamentSize; i++) {
            TradingStrategy candidate = population.get(random.nextInt(population.size()));
            if (best == null || candidate.getFitness() > bestFitness) {
                best = candidate;
                bestFitness = candidate.getFitness();
            }
        }
        
        return best;
    }
    
    private TradingStrategy crossover(TradingStrategy parent1, TradingStrategy parent2) {
        TradingStrategy child = new TradingStrategy();
        
        // Uniform crossover
        for (int i = 0; i < parent1.getGenes().length; i++) {
            child.getGenes()[i] = random.nextBoolean() ? 
                parent1.getGenes()[i] : parent2.getGenes()[i];
        }
        
        return child;
    }
    
    private void mutate(TradingStrategy strategy) {
        // Gaussian mutation
        for (int i = 0; i < strategy.getGenes().length; i++) {
            if (random.nextDouble() < MUTATION_RATE) {
                strategy.getGenes()[i] += random.nextGaussian() * 0.1;
            }
        }
    }
    
    private TradingStrategy createRandomStrategy() {
        TradingStrategy strategy = new TradingStrategy();
        for (int i = 0; i < strategy.getGenes().length; i++) {
            strategy.getGenes()[i] = random.nextDouble() * 2 - 1;
        }
        return strategy;
    }
    
    private TradingStrategy getBestStrategy() {
        return population.stream()
            .max(Comparator.comparingDouble(TradingStrategy::getFitness))
            .orElse(createRandomStrategy());
    }
    
    public void shutdown() {
        evolutionExecutor.shutdown();
        try {
            if (!evolutionExecutor.awaitTermination(60, TimeUnit.SECONDS)) {
                evolutionExecutor.shutdownNow();
            }
        } catch (InterruptedException e) {
            evolutionExecutor.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }
}
