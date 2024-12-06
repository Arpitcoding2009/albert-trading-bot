package com.albert.trading.bot.ml;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import java.util.*;
import java.util.concurrent.*;
import java.util.logging.Logger;

public class NeuralPredictor {
    private static final Logger LOGGER = Logger.getLogger(NeuralPredictor.class.getName());
    
    private MultiLayerNetwork network;
    private final ExecutorService trainingExecutor;
    private final Map<String, Queue<TrainingData>> trainingBuffer;
    private volatile boolean isTraining = false;
    
    private static final int INPUT_SIZE = 1000;
    private static final int HIDDEN_SIZE = 500;
    private static final int OUTPUT_SIZE = 3; // Buy, Sell, Hold
    private static final int BATCH_SIZE = 64;
    private static final double LEARNING_RATE = 0.001;
    private static final int BUFFER_SIZE = 10000;
    
    public NeuralPredictor() {
        this.trainingExecutor = Executors.newSingleThreadExecutor();
        this.trainingBuffer = new ConcurrentHashMap<>();
        initializeNetwork();
        startAutonomousTraining();
    }
    
    private void initializeNetwork() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(123)
            .weightInit(WeightInit.XAVIER)
            .updater(new Adam(LEARNING_RATE))
            .list()
            .layer(new DenseLayer.Builder()
                .nIn(INPUT_SIZE)
                .nOut(HIDDEN_SIZE)
                .activation(Activation.RELU)
                .dropOut(0.2)
                .build())
            .layer(new LSTM.Builder()
                .nIn(HIDDEN_SIZE)
                .nOut(HIDDEN_SIZE)
                .activation(Activation.TANH)
                .build())
            .layer(new DenseLayer.Builder()
                .nIn(HIDDEN_SIZE)
                .nOut(HIDDEN_SIZE / 2)
                .activation(Activation.RELU)
                .dropOut(0.2)
                .build())
            .layer(new DenseLayer.Builder()
                .nIn(HIDDEN_SIZE / 2)
                .nOut(HIDDEN_SIZE / 4)
                .activation(Activation.RELU)
                .build())
            .layer(new OutputLayer.Builder()
                .nIn(HIDDEN_SIZE / 4)
                .nOut(OUTPUT_SIZE)
                .activation(Activation.SOFTMAX)
                .build())
            .build();
        
        network = new MultiLayerNetwork(conf);
        network.init();
        network.setListeners(new ScoreIterationListener(100));
        
        LOGGER.info("Neural network initialized with " + 
                   network.numParams() + " parameters");
    }
    
    public CompletableFuture<PredictionResult> predict(MarketData data) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                INDArray input = preprocessData(data);
                INDArray output = network.output(input, false);
                return interpretPrediction(output, data.getConfidence());
            } catch (Exception e) {
                LOGGER.severe("Prediction failed: " + e.getMessage());
                return new PredictionResult(TradingSignal.HOLD, 0.0);
            }
        });
    }
    
    private INDArray preprocessData(MarketData data) {
        // Convert market data to neural network input format
        INDArray input = Nd4j.zeros(1, INPUT_SIZE);
        
        // Price data (normalized)
        double[] prices = data.getPrices();
        for (int i = 0; i < Math.min(prices.length, 100); i++) {
            input.putScalar(i, normalizePrice(prices[i]));
        }
        
        // Volume data
        double[] volumes = data.getVolumes();
        for (int i = 0; i < Math.min(volumes.length, 100); i++) {
            input.putScalar(100 + i, normalizeVolume(volumes[i]));
        }
        
        // Technical indicators
        Map<String, Double> indicators = data.getTechnicalIndicators();
        int offset = 200;
        for (Map.Entry<String, Double> entry : indicators.entrySet()) {
            input.putScalar(offset++, normalizeIndicator(entry.getValue()));
        }
        
        // Sentiment data
        double sentiment = data.getSentiment();
        input.putScalar(offset, normalizeSentiment(sentiment));
        
        return input;
    }
    
    private PredictionResult interpretPrediction(INDArray output, double confidence) {
        double[] probabilities = output.toDoubleVector();
        int maxIndex = 0;
        double maxProb = probabilities[0];
        
        for (int i = 1; i < probabilities.length; i++) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i];
                maxIndex = i;
            }
        }
        
        TradingSignal signal = TradingSignal.HOLD;
        switch (maxIndex) {
            case 0:
                signal = TradingSignal.BUY;
                break;
            case 1:
                signal = TradingSignal.SELL;
                break;
            case 2:
                signal = TradingSignal.HOLD;
                break;
        }
        
        double adjustedConfidence = maxProb * confidence;
        return new PredictionResult(signal, adjustedConfidence);
    }
    
    public void addTrainingData(String ticker, TrainingData data) {
        Queue<TrainingData> buffer = trainingBuffer.computeIfAbsent(
            ticker, k -> new ConcurrentLinkedQueue<>()
        );
        
        buffer.offer(data);
        while (buffer.size() > BUFFER_SIZE) {
            buffer.poll();
        }
    }
    
    private void startAutonomousTraining() {
        ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(1);
        scheduler.scheduleAtFixedRate(() -> {
            if (!isTraining && !trainingBuffer.isEmpty()) {
                trainOnBuffer();
            }
        }, 0, 5, TimeUnit.MINUTES);
    }
    
    private void trainOnBuffer() {
        isTraining = true;
        try {
            List<TrainingData> batchData = new ArrayList<>();
            trainingBuffer.values().forEach(buffer -> {
                buffer.stream()
                    .limit(BATCH_SIZE)
                    .forEach(batchData::add);
            });
            
            if (!batchData.isEmpty()) {
                INDArray input = Nd4j.zeros(batchData.size(), INPUT_SIZE);
                INDArray labels = Nd4j.zeros(batchData.size(), OUTPUT_SIZE);
                
                for (int i = 0; i < batchData.size(); i++) {
                    TrainingData data = batchData.get(i);
                    input.putRow(i, preprocessData(data.getMarketData()));
                    labels.putRow(i, createLabel(data.getActualOutcome()));
                }
                
                network.fit(input, labels);
                LOGGER.info("Completed training batch with " + 
                           batchData.size() + " samples");
            }
        } catch (Exception e) {
            LOGGER.severe("Training failed: " + e.getMessage());
        } finally {
            isTraining = false;
        }
    }
    
    private INDArray createLabel(TradingSignal signal) {
        INDArray label = Nd4j.zeros(OUTPUT_SIZE);
        switch (signal) {
            case BUY:
                label.putScalar(0, 1.0);
                break;
            case SELL:
                label.putScalar(1, 1.0);
                break;
            case HOLD:
                label.putScalar(2, 1.0);
                break;
        }
        return label;
    }
    
    private double normalizePrice(double price) {
        // Z-score normalization
        return (price - 100) / 50.0;
    }
    
    private double normalizeVolume(double volume) {
        // Log normalization
        return Math.log1p(volume) / 10.0;
    }
    
    private double normalizeIndicator(double value) {
        // Min-max normalization
        return (value + 1.0) / 2.0;
    }
    
    private double normalizeSentiment(double sentiment) {
        // Already normalized between -1 and 1
        return (sentiment + 1.0) / 2.0;
    }
    
    public void shutdown() {
        trainingExecutor.shutdown();
        try {
            if (!trainingExecutor.awaitTermination(60, TimeUnit.SECONDS)) {
                trainingExecutor.shutdownNow();
            }
        } catch (InterruptedException e) {
            trainingExecutor.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }
}
