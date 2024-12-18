package com.albert.trading.bot.ml;

import java.util.logging.Logger;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.util.ModelSerializer;
import org.deeplearning4j.nn.api.Model;
import java.io.File;
import java.io.IOException;
import org.nd4j.linalg.dataset.DataSet;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class DeepLearningModel implements AutoCloseable {
    private static final Logger LOGGER = Logger.getLogger(DeepLearningModel.class.getName());
    private MultiLayerNetwork network;
    private static final String MODEL_PATH = "models/pretrained_crypto_model.zip";

    public DeepLearningModel() {
        initializeNetwork();
    }

    public DeepLearningModel(MultiLayerNetwork network) {
        this.network = network;
    }

    private void initializeNetwork() {
        try {
            File modelFile = new File(MODEL_PATH);
            if (modelFile.exists()) {
                loadPreTrainedModel();
            } else {
                createNewNetwork();
            }
        } catch (Exception e) {
            LOGGER.severe("Failed to initialize network: " + e.getMessage());
            createNewNetwork();
        }
    }

    private void createNewNetwork() {
        try {
            int numInputs = 10;  // Features: price, volume, technical indicators, etc.
            int numOutputs = 3;  // Buy, Sell, Hold
            int numHiddenNodes = 50;

            network = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
                .seed(123)
                .updater(new Adam(0.001))
                .list()
                .layer(0, new DenseLayer.Builder()
                    .nIn(numInputs)
                    .nOut(numHiddenNodes)
                    .activation(Activation.RELU)
                    .build())
                .layer(1, new DenseLayer.Builder()
                    .nIn(numHiddenNodes)
                    .nOut(numHiddenNodes)
                    .activation(Activation.RELU)
                    .build())
                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                    .nIn(numHiddenNodes)
                    .nOut(numOutputs)
                    .activation(Activation.SOFTMAX)
                    .build())
                .build());
            
            network.init();
            LOGGER.info("Created new neural network");
        } catch (Exception e) {
            LOGGER.severe("Failed to create new network: " + e.getMessage());
        }
    }

    public MultiLayerNetwork getNetwork() {
        return network;
    }
    
    public INDArray predict(INDArray input) {
        try {
            if (network == null) {
                LOGGER.severe("Network not initialized");
                return Nd4j.zeros(1);
            }
            return network.output(input, false);
        } catch (Exception e) {
            LOGGER.severe("Prediction failed: " + e.getMessage());
            return Nd4j.zeros(1);
        }
    }

    public void train(DataSet trainingData) {
        try {
            if (network == null) {
                LOGGER.severe("Network not initialized");
                return;
            }
            network.fit(trainingData);
        } catch (Exception e) {
            LOGGER.severe("Training failed: " + e.getMessage());
        }
    }
    
    public void saveModel() {
        try {
            File modelFile = new File(MODEL_PATH);
            modelFile.getParentFile().mkdirs();
            ModelSerializer.writeModel(network, modelFile, true);
            LOGGER.info("Successfully saved model");
        } catch (IOException e) {
            LOGGER.severe("Failed to save model: " + e.getMessage());
        }
    }
    
    public void loadPreTrainedModel() {
        try {
            File modelFile = new File(MODEL_PATH);
            if (!modelFile.exists()) {
                LOGGER.warning("Pre-trained model not found. Will need to train from scratch.");
                createNewNetwork();
                return;
            }
            network = ModelSerializer.restoreMultiLayerNetwork(modelFile);
            LOGGER.info("Successfully loaded pre-trained model");
        } catch (IOException e) {
            LOGGER.severe("Failed to load pre-trained model: " + e.getMessage());
            createNewNetwork();
        }
    }
    
    @Override
    public void close() {
        try {
            if (network != null && network instanceof Model) {
                ((Model) network).close();
                Nd4j.getWorkspaceManager().destroyAllWorkspacesForCurrentThread();
            }
        } catch (Exception e) {
            LOGGER.warning("Error closing network: " + e.getMessage());
        } finally {
            network = null;
        }
    }
    
    public void shutdown() {
        close();
    }
}
