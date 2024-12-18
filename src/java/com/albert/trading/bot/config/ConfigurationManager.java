package com.albert.trading.bot.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class ConfigurationManager {
    private static final String CONFIG_DIR = "config";
    private static final String MAIN_CONFIG = "config.yaml";
    private static final String STRATEGY_CONFIG = "strategies.yaml";
    private static final String API_CONFIG = "api_config.yaml";
    
    private final ObjectMapper yamlMapper;
    private final Map<String, Object> mainConfig;
    private final Map<String, Map<String, Object>> strategyConfigs;
    private final Map<String, ApiConfig> apiConfigs;
    private final Map<String, Object> cache;
    
    private static ConfigurationManager instance;
    
    private ConfigurationManager() {
        this.yamlMapper = new ObjectMapper(new YAMLFactory());
        this.mainConfig = new HashMap<>();
        this.strategyConfigs = new HashMap<>();
        this.apiConfigs = new HashMap<>();
        this.cache = new ConcurrentHashMap<>();
        loadConfigurations();
    }
    
    public static synchronized ConfigurationManager getInstance() {
        if (instance == null) {
            instance = new ConfigurationManager();
        }
        return instance;
    }
    
    private void loadConfigurations() {
        try {
            // Load main configuration
            File mainConfigFile = new File(CONFIG_DIR, MAIN_CONFIG);
            if (mainConfigFile.exists()) {
                mainConfig.putAll(yamlMapper.readValue(mainConfigFile, Map.class));
            }
            
            // Load strategy configurations
            File strategyConfigFile = new File(CONFIG_DIR, STRATEGY_CONFIG);
            if (strategyConfigFile.exists()) {
                Map<String, Map<String, Object>> strategies = yamlMapper.readValue(
                    strategyConfigFile, Map.class);
                strategyConfigs.putAll(strategies);
            }
            
            // Load API configurations
            File apiConfigFile = new File(CONFIG_DIR, API_CONFIG);
            if (apiConfigFile.exists()) {
                Map<String, ApiConfig> apis = yamlMapper.readValue(
                    apiConfigFile, 
                    yamlMapper.getTypeFactory().constructMapType(
                        Map.class, String.class, ApiConfig.class));
                apiConfigs.putAll(apis);
            }
        } catch (IOException e) {
            throw new RuntimeException("Failed to load configurations", e);
        }
    }
    
    public void reloadConfigurations() {
        mainConfig.clear();
        strategyConfigs.clear();
        apiConfigs.clear();
        cache.clear();
        loadConfigurations();
    }
    
    @SuppressWarnings("unchecked")
    public <T> T getConfig(String key, Class<T> type) {
        return (T) cache.computeIfAbsent(key, k -> {
            Object value = mainConfig.get(k);
            if (value == null) {
                return null;
            }
            return yamlMapper.convertValue(value, type);
        });
    }
    
    public Map<String, Object> getStrategyConfig(String strategyName) {
        return new HashMap<>(strategyConfigs.getOrDefault(strategyName, new HashMap<>()));
    }
    
    public ApiConfig getApiConfig(String apiName) {
        return apiConfigs.get(apiName);
    }
    
    public void updateMainConfig(String key, Object value) {
        mainConfig.put(key, value);
        cache.remove(key);
        saveMainConfig();
    }
    
    public void updateStrategyConfig(String strategyName, Map<String, Object> config) {
        strategyConfigs.put(strategyName, new HashMap<>(config));
        saveStrategyConfig();
    }
    
    public void updateApiConfig(String apiName, ApiConfig config) {
        apiConfigs.put(apiName, config);
        saveApiConfig();
    }
    
    private void saveMainConfig() {
        try {
            File configFile = new File(CONFIG_DIR, MAIN_CONFIG);
            yamlMapper.writeValue(configFile, mainConfig);
        } catch (IOException e) {
            throw new RuntimeException("Failed to save main configuration", e);
        }
    }
    
    private void saveStrategyConfig() {
        try {
            File configFile = new File(CONFIG_DIR, STRATEGY_CONFIG);
            yamlMapper.writeValue(configFile, strategyConfigs);
        } catch (IOException e) {
            throw new RuntimeException("Failed to save strategy configurations", e);
        }
    }
    
    private void saveApiConfig() {
        try {
            File configFile = new File(CONFIG_DIR, API_CONFIG);
            yamlMapper.writeValue(configFile, apiConfigs);
        } catch (IOException e) {
            throw new RuntimeException("Failed to save API configurations", e);
        }
    }
    
    public Map<String, Object> getMainConfig() {
        return new HashMap<>(mainConfig);
    }
    
    public Map<String, Map<String, Object>> getAllStrategyConfigs() {
        return new HashMap<>(strategyConfigs);
    }
    
    public Map<String, ApiConfig> getAllApiConfigs() {
        return new HashMap<>(apiConfigs);
    }
}
