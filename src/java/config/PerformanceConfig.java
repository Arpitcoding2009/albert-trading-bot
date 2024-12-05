package com.albert.trading.bot.config;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.logging.Logger;
import java.util.logging.Level;

public class PerformanceConfig {
    private static final Logger LOGGER = Logger.getLogger(PerformanceConfig.class.getName());
    private static final String CONFIG_PATH = "src/config/performance.json";
    private static volatile PerformanceConfig instance;
    private Map<String, Object> config;

    private PerformanceConfig() {
        loadConfig();
    }

    public static PerformanceConfig getInstance() {
        if (instance == null) {
            synchronized (PerformanceConfig.class) {
                if (instance == null) {
                    instance = new PerformanceConfig();
                }
            }
        }
        return instance;
    }

    private void loadConfig() {
        try {
            ObjectMapper mapper = new ObjectMapper();
            config = mapper.readValue(new File(CONFIG_PATH), Map.class);
            LOGGER.info("Performance configuration loaded successfully");
        } catch (IOException e) {
            LOGGER.log(Level.SEVERE, "Failed to load performance configuration", e);
            throw new RuntimeException("Failed to load performance configuration", e);
        }
    }

    public Map<String, Object> getJavaConfig() {
        return getNestedConfig("performance.java");
    }

    public Map<String, Object> getPythonConfig() {
        return getNestedConfig("performance.python");
    }

    public Map<String, Object> getTradingConfig() {
        return getNestedConfig("performance.trading");
    }

    public Map<String, Object> getMemoryConfig() {
        return getNestedConfig("performance.memory_management");
    }

    public Map<String, Object> getLoggingConfig() {
        return getNestedConfig("performance.logging");
    }

    public Map<String, Object> getNetworkConfig() {
        return getNestedConfig("performance.network");
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> getNestedConfig(String path) {
        String[] parts = path.split("\\.");
        Map<String, Object> current = config;
        
        for (String part : parts) {
            Object value = current.get(part);
            if (value instanceof Map) {
                current = (Map<String, Object>) value;
            } else {
                return null;
            }
        }
        
        return current;
    }

    public void reloadConfig() {
        loadConfig();
    }
}
