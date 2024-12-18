package com.albert.trading.bot.config;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.HashMap;
import java.util.Map;

@JsonIgnoreProperties(ignoreUnknown = true)
public class ApiConfig {
    private final String name;
    private final String baseUrl;
    private final String version;
    private final Map<String, String> headers;
    private final Map<String, String> parameters;
    private final AuthConfig auth;
    private final RateLimitConfig rateLimit;
    
    public ApiConfig(
            @JsonProperty("name") String name,
            @JsonProperty("baseUrl") String baseUrl,
            @JsonProperty("version") String version,
            @JsonProperty("headers") Map<String, String> headers,
            @JsonProperty("parameters") Map<String, String> parameters,
            @JsonProperty("auth") AuthConfig auth,
            @JsonProperty("rateLimit") RateLimitConfig rateLimit) {
        this.name = name;
        this.baseUrl = baseUrl;
        this.version = version;
        this.headers = headers != null ? new HashMap<>(headers) : new HashMap<>();
        this.parameters = parameters != null ? new HashMap<>(parameters) : new HashMap<>();
        this.auth = auth;
        this.rateLimit = rateLimit;
    }
    
    public String getName() {
        return name;
    }
    
    public String getBaseUrl() {
        return baseUrl;
    }
    
    public String getVersion() {
        return version;
    }
    
    public Map<String, String> getHeaders() {
        return new HashMap<>(headers);
    }
    
    public Map<String, String> getParameters() {
        return new HashMap<>(parameters);
    }
    
    public AuthConfig getAuth() {
        return auth;
    }
    
    public RateLimitConfig getRateLimit() {
        return rateLimit;
    }
    
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class AuthConfig {
        private final String type;
        private final String apiKey;
        private final String apiSecret;
        private final Map<String, String> additionalParams;
        
        public AuthConfig(
                @JsonProperty("type") String type,
                @JsonProperty("apiKey") String apiKey,
                @JsonProperty("apiSecret") String apiSecret,
                @JsonProperty("additionalParams") Map<String, String> additionalParams) {
            this.type = type;
            this.apiKey = apiKey;
            this.apiSecret = apiSecret;
            this.additionalParams = additionalParams != null ? 
                new HashMap<>(additionalParams) : new HashMap<>();
        }
        
        public String getType() {
            return type;
        }
        
        public String getApiKey() {
            return apiKey;
        }
        
        public String getApiSecret() {
            return apiSecret;
        }
        
        public Map<String, String> getAdditionalParams() {
            return new HashMap<>(additionalParams);
        }
    }
    
    @JsonIgnoreProperties(ignoreUnknown = true)
    public static class RateLimitConfig {
        private final int requestsPerSecond;
        private final int requestsPerMinute;
        private final int requestsPerHour;
        private final Map<String, Integer> endpointLimits;
        
        public RateLimitConfig(
                @JsonProperty("requestsPerSecond") int requestsPerSecond,
                @JsonProperty("requestsPerMinute") int requestsPerMinute,
                @JsonProperty("requestsPerHour") int requestsPerHour,
                @JsonProperty("endpointLimits") Map<String, Integer> endpointLimits) {
            this.requestsPerSecond = requestsPerSecond;
            this.requestsPerMinute = requestsPerMinute;
            this.requestsPerHour = requestsPerHour;
            this.endpointLimits = endpointLimits != null ? 
                new HashMap<>(endpointLimits) : new HashMap<>();
        }
        
        public int getRequestsPerSecond() {
            return requestsPerSecond;
        }
        
        public int getRequestsPerMinute() {
            return requestsPerMinute;
        }
        
        public int getRequestsPerHour() {
            return requestsPerHour;
        }
        
        public Map<String, Integer> getEndpointLimits() {
            return new HashMap<>(endpointLimits);
        }
    }
}
