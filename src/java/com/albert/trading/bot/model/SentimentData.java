package com.albert.trading.bot.model;

import java.time.Instant;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class SentimentData {
    private final String symbol;
    private final double overallSentiment;
    private final Map<String, Double> sourceSentiments;
    private final List<String> keywords;
    private final Map<String, Integer> mentionCounts;
    private final Instant timestamp;
    
    private SentimentData(Builder builder) {
        this.symbol = builder.symbol;
        this.overallSentiment = builder.overallSentiment;
        this.sourceSentiments = new HashMap<>(builder.sourceSentiments);
        this.keywords = new ArrayList<>(builder.keywords);
        this.mentionCounts = new HashMap<>(builder.mentionCounts);
        this.timestamp = builder.timestamp;
    }
    
    public String getSymbol() {
        return symbol;
    }
    
    public double getOverallSentiment() {
        return overallSentiment;
    }
    
    public Map<String, Double> getSourceSentiments() {
        return new HashMap<>(sourceSentiments);
    }
    
    public List<String> getKeywords() {
        return new ArrayList<>(keywords);
    }
    
    public Map<String, Integer> getMentionCounts() {
        return new HashMap<>(mentionCounts);
    }
    
    public Instant getTimestamp() {
        return timestamp;
    }
    
    public static class Builder {
        private String symbol;
        private double overallSentiment;
        private final Map<String, Double> sourceSentiments = new HashMap<>();
        private final List<String> keywords = new ArrayList<>();
        private final Map<String, Integer> mentionCounts = new HashMap<>();
        private Instant timestamp = Instant.now();
        
        public Builder(String symbol) {
            this.symbol = symbol;
        }
        
        public Builder overallSentiment(double sentiment) {
            this.overallSentiment = sentiment;
            return this;
        }
        
        public Builder addSourceSentiment(String source, double sentiment) {
            this.sourceSentiments.put(source, sentiment);
            return this;
        }
        
        public Builder addKeyword(String keyword) {
            this.keywords.add(keyword);
            return this;
        }
        
        public Builder addMentionCount(String source, int count) {
            this.mentionCounts.put(source, count);
            return this;
        }
        
        public Builder timestamp(Instant timestamp) {
            this.timestamp = timestamp;
            return this;
        }
        
        public SentimentData build() {
            return new SentimentData(this);
        }
    }
    
    @Override
    public String toString() {
        return String.format("SentimentData[symbol=%s, sentiment=%.2f, sources=%d, keywords=%d]",
            symbol, overallSentiment, sourceSentiments.size(), keywords.size());
    }
}
