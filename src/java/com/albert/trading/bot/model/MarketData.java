package com.albert.trading.bot.model;

import java.time.Instant;
import java.util.Arrays;

public class MarketData {
    private final String ticker;
    private final double[] prices;
    private final double[] volumes;
    private final double[] technicalIndicators;
    private final Instant timestamp;
    
    public MarketData(String ticker, double[] prices, double[] volumes, double[] technicalIndicators) {
        this.ticker = ticker;
        this.prices = Arrays.copyOf(prices, prices.length);
        this.volumes = Arrays.copyOf(volumes, volumes.length);
        this.technicalIndicators = technicalIndicators != null ? 
            Arrays.copyOf(technicalIndicators, technicalIndicators.length) : new double[0];
        this.timestamp = Instant.now();
    }
    
    public String getTicker() {
        return ticker;
    }
    
    public double[] getPrices() {
        return Arrays.copyOf(prices, prices.length);
    }
    
    public double[] getVolumes() {
        return Arrays.copyOf(volumes, volumes.length);
    }
    
    public double[] getTechnicalIndicators() {
        return Arrays.copyOf(technicalIndicators, technicalIndicators.length);
    }
    
    public Instant getTimestamp() {
        return timestamp;
    }
    
    public double getLatestPrice() {
        return prices[prices.length - 1];
    }
    
    public double getLatestVolume() {
        return volumes[volumes.length - 1];
    }
    
    public MarketData withUpdatedPrice(double newPrice) {
        double[] newPrices = Arrays.copyOf(prices, prices.length);
        System.arraycopy(prices, 1, newPrices, 0, prices.length - 1);
        newPrices[prices.length - 1] = newPrice;
        return new MarketData(ticker, newPrices, volumes, technicalIndicators);
    }
    
    public MarketData withUpdatedVolume(double newVolume) {
        double[] newVolumes = Arrays.copyOf(volumes, volumes.length);
        System.arraycopy(volumes, 1, newVolumes, 0, volumes.length - 1);
        newVolumes[volumes.length - 1] = newVolume;
        return new MarketData(ticker, prices, newVolumes, technicalIndicators);
    }
    
    @Override
    public String toString() {
        return String.format("MarketData[ticker=%s, latest_price=%.2f, latest_volume=%.2f, timestamp=%s]",
            ticker, getLatestPrice(), getLatestVolume(), timestamp);
    }
}
