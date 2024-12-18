package com.albert.trading.bot.model;

import java.time.Instant;
import java.util.UUID;

import com.albert.trading.bot.model.TradingSignal;

public class TradeSignal {
    public static TradingSignal BUY;
    public static TradeSignal SELL;
    public static TradeSignal HOLD;
    private final String id;
    private final String ticker;
    private final TradingSignal signal;
    private final double amount;
    private final String exchange;
    private final Instant timestamp;
    
    public TradeSignal(String ticker, TradingSignal signal, double amount, String exchange) {
        this.id = UUID.randomUUID().toString();
        this.ticker = ticker;
        this.signal = signal;
        this.amount = amount;
        this.exchange = exchange;
        this.timestamp = Instant.now();
    }
    
    public String getId() {
        return id;
    }
    
    public String getTicker() {
        return ticker;
    }
    
    public TradingSignal getSignal() {
        return signal;
    }
    
    public double getAmount() {
        return amount;
    }
    
    public String getExchange() {
        return exchange;
    }
    
    public Instant getTimestamp() {
        return timestamp;
    }
}
