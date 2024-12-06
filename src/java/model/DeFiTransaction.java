package com.albert.trading.bot.model;

import java.math.BigInteger;
import java.util.UUID;

public class DeFiTransaction {
    private final String id;
    private final String protocol;
    private final String tokenIn;
    private final String tokenOut;
    private final BigInteger amount;
    private final DeFiQuote quote;
    
    public DeFiTransaction(String protocol, String tokenIn, String tokenOut, BigInteger amount) {
        this.id = UUID.randomUUID().toString();
        this.protocol = protocol;
        this.tokenIn = tokenIn;
        this.tokenOut = tokenOut;
        this.amount = amount;
        this.quote = null;
    }
    
    public DeFiTransaction(String protocol, String tokenIn, String tokenOut, 
                         BigInteger amount, DeFiQuote quote) {
        this.id = UUID.randomUUID().toString();
        this.protocol = protocol;
        this.tokenIn = tokenIn;
        this.tokenOut = tokenOut;
        this.amount = amount;
        this.quote = quote;
    }
    
    public String getId() {
        return id;
    }
    
    public String getProtocol() {
        return protocol;
    }
    
    public String getTokenIn() {
        return tokenIn;
    }
    
    public String getTokenOut() {
        return tokenOut;
    }
    
    public BigInteger getAmount() {
        return amount;
    }
    
    public DeFiQuote getQuote() {
        return quote;
    }
}
