package com.albert.trading.bot.model;

import java.math.BigInteger;

public class DeFiQuote {
    private final String protocol;
    private final String tokenIn;
    private final String tokenOut;
    private final BigInteger inputAmount;
    private final BigInteger outputAmount;
    private final BigInteger gasEstimate;
    
    public DeFiQuote(String protocol, String tokenIn, String tokenOut,
                     BigInteger inputAmount, BigInteger outputAmount, 
                     BigInteger gasEstimate) {
        this.protocol = protocol;
        this.tokenIn = tokenIn;
        this.tokenOut = tokenOut;
        this.inputAmount = inputAmount;
        this.outputAmount = outputAmount;
        this.gasEstimate = gasEstimate;
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
    
    public BigInteger getInputAmount() {
        return inputAmount;
    }
    
    public BigInteger getOutputAmount() {
        return outputAmount;
    }
    
    public BigInteger getGasEstimate() {
        return gasEstimate;
    }
}
