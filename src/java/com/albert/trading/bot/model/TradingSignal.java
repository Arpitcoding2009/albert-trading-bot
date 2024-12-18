package com.albert.trading.bot.model;

/**
 * Represents possible trading signals generated by the market analysis
 */
public enum TradingSignal {
    BUY("Buy signal indicating potential upward movement"),
    SELL("Sell signal indicating potential downward movement"),
    HOLD("Hold signal indicating no clear directional movement"),
    STRONG_BUY("Strong buy signal with high confidence"),
    STRONG_SELL("Strong sell signal with high confidence");

    private final String description;

    TradingSignal(String description) {
        this.description = description;
    }

    public String getDescription() {
        return description;
    }

    public boolean isStrong() {
        return this == STRONG_BUY || this == STRONG_SELL;
    }

    public boolean isBuySignal() {
        return this == BUY || this == STRONG_BUY;
    }

    public boolean isSellSignal() {
        return this == SELL || this == STRONG_SELL;
    }
}
