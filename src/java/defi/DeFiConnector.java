package com.albert.trading.bot.defi;

import org.web3j.protocol.Web3j;
import org.web3j.protocol.core.methods.response.EthGasPrice;
import org.web3j.protocol.http.HttpService;
import org.web3j.crypto.Credentials;
import org.web3j.tx.gas.ContractGasProvider;
import java.math.BigInteger;
import java.util.*;
import java.util.concurrent.*;
import java.util.logging.Logger;

public class DeFiConnector {
    private static final Logger LOGGER = Logger.getLogger(DeFiConnector.class.getName());
    
    private final Web3j web3j;
    private final Map<String, DeFiProtocol> protocols;
    private final ExecutorService executorService;
    private final BlockingQueue<DeFiTransaction> transactionQueue;
    
    private static final int MAX_PENDING_TRANSACTIONS = 1000;
    private static final double MAX_SLIPPAGE = 0.01; // 1%
    private static final long GAS_PRICE_UPDATE_INTERVAL = 15000; // 15 seconds
    private volatile BigInteger currentGasPrice;
    
    public DeFiConnector(String ethereumNode) {
        this.web3j = Web3j.build(new HttpService(ethereumNode));
        this.protocols = new ConcurrentHashMap<>();
        this.executorService = Executors.newWorkStealingPool();
        this.transactionQueue = new LinkedBlockingQueue<>(MAX_PENDING_TRANSACTIONS);
        
        initializeProtocols();
        startGasPriceMonitor();
        startTransactionProcessor();
    }
    
    private void initializeProtocols() {
        // Initialize major DeFi protocols
        protocols.put("uniswap", new UniswapProtocol(web3j));
        protocols.put("sushiswap", new SushiSwapProtocol(web3j));
        protocols.put("curve", new CurveProtocol(web3j));
        protocols.put("aave", new AaveProtocol(web3j));
        protocols.put("compound", new CompoundProtocol(web3j));
        
        LOGGER.info("Initialized " + protocols.size() + " DeFi protocols");
    }
    
    private void startGasPriceMonitor() {
        ScheduledExecutorService scheduler = Executors.newSingleThreadScheduledExecutor();
        scheduler.scheduleAtFixedRate(() -> {
            try {
                EthGasPrice gasPrice = web3j.ethGasPrice().send();
                currentGasPrice = gasPrice.getGasPrice();
                LOGGER.fine("Updated gas price: " + currentGasPrice);
            } catch (Exception e) {
                LOGGER.warning("Failed to update gas price: " + e.getMessage());
            }
        }, 0, GAS_PRICE_UPDATE_INTERVAL, TimeUnit.MILLISECONDS);
    }
    
    private void startTransactionProcessor() {
        Thread processor = new Thread(() -> {
            while (!Thread.currentThread().isInterrupted()) {
                try {
                    DeFiTransaction tx = transactionQueue.take();
                    processDeFiTransaction(tx);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        });
        processor.setDaemon(true);
        processor.start();
    }
    
    public CompletableFuture<DeFiQuote> getQuote(String protocol, String tokenIn, 
                                                String tokenOut, BigInteger amount) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                DeFiProtocol protocolImpl = protocols.get(protocol.toLowerCase());
                if (protocolImpl == null) {
                    throw new IllegalArgumentException("Unsupported protocol: " + protocol);
                }
                
                return protocolImpl.getQuote(tokenIn, tokenOut, amount);
            } catch (Exception e) {
                LOGGER.severe("Failed to get quote: " + e.getMessage());
                throw new CompletionException(e);
            }
        }, executorService);
    }
    
    public CompletableFuture<List<DeFiQuote>> getBestQuote(String tokenIn, 
                                                          String tokenOut, 
                                                          BigInteger amount) {
        List<CompletableFuture<DeFiQuote>> quotes = new ArrayList<>();
        
        for (Map.Entry<String, DeFiProtocol> entry : protocols.entrySet()) {
            quotes.add(getQuote(entry.getKey(), tokenIn, tokenOut, amount));
        }
        
        return CompletableFuture.allOf(quotes.toArray(new CompletableFuture[0]))
            .thenApply(v -> quotes.stream()
                .map(CompletableFuture::join)
                .sorted((q1, q2) -> q2.getOutputAmount().compareTo(q1.getOutputAmount()))
                .collect(Collectors.toList()));
    }
    
    public CompletableFuture<String> executeTrade(DeFiTransaction transaction) {
        return CompletableFuture.supplyAsync(() -> {
            try {
                validateTransaction(transaction);
                transactionQueue.put(transaction);
                return "Transaction queued: " + transaction.getId();
            } catch (Exception e) {
                LOGGER.severe("Failed to queue transaction: " + e.getMessage());
                throw new CompletionException(e);
            }
        });
    }
    
    private void processDeFiTransaction(DeFiTransaction tx) {
        try {
            DeFiProtocol protocol = protocols.get(tx.getProtocol().toLowerCase());
            if (protocol == null) {
                throw new IllegalArgumentException("Unsupported protocol: " + tx.getProtocol());
            }
            
            // Check current price against quote
            DeFiQuote currentQuote = protocol.getQuote(
                tx.getTokenIn(), 
                tx.getTokenOut(), 
                tx.getAmount()
            ).get();
            
            if (isSlippageAcceptable(tx.getQuote(), currentQuote)) {
                String txHash = protocol.executeTrade(tx, currentGasPrice).get();
                LOGGER.info("Transaction executed: " + txHash);
            } else {
                LOGGER.warning("Transaction rejected due to excessive slippage");
            }
        } catch (Exception e) {
            LOGGER.severe("Failed to process transaction: " + e.getMessage());
        }
    }
    
    private boolean isSlippageAcceptable(DeFiQuote originalQuote, DeFiQuote currentQuote) {
        BigInteger original = originalQuote.getOutputAmount();
        BigInteger current = currentQuote.getOutputAmount();
        
        double slippage = 1.0 - (current.doubleValue() / original.doubleValue());
        return slippage <= MAX_SLIPPAGE;
    }
    
    private void validateTransaction(DeFiTransaction tx) {
        if (tx.getAmount().compareTo(BigInteger.ZERO) <= 0) {
            throw new IllegalArgumentException("Invalid amount");
        }
        
        if (!protocols.containsKey(tx.getProtocol().toLowerCase())) {
            throw new IllegalArgumentException("Unsupported protocol: " + tx.getProtocol());
        }
    }
    
    public Map<String, TokenBalance> getTokenBalances(String address) {
        Map<String, TokenBalance> balances = new ConcurrentHashMap<>();
        List<CompletableFuture<Void>> futures = new ArrayList<>();
        
        for (DeFiProtocol protocol : protocols.values()) {
            futures.add(CompletableFuture.runAsync(() -> {
                try {
                    Map<String, TokenBalance> protocolBalances = 
                        protocol.getTokenBalances(address);
                    balances.putAll(protocolBalances);
                } catch (Exception e) {
                    LOGGER.warning("Failed to get balances from " + 
                                 protocol.getName() + ": " + e.getMessage());
                }
            }, executorService));
        }
        
        CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).join();
        return balances;
    }
    
    public void shutdown() {
        executorService.shutdown();
        try {
            if (!executorService.awaitTermination(60, TimeUnit.SECONDS)) {
                executorService.shutdownNow();
            }
        } catch (InterruptedException e) {
            executorService.shutdownNow();
            Thread.currentThread().interrupt();
        }
        
        web3j.shutdown();
    }
}
