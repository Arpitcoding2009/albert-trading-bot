// WebSocket handling and real-time updates
class WebSocketManager {
    constructor() {
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 5000; // 5 seconds
        this.heartbeatInterval = null;
        this.messageHandlers = new Map();
        
        // Initialize message handlers
        this.initializeMessageHandlers();
    }

    connect() {
        try {
            this.ws = new WebSocket(`ws://${window.location.host}/ws`);
            
            this.ws.onopen = this.handleOpen.bind(this);
            this.ws.onclose = this.handleClose.bind(this);
            this.ws.onerror = this.handleError.bind(this);
            this.ws.onmessage = this.handleMessage.bind(this);
            
        } catch (error) {
            console.error('WebSocket connection error:', error);
            this.handleError(error);
        }
    }

    initializeMessageHandlers() {
        // Market data updates
        this.messageHandlers.set('market_data', (data) => {
            this.updateMarketData(data);
        });
        
        // Trading signals
        this.messageHandlers.set('trade_signal', (data) => {
            this.handleTradeSignal(data);
        });
        
        // Portfolio updates
        this.messageHandlers.set('portfolio_update', (data) => {
            this.updatePortfolio(data);
        });
        
        // Alert messages
        this.messageHandlers.set('alert', (data) => {
            this.handleAlert(data);
        });
        
        // System status updates
        this.messageHandlers.set('status', (data) => {
            this.updateStatus(data);
        });
    }

    handleOpen() {
        console.log('WebSocket connected');
        this.reconnectAttempts = 0;
        this.startHeartbeat();
        showNotification('Connected to trading server', 'success');
    }

    handleClose(event) {
        console.log('WebSocket closed:', event);
        this.stopHeartbeat();
        
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            setTimeout(() => {
                this.reconnectAttempts++;
                this.connect();
            }, this.reconnectDelay);
        } else {
            showNotification('Connection lost. Please refresh the page.', 'error');
        }
    }

    handleError(error) {
        console.error('WebSocket error:', error);
        showNotification('Connection error occurred', 'error');
    }

    handleMessage(event) {
        try {
            const message = JSON.parse(event.data);
            
            if (message.type && this.messageHandlers.has(message.type)) {
                this.messageHandlers.get(message.type)(message.data);
            } else {
                console.warn('Unknown message type:', message.type);
            }
        } catch (error) {
            console.error('Error handling message:', error);
        }
    }

    startHeartbeat() {
        this.heartbeatInterval = setInterval(() => {
            if (this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify({ type: 'ping' }));
            }
        }, 30000); // Send heartbeat every 30 seconds
    }

    stopHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
    }

    updateMarketData(data) {
        // Update price displays
        const priceElements = document.querySelectorAll('[data-price]');
        priceElements.forEach(element => {
            const symbol = element.dataset.price;
            if (data[symbol]) {
                element.textContent = formatCurrency(data[symbol].price);
                
                // Update price change indicators
                const changeElement = element.nextElementSibling;
                if (changeElement && changeElement.dataset.priceChange) {
                    const change = data[symbol].change_24h;
                    changeElement.textContent = formatPercentage(change);
                    changeElement.className = `text-sm ${change >= 0 ? 'text-green-600' : 'text-red-600'}`;
                }
            }
        });
        
        // Update charts
        updatePriceChart(data);
    }

    handleTradeSignal(data) {
        // Add trade to recent trades table
        addTradeToTable(data);
        
        // Show notification
        const message = `${data.side.toUpperCase()} signal for ${data.symbol} at ${formatCurrency(data.price)}`;
        showNotification(message, 'info');
        
        // Play sound alert
        this.playTradeAlert();
    }

    updatePortfolio(data) {
        // Update portfolio value
        document.getElementById('portfolio-value').textContent = formatCurrency(data.total_value);
        
        // Update P/L displays
        const plElement = document.getElementById('daily-pnl');
        const plValue = data.daily_pnl;
        plElement.textContent = formatCurrency(plValue);
        plElement.className = `text-2xl font-bold ${plValue >= 0 ? 'text-green-600' : 'text-red-600'}`;
        
        // Update positions
        this.updatePositions(data.positions);
    }

    updatePositions(positions) {
        const container = document.getElementById('positions-container');
        container.innerHTML = ''; // Clear existing positions
        
        Object.entries(positions).forEach(([symbol, position]) => {
            const card = this.createPositionCard(symbol, position);
            container.appendChild(card);
        });
    }

    createPositionCard(symbol, position) {
        const card = document.createElement('div');
        card.className = 'bg-white rounded-lg shadow-md p-4 mb-4';
        
        const pnl = position.unrealized_pnl;
        const pnlClass = pnl >= 0 ? 'text-green-600' : 'text-red-600';
        
        card.innerHTML = `
            <div class="flex justify-between items-center mb-2">
                <h3 class="text-lg font-medium">${symbol}</h3>
                <span class="px-2 py-1 text-sm rounded ${position.side === 'buy' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}">
                    ${position.side.toUpperCase()}
                </span>
            </div>
            <div class="grid grid-cols-2 gap-4">
                <div>
                    <p class="text-sm text-gray-500">Entry Price</p>
                    <p class="text-sm font-medium">${formatCurrency(position.entry_price)}</p>
                </div>
                <div>
                    <p class="text-sm text-gray-500">Current Price</p>
                    <p class="text-sm font-medium">${formatCurrency(position.current_price)}</p>
                </div>
                <div>
                    <p class="text-sm text-gray-500">Size</p>
                    <p class="text-sm font-medium">${position.size}</p>
                </div>
                <div>
                    <p class="text-sm text-gray-500">P/L</p>
                    <p class="text-sm font-medium ${pnlClass}">${formatCurrency(pnl)} (${formatPercentage(position.pnl_percentage)})</p>
                </div>
            </div>
        `;
        
        return card;
    }

    handleAlert(data) {
        showNotification(data.message, data.severity);
        
        if (data.severity === 'error') {
            this.playAlertSound();
        }
    }

    updateStatus(data) {
        // Update trading status indicator
        const statusElement = document.getElementById('trading-status');
        statusElement.textContent = `Status: ${data.status}`;
        statusElement.className = `px-4 py-2 rounded-md text-sm font-medium ${
            data.status === 'active' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
        }`;
        
        // Update model info
        if (data.model_info) {
            document.getElementById('model-info').textContent = 
                `Active Model: ${data.model_info.name} (Confidence: ${formatPercentage(data.model_info.confidence)})`;
        }
    }

    playTradeAlert() {
        // Implement trade alert sound
        const audio = new Audio('/static/sounds/trade.mp3');
        audio.play().catch(error => console.log('Error playing sound:', error));
    }

    playAlertSound() {
        // Implement alert sound
        const audio = new Audio('/static/sounds/alert.mp3');
        audio.play().catch(error => console.log('Error playing sound:', error));
    }
}

// Initialize WebSocket manager
const wsManager = new WebSocketManager();

// Connect when document is ready
document.addEventListener('DOMContentLoaded', () => {
    wsManager.connect();
});
