// Main application logic
document.addEventListener('DOMContentLoaded', function() {
    // Initialize components
    initializeWebSocket();
    initializeCharts();
    initializeEventListeners();
    loadDashboardData();
});

// WebSocket connection
let ws;
function initializeWebSocket() {
    ws = new WebSocket(`ws://${window.location.host}/ws`);
    
    ws.onopen = () => {
        console.log('WebSocket connected');
        showNotification('Connected to server', 'success');
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWebSocketMessage(data);
    };
    
    ws.onclose = () => {
        console.log('WebSocket disconnected');
        showNotification('Disconnected from server', 'error');
        // Attempt to reconnect after 5 seconds
        setTimeout(initializeWebSocket, 5000);
    };
}

// Event listeners
function initializeEventListeners() {
    // Trading controls
    document.getElementById('start-trading').addEventListener('click', startTrading);
    document.getElementById('stop-trading').addEventListener('click', stopTrading);
    
    // Settings modal
    const settingsModal = document.getElementById('settings-modal');
    document.getElementById('save-settings').addEventListener('click', saveSettings);
    document.getElementById('close-settings').addEventListener('click', () => {
        settingsModal.classList.add('hidden');
    });
}

// Dashboard data
async function loadDashboardData() {
    try {
        const response = await fetch('/api/dashboard/summary');
        const data = await response.json();
        updateDashboard(data);
    } catch (error) {
        console.error('Error loading dashboard data:', error);
        showNotification('Error loading dashboard data', 'error');
    }
}

function updateDashboard(data) {
    // Update portfolio value
    document.getElementById('portfolio-value').textContent = 
        formatCurrency(data.portfolio.total_value);
    
    // Update portfolio change
    const changeElement = document.getElementById('portfolio-change');
    const changeValue = data.performance.pnl_percentage;
    changeElement.textContent = formatPercentage(changeValue);
    changeElement.className = `text-sm font-medium ${changeValue >= 0 ? 'text-green-600' : 'text-red-600'}`;
    
    // Update daily P/L
    document.getElementById('daily-pnl').textContent = 
        formatCurrency(data.performance.pnl);
    
    // Update win rate
    document.getElementById('win-rate').textContent = 
        `Win Rate: ${formatPercentage(data.performance.win_rate)}`;
    
    // Update positions
    document.getElementById('active-positions').textContent = 
        Object.keys(data.portfolio.positions).length;
    
    // Update total trades
    document.getElementById('total-trades').textContent = 
        `Total Trades: ${data.portfolio.trade_count}`;
    
    // Update trading status
    updateTradingStatus(data.trading_status.is_trading);
    
    // Update charts
    updateCharts(data);
}

// Trading controls
async function startTrading() {
    try {
        const response = await fetch('/api/trading/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(getTradeConfig())
        });
        
        if (response.ok) {
            showNotification('Trading started successfully', 'success');
            updateTradingStatus(true);
        } else {
            throw new Error('Failed to start trading');
        }
    } catch (error) {
        console.error('Error starting trading:', error);
        showNotification('Error starting trading', 'error');
    }
}

async function stopTrading() {
    try {
        const response = await fetch('/api/trading/stop', {
            method: 'POST'
        });
        
        if (response.ok) {
            showNotification('Trading stopped successfully', 'success');
            updateTradingStatus(false);
        } else {
            throw new Error('Failed to stop trading');
        }
    } catch (error) {
        console.error('Error stopping trading:', error);
        showNotification('Error stopping trading', 'error');
    }
}

function updateTradingStatus(isTrading) {
    const startButton = document.getElementById('start-trading');
    const stopButton = document.getElementById('stop-trading');
    const statusElement = document.getElementById('trading-status');
    
    if (isTrading) {
        startButton.classList.add('hidden');
        stopButton.classList.remove('hidden');
        statusElement.textContent = 'Status: Active';
        statusElement.classList.remove('text-red-600');
        statusElement.classList.add('text-green-600');
    } else {
        startButton.classList.remove('hidden');
        stopButton.classList.add('hidden');
        statusElement.textContent = 'Status: Inactive';
        statusElement.classList.remove('text-green-600');
        statusElement.classList.add('text-red-600');
    }
}

// Settings management
function getTradeConfig() {
    const form = document.getElementById('settings-form');
    const formData = new FormData(form);
    
    return {
        trading_pairs: formData.get('trading_pairs').split(',').map(p => p.trim()),
        risk: {
            max_position_size: parseFloat(formData.get('max_position_size')) / 100,
            stop_loss_percentage: parseFloat(formData.get('stop_loss')) / 100,
            take_profit_percentage: parseFloat(formData.get('take_profit')) / 100
        },
        exchanges: {
            primary: {
                api_key: formData.get('api_key'),
                api_secret: formData.get('api_secret')
            }
        }
    };
}

async function saveSettings() {
    try {
        const config = getTradeConfig();
        const response = await fetch('/api/settings', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        });
        
        if (response.ok) {
            showNotification('Settings saved successfully', 'success');
            document.getElementById('settings-modal').classList.add('hidden');
        } else {
            throw new Error('Failed to save settings');
        }
    } catch (error) {
        console.error('Error saving settings:', error);
        showNotification('Error saving settings', 'error');
    }
}

// WebSocket message handling
function handleWebSocketMessage(data) {
    switch (data.type) {
        case 'market_data':
            updateMarketData(data);
            break;
        case 'trade_signal':
            handleTradeSignal(data);
            break;
        case 'alert':
            handleAlert(data);
            break;
        case 'portfolio_update':
            handlePortfolioUpdate(data);
            break;
    }
}

function updateMarketData(data) {
    // Update price charts
    updatePriceChart(data);
    
    // Update current price display
    document.getElementById('current-price').textContent = 
        formatCurrency(data.price);
}

function handleTradeSignal(data) {
    // Add trade to recent trades table
    addTradeToTable(data);
    
    // Show notification
    showNotification(
        `New ${data.side} signal for ${data.symbol}`,
        'info'
    );
}

function handleAlert(data) {
    showNotification(data.message, data.severity);
}

function handlePortfolioUpdate(data) {
    // Update portfolio displays
    document.getElementById('portfolio-value').textContent = 
        formatCurrency(data.total_value);
    
    // Update positions display
    updatePositionsDisplay(data.positions);
}

// Utility functions
function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(value);
}

function formatPercentage(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'percent',
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    }).format(value / 100);
}

function showNotification(message, type) {
    const notification = document.getElementById('notification');
    const messageElement = document.getElementById('notification-message');
    const iconElement = document.getElementById('notification-icon');
    
    // Set message
    messageElement.textContent = message;
    
    // Set icon and color based on type
    switch (type) {
        case 'success':
            iconElement.innerHTML = '✓';
            iconElement.className = 'text-green-500';
            break;
        case 'error':
            iconElement.innerHTML = '✗';
            iconElement.className = 'text-red-500';
            break;
        case 'info':
            iconElement.innerHTML = 'ℹ';
            iconElement.className = 'text-blue-500';
            break;
    }
    
    // Show notification
    notification.classList.remove('hidden');
    
    // Hide after 3 seconds
    setTimeout(() => {
        notification.classList.add('hidden');
    }, 3000);
}

function addTradeToTable(trade) {
    const table = document.getElementById('trades-table');
    const row = document.createElement('tr');
    
    row.innerHTML = `
        <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
            ${new Date(trade.timestamp).toLocaleString()}
        </td>
        <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
            ${trade.symbol}
        </td>
        <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
            ${trade.side}
        </td>
        <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
            ${formatCurrency(trade.price)}
        </td>
        <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
            ${trade.amount}
        </td>
        <td class="px-6 py-4 whitespace-nowrap text-sm ${trade.pnl >= 0 ? 'text-green-600' : 'text-red-600'}">
            ${formatCurrency(trade.pnl)}
        </td>
    `;
    
    // Add to top of table
    table.insertBefore(row, table.firstChild);
    
    // Remove oldest row if more than 50 rows
    if (table.children.length > 50) {
        table.removeChild(table.lastChild);
    }
}

function updatePositionsDisplay(positions) {
    const positionsContainer = document.getElementById('positions-container');
    positionsContainer.innerHTML = '';
    
    for (const [symbol, position] of Object.entries(positions)) {
        const card = document.createElement('div');
        card.className = 'bg-white rounded-lg shadow p-4 mb-4';
        
        card.innerHTML = `
            <div class="flex justify-between items-center mb-2">
                <h4 class="text-lg font-medium text-gray-900">${symbol}</h4>
                <span class="px-2 py-1 rounded ${position.side === 'buy' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}">
                    ${position.side.toUpperCase()}
                </span>
            </div>
            <div class="grid grid-cols-2 gap-4">
                <div>
                    <p class="text-sm text-gray-500">Entry Price</p>
                    <p class="text-sm font-medium text-gray-900">${formatCurrency(position.entry_price)}</p>
                </div>
                <div>
                    <p class="text-sm text-gray-500">Amount</p>
                    <p class="text-sm font-medium text-gray-900">${position.amount}</p>
                </div>
                <div>
                    <p class="text-sm text-gray-500">Stop Loss</p>
                    <p class="text-sm font-medium text-gray-900">${formatCurrency(position.stop_loss)}</p>
                </div>
                <div>
                    <p class="text-sm text-gray-500">Take Profit</p>
                    <p class="text-sm font-medium text-gray-900">${formatCurrency(position.take_profit)}</p>
                </div>
            </div>
        `;
        
        positionsContainer.appendChild(card);
    }
}
