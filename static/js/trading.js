// Trading operations and real-time updates
class TradingManager {
    constructor() {
        this.isTrading = false;
        this.activeTrades = new Map();
        this.riskLevel = 'medium';
        this.profitTarget = 0;
        this.stopLoss = 0;
        this.activeStrategies = {
            trendFollowing: true,
            meanReversion: true,
            arbitrage: true
        };
        
        this.initializeEventListeners();
    }
    
    initializeEventListeners() {
        // Trading controls
        document.getElementById('start-trading').addEventListener('click', () => this.startTrading());
        document.getElementById('stop-trading').addEventListener('click', () => this.stopTrading());
        
        // Settings controls
        document.getElementById('risk-level').addEventListener('change', (e) => this.updateRiskLevel(e.target.value));
        document.getElementById('profit-target').addEventListener('input', (e) => this.updateProfitTarget(e.target.value));
        document.getElementById('stop-loss').addEventListener('input', (e) => this.updateStopLoss(e.target.value));
        
        // Strategy toggles
        document.getElementById('trend-following').addEventListener('change', (e) => this.toggleStrategy('trendFollowing', e.target.checked));
        document.getElementById('mean-reversion').addEventListener('change', (e) => this.toggleStrategy('meanReversion', e.target.checked));
        document.getElementById('arbitrage').addEventListener('change', (e) => this.toggleStrategy('arbitrage', e.target.checked));
    }
    
    async startTrading() {
        try {
            const response = await fetch('/api/trading/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(this.getTradeConfig())
            });
            
            if (response.ok) {
                this.isTrading = true;
                this.updateTradingStatus(true);
                showNotification('Trading started successfully', 'success');
            } else {
                throw new Error('Failed to start trading');
            }
        } catch (error) {
            console.error('Error starting trading:', error);
            showNotification('Failed to start trading', 'error');
        }
    }
    
    async stopTrading() {
        try {
            const response = await fetch('/api/trading/stop', {
                method: 'POST'
            });
            
            if (response.ok) {
                this.isTrading = false;
                this.updateTradingStatus(false);
                showNotification('Trading stopped successfully', 'success');
            } else {
                throw new Error('Failed to stop trading');
            }
        } catch (error) {
            console.error('Error stopping trading:', error);
            showNotification('Failed to stop trading', 'error');
        }
    }
    
    updateTradingStatus(isTrading) {
        const startButton = document.getElementById('start-trading');
        const stopButton = document.getElementById('stop-trading');
        const statusDot = document.querySelector('.status-dot');
        const statusText = document.querySelector('.status-text');
        
        if (isTrading) {
            startButton.classList.add('hidden');
            stopButton.classList.remove('hidden');
            statusDot.classList.add('active');
            statusText.textContent = 'Active';
        } else {
            startButton.classList.remove('hidden');
            stopButton.classList.add('hidden');
            statusDot.classList.remove('active');
            statusText.textContent = 'Inactive';
        }
    }
    
    updateRiskLevel(level) {
        this.riskLevel = level;
        if (this.isTrading) {
            this.updateTradeConfig();
        }
    }
    
    updateProfitTarget(target) {
        this.profitTarget = parseFloat(target);
        if (this.isTrading) {
            this.updateTradeConfig();
        }
    }
    
    updateStopLoss(stopLoss) {
        this.stopLoss = parseFloat(stopLoss);
        if (this.isTrading) {
            this.updateTradeConfig();
        }
    }
    
    toggleStrategy(strategy, enabled) {
        this.activeStrategies[strategy] = enabled;
        if (this.isTrading) {
            this.updateTradeConfig();
        }
    }
    
    getTradeConfig() {
        return {
            riskLevel: this.riskLevel,
            profitTarget: this.profitTarget,
            stopLoss: this.stopLoss,
            activeStrategies: this.activeStrategies
        };
    }
    
    async updateTradeConfig() {
        try {
            const response = await fetch('/api/trading/config', {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(this.getTradeConfig())
            });
            
            if (response.ok) {
                showNotification('Trading configuration updated', 'success');
            } else {
                throw new Error('Failed to update trading configuration');
            }
        } catch (error) {
            console.error('Error updating trade config:', error);
            showNotification('Failed to update trading configuration', 'error');
        }
    }
    
    handleNewTrade(trade) {
        this.activeTrades.set(trade.id, trade);
        this.updateTradesTable();
        
        // Update 3D visualization if trade is significant
        if (Math.abs(trade.amount) > 1000) {
            updateVisualizationWithTradeData({
                profitable: trade.pnl > 0,
                amount: trade.amount
            });
        }
    }
    
    updateTradesTable() {
        const tableBody = document.querySelector('#trades-table tbody');
        tableBody.innerHTML = '';
        
        this.activeTrades.forEach(trade => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${trade.pair}</td>
                <td>${trade.strategy}</td>
                <td>${formatCurrency(trade.entryPrice)}</td>
                <td>${formatCurrency(trade.currentPrice)}</td>
                <td class="${trade.pnl >= 0 ? 'text-green-500' : 'text-red-500'}">${formatCurrency(trade.pnl)}</td>
                <td>${formatDuration(trade.duration)}</td>
            `;
            tableBody.appendChild(row);
        });
    }
}

// Utility functions
function formatDuration(ms) {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    
    if (hours > 0) {
        return `${hours}h ${minutes % 60}m`;
    } else if (minutes > 0) {
        return `${minutes}m ${seconds % 60}s`;
    } else {
        return `${seconds}s`;
    }
}

// Initialize trading manager
const tradingManager = new TradingManager();
