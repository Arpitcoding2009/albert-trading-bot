// Chart configurations and updates
let portfolioChart, tradesChart;

function initializeCharts() {
    // Initialize Portfolio Performance Chart
    const portfolioCtx = document.getElementById('portfolio-chart').getContext('2d');
    portfolioChart = new Chart(portfolioCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Portfolio Value',
                data: [],
                borderColor: 'rgb(59, 130, 246)',
                backgroundColor: 'rgba(59, 130, 246, 0.1)',
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            return `Value: ${formatCurrency(context.raw)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    }
                },
                y: {
                    beginAtZero: false,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    },
                    ticks: {
                        callback: function(value) {
                            return formatCurrency(value);
                        }
                    }
                }
            }
        }
    });

    // Initialize Trading Activity Chart
    const tradesCtx = document.getElementById('trades-chart').getContext('2d');
    tradesChart = new Chart(tradesCtx, {
        type: 'bar',
        data: {
            labels: [],
            datasets: [{
                label: 'Profit/Loss',
                data: [],
                backgroundColor: function(context) {
                    const value = context.raw;
                    return value >= 0 ? 'rgba(34, 197, 94, 0.5)' : 'rgba(239, 68, 68, 0.5)';
                },
                borderColor: function(context) {
                    const value = context.raw;
                    return value >= 0 ? 'rgb(34, 197, 94)' : 'rgb(239, 68, 68)';
                },
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    callbacks: {
                        label: function(context) {
                            return `P/L: ${formatCurrency(context.raw)}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false
                    }
                },
                y: {
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    },
                    ticks: {
                        callback: function(value) {
                            return formatCurrency(value);
                        }
                    }
                }
            }
        }
    });
}

function updateCharts(data) {
    // Update Portfolio Chart
    const portfolioHistory = data.portfolio.history || [];
    portfolioChart.data.labels = portfolioHistory.map(h => 
        new Date(h.timestamp).toLocaleTimeString()
    );
    portfolioChart.data.datasets[0].data = portfolioHistory.map(h => h.value);
    portfolioChart.update();

    // Update Trades Chart
    const trades = data.performance.trades || [];
    tradesChart.data.labels = trades.map(t => 
        new Date(t.timestamp).toLocaleTimeString()
    );
    tradesChart.data.datasets[0].data = trades.map(t => t.pnl);
    tradesChart.update();
}

function updatePriceChart(marketData) {
    // Add new price data point
    const timestamp = new Date(marketData.timestamp).toLocaleTimeString();
    
    portfolioChart.data.labels.push(timestamp);
    portfolioChart.data.datasets[0].data.push(marketData.price);
    
    // Remove oldest data point if more than 100 points
    if (portfolioChart.data.labels.length > 100) {
        portfolioChart.data.labels.shift();
        portfolioChart.data.datasets[0].data.shift();
    }
    
    portfolioChart.update('none'); // Update without animation for better performance
}

// Utility function for currency formatting
function formatCurrency(value) {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD'
    }).format(value);
}
