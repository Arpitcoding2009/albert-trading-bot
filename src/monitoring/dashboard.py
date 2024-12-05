import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List
import asyncio

class TradingDashboard:
    def __init__(self, config: Dict):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Initialize Dash app
        self.app = dash.Dash(__name__)
        
        # Performance metrics
        self.update_interval = 5000  # 5 seconds
        self.max_data_points = 1000
        
        # Data storage
        self.performance_data = pd.DataFrame()
        self.risk_metrics = pd.DataFrame()
        self.trade_history = pd.DataFrame()
        
        # Initialize dashboard
        self._init_dashboard()
        
    def _init_dashboard(self):
        """Initialize dashboard layout"""
        try:
            self.app.layout = html.Div([
                # Header
                html.Div([
                    html.H1('Albert Trading Bot Dashboard'),
                    html.P('Real-time Trading Performance Monitor')
                ], className='header'),
                
                # Main content
                html.Div([
                    # Performance metrics
                    html.Div([
                        html.H2('Performance Metrics'),
                        self._create_performance_cards(),
                        dcc.Graph(id='performance-chart'),
                        dcc.Interval(
                            id='performance-interval',
                            interval=self.update_interval
                        )
                    ], className='metrics-container'),
                    
                    # Risk metrics
                    html.Div([
                        html.H2('Risk Analytics'),
                        self._create_risk_cards(),
                        dcc.Graph(id='risk-chart'),
                        dcc.Interval(
                            id='risk-interval',
                            interval=self.update_interval
                        )
                    ], className='risk-container'),
                    
                    # Trading activity
                    html.Div([
                        html.H2('Trading Activity'),
                        self._create_trade_table(),
                        dcc.Graph(id='trade-chart'),
                        dcc.Interval(
                            id='trade-interval',
                            interval=self.update_interval
                        )
                    ], className='trading-container')
                ], className='main-content')
            ], className='dashboard-container')
            
            # Register callbacks
            self._register_callbacks()
            
        except Exception as e:
            self.logger.error(f"Dashboard initialization error: {str(e)}")

    def _create_performance_cards(self) -> html.Div:
        """Create performance metric cards"""
        try:
            return html.Div([
                html.Div([
                    html.H3('Total Return'),
                    html.P(id='total-return')
                ], className='metric-card'),
                
                html.Div([
                    html.H3('Daily PnL'),
                    html.P(id='daily-pnl')
                ], className='metric-card'),
                
                html.Div([
                    html.H3('Sharpe Ratio'),
                    html.P(id='sharpe-ratio')
                ], className='metric-card'),
                
                html.Div([
                    html.H3('Win Rate'),
                    html.P(id='win-rate')
                ], className='metric-card')
            ], className='metrics-grid')
            
        except Exception as e:
            self.logger.error(f"Performance cards creation error: {str(e)}")
            return html.Div()

    def _create_risk_cards(self) -> html.Div:
        """Create risk metric cards"""
        try:
            return html.Div([
                html.Div([
                    html.H3('Value at Risk (95%)'),
                    html.P(id='var-95')
                ], className='metric-card'),
                
                html.Div([
                    html.H3('Max Drawdown'),
                    html.P(id='max-drawdown')
                ], className='metric-card'),
                
                html.Div([
                    html.H3('Portfolio Beta'),
                    html.P(id='portfolio-beta')
                ], className='metric-card'),
                
                html.Div([
                    html.H3('Volatility'),
                    html.P(id='portfolio-volatility')
                ], className='metric-card')
            ], className='metrics-grid')
            
        except Exception as e:
            self.logger.error(f"Risk cards creation error: {str(e)}")
            return html.Div()

    def _create_trade_table(self) -> html.Div:
        """Create trading activity table"""
        try:
            return html.Div([
                dash.dash_table.DataTable(
                    id='trade-table',
                    columns=[
                        {'name': 'Time', 'id': 'time'},
                        {'name': 'Symbol', 'id': 'symbol'},
                        {'name': 'Side', 'id': 'side'},
                        {'name': 'Price', 'id': 'price'},
                        {'name': 'Size', 'id': 'size'},
                        {'name': 'PnL', 'id': 'pnl'}
                    ],
                    style_table={'height': '300px', 'overflowY': 'auto'},
                    style_cell={'textAlign': 'left'},
                    style_header={
                        'backgroundColor': 'rgb(30, 30, 30)',
                        'color': 'white'
                    },
                    style_data={
                        'backgroundColor': 'rgb(50, 50, 50)',
                        'color': 'white'
                    }
                )
            ], className='table-container')
            
        except Exception as e:
            self.logger.error(f"Trade table creation error: {str(e)}")
            return html.Div()

    def _register_callbacks(self):
        """Register dashboard callbacks"""
        try:
            # Performance metrics callbacks
            @self.app.callback(
                [Output('total-return', 'children'),
                 Output('daily-pnl', 'children'),
                 Output('sharpe-ratio', 'children'),
                 Output('win-rate', 'children')],
                [Input('performance-interval', 'n_intervals')]
            )
            def update_performance_metrics(n):
                return self._get_performance_metrics()
            
            # Risk metrics callbacks
            @self.app.callback(
                [Output('var-95', 'children'),
                 Output('max-drawdown', 'children'),
                 Output('portfolio-beta', 'children'),
                 Output('portfolio-volatility', 'children')],
                [Input('risk-interval', 'n_intervals')]
            )
            def update_risk_metrics(n):
                return self._get_risk_metrics()
            
            # Performance chart callback
            @self.app.callback(
                Output('performance-chart', 'figure'),
                [Input('performance-interval', 'n_intervals')]
            )
            def update_performance_chart(n):
                return self._create_performance_chart()
            
            # Risk chart callback
            @self.app.callback(
                Output('risk-chart', 'figure'),
                [Input('risk-interval', 'n_intervals')]
            )
            def update_risk_chart(n):
                return self._create_risk_chart()
            
            # Trade table callback
            @self.app.callback(
                Output('trade-table', 'data'),
                [Input('trade-interval', 'n_intervals')]
            )
            def update_trade_table(n):
                return self._get_trade_data()
            
            # Trade chart callback
            @self.app.callback(
                Output('trade-chart', 'figure'),
                [Input('trade-interval', 'n_intervals')]
            )
            def update_trade_chart(n):
                return self._create_trade_chart()
            
        except Exception as e:
            self.logger.error(f"Callback registration error: {str(e)}")

    def _get_performance_metrics(self) -> tuple:
        """Get current performance metrics"""
        try:
            if len(self.performance_data) > 0:
                total_return = f"{self.performance_data['total_return'].iloc[-1]:.2f}%"
                daily_pnl = f"${self.performance_data['daily_pnl'].iloc[-1]:,.2f}"
                sharpe = f"{self.performance_data['sharpe_ratio'].iloc[-1]:.2f}"
                win_rate = f"{self.performance_data['win_rate'].iloc[-1]:.1f}%"
            else:
                total_return = "0.00%"
                daily_pnl = "$0.00"
                sharpe = "0.00"
                win_rate = "0.0%"
            
            return total_return, daily_pnl, sharpe, win_rate
            
        except Exception as e:
            self.logger.error(f"Performance metrics error: {str(e)}")
            return "0.00%", "$0.00", "0.00", "0.0%"

    def _get_risk_metrics(self) -> tuple:
        """Get current risk metrics"""
        try:
            if len(self.risk_metrics) > 0:
                var = f"{self.risk_metrics['var_95'].iloc[-1]:.2f}%"
                drawdown = f"{self.risk_metrics['max_drawdown'].iloc[-1]:.2f}%"
                beta = f"{self.risk_metrics['beta'].iloc[-1]:.2f}"
                vol = f"{self.risk_metrics['volatility'].iloc[-1]:.2f}%"
            else:
                var = "0.00%"
                drawdown = "0.00%"
                beta = "0.00"
                vol = "0.00%"
            
            return var, drawdown, beta, vol
            
        except Exception as e:
            self.logger.error(f"Risk metrics error: {str(e)}")
            return "0.00%", "0.00%", "0.00", "0.00%"

    def _create_performance_chart(self) -> go.Figure:
        """Create performance chart"""
        try:
            fig = go.Figure()
            
            if len(self.performance_data) > 0:
                # Add equity curve
                fig.add_trace(go.Scatter(
                    x=self.performance_data.index,
                    y=self.performance_data['total_return'],
                    name='Equity Curve',
                    line=dict(color='#00ff00', width=2)
                ))
                
                # Add drawdown
                fig.add_trace(go.Scatter(
                    x=self.performance_data.index,
                    y=self.performance_data['drawdown'],
                    name='Drawdown',
                    line=dict(color='#ff0000', width=1),
                    fill='tonexty'
                ))
            
            fig.update_layout(
                title='Portfolio Performance',
                xaxis_title='Time',
                yaxis_title='Return (%)',
                template='plotly_dark',
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Performance chart error: {str(e)}")
            return go.Figure()

    def _create_risk_chart(self) -> go.Figure:
        """Create risk metrics chart"""
        try:
            fig = go.Figure()
            
            if len(self.risk_metrics) > 0:
                # Add VaR
                fig.add_trace(go.Scatter(
                    x=self.risk_metrics.index,
                    y=self.risk_metrics['var_95'],
                    name='VaR (95%)',
                    line=dict(color='#ff0000', width=2)
                ))
                
                # Add volatility
                fig.add_trace(go.Scatter(
                    x=self.risk_metrics.index,
                    y=self.risk_metrics['volatility'],
                    name='Volatility',
                    line=dict(color='#ffff00', width=2)
                ))
            
            fig.update_layout(
                title='Risk Metrics',
                xaxis_title='Time',
                yaxis_title='Value (%)',
                template='plotly_dark',
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Risk chart error: {str(e)}")
            return go.Figure()

    def _create_trade_chart(self) -> go.Figure:
        """Create trading activity chart"""
        try:
            fig = go.Figure()
            
            if len(self.trade_history) > 0:
                # Add buy trades
                buys = self.trade_history[self.trade_history['side'] == 'buy']
                fig.add_trace(go.Scatter(
                    x=buys.index,
                    y=buys['price'],
                    mode='markers',
                    name='Buy',
                    marker=dict(
                        color='#00ff00',
                        size=10,
                        symbol='triangle-up'
                    )
                ))
                
                # Add sell trades
                sells = self.trade_history[self.trade_history['side'] == 'sell']
                fig.add_trace(go.Scatter(
                    x=sells.index,
                    y=sells['price'],
                    mode='markers',
                    name='Sell',
                    marker=dict(
                        color='#ff0000',
                        size=10,
                        symbol='triangle-down'
                    )
                ))
            
            fig.update_layout(
                title='Trading Activity',
                xaxis_title='Time',
                yaxis_title='Price',
                template='plotly_dark',
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Trade chart error: {str(e)}")
            return go.Figure()

    def _get_trade_data(self) -> List[Dict]:
        """Get recent trade data"""
        try:
            if len(self.trade_history) > 0:
                recent_trades = self.trade_history.tail(100)
                return recent_trades.to_dict('records')
            return []
            
        except Exception as e:
            self.logger.error(f"Trade data error: {str(e)}")
            return []

    async def update_data(self, performance_data: pd.DataFrame, 
                         risk_metrics: pd.DataFrame,
                         trade_history: pd.DataFrame):
        """Update dashboard data"""
        try:
            self.performance_data = performance_data
            self.risk_metrics = risk_metrics
            self.trade_history = trade_history
            
        except Exception as e:
            self.logger.error(f"Data update error: {str(e)}")

    def run(self, host: str = '0.0.0.0', port: int = 8050):
        """Run dashboard server"""
        try:
            self.app.run_server(host=host, port=port)
        except Exception as e:
            self.logger.error(f"Dashboard server error: {str(e)}")

    def shutdown(self):
        """Shutdown dashboard server"""
        try:
            self.app.server.shutdown()
        except Exception as e:
            self.logger.error(f"Dashboard shutdown error: {str(e)}")
