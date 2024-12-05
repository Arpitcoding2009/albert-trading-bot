from typing import List, Dict, Optional
import autogen
from pydantic import BaseModel, Field
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Pydantic Models for Type Safety
class MarketData(BaseModel):
    symbol: str
    price: float
    volume: float
    timestamp: int
    indicators: Dict[str, float] = Field(default_factory=dict)

class TradingSignal(BaseModel):
    action: str = Field(..., regex='^(BUY|SELL|HOLD)$')
    confidence: float = Field(..., ge=0, le=1)
    target_price: float
    stop_loss: float
    position_size: float = Field(..., ge=0, le=1)

class RiskMetrics(BaseModel):
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    var_95: float  # Value at Risk 95%

# Agent Configuration
config_list = [
    {
        'model': os.getenv('DEFAULT_MODEL', 'gpt-4'),
        'api_key': os.getenv('OPENAI_API_KEY'),
        'temperature': float(os.getenv('TEMPERATURE', '0.7')),
        'max_tokens': int(os.getenv('MAX_TOKENS', '2000'))
    }
]

# Initialize Agents with environment variables
market_analyst = autogen.AssistantAgent(
    name="market_analyst",
    llm_config={
        "config_list": config_list,
        "temperature": float(os.getenv('TEMPERATURE', '0.7')),
    },
    system_message="""You are an expert market analyst. Analyze market data and provide insights."""
)

risk_manager = autogen.AssistantAgent(
    name="risk_manager",
    llm_config={
        "config_list": config_list,
        "temperature": float(os.getenv('TEMPERATURE', '0.3')),
    },
    system_message="""You are a conservative risk manager. Evaluate and manage trading risks."""
)

strategy_agent = autogen.AssistantAgent(
    name="strategy_agent",
    llm_config={
        "config_list": config_list,
        "temperature": float(os.getenv('TEMPERATURE', '0.5')),
    },
    system_message="""You are a trading strategy expert. Generate optimal trading signals."""
)

execution_agent = autogen.AssistantAgent(
    name="execution_agent",
    llm_config={
        "config_list": config_list,
        "temperature": float(os.getenv('TEMPERATURE', '0.2')),
    },
    system_message="""You are a trade execution specialist. Execute trades with optimal timing and minimal slippage."""
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=10,
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config={"work_dir": "coding"},
)

class AgenticTradingSystem:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.agents = {
            'market_analyst': market_analyst,
            'risk_manager': risk_manager,
            'strategy_agent': strategy_agent,
            'execution_agent': execution_agent,
            'user_proxy': user_proxy
        }

    async def analyze_market(self, market_data: MarketData) -> Dict:
        """Market analysis using agentic framework"""
        response = await self.agents['market_analyst'].a_generate(
            message=f"Analyze market data: {market_data.dict()}"
        )
        return response.message

    async def assess_risk(self, market_analysis: Dict, current_portfolio: Dict) -> RiskMetrics:
        """Risk assessment using agentic framework"""
        response = await self.agents['risk_manager'].a_generate(
            message=f"Assess risk for market analysis: {market_analysis} and portfolio: {current_portfolio}"
        )
        return RiskMetrics(**response.message)

    async def generate_trading_signal(self, market_analysis: Dict, risk_metrics: RiskMetrics) -> TradingSignal:
        """Generate trading signals using agentic framework"""
        response = await self.agents['strategy_agent'].a_generate(
            message=f"Generate trading signal based on analysis: {market_analysis} and risk metrics: {risk_metrics}"
        )
        return TradingSignal(**response.message)

    async def execute_trade(self, trading_signal: TradingSignal) -> Dict:
        """Execute trade using agentic framework"""
        response = await self.agents['execution_agent'].a_generate(
            message=f"Execute trade with signal: {trading_signal.dict()}"
        )
        return response.message

    async def trading_cycle(self, market_data: MarketData, current_portfolio: Dict) -> Dict:
        """Complete trading cycle using multi-agent system"""
        try:
            # 1. Market Analysis
            market_analysis = await self.analyze_market(market_data)
            
            # 2. Risk Assessment
            risk_metrics = await self.assess_risk(market_analysis, current_portfolio)
            
            # 3. Signal Generation
            trading_signal = await self.generate_trading_signal(market_analysis, risk_metrics)
            
            # 4. Trade Execution (if signal indicates trading)
            if trading_signal.action != 'HOLD':
                execution_result = await self.execute_trade(trading_signal)
                return {
                    'status': 'success',
                    'market_analysis': market_analysis,
                    'risk_metrics': risk_metrics.dict(),
                    'trading_signal': trading_signal.dict(),
                    'execution_result': execution_result
                }
            
            return {
                'status': 'hold',
                'market_analysis': market_analysis,
                'risk_metrics': risk_metrics.dict(),
                'trading_signal': trading_signal.dict()
            }
            
        except Exception as e:
            self.logger.error(f"Trading cycle error: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
