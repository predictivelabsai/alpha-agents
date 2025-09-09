"""
Lohusalu Capital Management - 3-Agent System for Equity Portfolio Construction
"""

from .base_agent import BaseAgent, Stock, AgentAnalysis, InvestmentDecision, RiskTolerance, PortfolioRecommendation
from .fundamental_agent import FundamentalAgent
from .rationale_agent import RationaleAgent
from .ranker_agent import RankerAgent

# Legacy compatibility
def create_multi_agent_portfolio_system():
    """Legacy function for compatibility"""
    return None

class MultiAgentPortfolioSystem:
    """Legacy class for compatibility"""
    pass

__all__ = [
    'BaseAgent',
    'Stock', 
    'AgentAnalysis',
    'InvestmentDecision',
    'RiskTolerance',
    'PortfolioRecommendation',
    'FundamentalAgent',
    'RationaleAgent',
    'RankerAgent',
    'MultiAgentPortfolioSystem',
    'create_multi_agent_portfolio_system'
]

