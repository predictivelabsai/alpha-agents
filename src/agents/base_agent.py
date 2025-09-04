"""
Base Agent class for Alpha Agents equity portfolio construction system.
Adapted from the Alpha Agents paper for stock selection and portfolio management.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
from datetime import datetime
import logging

class InvestmentDecision(Enum):
    """Investment decision types."""
    BUY = "buy"
    SELL = "sell" 
    HOLD = "hold"
    AVOID = "avoid"

class RiskTolerance(Enum):
    """Risk tolerance levels for portfolio construction."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

@dataclass
class Stock:
    """Stock information for analysis."""
    symbol: str
    company_name: str
    sector: str
    market_cap: float
    current_price: float
    pe_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    beta: Optional[float] = None
    volume: Optional[int] = None
    
@dataclass
class AgentAnalysis:
    """Analysis result from an individual agent."""
    agent_name: str
    stock_symbol: str
    recommendation: InvestmentDecision
    confidence_score: float  # 0.0 to 1.0
    target_price: Optional[float] = None
    risk_assessment: str = "MODERATE"
    key_factors: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)
    reasoning: str = ""
    analysis_timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
@dataclass
class PortfolioRecommendation:
    """Portfolio construction recommendation."""
    stocks: List[Dict[str, Any]]  # List of {symbol, weight, rationale}
    expected_return: float
    risk_level: str
    diversification_score: float
    reasoning: str
    confidence_score: float
    rebalancing_frequency: str = "quarterly"

class BaseAgent(ABC):
    """
    Base class for all Alpha Agents in the equity portfolio construction system.
    Each agent specializes in a specific aspect of stock analysis.
    """
    
    def __init__(self, risk_tolerance: RiskTolerance, llm_client=None):
        """
        Initialize the base agent.
        
        Args:
            risk_tolerance: Risk tolerance level for analysis
            llm_client: Optional LLM client for enhanced analysis
        """
        self.risk_tolerance = risk_tolerance
        self.llm_client = llm_client
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @abstractmethod
    def analyze(self, stock: Stock, market_data: Optional[Dict] = None) -> AgentAnalysis:
        """
        Analyze a stock and provide investment recommendation.
        
        Args:
            stock: Stock information to analyze
            market_data: Optional market context data
            
        Returns:
            AgentAnalysis with recommendation and reasoning
        """
        pass
    
    def get_agent_type(self) -> str:
        """Return the agent type identifier."""
        return self.__class__.__name__.replace("Agent", "").lower()
    
    def adjust_for_risk_tolerance(self, base_confidence: float) -> float:
        """
        Adjust confidence score based on risk tolerance.
        
        Args:
            base_confidence: Base confidence score
            
        Returns:
            Adjusted confidence score
        """
        if self.risk_tolerance == RiskTolerance.CONSERVATIVE:
            return base_confidence * 0.8  # More cautious
        elif self.risk_tolerance == RiskTolerance.AGGRESSIVE:
            return min(base_confidence * 1.2, 1.0)  # More confident
        else:
            return base_confidence  # Moderate - no adjustment
    
    def format_currency(self, amount: float) -> str:
        """Format currency amounts."""
        if amount >= 1e9:
            return f"${amount/1e9:.1f}B"
        elif amount >= 1e6:
            return f"${amount/1e6:.1f}M"
        elif amount >= 1e3:
            return f"${amount/1e3:.1f}K"
        else:
            return f"${amount:.2f}"

