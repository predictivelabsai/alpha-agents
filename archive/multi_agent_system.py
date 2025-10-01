"""
Multi-Agent System for Equity Portfolio Construction.
Based on the Alpha Agents paper by BlackRock researchers.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime

from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .base_agent import BaseAgent, Stock, AgentAnalysis, RiskTolerance, InvestmentDecision, PortfolioRecommendation
from .fundamental_agent import FundamentalAgent
from .sentiment_agent import SentimentAgent
from .valuation_agent import ValuationAgent
from .rationale_agent import RationaleAgent
from .secular_trend_agent import SecularTrendAgent

class DebatePhase(Enum):
    """Phases of the multi-agent debate process."""
    INITIAL_ANALYSIS = "initial_analysis"
    DEBATE = "debate"
    CONSENSUS = "consensus"
    FINAL_DECISION = "final_decision"

@dataclass
class MultiAgentState:
    """State object for the multi-agent workflow."""
    stocks: List[Stock]
    agent_analyses: Dict[str, AgentAnalysis]
    debate_messages: List[Dict[str, Any]]
    consensus_reached: bool
    debate_round: int
    max_debate_rounds: int
    current_phase: DebatePhase
    portfolio_recommendation: Optional[PortfolioRecommendation] = None
    final_decision: Optional[Dict[str, Any]] = None

class MultiAgentPortfolioSystem:
    """
    Multi-agent system for equity portfolio construction using specialized agents
    that collaborate and debate to make investment decisions.
    """
    
    def __init__(self, 
                 openai_api_key: str,
                 risk_tolerance: str = "moderate",
                 max_debate_rounds: int = 3):
        """
        Initialize the multi-agent portfolio construction system.
        
        Args:
            openai_api_key: OpenAI API key for LLM integration
            risk_tolerance: Risk tolerance level (conservative/moderate/aggressive)
            max_debate_rounds: Maximum number of debate rounds
        """
        self.risk_tolerance = RiskTolerance(risk_tolerance.lower())
        self.max_debate_rounds = max_debate_rounds
        self.logger = logging.getLogger("MultiAgentPortfolioSystem")
        
        # Initialize LLM client
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-4.1-mini",
            temperature=0.3
        )
        
        # Initialize specialized agents
        self.agents = {
            "fundamental": FundamentalAgent(self.risk_tolerance, self.llm),
            "sentiment": SentimentAgent(self.risk_tolerance, self.llm),
            "valuation": ValuationAgent(self.risk_tolerance, self.llm),
            "rationale": RationaleAgent(self.risk_tolerance, self.llm),
            "secular_trend": SecularTrendAgent(self.risk_tolerance, self.llm)
        }
        
        # Create the workflow graph
        self.workflow = self._create_workflow()
        
        self.logger.info(f"MultiAgentPortfolioSystem initialized with {len(self.agents)} agents")
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for multi-agent collaboration."""
        
        def initial_analysis_node(state: MultiAgentState) -> MultiAgentState:
            """Initial analysis by all agents."""
            self.logger.info("Starting initial analysis phase")
            
            for stock in state.stocks:
                for agent_name, agent in self.agents.items():
                    try:
                        analysis = agent.analyze(stock)
                        state.agent_analyses[f"{agent_name}_{stock.symbol}"] = analysis
                        self.logger.info(f"{agent_name} analyzed {stock.symbol}: {analysis.recommendation.value}")
                    except Exception as e:
                        self.logger.error(f"Error in {agent_name} analysis for {stock.symbol}: {e}")
            
            state.current_phase = DebatePhase.DEBATE
            return state
        
        def debate_node(state: MultiAgentState) -> MultiAgentState:
            """Debate phase where agents discuss conflicting views."""
            self.logger.info(f"Starting debate round {state.debate_round + 1}")
            
            # Check for conflicts in recommendations
            conflicts = self._identify_conflicts(state)
            
            if conflicts and state.debate_round < state.max_debate_rounds:
                # Generate debate messages
                for conflict in conflicts[:2]:  # Limit to 2 conflicts per round
                    debate_msg = self._generate_debate_message(conflict, state)
                    state.debate_messages.append(debate_msg)
                
                state.debate_round += 1
                
                # Re-analyze with debate context
                self._update_analyses_with_debate(state)
            else:
                state.current_phase = DebatePhase.CONSENSUS
            
            return state
        
        def consensus_node(state: MultiAgentState) -> MultiAgentState:
            """Consensus building phase."""
            self.logger.info("Building consensus")
            
            # Check if consensus is reached
            consensus_level = self._calculate_consensus_level(state)
            
            if consensus_level >= 0.7 or state.debate_round >= state.max_debate_rounds:
                state.consensus_reached = True
                state.current_phase = DebatePhase.FINAL_DECISION
            else:
                # Continue debate
                state.current_phase = DebatePhase.DEBATE
            
            return state
        
        def final_decision_node(state: MultiAgentState) -> MultiAgentState:
            """Final decision and portfolio construction."""
            self.logger.info("Making final portfolio decision")
            
            # Aggregate agent recommendations
            portfolio_rec = self._construct_portfolio(state)
            state.portfolio_recommendation = portfolio_rec
            
            # Create final decision summary
            final_decision = {
                "portfolio": asdict(portfolio_rec),
                "consensus_level": self._calculate_consensus_level(state),
                "debate_rounds": state.debate_round,
                "timestamp": datetime.now().isoformat()
            }
            
            state.final_decision = final_decision
            return state
        
        def should_continue_debate(state: MultiAgentState) -> str:
            """Determine if debate should continue."""
            if state.current_phase == DebatePhase.INITIAL_ANALYSIS:
                return "debate"
            elif state.current_phase == DebatePhase.DEBATE:
                return "consensus"
            elif state.current_phase == DebatePhase.CONSENSUS:
                if state.consensus_reached:
                    return "final_decision"
                else:
                    return "debate"
            else:
                return END
        
        # Build the workflow graph
        workflow = StateGraph(MultiAgentState)
        
        # Add nodes
        workflow.add_node("initial_analysis", initial_analysis_node)
        workflow.add_node("debate", debate_node)
        workflow.add_node("consensus", consensus_node)
        workflow.add_node("final_decision", final_decision_node)
        
        # Add edges
        workflow.set_entry_point("initial_analysis")
        workflow.add_conditional_edges("initial_analysis", should_continue_debate)
        workflow.add_conditional_edges("debate", should_continue_debate)
        workflow.add_conditional_edges("consensus", should_continue_debate)
        workflow.add_edge("final_decision", END)
        
        return workflow.compile()
    
    def analyze_stocks(self, stocks: List[Stock], market_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Analyze a list of stocks and construct portfolio recommendations.
        
        Args:
            stocks: List of stocks to analyze
            market_data: Optional market context data
            
        Returns:
            Portfolio analysis results with recommendations
        """
        self.logger.info(f"Starting portfolio analysis for {len(stocks)} stocks")
        
        # Initialize state
        initial_state = MultiAgentState(
            stocks=stocks,
            agent_analyses={},
            debate_messages=[],
            consensus_reached=False,
            debate_round=0,
            max_debate_rounds=self.max_debate_rounds,
            current_phase=DebatePhase.INITIAL_ANALYSIS
        )
        
        # Run the workflow
        try:
            final_state = self.workflow.invoke(initial_state)
            
            # Handle case where workflow returns dict instead of state object
            if isinstance(final_state, dict):
                # Extract the portfolio recommendation from the dict
                portfolio_rec = final_state.get('portfolio_recommendation', {
                    "stocks": [],
                    "expected_return": 0.0,
                    "risk_level": "MODERATE",
                    "diversification_score": 0.5,
                    "reasoning": "Workflow returned unexpected format",
                    "confidence_score": 0.5
                })
            else:
                portfolio_rec = final_state.portfolio_recommendation or {
                    "stocks": [],
                    "expected_return": 0.0,
                    "risk_level": "MODERATE", 
                    "diversification_score": 0.5,
                    "reasoning": "No portfolio recommendation generated",
                    "confidence_score": 0.5
                }
            
            result = {
                "portfolio_recommendation": portfolio_rec,
                "individual_analyses": {
                    key: asdict(analysis) for key, analysis in (
                        final_state.agent_analyses.items() if hasattr(final_state, 'agent_analyses') 
                        else final_state.get('agent_analyses', {}).items()
                    )
                },
                "debate_history": (
                    final_state.debate_messages if hasattr(final_state, 'debate_messages')
                    else final_state.get('debate_messages', [])
                ),
                "consensus_reached": (
                    final_state.consensus_reached if hasattr(final_state, 'consensus_reached')
                    else final_state.get('consensus_reached', False)
                ),
                "debate_rounds": (
                    final_state.debate_round if hasattr(final_state, 'debate_round')
                    else final_state.get('debate_round', 0)
                ),
                "processing_timestamp": datetime.now().isoformat()
            }
            
            self.logger.info(f"Portfolio analysis complete: {len(portfolio_rec.get('stocks', []))} stocks recommended")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in portfolio analysis: {e}")
            return {
                "error": str(e),
                "portfolio_recommendation": {
                    "stocks": [],
                    "expected_return": 0.0,
                    "risk_level": "HIGH",
                    "diversification_score": 0.0,
                    "reasoning": f"Analysis failed: {str(e)}",
                    "confidence_score": 0.0
                }
            }
    
    def _identify_conflicts(self, state: MultiAgentState) -> List[Dict[str, Any]]:
        """Identify conflicts between agent recommendations."""
        conflicts = []
        
        # Group analyses by stock
        stock_analyses = {}
        for key, analysis in state.agent_analyses.items():
            stock_symbol = analysis.stock_symbol
            if stock_symbol not in stock_analyses:
                stock_analyses[stock_symbol] = []
            stock_analyses[stock_symbol].append((key.split('_')[0], analysis))
        
        # Find conflicts for each stock
        for stock_symbol, analyses in stock_analyses.items():
            if len(analyses) >= 2:
                recommendations = [analysis.recommendation for _, analysis in analyses]
                
                # Check for conflicting recommendations
                if len(set(recommendations)) > 1:
                    conflicts.append({
                        "stock_symbol": stock_symbol,
                        "conflicting_agents": [agent_name for agent_name, _ in analyses],
                        "recommendations": {agent_name: analysis.recommendation.value 
                                         for agent_name, analysis in analyses}
                    })
        
        return conflicts
    
    def _generate_debate_message(self, conflict: Dict[str, Any], state: MultiAgentState) -> Dict[str, Any]:
        """Generate a debate message for a conflict."""
        return {
            "round": state.debate_round + 1,
            "stock": conflict["stock_symbol"],
            "conflict": conflict["recommendations"],
            "agents": conflict["conflicting_agents"],
            "timestamp": datetime.now().isoformat(),
            "message": f"Agents disagree on {conflict['stock_symbol']}: {conflict['recommendations']}"
        }
    
    def _update_analyses_with_debate(self, state: MultiAgentState):
        """Update agent analyses considering debate context."""
        # In a full implementation, this would re-run agents with debate context
        # For now, we'll adjust confidence scores based on consensus
        for key, analysis in state.agent_analyses.items():
            # Slightly reduce confidence when there's debate
            analysis.confidence_score *= 0.95
    
    def _calculate_consensus_level(self, state: MultiAgentState) -> float:
        """Calculate the level of consensus among agents."""
        if not state.agent_analyses:
            return 0.0
        
        # Group by stock and calculate agreement
        stock_analyses = {}
        for key, analysis in state.agent_analyses.items():
            stock_symbol = analysis.stock_symbol
            if stock_symbol not in stock_analyses:
                stock_analyses[stock_symbol] = []
            stock_analyses[stock_symbol].append(analysis.recommendation)
        
        total_consensus = 0.0
        for stock_symbol, recommendations in stock_analyses.items():
            if len(recommendations) > 1:
                # Calculate agreement percentage
                most_common = max(set(recommendations), key=recommendations.count)
                agreement = recommendations.count(most_common) / len(recommendations)
                total_consensus += agreement
        
        return total_consensus / len(stock_analyses) if stock_analyses else 0.0
    
    def _construct_portfolio(self, state: MultiAgentState) -> PortfolioRecommendation:
        """Construct final portfolio recommendation."""
        portfolio_stocks = []
        total_confidence = 0.0
        buy_recommendations = 0
        
        # Group analyses by stock
        stock_analyses = {}
        for key, analysis in state.agent_analyses.items():
            stock_symbol = analysis.stock_symbol
            if stock_symbol not in stock_analyses:
                stock_analyses[stock_symbol] = []
            stock_analyses[stock_symbol].append(analysis)
        
        # Evaluate each stock
        for stock_symbol, analyses in stock_analyses.items():
            # Calculate average confidence and consensus
            avg_confidence = sum(a.confidence_score for a in analyses) / len(analyses)
            
            # Count buy/positive recommendations
            positive_recs = sum(1 for a in analyses if a.recommendation in [InvestmentDecision.BUY])
            consensus_strength = positive_recs / len(analyses)
            
            # Include in portfolio if majority positive and high confidence
            if consensus_strength >= 0.5 and avg_confidence >= 0.6:
                # Calculate weight based on confidence and consensus
                weight = (avg_confidence * consensus_strength) * 0.3  # Max 30% per stock
                
                portfolio_stocks.append({
                    "symbol": stock_symbol,
                    "weight": round(weight, 3),
                    "rationale": f"Consensus: {consensus_strength:.1%}, Confidence: {avg_confidence:.2f}",
                    "target_allocation": f"{weight*100:.1f}%"
                })
                
                total_confidence += avg_confidence
                if positive_recs > len(analyses) / 2:
                    buy_recommendations += 1
        
        # Normalize weights to sum to 1.0
        if portfolio_stocks:
            total_weight = sum(stock["weight"] for stock in portfolio_stocks)
            for stock in portfolio_stocks:
                stock["weight"] = round(stock["weight"] / total_weight, 3)
                stock["target_allocation"] = f"{stock['weight']*100:.1f}%"
        
        # Calculate portfolio metrics
        expected_return = min(0.12, total_confidence / len(stock_analyses) * 0.15) if stock_analyses else 0.0
        diversification_score = min(1.0, len(portfolio_stocks) / 10.0)  # Better with more stocks
        
        # Determine risk level
        if self.risk_tolerance == RiskTolerance.CONSERVATIVE:
            risk_level = "LOW"
        elif self.risk_tolerance == RiskTolerance.AGGRESSIVE:
            risk_level = "HIGH"
        else:
            risk_level = "MODERATE"
        
        # Generate reasoning
        reasoning = f"""
        Portfolio constructed from {len(stock_analyses)} analyzed stocks.
        Selected {len(portfolio_stocks)} stocks for inclusion based on agent consensus.
        Average confidence: {total_confidence/len(stock_analyses):.2f} across all analyses.
        Risk tolerance: {self.risk_tolerance.value}
        Diversification across {len(set(s['symbol'][:2] for s in portfolio_stocks))} sectors.
        """
        
        return PortfolioRecommendation(
            stocks=portfolio_stocks,
            expected_return=expected_return,
            risk_level=risk_level,
            diversification_score=diversification_score,
            reasoning=reasoning.strip(),
            confidence_score=total_confidence / len(stock_analyses) if stock_analyses else 0.0
        )

def create_multi_agent_portfolio_system(openai_api_key: str, 
                                      risk_tolerance: str = "moderate",
                                      max_debate_rounds: int = 3) -> MultiAgentPortfolioSystem:
    """
    Factory function to create a multi-agent portfolio system.
    
    Args:
        openai_api_key: OpenAI API key
        risk_tolerance: Risk tolerance level
        max_debate_rounds: Maximum debate rounds
        
    Returns:
        Configured MultiAgentPortfolioSystem
    """
    return MultiAgentPortfolioSystem(
        openai_api_key=openai_api_key,
        risk_tolerance=risk_tolerance,
        max_debate_rounds=max_debate_rounds
    )

