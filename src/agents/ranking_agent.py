"""
Ranking Agent - Aggregates analysis from all 5 agents and provides final recommendations
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from .base_agent import BaseAgent, Stock, InvestmentDecision, RiskTolerance

# Import prompts
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from prompts.ranking_agent import (
    RANKING_ANALYSIS_PROMPT,
    PORTFOLIO_RANKING_PROMPT,
    CONSENSUS_BUILDING_PROMPT,
    RISK_ASSESSMENT_PROMPT
)

class RankingAgent(BaseAgent):
    """
    Ranking Agent that synthesizes analysis from all other agents
    and provides final investment recommendations
    """
    
    def __init__(self, openai_api_key: str, risk_tolerance: RiskTolerance = RiskTolerance.MODERATE):
        super().__init__(openai_api_key, risk_tolerance)
        self.agent_type = "Ranking Agent"
        self.specialization = "Multi-agent synthesis and final recommendations"
        
        # Initialize LLM
        try:
            self.llm = ChatOpenAI(
                model="gpt-4",
                temperature=0.1,
                openai_api_key=openai_api_key
            )
        except Exception as e:
            logging.warning(f"Failed to initialize OpenAI client: {e}")
            self.llm = None
    
    def analyze_stock(self, stock: Stock, agent_analyses: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Synthesize analysis from all agents and provide final recommendation
        
        Args:
            stock: Stock object with basic information
            agent_analyses: Dictionary containing analysis from all 5 agents
            
        Returns:
            Dictionary containing final ranking analysis
        """
        try:
            if self.llm:
                return self._llm_analysis(stock, agent_analyses)
            else:
                return self._fallback_analysis(stock, agent_analyses)
        except Exception as e:
            logging.error(f"Error in ranking analysis: {e}")
            return self._fallback_analysis(stock, agent_analyses)
    
    def _llm_analysis(self, stock: Stock, agent_analyses: Dict[str, Dict]) -> Dict[str, Any]:
        """LLM-powered ranking analysis"""
        
        # Extract key information from each agent
        fundamental = agent_analyses.get('fundamental', {})
        sentiment = agent_analyses.get('sentiment', {})
        valuation = agent_analyses.get('valuation', {})
        rationale = agent_analyses.get('rationale', {})
        trend = agent_analyses.get('secular_trend', {})
        
        # Format prompt with agent analyses
        prompt = RANKING_ANALYSIS_PROMPT.format(
            stock_symbol=stock.symbol,
            company_name=stock.company_name,
            agent_analyses=json.dumps(agent_analyses, indent=2),
            fundamental_recommendation=fundamental.get('recommendation', 'HOLD'),
            fundamental_confidence=fundamental.get('confidence_score', 0.5),
            fundamental_insight=fundamental.get('reasoning', 'No analysis available'),
            sentiment_recommendation=sentiment.get('recommendation', 'HOLD'),
            sentiment_confidence=sentiment.get('confidence_score', 0.5),
            sentiment_insight=sentiment.get('reasoning', 'No analysis available'),
            valuation_recommendation=valuation.get('recommendation', 'HOLD'),
            valuation_confidence=valuation.get('confidence_score', 0.5),
            valuation_insight=valuation.get('reasoning', 'No analysis available'),
            rationale_recommendation=rationale.get('recommendation', 'HOLD'),
            rationale_confidence=rationale.get('confidence_score', 0.5),
            business_quality_score=rationale.get('business_quality_score', 35),
            rationale_insight=rationale.get('reasoning', 'No analysis available'),
            trend_recommendation=trend.get('recommendation', 'HOLD'),
            trend_confidence=trend.get('confidence_score', 0.5),
            trend_score=trend.get('trend_score', 5),
            trend_insight=trend.get('reasoning', 'No analysis available')
        )
        
        # Get LLM response
        response = self.llm.invoke([HumanMessage(content=prompt)])
        analysis_text = response.content
        
        # Parse the response and extract structured data
        return self._parse_ranking_response(analysis_text, agent_analyses)
    
    def _fallback_analysis(self, stock: Stock, agent_analyses: Dict[str, Dict]) -> Dict[str, Any]:
        """Fallback analysis without LLM"""
        
        # Calculate composite score based on agent recommendations
        recommendations = []
        confidences = []
        
        for agent_name, analysis in agent_analyses.items():
            rec = analysis.get('recommendation', 'HOLD')
            conf = analysis.get('confidence_score', 0.5)
            
            # Convert recommendation to numeric score
            rec_score = self._recommendation_to_score(rec)
            recommendations.append(rec_score * conf)
            confidences.append(conf)
        
        # Calculate weighted average
        if confidences:
            composite_score = sum(recommendations) / len(recommendations)
            avg_confidence = sum(confidences) / len(confidences)
        else:
            composite_score = 0.5
            avg_confidence = 0.5
        
        # Determine final recommendation
        final_rec = self._score_to_recommendation(composite_score)
        
        return {
            'agent_type': self.agent_type,
            'stock_symbol': stock.symbol,
            'company_name': stock.company_name,
            'recommendation': final_rec,
            'confidence_score': avg_confidence,
            'composite_score': int(composite_score * 100),
            'price_target': None,
            'investment_thesis': f"Consensus recommendation based on {len(agent_analyses)} agent analyses",
            'key_catalysts': ['Multi-agent consensus', 'Diversified analysis approach', 'Risk-adjusted assessment'],
            'key_risks': ['Agent disagreement', 'Market volatility', 'Execution risk'],
            'agent_consensus': 'Moderate' if len(set([a.get('recommendation', 'HOLD') for a in agent_analyses.values()])) <= 2 else 'Low',
            'reasoning': f"Final recommendation synthesized from {len(agent_analyses)} specialized agents with average confidence of {avg_confidence:.2f}",
            'analysis_timestamp': datetime.now().isoformat(),
            'agent_breakdown': agent_analyses
        }
    
    def rank_portfolio(self, stock_analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank multiple stocks for portfolio construction
        
        Args:
            stock_analyses: List of complete stock analyses from all agents
            
        Returns:
            List of stocks ranked by attractiveness
        """
        try:
            if self.llm:
                return self._llm_portfolio_ranking(stock_analyses)
            else:
                return self._fallback_portfolio_ranking(stock_analyses)
        except Exception as e:
            logging.error(f"Error in portfolio ranking: {e}")
            return self._fallback_portfolio_ranking(stock_analyses)
    
    def _llm_portfolio_ranking(self, stock_analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """LLM-powered portfolio ranking"""
        
        prompt = PORTFOLIO_RANKING_PROMPT.format(
            stock_analyses=json.dumps(stock_analyses, indent=2)
        )
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        ranking_text = response.content
        
        # Parse and return ranked list
        return self._parse_portfolio_ranking(ranking_text, stock_analyses)
    
    def _fallback_portfolio_ranking(self, stock_analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback portfolio ranking without LLM"""
        
        # Sort by composite score
        ranked_stocks = sorted(
            stock_analyses,
            key=lambda x: x.get('composite_score', 0),
            reverse=True
        )
        
        # Add ranking information
        for i, stock in enumerate(ranked_stocks):
            stock['portfolio_rank'] = i + 1
            stock['position_size'] = 'Large' if i < 3 else 'Medium' if i < 7 else 'Small'
        
        return ranked_stocks
    
    def build_consensus(self, agent_recommendations: Dict[str, str], agent_confidences: Dict[str, float]) -> Dict[str, Any]:
        """
        Build consensus when agents disagree
        
        Args:
            agent_recommendations: Dict of agent names to recommendations
            agent_confidences: Dict of agent names to confidence scores
            
        Returns:
            Consensus recommendation with reasoning
        """
        try:
            if self.llm:
                return self._llm_consensus_building(agent_recommendations, agent_confidences)
            else:
                return self._fallback_consensus_building(agent_recommendations, agent_confidences)
        except Exception as e:
            logging.error(f"Error in consensus building: {e}")
            return self._fallback_consensus_building(agent_recommendations, agent_confidences)
    
    def _recommendation_to_score(self, recommendation: str) -> float:
        """Convert recommendation to numeric score"""
        rec_map = {
            'STRONG_SELL': 0.0,
            'SELL': 0.25,
            'HOLD': 0.5,
            'BUY': 0.75,
            'STRONG_BUY': 1.0
        }
        return rec_map.get(recommendation.upper(), 0.5)
    
    def _score_to_recommendation(self, score: float) -> str:
        """Convert numeric score to recommendation"""
        if score >= 0.8:
            return 'STRONG_BUY'
        elif score >= 0.6:
            return 'BUY'
        elif score >= 0.4:
            return 'HOLD'
        elif score >= 0.2:
            return 'SELL'
        else:
            return 'STRONG_SELL'
    
    def _parse_ranking_response(self, response_text: str, agent_analyses: Dict[str, Dict]) -> Dict[str, Any]:
        """Parse LLM response into structured ranking data"""
        
        # Default values
        result = {
            'agent_type': self.agent_type,
            'recommendation': 'HOLD',
            'confidence_score': 0.5,
            'composite_score': 50,
            'price_target': None,
            'investment_thesis': 'Multi-agent synthesis analysis',
            'key_catalysts': [],
            'key_risks': [],
            'agent_consensus': 'Moderate',
            'reasoning': response_text,
            'analysis_timestamp': datetime.now().isoformat(),
            'agent_breakdown': agent_analyses
        }
        
        # Try to extract structured information from response
        lines = response_text.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('- **Final Recommendation**:'):
                result['recommendation'] = line.split(':')[1].strip()
            elif line.startswith('- **Confidence Score**:'):
                try:
                    result['confidence_score'] = float(line.split(':')[1].strip())
                except:
                    pass
            elif line.startswith('- **Composite Score**:'):
                try:
                    result['composite_score'] = int(line.split(':')[1].strip())
                except:
                    pass
            elif line.startswith('- **Price Target**:'):
                result['price_target'] = line.split(':')[1].strip()
            elif line.startswith('- **Investment Thesis**:'):
                result['investment_thesis'] = line.split(':')[1].strip()
            elif line.startswith('- **Agent Consensus**:'):
                result['agent_consensus'] = line.split(':')[1].strip()
        
        return result
    
    def _parse_portfolio_ranking(self, ranking_text: str, stock_analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Parse portfolio ranking response"""
        
        # For now, return sorted by composite score
        # In a full implementation, would parse the LLM response
        return sorted(
            stock_analyses,
            key=lambda x: x.get('composite_score', 0),
            reverse=True
        )
    
    def _llm_consensus_building(self, agent_recommendations: Dict[str, str], agent_confidences: Dict[str, float]) -> Dict[str, Any]:
        """LLM-powered consensus building"""
        
        prompt = CONSENSUS_BUILDING_PROMPT.format(
            fundamental_rec=agent_recommendations.get('fundamental', 'HOLD'),
            fundamental_conf=agent_confidences.get('fundamental', 0.5),
            sentiment_rec=agent_recommendations.get('sentiment', 'HOLD'),
            sentiment_conf=agent_confidences.get('sentiment', 0.5),
            valuation_rec=agent_recommendations.get('valuation', 'HOLD'),
            valuation_conf=agent_confidences.get('valuation', 0.5),
            rationale_rec=agent_recommendations.get('rationale', 'HOLD'),
            rationale_conf=agent_confidences.get('rationale', 0.5),
            trend_rec=agent_recommendations.get('secular_trend', 'HOLD'),
            trend_conf=agent_confidences.get('secular_trend', 0.5),
            disagreement_details="Agent recommendations vary across the spectrum"
        )
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        return {
            'consensus_recommendation': 'HOLD',
            'confidence_level': 0.5,
            'reasoning': response.content
        }
    
    def _fallback_consensus_building(self, agent_recommendations: Dict[str, str], agent_confidences: Dict[str, float]) -> Dict[str, Any]:
        """Fallback consensus building"""
        
        # Simple weighted average approach
        weighted_scores = []
        total_weight = 0
        
        for agent, rec in agent_recommendations.items():
            confidence = agent_confidences.get(agent, 0.5)
            score = self._recommendation_to_score(rec)
            weighted_scores.append(score * confidence)
            total_weight += confidence
        
        if total_weight > 0:
            consensus_score = sum(weighted_scores) / total_weight
        else:
            consensus_score = 0.5
        
        return {
            'consensus_recommendation': self._score_to_recommendation(consensus_score),
            'confidence_level': total_weight / len(agent_recommendations) if agent_recommendations else 0.5,
            'reasoning': f"Consensus built from {len(agent_recommendations)} agents using confidence weighting"
        }

