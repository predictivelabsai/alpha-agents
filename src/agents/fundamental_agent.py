"""
Fundamental Agent for equity portfolio construction.
Analyzes 10-K/10-Q reports, financial statements, and fundamental metrics.
Based on the Alpha Agents paper.
"""

from typing import Dict, List, Any, Optional
import json
import math
from datetime import datetime
from langchain_core.messages import HumanMessage
from .base_agent import BaseAgent, Stock, AgentAnalysis, InvestmentDecision, RiskTolerance

class FundamentalAgent(BaseAgent):
    """
    Fundamental Agent specializes in analyzing financial fundamentals,
    earnings reports, balance sheets, and company financial health.
    """
    
    def __init__(self, risk_tolerance: RiskTolerance, llm_client=None):
        super().__init__(risk_tolerance, llm_client)
        self.agent_name = "fundamental"
        
    def analyze(self, stock: Stock, market_data: Optional[Dict] = None) -> AgentAnalysis:
        """
        Analyze stock fundamentals and provide investment recommendation.
        
        Args:
            stock: Stock information to analyze
            market_data: Optional market context data
            
        Returns:
            AgentAnalysis with fundamental-based recommendation
        """
        try:
            # Perform fundamental analysis
            fundamental_metrics = self._calculate_fundamental_metrics(stock)
            financial_health = self._assess_financial_health(stock, fundamental_metrics)
            valuation_analysis = self._analyze_valuation(stock, fundamental_metrics)
            
            # Generate LLM-enhanced analysis if available
            if self.llm_client:
                llm_analysis = self._get_llm_analysis(stock, fundamental_metrics, financial_health, valuation_analysis)
                recommendation, confidence, reasoning = self._parse_llm_response(llm_analysis)
            else:
                recommendation, confidence, reasoning = self._fallback_analysis(fundamental_metrics, financial_health, valuation_analysis)
            
            # Adjust for risk tolerance
            confidence = self.adjust_for_risk_tolerance(confidence)
            
            # Extract key factors and concerns
            key_factors = self._extract_key_factors(fundamental_metrics, financial_health)
            concerns = self._identify_concerns(fundamental_metrics, financial_health)
            
            return AgentAnalysis(
                agent_name=self.agent_name,
                stock_symbol=stock.symbol,
                recommendation=recommendation,
                confidence_score=confidence,
                target_price=self._calculate_target_price(stock, fundamental_metrics),
                risk_assessment=self._assess_risk_level(fundamental_metrics, financial_health),
                key_factors=key_factors,
                concerns=concerns,
                reasoning=reasoning
            )
            
        except Exception as e:
            self.logger.error(f"Error in fundamental analysis for {stock.symbol}: {e}")
            return AgentAnalysis(
                agent_name=self.agent_name,
                stock_symbol=stock.symbol,
                recommendation=InvestmentDecision.HOLD,
                confidence_score=0.3,
                risk_assessment="HIGH",
                key_factors=[],
                concerns=["Analysis error occurred"],
                reasoning=f"Unable to complete fundamental analysis: {str(e)}"
            )
    
    def _calculate_fundamental_metrics(self, stock: Stock) -> Dict[str, float]:
        """Calculate key fundamental metrics."""
        metrics = {}
        
        # P/E Ratio Analysis
        if stock.pe_ratio:
            metrics['pe_ratio'] = stock.pe_ratio
            metrics['pe_score'] = self._score_pe_ratio(stock.pe_ratio, stock.sector)
        else:
            metrics['pe_ratio'] = None
            metrics['pe_score'] = 0.5
        
        # Market Cap Analysis
        metrics['market_cap'] = stock.market_cap
        metrics['market_cap_score'] = self._score_market_cap(stock.market_cap)
        
        # Dividend Analysis
        if stock.dividend_yield:
            metrics['dividend_yield'] = stock.dividend_yield
            metrics['dividend_score'] = self._score_dividend_yield(stock.dividend_yield)
        else:
            metrics['dividend_yield'] = 0.0
            metrics['dividend_score'] = 0.3
        
        # Beta Analysis (Risk)
        if stock.beta:
            metrics['beta'] = stock.beta
            metrics['beta_score'] = self._score_beta(stock.beta)
        else:
            metrics['beta'] = 1.0
            metrics['beta_score'] = 0.5
        
        return metrics
    
    def _score_pe_ratio(self, pe_ratio: float, sector: str) -> float:
        """Score P/E ratio based on sector benchmarks."""
        # Sector-specific P/E benchmarks (simplified)
        sector_benchmarks = {
            'Technology': 25.0,
            'Healthcare': 20.0,
            'Financial': 12.0,
            'Consumer': 18.0,
            'Industrial': 16.0,
            'Energy': 14.0,
            'Utilities': 15.0,
            'Materials': 14.0,
            'Real Estate': 20.0,
            'Telecommunications': 16.0
        }
        
        benchmark = sector_benchmarks.get(sector, 18.0)
        
        if pe_ratio <= 0:
            return 0.1  # Negative earnings
        elif pe_ratio < benchmark * 0.7:
            return 0.9  # Very attractive
        elif pe_ratio < benchmark:
            return 0.7  # Attractive
        elif pe_ratio < benchmark * 1.3:
            return 0.5  # Fair
        elif pe_ratio < benchmark * 1.8:
            return 0.3  # Expensive
        else:
            return 0.1  # Very expensive
    
    def _score_market_cap(self, market_cap: float) -> float:
        """Score based on market capitalization."""
        if market_cap >= 200e9:  # Mega cap
            return 0.8
        elif market_cap >= 10e9:  # Large cap
            return 0.7
        elif market_cap >= 2e9:  # Mid cap
            return 0.6
        elif market_cap >= 300e6:  # Small cap
            return 0.5
        else:  # Micro cap
            return 0.3
    
    def _score_dividend_yield(self, dividend_yield: float) -> float:
        """Score dividend yield."""
        if dividend_yield >= 0.05:  # 5%+
            return 0.9
        elif dividend_yield >= 0.03:  # 3-5%
            return 0.7
        elif dividend_yield >= 0.01:  # 1-3%
            return 0.5
        else:  # <1%
            return 0.3
    
    def _score_beta(self, beta: float) -> float:
        """Score beta (volatility risk)."""
        if self.risk_tolerance == RiskTolerance.CONSERVATIVE:
            if beta <= 0.8:
                return 0.9
            elif beta <= 1.2:
                return 0.6
            else:
                return 0.3
        elif self.risk_tolerance == RiskTolerance.AGGRESSIVE:
            if beta >= 1.5:
                return 0.8
            elif beta >= 1.0:
                return 0.7
            else:
                return 0.5
        else:  # Moderate
            if 0.8 <= beta <= 1.3:
                return 0.8
            else:
                return 0.5
    
    def _assess_financial_health(self, stock: Stock, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Assess overall financial health."""
        health_score = 0.0
        factors = []
        
        # P/E Score contribution
        pe_weight = 0.3
        health_score += metrics['pe_score'] * pe_weight
        if metrics['pe_score'] > 0.7:
            factors.append("Attractive valuation")
        elif metrics['pe_score'] < 0.3:
            factors.append("High valuation concern")
        
        # Market Cap Score contribution
        mc_weight = 0.2
        health_score += metrics['market_cap_score'] * mc_weight
        if metrics['market_cap'] >= 10e9:
            factors.append("Large, stable company")
        
        # Dividend Score contribution
        div_weight = 0.2
        health_score += metrics['dividend_score'] * div_weight
        if metrics['dividend_yield'] >= 0.03:
            factors.append("Strong dividend yield")
        
        # Beta Score contribution
        beta_weight = 0.3
        health_score += metrics['beta_score'] * beta_weight
        
        return {
            'overall_score': health_score,
            'health_factors': factors,
            'financial_strength': 'Strong' if health_score > 0.7 else 'Moderate' if health_score > 0.5 else 'Weak'
        }
    
    def _analyze_valuation(self, stock: Stock, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Analyze stock valuation."""
        valuation = {
            'current_price': stock.current_price,
            'pe_based_fair_value': None,
            'valuation_rating': 'Fair'
        }
        
        if stock.pe_ratio and stock.pe_ratio > 0:
            # Simple P/E based fair value estimation
            sector_avg_pe = 18.0  # Market average
            estimated_eps = stock.current_price / stock.pe_ratio
            fair_value = estimated_eps * sector_avg_pe
            valuation['pe_based_fair_value'] = fair_value
            
            price_to_fair = stock.current_price / fair_value
            if price_to_fair < 0.8:
                valuation['valuation_rating'] = 'Undervalued'
            elif price_to_fair > 1.2:
                valuation['valuation_rating'] = 'Overvalued'
        
        return valuation
    
    def _get_llm_analysis(self, stock: Stock, metrics: Dict, health: Dict, valuation: Dict) -> str:
        """Get enhanced analysis from LLM."""
        analysis_prompt = f"""
        As a Fundamental Analysis expert, analyze the following stock:
        
        Company: {stock.company_name} ({stock.symbol})
        Sector: {stock.sector}
        Current Price: ${stock.current_price:.2f}
        Market Cap: {self.format_currency(stock.market_cap)}
        
        Financial Metrics:
        - P/E Ratio: {stock.pe_ratio or 'N/A'}
        - Dividend Yield: {stock.dividend_yield*100 if stock.dividend_yield else 0:.2f}%
        - Beta: {stock.beta or 'N/A'}
        
        Analysis Scores:
        - P/E Score: {metrics['pe_score']:.2f}
        - Market Cap Score: {metrics['market_cap_score']:.2f}
        - Dividend Score: {metrics['dividend_score']:.2f}
        - Beta Score: {metrics['beta_score']:.2f}
        - Overall Health Score: {health['overall_score']:.2f}
        
        Valuation Assessment: {valuation['valuation_rating']}
        
        Risk Tolerance: {self.risk_tolerance.value}
        
        Based on fundamental analysis, provide:
        1. Investment recommendation (BUY/SELL/HOLD/AVOID)
        2. Confidence level (0.0 to 1.0)
        3. Target price estimate
        4. Key fundamental strengths
        5. Main concerns or risks
        6. Detailed reasoning
        
        Focus on financial health, earnings quality, balance sheet strength, and long-term growth prospects.
        """
        
        if self.llm_client:
            response = self.llm_client.invoke([HumanMessage(content=analysis_prompt)])
            return response.content
        else:
            return self._fallback_analysis(metrics, health, valuation)[2]
    
    def _parse_llm_response(self, response: str) -> tuple:
        """Parse LLM response into recommendation, confidence, and reasoning."""
        try:
            # Simple parsing logic
            response_lower = response.lower()
            
            # Extract recommendation
            if 'buy' in response_lower and 'avoid' not in response_lower:
                recommendation = InvestmentDecision.BUY
            elif 'sell' in response_lower:
                recommendation = InvestmentDecision.SELL
            elif 'avoid' in response_lower:
                recommendation = InvestmentDecision.AVOID
            else:
                recommendation = InvestmentDecision.HOLD
            
            # Extract confidence (look for numbers between 0 and 1)
            import re
            confidence_matches = re.findall(r'confidence[:\s]*([0-9]*\.?[0-9]+)', response_lower)
            if confidence_matches:
                confidence = float(confidence_matches[0])
                confidence = max(0.0, min(1.0, confidence))
            else:
                confidence = 0.6
            
            return recommendation, confidence, response
            
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}")
            return InvestmentDecision.HOLD, 0.5, response
    
    def _fallback_analysis(self, metrics: Dict, health: Dict, valuation: Dict) -> tuple:
        """Fallback analysis when LLM is not available."""
        overall_score = health['overall_score']
        
        # Determine recommendation based on scores
        if overall_score >= 0.75:
            recommendation = InvestmentDecision.BUY
            confidence = 0.8
        elif overall_score >= 0.6:
            recommendation = InvestmentDecision.HOLD
            confidence = 0.65
        elif overall_score >= 0.4:
            recommendation = InvestmentDecision.HOLD
            confidence = 0.5
        else:
            recommendation = InvestmentDecision.AVOID
            confidence = 0.7
        
        reasoning = f"""
        Fundamental Analysis Summary:
        - Overall Health Score: {overall_score:.2f}
        - Financial Strength: {health['financial_strength']}
        - Valuation: {valuation['valuation_rating']}
        - P/E Score: {metrics['pe_score']:.2f}
        - Market Cap: {self.format_currency(metrics['market_cap'])}
        
        Recommendation based on fundamental metrics and financial health assessment.
        """
        
        return recommendation, confidence, reasoning
    
    def _calculate_target_price(self, stock: Stock, metrics: Dict) -> Optional[float]:
        """Calculate target price based on fundamental analysis."""
        if stock.pe_ratio and stock.pe_ratio > 0:
            # Simple target price based on sector average P/E
            current_eps = stock.current_price / stock.pe_ratio
            target_pe = 18.0  # Market average
            target_price = current_eps * target_pe
            return round(target_price, 2)
        return None
    
    def _assess_risk_level(self, metrics: Dict, health: Dict) -> str:
        """Assess risk level based on fundamental metrics."""
        risk_factors = 0
        
        if metrics['pe_score'] < 0.3:
            risk_factors += 1
        if metrics['beta'] and metrics['beta'] > 1.5:
            risk_factors += 1
        if health['overall_score'] < 0.5:
            risk_factors += 1
        if metrics['market_cap'] < 2e9:  # Small cap
            risk_factors += 1
        
        if risk_factors >= 3:
            return "HIGH"
        elif risk_factors >= 2:
            return "MODERATE"
        else:
            return "LOW"
    
    def _extract_key_factors(self, metrics: Dict, health: Dict) -> List[str]:
        """Extract key positive factors."""
        factors = []
        
        if metrics['pe_score'] > 0.7:
            factors.append("Attractive P/E valuation")
        if metrics['dividend_yield'] and metrics['dividend_yield'] > 0.03:
            factors.append(f"Strong dividend yield ({metrics['dividend_yield']*100:.1f}%)")
        if metrics['market_cap'] >= 10e9:
            factors.append("Large-cap stability")
        if health['overall_score'] > 0.7:
            factors.append("Strong fundamental health")
        
        return factors[:5]  # Limit to top 5
    
    def _identify_concerns(self, metrics: Dict, health: Dict) -> List[str]:
        """Identify key concerns."""
        concerns = []
        
        if metrics['pe_score'] < 0.3:
            concerns.append("High valuation multiples")
        if metrics['beta'] and metrics['beta'] > 1.8:
            concerns.append("High volatility risk")
        if metrics['market_cap'] < 1e9:
            concerns.append("Small-cap liquidity risk")
        if health['overall_score'] < 0.4:
            concerns.append("Weak fundamental metrics")
        if not metrics['dividend_yield'] or metrics['dividend_yield'] < 0.01:
            concerns.append("No meaningful dividend income")
        
        return concerns[:5]  # Limit to top 5

