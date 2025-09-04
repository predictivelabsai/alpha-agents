"""
Valuation Agent for equity portfolio construction.
Analyzes stock prices, volumes, technical indicators, and valuation metrics.
Based on the Alpha Agents paper.
"""

from typing import Dict, List, Any, Optional
import json
import math
from datetime import datetime, timedelta
from langchain_core.messages import HumanMessage
from .base_agent import BaseAgent, Stock, AgentAnalysis, InvestmentDecision, RiskTolerance

class ValuationAgent(BaseAgent):
    """
    Valuation Agent specializes in analyzing stock prices, trading volumes,
    technical indicators, and relative valuation metrics.
    """
    
    def __init__(self, risk_tolerance: RiskTolerance, llm_client=None):
        super().__init__(risk_tolerance, llm_client)
        self.agent_name = "valuation"
        
    def analyze(self, stock: Stock, market_data: Optional[Dict] = None) -> AgentAnalysis:
        """
        Analyze stock valuation and provide investment recommendation.
        
        Args:
            stock: Stock information to analyze
            market_data: Optional market context data including price history
            
        Returns:
            AgentAnalysis with valuation-based recommendation
        """
        try:
            # Perform valuation analysis
            valuation_metrics = self._calculate_valuation_metrics(stock, market_data)
            technical_analysis = self._analyze_technical_indicators(stock, market_data)
            relative_valuation = self._analyze_relative_valuation(stock, market_data)
            liquidity_analysis = self._analyze_liquidity(stock, market_data)
            
            # Generate LLM-enhanced analysis if available
            if self.llm_client:
                llm_analysis = self._get_llm_analysis(stock, valuation_metrics, technical_analysis, relative_valuation, liquidity_analysis)
                recommendation, confidence, reasoning = self._parse_llm_response(llm_analysis)
            else:
                recommendation, confidence, reasoning = self._fallback_analysis(valuation_metrics, technical_analysis, relative_valuation)
            
            # Adjust for risk tolerance
            confidence = self.adjust_for_risk_tolerance(confidence)
            
            # Extract key factors and concerns
            key_factors = self._extract_key_factors(valuation_metrics, technical_analysis, relative_valuation)
            concerns = self._identify_concerns(valuation_metrics, technical_analysis, relative_valuation)
            
            return AgentAnalysis(
                agent_name=self.agent_name,
                stock_symbol=stock.symbol,
                recommendation=recommendation,
                confidence_score=confidence,
                target_price=self._calculate_valuation_target_price(stock, valuation_metrics, relative_valuation),
                risk_assessment=self._assess_valuation_risk(valuation_metrics, technical_analysis),
                key_factors=key_factors,
                concerns=concerns,
                reasoning=reasoning
            )
            
        except Exception as e:
            self.logger.error(f"Error in valuation analysis for {stock.symbol}: {e}")
            return AgentAnalysis(
                agent_name=self.agent_name,
                stock_symbol=stock.symbol,
                recommendation=InvestmentDecision.HOLD,
                confidence_score=0.3,
                risk_assessment="HIGH",
                key_factors=[],
                concerns=["Analysis error occurred"],
                reasoning=f"Unable to complete valuation analysis: {str(e)}"
            )
    
    def _calculate_valuation_metrics(self, stock: Stock, market_data: Optional[Dict]) -> Dict[str, Any]:
        """Calculate valuation metrics."""
        metrics = {
            'current_price': stock.current_price,
            'market_cap': stock.market_cap,
            'pe_ratio': stock.pe_ratio,
            'price_momentum': 0.0,
            'valuation_score': 0.5,
            'price_trend': 'neutral'
        }
        
        # P/E Ratio Analysis
        if stock.pe_ratio:
            if stock.pe_ratio < 15:
                metrics['pe_valuation'] = 'undervalued'
                metrics['pe_score'] = 0.8
            elif stock.pe_ratio < 25:
                metrics['pe_valuation'] = 'fair'
                metrics['pe_score'] = 0.6
            elif stock.pe_ratio < 35:
                metrics['pe_valuation'] = 'expensive'
                metrics['pe_score'] = 0.4
            else:
                metrics['pe_valuation'] = 'overvalued'
                metrics['pe_score'] = 0.2
        else:
            metrics['pe_valuation'] = 'unknown'
            metrics['pe_score'] = 0.5
        
        # Market Cap Valuation
        if stock.market_cap >= 200e9:  # Mega cap
            metrics['size_premium'] = 0.1
            metrics['liquidity_score'] = 0.9
        elif stock.market_cap >= 10e9:  # Large cap
            metrics['size_premium'] = 0.05
            metrics['liquidity_score'] = 0.8
        elif stock.market_cap >= 2e9:  # Mid cap
            metrics['size_premium'] = 0.0
            metrics['liquidity_score'] = 0.6
        else:  # Small cap
            metrics['size_premium'] = -0.05
            metrics['liquidity_score'] = 0.4
        
        # Dividend Yield Impact on Valuation
        if stock.dividend_yield:
            if stock.dividend_yield >= 0.04:  # 4%+
                metrics['dividend_value'] = 0.8
            elif stock.dividend_yield >= 0.02:  # 2-4%
                metrics['dividend_value'] = 0.6
            else:  # <2%
                metrics['dividend_value'] = 0.4
        else:
            metrics['dividend_value'] = 0.3  # No dividend
        
        # Integrate market data if available
        if market_data:
            if 'price_history' in market_data:
                metrics['price_momentum'] = self._calculate_price_momentum(market_data['price_history'])
            if 'valuation_data' in market_data:
                val_data = market_data['valuation_data']
                metrics.update(val_data)
        
        # Overall valuation score
        metrics['valuation_score'] = (
            metrics['pe_score'] * 0.4 +
            metrics['liquidity_score'] * 0.2 +
            metrics['dividend_value'] * 0.2 +
            (0.5 + metrics['size_premium']) * 0.2
        )
        
        return metrics
    
    def _analyze_technical_indicators(self, stock: Stock, market_data: Optional[Dict]) -> Dict[str, Any]:
        """Analyze technical indicators."""
        technical = {
            'volume_trend': 'neutral',
            'volume_score': 0.5,
            'momentum_score': 0.5,
            'volatility_score': 0.5,
            'technical_rating': 'neutral'
        }
        
        # Volume Analysis
        if stock.volume:
            # Simplified volume analysis (would need historical data in real implementation)
            if stock.volume > 1000000:  # High volume
                technical['volume_trend'] = 'high'
                technical['volume_score'] = 0.7
            elif stock.volume > 100000:  # Moderate volume
                technical['volume_trend'] = 'moderate'
                technical['volume_score'] = 0.6
            else:  # Low volume
                technical['volume_trend'] = 'low'
                technical['volume_score'] = 0.4
        
        # Beta-based volatility analysis
        if stock.beta:
            if stock.beta < 0.8:
                technical['volatility_score'] = 0.8  # Low volatility
                technical['volatility_rating'] = 'low'
            elif stock.beta < 1.3:
                technical['volatility_score'] = 0.6  # Moderate volatility
                technical['volatility_rating'] = 'moderate'
            else:
                technical['volatility_score'] = 0.4  # High volatility
                technical['volatility_rating'] = 'high'
        else:
            technical['volatility_score'] = 0.5
            technical['volatility_rating'] = 'unknown'
        
        # Sector-based momentum assumptions
        growth_sectors = ['Technology', 'Healthcare', 'Consumer']
        if stock.sector in growth_sectors:
            technical['momentum_score'] = 0.6
            technical['sector_momentum'] = 'positive'
        else:
            technical['momentum_score'] = 0.5
            technical['sector_momentum'] = 'neutral'
        
        # Overall technical rating
        tech_score = (
            technical['volume_score'] * 0.3 +
            technical['momentum_score'] * 0.4 +
            technical['volatility_score'] * 0.3
        )
        
        if tech_score >= 0.7:
            technical['technical_rating'] = 'bullish'
        elif tech_score >= 0.6:
            technical['technical_rating'] = 'positive'
        elif tech_score >= 0.4:
            technical['technical_rating'] = 'neutral'
        else:
            technical['technical_rating'] = 'bearish'
        
        # Integrate market data if available
        if market_data and 'technical_data' in market_data:
            tech_data = market_data['technical_data']
            technical.update(tech_data)
        
        return technical
    
    def _analyze_relative_valuation(self, stock: Stock, market_data: Optional[Dict]) -> Dict[str, Any]:
        """Analyze relative valuation vs peers and market."""
        relative_val = {
            'vs_sector': 'neutral',
            'vs_market': 'neutral',
            'peer_comparison': 'average',
            'relative_score': 0.5
        }
        
        # Sector-based relative valuation (simplified)
        sector_avg_pe = {
            'Technology': 28.0,
            'Healthcare': 22.0,
            'Financial': 13.0,
            'Consumer': 20.0,
            'Industrial': 18.0,
            'Energy': 15.0,
            'Utilities': 17.0,
            'Materials': 16.0,
            'Real Estate': 25.0,
            'Telecommunications': 18.0
        }
        
        market_avg_pe = 20.0
        sector_pe = sector_avg_pe.get(stock.sector, market_avg_pe)
        
        if stock.pe_ratio:
            # Vs Sector
            if stock.pe_ratio < sector_pe * 0.8:
                relative_val['vs_sector'] = 'undervalued'
                relative_val['sector_score'] = 0.8
            elif stock.pe_ratio < sector_pe * 1.2:
                relative_val['vs_sector'] = 'fair'
                relative_val['sector_score'] = 0.6
            else:
                relative_val['vs_sector'] = 'overvalued'
                relative_val['sector_score'] = 0.3
            
            # Vs Market
            if stock.pe_ratio < market_avg_pe * 0.8:
                relative_val['vs_market'] = 'undervalued'
                relative_val['market_score'] = 0.8
            elif stock.pe_ratio < market_avg_pe * 1.2:
                relative_val['vs_market'] = 'fair'
                relative_val['market_score'] = 0.6
            else:
                relative_val['vs_market'] = 'overvalued'
                relative_val['market_score'] = 0.3
        else:
            relative_val['sector_score'] = 0.5
            relative_val['market_score'] = 0.5
        
        # Market cap premium/discount
        if stock.market_cap >= 50e9:  # Mega cap premium
            relative_val['size_factor'] = 0.7
        elif stock.market_cap >= 10e9:  # Large cap
            relative_val['size_factor'] = 0.6
        elif stock.market_cap >= 2e9:  # Mid cap
            relative_val['size_factor'] = 0.5
        else:  # Small cap discount
            relative_val['size_factor'] = 0.4
        
        # Overall relative score
        relative_val['relative_score'] = (
            relative_val['sector_score'] * 0.4 +
            relative_val['market_score'] * 0.3 +
            relative_val['size_factor'] * 0.3
        )
        
        # Integrate market data if available
        if market_data and 'peer_data' in market_data:
            peer_data = market_data['peer_data']
            relative_val.update(peer_data)
        
        return relative_val
    
    def _analyze_liquidity(self, stock: Stock, market_data: Optional[Dict]) -> Dict[str, Any]:
        """Analyze stock liquidity."""
        liquidity = {
            'liquidity_score': 0.5,
            'trading_ease': 'moderate',
            'market_impact': 'moderate'
        }
        
        # Volume-based liquidity
        if stock.volume:
            if stock.volume >= 5000000:  # Very high volume
                liquidity['liquidity_score'] = 0.9
                liquidity['trading_ease'] = 'excellent'
                liquidity['market_impact'] = 'low'
            elif stock.volume >= 1000000:  # High volume
                liquidity['liquidity_score'] = 0.8
                liquidity['trading_ease'] = 'good'
                liquidity['market_impact'] = 'low'
            elif stock.volume >= 100000:  # Moderate volume
                liquidity['liquidity_score'] = 0.6
                liquidity['trading_ease'] = 'moderate'
                liquidity['market_impact'] = 'moderate'
            else:  # Low volume
                liquidity['liquidity_score'] = 0.3
                liquidity['trading_ease'] = 'poor'
                liquidity['market_impact'] = 'high'
        
        # Market cap impact on liquidity
        if stock.market_cap >= 10e9:  # Large cap
            liquidity['liquidity_score'] = max(liquidity['liquidity_score'], 0.7)
        elif stock.market_cap < 1e9:  # Small cap
            liquidity['liquidity_score'] = min(liquidity['liquidity_score'], 0.5)
        
        return liquidity
    
    def _calculate_price_momentum(self, price_history: List[float]) -> float:
        """Calculate price momentum from price history."""
        if len(price_history) < 2:
            return 0.0
        
        # Simple momentum calculation
        recent_price = price_history[-1]
        older_price = price_history[0]
        
        momentum = (recent_price - older_price) / older_price
        return momentum
    
    def _get_llm_analysis(self, stock: Stock, valuation_metrics: Dict, technical_analysis: Dict, 
                         relative_valuation: Dict, liquidity_analysis: Dict) -> str:
        """Get enhanced valuation analysis from LLM."""
        analysis_prompt = f"""
        As a Valuation and Technical Analysis expert, analyze the following stock:
        
        Company: {stock.company_name} ({stock.symbol})
        Sector: {stock.sector}
        Current Price: ${stock.current_price:.2f}
        Market Cap: {self.format_currency(stock.market_cap)}
        
        Valuation Metrics:
        - P/E Ratio: {stock.pe_ratio or 'N/A'}
        - P/E Valuation: {valuation_metrics.get('pe_valuation', 'unknown')}
        - Valuation Score: {valuation_metrics['valuation_score']:.2f}
        - Dividend Yield: {stock.dividend_yield*100 if stock.dividend_yield else 0:.2f}%
        
        Technical Analysis:
        - Volume: {stock.volume or 'N/A'}
        - Volume Trend: {technical_analysis['volume_trend']}
        - Beta: {stock.beta or 'N/A'}
        - Volatility Rating: {technical_analysis.get('volatility_rating', 'unknown')}
        - Technical Rating: {technical_analysis['technical_rating']}
        
        Relative Valuation:
        - Vs Sector: {relative_valuation['vs_sector']}
        - Vs Market: {relative_valuation['vs_market']}
        - Relative Score: {relative_valuation['relative_score']:.2f}
        
        Liquidity Analysis:
        - Liquidity Score: {liquidity_analysis['liquidity_score']:.2f}
        - Trading Ease: {liquidity_analysis['trading_ease']}
        - Market Impact: {liquidity_analysis['market_impact']}
        
        Risk Tolerance: {self.risk_tolerance.value}
        
        Based on valuation and technical analysis, provide:
        1. Investment recommendation (BUY/SELL/HOLD/AVOID)
        2. Confidence level (0.0 to 1.0)
        3. Target price estimate
        4. Key valuation strengths
        5. Technical concerns or risks
        6. Detailed reasoning
        
        Focus on price action, valuation multiples, technical momentum, and relative attractiveness.
        """
        
        if self.llm_client:
            response = self.llm_client.invoke([HumanMessage(content=analysis_prompt)])
            return response.content
        else:
            return self._fallback_analysis(valuation_metrics, technical_analysis, relative_valuation)[2]
    
    def _parse_llm_response(self, response: str) -> tuple:
        """Parse LLM response into recommendation, confidence, and reasoning."""
        try:
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
            
            # Extract confidence
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
    
    def _fallback_analysis(self, valuation_metrics: Dict, technical_analysis: Dict, relative_valuation: Dict) -> tuple:
        """Fallback analysis when LLM is not available."""
        # Weighted valuation score
        overall_score = (
            valuation_metrics['valuation_score'] * 0.4 +
            (0.5 if technical_analysis['technical_rating'] == 'neutral' else 
             0.7 if technical_analysis['technical_rating'] in ['bullish', 'positive'] else 0.3) * 0.3 +
            relative_valuation['relative_score'] * 0.3
        )
        
        # Determine recommendation based on overall score
        if overall_score >= 0.75:
            recommendation = InvestmentDecision.BUY
            confidence = 0.8
        elif overall_score >= 0.65:
            recommendation = InvestmentDecision.BUY
            confidence = 0.7
        elif overall_score >= 0.45:
            recommendation = InvestmentDecision.HOLD
            confidence = 0.6
        elif overall_score >= 0.35:
            recommendation = InvestmentDecision.HOLD
            confidence = 0.5
        else:
            recommendation = InvestmentDecision.AVOID
            confidence = 0.7
        
        reasoning = f"""
        Valuation Analysis Summary:
        - Overall Valuation Score: {overall_score:.2f}
        - P/E Valuation: {valuation_metrics.get('pe_valuation', 'unknown')}
        - Technical Rating: {technical_analysis['technical_rating']}
        - Relative Valuation Score: {relative_valuation['relative_score']:.2f}
        - Vs Sector: {relative_valuation['vs_sector']}
        - Vs Market: {relative_valuation['vs_market']}
        
        Recommendation based on valuation metrics, technical indicators, and relative analysis.
        """
        
        return recommendation, confidence, reasoning
    
    def _calculate_valuation_target_price(self, stock: Stock, valuation_metrics: Dict, relative_valuation: Dict) -> Optional[float]:
        """Calculate target price based on valuation analysis."""
        if stock.pe_ratio and stock.pe_ratio > 0:
            # Target based on fair value P/E
            current_eps = stock.current_price / stock.pe_ratio
            
            # Sector-based fair P/E
            sector_fair_pe = {
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
            
            fair_pe = sector_fair_pe.get(stock.sector, 18.0)
            target_price = current_eps * fair_pe
            
            # Adjust based on relative valuation
            if relative_valuation['relative_score'] > 0.7:
                target_price *= 1.1  # Premium for strong relative position
            elif relative_valuation['relative_score'] < 0.4:
                target_price *= 0.9  # Discount for weak relative position
            
            return round(target_price, 2)
        
        return None
    
    def _assess_valuation_risk(self, valuation_metrics: Dict, technical_analysis: Dict) -> str:
        """Assess risk level based on valuation metrics."""
        risk_factors = 0
        
        if valuation_metrics.get('pe_score', 0.5) < 0.3:
            risk_factors += 1  # High valuation
        if technical_analysis.get('volatility_rating') == 'high':
            risk_factors += 1  # High volatility
        if technical_analysis.get('volume_score', 0.5) < 0.4:
            risk_factors += 1  # Low liquidity
        if valuation_metrics.get('liquidity_score', 0.5) < 0.4:
            risk_factors += 1  # Poor liquidity
        
        if risk_factors >= 3:
            return "HIGH"
        elif risk_factors >= 2:
            return "MODERATE"
        else:
            return "LOW"
    
    def _extract_key_factors(self, valuation_metrics: Dict, technical_analysis: Dict, relative_valuation: Dict) -> List[str]:
        """Extract key positive valuation factors."""
        factors = []
        
        if valuation_metrics.get('pe_valuation') == 'undervalued':
            factors.append("Attractive P/E valuation")
        if relative_valuation['vs_sector'] == 'undervalued':
            factors.append("Undervalued vs sector peers")
        if relative_valuation['vs_market'] == 'undervalued':
            factors.append("Undervalued vs market")
        if technical_analysis['technical_rating'] in ['bullish', 'positive']:
            factors.append(f"Positive technical outlook ({technical_analysis['technical_rating']})")
        if valuation_metrics.get('dividend_value', 0) > 0.6:
            factors.append("Attractive dividend yield")
        if valuation_metrics.get('liquidity_score', 0.5) > 0.7:
            factors.append("Good liquidity")
        
        return factors[:5]  # Limit to top 5
    
    def _identify_concerns(self, valuation_metrics: Dict, technical_analysis: Dict, relative_valuation: Dict) -> List[str]:
        """Identify key valuation concerns."""
        concerns = []
        
        if valuation_metrics.get('pe_valuation') == 'overvalued':
            concerns.append("High P/E valuation")
        if relative_valuation['vs_sector'] == 'overvalued':
            concerns.append("Expensive vs sector peers")
        if relative_valuation['vs_market'] == 'overvalued':
            concerns.append("Expensive vs market")
        if technical_analysis['technical_rating'] == 'bearish':
            concerns.append("Negative technical signals")
        if technical_analysis.get('volatility_rating') == 'high':
            concerns.append("High price volatility")
        if valuation_metrics.get('liquidity_score', 0.5) < 0.4:
            concerns.append("Poor liquidity")
        
        return concerns[:5]  # Limit to top 5

