"""
Secular Trend Agent for equity portfolio construction.
Analyzes secular technology trends and identifies companies positioned to benefit 
from major technological waves (2025-2030).
"""

from typing import Dict, List, Any, Optional
import json
import math
from datetime import datetime
from langchain_core.messages import HumanMessage
from .base_agent import BaseAgent, Stock, AgentAnalysis, InvestmentDecision, RiskTolerance

class SecularTrendAgent(BaseAgent):
    """
    Secular Trend Agent specializes in analyzing secular technology trends and identifying
    companies positioned to benefit from major technological waves over the next 5 years.
    """
    
    def __init__(self, risk_tolerance: RiskTolerance, llm_client=None):
        super().__init__(risk_tolerance, llm_client)
        self.agent_name = "secular_trend"
        
        # Define the five secular technology waves (2025-2030)
        self.secular_trends = {
            'agentic_ai': {
                'name': 'Agentic AI & Autonomous Enterprise Software',
                'description': 'AI systems that can plan, act, and self-improve without constant human oversight',
                'market_size': 12e12,  # $12T global services spend
                'growth_rate': 0.40,   # 40% labor hour reduction potential
                'timeline': '2025-2030',
                'key_characteristics': [
                    'Proprietary data loops',
                    'Seat expansion inside Fortune 500 IT budgets',
                    'Cross-sell into systems of engagement'
                ],
                'winners': ['Microsoft', 'ServiceNow', 'CrowdStrike', 'Zscaler'],
                'margin_profile': 0.80  # 80%+ margins
            },
            'cloud_edge': {
                'name': 'Cloud Re-Acceleration & Sovereign/Edge Infrastructure',
                'description': 'Second-leg growth in public cloud plus sovereign clouds and edge nodes',
                'market_size': 110e9,  # $110B run-rate
                'growth_rate': 0.19,   # 19% YoY growth
                'timeline': '2025-2028',
                'key_characteristics': [
                    'Direct fiber access + power purchase agreements',
                    'GPU-as-a-Service contracts ≥3 years',
                    'Sovereign certification'
                ],
                'winners': ['Amazon', 'Microsoft', 'Nvidia'],
                'margin_profile': 0.65  # 60-70% margins
            },
            'ai_semiconductors': {
                'name': 'AI-Native Semiconductors & Advanced Packaging',
                'description': 'Custom silicon and advanced packaging for AI workloads',
                'market_size': 500e9,  # Estimated AI chip market
                'growth_rate': 0.50,   # 10x compute demand every 18 months
                'timeline': '2025-2030',
                'key_characteristics': [
                    'Chiplet architecture enabling rapid re-spins',
                    'Software SDK that third parties use',
                    'Long-term capacity agreements with foundry'
                ],
                'winners': ['Nvidia', 'Broadcom', 'TSMC'],
                'margin_profile': 0.75  # 65-90% margins
            },
            'cybersecurity_agentic': {
                'name': 'Cybersecurity for the Agentic Era',
                'description': 'AI-powered defense and zero-trust architectures for hybrid work and autonomous agents',
                'market_size': 200e9,  # Estimated cybersecurity market
                'growth_rate': 0.25,   # 93% of firms increasing cyber budgets
                'timeline': '2025-2030',
                'key_characteristics': [
                    'Single-platform architecture',
                    'AI trained on global telemetry >1 EB/day',
                    'FedRAMP / IL5 certification'
                ],
                'winners': ['CrowdStrike', 'Palo Alto Networks', 'Zscaler'],
                'margin_profile': 0.82  # 80-85% margins
            },
            'electrification_ai': {
                'name': 'Electrification & AI-Defined Vehicles',
                'description': 'EVs, charging networks, and software-defined vehicles with AI',
                'market_size': 800e9,  # Estimated EV market
                'growth_rate': 0.30,   # EU bans ICE by 2035
                'timeline': '2025-2035',
                'key_characteristics': [
                    'Over-the-air update capability',
                    'Proprietary charging connector or network exclusivity',
                    'Fleet-level data contracts'
                ],
                'winners': ['Tesla', 'NextEra Energy', 'CATL', 'BYD'],
                'margin_profile': 0.35  # 25-40% margins (rising)
            }
        }
        
        # Company trend mapping
        self.company_trend_mapping = {
            # Agentic AI
            'MSFT': ['agentic_ai', 'cloud_edge'],
            'NOW': ['agentic_ai'],
            'CRWD': ['agentic_ai', 'cybersecurity_agentic'],
            'ZS': ['agentic_ai', 'cybersecurity_agentic'],
            
            # Cloud & Edge
            'AMZN': ['cloud_edge'],
            'GOOGL': ['cloud_edge', 'agentic_ai'],
            'NVDA': ['ai_semiconductors', 'cloud_edge'],
            
            # AI Semiconductors
            'AVGO': ['ai_semiconductors'],
            'TSM': ['ai_semiconductors'],
            'AMD': ['ai_semiconductors'],
            'INTC': ['ai_semiconductors'],
            
            # Cybersecurity
            'PANW': ['cybersecurity_agentic'],
            'FTNT': ['cybersecurity_agentic'],
            'S': ['cybersecurity_agentic'],
            
            # Electrification
            'TSLA': ['electrification_ai'],
            'NEE': ['electrification_ai'],
            'BYD': ['electrification_ai'],
            'CATL': ['electrification_ai'],
            'F': ['electrification_ai'],
            'GM': ['electrification_ai']
        }
    
    def analyze(self, stock: Stock, market_data: Optional[Dict] = None) -> AgentAnalysis:
        """
        Analyze stock based on secular technology trends positioning.
        
        Args:
            stock: Stock information to analyze
            market_data: Optional market context data
            
        Returns:
            AgentAnalysis with secular trend-based recommendation
        """
        try:
            # Identify relevant secular trends for this stock
            relevant_trends = self._identify_relevant_trends(stock)
            trend_positioning = self._analyze_trend_positioning(stock, relevant_trends)
            market_opportunity = self._assess_market_opportunity(stock, relevant_trends)
            competitive_position = self._analyze_competitive_position(stock, relevant_trends)
            execution_capability = self._assess_execution_capability(stock, relevant_trends)
            
            # Generate LLM-enhanced analysis if available
            if self.llm_client:
                llm_analysis = self._get_llm_analysis(stock, relevant_trends, trend_positioning, 
                                                    market_opportunity, competitive_position, execution_capability)
                recommendation, confidence, reasoning = self._parse_llm_response(llm_analysis)
            else:
                recommendation, confidence, reasoning = self._fallback_analysis(
                    stock, relevant_trends, trend_positioning, market_opportunity, competitive_position, execution_capability)
            
            # Adjust for risk tolerance
            confidence = self.adjust_for_risk_tolerance(confidence)
            
            # Extract key factors and concerns
            key_factors = self._extract_key_factors(relevant_trends, trend_positioning, competitive_position)
            concerns = self._identify_concerns(stock, relevant_trends, execution_capability)
            
            return AgentAnalysis(
                agent_name=self.agent_name,
                stock_symbol=stock.symbol,
                recommendation=recommendation,
                confidence_score=confidence,
                target_price=self._calculate_trend_target_price(stock, relevant_trends, trend_positioning),
                risk_assessment=self._assess_trend_risk(stock, relevant_trends, execution_capability),
                key_factors=key_factors,
                concerns=concerns,
                reasoning=reasoning
            )
            
        except Exception as e:
            self.logger.error(f"Error in secular trend analysis for {stock.symbol}: {e}")
            return AgentAnalysis(
                agent_name=self.agent_name,
                stock_symbol=stock.symbol,
                recommendation=InvestmentDecision.HOLD,
                confidence_score=0.3,
                risk_assessment="HIGH",
                key_factors=[],
                concerns=["Analysis error occurred"],
                reasoning=f"Unable to complete secular trend analysis: {str(e)}"
            )
    
    def _identify_relevant_trends(self, stock: Stock) -> List[str]:
        """Identify which secular trends are relevant for this stock."""
        relevant_trends = []
        
        # Direct mapping from company symbol
        if stock.symbol in self.company_trend_mapping:
            relevant_trends.extend(self.company_trend_mapping[stock.symbol])
        
        # Sector-based trend mapping
        sector_trends = {
            'Technology': ['agentic_ai', 'cloud_edge', 'ai_semiconductors', 'cybersecurity_agentic'],
            'Communication Services': ['agentic_ai', 'cloud_edge'],
            'Consumer Discretionary': ['electrification_ai'],
            'Industrials': ['electrification_ai', 'ai_semiconductors'],
            'Energy': ['electrification_ai'],
            'Utilities': ['electrification_ai'],
            'Healthcare': ['agentic_ai'],
            'Financial Services': ['agentic_ai', 'cybersecurity_agentic']
        }
        
        if stock.sector in sector_trends:
            for trend in sector_trends[stock.sector]:
                if trend not in relevant_trends:
                    relevant_trends.append(trend)
        
        # Company name-based inference
        company_lower = stock.company_name.lower()
        if any(keyword in company_lower for keyword in ['cloud', 'software', 'data', 'ai', 'artificial']):
            if 'agentic_ai' not in relevant_trends:
                relevant_trends.append('agentic_ai')
        
        if any(keyword in company_lower for keyword in ['semiconductor', 'chip', 'processor', 'gpu']):
            if 'ai_semiconductors' not in relevant_trends:
                relevant_trends.append('ai_semiconductors')
        
        if any(keyword in company_lower for keyword in ['security', 'cyber', 'firewall', 'endpoint']):
            if 'cybersecurity_agentic' not in relevant_trends:
                relevant_trends.append('cybersecurity_agentic')
        
        if any(keyword in company_lower for keyword in ['electric', 'battery', 'charging', 'renewable', 'solar']):
            if 'electrification_ai' not in relevant_trends:
                relevant_trends.append('electrification_ai')
        
        return relevant_trends[:3]  # Limit to top 3 most relevant trends
    
    def _analyze_trend_positioning(self, stock: Stock, relevant_trends: List[str]) -> Dict[str, Any]:
        """Analyze how well positioned the company is for secular trends."""
        positioning = {
            'overall_positioning_score': 0.5,
            'trend_alignment': {},
            'market_timing': 'moderate',
            'competitive_advantage': 0.5
        }
        
        if not relevant_trends:
            return positioning
        
        total_score = 0
        trend_count = len(relevant_trends)
        
        for trend_key in relevant_trends:
            if trend_key not in self.secular_trends:
                continue
                
            trend = self.secular_trends[trend_key]
            trend_score = 0.5  # Base score
            
            # Market cap influence on trend positioning
            if stock.market_cap >= 100e9:  # Mega cap - established players
                if trend_key in ['agentic_ai', 'cloud_edge', 'ai_semiconductors']:
                    trend_score = 0.8  # Strong position in established trends
                else:
                    trend_score = 0.6  # Good position but may be slower to adapt
            elif stock.market_cap >= 10e9:  # Large cap
                trend_score = 0.7  # Good positioning across trends
            elif stock.market_cap >= 2e9:  # Mid cap
                trend_score = 0.6  # Moderate positioning, more agile
            else:  # Small cap
                trend_score = 0.4  # Higher risk but potential for disruption
            
            # Sector-specific adjustments
            if stock.sector == 'Technology':
                trend_score += 0.1  # Tech companies better positioned for tech trends
            elif stock.sector in ['Communication Services', 'Industrials']:
                trend_score += 0.05  # Some advantage
            
            # P/E ratio as market expectation indicator
            if stock.pe_ratio:
                if stock.pe_ratio > 30:  # High expectations
                    trend_score += 0.1  # Market expects strong trend participation
                elif stock.pe_ratio < 15:  # Low expectations
                    trend_score -= 0.05  # Market may not see strong trend positioning
            
            # Company-specific positioning (if directly mapped)
            if stock.symbol in self.company_trend_mapping:
                if trend_key in self.company_trend_mapping[stock.symbol]:
                    trend_score += 0.15  # Direct trend participant
            
            trend_score = max(0.1, min(0.9, trend_score))
            positioning['trend_alignment'][trend_key] = {
                'score': trend_score,
                'trend_name': trend['name'],
                'market_size': trend['market_size'],
                'growth_rate': trend['growth_rate']
            }
            total_score += trend_score
        
        positioning['overall_positioning_score'] = total_score / trend_count if trend_count > 0 else 0.5
        
        # Determine market timing
        if positioning['overall_positioning_score'] > 0.7:
            positioning['market_timing'] = 'early_leader'
        elif positioning['overall_positioning_score'] > 0.6:
            positioning['market_timing'] = 'well_positioned'
        elif positioning['overall_positioning_score'] > 0.4:
            positioning['market_timing'] = 'moderate'
        else:
            positioning['market_timing'] = 'lagging'
        
        return positioning
    
    def _assess_market_opportunity(self, stock: Stock, relevant_trends: List[str]) -> Dict[str, Any]:
        """Assess the market opportunity size and growth potential."""
        opportunity = {
            'total_addressable_market': 0,
            'weighted_growth_rate': 0.0,
            'opportunity_score': 0.5,
            'timeline_alignment': 'medium_term'
        }
        
        if not relevant_trends:
            return opportunity
        
        total_market = 0
        weighted_growth = 0
        
        for trend_key in relevant_trends:
            if trend_key not in self.secular_trends:
                continue
                
            trend = self.secular_trends[trend_key]
            total_market += trend['market_size']
            weighted_growth += trend['growth_rate'] * trend['market_size']
        
        opportunity['total_addressable_market'] = total_market
        opportunity['weighted_growth_rate'] = weighted_growth / total_market if total_market > 0 else 0.0
        
        # Score based on market size and growth
        if total_market >= 1e12:  # $1T+ market
            market_score = 0.9
        elif total_market >= 500e9:  # $500B+ market
            market_score = 0.8
        elif total_market >= 100e9:  # $100B+ market
            market_score = 0.7
        elif total_market >= 50e9:   # $50B+ market
            market_score = 0.6
        else:
            market_score = 0.4
        
        # Growth rate scoring
        if opportunity['weighted_growth_rate'] >= 0.4:  # 40%+ growth
            growth_score = 0.9
        elif opportunity['weighted_growth_rate'] >= 0.25:  # 25%+ growth
            growth_score = 0.8
        elif opportunity['weighted_growth_rate'] >= 0.15:  # 15%+ growth
            growth_score = 0.7
        elif opportunity['weighted_growth_rate'] >= 0.10:  # 10%+ growth
            growth_score = 0.6
        else:
            growth_score = 0.4
        
        opportunity['opportunity_score'] = (market_score * 0.6 + growth_score * 0.4)
        
        # Timeline alignment
        if opportunity['weighted_growth_rate'] >= 0.3:
            opportunity['timeline_alignment'] = 'near_term'  # 2025-2027
        elif opportunity['weighted_growth_rate'] >= 0.15:
            opportunity['timeline_alignment'] = 'medium_term'  # 2025-2030
        else:
            opportunity['timeline_alignment'] = 'long_term'  # 2025-2035
        
        return opportunity
    
    def _analyze_competitive_position(self, stock: Stock, relevant_trends: List[str]) -> Dict[str, Any]:
        """Analyze competitive position within secular trends."""
        competitive = {
            'market_leadership': 0.5,
            'differentiation': 0.5,
            'execution_track_record': 0.5,
            'competitive_moats': [],
            'overall_competitive_score': 0.5
        }
        
        # Market cap as proxy for market position
        if stock.market_cap >= 500e9:  # Mega cap
            competitive['market_leadership'] = 0.9
            competitive['execution_track_record'] = 0.8
        elif stock.market_cap >= 100e9:  # Large cap
            competitive['market_leadership'] = 0.8
            competitive['execution_track_record'] = 0.7
        elif stock.market_cap >= 10e9:  # Mid-large cap
            competitive['market_leadership'] = 0.6
            competitive['execution_track_record'] = 0.6
        else:  # Smaller companies
            competitive['market_leadership'] = 0.4
            competitive['execution_track_record'] = 0.5
        
        # Sector-based competitive advantages
        sector_advantages = {
            'Technology': ['Platform effects', 'Data network effects', 'Developer ecosystem'],
            'Communication Services': ['Scale advantages', 'Network effects'],
            'Industrials': ['Manufacturing scale', 'Distribution networks'],
            'Energy': ['Resource access', 'Infrastructure scale'],
            'Healthcare': ['Regulatory moats', 'Clinical data']
        }
        
        if stock.sector in sector_advantages:
            competitive['competitive_moats'] = sector_advantages[stock.sector]
            competitive['differentiation'] = 0.7
        
        # Company-specific competitive position
        known_leaders = {
            'MSFT': 0.9, 'AAPL': 0.9, 'GOOGL': 0.9, 'AMZN': 0.9, 'NVDA': 0.95,
            'TSLA': 0.8, 'META': 0.8, 'NFLX': 0.7, 'CRM': 0.7, 'NOW': 0.8,
            'CRWD': 0.8, 'ZS': 0.7, 'PANW': 0.8, 'AVGO': 0.8, 'TSM': 0.9
        }
        
        if stock.symbol in known_leaders:
            competitive['market_leadership'] = known_leaders[stock.symbol]
            competitive['execution_track_record'] = known_leaders[stock.symbol]
        
        # P/E ratio as market confidence indicator
        if stock.pe_ratio:
            if stock.pe_ratio > 40:  # Very high P/E suggests strong competitive position
                competitive['differentiation'] += 0.1
            elif stock.pe_ratio < 12:  # Low P/E might suggest competitive challenges
                competitive['differentiation'] -= 0.1
        
        competitive['overall_competitive_score'] = (
            competitive['market_leadership'] * 0.4 +
            competitive['differentiation'] * 0.3 +
            competitive['execution_track_record'] * 0.3
        )
        
        return competitive
    
    def _assess_execution_capability(self, stock: Stock, relevant_trends: List[str]) -> Dict[str, Any]:
        """Assess company's capability to execute on secular trends."""
        execution = {
            'financial_resources': 0.5,
            'innovation_capability': 0.5,
            'market_access': 0.5,
            'talent_acquisition': 0.5,
            'execution_risk': 'moderate',
            'overall_execution_score': 0.5
        }
        
        # Financial resources (market cap as proxy)
        if stock.market_cap >= 100e9:
            execution['financial_resources'] = 0.9
        elif stock.market_cap >= 10e9:
            execution['financial_resources'] = 0.7
        elif stock.market_cap >= 2e9:
            execution['financial_resources'] = 0.6
        else:
            execution['financial_resources'] = 0.4
        
        # Sector-based innovation capability
        innovation_sectors = {
            'Technology': 0.8,
            'Communication Services': 0.7,
            'Healthcare': 0.7,
            'Industrials': 0.6,
            'Consumer Discretionary': 0.6,
            'Energy': 0.5,
            'Utilities': 0.4,
            'Financial Services': 0.6
        }
        
        execution['innovation_capability'] = innovation_sectors.get(stock.sector, 0.5)
        
        # Market access (established companies have better access)
        if stock.market_cap >= 50e9:
            execution['market_access'] = 0.8
            execution['talent_acquisition'] = 0.8
        elif stock.market_cap >= 5e9:
            execution['market_access'] = 0.7
            execution['talent_acquisition'] = 0.7
        else:
            execution['market_access'] = 0.5
            execution['talent_acquisition'] = 0.5
        
        # Overall execution score
        execution['overall_execution_score'] = (
            execution['financial_resources'] * 0.3 +
            execution['innovation_capability'] * 0.3 +
            execution['market_access'] * 0.2 +
            execution['talent_acquisition'] * 0.2
        )
        
        # Execution risk assessment
        if execution['overall_execution_score'] >= 0.8:
            execution['execution_risk'] = 'low'
        elif execution['overall_execution_score'] >= 0.6:
            execution['execution_risk'] = 'moderate'
        else:
            execution['execution_risk'] = 'high'
        
        return execution
    
    def _get_llm_analysis(self, stock: Stock, relevant_trends: List[str], trend_positioning: Dict,
                         market_opportunity: Dict, competitive_position: Dict, execution_capability: Dict) -> str:
        """Get enhanced secular trend analysis from LLM."""
        
        trends_detail = ""
        for trend_key in relevant_trends:
            if trend_key in self.secular_trends:
                trend = self.secular_trends[trend_key]
                trends_detail += f"""
                {trend['name']}:
                - Market Size: {self.format_currency(trend['market_size'])}
                - Growth Rate: {trend['growth_rate']*100:.1f}%
                - Timeline: {trend['timeline']}
                - Margin Profile: {trend['margin_profile']*100:.0f}%
                - Key Characteristics: {', '.join(trend['key_characteristics'])}
                """
        
        analysis_prompt = f"""
        As a Secular Technology Trends expert analyzing investment opportunities for 2025-2030, evaluate:
        
        Company: {stock.company_name} ({stock.symbol})
        Sector: {stock.sector}
        Current Price: ${stock.current_price:.2f}
        Market Cap: {self.format_currency(stock.market_cap)}
        P/E Ratio: {stock.pe_ratio if stock.pe_ratio else 'N/A'}
        
        Relevant Secular Technology Trends:{trends_detail}
        
        Trend Positioning Analysis:
        - Overall Positioning Score: {trend_positioning['overall_positioning_score']:.2f}
        - Market Timing: {trend_positioning['market_timing']}
        - Trend Alignment: {len(trend_positioning['trend_alignment'])} relevant trends
        
        Market Opportunity:
        - Total Addressable Market: {self.format_currency(market_opportunity['total_addressable_market'])}
        - Weighted Growth Rate: {market_opportunity['weighted_growth_rate']*100:.1f}%
        - Opportunity Score: {market_opportunity['opportunity_score']:.2f}
        - Timeline Alignment: {market_opportunity['timeline_alignment']}
        
        Competitive Position:
        - Market Leadership: {competitive_position['market_leadership']:.2f}
        - Differentiation: {competitive_position['differentiation']:.2f}
        - Execution Track Record: {competitive_position['execution_track_record']:.2f}
        - Overall Competitive Score: {competitive_position['overall_competitive_score']:.2f}
        
        Execution Capability:
        - Financial Resources: {execution_capability['financial_resources']:.2f}
        - Innovation Capability: {execution_capability['innovation_capability']:.2f}
        - Market Access: {execution_capability['market_access']:.2f}
        - Execution Risk: {execution_capability['execution_risk']}
        - Overall Execution Score: {execution_capability['overall_execution_score']:.2f}
        
        Risk Tolerance: {self.risk_tolerance.value}
        
        Based on secular trend analysis, provide:
        1. Investment recommendation (BUY/SELL/HOLD/AVOID)
        2. Confidence level (0.0 to 1.0)
        3. Secular trend positioning assessment
        4. Key trend advantages
        5. Execution risks and challenges
        6. Detailed reasoning focusing on 2025-2030 secular opportunities
        
        Focus on long-term secular trends, market positioning, and execution capability.
        """
        
        if self.llm_client:
            response = self.llm_client.invoke([HumanMessage(content=analysis_prompt)])
            return response.content
        else:
            return self._fallback_analysis(stock, relevant_trends, trend_positioning, 
                                         market_opportunity, competitive_position, execution_capability)[2]
    
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
    
    def _fallback_analysis(self, stock: Stock, relevant_trends: List[str], trend_positioning: Dict,
                          market_opportunity: Dict, competitive_position: Dict, execution_capability: Dict) -> tuple:
        """Fallback analysis when LLM is not available."""
        
        # Weighted secular trend score
        overall_score = (
            trend_positioning['overall_positioning_score'] * 0.30 +
            market_opportunity['opportunity_score'] * 0.25 +
            competitive_position['overall_competitive_score'] * 0.25 +
            execution_capability['overall_execution_score'] * 0.20
        )
        
        # Determine recommendation based on secular trend positioning
        if overall_score >= 0.8:
            recommendation = InvestmentDecision.BUY
            confidence = 0.85
        elif overall_score >= 0.7:
            recommendation = InvestmentDecision.BUY
            confidence = 0.75
        elif overall_score >= 0.6:
            recommendation = InvestmentDecision.HOLD
            confidence = 0.65
        elif overall_score >= 0.4:
            recommendation = InvestmentDecision.HOLD
            confidence = 0.55
        else:
            recommendation = InvestmentDecision.AVOID
            confidence = 0.70
        
        trend_names = [self.secular_trends[t]['name'] for t in relevant_trends if t in self.secular_trends]
        
        reasoning = f"""
        Secular Technology Trends Analysis (2025-2030):
        - Overall Trend Score: {overall_score:.2f}
        - Relevant Trends: {', '.join(trend_names)}
        - Market Timing: {trend_positioning['market_timing']}
        - Total Addressable Market: {self.format_currency(market_opportunity['total_addressable_market'])}
        - Expected Growth Rate: {market_opportunity['weighted_growth_rate']*100:.1f}%
        - Competitive Position: {competitive_position['overall_competitive_score']:.2f}
        - Execution Capability: {execution_capability['overall_execution_score']:.2f}
        
        Recommendation based on positioning within major secular technology trends.
        """
        
        return recommendation, confidence, reasoning
    
    def _calculate_trend_target_price(self, stock: Stock, relevant_trends: List[str], trend_positioning: Dict) -> Optional[float]:
        """Calculate target price based on secular trend positioning."""
        if stock.pe_ratio and stock.pe_ratio > 0:
            current_eps = stock.current_price / stock.pe_ratio
            
            # Base P/E from sector
            sector_pe = {
                'Technology': 28.0,  # Higher for trend-exposed tech
                'Communication Services': 22.0,
                'Consumer Discretionary': 20.0,
                'Industrials': 18.0,
                'Energy': 16.0,
                'Healthcare': 18.0,
                'Financial Services': 14.0
            }
            
            base_pe = sector_pe.get(stock.sector, 20.0)
            
            # Adjust P/E based on trend positioning
            positioning_multiplier = 0.8 + (trend_positioning['overall_positioning_score'] * 0.6)  # 0.8 to 1.4
            
            # Growth premium for high-growth trends
            growth_premium = 1.0
            for trend_key in relevant_trends:
                if trend_key in self.secular_trends:
                    trend = self.secular_trends[trend_key]
                    if trend['growth_rate'] >= 0.3:  # 30%+ growth trends
                        growth_premium += 0.2
                    elif trend['growth_rate'] >= 0.2:  # 20%+ growth trends
                        growth_premium += 0.1
            
            growth_premium = min(1.5, growth_premium)  # Cap at 1.5x
            
            target_pe = base_pe * positioning_multiplier * growth_premium
            target_price = current_eps * target_pe
            
            return round(target_price, 2)
        
        return None
    
    def _assess_trend_risk(self, stock: Stock, relevant_trends: List[str], execution_capability: Dict) -> str:
        """Assess risk level based on secular trend exposure."""
        risk_factors = 0
        
        # Execution risk
        if execution_capability['execution_risk'] == 'high':
            risk_factors += 2
        elif execution_capability['execution_risk'] == 'moderate':
            risk_factors += 1
        
        # Market cap risk (smaller companies higher risk in trends)
        if stock.market_cap < 2e9:
            risk_factors += 2
        elif stock.market_cap < 10e9:
            risk_factors += 1
        
        # Trend concentration risk
        if len(relevant_trends) == 1:
            risk_factors += 1  # Single trend dependency
        
        # High growth trend risk
        high_growth_trends = 0
        for trend_key in relevant_trends:
            if trend_key in self.secular_trends:
                if self.secular_trends[trend_key]['growth_rate'] >= 0.4:
                    high_growth_trends += 1
        
        if high_growth_trends >= 2:
            risk_factors += 1  # Multiple high-growth trend exposure
        
        if risk_factors >= 4:
            return "HIGH"
        elif risk_factors >= 2:
            return "MODERATE"
        else:
            return "LOW"
    
    def _extract_key_factors(self, relevant_trends: List[str], trend_positioning: Dict, competitive_position: Dict) -> List[str]:
        """Extract key positive secular trend factors."""
        factors = []
        
        if trend_positioning['overall_positioning_score'] > 0.7:
            factors.append("Strong positioning in secular trends")
        
        if competitive_position['market_leadership'] > 0.7:
            factors.append("Market leadership in trend categories")
        
        # Add specific trend advantages
        for trend_key in relevant_trends:
            if trend_key in self.secular_trends:
                trend = self.secular_trends[trend_key]
                if trend['growth_rate'] >= 0.3:
                    factors.append(f"Exposure to high-growth {trend['name']} trend")
                elif trend['margin_profile'] >= 0.7:
                    factors.append(f"High-margin {trend['name']} opportunity")
        
        if trend_positioning['market_timing'] == 'early_leader':
            factors.append("Early leadership position in emerging trends")
        
        if len(relevant_trends) >= 2:
            factors.append("Diversified secular trend exposure")
        
        return factors[:5]  # Limit to top 5
    
    def _identify_concerns(self, stock: Stock, relevant_trends: List[str], execution_capability: Dict) -> List[str]:
        """Identify key secular trend concerns."""
        concerns = []
        
        if execution_capability['execution_risk'] == 'high':
            concerns.append("High execution risk for trend opportunities")
        
        if execution_capability['financial_resources'] < 0.5:
            concerns.append("Limited financial resources for trend investment")
        
        if not relevant_trends:
            concerns.append("Limited exposure to major secular trends")
        
        if len(relevant_trends) == 1:
            concerns.append("Single trend dependency risk")
        
        if stock.market_cap < 2e9:
            concerns.append("Small scale may limit trend participation")
        
        # Check for competitive threats
        if execution_capability['innovation_capability'] < 0.5:
            concerns.append("Innovation capability concerns")
        
        return concerns[:5]  # Limit to top 5

