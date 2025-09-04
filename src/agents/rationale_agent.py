"""
Rationale Agent for equity portfolio construction.
Analyzes business quality using a 7-step framework to determine if a company is a "Great Business".
Based on fundamental business analysis principles.
"""

from typing import Dict, List, Any, Optional
import json
import math
from datetime import datetime
from langchain_core.messages import HumanMessage
from .base_agent import BaseAgent, Stock, AgentAnalysis, InvestmentDecision, RiskTolerance

class RationaleAgent(BaseAgent):
    """
    Rationale Agent specializes in analyzing business quality using a comprehensive
    7-step framework to evaluate if a company represents a "Great Business" investment.
    """
    
    def __init__(self, risk_tolerance: RiskTolerance, llm_client=None):
        super().__init__(risk_tolerance, llm_client)
        self.agent_name = "rationale"
        
    def analyze(self, stock: Stock, market_data: Optional[Dict] = None) -> AgentAnalysis:
        """
        Analyze stock using the 7-step Great Business framework.
        
        Args:
            stock: Stock information to analyze
            market_data: Optional market context data
            
        Returns:
            AgentAnalysis with rationale-based recommendation
        """
        try:
            # Perform 7-step business quality analysis
            business_metrics = self._analyze_business_quality(stock, market_data)
            growth_analysis = self._analyze_growth_rates(stock, market_data)
            moat_analysis = self._analyze_competitive_advantage(stock, market_data)
            efficiency_analysis = self._analyze_operational_efficiency(stock, market_data)
            debt_analysis = self._analyze_debt_structure(stock, market_data)
            
            # Generate LLM-enhanced analysis if available
            if self.llm_client:
                llm_analysis = self._get_llm_analysis(stock, business_metrics, growth_analysis, 
                                                    moat_analysis, efficiency_analysis, debt_analysis)
                recommendation, confidence, reasoning = self._parse_llm_response(llm_analysis)
            else:
                recommendation, confidence, reasoning = self._fallback_analysis(
                    business_metrics, growth_analysis, moat_analysis, efficiency_analysis, debt_analysis)
            
            # Adjust for risk tolerance
            confidence = self.adjust_for_risk_tolerance(confidence)
            
            # Extract key factors and concerns
            key_factors = self._extract_key_factors(business_metrics, growth_analysis, moat_analysis, efficiency_analysis)
            concerns = self._identify_concerns(business_metrics, growth_analysis, debt_analysis)
            
            return AgentAnalysis(
                agent_name=self.agent_name,
                stock_symbol=stock.symbol,
                recommendation=recommendation,
                confidence_score=confidence,
                target_price=self._calculate_rationale_target_price(stock, business_metrics, growth_analysis),
                risk_assessment=self._assess_business_risk(business_metrics, debt_analysis, efficiency_analysis),
                key_factors=key_factors,
                concerns=concerns,
                reasoning=reasoning
            )
            
        except Exception as e:
            self.logger.error(f"Error in rationale analysis for {stock.symbol}: {e}")
            return AgentAnalysis(
                agent_name=self.agent_name,
                stock_symbol=stock.symbol,
                recommendation=InvestmentDecision.HOLD,
                confidence_score=0.3,
                risk_assessment="HIGH",
                key_factors=[],
                concerns=["Analysis error occurred"],
                reasoning=f"Unable to complete rationale analysis: {str(e)}"
            )
    
    def _analyze_business_quality(self, stock: Stock, market_data: Optional[Dict]) -> Dict[str, Any]:
        """Step 1: Analyze consistently increasing sales, net income, and cash flow."""
        metrics = {
            'sales_growth_consistency': 0.5,
            'net_income_consistency': 0.5,
            'cash_flow_consistency': 0.5,
            'margin_stability': 0.5,
            'overall_quality_score': 0.5
        }
        
        # Sector-based quality assumptions (simplified)
        quality_sectors = {
            'Technology': 0.7,  # Generally high quality
            'Healthcare': 0.65,
            'Consumer': 0.6,
            'Financial': 0.55,
            'Industrial': 0.5,
            'Utilities': 0.6,  # Stable but lower growth
            'Energy': 0.4,     # Cyclical
            'Materials': 0.45,
            'Real Estate': 0.5,
            'Telecommunications': 0.55
        }
        
        base_quality = quality_sectors.get(stock.sector, 0.5)
        
        # Market cap influence on business quality
        if stock.market_cap >= 100e9:  # Mega cap - proven quality
            metrics['business_maturity'] = 0.8
            base_quality += 0.1
        elif stock.market_cap >= 10e9:  # Large cap
            metrics['business_maturity'] = 0.7
            base_quality += 0.05
        elif stock.market_cap >= 2e9:  # Mid cap
            metrics['business_maturity'] = 0.6
        else:  # Small cap - higher risk
            metrics['business_maturity'] = 0.4
            base_quality -= 0.1
        
        # P/E ratio influence on quality perception
        if stock.pe_ratio:
            if 15 <= stock.pe_ratio <= 25:  # Reasonable valuation suggests quality
                base_quality += 0.05
            elif stock.pe_ratio > 35:  # Very high P/E might indicate speculation
                base_quality -= 0.1
        
        metrics['overall_quality_score'] = max(0.1, min(0.9, base_quality))
        
        # Integrate market data if available
        if market_data and 'financial_data' in market_data:
            fin_data = market_data['financial_data']
            if 'sales_growth_5y' in fin_data:
                metrics['sales_growth_consistency'] = min(0.9, fin_data['sales_growth_5y'] / 20.0)  # 20% = 0.9 score
            if 'margin_trend' in fin_data:
                metrics['margin_stability'] = fin_data['margin_trend']
        
        return metrics
    
    def _analyze_growth_rates(self, stock: Stock, market_data: Optional[Dict]) -> Dict[str, Any]:
        """Step 2: Analyze positive growth rates."""
        growth = {
            'eps_growth_5y': 0.0,
            'revenue_growth_estimate': 0.0,
            'growth_sustainability': 0.5,
            'growth_quality_score': 0.5
        }
        
        # Sector-based growth expectations
        growth_sectors = {
            'Technology': 0.15,      # 15% expected growth
            'Healthcare': 0.08,      # 8% expected growth
            'Consumer': 0.06,        # 6% expected growth
            'Financial': 0.05,       # 5% expected growth
            'Industrial': 0.04,      # 4% expected growth
            'Utilities': 0.03,       # 3% expected growth
            'Energy': 0.02,          # 2% expected growth (cyclical)
            'Materials': 0.03,       # 3% expected growth
            'Real Estate': 0.04,     # 4% expected growth
            'Telecommunications': 0.02  # 2% expected growth
        }
        
        expected_growth = growth_sectors.get(stock.sector, 0.05)
        growth['revenue_growth_estimate'] = expected_growth
        
        # Market cap influence on growth potential
        if stock.market_cap >= 500e9:  # Mega cap - slower growth
            growth['growth_sustainability'] = 0.6
            expected_growth *= 0.8
        elif stock.market_cap >= 50e9:  # Large cap
            growth['growth_sustainability'] = 0.7
        elif stock.market_cap >= 5e9:  # Mid cap - higher growth potential
            growth['growth_sustainability'] = 0.8
            expected_growth *= 1.2
        else:  # Small cap - highest growth potential but riskier
            growth['growth_sustainability'] = 0.6
            expected_growth *= 1.5
        
        # P/E ratio as growth indicator
        if stock.pe_ratio:
            # PEG-like analysis (simplified)
            implied_growth = stock.pe_ratio / 20.0  # Market average P/E
            if implied_growth > 1.5:  # High growth expectations
                growth['market_growth_expectation'] = 'high'
                growth['growth_quality_score'] = 0.7
            elif implied_growth > 1.0:  # Moderate growth
                growth['market_growth_expectation'] = 'moderate'
                growth['growth_quality_score'] = 0.6
            else:  # Low growth expectations
                growth['market_growth_expectation'] = 'low'
                growth['growth_quality_score'] = 0.4
        
        growth['eps_growth_5y'] = expected_growth
        
        # Integrate market data if available
        if market_data and 'growth_data' in market_data:
            growth_data = market_data['growth_data']
            growth.update(growth_data)
        
        return growth
    
    def _analyze_competitive_advantage(self, stock: Stock, market_data: Optional[Dict]) -> Dict[str, Any]:
        """Step 3: Analyze sustainable competitive advantage (Economic Moat)."""
        moat = {
            'moat_type': 'none',
            'moat_strength': 'narrow',
            'brand_power': 0.5,
            'barriers_to_entry': 0.5,
            'economies_of_scale': 0.5,
            'network_effects': 0.5,
            'switching_costs': 0.5,
            'overall_moat_score': 0.5
        }
        
        # Sector-based moat characteristics
        sector_moats = {
            'Technology': {
                'moat_type': 'network_effects',
                'moat_strength': 'wide',
                'brand_power': 0.7,
                'barriers_to_entry': 0.8,
                'economies_of_scale': 0.8,
                'network_effects': 0.9,
                'switching_costs': 0.8
            },
            'Healthcare': {
                'moat_type': 'regulatory',
                'moat_strength': 'wide',
                'brand_power': 0.6,
                'barriers_to_entry': 0.9,
                'economies_of_scale': 0.6,
                'network_effects': 0.4,
                'switching_costs': 0.7
            },
            'Consumer': {
                'moat_type': 'brand',
                'moat_strength': 'narrow',
                'brand_power': 0.8,
                'barriers_to_entry': 0.5,
                'economies_of_scale': 0.7,
                'network_effects': 0.3,
                'switching_costs': 0.6
            },
            'Financial': {
                'moat_type': 'regulatory',
                'moat_strength': 'narrow',
                'brand_power': 0.6,
                'barriers_to_entry': 0.7,
                'economies_of_scale': 0.8,
                'network_effects': 0.5,
                'switching_costs': 0.7
            },
            'Utilities': {
                'moat_type': 'regulatory',
                'moat_strength': 'wide',
                'brand_power': 0.4,
                'barriers_to_entry': 0.9,
                'economies_of_scale': 0.8,
                'network_effects': 0.2,
                'switching_costs': 0.9
            }
        }
        
        if stock.sector in sector_moats:
            sector_moat = sector_moats[stock.sector]
            moat.update(sector_moat)
        
        # Market cap influence on moat strength
        if stock.market_cap >= 100e9:  # Mega cap - strong moats
            moat['economies_of_scale'] = min(0.9, moat['economies_of_scale'] + 0.2)
            moat['brand_power'] = min(0.9, moat['brand_power'] + 0.1)
        elif stock.market_cap < 5e9:  # Small cap - weaker moats
            moat['economies_of_scale'] = max(0.2, moat['economies_of_scale'] - 0.2)
            moat['brand_power'] = max(0.2, moat['brand_power'] - 0.1)
        
        # Calculate overall moat score
        moat['overall_moat_score'] = (
            moat['brand_power'] * 0.2 +
            moat['barriers_to_entry'] * 0.25 +
            moat['economies_of_scale'] * 0.2 +
            moat['network_effects'] * 0.2 +
            moat['switching_costs'] * 0.15
        )
        
        # Integrate market data if available
        if market_data and 'competitive_data' in market_data:
            comp_data = market_data['competitive_data']
            moat.update(comp_data)
        
        return moat
    
    def _analyze_operational_efficiency(self, stock: Stock, market_data: Optional[Dict]) -> Dict[str, Any]:
        """Step 4: Analyze profitability and operational efficiency."""
        efficiency = {
            'roe_consistency': 0.5,
            'roic_consistency': 0.5,
            'revenue_efficiency': 0.5,
            'cash_conversion': 0.5,
            'overall_efficiency_score': 0.5
        }
        
        # Sector-based efficiency expectations
        efficiency_benchmarks = {
            'Technology': {'target_roe': 0.20, 'target_roic': 0.15, 'efficiency_score': 0.7},
            'Healthcare': {'target_roe': 0.15, 'target_roic': 0.12, 'efficiency_score': 0.65},
            'Consumer': {'target_roe': 0.15, 'target_roic': 0.12, 'efficiency_score': 0.6},
            'Financial': {'target_roe': 0.12, 'target_roic': 0.10, 'efficiency_score': 0.55},
            'Industrial': {'target_roe': 0.12, 'target_roic': 0.10, 'efficiency_score': 0.5},
            'Utilities': {'target_roe': 0.10, 'target_roic': 0.08, 'efficiency_score': 0.6},
            'Energy': {'target_roe': 0.08, 'target_roic': 0.06, 'efficiency_score': 0.4},
            'Materials': {'target_roe': 0.10, 'target_roic': 0.08, 'efficiency_score': 0.45}
        }
        
        benchmark = efficiency_benchmarks.get(stock.sector, 
                                            {'target_roe': 0.12, 'target_roic': 0.10, 'efficiency_score': 0.5})
        
        efficiency['target_roe'] = benchmark['target_roe']
        efficiency['target_roic'] = benchmark['target_roic']
        efficiency['overall_efficiency_score'] = benchmark['efficiency_score']
        
        # Market cap influence on efficiency
        if stock.market_cap >= 50e9:  # Large companies typically more efficient
            efficiency['overall_efficiency_score'] += 0.1
            efficiency['cash_conversion'] = 0.7
        elif stock.market_cap < 2e9:  # Small companies may be less efficient
            efficiency['overall_efficiency_score'] -= 0.1
            efficiency['cash_conversion'] = 0.4
        
        # P/E ratio as efficiency indicator
        if stock.pe_ratio:
            if stock.pe_ratio < 15:  # Low P/E might indicate high efficiency
                efficiency['market_efficiency_perception'] = 'high'
                efficiency['overall_efficiency_score'] += 0.05
            elif stock.pe_ratio > 30:  # High P/E might indicate lower current efficiency
                efficiency['market_efficiency_perception'] = 'growth_focused'
                efficiency['overall_efficiency_score'] -= 0.05
        
        # Integrate market data if available
        if market_data and 'efficiency_data' in market_data:
            eff_data = market_data['efficiency_data']
            if 'roe_ttm' in eff_data:
                actual_roe = eff_data['roe_ttm']
                efficiency['roe_consistency'] = min(0.9, actual_roe / benchmark['target_roe'])
            if 'roic_ttm' in eff_data:
                actual_roic = eff_data['roic_ttm']
                efficiency['roic_consistency'] = min(0.9, actual_roic / benchmark['target_roic'])
        
        return efficiency
    
    def _analyze_debt_structure(self, stock: Stock, market_data: Optional[Dict]) -> Dict[str, Any]:
        """Step 5: Analyze conservative debt structure."""
        debt = {
            'current_ratio': 1.5,  # Assumed default
            'debt_to_ebitda': 2.0,  # Assumed default
            'debt_service_ratio': 0.2,  # Assumed default
            'debt_quality_score': 0.5,
            'financial_strength': 'moderate'
        }
        
        # Sector-based debt tolerance
        debt_tolerance = {
            'Technology': {'max_debt_ebitda': 1.5, 'min_current_ratio': 1.5, 'tolerance_score': 0.8},
            'Healthcare': {'max_debt_ebitda': 2.0, 'min_current_ratio': 1.3, 'tolerance_score': 0.7},
            'Consumer': {'max_debt_ebitda': 2.5, 'min_current_ratio': 1.2, 'tolerance_score': 0.6},
            'Financial': {'max_debt_ebitda': 5.0, 'min_current_ratio': 1.1, 'tolerance_score': 0.5},  # Different metrics
            'Industrial': {'max_debt_ebitda': 3.0, 'min_current_ratio': 1.2, 'tolerance_score': 0.5},
            'Utilities': {'max_debt_ebitda': 4.0, 'min_current_ratio': 1.0, 'tolerance_score': 0.6},  # Capital intensive
            'Energy': {'max_debt_ebitda': 3.5, 'min_current_ratio': 1.1, 'tolerance_score': 0.4},
            'Materials': {'max_debt_ebitda': 3.0, 'min_current_ratio': 1.1, 'tolerance_score': 0.45}
        }
        
        tolerance = debt_tolerance.get(stock.sector, 
                                     {'max_debt_ebitda': 2.5, 'min_current_ratio': 1.2, 'tolerance_score': 0.5})
        
        debt['sector_max_debt_ebitda'] = tolerance['max_debt_ebitda']
        debt['sector_min_current_ratio'] = tolerance['min_current_ratio']
        
        # Market cap influence on debt capacity
        if stock.market_cap >= 50e9:  # Large companies can handle more debt
            debt['debt_capacity'] = 'high'
            debt['current_ratio'] = 2.0  # Assume better liquidity
            debt['debt_to_ebitda'] = 1.5  # Assume conservative debt
            debt['debt_quality_score'] = 0.7
        elif stock.market_cap >= 10e9:  # Mid-large cap
            debt['debt_capacity'] = 'moderate'
            debt['current_ratio'] = 1.5
            debt['debt_to_ebitda'] = 2.0
            debt['debt_quality_score'] = 0.6
        else:  # Smaller companies - should be more conservative
            debt['debt_capacity'] = 'limited'
            debt['current_ratio'] = 1.2
            debt['debt_to_ebitda'] = 2.5
            debt['debt_quality_score'] = 0.5
        
        # Assess financial strength
        if debt['current_ratio'] >= tolerance['min_current_ratio'] and debt['debt_to_ebitda'] <= tolerance['max_debt_ebitda']:
            debt['financial_strength'] = 'strong'
            debt['debt_quality_score'] += 0.2
        elif debt['debt_to_ebitda'] > tolerance['max_debt_ebitda'] * 1.5:
            debt['financial_strength'] = 'weak'
            debt['debt_quality_score'] -= 0.2
        
        debt['debt_quality_score'] = max(0.1, min(0.9, debt['debt_quality_score']))
        
        # Integrate market data if available
        if market_data and 'debt_data' in market_data:
            debt_data = market_data['debt_data']
            debt.update(debt_data)
        
        return debt
    
    def _get_llm_analysis(self, stock: Stock, business_metrics: Dict, growth_analysis: Dict, 
                         moat_analysis: Dict, efficiency_analysis: Dict, debt_analysis: Dict) -> str:
        """Get enhanced rationale analysis from LLM."""
        analysis_prompt = f"""
        As a Business Quality Analysis expert using the 7-Step Great Business Framework, analyze:
        
        Company: {stock.company_name} ({stock.symbol})
        Sector: {stock.sector}
        Current Price: ${stock.current_price:.2f}
        Market Cap: {self.format_currency(stock.market_cap)}
        
        7-Step Business Quality Analysis:
        
        1. Business Quality Metrics:
        - Overall Quality Score: {business_metrics['overall_quality_score']:.2f}
        - Business Maturity: {business_metrics.get('business_maturity', 'N/A')}
        - Sector Quality Rating: {business_metrics['overall_quality_score']:.2f}
        
        2. Growth Analysis:
        - Expected EPS Growth: {growth_analysis['eps_growth_5y']*100:.1f}%
        - Growth Sustainability: {growth_analysis['growth_sustainability']:.2f}
        - Market Growth Expectation: {growth_analysis.get('market_growth_expectation', 'moderate')}
        
        3. Competitive Advantage (Moat):
        - Moat Type: {moat_analysis['moat_type']}
        - Moat Strength: {moat_analysis['moat_strength']}
        - Overall Moat Score: {moat_analysis['overall_moat_score']:.2f}
        - Brand Power: {moat_analysis['brand_power']:.2f}
        - Barriers to Entry: {moat_analysis['barriers_to_entry']:.2f}
        
        4. Operational Efficiency:
        - Target ROE: {efficiency_analysis['target_roe']*100:.1f}%
        - Target ROIC: {efficiency_analysis['target_roic']*100:.1f}%
        - Overall Efficiency Score: {efficiency_analysis['overall_efficiency_score']:.2f}
        
        5. Debt Structure:
        - Current Ratio: {debt_analysis['current_ratio']:.2f}
        - Debt to EBITDA: {debt_analysis['debt_to_ebitda']:.2f}
        - Financial Strength: {debt_analysis['financial_strength']}
        - Debt Quality Score: {debt_analysis['debt_quality_score']:.2f}
        
        Risk Tolerance: {self.risk_tolerance.value}
        
        Based on the 7-step Great Business analysis, provide:
        1. Investment recommendation (BUY/SELL/HOLD/AVOID)
        2. Confidence level (0.0 to 1.0)
        3. Business quality assessment
        4. Key competitive advantages
        5. Main business risks
        6. Detailed reasoning
        
        Focus on long-term business quality, sustainable competitive advantages, and financial strength.
        """
        
        if self.llm_client:
            response = self.llm_client.invoke([HumanMessage(content=analysis_prompt)])
            return response.content
        else:
            return self._fallback_analysis(business_metrics, growth_analysis, moat_analysis, 
                                         efficiency_analysis, debt_analysis)[2]
    
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
    
    def _fallback_analysis(self, business_metrics: Dict, growth_analysis: Dict, moat_analysis: Dict, 
                          efficiency_analysis: Dict, debt_analysis: Dict) -> tuple:
        """Fallback analysis when LLM is not available."""
        # Weighted business quality score
        overall_score = (
            business_metrics['overall_quality_score'] * 0.25 +
            growth_analysis['growth_quality_score'] * 0.20 +
            moat_analysis['overall_moat_score'] * 0.25 +
            efficiency_analysis['overall_efficiency_score'] * 0.20 +
            debt_analysis['debt_quality_score'] * 0.10
        )
        
        # Determine recommendation based on business quality
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
            confidence = 0.75
        
        reasoning = f"""
        7-Step Great Business Analysis Summary:
        - Overall Business Quality Score: {overall_score:.2f}
        - Business Maturity: {business_metrics.get('business_maturity', 'N/A')}
        - Growth Quality: {growth_analysis['growth_quality_score']:.2f}
        - Competitive Moat: {moat_analysis['moat_strength']} ({moat_analysis['overall_moat_score']:.2f})
        - Operational Efficiency: {efficiency_analysis['overall_efficiency_score']:.2f}
        - Financial Strength: {debt_analysis['financial_strength']} ({debt_analysis['debt_quality_score']:.2f})
        
        Recommendation based on comprehensive business quality assessment using the 7-step framework.
        """
        
        return recommendation, confidence, reasoning
    
    def _calculate_rationale_target_price(self, stock: Stock, business_metrics: Dict, growth_analysis: Dict) -> Optional[float]:
        """Calculate target price based on business quality analysis."""
        if stock.pe_ratio and stock.pe_ratio > 0:
            # Quality-adjusted P/E target
            current_eps = stock.current_price / stock.pe_ratio
            
            # Base P/E from sector
            sector_pe = {
                'Technology': 25.0,
                'Healthcare': 20.0,
                'Consumer': 18.0,
                'Financial': 12.0,
                'Industrial': 16.0,
                'Utilities': 15.0,
                'Energy': 14.0,
                'Materials': 14.0
            }
            
            base_pe = sector_pe.get(stock.sector, 18.0)
            
            # Adjust P/E based on business quality
            quality_multiplier = 0.8 + (business_metrics['overall_quality_score'] * 0.4)  # 0.8 to 1.2
            growth_multiplier = 0.9 + (growth_analysis['growth_quality_score'] * 0.2)     # 0.9 to 1.1
            
            target_pe = base_pe * quality_multiplier * growth_multiplier
            target_price = current_eps * target_pe
            
            return round(target_price, 2)
        
        return None
    
    def _assess_business_risk(self, business_metrics: Dict, debt_analysis: Dict, efficiency_analysis: Dict) -> str:
        """Assess risk level based on business quality metrics."""
        risk_factors = 0
        
        if business_metrics['overall_quality_score'] < 0.4:
            risk_factors += 2  # Poor business quality
        elif business_metrics['overall_quality_score'] < 0.6:
            risk_factors += 1  # Moderate business quality
        
        if debt_analysis['financial_strength'] == 'weak':
            risk_factors += 2  # Poor financial strength
        elif debt_analysis['financial_strength'] == 'moderate':
            risk_factors += 1  # Moderate financial strength
        
        if efficiency_analysis['overall_efficiency_score'] < 0.4:
            risk_factors += 1  # Poor operational efficiency
        
        if risk_factors >= 4:
            return "HIGH"
        elif risk_factors >= 2:
            return "MODERATE"
        else:
            return "LOW"
    
    def _extract_key_factors(self, business_metrics: Dict, growth_analysis: Dict, 
                           moat_analysis: Dict, efficiency_analysis: Dict) -> List[str]:
        """Extract key positive business factors."""
        factors = []
        
        if business_metrics['overall_quality_score'] > 0.7:
            factors.append("High business quality score")
        
        if growth_analysis['growth_quality_score'] > 0.6:
            factors.append(f"Strong growth potential ({growth_analysis['eps_growth_5y']*100:.1f}% expected)")
        
        if moat_analysis['overall_moat_score'] > 0.7:
            factors.append(f"{moat_analysis['moat_strength'].title()} competitive moat ({moat_analysis['moat_type']})")
        
        if efficiency_analysis['overall_efficiency_score'] > 0.6:
            factors.append("Strong operational efficiency")
        
        if moat_analysis['brand_power'] > 0.7:
            factors.append("Strong brand power")
        
        if moat_analysis['barriers_to_entry'] > 0.7:
            factors.append("High barriers to entry")
        
        return factors[:5]  # Limit to top 5
    
    def _identify_concerns(self, business_metrics: Dict, growth_analysis: Dict, debt_analysis: Dict) -> List[str]:
        """Identify key business concerns."""
        concerns = []
        
        if business_metrics['overall_quality_score'] < 0.4:
            concerns.append("Low business quality metrics")
        
        if growth_analysis['growth_sustainability'] < 0.5:
            concerns.append("Questionable growth sustainability")
        
        if debt_analysis['financial_strength'] == 'weak':
            concerns.append("Weak financial position")
        
        if debt_analysis['debt_to_ebitda'] > debt_analysis.get('sector_max_debt_ebitda', 3.0):
            concerns.append("High debt levels")
        
        if debt_analysis['current_ratio'] < debt_analysis.get('sector_min_current_ratio', 1.2):
            concerns.append("Poor liquidity position")
        
        if business_metrics.get('business_maturity', 0.5) < 0.4:
            concerns.append("Business maturity concerns")
        
        return concerns[:5]  # Limit to top 5

