"""
Ranker Agent - Final Scoring and Ranking
Takes inputs from Fundamental Agent (quantitative) and Rationale Agent (qualitative)
Provides final scoring on scale 1-10 and investment recommendations
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from .fundamental_agent import QualifiedCompany
from .rationale_agent_updated import RationaleAnalysis


@dataclass
class RankedCompany:
    """Data class for final ranked companies with 1-10 scoring"""
    ticker: str
    company_name: str
    sector: str
    market_cap: float
    
    # Input Scores (normalized to 1-10)
    fundamental_score: float  # 1-10 from quantitative analysis
    qualitative_score: float  # 1-10 from rationale analysis
    
    # Component Scores (1-10 scale)
    growth_score: float
    profitability_score: float
    debt_health_score: float
    moat_score: float
    sentiment_score: float
    trend_score: float
    
    # Final Investment Score (1-10)
    final_investment_score: float
    
    # Investment Analysis
    investment_thesis: str
    why_good_investment: str  # Detailed reasoning for the score
    key_strengths: List[str]
    key_risks: List[str]
    
    # Recommendations
    recommendation: str  # STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
    confidence_level: str  # HIGH, MEDIUM, LOW
    position_size: str  # LARGE, MEDIUM, SMALL
    
    # Supporting Data
    price_target: Optional[float]
    citations: List[str]
    
    timestamp: str


class RankerAgent:
    """Agent for final scoring and ranking of investment candidates"""
    
    def __init__(self, api_key: str = None):
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM
        if api_key:
            try:
                self.llm = ChatOpenAI(
                    model="gpt-4",
                    temperature=0.1,
                    openai_api_key=api_key
                )
            except Exception as e:
                self.logger.warning(f"Failed to initialize LLM: {e}")
                self.llm = None
        else:
            self.llm = None
    
    def rank_companies(self, 
                      qualified_companies: List[QualifiedCompany],
                      rationale_analyses: Dict[str, RationaleAnalysis],
                      market_context: Optional[Dict] = None) -> List[RankedCompany]:
        """
        Rank companies based on fundamental (quantitative) and rationale (qualitative) analysis
        Provides final scoring on 1-10 scale with detailed reasoning
        """
        try:
            self.logger.info(f"Ranking {len(qualified_companies)} companies using 3-agent system")
            
            ranked_companies = []
            
            for company in qualified_companies:
                try:
                    # Get corresponding rationale analysis
                    rationale_analysis = rationale_analyses.get(company.ticker)
                    
                    if not rationale_analysis:
                        self.logger.warning(f"No rationale analysis found for {company.ticker}, skipping")
                        continue
                    
                    # Perform comprehensive ranking
                    ranked_company = self._rank_individual_company(company, rationale_analysis, market_context)
                    ranked_companies.append(ranked_company)
                    
                    self.logger.info(f"✓ Ranked {company.ticker}: {ranked_company.final_investment_score:.1f}/10 ({ranked_company.recommendation})")
                    
                except Exception as e:
                    self.logger.error(f"Error ranking {company.ticker}: {e}")
                    continue
            
            # Sort by final investment score
            ranked_companies.sort(key=lambda x: x.final_investment_score, reverse=True)
            
            # Apply portfolio-level considerations
            ranked_companies = self._apply_portfolio_optimization(ranked_companies)
            
            # Save ranking results
            self._save_ranking_results(ranked_companies)
            
            self.logger.info(f"Completed ranking of {len(ranked_companies)} companies")
            
            return ranked_companies
            
        except Exception as e:
            self.logger.error(f"Error in ranking process: {e}")
            return []
    
    def _rank_individual_company(self, 
                                company: QualifiedCompany, 
                                rationale_analysis: RationaleAnalysis,
                                market_context: Optional[Dict]) -> RankedCompany:
        """
        Rank individual company using both fundamental and rationale analysis
        Provides detailed scoring on 1-10 scale with reasoning
        """
        
        # Normalize fundamental scores to 1-10 scale
        fundamental_score = self._normalize_to_10_scale(company.overall_score, 0, 100)
        growth_score = self._normalize_to_10_scale(company.growth_score, 0, 100)
        profitability_score = self._normalize_to_10_scale(company.profitability_score, 0, 100)
        debt_health_score = self._normalize_to_10_scale(company.debt_score, 0, 100)
        
        # Get qualitative scores (already on 1-10 scale)
        qualitative_score = rationale_analysis.overall_qualitative_score
        moat_score = rationale_analysis.moat_score
        sentiment_score = rationale_analysis.sentiment_score
        trend_score = rationale_analysis.trend_score
        
        # Calculate weighted final investment score
        # Fundamental: 50%, Qualitative: 50%
        final_investment_score = (fundamental_score * 0.5) + (qualitative_score * 0.5)
        
        # Generate investment analysis
        if self.llm:
            analysis_result = self._generate_llm_investment_analysis(
                company, rationale_analysis, final_investment_score, market_context
            )
        else:
            analysis_result = self._generate_fallback_investment_analysis(
                company, rationale_analysis, final_investment_score
            )
        
        # Determine recommendation and confidence
        recommendation, confidence_level = self._determine_recommendation(final_investment_score, analysis_result)
        
        # Determine position sizing
        position_size = self._determine_position_size(final_investment_score, confidence_level, company.market_cap)
        
        ranked_company = RankedCompany(
            ticker=company.ticker,
            company_name=company.company_name,
            sector=company.sector,
            market_cap=company.market_cap,
            
            # Input scores
            fundamental_score=round(fundamental_score, 1),
            qualitative_score=round(qualitative_score, 1),
            
            # Component scores
            growth_score=round(growth_score, 1),
            profitability_score=round(profitability_score, 1),
            debt_health_score=round(debt_health_score, 1),
            moat_score=round(moat_score, 1),
            sentiment_score=round(sentiment_score, 1),
            trend_score=round(trend_score, 1),
            
            # Final score
            final_investment_score=round(final_investment_score, 1),
            
            # Analysis
            investment_thesis=analysis_result['thesis'],
            why_good_investment=analysis_result['reasoning'],
            key_strengths=analysis_result['strengths'],
            key_risks=analysis_result['risks'],
            
            # Recommendations
            recommendation=recommendation,
            confidence_level=confidence_level,
            position_size=position_size,
            
            # Supporting data
            price_target=analysis_result.get('price_target'),
            citations=rationale_analysis.citations,
            
            timestamp=datetime.now().isoformat()
        )
        
        return ranked_company
    
    def _normalize_to_10_scale(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize a value to 1-10 scale"""
        if max_val == min_val:
            return 5.0
        
        normalized = ((value - min_val) / (max_val - min_val)) * 9 + 1
        return max(1.0, min(10.0, normalized))

    def _generate_llm_investment_analysis(self, 
                                        company: QualifiedCompany, 
                                        rationale_analysis: RationaleAnalysis,
                                        final_score: float,
                                        market_context: Optional[Dict]) -> Dict:
        """Generate comprehensive LLM-powered investment analysis"""
        try:
            if not self.llm:
                return self._generate_fallback_investment_analysis(company, rationale_analysis, final_score)
            
            prompt = f"""
Provide comprehensive investment analysis for {company.company_name} ({company.ticker}):

QUANTITATIVE ANALYSIS (Fundamental Agent):
- Overall Fundamental Score: {self._normalize_to_10_scale(company.overall_score, 0, 100):.1f}/10
- Growth Score: {self._normalize_to_10_scale(company.growth_score, 0, 100):.1f}/10
- Profitability Score: {self._normalize_to_10_scale(company.profitability_score, 0, 100):.1f}/10
- Debt Health Score: {self._normalize_to_10_scale(company.debt_score, 0, 100):.1f}/10

Key Metrics:
- Revenue Growth (5Y): {company.revenue_growth_5y:.1f}%
- ROE: {company.roe_ttm:.1f}%
- ROIC: {company.roic_ttm:.1f}%
- Current Ratio: {company.current_ratio:.2f}
- Debt/EBITDA: {company.debt_to_ebitda:.2f}

QUALITATIVE ANALYSIS (Rationale Agent):
- Overall Qualitative Score: {rationale_analysis.overall_qualitative_score:.1f}/10
- Moat Score: {rationale_analysis.moat_score:.1f}/10 ({rationale_analysis.moat_strength} {rationale_analysis.moat_type})
- Sentiment Score: {rationale_analysis.sentiment_score:.1f}/10 ({rationale_analysis.sentiment_trend})
- Trend Score: {rationale_analysis.trend_score:.1f}/10 ({rationale_analysis.trend_alignment})

Research Insights:
{chr(10).join(f"- {insight}" for insight in rationale_analysis.key_insights)}

FINAL INVESTMENT SCORE: {final_score:.1f}/10

Provide your analysis in this exact format:

INVESTMENT_THESIS: [Compelling 2-3 sentence investment case]

WHY_GOOD_INVESTMENT: [Detailed reasoning explaining the {final_score:.1f}/10 score across multiple dimensions - quantitative strengths, qualitative advantages, market position, growth prospects, and risk factors]

KEY_STRENGTHS:
1. [Primary quantitative strength]
2. [Primary qualitative advantage]
3. [Additional competitive edge]

KEY_RISKS:
1. [Primary risk factor]
2. [Secondary concern]
3. [Additional risk consideration]

PRICE_TARGET: [12-month price target if applicable, or "N/A"]

Focus on explaining why this deserves the calculated investment score.
"""
            
            messages = [
                SystemMessage(content="You are a senior investment analyst providing final investment recommendations with detailed scoring rationale."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            return self._parse_llm_investment_response(response.content)
            
        except Exception as e:
            self.logger.error(f"Error generating LLM investment analysis: {e}")
            return self._generate_fallback_investment_analysis(company, rationale_analysis, final_score)

    def _parse_llm_investment_response(self, response: str) -> Dict:
        """Parse LLM investment analysis response"""
        try:
            lines = response.strip().split('\n')
            
            result = {
                'thesis': '',
                'reasoning': '',
                'strengths': [],
                'risks': [],
                'price_target': None
            }
            
            current_section = None
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('INVESTMENT_THESIS:'):
                    result['thesis'] = line.replace('INVESTMENT_THESIS:', '').strip()
                    current_section = 'thesis'
                elif line.startswith('WHY_GOOD_INVESTMENT:'):
                    result['reasoning'] = line.replace('WHY_GOOD_INVESTMENT:', '').strip()
                    current_section = 'reasoning'
                elif line.startswith('KEY_STRENGTHS:'):
                    current_section = 'strengths'
                elif line.startswith('KEY_RISKS:'):
                    current_section = 'risks'
                elif line.startswith('PRICE_TARGET:'):
                    try:
                        price_str = line.replace('PRICE_TARGET:', '').strip()
                        if price_str != 'N/A':
                            result['price_target'] = float(price_str.replace('$', '').replace(',', ''))
                    except:
                        result['price_target'] = None
                elif current_section == 'strengths' and line.startswith(('1.', '2.', '3.')):
                    result['strengths'].append(line[2:].strip())
                elif current_section == 'risks' and line.startswith(('1.', '2.', '3.')):
                    result['risks'].append(line[2:].strip())
                elif current_section in ['thesis', 'reasoning'] and line and not line.startswith(('KEY_', 'PRICE_', 'WHY_')):
                    result[current_section] += " " + line
            
            # Clean up text
            result['thesis'] = result['thesis'].strip()
            result['reasoning'] = result['reasoning'].strip()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error parsing LLM investment response: {e}")
            return {
                'thesis': 'Investment opportunity based on quantitative and qualitative analysis',
                'reasoning': 'Score reflects combination of fundamental metrics and business quality factors',
                'strengths': ['Strong fundamentals', 'Good business quality'],
                'risks': ['Market volatility', 'Execution risk'],
                'price_target': None
            }

    def _generate_fallback_investment_analysis(self, 
                                             company: QualifiedCompany, 
                                             rationale_analysis: RationaleAnalysis,
                                             final_score: float) -> Dict:
        """Generate fallback investment analysis without LLM"""
        
        # Generate thesis based on scores
        if final_score >= 8.5:
            thesis = f"Exceptional investment opportunity combining outstanding fundamentals with superior business quality in {company.sector}."
        elif final_score >= 7.5:
            thesis = f"Strong investment candidate with solid fundamentals and good competitive positioning in {company.sector}."
        elif final_score >= 6.5:
            thesis = f"Attractive investment with reasonable fundamentals and adequate business quality."
        elif final_score >= 5.5:
            thesis = f"Moderate investment opportunity with mixed fundamental and qualitative characteristics."
        else:
            thesis = f"Below-average investment prospect with concerning metrics or business quality issues."
        
        # Generate detailed reasoning
        reasoning = f"The {final_score:.1f}/10 investment score reflects: "
        reasoning_parts = []
        
        fundamental_score = self._normalize_to_10_scale(company.overall_score, 0, 100)
        if fundamental_score >= 7.0:
            reasoning_parts.append("strong quantitative fundamentals")
        elif fundamental_score >= 5.0:
            reasoning_parts.append("adequate quantitative metrics")
        else:
            reasoning_parts.append("weak fundamental performance")
        
        if rationale_analysis.overall_qualitative_score >= 7.0:
            reasoning_parts.append("superior business quality and competitive positioning")
        elif rationale_analysis.overall_qualitative_score >= 5.0:
            reasoning_parts.append("reasonable business quality characteristics")
        else:
            reasoning_parts.append("concerning business quality factors")
        
        if rationale_analysis.moat_score >= 7.0:
            reasoning_parts.append(f"strong {rationale_analysis.moat_type.lower()} competitive moat")
        
        if rationale_analysis.trend_score >= 7.0:
            reasoning_parts.append("favorable secular trend alignment")
        
        reasoning += ", ".join(reasoning_parts) + "."
        
        # Generate strengths
        strengths = []
        if company.growth_score >= 70:
            strengths.append("Strong revenue and earnings growth trajectory")
        if company.profitability_score >= 70:
            strengths.append("Superior profitability and operational efficiency")
        if company.debt_score >= 70:
            strengths.append("Conservative debt structure and financial strength")
        if rationale_analysis.moat_score >= 7.0:
            strengths.append(f"Strong {rationale_analysis.moat_type.lower()} competitive advantage")
        if rationale_analysis.sentiment_score >= 7.0:
            strengths.append("Positive market sentiment and momentum")
        if rationale_analysis.trend_score >= 7.0:
            strengths.append("Well-positioned for favorable industry trends")
        
        # Generate risks
        risks = []
        if company.growth_score < 50:
            risks.append("Slowing growth momentum and expansion challenges")
        if company.profitability_score < 50:
            risks.append("Margin pressure and operational efficiency concerns")
        if company.debt_score < 50:
            risks.append("Elevated debt levels and financial leverage risks")
        if rationale_analysis.moat_score < 5.0:
            risks.append("Weak competitive positioning and market share vulnerability")
        if rationale_analysis.sentiment_score < 4.0:
            risks.append("Negative sentiment and potential momentum reversal")
        if rationale_analysis.trend_score < 4.0:
            risks.append("Unfavorable industry trends and secular headwinds")
        
        # Ensure we have at least some strengths and risks
        if not strengths:
            strengths = ["Meets basic investment criteria", "Established market presence"]
        if not risks:
            risks = ["General market volatility", "Execution and operational risks"]
        
        return {
            'thesis': thesis,
            'reasoning': reasoning,
            'strengths': strengths[:3],  # Limit to top 3
            'risks': risks[:3],  # Limit to top 3
            'price_target': None
        }

    def _determine_recommendation(self, final_score: float, analysis_result: Dict) -> Tuple[str, str]:
        """Determine investment recommendation and confidence level"""
        
        # Base recommendation on score
        if final_score >= 8.5:
            recommendation = "STRONG_BUY"
            confidence = "HIGH"
        elif final_score >= 7.5:
            recommendation = "BUY"
            confidence = "HIGH"
        elif final_score >= 6.5:
            recommendation = "BUY"
            confidence = "MEDIUM"
        elif final_score >= 5.5:
            recommendation = "HOLD"
            confidence = "MEDIUM"
        elif final_score >= 4.0:
            recommendation = "HOLD"
            confidence = "LOW"
        else:
            recommendation = "SELL"
            confidence = "LOW"
        
        return recommendation, confidence

    def _determine_position_size(self, final_score: float, confidence_level: str, market_cap: float) -> str:
        """Determine appropriate position size"""
        
        # Base sizing on score and confidence
        if final_score >= 8.0 and confidence_level == "HIGH":
            base_size = "LARGE"
        elif final_score >= 7.0 and confidence_level in ["HIGH", "MEDIUM"]:
            base_size = "MEDIUM"
        else:
            base_size = "SMALL"
        
        # Adjust for market cap (liquidity considerations)
        if market_cap < 1e9:  # Micro cap - reduce size
            if base_size == "LARGE":
                return "MEDIUM"
            elif base_size == "MEDIUM":
                return "SMALL"
            else:
                return "SMALL"
        elif market_cap < 5e9:  # Small cap - slight reduction
            if base_size == "LARGE":
                return "MEDIUM"
            else:
                return base_size
        else:  # Large cap - no adjustment needed
            return base_size

    def _apply_portfolio_optimization(self, ranked_companies: List[RankedCompany]) -> List[RankedCompany]:
        """Apply portfolio-level optimization considerations"""
        
        # Sector diversification adjustments
        sector_counts = {}
        for company in ranked_companies:
            sector_counts[company.sector] = sector_counts.get(company.sector, 0) + 1
        
        # Adjust position sizes for over-concentration
        for company in ranked_companies:
            if sector_counts[company.sector] > 3:  # Too many in one sector
                if company.position_size == "LARGE":
                    company.position_size = "MEDIUM"
                elif company.position_size == "MEDIUM":
                    company.position_size = "SMALL"
        
        return ranked_companies

    def _save_ranking_results(self, companies: List[RankedCompany]):
        """Save ranking results to tracing folder"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"tracing/ranking_results_{timestamp}.json"
            
            results = {
                'timestamp': timestamp,
                'total_ranked': len(companies),
                'companies': [asdict(company) for company in companies],
                'summary': self._generate_ranking_summary(companies)
            }
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
                
            self.logger.info(f"Saved ranking results to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving ranking results: {e}")
    
    def _generate_ranking_summary(self, companies: List[RankedCompany]) -> Dict:
        """Generate summary statistics for ranking results"""
        if not companies:
            return {}
        
        recommendations = {}
        confidence_levels = {}
        position_sizes = {}
        
        for company in companies:
            recommendations[company.recommendation] = recommendations.get(company.recommendation, 0) + 1
            confidence_levels[company.confidence_level] = confidence_levels.get(company.confidence_level, 0) + 1
            position_sizes[company.position_size] = position_sizes.get(company.position_size, 0) + 1
        
        return {
            'total_companies': len(companies),
            'avg_final_score': sum(c.final_investment_score for c in companies) / len(companies),
            'avg_fundamental_score': sum(c.fundamental_score for c in companies) / len(companies),
            'avg_qualitative_score': sum(c.qualitative_score for c in companies) / len(companies),
            'top_5_tickers': [c.ticker for c in companies[:5]],
            'recommendation_distribution': recommendations,
            'confidence_distribution': confidence_levels,
            'position_size_distribution': position_sizes,
            'sectors_represented': list(set(c.sector for c in companies)),
            'high_score_count': len([c for c in companies if c.final_investment_score >= 8.0])
        }

    def get_portfolio_recommendations(self, ranked_companies: List[RankedCompany], max_positions: int = 10) -> Dict:
        """Generate final portfolio recommendations"""
        
        # Filter to top candidates
        top_candidates = [c for c in ranked_companies if c.recommendation in ['STRONG_BUY', 'BUY']]
        top_candidates = top_candidates[:max_positions]
        
        # Calculate position weights
        total_score = sum(c.final_investment_score for c in top_candidates)
        
        portfolio = []
        for company in top_candidates:
            weight = (company.final_investment_score / total_score) * 100 if total_score > 0 else 0
            
            # Adjust weight based on position size
            if company.position_size == "LARGE":
                weight *= 1.2
            elif company.position_size == "SMALL":
                weight *= 0.8
            
            portfolio.append({
                'ticker': company.ticker,
                'company_name': company.company_name,
                'weight': round(weight, 2),
                'recommendation': company.recommendation,
                'final_score': company.final_investment_score,
                'investment_thesis': company.investment_thesis
            })
        
        # Normalize weights to 100%
        total_weight = sum(p['weight'] for p in portfolio)
        if total_weight > 0:
            for position in portfolio:
                position['weight'] = round((position['weight'] / total_weight) * 100, 2)
        
        return {
            'portfolio': portfolio,
            'total_positions': len(portfolio),
            'avg_score': sum(p['final_score'] for p in portfolio) / len(portfolio) if portfolio else 0,
            'risk_profile': self._assess_portfolio_risk(top_candidates),
            'sector_allocation': self._calculate_sector_allocation(top_candidates)
        }
    
    def _assess_portfolio_risk(self, companies: List[RankedCompany]) -> str:
        """Assess overall portfolio risk level"""
        if not companies:
            return "UNKNOWN"
        
        # Risk based on confidence levels and scores
        high_confidence = len([c for c in companies if c.confidence_level == "HIGH"])
        total = len(companies)
        
        if high_confidence / total >= 0.7:
            return "LOW"
        elif high_confidence / total >= 0.4:
            return "MEDIUM"
        else:
            return "HIGH"
    
    def _calculate_sector_allocation(self, companies: List[RankedCompany]) -> Dict[str, float]:
        """Calculate sector allocation percentages"""
        if not companies:
            return {}
        
        sector_counts = {}
        for company in companies:
            sector_counts[company.sector] = sector_counts.get(company.sector, 0) + 1
        
        total = len(companies)
        return {sector: round((count / total) * 100, 1) for sector, count in sector_counts.items()}

