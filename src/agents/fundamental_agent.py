"""
Fundamental Agent - Quantitative Filtering
Filters stocks based on strict quantitative criteria without recommendations
"""

import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.yfinance_util import YFinanceUtil


@dataclass
class QualifiedCompany:
    """Data class for companies that meet quantitative criteria"""
    ticker: str
    company_name: str
    sector: str
    market_cap: float
    
    # Growth metrics
    revenue_growth_5y: float
    net_income_growth_5y: float
    cash_flow_growth_5y: float
    
    # Profitability metrics
    roe_ttm: float
    roic_ttm: float
    gross_margin: float
    profit_margin: float
    
    # Debt metrics
    current_ratio: float
    debt_to_ebitda: float
    debt_service_ratio: float
    
    # Quality scores
    growth_score: float
    profitability_score: float
    debt_score: float
    overall_score: float
    
    # Optional LLM commentary
    commentary: str
    
    timestamp: str


class FundamentalAgent:
    """Agent for quantitative fundamental filtering"""
    
    def __init__(self, api_key: str = None, use_llm: bool = False):
        self.logger = logging.getLogger(__name__)
        self.yfinance_util = YFinanceUtil()
        self.use_llm = use_llm
        
        # Initialize LLM only if requested
        if use_llm and api_key:
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
    
    def screen_sector(self, sector: str, market: str = 'US', max_market_cap: float = None, top_n: int = 20) -> List[QualifiedCompany]:
        """Screen sector for companies meeting quantitative criteria"""
        try:
            self.logger.info(f"Screening {sector} sector in {market} market (max cap: ${max_market_cap:,.0f})")
            
            # Get stock universe
            tickers = self.yfinance_util.get_sector_universe(sector, market, max_market_cap)
            
            qualified_companies = []
            
            for ticker in tickers:
                try:
                    # Get financial metrics
                    metrics = self.yfinance_util.get_fundamental_metrics(ticker)
                    
                    if 'error' in metrics:
                        continue
                    
                    # Apply quantitative filters
                    if self._meets_quantitative_criteria(metrics):
                        qualified_company = self._create_qualified_company(metrics, sector)
                        qualified_companies.append(qualified_company)
                        
                        self.logger.info(f"✓ {ticker} qualified with score {qualified_company.overall_score:.1f}")
                    
                except Exception as e:
                    self.logger.error(f"Error analyzing {ticker}: {e}")
                    continue
            
            # Sort by overall score
            qualified_companies.sort(key=lambda x: x.overall_score, reverse=True)
            
            # Limit results
            qualified_companies = qualified_companies[:top_n]
            
            # Save results
            self._save_screening_results(qualified_companies, sector, market)
            
            self.logger.info(f"Found {len(qualified_companies)} qualified companies in {sector}")
            
            return qualified_companies
            
        except Exception as e:
            self.logger.error(f"Error screening sector {sector}: {e}")
            return []
    
    def _meets_quantitative_criteria(self, metrics: Dict) -> bool:
        """Check if company meets strict quantitative criteria"""
        try:
            # 1. Growth Consistency Criteria
            revenue_growth = metrics.get('revenue_growth_5y', 0)
            income_growth = metrics.get('net_income_growth_5y', 0)
            cf_growth = metrics.get('cash_flow_growth_5y', 0)
            
            # Must have consistent positive growth
            if revenue_growth <= 0 or cf_growth <= 0:
                return False
            
            # If net income declining, check operating income (simplified check)
            if income_growth <= 0:
                # For now, we'll be strict and require positive net income growth
                return False
            
            # 2. Profitability and Efficiency Criteria
            roe = metrics.get('roe_ttm', 0)
            roic = metrics.get('roic_ttm', 0)
            
            # ROE and ROIC must be >= 12%
            if roe < 12 or roic < 12:
                return False
            
            # 3. Conservative Debt Criteria
            current_ratio = metrics.get('current_ratio', 0)
            debt_to_ebitda = metrics.get('debt_to_ebitda', float('inf'))
            debt_service_ratio = metrics.get('debt_service_ratio', 0)
            
            # Current ratio > 1.0
            if current_ratio <= 1.0:
                return False
            
            # Debt to EBITDA < 3.0
            if debt_to_ebitda >= 3.0:
                return False
            
            # Debt service ratio < 30% (we calculate as percentage)
            if debt_service_ratio > 0 and debt_service_ratio < 30:
                return False
            
            # 4. Additional Quality Checks
            gross_margin = metrics.get('gross_margin', 0)
            profit_margin = metrics.get('profit_margin', 0)
            
            # Must have reasonable margins
            if gross_margin <= 0 or profit_margin <= 0:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking criteria: {e}")
            return False
    
    def _create_qualified_company(self, metrics: Dict, sector: str) -> QualifiedCompany:
        """Create QualifiedCompany object from metrics"""
        
        # Calculate component scores (0-100)
        growth_score = self._calculate_growth_score(metrics)
        profitability_score = self._calculate_profitability_score(metrics)
        debt_score = self._calculate_debt_score(metrics)
        
        # Overall score (weighted average)
        overall_score = (growth_score * 0.4 + profitability_score * 0.4 + debt_score * 0.2)
        
        # Generate commentary if LLM available
        commentary = ""
        if self.llm:
            commentary = self._generate_commentary(metrics)
        
        qualified_company = QualifiedCompany(
            ticker=metrics.get('ticker', ''),
            company_name=metrics.get('company_name', 'N/A'),
            sector=metrics.get('sector', sector),
            market_cap=metrics.get('market_cap', 0),
            
            # Growth metrics
            revenue_growth_5y=metrics.get('revenue_growth_5y', 0),
            net_income_growth_5y=metrics.get('net_income_growth_5y', 0),
            cash_flow_growth_5y=metrics.get('cash_flow_growth_5y', 0),
            
            # Profitability metrics
            roe_ttm=metrics.get('roe_ttm', 0),
            roic_ttm=metrics.get('roic_ttm', 0),
            gross_margin=metrics.get('gross_margin', 0),
            profit_margin=metrics.get('profit_margin', 0),
            
            # Debt metrics
            current_ratio=metrics.get('current_ratio', 0),
            debt_to_ebitda=metrics.get('debt_to_ebitda', 0),
            debt_service_ratio=metrics.get('debt_service_ratio', 0),
            
            # Scores
            growth_score=growth_score,
            profitability_score=profitability_score,
            debt_score=debt_score,
            overall_score=overall_score,
            
            commentary=commentary,
            timestamp=datetime.now().isoformat()
        )
        
        return qualified_company
    
    def _calculate_growth_score(self, metrics: Dict) -> float:
        """Calculate growth quality score (0-100)"""
        score = 0
        
        # Revenue growth (0-40 points)
        revenue_growth = metrics.get('revenue_growth_5y', 0)
        if revenue_growth > 20:
            score += 40
        elif revenue_growth > 10:
            score += 30
        elif revenue_growth > 5:
            score += 20
        elif revenue_growth > 0:
            score += 10
        
        # Income growth (0-30 points)
        income_growth = metrics.get('net_income_growth_5y', 0)
        if income_growth > 20:
            score += 30
        elif income_growth > 10:
            score += 20
        elif income_growth > 5:
            score += 15
        elif income_growth > 0:
            score += 10
        
        # Cash flow growth (0-30 points)
        cf_growth = metrics.get('cash_flow_growth_5y', 0)
        if cf_growth > 20:
            score += 30
        elif cf_growth > 10:
            score += 20
        elif cf_growth > 5:
            score += 15
        elif cf_growth > 0:
            score += 10
        
        return min(score, 100)
    
    def _calculate_profitability_score(self, metrics: Dict) -> float:
        """Calculate profitability quality score (0-100)"""
        score = 0
        
        # ROE (0-40 points)
        roe = metrics.get('roe_ttm', 0)
        if roe > 25:
            score += 40
        elif roe > 20:
            score += 35
        elif roe > 15:
            score += 30
        elif roe > 12:
            score += 20
        
        # ROIC (0-40 points)
        roic = metrics.get('roic_ttm', 0)
        if roic > 25:
            score += 40
        elif roic > 20:
            score += 35
        elif roic > 15:
            score += 30
        elif roic > 12:
            score += 20
        
        # Margins (0-20 points)
        gross_margin = metrics.get('gross_margin', 0)
        profit_margin = metrics.get('profit_margin', 0)
        
        if gross_margin > 50 and profit_margin > 20:
            score += 20
        elif gross_margin > 30 and profit_margin > 10:
            score += 15
        elif gross_margin > 20 and profit_margin > 5:
            score += 10
        elif gross_margin > 0 and profit_margin > 0:
            score += 5
        
        return min(score, 100)
    
    def _calculate_debt_score(self, metrics: Dict) -> float:
        """Calculate debt health score (0-100)"""
        score = 0
        
        # Current ratio (0-40 points)
        current_ratio = metrics.get('current_ratio', 0)
        if current_ratio > 2.0:
            score += 40
        elif current_ratio > 1.5:
            score += 30
        elif current_ratio > 1.2:
            score += 20
        elif current_ratio > 1.0:
            score += 10
        
        # Debt to EBITDA (0-40 points)
        debt_to_ebitda = metrics.get('debt_to_ebitda', float('inf'))
        if debt_to_ebitda < 0.5:
            score += 40
        elif debt_to_ebitda < 1.0:
            score += 35
        elif debt_to_ebitda < 2.0:
            score += 25
        elif debt_to_ebitda < 3.0:
            score += 15
        
        # Debt service (0-20 points)
        debt_service = metrics.get('debt_service_ratio', 0)
        if debt_service > 100:  # Very strong coverage
            score += 20
        elif debt_service > 50:
            score += 15
        elif debt_service > 30:
            score += 10
        elif debt_service > 0:
            score += 5
        
        return min(score, 100)
    
    def _generate_commentary(self, metrics: Dict) -> str:
        """Generate LLM commentary on the qualified company"""
        try:
            if not self.llm:
                return ""
            
            prompt = f"""
Provide brief commentary on this qualified company's financial strength:

Company: {metrics.get('company_name', 'N/A')}
Sector: {metrics.get('sector', 'N/A')}

Key Metrics:
- Revenue Growth (5Y): {metrics.get('revenue_growth_5y', 0):.1f}%
- ROE: {metrics.get('roe_ttm', 0):.1f}%
- ROIC: {metrics.get('roic_ttm', 0):.1f}%
- Current Ratio: {metrics.get('current_ratio', 0):.2f}
- Debt/EBITDA: {metrics.get('debt_to_ebitda', 0):.2f}

Provide 2-3 sentences highlighting the key financial strengths that make this a qualified candidate.
"""
            
            messages = [
                SystemMessage(content="You are a financial analyst providing brief commentary on qualified companies."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            return response.content.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating commentary: {e}")
            return ""
    
    def _save_screening_results(self, companies: List[QualifiedCompany], sector: str, market: str):
        """Save screening results to tracing folder"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"tracing/fundamental_screening_{sector}_{market}_{timestamp}.json"
            
            results = {
                'sector': sector,
                'market': market,
                'timestamp': timestamp,
                'total_qualified': len(companies),
                'companies': [asdict(company) for company in companies]
            }
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
                
            self.logger.info(f"Saved screening results to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving screening results: {e}")
    
    def get_screening_summary(self, companies: List[QualifiedCompany]) -> Dict:
        """Get summary statistics for screening results"""
        if not companies:
            return {}
        
        return {
            'total_qualified': len(companies),
            'avg_overall_score': sum(c.overall_score for c in companies) / len(companies),
            'avg_growth_score': sum(c.growth_score for c in companies) / len(companies),
            'avg_profitability_score': sum(c.profitability_score for c in companies) / len(companies),
            'avg_debt_score': sum(c.debt_score for c in companies) / len(companies),
            'top_performers': [c.ticker for c in companies[:5]],
            'sectors_represented': list(set(c.sector for c in companies)),
            'market_cap_range': {
                'min': min(c.market_cap for c in companies),
                'max': max(c.market_cap for c in companies),
                'avg': sum(c.market_cap for c in companies) / len(companies)
            }
        }

