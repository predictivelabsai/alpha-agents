"""
Ranker Agent - Lohusalu Capital Management
Final scoring and ranking system combining fundamental and qualitative analysis
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_anthropic import ChatAnthropic
# from langchain_mistralai import ChatMistralAI

@dataclass
class InvestmentScore:
    """Data class for individual investment scoring"""
    ticker: str
    company_name: str
    sector: str
    
    # Component scores
    fundamental_score: float
    qualitative_score: float
    
    # Detailed scoring breakdown
    growth_score: float
    profitability_score: float
    financial_strength_score: float
    valuation_score: float
    moat_score: float
    sentiment_score: float
    trends_score: float
    competitive_score: float
    
    # Final composite score
    composite_score: float
    investment_grade: str  # A+, A, A-, B+, B, B-, C+, C, C-, D
    
    # Investment thesis
    investment_thesis: str
    key_strengths: List[str]
    key_risks: List[str]
    catalysts: List[str]
    time_horizon: str
    
    # Quantitative metrics
    upside_potential: float
    risk_rating: str  # Low, Medium, High
    conviction_level: str  # High, Medium, Low
    
    reasoning: str

@dataclass
class PortfolioRecommendation:
    """Data class for portfolio-level recommendations"""
    recommended_stocks: List[InvestmentScore]
    portfolio_composition: Dict[str, float]  # sector allocations
    risk_profile: str
    expected_return: float
    portfolio_thesis: str
    diversification_score: float
    overall_conviction: str

class RankerAgent:
    """
    Ranker Agent for final scoring and investment recommendations
    
    Key Functions:
    1. Combine fundamental and qualitative scores
    2. Apply sophisticated scoring methodology
    3. Generate investment grades and rankings
    4. Provide detailed investment thesis for each stock
    5. Create portfolio-level recommendations
    6. Score prospects across multiple dimensions
    """
    
    def __init__(self, model_provider: str = "openai", model_name: str = "gpt-4"):
        self.logger = logging.getLogger(__name__)
        self.model_provider = model_provider
        self.model_name = model_name
        self.llm = self._initialize_llm()
        
        # Scoring methodology weights
        self.scoring_weights = {
            'fundamental': 0.60,  # 60% weight on fundamental analysis
            'qualitative': 0.40   # 40% weight on qualitative analysis
        }
        
        # Detailed component weights (within fundamental and qualitative)
        self.fundamental_weights = {
            'growth': 0.30,
            'profitability': 0.25,
            'financial_strength': 0.25,
            'valuation': 0.20
        }
        
        self.qualitative_weights = {
            'moat': 0.35,
            'sentiment': 0.25,
            'trends': 0.25,
            'competitive': 0.15
        }
        
        # Investment grade thresholds
        self.grade_thresholds = {
            'A+': 90, 'A': 85, 'A-': 80,
            'B+': 75, 'B': 70, 'B-': 65,
            'C+': 60, 'C': 55, 'C-': 50,
            'D': 0
        }
        
        # Risk assessment criteria
        self.risk_criteria = {
            'low': {'debt_to_equity': 0.3, 'current_ratio': 2.0, 'volatility': 15},
            'medium': {'debt_to_equity': 0.6, 'current_ratio': 1.5, 'volatility': 25},
            'high': {'debt_to_equity': 1.0, 'current_ratio': 1.0, 'volatility': 35}
        }
    
    def _initialize_llm(self):
        """Initialize the language model based on provider"""
        try:
            if self.model_provider.lower() == "openai":
                return ChatOpenAI(model=self.model_name, temperature=0.1)
            elif self.model_provider.lower() == "google":
                return ChatGoogleGenerativeAI(model=self.model_name, temperature=0.1)
            elif self.model_provider.lower() == "anthropic":
                return ChatAnthropic(model=self.model_name, temperature=0.1)
            elif self.model_provider.lower() == "mistral":
                return ChatMistralAI(model=self.model_name, temperature=0.1)
            else:
                return ChatOpenAI(model="gpt-4", temperature=0.1)
        except Exception as e:
            self.logger.error(f"Error initializing LLM: {e}")
            return ChatOpenAI(model="gpt-4", temperature=0.1)
    
    def calculate_composite_score(self, fundamental_data: Dict, qualitative_data: Dict) -> Tuple[float, Dict[str, float]]:
        """
        Calculate composite investment score from fundamental and qualitative data
        
        Args:
            fundamental_data: Output from Fundamental Agent
            qualitative_data: Output from Rationale Agent
            
        Returns:
            Tuple of (composite_score, component_scores)
        """
        try:
            # Extract fundamental component scores
            fundamental_score = fundamental_data.get('fundamental_score', 50)
            
            # Break down fundamental score into components
            metrics = fundamental_data.get('metrics', {})
            growth_score = self._calculate_growth_score(metrics)
            profitability_score = self._calculate_profitability_score(metrics)
            financial_strength_score = self._calculate_financial_strength_score(metrics)
            valuation_score = self._calculate_valuation_score(metrics, fundamental_data)
            
            # Extract qualitative component scores
            qualitative_score = qualitative_data.get('qualitative_score', 50)
            moat_score = qualitative_data.get('moat_analysis', {}).get('moat_score', 50)
            sentiment_score = qualitative_data.get('sentiment_analysis', {}).get('sentiment_score', 50)
            trends_score = qualitative_data.get('secular_trends', {}).get('trend_score', 50)
            competitive_score = qualitative_data.get('competitive_position', {}).get('competitive_score', 50)
            
            # Calculate weighted fundamental score
            weighted_fundamental = (
                growth_score * self.fundamental_weights['growth'] +
                profitability_score * self.fundamental_weights['profitability'] +
                financial_strength_score * self.fundamental_weights['financial_strength'] +
                valuation_score * self.fundamental_weights['valuation']
            )
            
            # Calculate weighted qualitative score
            weighted_qualitative = (
                moat_score * self.qualitative_weights['moat'] +
                sentiment_score * self.qualitative_weights['sentiment'] +
                trends_score * self.qualitative_weights['trends'] +
                competitive_score * self.qualitative_weights['competitive']
            )
            
            # Calculate final composite score
            composite_score = (
                weighted_fundamental * self.scoring_weights['fundamental'] +
                weighted_qualitative * self.scoring_weights['qualitative']
            )
            
            component_scores = {
                'fundamental_score': weighted_fundamental,
                'qualitative_score': weighted_qualitative,
                'growth_score': growth_score,
                'profitability_score': profitability_score,
                'financial_strength_score': financial_strength_score,
                'valuation_score': valuation_score,
                'moat_score': moat_score,
                'sentiment_score': sentiment_score,
                'trends_score': trends_score,
                'competitive_score': competitive_score
            }
            
            return round(composite_score, 1), component_scores
            
        except Exception as e:
            self.logger.error(f"Error calculating composite score: {e}")
            return 50.0, {}
    
    def _calculate_growth_score(self, metrics: Dict) -> float:
        """Calculate growth component score"""
        try:
            revenue_growth = metrics.get('revenue_growth', 0)
            net_income_growth = metrics.get('net_income_growth', 0)
            
            # Score based on growth rates
            growth_score = 0
            
            # Revenue growth scoring
            if revenue_growth > 25:
                growth_score += 50
            elif revenue_growth > 15:
                growth_score += 40
            elif revenue_growth > 10:
                growth_score += 30
            elif revenue_growth > 5:
                growth_score += 20
            elif revenue_growth > 0:
                growth_score += 10
            
            # Net income growth scoring
            if net_income_growth > 25:
                growth_score += 50
            elif net_income_growth > 15:
                growth_score += 40
            elif net_income_growth > 10:
                growth_score += 30
            elif net_income_growth > 5:
                growth_score += 20
            elif net_income_growth > 0:
                growth_score += 10
            
            return min(growth_score, 100)
            
        except Exception:
            return 50
    
    def _calculate_profitability_score(self, metrics: Dict) -> float:
        """Calculate profitability component score"""
        try:
            roe = metrics.get('roe', 0)
            roic = metrics.get('roic', 0)
            profit_margin = metrics.get('profit_margin', 0)
            gross_margin = metrics.get('gross_margin', 0)
            
            profitability_score = 0
            
            # ROE scoring
            if roe > 20:
                profitability_score += 25
            elif roe > 15:
                profitability_score += 20
            elif roe > 10:
                profitability_score += 15
            elif roe > 5:
                profitability_score += 10
            
            # ROIC scoring
            if roic > 15:
                profitability_score += 25
            elif roic > 12:
                profitability_score += 20
            elif roic > 8:
                profitability_score += 15
            elif roic > 5:
                profitability_score += 10
            
            # Profit margin scoring
            if profit_margin > 15:
                profitability_score += 25
            elif profit_margin > 10:
                profitability_score += 20
            elif profit_margin > 5:
                profitability_score += 15
            elif profit_margin > 2:
                profitability_score += 10
            
            # Gross margin scoring
            if gross_margin > 60:
                profitability_score += 25
            elif gross_margin > 40:
                profitability_score += 20
            elif gross_margin > 25:
                profitability_score += 15
            elif gross_margin > 15:
                profitability_score += 10
            
            return min(profitability_score, 100)
            
        except Exception:
            return 50
    
    def _calculate_financial_strength_score(self, metrics: Dict) -> float:
        """Calculate financial strength component score"""
        try:
            current_ratio = metrics.get('current_ratio', 0)
            debt_to_equity = metrics.get('debt_to_equity', 1)
            
            financial_score = 0
            
            # Current ratio scoring
            if current_ratio > 3:
                financial_score += 50
            elif current_ratio > 2:
                financial_score += 40
            elif current_ratio > 1.5:
                financial_score += 30
            elif current_ratio > 1.2:
                financial_score += 20
            elif current_ratio > 1.0:
                financial_score += 10
            
            # Debt-to-equity scoring (lower is better)
            if debt_to_equity < 0.2:
                financial_score += 50
            elif debt_to_equity < 0.4:
                financial_score += 40
            elif debt_to_equity < 0.6:
                financial_score += 30
            elif debt_to_equity < 0.8:
                financial_score += 20
            elif debt_to_equity < 1.0:
                financial_score += 10
            
            return min(financial_score, 100)
            
        except Exception:
            return 50
    
    def _calculate_valuation_score(self, metrics: Dict, fundamental_data: Dict) -> float:
        """Calculate valuation component score"""
        try:
            pe_ratio = metrics.get('pe_ratio', 25)
            pb_ratio = metrics.get('pb_ratio', 3)
            upside_potential = fundamental_data.get('upside_potential', 0)
            
            valuation_score = 0
            
            # P/E ratio scoring (lower is generally better, but consider growth)
            if pe_ratio < 12:
                valuation_score += 30
            elif pe_ratio < 18:
                valuation_score += 25
            elif pe_ratio < 25:
                valuation_score += 20
            elif pe_ratio < 35:
                valuation_score += 15
            elif pe_ratio < 50:
                valuation_score += 10
            
            # P/B ratio scoring
            if pb_ratio < 1.5:
                valuation_score += 25
            elif pb_ratio < 2.5:
                valuation_score += 20
            elif pb_ratio < 4:
                valuation_score += 15
            elif pb_ratio < 6:
                valuation_score += 10
            
            # Upside potential scoring
            if upside_potential > 50:
                valuation_score += 45
            elif upside_potential > 30:
                valuation_score += 35
            elif upside_potential > 15:
                valuation_score += 25
            elif upside_potential > 5:
                valuation_score += 15
            elif upside_potential > 0:
                valuation_score += 10
            
            return min(valuation_score, 100)
            
        except Exception:
            return 50
    
    def assign_investment_grade(self, composite_score: float) -> str:
        """Assign investment grade based on composite score"""
        for grade, threshold in self.grade_thresholds.items():
            if composite_score >= threshold:
                return grade
        return 'D'
    
    def assess_risk_rating(self, fundamental_data: Dict, qualitative_data: Dict) -> str:
        """Assess risk rating based on multiple factors"""
        try:
            metrics = fundamental_data.get('metrics', {})
            
            # Financial risk factors
            debt_to_equity = metrics.get('debt_to_equity', 0.5)
            current_ratio = metrics.get('current_ratio', 1.5)
            
            # Market risk factors (simplified)
            market_cap = fundamental_data.get('market_cap', 1e9)
            
            # Qualitative risk factors
            moat_strength = qualitative_data.get('moat_analysis', {}).get('moat_strength', 'None')
            sentiment = qualitative_data.get('sentiment_analysis', {}).get('overall_sentiment', 'Neutral')
            
            risk_score = 0
            
            # Financial strength impact on risk
            if debt_to_equity < 0.3 and current_ratio > 2.0:
                risk_score += 30  # Lower risk
            elif debt_to_equity < 0.6 and current_ratio > 1.5:
                risk_score += 20
            else:
                risk_score += 10  # Higher risk
            
            # Market cap impact (smaller = riskier)
            if market_cap > 10e9:
                risk_score += 25
            elif market_cap > 2e9:
                risk_score += 20
            elif market_cap > 500e6:
                risk_score += 15
            else:
                risk_score += 10
            
            # Moat strength impact
            if moat_strength == 'Wide':
                risk_score += 25
            elif moat_strength == 'Narrow':
                risk_score += 15
            else:
                risk_score += 5
            
            # Sentiment impact
            if sentiment == 'Positive':
                risk_score += 20
            elif sentiment == 'Neutral':
                risk_score += 15
            else:
                risk_score += 5
            
            # Determine risk rating
            if risk_score >= 75:
                return 'Low'
            elif risk_score >= 50:
                return 'Medium'
            else:
                return 'High'
                
        except Exception as e:
            self.logger.error(f"Error assessing risk rating: {e}")
            return 'Medium'
    
    def determine_conviction_level(self, composite_score: float, component_scores: Dict) -> str:
        """Determine conviction level based on score consistency"""
        try:
            # Check score consistency across components
            scores = [
                component_scores.get('growth_score', 50),
                component_scores.get('profitability_score', 50),
                component_scores.get('financial_strength_score', 50),
                component_scores.get('valuation_score', 50),
                component_scores.get('moat_score', 50),
                component_scores.get('sentiment_score', 50),
                component_scores.get('trends_score', 50),
                component_scores.get('competitive_score', 50)
            ]
            
            # Calculate standard deviation of scores
            score_std = np.std(scores)
            score_mean = np.mean(scores)
            
            # High conviction: high average score with low deviation
            if score_mean >= 75 and score_std <= 10:
                return 'High'
            elif score_mean >= 65 and score_std <= 15:
                return 'Medium'
            elif score_mean >= 55 and score_std <= 20:
                return 'Medium'
            else:
                return 'Low'
                
        except Exception:
            return 'Medium'
    
    def generate_investment_thesis(self, ticker: str, company_name: str, 
                                 fundamental_data: Dict, qualitative_data: Dict,
                                 composite_score: float, component_scores: Dict) -> str:
        """Generate comprehensive investment thesis using LLM"""
        try:
            prompt = f"""
            Generate a comprehensive investment thesis for {company_name} ({ticker}) based on the following analysis:

            FUNDAMENTAL ANALYSIS:
            - Overall Fundamental Score: {fundamental_data.get('fundamental_score', 0):.1f}/100
            - Growth Score: {component_scores.get('growth_score', 0):.1f}/100
            - Profitability Score: {component_scores.get('profitability_score', 0):.1f}/100
            - Financial Strength Score: {component_scores.get('financial_strength_score', 0):.1f}/100
            - Valuation Score: {component_scores.get('valuation_score', 0):.1f}/100
            - Upside Potential: {fundamental_data.get('upside_potential', 0):.1f}%

            QUALITATIVE ANALYSIS:
            - Overall Qualitative Score: {qualitative_data.get('qualitative_score', 0):.1f}/100
            - Economic Moat: {qualitative_data.get('moat_analysis', {}).get('moat_strength', 'Unknown')} ({component_scores.get('moat_score', 0):.1f}/100)
            - Sentiment: {qualitative_data.get('sentiment_analysis', {}).get('overall_sentiment', 'Unknown')} ({component_scores.get('sentiment_score', 0):.1f}/100)
            - Secular Trends: {qualitative_data.get('secular_trends', {}).get('trend_alignment', 'Unknown')} ({component_scores.get('trends_score', 0):.1f}/100)
            - Competitive Position: {qualitative_data.get('competitive_position', {}).get('market_position', 'Unknown')} ({component_scores.get('competitive_score', 0):.1f}/100)

            COMPOSITE SCORE: {composite_score:.1f}/100

            Provide a comprehensive investment thesis that includes:

            1. **Investment Strengths** (2-3 key strengths)
            2. **Investment Risks** (2-3 main risks)
            3. **Key Catalysts** (factors that could drive outperformance)
            4. **Investment Recommendation** (clear conclusion based on analysis)

            Write a professional, balanced analysis suitable for investment decision-making. Focus on the most material factors and provide specific reasoning based on the quantitative and qualitative analysis.
            """
            
            messages = [
                SystemMessage(content="You are an expert investment analyst providing comprehensive investment thesis and recommendations."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            self.logger.error(f"Error generating investment thesis: {e}")
            return f"Investment thesis for {company_name} with composite score {composite_score:.1f}/100."
    
    def extract_key_factors(self, investment_thesis: str) -> Tuple[List[str], List[str], List[str]]:
        """Extract key strengths, risks, and catalysts from investment thesis"""
        try:
            prompt = f"""
            Extract key investment factors from the following investment thesis:

            {investment_thesis}

            Provide the output in JSON format:
            {{
                "key_strengths": ["<strength1>", "<strength2>", "<strength3>"],
                "key_risks": ["<risk1>", "<risk2>", "<risk3>"],
                "catalysts": ["<catalyst1>", "<catalyst2>", "<catalyst3>"]
            }}

            Focus on the most important factors mentioned in the thesis.
            """
            
            messages = [
                SystemMessage(content="You are an expert at extracting key investment factors from analysis."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            try:
                factors = json.loads(response.content)
                return (
                    factors.get('key_strengths', []),
                    factors.get('key_risks', []),
                    factors.get('catalysts', [])
                )
            except json.JSONDecodeError:
                # Fallback extraction
                return (
                    ["Strong fundamentals", "Competitive advantages"],
                    ["Market risks", "Execution risks"],
                    ["Growth catalysts", "Market expansion"]
                )
                
        except Exception as e:
            self.logger.error(f"Error extracting key factors: {e}")
            return ([], [], [])
    
    def determine_time_horizon(self, qualitative_data: Dict, composite_score: float) -> str:
        """Determine appropriate investment time horizon"""
        try:
            # Consider secular trends time horizon
            trends_horizon = qualitative_data.get('secular_trends', {}).get('time_horizon', 'Medium-term')
            moat_strength = qualitative_data.get('moat_analysis', {}).get('moat_strength', 'None')
            
            # Strong moats and long-term trends suggest longer horizons
            if moat_strength == 'Wide' and 'Long-term' in trends_horizon:
                return 'Long-term (5+ years)'
            elif moat_strength in ['Wide', 'Narrow'] and composite_score >= 70:
                return 'Medium-term (2-5 years)'
            elif composite_score >= 60:
                return 'Medium-term (2-5 years)'
            else:
                return 'Short-term (1-2 years)'
                
        except Exception:
            return 'Medium-term (2-5 years)'
    
    def score_investment(self, ticker: str, company_name: str, sector: str,
                        fundamental_data: Dict, qualitative_data: Dict) -> InvestmentScore:
        """
        Score a single investment opportunity
        
        Args:
            ticker: Stock ticker symbol
            company_name: Full company name
            sector: Company sector
            fundamental_data: Output from Fundamental Agent
            qualitative_data: Output from Rationale Agent
            
        Returns:
            InvestmentScore object with complete scoring analysis
        """
        self.logger.info(f"Scoring investment for {ticker}")
        
        try:
            # Calculate composite score and component breakdown
            composite_score, component_scores = self.calculate_composite_score(
                fundamental_data, qualitative_data
            )
            
            # Assign investment grade
            investment_grade = self.assign_investment_grade(composite_score)
            
            # Assess risk and conviction
            risk_rating = self.assess_risk_rating(fundamental_data, qualitative_data)
            conviction_level = self.determine_conviction_level(composite_score, component_scores)
            
            # Generate investment thesis
            investment_thesis = self.generate_investment_thesis(
                ticker, company_name, fundamental_data, qualitative_data,
                composite_score, component_scores
            )
            
            # Extract key factors
            key_strengths, key_risks, catalysts = self.extract_key_factors(investment_thesis)
            
            # Determine time horizon
            time_horizon = self.determine_time_horizon(qualitative_data, composite_score)
            
            # Generate reasoning
            reasoning = self._generate_scoring_reasoning(
                ticker, composite_score, component_scores, investment_grade
            )
            
            return InvestmentScore(
                ticker=ticker,
                company_name=company_name,
                sector=sector,
                fundamental_score=component_scores.get('fundamental_score', 50),
                qualitative_score=component_scores.get('qualitative_score', 50),
                growth_score=component_scores.get('growth_score', 50),
                profitability_score=component_scores.get('profitability_score', 50),
                financial_strength_score=component_scores.get('financial_strength_score', 50),
                valuation_score=component_scores.get('valuation_score', 50),
                moat_score=component_scores.get('moat_score', 50),
                sentiment_score=component_scores.get('sentiment_score', 50),
                trends_score=component_scores.get('trends_score', 50),
                competitive_score=component_scores.get('competitive_score', 50),
                composite_score=composite_score,
                investment_grade=investment_grade,
                investment_thesis=investment_thesis,
                key_strengths=key_strengths,
                key_risks=key_risks,
                catalysts=catalysts,
                time_horizon=time_horizon,
                upside_potential=fundamental_data.get('upside_potential', 0),
                risk_rating=risk_rating,
                conviction_level=conviction_level,
                reasoning=reasoning
            )
            
        except Exception as e:
            self.logger.error(f"Error scoring investment for {ticker}: {e}")
            # Return error score
            return InvestmentScore(
                ticker=ticker,
                company_name=company_name,
                sector=sector,
                fundamental_score=0,
                qualitative_score=0,
                growth_score=0,
                profitability_score=0,
                financial_strength_score=0,
                valuation_score=0,
                moat_score=0,
                sentiment_score=0,
                trends_score=0,
                competitive_score=0,
                composite_score=0,
                investment_grade='D',
                investment_thesis=f"Error in analysis for {company_name}",
                key_strengths=[],
                key_risks=[],
                catalysts=[],
                time_horizon='Unknown',
                upside_potential=0,
                risk_rating='High',
                conviction_level='Low',
                reasoning=f"Error in scoring: {str(e)}"
            )
    
    def _generate_scoring_reasoning(self, ticker: str, composite_score: float,
                                  component_scores: Dict, investment_grade: str) -> str:
        """Generate reasoning for scoring methodology"""
        try:
            fundamental_score = component_scores.get('fundamental_score', 50)
            qualitative_score = component_scores.get('qualitative_score', 50)
            
            reasoning = f"""
            {ticker} receives a composite score of {composite_score:.1f}/100 (Grade: {investment_grade}) based on:
            
            Fundamental Analysis ({fundamental_score:.1f}/100):
            - Growth: {component_scores.get('growth_score', 0):.1f}/100
            - Profitability: {component_scores.get('profitability_score', 0):.1f}/100
            - Financial Strength: {component_scores.get('financial_strength_score', 0):.1f}/100
            - Valuation: {component_scores.get('valuation_score', 0):.1f}/100
            
            Qualitative Analysis ({qualitative_score:.1f}/100):
            - Economic Moat: {component_scores.get('moat_score', 0):.1f}/100
            - Market Sentiment: {component_scores.get('sentiment_score', 0):.1f}/100
            - Secular Trends: {component_scores.get('trends_score', 0):.1f}/100
            - Competitive Position: {component_scores.get('competitive_score', 0):.1f}/100
            
            The scoring methodology weights fundamental analysis at 60% and qualitative analysis at 40%.
            """
            
            return reasoning.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating scoring reasoning: {e}")
            return f"Scoring analysis for {ticker} with composite score {composite_score:.1f}/100."
    
    def rank_investments(self, investment_scores: List[InvestmentScore]) -> List[InvestmentScore]:
        """Rank investments by composite score and other factors"""
        try:
            # Sort by composite score (primary), then by conviction level and upside potential
            def sort_key(score: InvestmentScore):
                conviction_weight = {'High': 3, 'Medium': 2, 'Low': 1}.get(score.conviction_level, 1)
                return (score.composite_score, conviction_weight, score.upside_potential)
            
            ranked_scores = sorted(investment_scores, key=sort_key, reverse=True)
            
            self.logger.info(f"Ranked {len(ranked_scores)} investments")
            return ranked_scores
            
        except Exception as e:
            self.logger.error(f"Error ranking investments: {e}")
            return investment_scores
    
    def create_portfolio_recommendation(self, investment_scores: List[InvestmentScore],
                                      max_positions: int = 10) -> PortfolioRecommendation:
        """Create portfolio-level recommendation from individual scores"""
        try:
            # Select top investments
            top_investments = investment_scores[:max_positions]
            
            # Calculate sector allocation
            sector_allocation = {}
            for investment in top_investments:
                sector = investment.sector
                sector_allocation[sector] = sector_allocation.get(sector, 0) + 1
            
            # Convert to percentages
            total_positions = len(top_investments)
            portfolio_composition = {
                sector: (count / total_positions) * 100 
                for sector, count in sector_allocation.items()
            }
            
            # Calculate portfolio metrics
            avg_score = np.mean([inv.composite_score for inv in top_investments])
            avg_upside = np.mean([inv.upside_potential for inv in top_investments])
            
            # Assess portfolio risk
            risk_counts = {'Low': 0, 'Medium': 0, 'High': 0}
            for investment in top_investments:
                risk_counts[investment.risk_rating] += 1
            
            if risk_counts['High'] > risk_counts['Low'] + risk_counts['Medium']:
                portfolio_risk = 'High'
            elif risk_counts['Low'] > risk_counts['Medium'] + risk_counts['High']:
                portfolio_risk = 'Low'
            else:
                portfolio_risk = 'Medium'
            
            # Calculate diversification score
            diversification_score = min(len(sector_allocation) * 20, 100)  # More sectors = better diversification
            
            # Determine overall conviction
            high_conviction_count = sum(1 for inv in top_investments if inv.conviction_level == 'High')
            if high_conviction_count >= len(top_investments) * 0.6:
                overall_conviction = 'High'
            elif high_conviction_count >= len(top_investments) * 0.3:
                overall_conviction = 'Medium'
            else:
                overall_conviction = 'Low'
            
            # Generate portfolio thesis
            portfolio_thesis = self._generate_portfolio_thesis(
                top_investments, portfolio_composition, avg_score, portfolio_risk
            )
            
            return PortfolioRecommendation(
                recommended_stocks=top_investments,
                portfolio_composition=portfolio_composition,
                risk_profile=portfolio_risk,
                expected_return=avg_upside,
                portfolio_thesis=portfolio_thesis,
                diversification_score=diversification_score,
                overall_conviction=overall_conviction
            )
            
        except Exception as e:
            self.logger.error(f"Error creating portfolio recommendation: {e}")
            return PortfolioRecommendation(
                recommended_stocks=[],
                portfolio_composition={},
                risk_profile='Unknown',
                expected_return=0,
                portfolio_thesis='Error in portfolio analysis',
                diversification_score=0,
                overall_conviction='Low'
            )
    
    def _generate_portfolio_thesis(self, investments: List[InvestmentScore],
                                 composition: Dict[str, float], avg_score: float,
                                 risk_profile: str) -> str:
        """Generate portfolio-level investment thesis"""
        try:
            top_sectors = sorted(composition.items(), key=lambda x: x[1], reverse=True)[:3]
            
            thesis = f"""
            Portfolio Recommendation (Average Score: {avg_score:.1f}/100, Risk: {risk_profile})
            
            This portfolio consists of {len(investments)} carefully selected investments with strong fundamental and qualitative characteristics. 
            
            Sector Allocation: The portfolio is diversified across {len(composition)} sectors, with primary exposure to {top_sectors[0][0]} ({top_sectors[0][1]:.1f}%), 
            {top_sectors[1][0] if len(top_sectors) > 1 else 'N/A'} ({top_sectors[1][1]:.1f if len(top_sectors) > 1 else 0}%), 
            and {top_sectors[2][0] if len(top_sectors) > 2 else 'N/A'} ({top_sectors[2][1]:.1f if len(top_sectors) > 2 else 0}%).
            
            The selected companies demonstrate strong competitive positions, attractive valuations, and alignment with positive secular trends.
            """
            
            return thesis.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating portfolio thesis: {e}")
            return f"Portfolio of {len(investments)} investments with average score {avg_score:.1f}/100."
    
    def save_analysis_trace(self, investment_scores: List[InvestmentScore],
                           portfolio_recommendation: PortfolioRecommendation,
                           output_dir: str = "tracing") -> str:
        """Save ranking analysis trace to JSON file"""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Create trace data
            trace_data = {
                'timestamp': datetime.now().isoformat(),
                'agent': 'RankerAgent',
                'model_provider': self.model_provider,
                'model_name': self.model_name,
                'scoring_methodology': {
                    'scoring_weights': self.scoring_weights,
                    'fundamental_weights': self.fundamental_weights,
                    'qualitative_weights': self.qualitative_weights,
                    'grade_thresholds': self.grade_thresholds
                },
                'investment_scores': [
                    {
                        'ticker': score.ticker,
                        'company_name': score.company_name,
                        'sector': score.sector,
                        'composite_score': score.composite_score,
                        'investment_grade': score.investment_grade,
                        'component_scores': {
                            'fundamental_score': score.fundamental_score,
                            'qualitative_score': score.qualitative_score,
                            'growth_score': score.growth_score,
                            'profitability_score': score.profitability_score,
                            'financial_strength_score': score.financial_strength_score,
                            'valuation_score': score.valuation_score,
                            'moat_score': score.moat_score,
                            'sentiment_score': score.sentiment_score,
                            'trends_score': score.trends_score,
                            'competitive_score': score.competitive_score
                        },
                        'investment_thesis': score.investment_thesis,
                        'key_strengths': score.key_strengths,
                        'key_risks': score.key_risks,
                        'catalysts': score.catalysts,
                        'time_horizon': score.time_horizon,
                        'upside_potential': score.upside_potential,
                        'risk_rating': score.risk_rating,
                        'conviction_level': score.conviction_level,
                        'reasoning': score.reasoning
                    } for score in investment_scores
                ],
                'portfolio_recommendation': {
                    'portfolio_composition': portfolio_recommendation.portfolio_composition,
                    'risk_profile': portfolio_recommendation.risk_profile,
                    'expected_return': portfolio_recommendation.expected_return,
                    'portfolio_thesis': portfolio_recommendation.portfolio_thesis,
                    'diversification_score': portfolio_recommendation.diversification_score,
                    'overall_conviction': portfolio_recommendation.overall_conviction
                }
            }
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ranker_agent_trace_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(trace_data, f, indent=2, default=str)
            
            self.logger.info(f"Analysis trace saved to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error saving analysis trace: {e}")
            return ""

if __name__ == "__main__":
    # Test the agent
    agent = RankerAgent()
    
    # Mock data for testing
    fundamental_data = {
        'fundamental_score': 75,
        'metrics': {
            'revenue_growth': 15,
            'roe': 18,
            'current_ratio': 2.1,
            'debt_to_equity': 0.3,
            'pe_ratio': 22
        },
        'upside_potential': 25,
        'market_cap': 5e9
    }
    
    qualitative_data = {
        'qualitative_score': 80,
        'moat_analysis': {'moat_strength': 'Wide', 'moat_score': 85},
        'sentiment_analysis': {'overall_sentiment': 'Positive', 'sentiment_score': 75},
        'secular_trends': {'trend_alignment': 'Strong', 'trend_score': 80},
        'competitive_position': {'market_position': 'Leader', 'competitive_score': 85}
    }
    
    score = agent.score_investment("AAPL", "Apple Inc.", "Technology", fundamental_data, qualitative_data)
    
    print(f"Investment Score for {score.ticker}:")
    print(f"Composite Score: {score.composite_score}/100")
    print(f"Investment Grade: {score.investment_grade}")
    print(f"Risk Rating: {score.risk_rating}")
    print(f"Conviction: {score.conviction_level}")
    print(f"Thesis: {score.investment_thesis[:200]}...")

