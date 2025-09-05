"""
Rationale Agent - Qualitative Analysis with Web Search
Analyzes business moats, sentiment, secular trends using Tavily search API
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from tavily import TavilyClient

from .fundamental_agent import QualifiedCompany


@dataclass
class RationaleAnalysis:
    """Data class for rationale analysis results"""
    ticker: str
    company_name: str
    
    # Moat Analysis
    moat_type: str
    moat_strength: str  # WIDE, NARROW, NONE
    moat_score: float  # 0-10
    
    # Sentiment Analysis
    sentiment_score: float  # 0-10
    sentiment_trend: str  # IMPROVING, STABLE, DECLINING
    news_sentiment: str
    
    # Secular Trends
    trend_alignment: str  # STRONG, MODERATE, WEAK
    trend_score: float  # 0-10
    industry_outlook: str
    
    # Overall Qualitative Score
    overall_qualitative_score: float  # 0-10
    
    # Supporting Evidence
    key_insights: List[str]
    citations: List[str]
    research_summary: str
    
    timestamp: str


class RationaleAgent:
    """Agent for qualitative analysis using web search and LLM reasoning"""
    
    def __init__(self, api_key: str = None, tavily_api_key: str = None):
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
        
        # Initialize Tavily client
        tavily_key = tavily_api_key or os.getenv('TAVILY_API_KEY')
        if tavily_key:
            try:
                self.tavily_client = TavilyClient(api_key=tavily_key)
            except Exception as e:
                self.logger.warning(f"Failed to initialize Tavily client: {e}")
                self.tavily_client = None
        else:
            self.logger.warning("No Tavily API key provided")
            self.tavily_client = None
    
    def analyze_companies(self, qualified_companies: List[QualifiedCompany]) -> Dict[str, RationaleAnalysis]:
        """Analyze qualified companies for qualitative factors"""
        try:
            self.logger.info(f"Starting rationale analysis for {len(qualified_companies)} companies")
            
            analyses = {}
            
            for company in qualified_companies:
                try:
                    analysis = self._analyze_individual_company(company)
                    analyses[company.ticker] = analysis
                    
                    self.logger.info(f"✓ Completed rationale analysis for {company.ticker}: {analysis.overall_qualitative_score:.1f}/10")
                    
                except Exception as e:
                    self.logger.error(f"Error analyzing {company.ticker}: {e}")
                    continue
            
            # Save analysis results
            self._save_rationale_results(analyses)
            
            self.logger.info(f"Completed rationale analysis for {len(analyses)} companies")
            
            return analyses
            
        except Exception as e:
            self.logger.error(f"Error in rationale analysis process: {e}")
            return {}
    
    def _analyze_individual_company(self, company: QualifiedCompany) -> RationaleAnalysis:
        """Perform comprehensive qualitative analysis for a single company"""
        
        # Gather web research data
        research_data = self._conduct_web_research(company)
        
        # Analyze competitive moat
        moat_analysis = self._analyze_competitive_moat(company, research_data)
        
        # Analyze sentiment and news
        sentiment_analysis = self._analyze_sentiment(company, research_data)
        
        # Analyze secular trends
        trend_analysis = self._analyze_secular_trends(company, research_data)
        
        # Generate overall qualitative assessment
        if self.llm:
            overall_assessment = self._generate_llm_assessment(company, research_data, moat_analysis, sentiment_analysis, trend_analysis)
        else:
            overall_assessment = self._generate_fallback_assessment(moat_analysis, sentiment_analysis, trend_analysis)
        
        # Calculate overall qualitative score
        overall_score = (moat_analysis['score'] * 0.4 + 
                        sentiment_analysis['score'] * 0.3 + 
                        trend_analysis['score'] * 0.3)
        
        analysis = RationaleAnalysis(
            ticker=company.ticker,
            company_name=company.company_name,
            
            # Moat analysis
            moat_type=moat_analysis['type'],
            moat_strength=moat_analysis['strength'],
            moat_score=moat_analysis['score'],
            
            # Sentiment analysis
            sentiment_score=sentiment_analysis['score'],
            sentiment_trend=sentiment_analysis['trend'],
            news_sentiment=sentiment_analysis['news_sentiment'],
            
            # Trend analysis
            trend_alignment=trend_analysis['alignment'],
            trend_score=trend_analysis['score'],
            industry_outlook=trend_analysis['outlook'],
            
            # Overall assessment
            overall_qualitative_score=overall_score,
            key_insights=overall_assessment['insights'],
            citations=research_data.get('citations', []),
            research_summary=overall_assessment['summary'],
            
            timestamp=datetime.now().isoformat()
        )
        
        return analysis
    
    def _conduct_web_research(self, company: QualifiedCompany) -> Dict:
        """Conduct comprehensive web research using Tavily"""
        research_data = {
            'company_info': {},
            'competitive_landscape': {},
            'recent_news': {},
            'industry_trends': {},
            'citations': []
        }
        
        if not self.tavily_client:
            self.logger.warning(f"No Tavily client available for {company.ticker}")
            return research_data
        
        try:
            # Research queries
            queries = [
                f"{company.company_name} competitive advantage moat business model",
                f"{company.company_name} recent news earnings financial performance",
                f"{company.sector} industry trends outlook 2024 2025",
                f"{company.company_name} market position competitors analysis",
                f"{company.ticker} analyst sentiment investor opinion"
            ]
            
            all_results = []
            
            for query in queries:
                try:
                    self.logger.info(f"Searching: {query}")
                    results = self.tavily_client.search(
                        query=query,
                        search_depth="advanced",
                        max_results=5,
                        include_domains=["reuters.com", "bloomberg.com", "wsj.com", "ft.com", "marketwatch.com", "seekingalpha.com", "fool.com"]
                    )
                    
                    if results and 'results' in results:
                        all_results.extend(results['results'])
                        
                        # Extract citations
                        for result in results['results']:
                            if 'url' in result:
                                research_data['citations'].append(result['url'])
                    
                except Exception as e:
                    self.logger.error(f"Error searching '{query}': {e}")
                    continue
            
            # Process and categorize results
            research_data = self._process_search_results(all_results, company)
            
        except Exception as e:
            self.logger.error(f"Error conducting web research for {company.ticker}: {e}")
        
        return research_data
    
    def _process_search_results(self, results: List[Dict], company: QualifiedCompany) -> Dict:
        """Process and categorize search results"""
        processed_data = {
            'company_info': [],
            'competitive_landscape': [],
            'recent_news': [],
            'industry_trends': [],
            'citations': []
        }
        
        for result in results:
            try:
                title = result.get('title', '').lower()
                content = result.get('content', '').lower()
                url = result.get('url', '')
                
                # Categorize based on content
                if any(keyword in title or keyword in content for keyword in ['moat', 'competitive advantage', 'business model']):
                    processed_data['competitive_landscape'].append({
                        'title': result.get('title', ''),
                        'content': result.get('content', ''),
                        'url': url
                    })
                elif any(keyword in title or keyword in content for keyword in ['earnings', 'financial', 'revenue', 'profit']):
                    processed_data['recent_news'].append({
                        'title': result.get('title', ''),
                        'content': result.get('content', ''),
                        'url': url
                    })
                elif any(keyword in title or keyword in content for keyword in ['industry', 'sector', 'trend', 'outlook']):
                    processed_data['industry_trends'].append({
                        'title': result.get('title', ''),
                        'content': result.get('content', ''),
                        'url': url
                    })
                else:
                    processed_data['company_info'].append({
                        'title': result.get('title', ''),
                        'content': result.get('content', ''),
                        'url': url
                    })
                
                processed_data['citations'].append(url)
                
            except Exception as e:
                self.logger.error(f"Error processing search result: {e}")
                continue
        
        return processed_data
    
    def _analyze_competitive_moat(self, company: QualifiedCompany, research_data: Dict) -> Dict:
        """Analyze competitive moat based on research data"""
        
        # Default moat characteristics by sector
        sector_moats = {
            'Technology': {'type': 'Network Effects', 'base_score': 7.0},
            'Healthcare': {'type': 'Regulatory', 'base_score': 6.5},
            'Consumer Cyclical': {'type': 'Brand', 'base_score': 5.5},
            'Consumer Defensive': {'type': 'Brand', 'base_score': 6.0},
            'Financial Services': {'type': 'Switching Costs', 'base_score': 5.0},
            'Industrial': {'type': 'Economies of Scale', 'base_score': 4.5},
            'Utilities': {'type': 'Regulatory', 'base_score': 7.5},
            'Energy': {'type': 'Cost Advantage', 'base_score': 4.0},
            'Materials': {'type': 'Cost Advantage', 'base_score': 4.0},
            'Real Estate': {'type': 'Location', 'base_score': 5.5}
        }
        
        sector_info = sector_moats.get(company.sector, {'type': 'Unknown', 'base_score': 5.0})
        moat_score = sector_info['base_score']
        
        # Adjust based on market cap (larger companies typically have stronger moats)
        if company.market_cap >= 100e9:  # Mega cap
            moat_score += 1.5
        elif company.market_cap >= 10e9:  # Large cap
            moat_score += 1.0
        elif company.market_cap >= 2e9:  # Mid cap
            moat_score += 0.5
        else:  # Small cap
            moat_score -= 0.5
        
        # Adjust based on research findings
        competitive_content = research_data.get('competitive_landscape', [])
        if competitive_content:
            # Look for moat indicators in research
            moat_indicators = ['competitive advantage', 'market leader', 'dominant position', 'barriers to entry', 'switching costs']
            positive_mentions = 0
            
            for item in competitive_content:
                content = (item.get('title', '') + ' ' + item.get('content', '')).lower()
                for indicator in moat_indicators:
                    if indicator in content:
                        positive_mentions += 1
            
            if positive_mentions >= 3:
                moat_score += 1.0
            elif positive_mentions >= 1:
                moat_score += 0.5
        
        # Determine moat strength
        if moat_score >= 8.0:
            strength = 'WIDE'
        elif moat_score >= 6.0:
            strength = 'NARROW'
        else:
            strength = 'NONE'
        
        moat_score = min(10.0, max(0.0, moat_score))
        
        return {
            'type': sector_info['type'],
            'strength': strength,
            'score': round(moat_score, 1)
        }
    
    def _analyze_sentiment(self, company: QualifiedCompany, research_data: Dict) -> Dict:
        """Analyze sentiment based on news and research"""
        
        base_sentiment = 5.0  # Neutral starting point
        
        # Analyze recent news sentiment
        news_items = research_data.get('recent_news', [])
        positive_keywords = ['growth', 'strong', 'beat', 'exceed', 'positive', 'bullish', 'upgrade', 'outperform']
        negative_keywords = ['decline', 'weak', 'miss', 'disappoint', 'negative', 'bearish', 'downgrade', 'underperform']
        
        sentiment_signals = 0
        total_signals = 0
        
        for item in news_items:
            content = (item.get('title', '') + ' ' + item.get('content', '')).lower()
            
            for keyword in positive_keywords:
                if keyword in content:
                    sentiment_signals += 1
                    total_signals += 1
            
            for keyword in negative_keywords:
                if keyword in content:
                    sentiment_signals -= 1
                    total_signals += 1
        
        # Adjust sentiment score based on signals
        if total_signals > 0:
            sentiment_adjustment = (sentiment_signals / total_signals) * 2.0
            base_sentiment += sentiment_adjustment
        
        # Determine sentiment trend
        if sentiment_signals > 2:
            trend = 'IMPROVING'
        elif sentiment_signals < -2:
            trend = 'DECLINING'
        else:
            trend = 'STABLE'
        
        # Generate news sentiment summary
        if base_sentiment >= 7.0:
            news_sentiment = 'Positive'
        elif base_sentiment >= 4.0:
            news_sentiment = 'Neutral'
        else:
            news_sentiment = 'Negative'
        
        sentiment_score = min(10.0, max(0.0, base_sentiment))
        
        return {
            'score': round(sentiment_score, 1),
            'trend': trend,
            'news_sentiment': news_sentiment
        }
    
    def _analyze_secular_trends(self, company: QualifiedCompany, research_data: Dict) -> Dict:
        """Analyze secular trends and industry outlook"""
        
        # Sector trend scores (based on general market outlook)
        sector_trends = {
            'Technology': {'base_score': 8.0, 'outlook': 'Strong growth driven by AI, cloud, and digital transformation'},
            'Healthcare': {'base_score': 7.5, 'outlook': 'Aging demographics and innovation driving growth'},
            'Consumer Cyclical': {'base_score': 5.5, 'outlook': 'Mixed outlook dependent on economic conditions'},
            'Consumer Defensive': {'base_score': 6.0, 'outlook': 'Stable demand with inflation pressures'},
            'Financial Services': {'base_score': 6.5, 'outlook': 'Benefiting from higher interest rates'},
            'Industrial': {'base_score': 6.0, 'outlook': 'Infrastructure spending and automation trends'},
            'Utilities': {'base_score': 5.0, 'outlook': 'Stable but facing energy transition challenges'},
            'Energy': {'base_score': 4.5, 'outlook': 'Transition to renewable energy creating headwinds'},
            'Materials': {'base_score': 5.0, 'outlook': 'Cyclical with infrastructure demand support'},
            'Real Estate': {'base_score': 4.0, 'outlook': 'Facing headwinds from higher interest rates'}
        }
        
        sector_info = sector_trends.get(company.sector, {'base_score': 5.0, 'outlook': 'Mixed outlook'})
        trend_score = sector_info['base_score']
        
        # Analyze industry trend content
        trend_content = research_data.get('industry_trends', [])
        if trend_content:
            positive_trend_keywords = ['growth', 'expansion', 'opportunity', 'innovation', 'demand', 'bullish']
            negative_trend_keywords = ['decline', 'headwinds', 'challenges', 'disruption', 'bearish', 'contraction']
            
            trend_signals = 0
            for item in trend_content:
                content = (item.get('title', '') + ' ' + item.get('content', '')).lower()
                
                for keyword in positive_trend_keywords:
                    if keyword in content:
                        trend_signals += 1
                
                for keyword in negative_trend_keywords:
                    if keyword in content:
                        trend_signals -= 1
            
            # Adjust trend score based on research
            if trend_signals > 2:
                trend_score += 1.0
            elif trend_signals > 0:
                trend_score += 0.5
            elif trend_signals < -2:
                trend_score -= 1.0
            elif trend_signals < 0:
                trend_score -= 0.5
        
        # Determine trend alignment
        if trend_score >= 7.5:
            alignment = 'STRONG'
        elif trend_score >= 5.5:
            alignment = 'MODERATE'
        else:
            alignment = 'WEAK'
        
        trend_score = min(10.0, max(0.0, trend_score))
        
        return {
            'alignment': alignment,
            'score': round(trend_score, 1),
            'outlook': sector_info['outlook']
        }
    
    def _generate_llm_assessment(self, company: QualifiedCompany, research_data: Dict, 
                                moat_analysis: Dict, sentiment_analysis: Dict, trend_analysis: Dict) -> Dict:
        """Generate LLM-powered qualitative assessment"""
        try:
            if not self.llm:
                return self._generate_fallback_assessment(moat_analysis, sentiment_analysis, trend_analysis)
            
            # Prepare research summary for LLM
            research_summary = self._prepare_research_summary(research_data)
            
            prompt = f"""
Provide qualitative investment analysis for {company.company_name} ({company.ticker}):

COMPANY PROFILE:
- Sector: {company.sector}
- Market Cap: ${company.market_cap:,.0f}

RESEARCH FINDINGS:
{research_summary}

QUANTITATIVE ANALYSIS RESULTS:
- Moat Type: {moat_analysis['type']}
- Moat Strength: {moat_analysis['strength']} (Score: {moat_analysis['score']}/10)
- Sentiment: {sentiment_analysis['news_sentiment']} (Score: {sentiment_analysis['score']}/10)
- Trend Alignment: {trend_analysis['alignment']} (Score: {trend_analysis['score']}/10)

Provide your qualitative assessment:

KEY_INSIGHTS:
1. [Primary qualitative strength]
2. [Secondary competitive advantage]
3. [Key risk or concern]

RESEARCH_SUMMARY: [2-3 sentence synthesis of research findings]

Focus on business quality, competitive positioning, and long-term prospects.
"""
            
            messages = [
                SystemMessage(content="You are a qualitative equity analyst providing business quality assessment."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            return self._parse_llm_assessment(response.content)
            
        except Exception as e:
            self.logger.error(f"Error generating LLM assessment: {e}")
            return self._generate_fallback_assessment(moat_analysis, sentiment_analysis, trend_analysis)
    
    def _prepare_research_summary(self, research_data: Dict) -> str:
        """Prepare research summary for LLM analysis"""
        summary_parts = []
        
        # Company info
        company_info = research_data.get('company_info', [])
        if company_info:
            summary_parts.append("Company Information:")
            for item in company_info[:2]:  # Limit to top 2 items
                summary_parts.append(f"- {item.get('title', '')}")
        
        # Competitive landscape
        competitive_info = research_data.get('competitive_landscape', [])
        if competitive_info:
            summary_parts.append("\nCompetitive Position:")
            for item in competitive_info[:2]:
                summary_parts.append(f"- {item.get('title', '')}")
        
        # Recent news
        news_info = research_data.get('recent_news', [])
        if news_info:
            summary_parts.append("\nRecent Developments:")
            for item in news_info[:2]:
                summary_parts.append(f"- {item.get('title', '')}")
        
        # Industry trends
        trend_info = research_data.get('industry_trends', [])
        if trend_info:
            summary_parts.append("\nIndustry Outlook:")
            for item in trend_info[:2]:
                summary_parts.append(f"- {item.get('title', '')}")
        
        return '\n'.join(summary_parts)
    
    def _parse_llm_assessment(self, response: str) -> Dict:
        """Parse LLM assessment response"""
        try:
            lines = response.strip().split('\n')
            
            insights = []
            summary = ""
            
            current_section = None
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('KEY_INSIGHTS:'):
                    current_section = 'insights'
                elif line.startswith('RESEARCH_SUMMARY:'):
                    current_section = 'summary'
                elif current_section == 'insights' and line.startswith(('1.', '2.', '3.')):
                    insights.append(line[2:].strip())
                elif current_section == 'summary' and line:
                    summary += " " + line
            
            return {
                'insights': insights,
                'summary': summary.strip()
            }
            
        except Exception as e:
            self.logger.error(f"Error parsing LLM assessment: {e}")
            return self._generate_fallback_assessment({}, {}, {})
    
    def _generate_fallback_assessment(self, moat_analysis: Dict, sentiment_analysis: Dict, trend_analysis: Dict) -> Dict:
        """Generate fallback assessment without LLM"""
        
        insights = []
        
        # Generate insights based on scores
        if moat_analysis.get('score', 0) >= 7.0:
            insights.append(f"Strong {moat_analysis.get('type', 'competitive')} advantage provides defensive moat")
        
        if sentiment_analysis.get('score', 0) >= 7.0:
            insights.append("Positive market sentiment and news coverage")
        elif sentiment_analysis.get('score', 0) <= 3.0:
            insights.append("Negative sentiment presents potential risk")
        
        if trend_analysis.get('score', 0) >= 7.0:
            insights.append("Well-positioned for favorable secular trends")
        elif trend_analysis.get('score', 0) <= 4.0:
            insights.append("Facing headwinds from industry trends")
        
        summary = f"Qualitative analysis shows {moat_analysis.get('strength', 'moderate').lower()} competitive position with {sentiment_analysis.get('news_sentiment', 'neutral').lower()} sentiment."
        
        return {
            'insights': insights,
            'summary': summary
        }
    
    def _save_rationale_results(self, analyses: Dict[str, RationaleAnalysis]):
        """Save rationale analysis results"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"tracing/rationale_analysis_{timestamp}.json"
            
            results = {
                'timestamp': timestamp,
                'total_analyzed': len(analyses),
                'analyses': {ticker: asdict(analysis) for ticker, analysis in analyses.items()},
                'summary': self._generate_analysis_summary(analyses)
            }
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
                
            self.logger.info(f"Saved rationale analysis results to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error saving rationale results: {e}")
    
    def _generate_analysis_summary(self, analyses: Dict[str, RationaleAnalysis]) -> Dict:
        """Generate summary statistics for rationale analysis"""
        if not analyses:
            return {}
        
        analysis_list = list(analyses.values())
        
        return {
            'total_companies': len(analysis_list),
            'avg_qualitative_score': sum(a.overall_qualitative_score for a in analysis_list) / len(analysis_list),
            'avg_moat_score': sum(a.moat_score for a in analysis_list) / len(analysis_list),
            'avg_sentiment_score': sum(a.sentiment_score for a in analysis_list) / len(analysis_list),
            'avg_trend_score': sum(a.trend_score for a in analysis_list) / len(analysis_list),
            'strong_moats': len([a for a in analysis_list if a.moat_strength == 'WIDE']),
            'positive_sentiment': len([a for a in analysis_list if a.sentiment_score >= 7.0]),
            'strong_trends': len([a for a in analysis_list if a.trend_alignment == 'STRONG']),
            'total_citations': sum(len(a.citations) for a in analysis_list)
        }

