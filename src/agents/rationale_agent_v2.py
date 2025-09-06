"""
Rationale Agent - Lohusalu Capital Management
Qualitative analysis with web search for moats, sentiment, and secular trends
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
# # from langchain_google_genai import ChatGoogleGenerativeAI
# # from langchain_anthropic import ChatAnthropic
# # from langchain_mistralai import ChatMistralAI
from tavily import TavilyClient
import requests
import time

@dataclass
class QualitativeAnalysis:
    """Data class for qualitative analysis results"""
    ticker: str
    company_name: str
    moat_analysis: Dict[str, Any]
    sentiment_analysis: Dict[str, Any]
    secular_trends: Dict[str, Any]
    competitive_position: Dict[str, Any]
    qualitative_score: float
    reasoning: str
    citations: List[str]
    search_queries_used: List[str]

@dataclass
class MoatAnalysis:
    """Data class for economic moat analysis"""
    moat_type: str
    moat_strength: str  # "Wide", "Narrow", "None"
    moat_score: float  # 0-100
    network_effects: bool
    switching_costs: bool
    cost_advantages: bool
    intangible_assets: bool
    efficient_scale: bool
    reasoning: str

@dataclass
class SentimentAnalysis:
    """Data class for sentiment analysis"""
    overall_sentiment: str  # "Positive", "Neutral", "Negative"
    sentiment_score: float  # 0-100
    analyst_sentiment: str
    news_sentiment: str
    social_sentiment: str
    recent_developments: List[str]
    reasoning: str

@dataclass
class SecularTrends:
    """Data class for secular trends analysis"""
    primary_trends: List[str]
    trend_alignment: str  # "Strong", "Moderate", "Weak", "Negative"
    trend_score: float  # 0-100
    growth_drivers: List[str]
    headwinds: List[str]
    time_horizon: str  # "Short-term", "Medium-term", "Long-term"
    reasoning: str

class RationaleAgent:
    """
    Rationale Agent for qualitative analysis using web search
    
    Key Functions:
    1. Analyze economic moats and competitive advantages
    2. Assess market sentiment and recent developments
    3. Identify secular trends and growth drivers
    4. Evaluate competitive positioning
    5. Provide comprehensive qualitative scoring
    """
    
    def __init__(self, model_provider: str = "openai", model_name: str = "gpt-4"):
        self.logger = logging.getLogger(__name__)
        self.model_provider = model_provider
        self.model_name = model_name
        self.llm = self._initialize_llm()
        
        # Initialize Tavily client
        self.tavily_client = self._initialize_tavily()
        
        # Search configuration
        self.max_search_results = 10
        self.search_timeout = 30
        
        # Scoring weights
        self.scoring_weights = {
            'moat_strength': 0.30,
            'sentiment': 0.25,
            'secular_trends': 0.25,
            'competitive_position': 0.20
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
    
    def _initialize_tavily(self):
        """Initialize Tavily search client"""
        try:
            api_key = os.getenv('TAVILY_API_KEY')
            if not api_key:
                self.logger.warning("TAVILY_API_KEY not found in environment variables")
                return None
            return TavilyClient(api_key=api_key)
        except Exception as e:
            self.logger.error(f"Error initializing Tavily client: {e}")
            return None
    
    def search_company_information(self, ticker: str, company_name: str, search_type: str) -> Dict[str, Any]:
        """
        Search for company information using Tavily
        
        Args:
            ticker: Stock ticker symbol
            company_name: Full company name
            search_type: Type of search ("moat", "sentiment", "trends", "competitive")
        """
        if not self.tavily_client:
            self.logger.warning("Tavily client not available, using fallback search")
            return self._fallback_search(ticker, company_name, search_type)
        
        try:
            # Define search queries based on type
            queries = self._get_search_queries(ticker, company_name, search_type)
            
            all_results = []
            citations = []
            
            for query in queries:
                try:
                    # Search with Tavily
                    search_results = self.tavily_client.search(
                        query=query,
                        search_depth="advanced",
                        max_results=self.max_search_results,
                        include_domains=["reuters.com", "bloomberg.com", "wsj.com", "ft.com", 
                                       "seekingalpha.com", "morningstar.com", "fool.com",
                                       "marketwatch.com", "cnbc.com", "yahoo.com"],
                        exclude_domains=["reddit.com", "twitter.com", "facebook.com"]
                    )
                    
                    # Process results
                    for result in search_results.get('results', []):
                        all_results.append({
                            'title': result.get('title', ''),
                            'content': result.get('content', ''),
                            'url': result.get('url', ''),
                            'published_date': result.get('published_date', ''),
                            'score': result.get('score', 0),
                            'query': query
                        })
                        
                        # Add citation
                        if result.get('url'):
                            citations.append(result['url'])
                    
                    # Rate limiting
                    time.sleep(0.5)
                    
                except Exception as e:
                    self.logger.error(f"Error in Tavily search for query '{query}': {e}")
                    continue
            
            return {
                'results': all_results,
                'citations': list(set(citations)),
                'queries_used': queries,
                'search_type': search_type
            }
            
        except Exception as e:
            self.logger.error(f"Error in company search: {e}")
            return {'results': [], 'citations': [], 'queries_used': [], 'search_type': search_type}
    
    def _get_search_queries(self, ticker: str, company_name: str, search_type: str) -> List[str]:
        """Generate search queries based on search type"""
        base_queries = [f"{company_name} {ticker}", f"{ticker} stock"]
        
        if search_type == "moat":
            return base_queries + [
                f"{company_name} competitive advantage moat",
                f"{ticker} economic moat competitive position",
                f"{company_name} barriers to entry competitive advantages",
                f"{ticker} network effects switching costs",
                f"{company_name} brand strength intangible assets"
            ]
        
        elif search_type == "sentiment":
            return base_queries + [
                f"{ticker} analyst ratings recent news",
                f"{company_name} earnings call sentiment",
                f"{ticker} stock news sentiment analysis",
                f"{company_name} recent developments outlook",
                f"{ticker} analyst upgrades downgrades"
            ]
        
        elif search_type == "trends":
            return base_queries + [
                f"{company_name} secular trends growth drivers",
                f"{ticker} industry trends market opportunity",
                f"{company_name} long term growth prospects",
                f"{ticker} market trends tailwinds headwinds",
                f"{company_name} future growth catalysts"
            ]
        
        elif search_type == "competitive":
            return base_queries + [
                f"{company_name} competitive position market share",
                f"{ticker} competitors competitive landscape",
                f"{company_name} market leadership position",
                f"{ticker} competitive threats opportunities",
                f"{company_name} industry position ranking"
            ]
        
        else:
            return base_queries
    
    def _fallback_search(self, ticker: str, company_name: str, search_type: str) -> Dict[str, Any]:
        """Fallback search method when Tavily is not available"""
        self.logger.info(f"Using fallback search for {ticker} - {search_type}")
        
        # This would implement alternative search methods
        # For now, return empty results
        return {
            'results': [],
            'citations': [],
            'queries_used': [f"{company_name} {ticker} {search_type}"],
            'search_type': search_type
        }
    
    def analyze_economic_moat(self, ticker: str, company_name: str) -> MoatAnalysis:
        """Analyze economic moat and competitive advantages"""
        self.logger.info(f"Analyzing economic moat for {ticker}")
        
        # Search for moat-related information
        search_data = self.search_company_information(ticker, company_name, "moat")
        
        # Analyze with LLM
        moat_analysis = self._analyze_moat_with_llm(ticker, company_name, search_data)
        
        return moat_analysis
    
    def _analyze_moat_with_llm(self, ticker: str, company_name: str, search_data: Dict) -> MoatAnalysis:
        """Analyze economic moat using LLM"""
        try:
            # Prepare search content
            search_content = ""
            for result in search_data.get('results', [])[:5]:  # Use top 5 results
                search_content += f"Title: {result.get('title', '')}\n"
                search_content += f"Content: {result.get('content', '')[:500]}...\n\n"
            
            prompt = f"""
            Analyze the economic moat and competitive advantages for {company_name} ({ticker}) based on the following information:

            Search Results:
            {search_content}

            Evaluate the company's economic moat across these dimensions:

            1. **Network Effects**: Does the company benefit from network effects where value increases with more users?
            2. **Switching Costs**: Are there high costs for customers to switch to competitors?
            3. **Cost Advantages**: Does the company have sustainable cost advantages (scale, location, unique assets)?
            4. **Intangible Assets**: Strong brands, patents, regulatory licenses, or other intangible assets?
            5. **Efficient Scale**: Does the company operate in a market with limited room for competitors?

            Provide your analysis in JSON format:
            {{
                "moat_type": "<primary moat type>",
                "moat_strength": "<Wide/Narrow/None>",
                "moat_score": <0-100>,
                "network_effects": <true/false>,
                "switching_costs": <true/false>,
                "cost_advantages": <true/false>,
                "intangible_assets": <true/false>,
                "efficient_scale": <true/false>,
                "reasoning": "<detailed explanation of moat analysis>"
            }}

            Scoring Guidelines:
            - Wide Moat (80-100): Multiple strong competitive advantages, very difficult to replicate
            - Narrow Moat (50-79): Some competitive advantages, moderately difficult to replicate  
            - No Moat (0-49): Limited or no sustainable competitive advantages
            """
            
            messages = [
                SystemMessage(content="You are an expert analyst specializing in competitive advantage and economic moat analysis."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Parse JSON response
            try:
                moat_data = json.loads(response.content)
                return MoatAnalysis(
                    moat_type=moat_data.get('moat_type', 'Unknown'),
                    moat_strength=moat_data.get('moat_strength', 'None'),
                    moat_score=moat_data.get('moat_score', 0),
                    network_effects=moat_data.get('network_effects', False),
                    switching_costs=moat_data.get('switching_costs', False),
                    cost_advantages=moat_data.get('cost_advantages', False),
                    intangible_assets=moat_data.get('intangible_assets', False),
                    efficient_scale=moat_data.get('efficient_scale', False),
                    reasoning=moat_data.get('reasoning', '')
                )
            except json.JSONDecodeError:
                # Fallback parsing
                return MoatAnalysis(
                    moat_type='Unknown',
                    moat_strength='None',
                    moat_score=0,
                    network_effects=False,
                    switching_costs=False,
                    cost_advantages=False,
                    intangible_assets=False,
                    efficient_scale=False,
                    reasoning=response.content[:500]
                )
                
        except Exception as e:
            self.logger.error(f"Error in moat analysis: {e}")
            return MoatAnalysis(
                moat_type='Error',
                moat_strength='None',
                moat_score=0,
                network_effects=False,
                switching_costs=False,
                cost_advantages=False,
                intangible_assets=False,
                efficient_scale=False,
                reasoning=f"Error in analysis: {str(e)}"
            )
    
    def analyze_sentiment(self, ticker: str, company_name: str) -> SentimentAnalysis:
        """Analyze market sentiment and recent developments"""
        self.logger.info(f"Analyzing sentiment for {ticker}")
        
        # Search for sentiment-related information
        search_data = self.search_company_information(ticker, company_name, "sentiment")
        
        # Analyze with LLM
        sentiment_analysis = self._analyze_sentiment_with_llm(ticker, company_name, search_data)
        
        return sentiment_analysis
    
    def _analyze_sentiment_with_llm(self, ticker: str, company_name: str, search_data: Dict) -> SentimentAnalysis:
        """Analyze sentiment using LLM"""
        try:
            # Prepare search content
            search_content = ""
            recent_developments = []
            
            for result in search_data.get('results', [])[:8]:  # Use top 8 results
                search_content += f"Title: {result.get('title', '')}\n"
                search_content += f"Content: {result.get('content', '')[:400]}...\n"
                search_content += f"Date: {result.get('published_date', 'Unknown')}\n\n"
                
                # Extract recent developments
                if result.get('title'):
                    recent_developments.append(result['title'])
            
            prompt = f"""
            Analyze the market sentiment for {company_name} ({ticker}) based on recent news and developments:

            Recent News and Analysis:
            {search_content}

            Evaluate sentiment across these dimensions:

            1. **Overall Sentiment**: General market sentiment towards the company
            2. **Analyst Sentiment**: Professional analyst opinions and ratings
            3. **News Sentiment**: Tone and content of recent news coverage
            4. **Social Sentiment**: Social media and retail investor sentiment (if available)

            Consider:
            - Recent earnings results and guidance
            - Management commentary and outlook
            - Analyst upgrades/downgrades
            - Industry developments affecting the company
            - Regulatory or legal developments
            - Product launches or business developments

            Provide your analysis in JSON format:
            {{
                "overall_sentiment": "<Positive/Neutral/Negative>",
                "sentiment_score": <0-100>,
                "analyst_sentiment": "<description>",
                "news_sentiment": "<description>",
                "social_sentiment": "<description>",
                "recent_developments": {recent_developments[:5]},
                "reasoning": "<detailed explanation of sentiment analysis>"
            }}

            Scoring Guidelines:
            - 80-100: Very positive sentiment, strong bullish indicators
            - 60-79: Positive sentiment, generally favorable outlook
            - 40-59: Neutral sentiment, mixed signals
            - 20-39: Negative sentiment, concerns present
            - 0-19: Very negative sentiment, significant bearish indicators
            """
            
            messages = [
                SystemMessage(content="You are an expert sentiment analyst specializing in market sentiment and investor psychology."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Parse JSON response
            try:
                sentiment_data = json.loads(response.content)
                return SentimentAnalysis(
                    overall_sentiment=sentiment_data.get('overall_sentiment', 'Neutral'),
                    sentiment_score=sentiment_data.get('sentiment_score', 50),
                    analyst_sentiment=sentiment_data.get('analyst_sentiment', ''),
                    news_sentiment=sentiment_data.get('news_sentiment', ''),
                    social_sentiment=sentiment_data.get('social_sentiment', ''),
                    recent_developments=sentiment_data.get('recent_developments', []),
                    reasoning=sentiment_data.get('reasoning', '')
                )
            except json.JSONDecodeError:
                return SentimentAnalysis(
                    overall_sentiment='Neutral',
                    sentiment_score=50,
                    analyst_sentiment='',
                    news_sentiment='',
                    social_sentiment='',
                    recent_developments=recent_developments[:5],
                    reasoning=response.content[:500]
                )
                
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis: {e}")
            return SentimentAnalysis(
                overall_sentiment='Error',
                sentiment_score=0,
                analyst_sentiment='',
                news_sentiment='',
                social_sentiment='',
                recent_developments=[],
                reasoning=f"Error in analysis: {str(e)}"
            )
    
    def analyze_secular_trends(self, ticker: str, company_name: str) -> SecularTrends:
        """Analyze secular trends and growth drivers"""
        self.logger.info(f"Analyzing secular trends for {ticker}")
        
        # Search for trends-related information
        search_data = self.search_company_information(ticker, company_name, "trends")
        
        # Analyze with LLM
        trends_analysis = self._analyze_trends_with_llm(ticker, company_name, search_data)
        
        return trends_analysis
    
    def _analyze_trends_with_llm(self, ticker: str, company_name: str, search_data: Dict) -> SecularTrends:
        """Analyze secular trends using LLM"""
        try:
            # Prepare search content
            search_content = ""
            for result in search_data.get('results', [])[:6]:  # Use top 6 results
                search_content += f"Title: {result.get('title', '')}\n"
                search_content += f"Content: {result.get('content', '')[:500]}...\n\n"
            
            prompt = f"""
            Analyze the secular trends and long-term growth drivers for {company_name} ({ticker}):

            Market and Industry Information:
            {search_content}

            Evaluate:

            1. **Primary Secular Trends**: Major long-term trends affecting the company
            2. **Trend Alignment**: How well positioned is the company to benefit from these trends
            3. **Growth Drivers**: Specific factors that could drive long-term growth
            4. **Headwinds**: Potential challenges or negative trends
            5. **Time Horizon**: When these trends are expected to materialize

            Consider:
            - Demographic shifts
            - Technology adoption cycles
            - Regulatory changes
            - Environmental and sustainability trends
            - Globalization and trade patterns
            - Consumer behavior changes
            - Industry disruption and innovation

            Provide your analysis in JSON format:
            {{
                "primary_trends": ["<trend1>", "<trend2>", "<trend3>"],
                "trend_alignment": "<Strong/Moderate/Weak/Negative>",
                "trend_score": <0-100>,
                "growth_drivers": ["<driver1>", "<driver2>", "<driver3>"],
                "headwinds": ["<headwind1>", "<headwind2>"],
                "time_horizon": "<Short-term/Medium-term/Long-term>",
                "reasoning": "<detailed explanation of trends analysis>"
            }}

            Scoring Guidelines:
            - 80-100: Strong alignment with multiple powerful secular trends
            - 60-79: Good alignment with some secular trends
            - 40-59: Moderate alignment, mixed trend exposure
            - 20-39: Weak alignment, limited trend benefits
            - 0-19: Negative alignment, facing secular headwinds
            """
            
            messages = [
                SystemMessage(content="You are an expert analyst specializing in secular trends and long-term market dynamics."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Parse JSON response
            try:
                trends_data = json.loads(response.content)
                return SecularTrends(
                    primary_trends=trends_data.get('primary_trends', []),
                    trend_alignment=trends_data.get('trend_alignment', 'Moderate'),
                    trend_score=trends_data.get('trend_score', 50),
                    growth_drivers=trends_data.get('growth_drivers', []),
                    headwinds=trends_data.get('headwinds', []),
                    time_horizon=trends_data.get('time_horizon', 'Medium-term'),
                    reasoning=trends_data.get('reasoning', '')
                )
            except json.JSONDecodeError:
                return SecularTrends(
                    primary_trends=[],
                    trend_alignment='Moderate',
                    trend_score=50,
                    growth_drivers=[],
                    headwinds=[],
                    time_horizon='Medium-term',
                    reasoning=response.content[:500]
                )
                
        except Exception as e:
            self.logger.error(f"Error in trends analysis: {e}")
            return SecularTrends(
                primary_trends=[],
                trend_alignment='Error',
                trend_score=0,
                growth_drivers=[],
                headwinds=[],
                time_horizon='Unknown',
                reasoning=f"Error in analysis: {str(e)}"
            )
    
    def analyze_competitive_position(self, ticker: str, company_name: str) -> Dict[str, Any]:
        """Analyze competitive position and market dynamics"""
        self.logger.info(f"Analyzing competitive position for {ticker}")
        
        # Search for competitive information
        search_data = self.search_company_information(ticker, company_name, "competitive")
        
        # Analyze with LLM
        competitive_analysis = self._analyze_competitive_with_llm(ticker, company_name, search_data)
        
        return competitive_analysis
    
    def _analyze_competitive_with_llm(self, ticker: str, company_name: str, search_data: Dict) -> Dict[str, Any]:
        """Analyze competitive position using LLM"""
        try:
            # Prepare search content
            search_content = ""
            for result in search_data.get('results', [])[:5]:
                search_content += f"Title: {result.get('title', '')}\n"
                search_content += f"Content: {result.get('content', '')[:400]}...\n\n"
            
            prompt = f"""
            Analyze the competitive position of {company_name} ({ticker}) in its industry:

            Competitive Information:
            {search_content}

            Evaluate:

            1. **Market Position**: Leadership position in the industry
            2. **Market Share**: Relative market share and trends
            3. **Competitive Threats**: Key competitors and competitive pressures
            4. **Competitive Advantages**: Unique strengths vs competitors
            5. **Industry Dynamics**: Overall industry competitiveness

            Provide your analysis in JSON format:
            {{
                "market_position": "<Leader/Strong/Moderate/Weak>",
                "market_share_trend": "<Gaining/Stable/Losing>",
                "competitive_score": <0-100>,
                "key_competitors": ["<competitor1>", "<competitor2>"],
                "competitive_threats": ["<threat1>", "<threat2>"],
                "competitive_advantages": ["<advantage1>", "<advantage2>"],
                "industry_attractiveness": "<High/Medium/Low>",
                "reasoning": "<detailed competitive analysis>"
            }}
            """
            
            messages = [
                SystemMessage(content="You are an expert competitive analyst specializing in industry dynamics and market positioning."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Parse JSON response
            try:
                return json.loads(response.content)
            except json.JSONDecodeError:
                return {
                    'market_position': 'Moderate',
                    'market_share_trend': 'Stable',
                    'competitive_score': 50,
                    'key_competitors': [],
                    'competitive_threats': [],
                    'competitive_advantages': [],
                    'industry_attractiveness': 'Medium',
                    'reasoning': response.content[:500]
                }
                
        except Exception as e:
            self.logger.error(f"Error in competitive analysis: {e}")
            return {
                'market_position': 'Error',
                'market_share_trend': 'Unknown',
                'competitive_score': 0,
                'key_competitors': [],
                'competitive_threats': [],
                'competitive_advantages': [],
                'industry_attractiveness': 'Unknown',
                'reasoning': f"Error in analysis: {str(e)}"
            }
    
    def calculate_qualitative_score(self, moat_analysis: MoatAnalysis, 
                                  sentiment_analysis: SentimentAnalysis,
                                  trends_analysis: SecularTrends,
                                  competitive_analysis: Dict[str, Any]) -> float:
        """Calculate overall qualitative score"""
        try:
            # Extract scores
            moat_score = moat_analysis.moat_score
            sentiment_score = sentiment_analysis.sentiment_score
            trends_score = trends_analysis.trend_score
            competitive_score = competitive_analysis.get('competitive_score', 50)
            
            # Calculate weighted score
            qualitative_score = (
                moat_score * self.scoring_weights['moat_strength'] +
                sentiment_score * self.scoring_weights['sentiment'] +
                trends_score * self.scoring_weights['secular_trends'] +
                competitive_score * self.scoring_weights['competitive_position']
            )
            
            return round(qualitative_score, 1)
            
        except Exception as e:
            self.logger.error(f"Error calculating qualitative score: {e}")
            return 50.0
    
    def run_qualitative_analysis(self, ticker: str, company_name: str) -> QualitativeAnalysis:
        """
        Run complete qualitative analysis for a stock
        
        Args:
            ticker: Stock ticker symbol
            company_name: Full company name
            
        Returns:
            QualitativeAnalysis object with complete analysis
        """
        self.logger.info(f"Starting qualitative analysis for {ticker}")
        
        try:
            # Run all analyses
            moat_analysis = self.analyze_economic_moat(ticker, company_name)
            sentiment_analysis = self.analyze_sentiment(ticker, company_name)
            trends_analysis = self.analyze_secular_trends(ticker, company_name)
            competitive_analysis = self.analyze_competitive_position(ticker, company_name)
            
            # Calculate overall score
            qualitative_score = self.calculate_qualitative_score(
                moat_analysis, sentiment_analysis, trends_analysis, competitive_analysis
            )
            
            # Collect all citations
            all_citations = []
            all_queries = []
            
            # Generate comprehensive reasoning
            reasoning = self._generate_comprehensive_reasoning(
                ticker, company_name, moat_analysis, sentiment_analysis, 
                trends_analysis, competitive_analysis, qualitative_score
            )
            
            return QualitativeAnalysis(
                ticker=ticker,
                company_name=company_name,
                moat_analysis={
                    'moat_type': moat_analysis.moat_type,
                    'moat_strength': moat_analysis.moat_strength,
                    'moat_score': moat_analysis.moat_score,
                    'network_effects': moat_analysis.network_effects,
                    'switching_costs': moat_analysis.switching_costs,
                    'cost_advantages': moat_analysis.cost_advantages,
                    'intangible_assets': moat_analysis.intangible_assets,
                    'efficient_scale': moat_analysis.efficient_scale,
                    'reasoning': moat_analysis.reasoning
                },
                sentiment_analysis={
                    'overall_sentiment': sentiment_analysis.overall_sentiment,
                    'sentiment_score': sentiment_analysis.sentiment_score,
                    'analyst_sentiment': sentiment_analysis.analyst_sentiment,
                    'news_sentiment': sentiment_analysis.news_sentiment,
                    'social_sentiment': sentiment_analysis.social_sentiment,
                    'recent_developments': sentiment_analysis.recent_developments,
                    'reasoning': sentiment_analysis.reasoning
                },
                secular_trends={
                    'primary_trends': trends_analysis.primary_trends,
                    'trend_alignment': trends_analysis.trend_alignment,
                    'trend_score': trends_analysis.trend_score,
                    'growth_drivers': trends_analysis.growth_drivers,
                    'headwinds': trends_analysis.headwinds,
                    'time_horizon': trends_analysis.time_horizon,
                    'reasoning': trends_analysis.reasoning
                },
                competitive_position=competitive_analysis,
                qualitative_score=qualitative_score,
                reasoning=reasoning,
                citations=all_citations,
                search_queries_used=all_queries
            )
            
        except Exception as e:
            self.logger.error(f"Error in qualitative analysis for {ticker}: {e}")
            # Return error analysis
            return QualitativeAnalysis(
                ticker=ticker,
                company_name=company_name,
                moat_analysis={},
                sentiment_analysis={},
                secular_trends={},
                competitive_position={},
                qualitative_score=0,
                reasoning=f"Error in qualitative analysis: {str(e)}",
                citations=[],
                search_queries_used=[]
            )
    
    def _generate_comprehensive_reasoning(self, ticker: str, company_name: str,
                                        moat_analysis: MoatAnalysis,
                                        sentiment_analysis: SentimentAnalysis,
                                        trends_analysis: SecularTrends,
                                        competitive_analysis: Dict[str, Any],
                                        qualitative_score: float) -> str:
        """Generate comprehensive reasoning for qualitative analysis"""
        try:
            prompt = f"""
            Provide a comprehensive qualitative investment thesis for {company_name} ({ticker}) based on the following analysis:

            Economic Moat Analysis:
            - Moat Strength: {moat_analysis.moat_strength}
            - Moat Score: {moat_analysis.moat_score}/100
            - Key Advantages: Network Effects: {moat_analysis.network_effects}, Switching Costs: {moat_analysis.switching_costs}

            Sentiment Analysis:
            - Overall Sentiment: {sentiment_analysis.overall_sentiment}
            - Sentiment Score: {sentiment_analysis.sentiment_score}/100
            - Recent Developments: {sentiment_analysis.recent_developments[:3]}

            Secular Trends:
            - Trend Alignment: {trends_analysis.trend_alignment}
            - Trend Score: {trends_analysis.trend_score}/100
            - Primary Trends: {trends_analysis.primary_trends[:3]}

            Competitive Position:
            - Market Position: {competitive_analysis.get('market_position', 'Unknown')}
            - Competitive Score: {competitive_analysis.get('competitive_score', 0)}/100

            Overall Qualitative Score: {qualitative_score}/100

            Provide a 4-5 sentence investment thesis that synthesizes these qualitative factors and explains the overall attractiveness of the investment opportunity.
            """
            
            messages = [
                SystemMessage(content="You are an expert investment analyst providing comprehensive qualitative investment thesis."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            self.logger.error(f"Error generating reasoning: {e}")
            return f"Qualitative analysis for {company_name} with overall score {qualitative_score}/100."
    
    def save_analysis_trace(self, analysis: QualitativeAnalysis, output_dir: str = "tracing") -> str:
        """Save qualitative analysis trace to JSON file"""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Create trace data
            trace_data = {
                'timestamp': datetime.now().isoformat(),
                'agent': 'RationaleAgent',
                'model_provider': self.model_provider,
                'model_name': self.model_name,
                'analysis': {
                    'ticker': analysis.ticker,
                    'company_name': analysis.company_name,
                    'moat_analysis': analysis.moat_analysis,
                    'sentiment_analysis': analysis.sentiment_analysis,
                    'secular_trends': analysis.secular_trends,
                    'competitive_position': analysis.competitive_position,
                    'qualitative_score': analysis.qualitative_score,
                    'reasoning': analysis.reasoning,
                    'citations': analysis.citations,
                    'search_queries_used': analysis.search_queries_used
                },
                'scoring_weights': self.scoring_weights
            }
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rationale_agent_trace_{analysis.ticker}_{timestamp}.json"
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
    agent = RationaleAgent()
    analysis = agent.run_qualitative_analysis("AAPL", "Apple Inc.")
    
    print(f"Qualitative Analysis for {analysis.ticker}:")
    print(f"Overall Score: {analysis.qualitative_score}/100")
    print(f"Moat Strength: {analysis.moat_analysis.get('moat_strength', 'Unknown')}")
    print(f"Sentiment: {analysis.sentiment_analysis.get('overall_sentiment', 'Unknown')}")
    print(f"Reasoning: {analysis.reasoning[:200]}...")

