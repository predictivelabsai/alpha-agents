"""
Sentiment Agent for equity portfolio construction.
Analyzes financial news, analyst ratings, and market sentiment.
Based on the Alpha Agents paper.
"""

from typing import Dict, List, Any, Optional
import json
import re
from datetime import datetime, timedelta
from langchain_core.messages import HumanMessage
from .base_agent import BaseAgent, Stock, AgentAnalysis, InvestmentDecision, RiskTolerance

class SentimentAgent(BaseAgent):
    """
    Sentiment Agent specializes in analyzing market sentiment,
    news sentiment, analyst ratings, and social media trends.
    """
    
    def __init__(self, risk_tolerance: RiskTolerance, llm_client=None):
        super().__init__(risk_tolerance, llm_client)
        self.agent_name = "sentiment"
        
    def analyze(self, stock: Stock, market_data: Optional[Dict] = None) -> AgentAnalysis:
        """
        Analyze stock sentiment and provide investment recommendation.
        
        Args:
            stock: Stock information to analyze
            market_data: Optional market context data including news and sentiment
            
        Returns:
            AgentAnalysis with sentiment-based recommendation
        """
        try:
            # Perform sentiment analysis
            sentiment_metrics = self._calculate_sentiment_metrics(stock, market_data)
            news_analysis = self._analyze_news_sentiment(stock, market_data)
            analyst_consensus = self._analyze_analyst_ratings(stock, market_data)
            social_sentiment = self._analyze_social_sentiment(stock, market_data)
            
            # Generate LLM-enhanced analysis if available
            if self.llm_client:
                llm_analysis = self._get_llm_analysis(stock, sentiment_metrics, news_analysis, analyst_consensus, social_sentiment)
                recommendation, confidence, reasoning = self._parse_llm_response(llm_analysis)
            else:
                recommendation, confidence, reasoning = self._fallback_analysis(sentiment_metrics, news_analysis, analyst_consensus)
            
            # Adjust for risk tolerance
            confidence = self.adjust_for_risk_tolerance(confidence)
            
            # Extract key factors and concerns
            key_factors = self._extract_key_factors(sentiment_metrics, news_analysis, analyst_consensus)
            concerns = self._identify_concerns(sentiment_metrics, news_analysis, analyst_consensus)
            
            return AgentAnalysis(
                agent_name=self.agent_name,
                stock_symbol=stock.symbol,
                recommendation=recommendation,
                confidence_score=confidence,
                target_price=self._calculate_sentiment_target_price(stock, analyst_consensus),
                risk_assessment=self._assess_sentiment_risk(sentiment_metrics, news_analysis),
                key_factors=key_factors,
                concerns=concerns,
                reasoning=reasoning
            )
            
        except Exception as e:
            self.logger.error(f"Error in sentiment analysis for {stock.symbol}: {e}")
            return AgentAnalysis(
                agent_name=self.agent_name,
                stock_symbol=stock.symbol,
                recommendation=InvestmentDecision.HOLD,
                confidence_score=0.3,
                risk_assessment="HIGH",
                key_factors=[],
                concerns=["Analysis error occurred"],
                reasoning=f"Unable to complete sentiment analysis: {str(e)}"
            )
    
    def _calculate_sentiment_metrics(self, stock: Stock, market_data: Optional[Dict]) -> Dict[str, Any]:
        """Calculate sentiment-based metrics."""
        metrics = {
            'overall_sentiment': 0.5,  # Neutral baseline
            'sentiment_strength': 0.5,
            'sentiment_trend': 'neutral',
            'volume_sentiment': 0.5,
            'momentum_score': 0.5
        }
        
        # Volume-based sentiment (high volume can indicate strong sentiment)
        if stock.volume:
            # Assume average daily volume is around current volume (simplified)
            avg_volume = stock.volume * 0.8  # Simplified assumption
            volume_ratio = stock.volume / avg_volume if avg_volume > 0 else 1.0
            
            if volume_ratio > 1.5:
                metrics['volume_sentiment'] = 0.7  # High interest
            elif volume_ratio > 1.2:
                metrics['volume_sentiment'] = 0.6  # Moderate interest
            else:
                metrics['volume_sentiment'] = 0.4  # Low interest
        
        # Price momentum (simplified - would need historical data in real implementation)
        # For now, use sector and market cap as proxies
        if stock.sector in ['Technology', 'Healthcare', 'Consumer']:
            metrics['momentum_score'] = 0.6  # Growth sectors
        else:
            metrics['momentum_score'] = 0.5  # Neutral
        
        # Market data integration
        if market_data:
            if 'sentiment_score' in market_data:
                metrics['overall_sentiment'] = market_data['sentiment_score']
            if 'news_sentiment' in market_data:
                metrics['news_sentiment'] = market_data['news_sentiment']
            if 'social_sentiment' in market_data:
                metrics['social_sentiment'] = market_data['social_sentiment']
        
        return metrics
    
    def _analyze_news_sentiment(self, stock: Stock, market_data: Optional[Dict]) -> Dict[str, Any]:
        """Analyze news sentiment."""
        news_analysis = {
            'sentiment_score': 0.5,  # Neutral
            'news_volume': 'moderate',
            'key_themes': [],
            'recent_catalysts': [],
            'sentiment_trend': 'stable'
        }
        
        # Sector-based news sentiment assumptions (simplified)
        sector_sentiment = {
            'Technology': 0.6,  # Generally positive
            'Healthcare': 0.55,
            'Financial': 0.5,
            'Consumer': 0.55,
            'Industrial': 0.5,
            'Energy': 0.45,  # Often volatile
            'Utilities': 0.5,
            'Materials': 0.5,
            'Real Estate': 0.5,
            'Telecommunications': 0.45
        }
        
        news_analysis['sentiment_score'] = sector_sentiment.get(stock.sector, 0.5)
        
        # Market cap influence on news coverage
        if stock.market_cap >= 50e9:  # Mega cap
            news_analysis['news_volume'] = 'high'
            news_analysis['key_themes'] = ['Market leadership', 'Institutional focus']
        elif stock.market_cap >= 10e9:  # Large cap
            news_analysis['news_volume'] = 'moderate'
            news_analysis['key_themes'] = ['Sector performance', 'Earnings focus']
        else:
            news_analysis['news_volume'] = 'low'
            news_analysis['key_themes'] = ['Growth potential', 'Niche market']
        
        # Integrate market data if available
        if market_data and 'news_data' in market_data:
            news_data = market_data['news_data']
            if 'sentiment' in news_data:
                news_analysis['sentiment_score'] = news_data['sentiment']
            if 'themes' in news_data:
                news_analysis['key_themes'] = news_data['themes']
        
        return news_analysis
    
    def _analyze_analyst_ratings(self, stock: Stock, market_data: Optional[Dict]) -> Dict[str, Any]:
        """Analyze analyst ratings and consensus."""
        analyst_analysis = {
            'consensus_rating': 'hold',
            'rating_score': 0.5,
            'price_target_vs_current': 0.0,
            'analyst_count': 0,
            'rating_trend': 'stable'
        }
        
        # Market cap based analyst coverage assumptions
        if stock.market_cap >= 10e9:  # Large cap
            analyst_analysis['analyst_count'] = 15  # High coverage
            analyst_analysis['consensus_rating'] = 'buy'
            analyst_analysis['rating_score'] = 0.65
        elif stock.market_cap >= 2e9:  # Mid cap
            analyst_analysis['analyst_count'] = 8  # Moderate coverage
            analyst_analysis['consensus_rating'] = 'hold'
            analyst_analysis['rating_score'] = 0.55
        else:  # Small cap
            analyst_analysis['analyst_count'] = 3  # Limited coverage
            analyst_analysis['consensus_rating'] = 'hold'
            analyst_analysis['rating_score'] = 0.5
        
        # Sector-based rating adjustments
        growth_sectors = ['Technology', 'Healthcare', 'Consumer']
        if stock.sector in growth_sectors:
            analyst_analysis['rating_score'] += 0.1
            analyst_analysis['consensus_rating'] = 'buy' if analyst_analysis['rating_score'] > 0.6 else 'hold'
        
        # Integrate market data if available
        if market_data and 'analyst_data' in market_data:
            analyst_data = market_data['analyst_data']
            if 'consensus' in analyst_data:
                analyst_analysis['consensus_rating'] = analyst_data['consensus']
            if 'target_price' in analyst_data:
                target_price = analyst_data['target_price']
                analyst_analysis['price_target_vs_current'] = (target_price - stock.current_price) / stock.current_price
        
        return analyst_analysis
    
    def _analyze_social_sentiment(self, stock: Stock, market_data: Optional[Dict]) -> Dict[str, Any]:
        """Analyze social media sentiment."""
        social_analysis = {
            'social_sentiment': 0.5,
            'mention_volume': 'low',
            'trending_topics': [],
            'retail_interest': 'moderate'
        }
        
        # Market cap and sector influence on social sentiment
        if stock.market_cap >= 100e9:  # Mega cap
            social_analysis['mention_volume'] = 'high'
            social_analysis['retail_interest'] = 'high'
        elif stock.market_cap >= 10e9:  # Large cap
            social_analysis['mention_volume'] = 'moderate'
            social_analysis['retail_interest'] = 'moderate'
        
        # Tech stocks tend to have higher social media presence
        if stock.sector == 'Technology':
            social_analysis['social_sentiment'] = 0.6
            social_analysis['trending_topics'] = ['Innovation', 'Growth', 'AI/Tech trends']
        
        # Integrate market data if available
        if market_data and 'social_data' in market_data:
            social_data = market_data['social_data']
            if 'sentiment' in social_data:
                social_analysis['social_sentiment'] = social_data['sentiment']
            if 'topics' in social_data:
                social_analysis['trending_topics'] = social_data['topics']
        
        return social_analysis
    
    def _get_llm_analysis(self, stock: Stock, sentiment_metrics: Dict, news_analysis: Dict, 
                         analyst_consensus: Dict, social_sentiment: Dict) -> str:
        """Get enhanced sentiment analysis from LLM."""
        analysis_prompt = f"""
        As a Market Sentiment Analysis expert, analyze the following stock:
        
        Company: {stock.company_name} ({stock.symbol})
        Sector: {stock.sector}
        Current Price: ${stock.current_price:.2f}
        Market Cap: {self.format_currency(stock.market_cap)}
        
        Sentiment Metrics:
        - Overall Sentiment Score: {sentiment_metrics['overall_sentiment']:.2f}
        - Volume Sentiment: {sentiment_metrics['volume_sentiment']:.2f}
        - Momentum Score: {sentiment_metrics['momentum_score']:.2f}
        
        News Analysis:
        - News Sentiment: {news_analysis['sentiment_score']:.2f}
        - News Volume: {news_analysis['news_volume']}
        - Key Themes: {', '.join(news_analysis['key_themes'])}
        
        Analyst Consensus:
        - Consensus Rating: {analyst_consensus['consensus_rating']}
        - Rating Score: {analyst_consensus['rating_score']:.2f}
        - Analyst Coverage: {analyst_consensus['analyst_count']} analysts
        
        Social Sentiment:
        - Social Sentiment: {social_sentiment['social_sentiment']:.2f}
        - Mention Volume: {social_sentiment['mention_volume']}
        - Retail Interest: {social_sentiment['retail_interest']}
        
        Risk Tolerance: {self.risk_tolerance.value}
        
        Based on sentiment analysis, provide:
        1. Investment recommendation (BUY/SELL/HOLD/AVOID)
        2. Confidence level (0.0 to 1.0)
        3. Key sentiment drivers
        4. Market perception insights
        5. Potential sentiment risks
        6. Detailed reasoning
        
        Focus on market psychology, investor sentiment trends, and momentum indicators.
        """
        
        if self.llm_client:
            response = self.llm_client.invoke([HumanMessage(content=analysis_prompt)])
            return response.content
        else:
            return self._fallback_analysis(sentiment_metrics, news_analysis, analyst_consensus)[2]
    
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
    
    def _fallback_analysis(self, sentiment_metrics: Dict, news_analysis: Dict, analyst_consensus: Dict) -> tuple:
        """Fallback analysis when LLM is not available."""
        # Weighted sentiment score
        overall_sentiment = (
            sentiment_metrics['overall_sentiment'] * 0.3 +
            news_analysis['sentiment_score'] * 0.3 +
            analyst_consensus['rating_score'] * 0.4
        )
        
        # Determine recommendation based on sentiment
        if overall_sentiment >= 0.7:
            recommendation = InvestmentDecision.BUY
            confidence = 0.75
        elif overall_sentiment >= 0.6:
            recommendation = InvestmentDecision.BUY
            confidence = 0.65
        elif overall_sentiment >= 0.4:
            recommendation = InvestmentDecision.HOLD
            confidence = 0.6
        else:
            recommendation = InvestmentDecision.AVOID
            confidence = 0.7
        
        reasoning = f"""
        Sentiment Analysis Summary:
        - Overall Sentiment Score: {overall_sentiment:.2f}
        - News Sentiment: {news_analysis['sentiment_score']:.2f} ({news_analysis['news_volume']} volume)
        - Analyst Consensus: {analyst_consensus['consensus_rating']} ({analyst_consensus['analyst_count']} analysts)
        - Volume Sentiment: {sentiment_metrics['volume_sentiment']:.2f}
        
        Recommendation based on market sentiment, news analysis, and analyst consensus.
        """
        
        return recommendation, confidence, reasoning
    
    def _calculate_sentiment_target_price(self, stock: Stock, analyst_consensus: Dict) -> Optional[float]:
        """Calculate target price based on sentiment analysis."""
        if analyst_consensus['price_target_vs_current'] != 0.0:
            # Use analyst target if available
            target_price = stock.current_price * (1 + analyst_consensus['price_target_vs_current'])
            return round(target_price, 2)
        else:
            # Simple sentiment-based target
            sentiment_multiplier = 1.0
            if analyst_consensus['consensus_rating'] == 'buy':
                sentiment_multiplier = 1.15  # 15% upside
            elif analyst_consensus['consensus_rating'] == 'sell':
                sentiment_multiplier = 0.90  # 10% downside
            
            return round(stock.current_price * sentiment_multiplier, 2)
    
    def _assess_sentiment_risk(self, sentiment_metrics: Dict, news_analysis: Dict) -> str:
        """Assess risk level based on sentiment metrics."""
        risk_factors = 0
        
        if sentiment_metrics['overall_sentiment'] < 0.3:
            risk_factors += 2  # Very negative sentiment
        elif sentiment_metrics['overall_sentiment'] < 0.4:
            risk_factors += 1  # Negative sentiment
        
        if news_analysis['sentiment_score'] < 0.3:
            risk_factors += 1  # Negative news sentiment
        
        if sentiment_metrics['volume_sentiment'] > 0.8:
            risk_factors += 1  # Potentially overheated
        
        if risk_factors >= 3:
            return "HIGH"
        elif risk_factors >= 2:
            return "MODERATE"
        else:
            return "LOW"
    
    def _extract_key_factors(self, sentiment_metrics: Dict, news_analysis: Dict, analyst_consensus: Dict) -> List[str]:
        """Extract key positive sentiment factors."""
        factors = []
        
        if sentiment_metrics['overall_sentiment'] > 0.6:
            factors.append("Positive market sentiment")
        if news_analysis['sentiment_score'] > 0.6:
            factors.append("Favorable news coverage")
        if analyst_consensus['consensus_rating'] == 'buy':
            factors.append(f"Analyst consensus: {analyst_consensus['consensus_rating'].upper()}")
        if analyst_consensus['analyst_count'] >= 10:
            factors.append("Strong analyst coverage")
        if sentiment_metrics['volume_sentiment'] > 0.6:
            factors.append("High investor interest")
        
        # Add key themes from news
        factors.extend(news_analysis['key_themes'][:2])
        
        return factors[:5]  # Limit to top 5
    
    def _identify_concerns(self, sentiment_metrics: Dict, news_analysis: Dict, analyst_consensus: Dict) -> List[str]:
        """Identify key sentiment concerns."""
        concerns = []
        
        if sentiment_metrics['overall_sentiment'] < 0.4:
            concerns.append("Negative market sentiment")
        if news_analysis['sentiment_score'] < 0.4:
            concerns.append("Unfavorable news sentiment")
        if analyst_consensus['consensus_rating'] == 'sell':
            concerns.append("Negative analyst consensus")
        if analyst_consensus['analyst_count'] < 3:
            concerns.append("Limited analyst coverage")
        if sentiment_metrics['volume_sentiment'] < 0.3:
            concerns.append("Low investor interest")
        if news_analysis['news_volume'] == 'low':
            concerns.append("Limited news coverage")
        
        return concerns[:5]  # Limit to top 5

