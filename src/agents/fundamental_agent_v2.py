"""
Fundamental Agent - Lohusalu Capital Management
Selects trending sectors and screens against quantitative metrics
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import json
import os
from dataclasses import dataclass
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
# # from langchain_google_genai import ChatGoogleGenerativeAI
# # from langchain_anthropic import ChatAnthropic
# # from langchain_mistralai import ChatMistralAI

@dataclass
class SectorAnalysis:
    """Data class for sector analysis results"""
    sector: str
    weight: float
    momentum_score: float
    growth_potential: float
    reasoning: str
    top_stocks: List[str]

@dataclass
class StockScreening:
    """Data class for stock screening results"""
    ticker: str
    sector: str
    market_cap: float
    fundamental_score: float
    intrinsic_value: float
    current_price: float
    upside_potential: float
    metrics: Dict[str, Any]
    reasoning: str

class FundamentalAgent:
    """
    Fundamental Agent for sector selection and quantitative stock screening
    
    Key Functions:
    1. Identify trending sectors and assign weights
    2. Screen stocks against quantitative metrics
    3. Calculate intrinsic value estimates
    4. Provide fundamental analysis reasoning
    """
    
    def __init__(self, model_provider: str = "openai", model_name: str = "gpt-4"):
        self.logger = logging.getLogger(__name__)
        self.model_provider = model_provider
        self.model_name = model_name
        self.llm = self._initialize_llm()
        
        # Sector universe
        self.sectors = [
            'Technology', 'Healthcare', 'Financial Services', 'Consumer Cyclical',
            'Communication Services', 'Industrial', 'Consumer Defensive', 'Energy',
            'Utilities', 'Real Estate', 'Materials'
        ]
        
        # Quantitative screening criteria
        self.screening_criteria = {
            'min_market_cap': 100e6,  # $100M minimum
            'max_market_cap': 10e9,   # $10B maximum (focus on small-mid cap)
            'min_revenue_growth': 10,  # 10% minimum revenue growth
            'min_roe': 12,            # 12% minimum ROE
            'min_roic': 10,           # 10% minimum ROIC
            'max_debt_to_equity': 0.5, # Maximum 50% debt-to-equity
            'min_current_ratio': 1.2,  # Minimum current ratio
            'min_gross_margin': 20,    # 20% minimum gross margin
            'min_profit_margin': 5     # 5% minimum profit margin
        }
    
    def _initialize_llm(self):
        """Initialize the language model based on provider"""
        try:
            # Initialize LLM based on provider
            if self.model_provider == "openai":
                self.llm = ChatOpenAI(model=self.model_name, temperature=0.1)
            else:
                # Default to OpenAI for now
                self.llm = ChatOpenAI(model="gpt-4", temperature=0.1)
        except Exception as e:
            self.logger.error(f"Error initializing LLM: {e}")
            # Fallback to basic OpenAI model
            self.llm = ChatOpenAI(model="gpt-4", temperature=0.1)
    
    def analyze_sectors(self) -> List[SectorAnalysis]:
        """
        Analyze sectors to identify trending opportunities and assign weights
        """
        self.logger.info("Starting sector analysis...")
        
        sector_analyses = []
        
        for sector in self.sectors:
            try:
                # Get sector performance data
                sector_data = self._get_sector_performance(sector)
                
                # Analyze with LLM
                analysis = self._analyze_sector_with_llm(sector, sector_data)
                
                sector_analyses.append(analysis)
                
            except Exception as e:
                self.logger.error(f"Error analyzing sector {sector}: {e}")
                continue
        
        # Sort by weight (highest first)
        sector_analyses.sort(key=lambda x: x.weight, reverse=True)
        
        self.logger.info(f"Completed sector analysis for {len(sector_analyses)} sectors")
        return sector_analyses
    
    def _get_sector_performance(self, sector: str) -> Dict[str, Any]:
        """Get sector performance metrics"""
        try:
            # Get representative ETFs for sectors
            sector_etfs = {
                'Technology': 'XLK',
                'Healthcare': 'XLV',
                'Financial Services': 'XLF',
                'Consumer Cyclical': 'XLY',
                'Communication Services': 'XLC',
                'Industrial': 'XLI',
                'Consumer Defensive': 'XLP',
                'Energy': 'XLE',
                'Utilities': 'XLU',
                'Real Estate': 'XLRE',
                'Materials': 'XLB'
            }
            
            etf_ticker = sector_etfs.get(sector, 'SPY')
            etf = yf.Ticker(etf_ticker)
            
            # Get historical data
            hist_1y = etf.history(period="1y")
            hist_3m = etf.history(period="3mo")
            hist_1m = etf.history(period="1mo")
            
            # Calculate performance metrics
            performance_1y = ((hist_1y['Close'].iloc[-1] / hist_1y['Close'].iloc[0]) - 1) * 100
            performance_3m = ((hist_3m['Close'].iloc[-1] / hist_3m['Close'].iloc[0]) - 1) * 100
            performance_1m = ((hist_1m['Close'].iloc[-1] / hist_1m['Close'].iloc[0]) - 1) * 100
            
            # Calculate volatility
            volatility = hist_3m['Close'].pct_change().std() * np.sqrt(252) * 100
            
            # Calculate momentum (recent performance vs longer term)
            momentum = performance_1m - performance_3m
            
            return {
                'performance_1y': performance_1y,
                'performance_3m': performance_3m,
                'performance_1m': performance_1m,
                'volatility': volatility,
                'momentum': momentum,
                'volume_trend': hist_1m['Volume'].mean() / hist_3m['Volume'].mean()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting sector performance for {sector}: {e}")
            return {}
    
    def _analyze_sector_with_llm(self, sector: str, sector_data: Dict) -> SectorAnalysis:
        """Analyze sector using LLM"""
        try:
            prompt = f"""
            As a fundamental analyst, analyze the {sector} sector based on the following performance data:
            
            Performance Metrics:
            - 1-Year Performance: {sector_data.get('performance_1y', 0):.2f}%
            - 3-Month Performance: {sector_data.get('performance_3m', 0):.2f}%
            - 1-Month Performance: {sector_data.get('performance_1m', 0):.2f}%
            - Volatility: {sector_data.get('volatility', 0):.2f}%
            - Momentum: {sector_data.get('momentum', 0):.2f}%
            - Volume Trend: {sector_data.get('volume_trend', 1):.2f}x
            
            Provide:
            1. Weight (0-100): How attractive is this sector for investment?
            2. Momentum Score (0-100): Current momentum and trend strength
            3. Growth Potential (0-100): Long-term growth prospects
            4. Reasoning: 2-3 sentences explaining your analysis
            5. Top 3 stock tickers to focus on in this sector
            
            Format your response as JSON:
            {{
                "weight": <number>,
                "momentum_score": <number>,
                "growth_potential": <number>,
                "reasoning": "<text>",
                "top_stocks": ["<ticker1>", "<ticker2>", "<ticker3>"]
            }}
            """
            
            messages = [
                SystemMessage(content="You are an expert fundamental analyst specializing in sector analysis and stock selection."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Parse JSON response
            try:
                analysis_data = json.loads(response.content)
                return SectorAnalysis(
                    sector=sector,
                    weight=analysis_data.get('weight', 50),
                    momentum_score=analysis_data.get('momentum_score', 50),
                    growth_potential=analysis_data.get('growth_potential', 50),
                    reasoning=analysis_data.get('reasoning', ''),
                    top_stocks=analysis_data.get('top_stocks', [])
                )
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                return SectorAnalysis(
                    sector=sector,
                    weight=50,
                    momentum_score=50,
                    growth_potential=50,
                    reasoning=response.content[:200],
                    top_stocks=[]
                )
                
        except Exception as e:
            self.logger.error(f"Error in LLM sector analysis for {sector}: {e}")
            return SectorAnalysis(
                sector=sector,
                weight=0,
                momentum_score=0,
                growth_potential=0,
                reasoning=f"Error in analysis: {str(e)}",
                top_stocks=[]
            )
    
    def screen_stocks(self, sector_analyses: List[SectorAnalysis], max_stocks: int = 20) -> List[StockScreening]:
        """
        Screen stocks based on quantitative metrics and intrinsic value
        """
        self.logger.info("Starting quantitative stock screening...")
        
        all_stocks = []
        
        # Get stocks from top sectors
        top_sectors = sector_analyses[:5]  # Focus on top 5 sectors
        
        for sector_analysis in top_sectors:
            sector_stocks = self._screen_sector_stocks(sector_analysis)
            all_stocks.extend(sector_stocks)
        
        # Sort by fundamental score
        all_stocks.sort(key=lambda x: x.fundamental_score, reverse=True)
        
        # Return top stocks
        result = all_stocks[:max_stocks]
        self.logger.info(f"Completed screening, found {len(result)} qualifying stocks")
        
        return result
    
    def _screen_sector_stocks(self, sector_analysis: SectorAnalysis) -> List[StockScreening]:
        """Screen stocks within a specific sector"""
        sector_stocks = []
        
        # Get expanded stock list for the sector
        stock_universe = self._get_sector_stock_universe(sector_analysis.sector)
        
        # Include top stocks from sector analysis
        if sector_analysis.top_stocks:
            stock_universe.extend(sector_analysis.top_stocks)
        
        # Remove duplicates
        stock_universe = list(set(stock_universe))
        
        for ticker in stock_universe:
            try:
                stock_screening = self._analyze_stock_fundamentals(ticker, sector_analysis.sector)
                if stock_screening and self._passes_screening_criteria(stock_screening):
                    sector_stocks.append(stock_screening)
            except Exception as e:
                self.logger.error(f"Error screening stock {ticker}: {e}")
                continue
        
        return sector_stocks
    
    def _get_sector_stock_universe(self, sector: str) -> List[str]:
        """Get expanded stock universe for a sector"""
        # This would ideally connect to a stock screener API or database
        # For now, using a curated list of stocks by sector
        
        sector_stocks = {
            'Technology': [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'ADBE', 'CRM', 'ORCL', 'IBM',
                'INTC', 'AMD', 'QCOM', 'CSCO', 'AVGO', 'NOW', 'INTU', 'TXN', 'MU', 'AMAT',
                'SNOW', 'PLTR', 'CRWD', 'ZS', 'OKTA', 'DDOG', 'NET', 'FTNT', 'PANW'
            ],
            'Healthcare': [
                'JNJ', 'UNH', 'PFE', 'ABT', 'TMO', 'MDT', 'DHR', 'BMY', 'ABBV', 'MRK',
                'LLY', 'GILD', 'AMGN', 'ISRG', 'SYK', 'BSX', 'ZBH', 'BAX', 'BDX', 'EW',
                'VEEV', 'DXCM', 'HOLX', 'TDOC', 'TELADOC'
            ],
            'Financial Services': [
                'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SCHW', 'USB',
                'PNC', 'TFC', 'COF', 'CME', 'ICE', 'SPGI', 'MCO', 'V', 'MA', 'PYPL',
                'SQ', 'AFRM', 'SOFI', 'LC'
            ],
            'Consumer Cyclical': [
                'AMZN', 'TSLA', 'HD', 'NKE', 'MCD', 'SBUX', 'LOW', 'TJX', 'F', 'GM',
                'BKNG', 'MAR', 'RCL', 'CCL', 'NCLH', 'LULU', 'ULTA', 'ETSY', 'W', 'CHWY'
            ],
            'Communication Services': [
                'GOOGL', 'META', 'DIS', 'NFLX', 'CMCSA', 'VZ', 'T', 'TMUS', 'CHTR', 'DISH',
                'SNAP', 'PINS', 'ROKU', 'SPOT', 'ZM', 'DOCU', 'TEAM', 'TWLO', 'RBLX'
            ]
        }
        
        return sector_stocks.get(sector, [])
    
    def _analyze_stock_fundamentals(self, ticker: str, sector: str) -> Optional[StockScreening]:
        """Analyze individual stock fundamentals"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get financial statements
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow
            
            # Calculate key metrics
            metrics = self._calculate_fundamental_metrics(info, financials, balance_sheet, cash_flow)
            
            # Calculate intrinsic value
            intrinsic_value = self._calculate_intrinsic_value(metrics, info)
            
            # Calculate fundamental score
            fundamental_score = self._calculate_fundamental_score(metrics)
            
            # Get current price
            current_price = info.get('currentPrice', 0)
            
            # Calculate upside potential
            upside_potential = ((intrinsic_value - current_price) / current_price * 100) if current_price > 0 else 0
            
            # Generate reasoning
            reasoning = self._generate_fundamental_reasoning(ticker, metrics, fundamental_score)
            
            return StockScreening(
                ticker=ticker,
                sector=sector,
                market_cap=info.get('marketCap', 0),
                fundamental_score=fundamental_score,
                intrinsic_value=intrinsic_value,
                current_price=current_price,
                upside_potential=upside_potential,
                metrics=metrics,
                reasoning=reasoning
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing fundamentals for {ticker}: {e}")
            return None
    
    def _calculate_fundamental_metrics(self, info: Dict, financials: pd.DataFrame, 
                                     balance_sheet: pd.DataFrame, cash_flow: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive fundamental metrics"""
        metrics = {}
        
        try:
            # Basic metrics from info
            metrics['market_cap'] = info.get('marketCap', 0)
            metrics['pe_ratio'] = info.get('trailingPE', 0)
            metrics['pb_ratio'] = info.get('priceToBook', 0)
            metrics['roe'] = info.get('returnOnEquity', 0) * 100 if info.get('returnOnEquity') else 0
            metrics['gross_margin'] = info.get('grossMargins', 0) * 100 if info.get('grossMargins') else 0
            metrics['profit_margin'] = info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0
            metrics['current_ratio'] = info.get('currentRatio', 0)
            metrics['debt_to_equity'] = info.get('debtToEquity', 0) / 100 if info.get('debtToEquity') else 0
            
            # Growth metrics from financials
            if not financials.empty:
                # Revenue growth
                if 'Total Revenue' in financials.index:
                    revenues = financials.loc['Total Revenue'].dropna()
                    if len(revenues) >= 2:
                        revenue_growth = ((revenues.iloc[0] / revenues.iloc[-1]) ** (1/len(revenues)) - 1) * 100
                        metrics['revenue_growth'] = revenue_growth
                
                # Net income growth
                if 'Net Income' in financials.index:
                    net_incomes = financials.loc['Net Income'].dropna()
                    if len(net_incomes) >= 2:
                        ni_growth = ((net_incomes.iloc[0] / net_incomes.iloc[-1]) ** (1/len(net_incomes)) - 1) * 100
                        metrics['net_income_growth'] = ni_growth
            
            # ROIC calculation
            if not financials.empty and not balance_sheet.empty:
                try:
                    net_income = financials.loc['Net Income'].iloc[0] if 'Net Income' in financials.index else 0
                    total_equity = balance_sheet.loc['Total Stockholder Equity'].iloc[0] if 'Total Stockholder Equity' in balance_sheet.index else 0
                    total_debt = balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in balance_sheet.index else 0
                    
                    invested_capital = total_equity + total_debt
                    if invested_capital > 0:
                        metrics['roic'] = (net_income / invested_capital) * 100
                except:
                    metrics['roic'] = 0
            
            # Cash flow metrics
            if not cash_flow.empty:
                if 'Operating Cash Flow' in cash_flow.index:
                    operating_cf = cash_flow.loc['Operating Cash Flow'].iloc[0]
                    metrics['operating_cash_flow'] = operating_cf
                    
                    # Free cash flow
                    if 'Capital Expenditure' in cash_flow.index:
                        capex = cash_flow.loc['Capital Expenditure'].iloc[0]
                        metrics['free_cash_flow'] = operating_cf + capex  # capex is negative
            
        except Exception as e:
            self.logger.error(f"Error calculating fundamental metrics: {e}")
        
        return metrics
    
    def _calculate_intrinsic_value(self, metrics: Dict, info: Dict) -> float:
        """Calculate intrinsic value using DCF and multiple valuation methods"""
        try:
            # Simple DCF approximation
            free_cash_flow = metrics.get('free_cash_flow', 0)
            growth_rate = min(metrics.get('revenue_growth', 5), 15) / 100  # Cap at 15%
            discount_rate = 0.10  # 10% discount rate
            terminal_growth = 0.03  # 3% terminal growth
            
            if free_cash_flow > 0:
                # 5-year DCF
                dcf_value = 0
                for year in range(1, 6):
                    future_cf = free_cash_flow * ((1 + growth_rate) ** year)
                    present_value = future_cf / ((1 + discount_rate) ** year)
                    dcf_value += present_value
                
                # Terminal value
                terminal_cf = free_cash_flow * ((1 + growth_rate) ** 5) * (1 + terminal_growth)
                terminal_value = terminal_cf / (discount_rate - terminal_growth)
                terminal_pv = terminal_value / ((1 + discount_rate) ** 5)
                
                total_value = dcf_value + terminal_pv
                
                # Convert to per-share value
                shares_outstanding = info.get('sharesOutstanding', 1)
                intrinsic_value = total_value / shares_outstanding if shares_outstanding > 0 else 0
                
                return max(intrinsic_value, 0)
            
            # Fallback to P/E based valuation
            earnings_per_share = info.get('trailingEps', 0)
            if earnings_per_share > 0:
                # Use sector average P/E or conservative multiple
                conservative_pe = 15
                return earnings_per_share * conservative_pe
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Error calculating intrinsic value: {e}")
            return 0
    
    def _calculate_fundamental_score(self, metrics: Dict) -> float:
        """Calculate overall fundamental score (0-100)"""
        score = 0
        max_score = 100
        
        try:
            # Growth score (25 points)
            revenue_growth = metrics.get('revenue_growth', 0)
            if revenue_growth > 20:
                score += 25
            elif revenue_growth > 10:
                score += 20
            elif revenue_growth > 5:
                score += 15
            elif revenue_growth > 0:
                score += 10
            
            # Profitability score (25 points)
            roe = metrics.get('roe', 0)
            if roe > 20:
                score += 25
            elif roe > 15:
                score += 20
            elif roe > 10:
                score += 15
            elif roe > 5:
                score += 10
            
            # Financial strength score (25 points)
            current_ratio = metrics.get('current_ratio', 0)
            debt_to_equity = metrics.get('debt_to_equity', 1)
            
            if current_ratio > 2 and debt_to_equity < 0.3:
                score += 25
            elif current_ratio > 1.5 and debt_to_equity < 0.5:
                score += 20
            elif current_ratio > 1.2 and debt_to_equity < 0.7:
                score += 15
            elif current_ratio > 1.0:
                score += 10
            
            # Valuation score (25 points)
            pe_ratio = metrics.get('pe_ratio', 50)
            pb_ratio = metrics.get('pb_ratio', 10)
            
            if pe_ratio < 15 and pb_ratio < 2:
                score += 25
            elif pe_ratio < 20 and pb_ratio < 3:
                score += 20
            elif pe_ratio < 25 and pb_ratio < 4:
                score += 15
            elif pe_ratio < 30:
                score += 10
            
        except Exception as e:
            self.logger.error(f"Error calculating fundamental score: {e}")
        
        return min(score, max_score)
    
    def _passes_screening_criteria(self, stock: StockScreening) -> bool:
        """Check if stock passes screening criteria"""
        try:
            criteria = self.screening_criteria
            metrics = stock.metrics
            
            # Market cap check
            if not (criteria['min_market_cap'] <= stock.market_cap <= criteria['max_market_cap']):
                return False
            
            # Growth check
            if metrics.get('revenue_growth', 0) < criteria['min_revenue_growth']:
                return False
            
            # Profitability checks
            if metrics.get('roe', 0) < criteria['min_roe']:
                return False
            
            if metrics.get('roic', 0) < criteria['min_roic']:
                return False
            
            if metrics.get('gross_margin', 0) < criteria['min_gross_margin']:
                return False
            
            if metrics.get('profit_margin', 0) < criteria['min_profit_margin']:
                return False
            
            # Financial strength checks
            if metrics.get('debt_to_equity', 1) > criteria['max_debt_to_equity']:
                return False
            
            if metrics.get('current_ratio', 0) < criteria['min_current_ratio']:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking screening criteria: {e}")
            return False
    
    def _generate_fundamental_reasoning(self, ticker: str, metrics: Dict, score: float) -> str:
        """Generate reasoning for fundamental analysis"""
        try:
            prompt = f"""
            Provide a concise fundamental analysis reasoning for {ticker} based on these metrics:
            
            Financial Metrics:
            - Revenue Growth: {metrics.get('revenue_growth', 0):.1f}%
            - ROE: {metrics.get('roe', 0):.1f}%
            - ROIC: {metrics.get('roic', 0):.1f}%
            - Gross Margin: {metrics.get('gross_margin', 0):.1f}%
            - Profit Margin: {metrics.get('profit_margin', 0):.1f}%
            - Current Ratio: {metrics.get('current_ratio', 0):.2f}
            - Debt-to-Equity: {metrics.get('debt_to_equity', 0):.2f}
            - P/E Ratio: {metrics.get('pe_ratio', 0):.1f}
            - Fundamental Score: {score:.1f}/100
            
            Provide 2-3 sentences explaining the key strengths and weaknesses from a fundamental perspective.
            """
            
            messages = [
                SystemMessage(content="You are an expert fundamental analyst. Provide concise, factual analysis."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            self.logger.error(f"Error generating reasoning: {e}")
            return f"Fundamental analysis for {ticker} with score {score:.1f}/100 based on quantitative metrics."
    
    def save_analysis_trace(self, sector_analyses: List[SectorAnalysis], 
                           stock_screenings: List[StockScreening], 
                           output_dir: str = "tracing") -> str:
        """Save analysis trace to JSON file"""
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Create trace data
            trace_data = {
                'timestamp': datetime.now().isoformat(),
                'agent': 'FundamentalAgent',
                'model_provider': self.model_provider,
                'model_name': self.model_name,
                'sector_analyses': [
                    {
                        'sector': sa.sector,
                        'weight': sa.weight,
                        'momentum_score': sa.momentum_score,
                        'growth_potential': sa.growth_potential,
                        'reasoning': sa.reasoning,
                        'top_stocks': sa.top_stocks
                    } for sa in sector_analyses
                ],
                'stock_screenings': [
                    {
                        'ticker': ss.ticker,
                        'sector': ss.sector,
                        'market_cap': ss.market_cap,
                        'fundamental_score': ss.fundamental_score,
                        'intrinsic_value': ss.intrinsic_value,
                        'current_price': ss.current_price,
                        'upside_potential': ss.upside_potential,
                        'metrics': ss.metrics,
                        'reasoning': ss.reasoning
                    } for ss in stock_screenings
                ],
                'screening_criteria': self.screening_criteria
            }
            
            # Save to file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fundamental_agent_trace_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w') as f:
                json.dump(trace_data, f, indent=2, default=str)
            
            self.logger.info(f"Analysis trace saved to {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error saving analysis trace: {e}")
            return ""
    
    def run_analysis(self, save_trace: bool = True) -> Tuple[List[SectorAnalysis], List[StockScreening]]:
        """
        Run complete fundamental analysis pipeline
        
        Returns:
            Tuple of (sector_analyses, stock_screenings)
        """
        self.logger.info("Starting Fundamental Agent analysis...")
        
        # Step 1: Analyze sectors
        sector_analyses = self.analyze_sectors()
        
        # Step 2: Screen stocks
        stock_screenings = self.screen_stocks(sector_analyses)
        
        # Step 3: Save trace if requested
        if save_trace:
            self.save_analysis_trace(sector_analyses, stock_screenings)
        
        self.logger.info("Fundamental Agent analysis completed")
        
        return sector_analyses, stock_screenings

if __name__ == "__main__":
    # Test the agent
    agent = FundamentalAgent()
    sector_analyses, stock_screenings = agent.run_analysis()
    
    print(f"Found {len(sector_analyses)} sector analyses")
    print(f"Found {len(stock_screenings)} qualifying stocks")
    
    # Print top results
    print("\nTop Sectors:")
    for sa in sector_analyses[:3]:
        print(f"- {sa.sector}: Weight {sa.weight}, Reasoning: {sa.reasoning[:100]}...")
    
    print("\nTop Stocks:")
    for ss in stock_screenings[:5]:
        print(f"- {ss.ticker}: Score {ss.fundamental_score:.1f}, Upside {ss.upside_potential:.1f}%")

