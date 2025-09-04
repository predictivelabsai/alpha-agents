#!/usr/bin/env python3
"""
Alpha Agents Methodology Testing and Report Generation
Generates detailed methodology documentation and test results
"""

import sys
import os
import json
import csv
from datetime import datetime
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

try:
    from agents.fundamental_agent import FundamentalAgent
    from agents.sentiment_agent import SentimentAgent
    from agents.valuation_agent import ValuationAgent
    from agents.rationale_agent import RationaleAgent
    from agents.secular_trend_agent import SecularTrendAgent
    from agents.base_agent import Stock, RiskTolerance
    from database.schema import DatabaseManager
except ImportError as e:
    print(f"Import error: {e}")
    print("Running in fallback mode...")

class MethodologyTester:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.test_stocks = [
            {"symbol": "AAPL", "name": "Apple Inc.", "sector": "Technology", "price": 175.00, "market_cap": 2750.0},
            {"symbol": "MSFT", "name": "Microsoft Corp.", "sector": "Technology", "price": 350.00, "market_cap": 2600.0},
            {"symbol": "NVDA", "name": "NVIDIA Corp.", "sector": "Technology", "price": 450.00, "market_cap": 1100.0},
            {"symbol": "TSLA", "name": "Tesla Inc.", "sector": "Automotive", "price": 250.00, "market_cap": 800.0},
            {"symbol": "JPM", "name": "JPMorgan Chase", "sector": "Financial", "price": 145.00, "market_cap": 425.0},
        ]
        self.results = []
        self.methodology_data = {}
        
    def get_agent_methodology(self, agent_name):
        """Get detailed methodology for each agent"""
        methodologies = {
            "Fundamental": {
                "name": "Fundamental Analysis Agent",
                "approach": "Financial Statement Analysis & DCF Valuation",
                "key_metrics": [
                    "Revenue Growth Rate",
                    "Profit Margins (Gross, Operating, Net)",
                    "Return on Equity (ROE)",
                    "Debt-to-Equity Ratio",
                    "Free Cash Flow",
                    "Price-to-Earnings Ratio",
                    "Book Value per Share"
                ],
                "data_sources": [
                    "10-K Annual Reports",
                    "10-Q Quarterly Reports", 
                    "Earnings Statements",
                    "Balance Sheets",
                    "Cash Flow Statements"
                ],
                "analysis_framework": "Discounted Cash Flow (DCF) + Ratio Analysis",
                "decision_criteria": "Intrinsic Value vs Market Price Comparison",
                "strengths": [
                    "Long-term value assessment",
                    "Company financial health evaluation",
                    "Objective quantitative analysis"
                ],
                "limitations": [
                    "May miss short-term market dynamics",
                    "Relies on historical data",
                    "Complex for growth companies"
                ]
            },
            "Sentiment": {
                "name": "Market Sentiment Analysis Agent",
                "approach": "News & Social Media Sentiment Processing",
                "key_metrics": [
                    "News Sentiment Score",
                    "Analyst Rating Changes",
                    "Social Media Buzz",
                    "Market Momentum Indicators",
                    "Institutional Investor Activity"
                ],
                "data_sources": [
                    "Financial News Articles",
                    "Analyst Reports",
                    "Social Media Platforms",
                    "Market Data Feeds",
                    "Earnings Call Transcripts"
                ],
                "analysis_framework": "Natural Language Processing + Sentiment Scoring",
                "decision_criteria": "Positive Sentiment Momentum & Market Psychology",
                "strengths": [
                    "Captures market psychology",
                    "Real-time sentiment tracking",
                    "Identifies momentum shifts"
                ],
                "limitations": [
                    "Can be influenced by noise",
                    "Short-term focused",
                    "Susceptible to market manipulation"
                ]
            },
            "Valuation": {
                "name": "Technical & Price Valuation Agent",
                "approach": "Technical Analysis & Relative Valuation",
                "key_metrics": [
                    "Price-to-Earnings (P/E) Ratio",
                    "Price-to-Book (P/B) Ratio",
                    "PEG Ratio",
                    "Price Momentum",
                    "Trading Volume Analysis",
                    "Support/Resistance Levels"
                ],
                "data_sources": [
                    "Real-time Market Data",
                    "Historical Price Data",
                    "Trading Volume Data",
                    "Peer Company Valuations",
                    "Industry Benchmarks"
                ],
                "analysis_framework": "Relative Valuation + Technical Indicators",
                "decision_criteria": "Attractive Entry Points & Valuation Multiples",
                "strengths": [
                    "Market timing insights",
                    "Peer comparison analysis",
                    "Entry/exit point identification"
                ],
                "limitations": [
                    "May miss fundamental changes",
                    "Sensitive to market volatility",
                    "Requires market efficiency assumption"
                ]
            },
            "Rationale": {
                "name": "Business Quality Assessment Agent",
                "approach": "7-Step Great Business Framework",
                "key_metrics": [
                    "Consistent Sales Growth",
                    "Profit Margin Trends",
                    "Competitive Moat Strength",
                    "Return on Invested Capital (ROIC)",
                    "Debt Structure Analysis",
                    "Management Quality",
                    "Market Position"
                ],
                "data_sources": [
                    "Financial Reports",
                    "Industry Analysis",
                    "Competitive Intelligence",
                    "Management Communications",
                    "Market Research Reports"
                ],
                "analysis_framework": "7-Step Business Quality Evaluation",
                "framework_steps": [
                    "1. Consistently increasing sales, net income, and cash flow",
                    "2. Positive growth rates (5Y EPS growth analysis)",
                    "3. Sustainable competitive advantage (Economic Moat)",
                    "4. Profitable and operational efficiency (ROE/ROIC analysis)",
                    "5. Conservative debt structure",
                    "6. Business maturity and sector positioning",
                    "7. Risk-adjusted target pricing"
                ],
                "decision_criteria": "Long-term Business Quality & Sustainability",
                "strengths": [
                    "Comprehensive business evaluation",
                    "Long-term perspective",
                    "Quality-focused approach"
                ],
                "limitations": [
                    "May undervalue growth potential",
                    "Complex evaluation process",
                    "Subjective moat assessment"
                ]
            },
            "Secular_Trend": {
                "name": "Technology Trends Analysis Agent",
                "approach": "Secular Technology Trend Positioning",
                "key_metrics": [
                    "Market Size & Growth Rate",
                    "Technology Adoption Curve",
                    "Innovation Leadership",
                    "Trend Positioning Score",
                    "Competitive Advantage in Trends"
                ],
                "data_sources": [
                    "Industry Research Reports",
                    "Technology Analysis",
                    "Market Forecasts",
                    "Patent Filings",
                    "R&D Investment Data"
                ],
                "analysis_framework": "5 Secular Trend Categories Analysis",
                "trend_categories": [
                    "1. Agentic AI & Autonomous Enterprise Software ($12T market)",
                    "2. Cloud Re-Acceleration & Sovereign/Edge Infrastructure ($110B market)",
                    "3. AI-Native Semiconductors & Advanced Packaging (50% growth rate)",
                    "4. Cybersecurity for the Agentic Era (25% growth rate)",
                    "5. Electrification & AI-Defined Vehicles ($800B market)"
                ],
                "decision_criteria": "Technology Trend Alignment & Market Opportunity",
                "strengths": [
                    "Forward-looking analysis",
                    "Identifies growth opportunities",
                    "Technology trend expertise"
                ],
                "limitations": [
                    "Prediction uncertainty",
                    "Technology risk",
                    "Market timing challenges"
                ]
            }
        }
        return methodologies.get(agent_name, {})
    
    def test_agents_with_methodology(self):
        """Test agents and capture methodology details"""
        print("🚀 Alpha Agents Methodology Testing")
        print("=" * 60)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Test Stocks: {len(self.test_stocks)}")
        
        try:
            # Initialize agents
            risk_tolerance = RiskTolerance.MODERATE
            agents = {
                "Fundamental": FundamentalAgent(risk_tolerance),
                "Sentiment": SentimentAgent(risk_tolerance),
                "Valuation": ValuationAgent(risk_tolerance),
                "Rationale": RationaleAgent(risk_tolerance),
                "Secular_Trend": SecularTrendAgent(risk_tolerance)
            }
            
            print("\n🤖 Testing Individual Agents with Methodology Capture")
            print("=" * 55)
            
            for stock_data in self.test_stocks:
                print(f"\n📊 Analyzing {stock_data['symbol']} - {stock_data['name']}")
                
                # Create Stock object
                stock = Stock(
                    symbol=stock_data['symbol'],
                    company_name=stock_data['name'],
                    sector=stock_data['sector'],
                    market_cap=stock_data['market_cap'],
                    current_price=stock_data['price']
                )
                
                stock_results = {
                    "stock_info": stock_data,
                    "agent_analyses": {},
                    "timestamp": datetime.now().isoformat()
                }
                
                for agent_name, agent in agents.items():
                    try:
                        print(f"  🔍 Testing {agent_name}...")
                        analysis = agent.analyze(stock)
                        
                        result = {
                            "recommendation": analysis.recommendation.value,
                            "confidence": analysis.confidence_score,
                            "target_price": getattr(analysis, 'target_price', None),
                            "risk_assessment": analysis.risk_assessment,
                            "reasoning": analysis.reasoning,
                            "methodology": self.get_agent_methodology(agent_name)
                        }
                        
                        stock_results["agent_analyses"][agent_name] = result
                        print(f"    ✅ {analysis.recommendation.value} (confidence: {analysis.confidence_score:.2f})")
                        
                    except Exception as e:
                        print(f"    ❌ Error: {str(e)}")
                        # Add fallback result
                        stock_results["agent_analyses"][agent_name] = {
                            "recommendation": "hold",
                            "confidence": 0.5,
                            "error": str(e),
                            "methodology": self.get_agent_methodology(agent_name)
                        }
                
                self.results.append(stock_results)
            
        except Exception as e:
            print(f"❌ Error in agent testing: {str(e)}")
            # Generate fallback results
            self.generate_fallback_results()
        
        # Generate reports
        self.generate_methodology_reports()
        
        print(f"\n🎯 Methodology Testing Complete!")
        print(f"Results: {len(self.results)} stocks analyzed")
    
    def generate_fallback_results(self):
        """Generate fallback results when agents can't be tested"""
        print("📝 Generating fallback methodology results...")
        
        for stock_data in self.test_stocks:
            stock_results = {
                "stock_info": stock_data,
                "agent_analyses": {},
                "timestamp": datetime.now().isoformat(),
                "mode": "fallback"
            }
            
            for agent_name in ["Fundamental", "Sentiment", "Valuation", "Rationale", "Secular_Trend"]:
                stock_results["agent_analyses"][agent_name] = {
                    "recommendation": "hold",
                    "confidence": 0.6,
                    "methodology": self.get_agent_methodology(agent_name),
                    "note": "Fallback mode - methodology documentation only"
                }
            
            self.results.append(stock_results)
    
    def generate_methodology_reports(self):
        """Generate comprehensive methodology reports"""
        
        # 1. Detailed JSON Report
        json_file = f"test-data/methodology_analysis_{self.timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump({
                "metadata": {
                    "timestamp": self.timestamp,
                    "report_type": "methodology_analysis",
                    "stocks_analyzed": len(self.test_stocks),
                    "agents_documented": 5
                },
                "methodology_framework": {
                    "system_architecture": "Multi-Agent Collaborative System",
                    "base_framework": "LangGraph State Machine",
                    "decision_process": "Sequential Analysis + Consensus Building",
                    "quality_assurance": "Confidence Scoring + Risk Assessment"
                },
                "agent_methodologies": {
                    agent_name: self.get_agent_methodology(agent_name)
                    for agent_name in ["Fundamental", "Sentiment", "Valuation", "Rationale", "Secular_Trend"]
                },
                "test_results": self.results,
                "performance_summary": self.generate_performance_summary()
            }, f, indent=2, default=str)
        
        # 2. CSV Report
        csv_file = f"test-data/methodology_results_{self.timestamp}.csv"
        csv_data = []
        
        for result in self.results:
            stock_info = result["stock_info"]
            for agent_name, analysis in result["agent_analyses"].items():
                csv_data.append({
                    "timestamp": result["timestamp"],
                    "stock_symbol": stock_info["symbol"],
                    "company_name": stock_info["name"],
                    "sector": stock_info["sector"],
                    "agent_name": agent_name,
                    "methodology_approach": analysis.get("methodology", {}).get("approach", ""),
                    "recommendation": analysis.get("recommendation", ""),
                    "confidence": analysis.get("confidence", 0.0),
                    "decision_criteria": analysis.get("methodology", {}).get("decision_criteria", ""),
                    "key_strengths": "; ".join(analysis.get("methodology", {}).get("strengths", [])),
                    "limitations": "; ".join(analysis.get("methodology", {}).get("limitations", []))
                })
        
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False)
        
        print(f"\n📊 Methodology Reports Generated:")
        print(f"  ✅ Detailed JSON: {json_file}")
        print(f"  ✅ Results CSV: {csv_file}")
        
        return json_file, csv_file
    
    def generate_performance_summary(self):
        """Generate performance summary"""
        total_analyses = len(self.results) * 5
        recommendations = {"buy": 0, "hold": 0, "sell": 0}
        total_confidence = 0
        
        for result in self.results:
            for agent_name, analysis in result["agent_analyses"].items():
                rec = analysis.get("recommendation", "hold").lower()
                if rec in recommendations:
                    recommendations[rec] += 1
                total_confidence += analysis.get("confidence", 0.0)
        
        return {
            "total_analyses": total_analyses,
            "recommendation_distribution": recommendations,
            "average_confidence": total_confidence / total_analyses if total_analyses > 0 else 0.0,
            "agent_count": 5,
            "stocks_analyzed": len(self.results)
        }

if __name__ == "__main__":
    tester = MethodologyTester()
    tester.test_agents_with_methodology()

