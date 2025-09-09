#!/usr/bin/env python3
"""
Test script for Ranker Agent
Tests investment scoring and portfolio recommendation functionality
"""

import os
import sys
import json
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from agents.ranker_agent import RankerAgent
from agents.fundamental_agent import FundamentalAgent, SectorAnalysis, StockScreening
from agents.rationale_agent import RationaleAgent, QualitativeAnalysis

def setup_logging():
    """Setup logging for test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def create_mock_fundamental_data():
    """Create mock fundamental analysis data"""
    return StockScreening(
        symbol="AAPL",
        company_name="Apple Inc.",
        sector="Technology",
        fundamental_score=85,
        intrinsic_value=185.50,
        current_price=175.25,
        upside_potential=5.8,
        financial_metrics={
            "revenue_growth": 8.2,
            "roe": 26.4,
            "roic": 18.7,
            "debt_to_equity": 0.31,
            "current_ratio": 1.05,
            "gross_margin": 43.3,
            "profit_margin": 25.1
        },
        reasoning="Strong fundamentals with consistent growth and excellent profitability metrics"
    )

def create_mock_qualitative_data():
    """Create mock qualitative analysis data"""
    return QualitativeAnalysis(
        symbol="AAPL",
        company_name="Apple Inc.",
        economic_moat={
            "strength": 92,
            "durability": 88,
            "components": {
                "brand_loyalty": 95,
                "switching_costs": 85,
                "network_effects": 80,
                "regulatory_barriers": 70
            },
            "reasoning": "Exceptional brand loyalty with strong ecosystem lock-in effects"
        },
        sentiment_analysis={
            "overall_sentiment": 78,
            "news_sentiment": 82,
            "analyst_sentiment": 75,
            "social_sentiment": 76,
            "reasoning": "Generally positive sentiment with strong earnings outlook"
        },
        secular_trends={
            "alignment_score": 85,
            "key_trends": ["AI Integration", "Services Growth", "Sustainability"],
            "trend_analysis": {
                "ai_integration": 90,
                "services_growth": 85,
                "sustainability": 80
            },
            "reasoning": "Well-positioned for AI revolution and services expansion"
        },
        competitive_position={
            "market_position": 90,
            "competitive_threats": 25,
            "innovation_capability": 88,
            "reasoning": "Dominant market position with strong innovation pipeline"
        },
        qualitative_score=86,
        confidence_level=0.85,
        reasoning="Strong competitive moats and positive secular trend alignment",
        citations=[
            {
                "source": "Reuters",
                "url": "https://example.com/news",
                "title": "Apple Reports Strong Q3 Results",
                "relevance": "earnings_analysis"
            }
        ],
        search_queries_used=[
            "Apple competitive advantages 2024",
            "iPhone market share trends",
            "Apple AI strategy analysis"
        ]
    )

def test_ranker_agent():
    """Test Ranker Agent functionality"""
    print("🧪 Testing Ranker Agent...")
    
    # Initialize agent
    agent = RankerAgent(model_provider="openai", model_name="gpt-4.1-mini")
    
    # Test 1: Agent initialization
    print("\n🎯 Test 1: Agent Initialization")
    print(f"✅ Agent created successfully")
    print(f"✅ Model Provider: {agent.model_provider}")
    print(f"✅ Model Name: {agent.model_name}")
    print(f"✅ LLM Initialized: {agent.llm is not None}")
    print(f"✅ Fundamental Weight: {agent.fundamental_weight}")
    print(f"✅ Qualitative Weight: {agent.qualitative_weight}")
    
    # Test 2: Single Stock Analysis
    print("\n📊 Test 2: Single Stock Investment Scoring")
    try:
        # Create mock data
        fundamental_data = create_mock_fundamental_data()
        qualitative_data = create_mock_qualitative_data()
        
        print(f"  📈 Testing with {fundamental_data.symbol} ({fundamental_data.company_name})")
        print(f"  📊 Fundamental Score: {fundamental_data.fundamental_score}")
        print(f"  🧠 Qualitative Score: {qualitative_data.qualitative_score}")
        
        # Analyze single stock
        investment_score = agent.analyze_stock(fundamental_data, qualitative_data)
        
        print(f"✅ Investment analysis completed")
        print(f"  🎯 Composite Score: {investment_score.composite_score}")
        print(f"  🏆 Investment Grade: {investment_score.investment_grade}")
        print(f"  💪 Conviction Level: {investment_score.conviction_level}")
        print(f"  📊 Component Scores:")
        for component, score in investment_score.component_scores.items():
            print(f"    {component}: {score}")
        print(f"  🎯 Confidence Level: {investment_score.confidence_level}")
        
    except Exception as e:
        print(f"❌ Single Stock Analysis Error: {e}")
        investment_score = None
    
    # Test 3: Investment Thesis Generation
    print("\n📝 Test 3: Investment Thesis Generation")
    try:
        if investment_score:
            thesis = agent.generate_investment_thesis(
                fundamental_data, qualitative_data, investment_score
            )
            
            print(f"✅ Investment thesis generated")
            print(f"  📋 Executive Summary: {thesis.get('executive_summary', '')[:100]}...")
            print(f"  💪 Strengths Count: {len(thesis.get('strengths', []))}")
            print(f"  ⚠️ Risks Count: {len(thesis.get('risks', []))}")
            print(f"  🚀 Near-term Catalysts: {len(thesis.get('catalysts', {}).get('near_term', []))}")
            print(f"  📈 Medium-term Catalysts: {len(thesis.get('catalysts', {}).get('medium_term', []))}")
            print(f"  🔮 Long-term Catalysts: {len(thesis.get('catalysts', {}).get('long_term', []))}")
            
        else:
            print("⚠️ Skipping thesis generation - no investment score available")
            thesis = {}
            
    except Exception as e:
        print(f"❌ Investment Thesis Error: {e}")
        thesis = {}
    
    # Test 4: Portfolio Recommendation
    print("\n🎯 Test 4: Portfolio Recommendation")
    try:
        if investment_score:
            # Create a list with our test stock for portfolio analysis
            stock_analyses = [(fundamental_data, qualitative_data, investment_score)]
            
            portfolio_rec = agent.construct_portfolio(stock_analyses)
            
            print(f"✅ Portfolio recommendation generated")
            print(f"  📊 Total Stocks: {len(portfolio_rec.stock_allocations)}")
            print(f"  🎯 Portfolio Score: {portfolio_rec.portfolio_score}")
            print(f"  ⚖️ Risk Rating: {portfolio_rec.risk_rating}")
            print(f"  📈 Expected Return: {portfolio_rec.expected_return:.1f}%")
            print(f"  📊 Sector Allocation:")
            for sector, allocation in portfolio_rec.sector_allocation.items():
                print(f"    {sector}: {allocation:.1f}%")
            
            # Display individual stock allocation
            for allocation in portfolio_rec.stock_allocations:
                print(f"  🏢 {allocation['symbol']}: {allocation['allocation']:.1f}% (Grade: {allocation['grade']})")
            
        else:
            print("⚠️ Skipping portfolio recommendation - no investment score available")
            portfolio_rec = None
            
    except Exception as e:
        print(f"❌ Portfolio Recommendation Error: {e}")
        portfolio_rec = None
    
    # Test 5: Save Analysis Trace
    print("\n💾 Test 5: Save Analysis Trace")
    try:
        if investment_score and portfolio_rec:
            trace_file = agent.save_analysis_trace(
                fundamental_data.symbol, investment_score, thesis, portfolio_rec
            )
            if trace_file:
                print(f"✅ Trace saved to: {trace_file}")
                
                # Read and display trace
                with open(trace_file, 'r') as f:
                    trace_data = json.load(f)
                
                print(f"✅ Trace contains:")
                print(f"  📊 Timestamp: {trace_data.get('timestamp')}")
                print(f"  🤖 Agent: {trace_data.get('agent')}")
                print(f"  🧠 Model: {trace_data.get('model_provider')}/{trace_data.get('model_name')}")
                print(f"  📈 Symbol: {trace_data.get('symbol')}")
                print(f"  🎯 Investment Grade: {trace_data.get('investment_score', {}).get('investment_grade')}")
                print(f"  📊 Portfolio Stocks: {len(trace_data.get('portfolio_recommendation', {}).get('stock_allocations', []))}")
                
                return trace_file, trace_data
            else:
                print("❌ Failed to save trace")
                return None, None
        else:
            print("⚠️ Skipping trace save - incomplete analysis data")
            return None, None
            
    except Exception as e:
        print(f"❌ Save Trace Error: {e}")
        return None, None

def main():
    """Main test function"""
    setup_logging()
    
    print("=" * 60)
    print("🧪 RANKER AGENT TEST SUITE")
    print("=" * 60)
    
    # Set environment variables
    os.environ.setdefault('TAVILY_API_KEY', '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')
    
    try:
        trace_file, trace_data = test_ranker_agent()
        
        print("\n" + "=" * 60)
        print("✅ RANKER AGENT TEST COMPLETED")
        print("=" * 60)
        
        if trace_file:
            print(f"📄 Full trace available at: {trace_file}")
            
        return trace_data
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()

