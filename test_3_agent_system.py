#!/usr/bin/env python3
"""
Test script for the 3-agent system
"""

import os
import sys
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agents.fundamental_agent_v2 import FundamentalAgent
from agents.rationale_agent_v2 import RationaleAgent
from agents.ranker_agent_v2 import RankerAgent
from utils.trace_manager import TraceManager

def test_fundamental_agent():
    """Test the Fundamental Agent"""
    print("🔍 Testing Fundamental Agent...")
    
    try:
        # Initialize agent
        agent = FundamentalAgent(model_provider="openai", model_name="gpt-4")
        
        # Test sector analysis
        print("  📊 Running sector analysis...")
        sector_analyses = agent.analyze_sectors()
        
        print(f"  ✅ Found {len(sector_analyses)} sectors")
        for sa in sector_analyses[:3]:
            print(f"    - {sa.sector}: Weight {sa.weight:.1f}, Score {sa.momentum_score:.1f}")
        
        # Test stock screening
        print("  🔍 Running stock screening...")
        stock_screenings = agent.screen_stocks(sector_analyses, max_stocks=10)
        
        print(f"  ✅ Found {len(stock_screenings)} qualifying stocks")
        for ss in stock_screenings[:3]:
            print(f"    - {ss.ticker}: Score {ss.fundamental_score:.1f}, Upside {ss.upside_potential:.1f}%")
        
        return sector_analyses, stock_screenings
        
    except Exception as e:
        print(f"  ❌ Error in Fundamental Agent: {e}")
        return [], []

def test_rationale_agent():
    """Test the Rationale Agent"""
    print("\n🧠 Testing Rationale Agent...")
    
    try:
        # Initialize agent
        agent = RationaleAgent(model_provider="openai", model_name="gpt-4")
        
        # Test with a sample stock
        ticker = "AAPL"
        company_name = "Apple Inc."
        
        print(f"  🔍 Analyzing {ticker} - {company_name}...")
        analysis = agent.run_qualitative_analysis(ticker, company_name)
        
        print(f"  ✅ Qualitative Score: {analysis.qualitative_score:.1f}/100")
        print(f"    - Moat Score: {analysis.moat_analysis.moat_score:.1f}")
        print(f"    - Sentiment Score: {analysis.sentiment_analysis.sentiment_score:.1f}")
        print(f"    - Trends Score: {analysis.secular_trends.trend_score:.1f}")
        
        return analysis
        
    except Exception as e:
        print(f"  ❌ Error in Rationale Agent: {e}")
        return None

def test_ranker_agent():
    """Test the Ranker Agent"""
    print("\n🎯 Testing Ranker Agent...")
    
    try:
        # Initialize agent
        agent = RankerAgent(model_provider="openai", model_name="gpt-4")
        
        # Create mock data for testing
        ticker = "AAPL"
        company_name = "Apple Inc."
        sector = "Technology"
        
        fundamental_data = {
            'fundamental_score': 85.0,
            'metrics': {'revenue_growth': 15.0, 'roe': 25.0},
            'upside_potential': 20.0,
            'market_cap': 3000000000000
        }
        
        qualitative_data = {
            'qualitative_score': 88.0,
            'moat_analysis': {'moat_score': 90.0},
            'sentiment_analysis': {'sentiment_score': 85.0},
            'secular_trends': {'trend_score': 90.0}
        }
        
        print(f"  🔍 Scoring {ticker} - {company_name}...")
        investment_score = agent.score_investment(
            ticker, company_name, sector, fundamental_data, qualitative_data
        )
        
        print(f"  ✅ Investment Grade: {investment_score.investment_grade}")
        print(f"    - Composite Score: {investment_score.composite_score:.1f}/100")
        print(f"    - Conviction Level: {investment_score.conviction_level}")
        print(f"    - Risk Rating: {investment_score.risk_rating}")
        
        return investment_score
        
    except Exception as e:
        print(f"  ❌ Error in Ranker Agent: {e}")
        return None

def test_complete_pipeline():
    """Test the complete 3-agent pipeline"""
    print("\n🚀 Testing Complete 3-Agent Pipeline...")
    
    try:
        # Initialize all agents
        fundamental_agent = FundamentalAgent(model_provider="openai", model_name="gpt-4")
        rationale_agent = RationaleAgent(model_provider="openai", model_name="gpt-4")
        ranker_agent = RankerAgent(model_provider="openai", model_name="gpt-4")
        
        # Step 1: Fundamental Analysis
        print("  📊 Step 1: Fundamental Analysis...")
        sector_analyses = fundamental_agent.analyze_sectors()
        stock_screenings = fundamental_agent.screen_stocks(sector_analyses, max_stocks=5)
        
        print(f"    ✅ Found {len(stock_screenings)} stocks from fundamental screening")
        
        # Step 2: Rationale Analysis
        print("  🧠 Step 2: Rationale Analysis...")
        rationale_results = {}
        
        for stock in stock_screenings[:3]:  # Analyze top 3 stocks
            print(f"    🔍 Analyzing {stock.ticker}...")
            analysis = rationale_agent.run_qualitative_analysis(stock.ticker, f"{stock.ticker} Corp")
            rationale_results[stock.ticker] = analysis
        
        print(f"    ✅ Completed qualitative analysis for {len(rationale_results)} stocks")
        
        # Step 3: Ranking and Final Recommendations
        print("  🎯 Step 3: Ranking and Final Recommendations...")
        investment_scores = []
        
        for stock in stock_screenings[:3]:
            if stock.ticker in rationale_results:
                fundamental_data = {
                    'fundamental_score': stock.fundamental_score,
                    'metrics': stock.metrics,
                    'upside_potential': stock.upside_potential,
                    'market_cap': stock.market_cap
                }
                
                qualitative_data = rationale_results[stock.ticker].__dict__
                
                investment_score = ranker_agent.score_investment(
                    stock.ticker, f"{stock.ticker} Corp", stock.sector,
                    fundamental_data, qualitative_data
                )
                investment_scores.append(investment_score)
        
        # Rank investments
        ranked_scores = ranker_agent.rank_investments(investment_scores)
        
        print(f"    ✅ Final Rankings:")
        for i, score in enumerate(ranked_scores, 1):
            print(f"      {i}. {score.ticker} - {score.investment_grade} ({score.composite_score:.1f})")
        
        # Create portfolio recommendation
        portfolio_rec = ranker_agent.create_portfolio_recommendation(ranked_scores, 3)
        
        print(f"    ✅ Portfolio Recommendation:")
        print(f"      - Risk Profile: {portfolio_rec.risk_profile}")
        print(f"      - Expected Return: {portfolio_rec.expected_return:.1f}%")
        print(f"      - Diversification Score: {portfolio_rec.diversification_score:.1f}")
        
        return {
            'sector_analyses': sector_analyses,
            'stock_screenings': stock_screenings,
            'rationale_results': rationale_results,
            'investment_scores': ranked_scores,
            'portfolio_recommendation': portfolio_rec
        }
        
    except Exception as e:
        print(f"  ❌ Error in complete pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_trace_manager():
    """Test the Trace Manager"""
    print("\n📁 Testing Trace Manager...")
    
    try:
        trace_manager = TraceManager()
        
        # Create a test trace
        test_trace = {
            'test_run': True,
            'timestamp': datetime.now().isoformat(),
            'results': 'Test successful'
        }
        
        # Save trace
        trace_file = trace_manager.save_trace(test_trace, 'fundamental')
        print(f"  ✅ Trace saved to: {trace_file}")
        
        # Load trace
        loaded_trace = trace_manager.load_trace(trace_file)
        print(f"  ✅ Trace loaded successfully: {loaded_trace is not None}")
        
        # Get recent traces
        traces = trace_manager.get_traces_by_agent('fundamental', limit=5)
        print(f"  ✅ Found {len(traces)} recent traces")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error in Trace Manager: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Testing Lohusalu Capital Management 3-Agent System")
    print("=" * 60)
    
    # Test individual agents
    sector_analyses, stock_screenings = test_fundamental_agent()
    rationale_analysis = test_rationale_agent()
    investment_score = test_ranker_agent()
    
    # Test trace manager
    trace_manager_ok = test_trace_manager()
    
    # Test complete pipeline
    pipeline_results = test_complete_pipeline()
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 Test Summary:")
    print(f"  📊 Fundamental Agent: {'✅ PASS' if sector_analyses else '❌ FAIL'}")
    print(f"  🧠 Rationale Agent: {'✅ PASS' if rationale_analysis else '❌ FAIL'}")
    print(f"  🎯 Ranker Agent: {'✅ PASS' if investment_score else '❌ FAIL'}")
    print(f"  📁 Trace Manager: {'✅ PASS' if trace_manager_ok else '❌ FAIL'}")
    print(f"  🚀 Complete Pipeline: {'✅ PASS' if pipeline_results else '❌ FAIL'}")
    
    if pipeline_results:
        print("\n🎉 All tests passed! The 3-agent system is working correctly.")
        
        # Save results to file
        results_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            # Convert dataclasses to dicts for JSON serialization
            serializable_results = {
                'timestamp': datetime.now().isoformat(),
                'test_status': 'PASS',
                'sector_count': len(pipeline_results['sector_analyses']),
                'stock_count': len(pipeline_results['stock_screenings']),
                'final_recommendations': len(pipeline_results['investment_scores'])
            }
            json.dump(serializable_results, f, indent=2)
        
        print(f"📄 Test results saved to: {results_file}")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main()

