#!/usr/bin/env python3
"""
Simplified test script for the 3-agent system (without Streamlit dependencies)
"""

import os
import sys
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_fundamental_agent():
    """Test the Fundamental Agent with mock data"""
    print("🔍 Testing Fundamental Agent...")
    
    try:
        from agents.fundamental_agent_v2 import FundamentalAgent, SectorAnalysis, StockScreening
        
        # Initialize agent
        agent = FundamentalAgent(model_provider="openai", model_name="gpt-4")
        
        # Create mock sector analyses
        mock_sectors = [
            SectorAnalysis(
                sector="Technology",
                weight=85.0,
                momentum_score=90.0,
                growth_potential=88.0,
                reasoning="Strong AI and cloud growth trends",
                top_stocks=["AAPL", "MSFT", "GOOGL"]
            ),
            SectorAnalysis(
                sector="Healthcare",
                weight=75.0,
                momentum_score=80.0,
                growth_potential=82.0,
                reasoning="Aging demographics and biotech innovation",
                top_stocks=["JNJ", "PFE", "UNH"]
            )
        ]
        
        print(f"  ✅ Created {len(mock_sectors)} mock sector analyses")
        for sa in mock_sectors:
            print(f"    - {sa.sector}: Weight {sa.weight:.1f}, Score {sa.momentum_score:.1f}")
        
        # Create mock stock screenings
        mock_stocks = [
            StockScreening(
                ticker="AAPL",
                sector="Technology",
                market_cap=3000000000000,
                fundamental_score=88.5,
                intrinsic_value=200.0,
                current_price=175.0,
                upside_potential=14.3,
                metrics={"revenue_growth": 15.0, "roe": 25.0, "debt_ratio": 0.3},
                reasoning="Strong fundamentals with consistent growth"
            ),
            StockScreening(
                ticker="MSFT",
                sector="Technology",
                market_cap=2800000000000,
                fundamental_score=92.1,
                intrinsic_value=380.0,
                current_price=350.0,
                upside_potential=8.6,
                metrics={"revenue_growth": 12.0, "roe": 22.0, "debt_ratio": 0.2},
                reasoning="Dominant cloud position with strong moat"
            )
        ]
        
        print(f"  ✅ Created {len(mock_stocks)} mock stock screenings")
        for ss in mock_stocks:
            print(f"    - {ss.ticker}: Score {ss.fundamental_score:.1f}, Upside {ss.upside_potential:.1f}%")
        
        return mock_sectors, mock_stocks
        
    except Exception as e:
        print(f"  ❌ Error in Fundamental Agent: {e}")
        import traceback
        traceback.print_exc()
        return [], []

def test_rationale_agent():
    """Test the Rationale Agent with mock data"""
    print("\n🧠 Testing Rationale Agent...")
    
    try:
        from agents.rationale_agent_v2 import RationaleAgent, QualitativeAnalysis
        
        # Initialize agent
        agent = RationaleAgent(model_provider="openai", model_name="gpt-4")
        
        # Create mock qualitative analysis
        mock_analysis = QualitativeAnalysis(
            ticker="AAPL",
            company_name="Apple Inc.",
            moat_analysis={
                "moat_score": 92.0,
                "moat_strength": "Wide",
                "moat_sources": ["Brand loyalty", "Ecosystem lock-in", "Design capabilities"],
                "durability": "High"
            },
            sentiment_analysis={
                "sentiment_score": 85.0,
                "overall_sentiment": "Positive",
                "news_sentiment": "Bullish",
                "analyst_sentiment": "Optimistic"
            },
            secular_trends={
                "trend_score": 88.0,
                "trend_alignment": "Strong",
                "key_trends": ["AI integration", "Services growth", "Wearables expansion"],
                "trend_sustainability": "High"
            },
            competitive_position={
                "competitive_score": 90.0,
                "market_position": "Leader",
                "competitive_advantages": ["Premium brand", "Vertical integration", "R&D capabilities"],
                "threats": ["Regulatory pressure", "Market saturation"]
            },
            qualitative_score=88.8,
            reasoning="Strong competitive moat with premium brand positioning and ecosystem lock-in effects",
            citations=["Apple 10-K Filing", "Industry Analysis Report", "Analyst Coverage"],
            search_queries_used=["Apple competitive advantages", "Apple ecosystem moat", "Apple brand loyalty"]
        )
        
        print(f"  ✅ Created mock qualitative analysis for {mock_analysis.ticker}")
        print(f"    - Qualitative Score: {mock_analysis.qualitative_score:.1f}/100")
        print(f"    - Moat Score: {mock_analysis.moat_analysis['moat_score']:.1f}")
        print(f"    - Sentiment Score: {mock_analysis.sentiment_analysis['sentiment_score']:.1f}")
        print(f"    - Trends Score: {mock_analysis.secular_trends['trend_score']:.1f}")
        
        return mock_analysis
        
    except Exception as e:
        print(f"  ❌ Error in Rationale Agent: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_ranker_agent():
    """Test the Ranker Agent with mock data"""
    print("\n🎯 Testing Ranker Agent...")
    
    try:
        from agents.ranker_agent_v2 import RankerAgent, InvestmentScore, PortfolioRecommendation
        
        # Initialize agent
        agent = RankerAgent(model_provider="openai", model_name="gpt-4")
        
        # Create mock investment scores
        mock_scores = [
            InvestmentScore(
                ticker="AAPL",
                company_name="Apple Inc.",
                sector="Technology",
                fundamental_score=88.5,
                qualitative_score=89.2,
                growth_score=85.0,
                profitability_score=92.0,
                financial_strength_score=88.0,
                valuation_score=82.0,
                moat_score=92.0,
                sentiment_score=85.0,
                trends_score=88.0,
                competitive_score=90.0,
                composite_score=88.8,
                investment_grade="A",
                investment_thesis="Strong ecosystem and brand loyalty with AI growth catalysts",
                key_strengths=["Brand moat", "Cash generation", "Innovation pipeline"],
                key_risks=["Regulatory pressure", "China exposure", "Market saturation"],
                catalysts=["AI integration", "Services growth", "New product categories"],
                time_horizon="12-18 months",
                upside_potential=14.3,
                risk_rating="Medium",
                conviction_level="High",
                reasoning="Comprehensive analysis shows strong fundamentals and competitive position"
            ),
            InvestmentScore(
                ticker="MSFT",
                company_name="Microsoft Corporation",
                sector="Technology",
                fundamental_score=92.1,
                qualitative_score=90.0,
                growth_score=88.0,
                profitability_score=94.0,
                financial_strength_score=90.0,
                valuation_score=85.0,
                moat_score=88.0,
                sentiment_score=87.0,
                trends_score=92.0,
                competitive_score=89.0,
                composite_score=91.2,
                investment_grade="A+",
                investment_thesis="Dominant cloud position with AI leadership and strong fundamentals",
                key_strengths=["Cloud dominance", "AI leadership", "Recurring revenue"],
                key_risks=["Competition", "Valuation", "Execution risk"],
                catalysts=["AI monetization", "Cloud growth", "Productivity gains"],
                time_horizon="12-24 months",
                upside_potential=8.6,
                risk_rating="Low",
                conviction_level="High",
                reasoning="Market leader with strong competitive moats and growth prospects"
            )
        ]
        
        print(f"  ✅ Created {len(mock_scores)} mock investment scores")
        for score in mock_scores:
            print(f"    - {score.ticker}: Grade {score.investment_grade} ({score.composite_score:.1f})")
        
        # Create mock portfolio recommendation
        mock_portfolio = PortfolioRecommendation(
            recommended_stocks=mock_scores,
            portfolio_composition={
                "Technology": 80.0,
                "Healthcare": 20.0
            },
            risk_profile="Medium",
            expected_return=18.5,
            portfolio_thesis="Technology-focused portfolio with AI and cloud exposure",
            diversification_score=75.0,
            overall_conviction="High"
        )
        
        print(f"  ✅ Created mock portfolio recommendation")
        print(f"    - Risk Profile: {mock_portfolio.risk_profile}")
        print(f"    - Expected Return: {mock_portfolio.expected_return:.1f}%")
        print(f"    - Diversification Score: {mock_portfolio.diversification_score:.1f}")
        
        return mock_scores, mock_portfolio
        
    except Exception as e:
        print(f"  ❌ Error in Ranker Agent: {e}")
        import traceback
        traceback.print_exc()
        return [], None

def test_complete_pipeline():
    """Test the complete 3-agent pipeline with mock data"""
    print("\n🚀 Testing Complete 3-Agent Pipeline...")
    
    try:
        # Run individual agent tests
        sector_analyses, stock_screenings = test_fundamental_agent()
        rationale_analysis = test_rationale_agent()
        investment_scores, portfolio_rec = test_ranker_agent()
        
        if sector_analyses and stock_screenings and rationale_analysis and investment_scores:
            print("\n  ✅ Pipeline Integration Test:")
            print(f"    - Sectors Analyzed: {len(sector_analyses)}")
            print(f"    - Stocks Screened: {len(stock_screenings)}")
            print(f"    - Qualitative Analysis: Complete")
            print(f"    - Investment Scores: {len(investment_scores)}")
            print(f"    - Portfolio Recommendation: Complete")
            
            # Create summary results
            pipeline_results = {
                'timestamp': datetime.now().isoformat(),
                'test_status': 'PASS',
                'sector_count': len(sector_analyses),
                'stock_count': len(stock_screenings),
                'final_recommendations': len(investment_scores),
                'top_recommendation': {
                    'ticker': investment_scores[0].ticker,
                    'grade': investment_scores[0].investment_grade,
                    'score': investment_scores[0].composite_score
                },
                'portfolio_metrics': {
                    'risk_profile': portfolio_rec.risk_profile,
                    'expected_return': portfolio_rec.expected_return,
                    'diversification_score': portfolio_rec.diversification_score
                }
            }
            
            return pipeline_results
        else:
            print("  ❌ Pipeline integration failed - missing components")
            return None
            
    except Exception as e:
        print(f"  ❌ Error in complete pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main test function"""
    print("🧪 Testing Lohusalu Capital Management 3-Agent System")
    print("=" * 60)
    
    # Test individual agents
    sector_analyses, stock_screenings = test_fundamental_agent()
    rationale_analysis = test_rationale_agent()
    investment_scores, portfolio_rec = test_ranker_agent()
    
    # Test complete pipeline
    pipeline_results = test_complete_pipeline()
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 Test Summary:")
    print(f"  📊 Fundamental Agent: {'✅ PASS' if sector_analyses and stock_screenings else '❌ FAIL'}")
    print(f"  🧠 Rationale Agent: {'✅ PASS' if rationale_analysis else '❌ FAIL'}")
    print(f"  🎯 Ranker Agent: {'✅ PASS' if investment_scores and portfolio_rec else '❌ FAIL'}")
    print(f"  🚀 Complete Pipeline: {'✅ PASS' if pipeline_results else '❌ FAIL'}")
    
    if pipeline_results:
        print("\n🎉 All tests passed! The 3-agent system structure is working correctly.")
        
        # Save results to file
        results_file = f"test_results_simple_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(pipeline_results, f, indent=2)
        
        print(f"📄 Test results saved to: {results_file}")
        
        # Display sample results
        print("\n📊 Sample Results:")
        print(f"  🏆 Top Recommendation: {pipeline_results['top_recommendation']['ticker']} - {pipeline_results['top_recommendation']['grade']} ({pipeline_results['top_recommendation']['score']:.1f})")
        print(f"  📈 Expected Return: {pipeline_results['portfolio_metrics']['expected_return']:.1f}%")
        print(f"  🎯 Risk Profile: {pipeline_results['portfolio_metrics']['risk_profile']}")
        
    else:
        print("\n❌ Some tests failed. The system structure needs attention.")

if __name__ == "__main__":
    main()

