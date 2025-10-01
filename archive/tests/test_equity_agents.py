#!/usr/bin/env python3
"""
Test script for Alpha Agents equity portfolio construction system.
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agents import Stock, create_multi_agent_portfolio_system, RiskTolerance, FundamentalAgent, SentimentAgent, ValuationAgent

def test_individual_agents():
    """Test individual agents without LLM."""
    
    print("Testing Individual Equity Agents")
    print("=" * 50)
    
    # Create sample stocks
    apple = Stock(
        symbol="AAPL",
        company_name="Apple Inc.",
        sector="Technology",
        market_cap=2500e9,  # $2.5T
        current_price=150.0,
        pe_ratio=25.0,
        dividend_yield=0.005,  # 0.5%
        beta=1.2,
        volume=50000000
    )
    
    microsoft = Stock(
        symbol="MSFT",
        company_name="Microsoft Corporation",
        sector="Technology",
        market_cap=2000e9,  # $2T
        current_price=300.0,
        pe_ratio=28.0,
        dividend_yield=0.02,  # 2%
        beta=0.9,
        volume=30000000
    )
    
    print(f"Sample Stocks:")
    print(f"  {apple.symbol}: ${apple.current_price:.2f}, P/E: {apple.pe_ratio}, Market Cap: ${apple.market_cap/1e9:.1f}B")
    print(f"  {microsoft.symbol}: ${microsoft.current_price:.2f}, P/E: {microsoft.pe_ratio}, Market Cap: ${microsoft.market_cap/1e9:.1f}B")
    print()
    
    try:
        # Test Fundamental Agent
        print("Testing Fundamental Agent...")
        fundamental_agent = FundamentalAgent(RiskTolerance.MODERATE, llm_client=None)
        
        apple_fundamental = fundamental_agent.analyze(apple)
        print(f"  AAPL Fundamental Analysis:")
        print(f"    Recommendation: {apple_fundamental.recommendation.value}")
        print(f"    Confidence: {apple_fundamental.confidence_score:.2f}")
        print(f"    Risk Assessment: {apple_fundamental.risk_assessment}")
        print(f"    Target Price: ${apple_fundamental.target_price or 'N/A'}")
        print()
        
        # Test Sentiment Agent
        print("Testing Sentiment Agent...")
        sentiment_agent = SentimentAgent(RiskTolerance.MODERATE, llm_client=None)
        
        apple_sentiment = sentiment_agent.analyze(apple)
        print(f"  AAPL Sentiment Analysis:")
        print(f"    Recommendation: {apple_sentiment.recommendation.value}")
        print(f"    Confidence: {apple_sentiment.confidence_score:.2f}")
        print(f"    Risk Assessment: {apple_sentiment.risk_assessment}")
        print()
        
        # Test Valuation Agent
        print("Testing Valuation Agent...")
        valuation_agent = ValuationAgent(RiskTolerance.MODERATE, llm_client=None)
        
        apple_valuation = valuation_agent.analyze(apple)
        print(f"  AAPL Valuation Analysis:")
        print(f"    Recommendation: {apple_valuation.recommendation.value}")
        print(f"    Confidence: {apple_valuation.confidence_score:.2f}")
        print(f"    Risk Assessment: {apple_valuation.risk_assessment}")
        print()
        
        print("All individual agent tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during individual agent testing: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_portfolio_system():
    """Test the multi-agent portfolio system."""
    
    print("\nTesting Multi-Agent Portfolio System")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Get OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Warning: OPENAI_API_KEY not found. Testing with fallback analysis only.")
    
    # Create sample portfolio
    stocks = [
        Stock(
            symbol="AAPL",
            company_name="Apple Inc.",
            sector="Technology",
            market_cap=2500e9,
            current_price=150.0,
            pe_ratio=25.0,
            dividend_yield=0.005,
            beta=1.2,
            volume=50000000
        ),
        Stock(
            symbol="MSFT",
            company_name="Microsoft Corporation",
            sector="Technology",
            market_cap=2000e9,
            current_price=300.0,
            pe_ratio=28.0,
            dividend_yield=0.02,
            beta=0.9,
            volume=30000000
        ),
        Stock(
            symbol="JNJ",
            company_name="Johnson & Johnson",
            sector="Healthcare",
            market_cap=400e9,
            current_price=160.0,
            pe_ratio=15.0,
            dividend_yield=0.03,
            beta=0.7,
            volume=10000000
        )
    ]
    
    print(f"Portfolio Stocks:")
    for stock in stocks:
        print(f"  {stock.symbol} ({stock.sector}): ${stock.current_price:.2f}")
    print()
    
    try:
        # Create multi-agent system
        print("Initializing multi-agent portfolio system...")
        if api_key:
            mas = create_multi_agent_portfolio_system(
                openai_api_key=api_key,
                risk_tolerance="moderate",
                max_debate_rounds=2
            )
        else:
            # Create system without LLM for testing
            mas = create_multi_agent_portfolio_system(
                openai_api_key="dummy_key",  # Will use fallback analysis
                risk_tolerance="moderate",
                max_debate_rounds=1
            )
            # Set LLM to None for all agents to force fallback
            for agent in mas.agents.values():
                agent.llm_client = None
        
        print("Analyzing portfolio with multi-agent system...")
        print("This may take a moment as agents analyze and debate...")
        print()
        
        # Analyze the portfolio
        result = mas.analyze_stocks(stocks)
        
        # Display results
        print("PORTFOLIO ANALYSIS RESULTS")
        print("=" * 50)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            return False
        
        portfolio = result['portfolio_recommendation']
        print(f"Expected Return: {portfolio['expected_return']*100:.1f}%")
        print(f"Risk Level: {portfolio['risk_level']}")
        print(f"Diversification Score: {portfolio['diversification_score']:.2f}")
        print(f"Confidence Score: {portfolio['confidence_score']:.2f}")
        print()
        
        print("PORTFOLIO COMPOSITION")
        print("-" * 30)
        portfolio_stocks = portfolio['stocks']
        
        if portfolio_stocks:
            for stock in portfolio_stocks:
                print(f"{stock['symbol']}: {stock['target_allocation']} - {stock['rationale']}")
        else:
            print("No stocks recommended for portfolio inclusion.")
        print()
        
        print("INDIVIDUAL STOCK ANALYSES")
        print("-" * 30)
        
        analyses = result['individual_analyses']
        for analysis_key, analysis in analyses.items():
            agent_name, stock_symbol = analysis_key.split('_', 1)
            print(f"{agent_name.title()} Agent - {stock_symbol}:")
            print(f"  Recommendation: {analysis['recommendation']}")
            print(f"  Confidence: {analysis['confidence_score']:.2f}")
            print(f"  Risk: {analysis['risk_assessment']}")
            print()
        
        if result['debate_history']:
            print("DEBATE SUMMARY")
            print("-" * 30)
            print(f"Debate Rounds: {result['debate_rounds']}")
            print(f"Consensus Reached: {result['consensus_reached']}")
            print(f"Total Conflicts: {len(result['debate_history'])}")
            print()
        
        print("Portfolio system test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during portfolio system testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Alpha Agents Equity Portfolio System Test")
    print("=" * 60)
    
    # Test individual agents first
    individual_success = test_individual_agents()
    
    # Test portfolio system
    portfolio_success = test_portfolio_system()
    
    # Summary
    print("\nTEST SUMMARY")
    print("=" * 30)
    print(f"Individual Agents: {'✅ PASS' if individual_success else '❌ FAIL'}")
    print(f"Portfolio System: {'✅ PASS' if portfolio_success else '❌ FAIL'}")
    
    overall_success = individual_success and portfolio_success
    print(f"Overall Result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    sys.exit(0 if overall_success else 1)

