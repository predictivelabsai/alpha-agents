#!/usr/bin/env python3
"""
Test script for the new RationaleAgent and SecularTrendAgent.
"""

import sys
import os
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agents import Stock, RiskTolerance, RationaleAgent, SecularTrendAgent
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

def test_new_agents():
    """Test the new RationaleAgent and SecularTrendAgent."""
    
    # Get API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ OpenAI API key not found")
        return False
    
    # Initialize LLM client
    llm = ChatOpenAI(
        api_key=api_key,
        model="gpt-4o-mini",
        temperature=0.3
    )
    
    # Create test stock
    test_stock = Stock(
        symbol="AAPL",
        company_name="Apple Inc.",
        sector="Technology",
        current_price=150.0,
        market_cap=2500e9,  # $2.5T
        pe_ratio=25.0,
        dividend_yield=0.5,
        beta=1.2,
        volume=50000000
    )
    
    print("🧪 Testing New Alpha Agents")
    print("=" * 50)
    print(f"Test Stock: {test_stock.company_name} ({test_stock.symbol})")
    print(f"Sector: {test_stock.sector}")
    print(f"Market Cap: ${test_stock.market_cap/1e9:.1f}B")
    print()
    
    # Test RationaleAgent
    print("🧠 Testing RationaleAgent...")
    try:
        rationale_agent = RationaleAgent(RiskTolerance.MODERATE, llm)
        rationale_analysis = rationale_agent.analyze(test_stock)
        
        print(f"✅ RationaleAgent Analysis Complete")
        print(f"   Recommendation: {rationale_analysis.recommendation.value}")
        print(f"   Confidence: {rationale_analysis.confidence_score:.2f}")
        print(f"   Risk Assessment: {rationale_analysis.risk_assessment}")
        print(f"   Key Factors: {len(rationale_analysis.key_factors)} identified")
        print(f"   Concerns: {len(rationale_analysis.concerns)} identified")
        if rationale_analysis.target_price:
            print(f"   Target Price: ${rationale_analysis.target_price:.2f}")
        print()
        
    except Exception as e:
        print(f"❌ RationaleAgent Error: {e}")
        return False
    
    # Test SecularTrendAgent
    print("🚀 Testing SecularTrendAgent...")
    try:
        trend_agent = SecularTrendAgent(RiskTolerance.MODERATE, llm)
        trend_analysis = trend_agent.analyze(test_stock)
        
        print(f"✅ SecularTrendAgent Analysis Complete")
        print(f"   Recommendation: {trend_analysis.recommendation.value}")
        print(f"   Confidence: {trend_analysis.confidence_score:.2f}")
        print(f"   Risk Assessment: {trend_analysis.risk_assessment}")
        print(f"   Key Factors: {len(trend_analysis.key_factors)} identified")
        print(f"   Concerns: {len(trend_analysis.concerns)} identified")
        if trend_analysis.target_price:
            print(f"   Target Price: ${trend_analysis.target_price:.2f}")
        print()
        
    except Exception as e:
        print(f"❌ SecularTrendAgent Error: {e}")
        return False
    
    # Test without LLM (fallback mode)
    print("🔄 Testing Fallback Mode (No LLM)...")
    try:
        rationale_agent_fallback = RationaleAgent(RiskTolerance.MODERATE, None)
        trend_agent_fallback = SecularTrendAgent(RiskTolerance.MODERATE, None)
        
        rationale_fallback = rationale_agent_fallback.analyze(test_stock)
        trend_fallback = trend_agent_fallback.analyze(test_stock)
        
        print(f"✅ Fallback Mode Working")
        print(f"   RationaleAgent (Fallback): {rationale_fallback.recommendation.value}")
        print(f"   SecularTrendAgent (Fallback): {trend_fallback.recommendation.value}")
        print()
        
    except Exception as e:
        print(f"❌ Fallback Mode Error: {e}")
        return False
    
    # Test multi-agent system integration
    print("🤝 Testing Multi-Agent System Integration...")
    try:
        from agents import create_multi_agent_portfolio_system
        
        mas = create_multi_agent_portfolio_system(
            openai_api_key=api_key,
            risk_tolerance="moderate",
            max_debate_rounds=1
        )
        
        print(f"✅ Multi-Agent System Created")
        print(f"   Total Agents: {len(mas.agents)}")
        print(f"   Agent Names: {list(mas.agents.keys())}")
        
        # Verify new agents are included
        if 'rationale' in mas.agents and 'secular_trend' in mas.agents:
            print(f"✅ New agents successfully integrated")
        else:
            print(f"❌ New agents not found in system")
            return False
        
    except Exception as e:
        print(f"❌ Multi-Agent System Error: {e}")
        return False
    
    print("🎉 All tests passed! New agents are working correctly.")
    return True

if __name__ == "__main__":
    success = test_new_agents()
    sys.exit(0 if success else 1)

