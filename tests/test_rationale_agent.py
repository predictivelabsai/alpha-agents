#!/usr/bin/env python3
"""
Test script for Rationale Agent
Tests qualitative analysis with web search functionality
"""

import os
import sys
import json
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from agents.rationale_agent import RationaleAgent

def setup_logging():
    """Setup logging for test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_rationale_agent():
    """Test Rationale Agent functionality"""
    print("🧪 Testing Rationale Agent...")
    
    # Initialize agent
    agent = RationaleAgent(model_provider="openai", model_name="gpt-4.1-mini")
    
    # Test 1: Agent initialization
    print("\n🔍 Test 1: Agent Initialization")
    print(f"✅ Agent created successfully")
    print(f"✅ Model Provider: {agent.model_provider}")
    print(f"✅ Model Name: {agent.model_name}")
    print(f"✅ LLM Initialized: {agent.llm is not None}")
    print(f"✅ Tavily Client Initialized: {agent.tavily_client is not None}")
    
    # Test 2: Individual Analysis Components
    print("\n🧠 Test 2: Individual Analysis Components")
    test_symbol = "AAPL"
    
    try:
        # Test economic moat analysis
        print(f"  📊 Testing Economic Moat Analysis for {test_symbol}...")
        moat_analysis = agent.analyze_economic_moat(test_symbol)
        print(f"    ✅ Moat Strength: {moat_analysis.get('strength', 'N/A')}")
        print(f"    ✅ Durability: {moat_analysis.get('durability', 'N/A')}")
        print(f"    ✅ Components: {list(moat_analysis.get('components', {}).keys())}")
        
    except Exception as e:
        print(f"    ❌ Economic Moat Analysis Error: {e}")
        moat_analysis = {}
    
    try:
        # Test sentiment analysis
        print(f"  📰 Testing Sentiment Analysis for {test_symbol}...")
        sentiment_analysis = agent.analyze_sentiment(test_symbol)
        print(f"    ✅ Overall Sentiment: {sentiment_analysis.get('overall_sentiment', 'N/A')}")
        print(f"    ✅ News Sentiment: {sentiment_analysis.get('news_sentiment', 'N/A')}")
        print(f"    ✅ Analyst Sentiment: {sentiment_analysis.get('analyst_sentiment', 'N/A')}")
        
    except Exception as e:
        print(f"    ❌ Sentiment Analysis Error: {e}")
        sentiment_analysis = {}
    
    try:
        # Test secular trends analysis
        print(f"  🚀 Testing Secular Trends Analysis for {test_symbol}...")
        trends_analysis = agent.analyze_secular_trends(test_symbol)
        print(f"    ✅ Alignment Score: {trends_analysis.get('alignment_score', 'N/A')}")
        print(f"    ✅ Key Trends: {trends_analysis.get('key_trends', [])}")
        
    except Exception as e:
        print(f"    ❌ Secular Trends Analysis Error: {e}")
        trends_analysis = {}
    
    # Test 3: Comprehensive Analysis
    print("\n🎯 Test 3: Comprehensive Qualitative Analysis")
    try:
        analysis = agent.analyze_stock(test_symbol)
        
        print(f"✅ Comprehensive analysis completed for {test_symbol}")
        print(f"  📊 Qualitative Score: {analysis.qualitative_score}")
        print(f"  🎯 Confidence Level: {analysis.confidence_level}")
        print(f"  🔍 Citations Count: {len(analysis.citations)}")
        print(f"  🔎 Search Queries Used: {len(analysis.search_queries_used)}")
        
        # Display key components
        if hasattr(analysis, 'economic_moat'):
            print(f"  🏰 Economic Moat Strength: {analysis.economic_moat.get('strength', 'N/A')}")
        if hasattr(analysis, 'sentiment_analysis'):
            print(f"  📈 Overall Sentiment: {analysis.sentiment_analysis.get('overall_sentiment', 'N/A')}")
        if hasattr(analysis, 'secular_trends'):
            print(f"  🚀 Trend Alignment: {analysis.secular_trends.get('alignment_score', 'N/A')}")
        if hasattr(analysis, 'competitive_position'):
            print(f"  🥇 Market Position: {analysis.competitive_position.get('market_position', 'N/A')}")
        
        print(f"  💭 Reasoning: {analysis.reasoning[:150]}...")
        
    except Exception as e:
        print(f"❌ Comprehensive Analysis Error: {e}")
        analysis = None
    
    # Test 4: Save Analysis Trace
    print("\n💾 Test 4: Save Analysis Trace")
    try:
        if analysis:
            trace_file = agent.save_analysis_trace(test_symbol, analysis)
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
                print(f"  🔍 Citations: {len(trace_data.get('analysis', {}).get('citations', []))}")
                
                return trace_file, trace_data
            else:
                print("❌ Failed to save trace")
                return None, None
        else:
            print("⚠️ Skipping trace save - no analysis available")
            return None, None
            
    except Exception as e:
        print(f"❌ Save Trace Error: {e}")
        return None, None

def main():
    """Main test function"""
    setup_logging()
    
    print("=" * 60)
    print("🧪 RATIONALE AGENT TEST SUITE")
    print("=" * 60)
    
    # Set environment variables
    os.environ.setdefault('TAVILY_API_KEY', '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')
    
    try:
        trace_file, trace_data = test_rationale_agent()
        
        print("\n" + "=" * 60)
        print("✅ RATIONALE AGENT TEST COMPLETED")
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

