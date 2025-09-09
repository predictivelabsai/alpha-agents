#!/usr/bin/env python3
"""
Test script for Fundamental Agent
Tests sector analysis and stock screening functionality
"""

import os
import sys
import json
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from agents.fundamental_agent import FundamentalAgent

def setup_logging():
    """Setup logging for test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_fundamental_agent():
    """Test Fundamental Agent functionality"""
    print("🧪 Testing Fundamental Agent...")
    
    # Initialize agent
    agent = FundamentalAgent(model_provider="openai", model_name="gpt-4.1-mini")
    
    # Test 1: Agent initialization
    print("\n📊 Test 1: Agent Initialization")
    print(f"✅ Agent created successfully")
    print(f"✅ Model Provider: {agent.model_provider}")
    print(f"✅ Model Name: {agent.model_name}")
    print(f"✅ LLM Initialized: {agent.llm is not None}")
    print(f"✅ Sectors Count: {len(agent.sectors)}")
    print(f"✅ Screening Criteria: {len(agent.screening_criteria)} criteria")
    
    # Test 2: Sector Analysis
    print("\n🏢 Test 2: Sector Analysis")
    try:
        # Analyze first 3 sectors for testing
        test_sectors = agent.sectors[:3]
        agent.sectors = test_sectors
        
        sector_analyses = agent.analyze_sectors()
        
        print(f"✅ Analyzed {len(sector_analyses)} sectors")
        
        for analysis in sector_analyses:
            print(f"  📈 {analysis.sector}:")
            print(f"    Weight: {analysis.weight}")
            print(f"    Momentum: {analysis.momentum_score}")
            print(f"    Growth Potential: {analysis.growth_potential}")
            print(f"    Top Stocks: {analysis.top_stocks}")
            print(f"    Reasoning: {analysis.reasoning[:100]}...")
            
    except Exception as e:
        print(f"❌ Sector Analysis Error: {e}")
        sector_analyses = []
    
    # Test 3: Stock Screening
    print("\n📈 Test 3: Stock Screening")
    try:
        if sector_analyses:
            stock_screenings = agent.screen_stocks(sector_analyses, max_stocks=5)
            
            print(f"✅ Screened {len(stock_screenings)} stocks")
            
            for screening in stock_screenings:
                print(f"  🏢 {screening.symbol} ({screening.company_name}):")
                print(f"    Sector: {screening.sector}")
                print(f"    Fundamental Score: {screening.fundamental_score}")
                print(f"    Current Price: ${screening.current_price:.2f}")
                print(f"    Intrinsic Value: ${screening.intrinsic_value:.2f}")
                print(f"    Upside Potential: {screening.upside_potential:.1f}%")
                print(f"    Reasoning: {screening.reasoning[:100]}...")
        else:
            print("⚠️ Skipping stock screening - no sector analyses available")
            stock_screenings = []
            
    except Exception as e:
        print(f"❌ Stock Screening Error: {e}")
        stock_screenings = []
    
    # Test 4: Save Analysis Trace
    print("\n💾 Test 4: Save Analysis Trace")
    try:
        trace_file = agent.save_analysis_trace(sector_analyses, stock_screenings)
        if trace_file:
            print(f"✅ Trace saved to: {trace_file}")
            
            # Read and display trace
            with open(trace_file, 'r') as f:
                trace_data = json.load(f)
            
            print(f"✅ Trace contains:")
            print(f"  📊 Timestamp: {trace_data.get('timestamp')}")
            print(f"  🤖 Agent: {trace_data.get('agent')}")
            print(f"  🧠 Model: {trace_data.get('model_provider')}/{trace_data.get('model_name')}")
            print(f"  🏢 Sectors Analyzed: {len(trace_data.get('sector_analyses', []))}")
            print(f"  📈 Stocks Screened: {len(trace_data.get('stock_screenings', []))}")
            
            return trace_file, trace_data
        else:
            print("❌ Failed to save trace")
            return None, None
            
    except Exception as e:
        print(f"❌ Save Trace Error: {e}")
        return None, None

def main():
    """Main test function"""
    setup_logging()
    
    print("=" * 60)
    print("🧪 FUNDAMENTAL AGENT TEST SUITE")
    print("=" * 60)
    
    # Set environment variables
    os.environ.setdefault('TAVILY_API_KEY', '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')
    
    try:
        trace_file, trace_data = test_fundamental_agent()
        
        print("\n" + "=" * 60)
        print("✅ FUNDAMENTAL AGENT TEST COMPLETED")
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

