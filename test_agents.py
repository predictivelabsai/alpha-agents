#!/usr/bin/env python3
"""
Test script for Alpha Agents multi-agent system.
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agents import LoanApplication, create_multi_agent_system, RiskTolerance

def test_multi_agent_system():
    """Test the multi-agent system with a sample loan application."""
    
    # Load environment variables
    load_dotenv()
    
    # Get OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment")
        return False
    
    print("Testing Alpha Agents Multi-Agent System")
    print("=" * 50)
    
    # Create sample loan application
    sample_application = LoanApplication(
        id=1,
        applicant_name="John Doe",
        email="john.doe@email.com",
        phone="555-123-4567",
        loan_amount=25000.0,
        loan_purpose="debt consolidation",
        annual_income=75000.0,
        credit_score=720,
        employment_status="full-time",
        debt_to_income_ratio=0.35
    )
    
    print(f"Sample Application:")
    print(f"  Applicant: {sample_application.applicant_name}")
    print(f"  Loan Amount: ${sample_application.loan_amount:,.2f}")
    print(f"  Annual Income: ${sample_application.annual_income:,.2f}")
    print(f"  Credit Score: {sample_application.credit_score}")
    print(f"  Purpose: {sample_application.loan_purpose}")
    print(f"  DTI Ratio: {sample_application.debt_to_income_ratio:.1%}")
    print()
    
    try:
        # Create multi-agent system
        print("Initializing multi-agent system...")
        mas = create_multi_agent_system(
            openai_api_key=api_key,
            risk_tolerance="moderate",
            max_debate_rounds=2
        )
        
        print("Processing application through multi-agent system...")
        print("This may take a moment as agents analyze and debate...")
        print()
        
        # Process the application
        result = mas.process_application(sample_application)
        
        # Display results
        print("ANALYSIS RESULTS")
        print("=" * 50)
        
        if 'error' in result:
            print(f"Error: {result['error']}")
            return False
        
        final_decision = result['final_decision']
        print(f"Final Decision: {final_decision['decision'].upper()}")
        print(f"Confidence Score: {final_decision['confidence_score']:.2f}")
        print(f"Risk Level: {final_decision['risk_level']}")
        print(f"Reasoning: {final_decision['reasoning']}")
        print()
        
        print("INDIVIDUAL AGENT ANALYSES")
        print("-" * 30)
        
        for agent_name, analysis in result['agent_analyses'].items():
            print(f"{agent_name.replace('_', ' ').title()}:")
            recommendation = analysis['recommendation']
            if hasattr(recommendation, 'value'):
                rec_display = recommendation.value
            else:
                rec_display = str(recommendation).replace('LendingDecision.', '')
            print(f"  Recommendation: {rec_display}")
            print(f"  Confidence: {analysis['confidence_score']:.2f}")
            print(f"  Key Factors: {', '.join(analysis['key_factors'][:2])}")
            if analysis['concerns']:
                print(f"  Concerns: {', '.join(analysis['concerns'][:2])}")
            print()
        
        if result['debate_history']:
            print("DEBATE SUMMARY")
            print("-" * 30)
            print(f"Debate Rounds: {result['debate_rounds']}")
            print(f"Consensus Reached: {result['consensus_reached']}")
            print(f"Total Messages: {len(result['debate_history'])}")
            print()
        
        print("Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_multi_agent_system()
    sys.exit(0 if success else 1)

