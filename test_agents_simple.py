#!/usr/bin/env python3
"""
Simple test script for Alpha Agents multi-agent system without LLM.
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agents import LoanApplication, CreditRiskAgent, IncomeVerificationAgent, BehavioralAnalysisAgent, RiskTolerance

def test_agents_without_llm():
    """Test individual agents without LLM."""
    
    print("Testing Alpha Agents Individual Components")
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
        # Test Credit Risk Agent
        print("Testing Credit Risk Agent...")
        credit_agent = CreditRiskAgent(RiskTolerance.MODERATE, llm_client=None)
        credit_analysis = credit_agent.analyze(sample_application)
        
        print(f"Credit Risk Analysis:")
        print(f"  Recommendation: {credit_analysis.recommendation.value}")
        print(f"  Confidence: {credit_analysis.confidence_score:.2f}")
        print(f"  Risk Assessment: {credit_analysis.risk_assessment}")
        print(f"  Key Factors: {', '.join(credit_analysis.key_factors[:2])}")
        print()
        
        # Test Income Verification Agent
        print("Testing Income Verification Agent...")
        income_agent = IncomeVerificationAgent(RiskTolerance.MODERATE, llm_client=None)
        income_analysis = income_agent.analyze(sample_application)
        
        print(f"Income Verification Analysis:")
        print(f"  Recommendation: {income_analysis.recommendation.value}")
        print(f"  Confidence: {income_analysis.confidence_score:.2f}")
        print(f"  Risk Assessment: {income_analysis.risk_assessment}")
        print(f"  Key Factors: {', '.join(income_analysis.key_factors[:2])}")
        print()
        
        # Test Behavioral Analysis Agent
        print("Testing Behavioral Analysis Agent...")
        behavioral_agent = BehavioralAnalysisAgent(RiskTolerance.MODERATE, llm_client=None)
        behavioral_analysis = behavioral_agent.analyze(sample_application)
        
        print(f"Behavioral Analysis:")
        print(f"  Recommendation: {behavioral_analysis.recommendation.value}")
        print(f"  Confidence: {behavioral_analysis.confidence_score:.2f}")
        print(f"  Risk Assessment: {behavioral_analysis.risk_assessment}")
        print(f"  Key Factors: {', '.join(behavioral_analysis.key_factors[:2])}")
        print()
        
        print("All individual agent tests completed successfully!")
        print("The multi-agent system structure is working correctly.")
        print("LLM integration is available but not required for basic functionality.")
        
        return True
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_agents_without_llm()
    sys.exit(0 if success else 1)

