"""
Income Verification Agent for consumer lending analysis.
Adapted from the Sentiment Agent in the Alpha Agents paper.
"""

from typing import Dict, List, Any
import json
import re
from datetime import datetime
from langchain_core.messages import HumanMessage
from .base_agent import BaseAgent, LoanApplication, AgentAnalysis, RiskTolerance, LendingDecision

class IncomeVerificationAgent(BaseAgent):
    """
    Income Verification Agent specializes in analyzing income sources, employment verification,
    and financial documentation. Equivalent to the Sentiment Agent in the original Alpha Agents paper.
    """
    
    def __init__(self, risk_tolerance: RiskTolerance = RiskTolerance.MODERATE, llm_client=None):
        super().__init__("IncomeVerification", risk_tolerance, llm_client)
        
    def get_role_prompt(self) -> str:
        """Get the role-specific prompt for the Income Verification Agent."""
        return """
        You are an Income Verification Specialist for consumer lending. Your primary responsibility 
        is to analyze and verify income sources, employment status, and financial documentation 
        authenticity. You focus on the reliability and sustainability of the applicant's income.
        
        Your analysis should focus on:
        1. Income source verification and reliability
        2. Employment history and job stability
        3. Income trend analysis and growth patterns
        4. Documentation completeness and authenticity
        5. Alternative income sources evaluation
        6. Seasonal or variable income considerations
        
        Provide recommendations based on income verification standards and employment stability metrics.
        Consider the sustainability and reliability of income for loan repayment capacity.
        
        Always structure your response with:
        - Recommendation: APPROVE/REJECT/CONDITIONAL/REVIEW
        - Confidence: Score from 0.0 to 1.0
        - Risk: LOW/MODERATE/HIGH
        - Reasoning: Detailed explanation
        - Key Factors: List of supporting factors
        - Concerns: List of verification issues or concerns
        """
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get the tools available to the Income Verification Agent."""
        return [
            {
                "name": "income_analyzer",
                "description": "Analyze income patterns and sustainability",
                "function": self.analyze_income_patterns
            },
            {
                "name": "employment_verifier",
                "description": "Verify employment status and history",
                "function": self.verify_employment
            },
            {
                "name": "documentation_checker",
                "description": "Check documentation completeness and authenticity",
                "function": self.check_documentation
            },
            {
                "name": "income_ratio_calculator",
                "description": "Calculate income-to-loan ratios and affordability",
                "function": self.calculate_income_ratios
            }
        ]
    
    def analyze_income_patterns(self, annual_income: float, employment_status: str) -> Dict[str, Any]:
        """Analyze income patterns and sustainability."""
        income_stability = "STABLE"
        sustainability_score = 0.7
        
        # Analyze income level
        if annual_income >= 100000:
            income_tier = "HIGH"
            tier_score = 0.9
        elif annual_income >= 60000:
            income_tier = "UPPER_MIDDLE"
            tier_score = 0.8
        elif annual_income >= 40000:
            income_tier = "MIDDLE"
            tier_score = 0.7
        elif annual_income >= 25000:
            income_tier = "LOWER_MIDDLE"
            tier_score = 0.6
        else:
            income_tier = "LOW"
            tier_score = 0.4
        
        # Adjust based on employment type
        if employment_status.lower() in ["self-employed", "freelance", "contract"]:
            income_stability = "VARIABLE"
            sustainability_score *= 0.8
        elif employment_status.lower() in ["part-time"]:
            income_stability = "LIMITED"
            sustainability_score *= 0.9
        elif employment_status.lower() in ["unemployed"]:
            income_stability = "NONE"
            sustainability_score = 0.1
        
        return {
            "annual_income": annual_income,
            "income_tier": income_tier,
            "tier_score": tier_score,
            "income_stability": income_stability,
            "sustainability_score": sustainability_score,
            "analysis": f"Income of ${annual_income:,.2f} classified as {income_tier} with {income_stability} stability."
        }
    
    def verify_employment(self, employment_status: str, annual_income: float) -> Dict[str, Any]:
        """Verify employment status and assess job security."""
        verification_score = 0.5
        job_security = "MODERATE"
        
        # Employment status verification
        if employment_status.lower() in ["full-time", "permanent"]:
            verification_score = 0.9
            job_security = "HIGH"
            employment_risk = "LOW"
        elif employment_status.lower() in ["part-time"]:
            verification_score = 0.7
            job_security = "MODERATE"
            employment_risk = "MODERATE"
        elif employment_status.lower() in ["contract", "temporary"]:
            verification_score = 0.6
            job_security = "LOW"
            employment_risk = "HIGH"
        elif employment_status.lower() in ["self-employed", "freelance"]:
            verification_score = 0.5
            job_security = "VARIABLE"
            employment_risk = "HIGH"
        elif employment_status.lower() in ["unemployed"]:
            verification_score = 0.1
            job_security = "NONE"
            employment_risk = "VERY_HIGH"
        else:
            employment_risk = "MODERATE"
        
        # Income consistency check
        income_consistency = "CONSISTENT"
        if employment_status.lower() in ["self-employed", "freelance", "contract"]:
            income_consistency = "VARIABLE"
        elif employment_status.lower() in ["unemployed"]:
            income_consistency = "NONE"
        
        return {
            "employment_status": employment_status,
            "verification_score": verification_score,
            "job_security": job_security,
            "employment_risk": employment_risk,
            "income_consistency": income_consistency,
            "assessment": f"Employment verification shows {job_security} job security with {employment_risk} risk level."
        }
    
    def check_documentation(self, application: LoanApplication) -> Dict[str, Any]:
        """Check documentation completeness and authenticity indicators."""
        doc_score = 0.0
        completeness_issues = []
        authenticity_flags = []
        
        # Basic information completeness
        required_fields = [
            ('applicant_name', application.applicant_name),
            ('email', application.email),
            ('phone', application.phone),
            ('annual_income', application.annual_income),
            ('credit_score', application.credit_score),
            ('employment_status', application.employment_status)
        ]
        
        complete_fields = 0
        for field_name, field_value in required_fields:
            if field_value and str(field_value).strip():
                complete_fields += 1
            else:
                completeness_issues.append(f"Missing {field_name}")
        
        doc_score = complete_fields / len(required_fields)
        
        # Basic authenticity checks
        if application.email and '@' in application.email and '.' in application.email:
            doc_score += 0.1
        else:
            authenticity_flags.append("Invalid email format")
        
        if application.phone and len(re.sub(r'[^\d]', '', application.phone)) >= 10:
            doc_score += 0.1
        else:
            authenticity_flags.append("Invalid phone format")
        
        # Income reasonableness check
        if application.annual_income > 0 and application.annual_income < 1000000:
            doc_score += 0.1
        else:
            authenticity_flags.append("Unreasonable income amount")
        
        doc_score = min(1.0, doc_score)
        
        if doc_score >= 0.9:
            doc_quality = "EXCELLENT"
        elif doc_score >= 0.7:
            doc_quality = "GOOD"
        elif doc_score >= 0.5:
            doc_quality = "FAIR"
        else:
            doc_quality = "POOR"
        
        return {
            "completeness_score": doc_score,
            "documentation_quality": doc_quality,
            "completeness_issues": completeness_issues,
            "authenticity_flags": authenticity_flags,
            "assessment": f"Documentation quality rated as {doc_quality} with score {doc_score:.2f}"
        }
    
    def calculate_income_ratios(self, annual_income: float, loan_amount: float) -> Dict[str, Any]:
        """Calculate income-to-loan ratios and affordability metrics."""
        monthly_income = annual_income / 12
        
        # Loan-to-income ratio
        loan_to_income_ratio = loan_amount / annual_income
        
        # Estimated monthly payment (5% interest, 5 years)
        monthly_rate = 0.05 / 12
        num_payments = 60
        monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
        
        # Payment-to-income ratio
        payment_to_income_ratio = monthly_payment / monthly_income
        
        # Affordability assessment
        if payment_to_income_ratio <= 0.10:
            affordability = "EXCELLENT"
            affordability_score = 0.9
        elif payment_to_income_ratio <= 0.15:
            affordability = "GOOD"
            affordability_score = 0.8
        elif payment_to_income_ratio <= 0.20:
            affordability = "FAIR"
            affordability_score = 0.6
        elif payment_to_income_ratio <= 0.25:
            affordability = "POOR"
            affordability_score = 0.4
        else:
            affordability = "UNAFFORDABLE"
            affordability_score = 0.2
        
        return {
            "monthly_income": monthly_income,
            "loan_to_income_ratio": loan_to_income_ratio,
            "monthly_payment": monthly_payment,
            "payment_to_income_ratio": payment_to_income_ratio,
            "affordability": affordability,
            "affordability_score": affordability_score,
            "assessment": f"Loan affordability rated as {affordability} with {payment_to_income_ratio:.1%} payment-to-income ratio."
        }
    
    def analyze(self, application: LoanApplication) -> AgentAnalysis:
        """Analyze a loan application from income verification perspective."""
        try:
            # Use tools to analyze different aspects
            income_analysis = self.analyze_income_patterns(
                application.annual_income, 
                application.employment_status
            )
            employment_analysis = self.verify_employment(
                application.employment_status, 
                application.annual_income
            )
            documentation_analysis = self.check_documentation(application)
            ratio_analysis = self.calculate_income_ratios(
                application.annual_income, 
                application.loan_amount
            )
            
            # Create comprehensive analysis prompt
            analysis_prompt = f"""
            {self.get_role_prompt()}
            
            {self.format_application_data(application)}
            
            Income Pattern Analysis:
            {json.dumps(income_analysis, indent=2)}
            
            Employment Verification:
            {json.dumps(employment_analysis, indent=2)}
            
            Documentation Check:
            {json.dumps(documentation_analysis, indent=2)}
            
            Income Ratio Analysis:
            {json.dumps(ratio_analysis, indent=2)}
            
            Based on this comprehensive income verification analysis, provide your recommendation.
            """
            
            if self.llm_client:
                response = self.llm_client.invoke([HumanMessage(content=analysis_prompt)])
                llm_response = response.content
            else:
                # Fallback analysis without LLM
                llm_response = self._fallback_analysis(
                    income_analysis, employment_analysis, 
                    documentation_analysis, ratio_analysis
                )
            
            return self.create_analysis_result(application, llm_response)
            
        except Exception as e:
            self.logger.error(f"Error in income verification analysis: {e}")
            return AgentAnalysis(
                agent_type=self.agent_type,
                recommendation=LendingDecision.REVIEW,
                confidence_score=0.3,
                risk_assessment="UNKNOWN",
                reasoning=f"Error in analysis: {str(e)}",
                key_factors=[],
                concerns=["Analysis error occurred"],
                timestamp=datetime.now()
            )
    
    def _fallback_analysis(self, income_analysis: Dict, employment_analysis: Dict, 
                          documentation_analysis: Dict, ratio_analysis: Dict) -> str:
        """Provide fallback analysis when LLM is not available."""
        score = 0
        factors = []
        concerns = []
        
        # Income sustainability (30% weight)
        sustainability = income_analysis["sustainability_score"]
        score += sustainability * 0.3
        if sustainability >= 0.8:
            factors.append("Sustainable income source")
        elif sustainability < 0.5:
            concerns.append("Income sustainability concerns")
        
        # Employment verification (25% weight)
        employment_score = employment_analysis["verification_score"]
        score += employment_score * 0.25
        if employment_score >= 0.8:
            factors.append("Stable employment verified")
        elif employment_score < 0.5:
            concerns.append("Employment verification issues")
        
        # Documentation quality (20% weight)
        doc_score = documentation_analysis["completeness_score"]
        score += doc_score * 0.2
        if doc_score >= 0.8:
            factors.append("Complete documentation")
        elif doc_score < 0.6:
            concerns.append("Documentation deficiencies")
        
        # Affordability (25% weight)
        affordability_score = ratio_analysis["affordability_score"]
        score += affordability_score * 0.25
        if affordability_score >= 0.8:
            factors.append("Excellent loan affordability")
        elif affordability_score < 0.5:
            concerns.append("Affordability concerns")
        
        # Determine recommendation
        if score >= 0.75:
            recommendation = "APPROVE"
            risk = "LOW"
        elif score >= 0.6:
            recommendation = "CONDITIONAL"
            risk = "MODERATE"
        elif score >= 0.4:
            recommendation = "REVIEW"
            risk = "HIGH"
        else:
            recommendation = "REJECT"
            risk = "HIGH"
        
        return f"""
        Recommendation: {recommendation}
        Confidence: {score:.2f}
        Risk: {risk}
        Reasoning: Income verification analysis shows {risk.lower()} risk based on employment stability and income sustainability.
        Key Factors: {'; '.join(factors)}
        Concerns: {'; '.join(concerns) if concerns else 'None identified'}
        """

