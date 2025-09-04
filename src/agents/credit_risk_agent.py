"""
Credit Risk Agent for consumer lending analysis.
Adapted from the Fundamental Agent in the Alpha Agents paper.
"""

from typing import Dict, List, Any
import json
import math
from datetime import datetime
from langchain_core.messages import HumanMessage
from .base_agent import BaseAgent, LoanApplication, AgentAnalysis, RiskTolerance

class CreditRiskAgent(BaseAgent):
    """
    Credit Risk Agent specializes in analyzing creditworthiness and financial fundamentals.
    Equivalent to the Fundamental Agent in the original Alpha Agents paper.
    """
    
    def __init__(self, risk_tolerance: RiskTolerance = RiskTolerance.MODERATE, llm_client=None):
        super().__init__("CreditRisk", risk_tolerance, llm_client)
        
    def get_role_prompt(self) -> str:
        """Get the role-specific prompt for the Credit Risk Agent."""
        return """
        You are a Credit Risk Analyst specializing in consumer lending. Your primary responsibility 
        is to analyze the creditworthiness and financial fundamentals of loan applicants. You have 
        access to credit scores, debt-to-income ratios, employment history, and financial statements.
        
        Your analysis should focus on:
        1. Credit score evaluation and trends
        2. Debt-to-income ratio assessment
        3. Employment stability and income verification
        4. Payment history and credit utilization
        5. Overall financial health indicators
        
        Provide recommendations based on quantitative financial metrics and credit risk models.
        Consider the applicant's ability to repay the loan based on their financial profile.
        
        Always structure your response with:
        - Recommendation: APPROVE/REJECT/CONDITIONAL/REVIEW
        - Confidence: Score from 0.0 to 1.0
        - Risk: LOW/MODERATE/HIGH
        - Reasoning: Detailed explanation
        - Key Factors: List of supporting factors
        - Concerns: List of risk factors or concerns
        """
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get the tools available to the Credit Risk Agent."""
        return [
            {
                "name": "credit_score_analyzer",
                "description": "Analyze credit score and provide risk assessment",
                "function": self.analyze_credit_score
            },
            {
                "name": "debt_calculator",
                "description": "Calculate debt-to-income ratios and debt burden",
                "function": self.calculate_debt_metrics
            },
            {
                "name": "income_stability_checker",
                "description": "Assess employment and income stability",
                "function": self.assess_income_stability
            }
        ]
    
    def analyze_credit_score(self, credit_score: int) -> Dict[str, Any]:
        """Analyze credit score and provide risk assessment."""
        if credit_score >= 800:
            risk_level = "VERY_LOW"
            score_category = "Excellent"
            approval_likelihood = 0.95
        elif credit_score >= 740:
            risk_level = "LOW"
            score_category = "Very Good"
            approval_likelihood = 0.85
        elif credit_score >= 670:
            risk_level = "MODERATE"
            score_category = "Good"
            approval_likelihood = 0.70
        elif credit_score >= 580:
            risk_level = "HIGH"
            score_category = "Fair"
            approval_likelihood = 0.40
        else:
            risk_level = "VERY_HIGH"
            score_category = "Poor"
            approval_likelihood = 0.15
        
        return {
            "credit_score": credit_score,
            "risk_level": risk_level,
            "category": score_category,
            "approval_likelihood": approval_likelihood,
            "analysis": f"Credit score of {credit_score} falls in the {score_category} range with {risk_level} risk level."
        }
    
    def calculate_debt_metrics(self, annual_income: float, debt_to_income_ratio: float, loan_amount: float) -> Dict[str, Any]:
        """Calculate debt-to-income ratios and debt burden."""
        monthly_income = annual_income / 12
        current_monthly_debt = monthly_income * debt_to_income_ratio
        
        # Estimate monthly loan payment (assuming 5% interest, 5-year term)
        monthly_rate = 0.05 / 12
        num_payments = 60  # 5 years
        monthly_payment = loan_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / ((1 + monthly_rate)**num_payments - 1)
        
        new_debt_to_income = (current_monthly_debt + monthly_payment) / monthly_income
        
        # Risk assessment based on DTI
        if new_debt_to_income <= 0.28:
            dti_risk = "LOW"
            dti_assessment = "Excellent debt management"
        elif new_debt_to_income <= 0.36:
            dti_risk = "MODERATE"
            dti_assessment = "Acceptable debt levels"
        elif new_debt_to_income <= 0.43:
            dti_risk = "HIGH"
            dti_assessment = "High debt burden"
        else:
            dti_risk = "VERY_HIGH"
            dti_assessment = "Excessive debt burden"
        
        return {
            "current_dti": debt_to_income_ratio,
            "projected_dti": new_debt_to_income,
            "monthly_income": monthly_income,
            "estimated_payment": monthly_payment,
            "dti_risk": dti_risk,
            "assessment": dti_assessment
        }
    
    def assess_income_stability(self, employment_status: str, annual_income: float) -> Dict[str, Any]:
        """Assess employment and income stability."""
        stability_score = 0.5  # Default moderate stability
        
        if employment_status.lower() in ["full-time", "permanent"]:
            stability_score = 0.8
            stability_level = "HIGH"
        elif employment_status.lower() in ["part-time", "contract"]:
            stability_score = 0.6
            stability_level = "MODERATE"
        elif employment_status.lower() in ["self-employed", "freelance"]:
            stability_score = 0.4
            stability_level = "MODERATE"
        elif employment_status.lower() in ["unemployed", "retired"]:
            stability_score = 0.2
            stability_level = "LOW"
        else:
            stability_level = "MODERATE"
        
        # Adjust based on income level
        if annual_income >= 100000:
            stability_score += 0.1
        elif annual_income >= 50000:
            stability_score += 0.05
        elif annual_income < 25000:
            stability_score -= 0.1
        
        stability_score = max(0.0, min(1.0, stability_score))
        
        return {
            "employment_status": employment_status,
            "annual_income": annual_income,
            "stability_score": stability_score,
            "stability_level": stability_level,
            "assessment": f"Employment stability rated as {stability_level} based on {employment_status} status and income level."
        }
    
    def analyze(self, application: LoanApplication) -> AgentAnalysis:
        """Analyze a loan application from credit risk perspective."""
        try:
            # Use tools to analyze different aspects
            credit_analysis = self.analyze_credit_score(application.credit_score)
            debt_analysis = self.calculate_debt_metrics(
                application.annual_income, 
                application.debt_to_income_ratio, 
                application.loan_amount
            )
            income_analysis = self.assess_income_stability(
                application.employment_status, 
                application.annual_income
            )
            
            # Create comprehensive analysis prompt
            analysis_prompt = f"""
            {self.get_role_prompt()}
            
            {self.format_application_data(application)}
            
            Credit Analysis Results:
            {json.dumps(credit_analysis, indent=2)}
            
            Debt Analysis Results:
            {json.dumps(debt_analysis, indent=2)}
            
            Income Stability Analysis:
            {json.dumps(income_analysis, indent=2)}
            
            Based on this comprehensive credit risk analysis, provide your recommendation.
            """
            
            if self.llm_client:
                response = self.llm_client.invoke([HumanMessage(content=analysis_prompt)])
                llm_response = response.content
            else:
                # Fallback analysis without LLM
                llm_response = self._fallback_analysis(credit_analysis, debt_analysis, income_analysis)
            
            return self.create_analysis_result(application, llm_response)
            
        except Exception as e:
            self.logger.error(f"Error in credit risk analysis: {e}")
            return AgentAnalysis(
                agent_type=self.agent_type,
                recommendation="REVIEW",
                confidence_score=0.3,
                risk_assessment="UNKNOWN",
                reasoning=f"Error in analysis: {str(e)}",
                key_factors=[],
                concerns=["Analysis error occurred"],
                timestamp=datetime.now()
            )
    
    def _fallback_analysis(self, credit_analysis: Dict, debt_analysis: Dict, income_analysis: Dict) -> str:
        """Provide fallback analysis when LLM is not available."""
        # Simple rule-based analysis
        score = 0
        factors = []
        concerns = []
        
        # Credit score factor (40% weight)
        if credit_analysis["approval_likelihood"] >= 0.8:
            score += 0.4
            factors.append(f"Excellent credit score ({credit_analysis['credit_score']})")
        elif credit_analysis["approval_likelihood"] >= 0.6:
            score += 0.3
            factors.append(f"Good credit score ({credit_analysis['credit_score']})")
        else:
            score += 0.1
            concerns.append(f"Low credit score ({credit_analysis['credit_score']})")
        
        # DTI factor (35% weight)
        if debt_analysis["projected_dti"] <= 0.28:
            score += 0.35
            factors.append("Low debt-to-income ratio")
        elif debt_analysis["projected_dti"] <= 0.36:
            score += 0.25
            factors.append("Acceptable debt-to-income ratio")
        else:
            score += 0.1
            concerns.append("High debt-to-income ratio")
        
        # Income stability factor (25% weight)
        if income_analysis["stability_score"] >= 0.7:
            score += 0.25
            factors.append("Stable employment and income")
        elif income_analysis["stability_score"] >= 0.5:
            score += 0.15
        else:
            score += 0.05
            concerns.append("Income stability concerns")
        
        # Determine recommendation
        if score >= 0.7:
            recommendation = "APPROVE"
            risk = "LOW"
        elif score >= 0.5:
            recommendation = "CONDITIONAL"
            risk = "MODERATE"
        elif score >= 0.3:
            recommendation = "REVIEW"
            risk = "HIGH"
        else:
            recommendation = "REJECT"
            risk = "HIGH"
        
        return f"""
        Recommendation: {recommendation}
        Confidence: {score:.2f}
        Risk: {risk}
        Reasoning: Credit risk analysis based on quantitative metrics shows {risk.lower()} risk profile.
        Key Factors: {'; '.join(factors)}
        Concerns: {'; '.join(concerns) if concerns else 'None identified'}
        """

