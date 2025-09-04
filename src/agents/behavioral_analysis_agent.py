"""
Behavioral Analysis Agent for consumer lending analysis.
Adapted from the Valuation Agent in the Alpha Agents paper.
"""

from typing import Dict, List, Any
import json
import math
from datetime import datetime
from langchain_core.messages import HumanMessage
from .base_agent import BaseAgent, LoanApplication, AgentAnalysis, RiskTolerance, LendingDecision

class BehavioralAnalysisAgent(BaseAgent):
    """
    Behavioral Analysis Agent specializes in analyzing financial behavior patterns,
    spending habits, and psychological risk factors. Equivalent to the Valuation Agent 
    in the original Alpha Agents paper.
    """
    
    def __init__(self, risk_tolerance: RiskTolerance = RiskTolerance.MODERATE, llm_client=None):
        super().__init__("BehavioralAnalysis", risk_tolerance, llm_client)
        
    def get_role_prompt(self) -> str:
        """Get the role-specific prompt for the Behavioral Analysis Agent."""
        return """
        You are a Behavioral Finance Analyst specializing in consumer lending risk assessment. 
        Your primary responsibility is to analyze financial behavior patterns, spending habits, 
        and psychological factors that may impact loan repayment probability.
        
        Your analysis should focus on:
        1. Financial behavior patterns and trends
        2. Spending habit analysis and impulse control indicators
        3. Debt management history and patterns
        4. Risk-taking behavior in financial decisions
        5. Psychological factors affecting financial responsibility
        6. Behavioral red flags and positive indicators
        
        Provide recommendations based on behavioral finance principles and risk psychology.
        Consider how behavioral patterns may affect the applicant's ability and willingness 
        to repay the loan consistently.
        
        Always structure your response with:
        - Recommendation: APPROVE/REJECT/CONDITIONAL/REVIEW
        - Confidence: Score from 0.0 to 1.0
        - Risk: LOW/MODERATE/HIGH
        - Reasoning: Detailed explanation
        - Key Factors: List of positive behavioral indicators
        - Concerns: List of behavioral risk factors
        """
    
    def get_tools(self) -> List[Dict[str, Any]]:
        """Get the tools available to the Behavioral Analysis Agent."""
        return [
            {
                "name": "behavior_pattern_analyzer",
                "description": "Analyze financial behavior patterns and trends",
                "function": self.analyze_behavior_patterns
            },
            {
                "name": "risk_psychology_assessor",
                "description": "Assess psychological risk factors",
                "function": self.assess_risk_psychology
            },
            {
                "name": "spending_habit_evaluator",
                "description": "Evaluate spending habits and financial discipline",
                "function": self.evaluate_spending_habits
            },
            {
                "name": "debt_behavior_analyzer",
                "description": "Analyze debt management behavior",
                "function": self.analyze_debt_behavior
            }
        ]
    
    def analyze_behavior_patterns(self, application: LoanApplication) -> Dict[str, Any]:
        """Analyze financial behavior patterns based on available data."""
        behavior_score = 0.5
        patterns = []
        risk_indicators = []
        
        # Analyze loan purpose for behavioral insights
        purpose = application.loan_purpose.lower() if application.loan_purpose else ""
        
        if any(word in purpose for word in ["consolidation", "debt", "refinance"]):
            patterns.append("Debt consolidation seeking")
            behavior_score += 0.1  # Positive: trying to manage debt
        elif any(word in purpose for word in ["home", "education", "medical"]):
            patterns.append("Essential/investment purpose")
            behavior_score += 0.2  # Very positive: responsible borrowing
        elif any(word in purpose for word in ["vacation", "luxury", "entertainment"]):
            patterns.append("Discretionary spending")
            behavior_score -= 0.1  # Negative: non-essential borrowing
            risk_indicators.append("Borrowing for discretionary expenses")
        elif any(word in purpose for word in ["business", "investment"]):
            patterns.append("Investment/business purpose")
            behavior_score += 0.05  # Slightly positive but riskier
        
        # Analyze income vs loan amount ratio for spending behavior
        income_loan_ratio = application.annual_income / application.loan_amount if application.loan_amount > 0 else 0
        
        if income_loan_ratio >= 10:
            patterns.append("Conservative borrowing relative to income")
            behavior_score += 0.15
        elif income_loan_ratio >= 5:
            patterns.append("Moderate borrowing relative to income")
            behavior_score += 0.1
        elif income_loan_ratio >= 2:
            patterns.append("Significant borrowing relative to income")
            behavior_score -= 0.05
        else:
            patterns.append("High borrowing relative to income")
            behavior_score -= 0.15
            risk_indicators.append("Borrowing amount high relative to income")
        
        # Analyze debt-to-income for financial discipline
        if application.debt_to_income_ratio <= 0.2:
            patterns.append("Excellent debt management")
            behavior_score += 0.2
        elif application.debt_to_income_ratio <= 0.3:
            patterns.append("Good debt management")
            behavior_score += 0.1
        elif application.debt_to_income_ratio <= 0.4:
            patterns.append("Moderate debt levels")
            behavior_score += 0.0
        else:
            patterns.append("High existing debt burden")
            behavior_score -= 0.1
            risk_indicators.append("High existing debt levels")
        
        behavior_score = max(0.0, min(1.0, behavior_score))
        
        return {
            "behavior_score": behavior_score,
            "patterns_identified": patterns,
            "risk_indicators": risk_indicators,
            "income_loan_ratio": income_loan_ratio,
            "assessment": f"Behavioral analysis shows {'positive' if behavior_score > 0.6 else 'concerning' if behavior_score < 0.4 else 'mixed'} patterns."
        }
    
    def assess_risk_psychology(self, application: LoanApplication) -> Dict[str, Any]:
        """Assess psychological risk factors based on application data."""
        psychology_score = 0.5
        risk_factors = []
        positive_factors = []
        
        # Employment stability as indicator of planning and stability
        employment = application.employment_status.lower() if application.employment_status else ""
        
        if employment in ["full-time", "permanent"]:
            psychology_score += 0.2
            positive_factors.append("Stable employment indicates planning ability")
        elif employment in ["self-employed", "freelance"]:
            psychology_score += 0.1
            positive_factors.append("Self-employment shows initiative")
            risk_factors.append("Variable income may indicate higher risk tolerance")
        elif employment in ["part-time", "contract"]:
            psychology_score -= 0.05
            risk_factors.append("Employment instability may indicate poor planning")
        elif employment in ["unemployed"]:
            psychology_score -= 0.3
            risk_factors.append("Unemployment indicates financial stress")
        
        # Credit score as indicator of financial responsibility
        if application.credit_score >= 750:
            psychology_score += 0.2
            positive_factors.append("High credit score indicates financial responsibility")
        elif application.credit_score >= 650:
            psychology_score += 0.1
            positive_factors.append("Good credit score shows financial awareness")
        elif application.credit_score < 600:
            psychology_score -= 0.2
            risk_factors.append("Low credit score may indicate poor financial habits")
        
        # Loan amount relative to income as risk tolerance indicator
        loan_income_ratio = application.loan_amount / application.annual_income if application.annual_income > 0 else 1
        
        if loan_income_ratio > 0.5:
            psychology_score -= 0.15
            risk_factors.append("High loan-to-income ratio suggests high risk tolerance")
        elif loan_income_ratio < 0.1:
            psychology_score += 0.1
            positive_factors.append("Conservative loan amount suggests prudent risk management")
        
        psychology_score = max(0.0, min(1.0, psychology_score))
        
        # Determine risk profile
        if psychology_score >= 0.7:
            risk_profile = "LOW_RISK"
        elif psychology_score >= 0.5:
            risk_profile = "MODERATE_RISK"
        else:
            risk_profile = "HIGH_RISK"
        
        return {
            "psychology_score": psychology_score,
            "risk_profile": risk_profile,
            "positive_factors": positive_factors,
            "risk_factors": risk_factors,
            "assessment": f"Psychological risk assessment indicates {risk_profile.replace('_', ' ').lower()} borrower profile."
        }
    
    def evaluate_spending_habits(self, application: LoanApplication) -> Dict[str, Any]:
        """Evaluate spending habits and financial discipline indicators."""
        spending_score = 0.5
        discipline_indicators = []
        concern_indicators = []
        
        # Analyze debt-to-income as spending discipline indicator
        dti = application.debt_to_income_ratio
        
        if dti <= 0.15:
            spending_score += 0.25
            discipline_indicators.append("Very low debt levels indicate excellent spending control")
        elif dti <= 0.25:
            spending_score += 0.15
            discipline_indicators.append("Low debt levels indicate good spending discipline")
        elif dti <= 0.35:
            spending_score += 0.05
            discipline_indicators.append("Moderate debt levels")
        elif dti <= 0.45:
            spending_score -= 0.1
            concern_indicators.append("High debt levels may indicate spending issues")
        else:
            spending_score -= 0.2
            concern_indicators.append("Very high debt levels suggest poor spending control")
        
        # Analyze loan purpose for spending priorities
        purpose = application.loan_purpose.lower() if application.loan_purpose else ""
        
        if any(word in purpose for word in ["emergency", "medical", "repair"]):
            spending_score += 0.1
            discipline_indicators.append("Borrowing for emergencies shows responsible priorities")
        elif any(word in purpose for word in ["education", "home", "investment"]):
            spending_score += 0.15
            discipline_indicators.append("Borrowing for investments shows good financial planning")
        elif any(word in purpose for word in ["consolidation", "refinance"]):
            spending_score += 0.05
            discipline_indicators.append("Debt consolidation shows attempt to improve finances")
        elif any(word in purpose for word in ["vacation", "wedding", "luxury"]):
            spending_score -= 0.1
            concern_indicators.append("Borrowing for discretionary items may indicate poor priorities")
        
        # Income level analysis for spending capacity
        if application.annual_income >= 75000:
            spending_score += 0.1
            discipline_indicators.append("Higher income provides more spending flexibility")
        elif application.annual_income < 30000:
            spending_score -= 0.05
            concern_indicators.append("Lower income limits spending flexibility")
        
        spending_score = max(0.0, min(1.0, spending_score))
        
        # Determine spending discipline level
        if spending_score >= 0.7:
            discipline_level = "EXCELLENT"
        elif spending_score >= 0.6:
            discipline_level = "GOOD"
        elif spending_score >= 0.4:
            discipline_level = "FAIR"
        else:
            discipline_level = "POOR"
        
        return {
            "spending_score": spending_score,
            "discipline_level": discipline_level,
            "discipline_indicators": discipline_indicators,
            "concern_indicators": concern_indicators,
            "assessment": f"Spending habit analysis indicates {discipline_level.lower()} financial discipline."
        }
    
    def analyze_debt_behavior(self, application: LoanApplication) -> Dict[str, Any]:
        """Analyze debt management behavior patterns."""
        debt_behavior_score = 0.5
        positive_behaviors = []
        negative_behaviors = []
        
        # Current debt level analysis
        current_dti = application.debt_to_income_ratio
        
        if current_dti <= 0.2:
            debt_behavior_score += 0.3
            positive_behaviors.append("Maintains low debt levels")
        elif current_dti <= 0.3:
            debt_behavior_score += 0.2
            positive_behaviors.append("Manages debt at reasonable levels")
        elif current_dti <= 0.4:
            debt_behavior_score += 0.0
        else:
            debt_behavior_score -= 0.2
            negative_behaviors.append("High existing debt burden")
        
        # Credit score as debt management indicator
        if application.credit_score >= 750:
            debt_behavior_score += 0.2
            positive_behaviors.append("Excellent credit history indicates good debt management")
        elif application.credit_score >= 650:
            debt_behavior_score += 0.1
            positive_behaviors.append("Good credit history")
        elif application.credit_score < 600:
            debt_behavior_score -= 0.15
            negative_behaviors.append("Poor credit history may indicate debt management issues")
        
        # Loan purpose analysis for debt behavior
        purpose = application.loan_purpose.lower() if application.loan_purpose else ""
        
        if "consolidation" in purpose or "refinance" in purpose:
            debt_behavior_score += 0.1
            positive_behaviors.append("Seeking debt consolidation shows proactive debt management")
        elif any(word in purpose for word in ["pay off", "payoff", "debt"]):
            debt_behavior_score += 0.15
            positive_behaviors.append("Using loan to pay off debt shows good debt strategy")
        
        # Calculate projected debt behavior
        monthly_income = application.annual_income / 12
        estimated_payment = application.loan_amount * 0.02  # Rough estimate
        projected_dti = (monthly_income * current_dti + estimated_payment) / monthly_income
        
        if projected_dti <= 0.35:
            positive_behaviors.append("Projected debt levels remain manageable")
        elif projected_dti > 0.5:
            debt_behavior_score -= 0.1
            negative_behaviors.append("Projected debt levels may become unmanageable")
        
        debt_behavior_score = max(0.0, min(1.0, debt_behavior_score))
        
        # Determine debt management capability
        if debt_behavior_score >= 0.7:
            debt_management = "EXCELLENT"
        elif debt_behavior_score >= 0.6:
            debt_management = "GOOD"
        elif debt_behavior_score >= 0.4:
            debt_management = "FAIR"
        else:
            debt_management = "POOR"
        
        return {
            "debt_behavior_score": debt_behavior_score,
            "debt_management": debt_management,
            "positive_behaviors": positive_behaviors,
            "negative_behaviors": negative_behaviors,
            "projected_dti": projected_dti,
            "assessment": f"Debt management behavior rated as {debt_management.lower()}."
        }
    
    def analyze(self, application: LoanApplication) -> AgentAnalysis:
        """Analyze a loan application from behavioral analysis perspective."""
        try:
            # Use tools to analyze different behavioral aspects
            behavior_analysis = self.analyze_behavior_patterns(application)
            psychology_analysis = self.assess_risk_psychology(application)
            spending_analysis = self.evaluate_spending_habits(application)
            debt_analysis = self.analyze_debt_behavior(application)
            
            # Create comprehensive analysis prompt
            analysis_prompt = f"""
            {self.get_role_prompt()}
            
            {self.format_application_data(application)}
            
            Behavior Pattern Analysis:
            {json.dumps(behavior_analysis, indent=2)}
            
            Risk Psychology Assessment:
            {json.dumps(psychology_analysis, indent=2)}
            
            Spending Habits Evaluation:
            {json.dumps(spending_analysis, indent=2)}
            
            Debt Behavior Analysis:
            {json.dumps(debt_analysis, indent=2)}
            
            Based on this comprehensive behavioral analysis, provide your recommendation.
            """
            
            if self.llm_client:
                response = self.llm_client.invoke([HumanMessage(content=analysis_prompt)])
                llm_response = response.content
            else:
                # Fallback analysis without LLM
                llm_response = self._fallback_analysis(
                    behavior_analysis, psychology_analysis, 
                    spending_analysis, debt_analysis
                )
            
            return self.create_analysis_result(application, llm_response)
            
        except Exception as e:
            self.logger.error(f"Error in behavioral analysis: {e}")
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
    
    def _fallback_analysis(self, behavior_analysis: Dict, psychology_analysis: Dict, 
                          spending_analysis: Dict, debt_analysis: Dict) -> str:
        """Provide fallback analysis when LLM is not available."""
        # Weighted scoring
        total_score = (
            behavior_analysis["behavior_score"] * 0.25 +
            psychology_analysis["psychology_score"] * 0.25 +
            spending_analysis["spending_score"] * 0.25 +
            debt_analysis["debt_behavior_score"] * 0.25
        )
        
        # Collect factors and concerns
        factors = []
        concerns = []
        
        if behavior_analysis["behavior_score"] >= 0.6:
            factors.extend(behavior_analysis["patterns_identified"][:2])
        else:
            concerns.extend(behavior_analysis["risk_indicators"])
        
        factors.extend(psychology_analysis["positive_factors"][:2])
        concerns.extend(psychology_analysis["risk_factors"][:2])
        
        factors.extend(spending_analysis["discipline_indicators"][:2])
        concerns.extend(spending_analysis["concern_indicators"][:2])
        
        factors.extend(debt_analysis["positive_behaviors"][:2])
        concerns.extend(debt_analysis["negative_behaviors"][:2])
        
        # Determine recommendation
        if total_score >= 0.7:
            recommendation = "APPROVE"
            risk = "LOW"
        elif total_score >= 0.6:
            recommendation = "CONDITIONAL"
            risk = "MODERATE"
        elif total_score >= 0.4:
            recommendation = "REVIEW"
            risk = "HIGH"
        else:
            recommendation = "REJECT"
            risk = "HIGH"
        
        return f"""
        Recommendation: {recommendation}
        Confidence: {total_score:.2f}
        Risk: {risk}
        Reasoning: Behavioral analysis indicates {risk.lower()} risk based on financial behavior patterns and psychological factors.
        Key Factors: {'; '.join(factors[:3])}
        Concerns: {'; '.join(concerns[:3]) if concerns else 'None identified'}
        """

