#!/usr/bin/env python3
"""
Unit tests for FundamentalAgent
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from agents import FundamentalAgent, Stock, RiskTolerance, InvestmentDecision

class TestFundamentalAgent(unittest.TestCase):
    """Test cases for FundamentalAgent"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.risk_tolerance = RiskTolerance.MODERATE
        self.agent = FundamentalAgent(self.risk_tolerance, None)  # No LLM for testing
        
        # Create test stocks
        self.tech_stock = Stock(
            symbol="AAPL",
            company_name="Apple Inc.",
            sector="Technology",
            current_price=150.0,
            market_cap=2500e9,
            pe_ratio=25.0,
            dividend_yield=0.5,
            beta=1.2,
            volume=50000000
        )
        
        self.utility_stock = Stock(
            symbol="NEE",
            company_name="NextEra Energy",
            sector="Utilities",
            current_price=80.0,
            market_cap=160e9,
            pe_ratio=22.0,
            dividend_yield=3.2,
            beta=0.8,
            volume=8000000
        )
        
        self.small_cap_stock = Stock(
            symbol="SMALL",
            company_name="Small Cap Corp",
            sector="Technology",
            current_price=25.0,
            market_cap=1e9,
            pe_ratio=35.0,
            dividend_yield=0.0,
            beta=1.8,
            volume=2000000
        )
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        self.assertEqual(self.agent.agent_name, "fundamental")
        self.assertEqual(self.agent.risk_tolerance, RiskTolerance.MODERATE)
        self.assertIsNone(self.agent.llm_client)
    
    def test_analyze_tech_stock(self):
        """Test analysis of technology stock"""
        analysis = self.agent.analyze(self.tech_stock)
        
        self.assertEqual(analysis.agent_name, "fundamental")
        self.assertEqual(analysis.stock_symbol, "AAPL")
        self.assertIsInstance(analysis.recommendation, InvestmentDecision)
        self.assertGreaterEqual(analysis.confidence_score, 0.0)
        self.assertLessEqual(analysis.confidence_score, 1.0)
        self.assertIn(analysis.risk_assessment, ["LOW", "MODERATE", "HIGH"])
        self.assertIsInstance(analysis.key_factors, list)
        self.assertIsInstance(analysis.concerns, list)
        self.assertIsInstance(analysis.reasoning, str)
    
    def test_analyze_utility_stock(self):
        """Test analysis of utility stock"""
        analysis = self.agent.analyze(self.utility_stock)
        
        self.assertEqual(analysis.stock_symbol, "NEE")
        self.assertIsInstance(analysis.recommendation, InvestmentDecision)
        # Utilities typically have lower risk
        self.assertIn(analysis.risk_assessment, ["LOW", "MODERATE"])
    
    def test_analyze_small_cap_stock(self):
        """Test analysis of small cap stock"""
        analysis = self.agent.analyze(self.small_cap_stock)
        
        self.assertEqual(analysis.stock_symbol, "SMALL")
        # Small cap should typically have higher risk
        self.assertIn(analysis.risk_assessment, ["MODERATE", "HIGH"])
    
    def test_financial_strength_analysis(self):
        """Test financial strength analysis"""
        metrics = self.agent._analyze_financial_strength(self.tech_stock, None)
        
        self.assertIn('debt_to_equity', metrics)
        self.assertIn('current_ratio', metrics)
        self.assertIn('interest_coverage', metrics)
        self.assertIn('financial_strength_score', metrics)
        self.assertGreaterEqual(metrics['financial_strength_score'], 0.0)
        self.assertLessEqual(metrics['financial_strength_score'], 1.0)
    
    def test_earnings_quality_analysis(self):
        """Test earnings quality analysis"""
        quality = self.agent._analyze_earnings_quality(self.tech_stock, None)
        
        self.assertIn('earnings_consistency', quality)
        self.assertIn('revenue_growth', quality)
        self.assertIn('margin_stability', quality)
        self.assertIn('earnings_quality_score', quality)
        self.assertGreaterEqual(quality['earnings_quality_score'], 0.0)
        self.assertLessEqual(quality['earnings_quality_score'], 1.0)
    
    def test_growth_prospects_analysis(self):
        """Test growth prospects analysis"""
        growth = self.agent._analyze_growth_prospects(self.tech_stock, None)
        
        self.assertIn('revenue_growth_trend', growth)
        self.assertIn('earnings_growth_trend', growth)
        self.assertIn('market_expansion', growth)
        self.assertIn('growth_score', growth)
        self.assertGreaterEqual(growth['growth_score'], 0.0)
        self.assertLessEqual(growth['growth_score'], 1.0)
    
    def test_target_price_calculation(self):
        """Test target price calculation"""
        financial_strength = {'financial_strength_score': 0.8}
        earnings_quality = {'earnings_quality_score': 0.7}
        growth_prospects = {'growth_score': 0.6}
        
        target_price = self.agent._calculate_fundamental_target_price(
            self.tech_stock, financial_strength, earnings_quality, growth_prospects
        )
        
        if target_price is not None:
            self.assertGreater(target_price, 0)
            self.assertIsInstance(target_price, float)
    
    def test_risk_assessment(self):
        """Test risk assessment"""
        financial_strength = {'financial_strength_score': 0.8}
        earnings_quality = {'earnings_quality_score': 0.7}
        
        risk = self.agent._assess_fundamental_risk(
            self.tech_stock, financial_strength, earnings_quality
        )
        
        self.assertIn(risk, ["LOW", "MODERATE", "HIGH"])
    
    def test_key_factors_extraction(self):
        """Test key factors extraction"""
        financial_strength = {'financial_strength_score': 0.8, 'current_ratio': 2.0}
        earnings_quality = {'earnings_quality_score': 0.7, 'earnings_consistency': 0.8}
        growth_prospects = {'growth_score': 0.6, 'revenue_growth_trend': 0.7}
        
        factors = self.agent._extract_key_factors(
            financial_strength, earnings_quality, growth_prospects
        )
        
        self.assertIsInstance(factors, list)
        self.assertLessEqual(len(factors), 5)  # Should limit to 5 factors
    
    def test_concerns_identification(self):
        """Test concerns identification"""
        financial_strength = {'financial_strength_score': 0.3, 'debt_to_equity': 3.0}
        earnings_quality = {'earnings_quality_score': 0.4}
        
        concerns = self.agent._identify_concerns(
            self.tech_stock, financial_strength, earnings_quality
        )
        
        self.assertIsInstance(concerns, list)
        self.assertLessEqual(len(concerns), 5)  # Should limit to 5 concerns
    
    def test_risk_tolerance_adjustment(self):
        """Test risk tolerance adjustment"""
        base_confidence = 0.7
        
        # Test conservative adjustment
        conservative_agent = FundamentalAgent(RiskTolerance.CONSERVATIVE, None)
        conservative_confidence = conservative_agent.adjust_for_risk_tolerance(base_confidence)
        self.assertLessEqual(conservative_confidence, base_confidence)
        
        # Test aggressive adjustment
        aggressive_agent = FundamentalAgent(RiskTolerance.AGGRESSIVE, None)
        aggressive_confidence = aggressive_agent.adjust_for_risk_tolerance(base_confidence)
        self.assertGreaterEqual(aggressive_confidence, base_confidence)
    
    def test_error_handling(self):
        """Test error handling in analysis"""
        # Test with invalid stock data
        invalid_stock = Stock(
            symbol="INVALID",
            company_name="Invalid Corp",
            sector="Unknown",
            current_price=-10.0,  # Invalid price
            market_cap=-1000,     # Invalid market cap
            pe_ratio=None,
            dividend_yield=None,
            beta=None,
            volume=0
        )
        
        analysis = self.agent.analyze(invalid_stock)
        
        # Should still return a valid analysis object
        self.assertEqual(analysis.agent_name, "fundamental")
        self.assertEqual(analysis.stock_symbol, "INVALID")
        self.assertIsInstance(analysis.recommendation, InvestmentDecision)

if __name__ == '__main__':
    unittest.main()

