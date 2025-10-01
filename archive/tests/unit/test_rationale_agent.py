#!/usr/bin/env python3
"""
Unit tests for RationaleAgent
"""

import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from agents import RationaleAgent, Stock, RiskTolerance, InvestmentDecision

class TestRationaleAgent(unittest.TestCase):
    """Test cases for RationaleAgent"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.risk_tolerance = RiskTolerance.MODERATE
        self.agent = RationaleAgent(self.risk_tolerance, None)  # No LLM for testing
        
        # Create test stocks for different scenarios
        self.quality_stock = Stock(
            symbol="MSFT",
            company_name="Microsoft Corporation",
            sector="Technology",
            current_price=300.0,
            market_cap=2200e9,
            pe_ratio=28.0,
            dividend_yield=0.8,
            beta=0.9,
            volume=25000000
        )
        
        self.value_stock = Stock(
            symbol="BRK.B",
            company_name="Berkshire Hathaway",
            sector="Financial",
            current_price=350.0,
            market_cap=800e9,
            pe_ratio=15.0,
            dividend_yield=0.0,
            beta=0.8,
            volume=3000000
        )
        
        self.growth_stock = Stock(
            symbol="NVDA",
            company_name="NVIDIA Corporation",
            sector="Technology",
            current_price=450.0,
            market_cap=1100e9,
            pe_ratio=65.0,
            dividend_yield=0.1,
            beta=1.7,
            volume=40000000
        )
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        self.assertEqual(self.agent.agent_name, "rationale")
        self.assertEqual(self.agent.risk_tolerance, RiskTolerance.MODERATE)
        self.assertIsNone(self.agent.llm_client)
    
    def test_analyze_quality_stock(self):
        """Test analysis of high-quality stock"""
        analysis = self.agent.analyze(self.quality_stock)
        
        self.assertEqual(analysis.agent_name, "rationale")
        self.assertEqual(analysis.stock_symbol, "MSFT")
        self.assertIsInstance(analysis.recommendation, InvestmentDecision)
        self.assertGreaterEqual(analysis.confidence_score, 0.0)
        self.assertLessEqual(analysis.confidence_score, 1.0)
        self.assertIn(analysis.risk_assessment, ["LOW", "MODERATE", "HIGH"])
        self.assertIsInstance(analysis.key_factors, list)
        self.assertIsInstance(analysis.concerns, list)
        self.assertIsInstance(analysis.reasoning, str)
    
    def test_business_quality_analysis(self):
        """Test 7-step business quality analysis - Step 1"""
        metrics = self.agent._analyze_business_quality(self.quality_stock, None)
        
        self.assertIn('sales_growth_consistency', metrics)
        self.assertIn('net_income_consistency', metrics)
        self.assertIn('cash_flow_consistency', metrics)
        self.assertIn('margin_stability', metrics)
        self.assertIn('overall_quality_score', metrics)
        self.assertIn('business_maturity', metrics)
        
        # Quality scores should be between 0 and 1
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)
    
    def test_growth_rates_analysis(self):
        """Test growth rates analysis - Step 2"""
        growth = self.agent._analyze_growth_rates(self.quality_stock, None)
        
        self.assertIn('eps_growth_5y', growth)
        self.assertIn('revenue_growth_estimate', growth)
        self.assertIn('growth_sustainability', growth)
        self.assertIn('growth_quality_score', growth)
        
        # Growth rates should be reasonable
        self.assertGreaterEqual(growth['eps_growth_5y'], 0.0)
        self.assertLessEqual(growth['eps_growth_5y'], 1.0)  # 100% max
        self.assertGreaterEqual(growth['growth_sustainability'], 0.0)
        self.assertLessEqual(growth['growth_sustainability'], 1.0)
    
    def test_competitive_advantage_analysis(self):
        """Test competitive advantage analysis - Step 3"""
        moat = self.agent._analyze_competitive_advantage(self.quality_stock, None)
        
        self.assertIn('moat_type', moat)
        self.assertIn('moat_strength', moat)
        self.assertIn('brand_power', moat)
        self.assertIn('barriers_to_entry', moat)
        self.assertIn('economies_of_scale', moat)
        self.assertIn('network_effects', moat)
        self.assertIn('switching_costs', moat)
        self.assertIn('overall_moat_score', moat)
        
        # Moat strength should be valid
        self.assertIn(moat['moat_strength'], ['narrow', 'wide'])
        self.assertGreaterEqual(moat['overall_moat_score'], 0.0)
        self.assertLessEqual(moat['overall_moat_score'], 1.0)
    
    def test_operational_efficiency_analysis(self):
        """Test operational efficiency analysis - Step 4"""
        efficiency = self.agent._analyze_operational_efficiency(self.quality_stock, None)
        
        self.assertIn('roe_consistency', efficiency)
        self.assertIn('roic_consistency', efficiency)
        self.assertIn('revenue_efficiency', efficiency)
        self.assertIn('cash_conversion', efficiency)
        self.assertIn('overall_efficiency_score', efficiency)
        self.assertIn('target_roe', efficiency)
        self.assertIn('target_roic', efficiency)
        
        # Efficiency metrics should be reasonable
        self.assertGreaterEqual(efficiency['overall_efficiency_score'], 0.0)
        self.assertLessEqual(efficiency['overall_efficiency_score'], 1.0)
        self.assertGreater(efficiency['target_roe'], 0.0)
        self.assertGreater(efficiency['target_roic'], 0.0)
    
    def test_debt_structure_analysis(self):
        """Test debt structure analysis - Step 5"""
        debt = self.agent._analyze_debt_structure(self.quality_stock, None)
        
        self.assertIn('current_ratio', debt)
        self.assertIn('debt_to_ebitda', debt)
        self.assertIn('debt_service_ratio', debt)
        self.assertIn('debt_quality_score', debt)
        self.assertIn('financial_strength', debt)
        self.assertIn('sector_max_debt_ebitda', debt)
        self.assertIn('sector_min_current_ratio', debt)
        
        # Debt metrics should be reasonable
        self.assertGreater(debt['current_ratio'], 0.0)
        self.assertGreater(debt['debt_to_ebitda'], 0.0)
        self.assertIn(debt['financial_strength'], ['weak', 'moderate', 'strong'])
        self.assertGreaterEqual(debt['debt_quality_score'], 0.0)
        self.assertLessEqual(debt['debt_quality_score'], 1.0)
    
    def test_sector_specific_analysis(self):
        """Test sector-specific analysis differences"""
        # Technology stock should have different characteristics than Financial
        tech_analysis = self.agent.analyze(self.quality_stock)  # Technology
        financial_analysis = self.agent.analyze(self.value_stock)  # Financial
        
        # Both should be valid but potentially different recommendations
        self.assertIsInstance(tech_analysis.recommendation, InvestmentDecision)
        self.assertIsInstance(financial_analysis.recommendation, InvestmentDecision)
        
        # Technology typically has higher growth expectations
        tech_business = self.agent._analyze_business_quality(self.quality_stock, None)
        financial_business = self.agent._analyze_business_quality(self.value_stock, None)
        
        # Tech should generally have higher quality score due to sector
        self.assertGreaterEqual(tech_business['overall_quality_score'], 0.5)
    
    def test_market_cap_influence(self):
        """Test market cap influence on analysis"""
        # Large cap should have different characteristics than small cap
        large_cap_stock = Stock(
            symbol="LARGE",
            company_name="Large Cap Corp",
            sector="Technology",
            current_price=100.0,
            market_cap=200e9,  # Large cap
            pe_ratio=25.0,
            dividend_yield=1.0,
            beta=1.0,
            volume=10000000
        )
        
        small_cap_stock = Stock(
            symbol="SMALL",
            company_name="Small Cap Corp",
            sector="Technology",
            current_price=50.0,
            market_cap=1e9,  # Small cap
            pe_ratio=30.0,
            dividend_yield=0.0,
            beta=1.5,
            volume=1000000
        )
        
        large_business = self.agent._analyze_business_quality(large_cap_stock, None)
        small_business = self.agent._analyze_business_quality(small_cap_stock, None)
        
        # Large cap should have higher business maturity
        self.assertGreater(large_business['business_maturity'], small_business['business_maturity'])
    
    def test_target_price_calculation(self):
        """Test target price calculation"""
        business_metrics = {'overall_quality_score': 0.8}
        growth_analysis = {'growth_quality_score': 0.7}
        
        target_price = self.agent._calculate_rationale_target_price(
            self.quality_stock, business_metrics, growth_analysis
        )
        
        if target_price is not None:
            self.assertGreater(target_price, 0)
            self.assertIsInstance(target_price, float)
            # Target price should be reasonable relative to current price
            self.assertGreater(target_price, self.quality_stock.current_price * 0.5)
            self.assertLess(target_price, self.quality_stock.current_price * 3.0)
    
    def test_risk_assessment(self):
        """Test business risk assessment"""
        business_metrics = {'overall_quality_score': 0.8}
        debt_analysis = {'financial_strength': 'strong', 'debt_quality_score': 0.8}
        efficiency_analysis = {'overall_efficiency_score': 0.7}
        
        risk = self.agent._assess_business_risk(
            business_metrics, debt_analysis, efficiency_analysis
        )
        
        self.assertIn(risk, ["LOW", "MODERATE", "HIGH"])
        
        # High quality should result in lower risk
        self.assertEqual(risk, "LOW")
    
    def test_key_factors_extraction(self):
        """Test key factors extraction"""
        business_metrics = {'overall_quality_score': 0.8}
        growth_analysis = {'growth_quality_score': 0.7, 'eps_growth_5y': 0.15}
        moat_analysis = {'overall_moat_score': 0.8, 'moat_strength': 'wide', 'moat_type': 'network_effects', 'brand_power': 0.8}
        efficiency_analysis = {'overall_efficiency_score': 0.7}
        
        factors = self.agent._extract_key_factors(
            business_metrics, growth_analysis, moat_analysis, efficiency_analysis
        )
        
        self.assertIsInstance(factors, list)
        self.assertLessEqual(len(factors), 5)
        
        # Should identify high quality factors
        factor_text = ' '.join(factors)
        self.assertTrue(any(keyword in factor_text.lower() for keyword in ['quality', 'growth', 'moat', 'efficiency']))
    
    def test_concerns_identification(self):
        """Test concerns identification"""
        business_metrics = {'overall_quality_score': 0.3}  # Low quality
        growth_analysis = {'growth_sustainability': 0.4}   # Poor sustainability
        debt_analysis = {'financial_strength': 'weak', 'debt_to_ebitda': 5.0, 'current_ratio': 0.8}
        
        concerns = self.agent._identify_concerns(
            business_metrics, growth_analysis, debt_analysis
        )
        
        self.assertIsInstance(concerns, list)
        self.assertLessEqual(len(concerns), 5)
        self.assertGreater(len(concerns), 0)  # Should identify concerns with poor metrics
    
    def test_fallback_analysis(self):
        """Test fallback analysis without LLM"""
        business_metrics = {'overall_quality_score': 0.8}
        growth_analysis = {'growth_quality_score': 0.7}
        moat_analysis = {'overall_moat_score': 0.8, 'moat_strength': 'wide'}
        efficiency_analysis = {'overall_efficiency_score': 0.7}
        debt_analysis = {'debt_quality_score': 0.8, 'financial_strength': 'strong'}
        
        recommendation, confidence, reasoning = self.agent._fallback_analysis(
            business_metrics, growth_analysis, moat_analysis, efficiency_analysis, debt_analysis
        )
        
        self.assertIsInstance(recommendation, InvestmentDecision)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        self.assertIsInstance(reasoning, str)
        self.assertIn("7-Step Great Business Analysis", reasoning)

if __name__ == '__main__':
    unittest.main()

