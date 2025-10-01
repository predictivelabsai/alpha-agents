#!/usr/bin/env python3
"""
Unit tests for SecularTrendAgent
"""

import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from agents import SecularTrendAgent, Stock, RiskTolerance, InvestmentDecision

class TestSecularTrendAgent(unittest.TestCase):
    """Test cases for SecularTrendAgent"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.risk_tolerance = RiskTolerance.MODERATE
        self.agent = SecularTrendAgent(self.risk_tolerance, None)  # No LLM for testing
        
        # Create test stocks for different trend exposures
        self.ai_stock = Stock(
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
        
        self.cloud_stock = Stock(
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
        
        self.ev_stock = Stock(
            symbol="TSLA",
            company_name="Tesla Inc.",
            sector="Consumer Discretionary",
            current_price=200.0,
            market_cap=650e9,
            pe_ratio=45.0,
            dividend_yield=0.0,
            beta=2.0,
            volume=80000000
        )
        
        self.cyber_stock = Stock(
            symbol="CRWD",
            company_name="CrowdStrike Holdings",
            sector="Technology",
            current_price=180.0,
            market_cap=45e9,
            pe_ratio=85.0,
            dividend_yield=0.0,
            beta=1.4,
            volume=5000000
        )
        
        self.traditional_stock = Stock(
            symbol="KO",
            company_name="The Coca-Cola Company",
            sector="Consumer Staples",
            current_price=60.0,
            market_cap=260e9,
            pe_ratio=25.0,
            dividend_yield=3.0,
            beta=0.6,
            volume=15000000
        )
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        self.assertEqual(self.agent.agent_name, "secular_trend")
        self.assertEqual(self.agent.risk_tolerance, RiskTolerance.MODERATE)
        self.assertIsNone(self.agent.llm_client)
        
        # Test secular trends are properly defined
        self.assertEqual(len(self.agent.secular_trends), 5)
        expected_trends = ['agentic_ai', 'cloud_edge', 'ai_semiconductors', 'cybersecurity_agentic', 'electrification_ai']
        for trend in expected_trends:
            self.assertIn(trend, self.agent.secular_trends)
            
        # Test company trend mapping exists
        self.assertIsInstance(self.agent.company_trend_mapping, dict)
        self.assertIn('NVDA', self.agent.company_trend_mapping)
        self.assertIn('MSFT', self.agent.company_trend_mapping)
    
    def test_secular_trends_structure(self):
        """Test secular trends data structure"""
        for trend_key, trend_data in self.agent.secular_trends.items():
            self.assertIn('name', trend_data)
            self.assertIn('description', trend_data)
            self.assertIn('market_size', trend_data)
            self.assertIn('growth_rate', trend_data)
            self.assertIn('timeline', trend_data)
            self.assertIn('key_characteristics', trend_data)
            self.assertIn('winners', trend_data)
            self.assertIn('margin_profile', trend_data)
            
            # Validate data types and ranges
            self.assertIsInstance(trend_data['name'], str)
            self.assertIsInstance(trend_data['description'], str)
            self.assertGreater(trend_data['market_size'], 0)
            self.assertGreaterEqual(trend_data['growth_rate'], 0.0)
            self.assertLessEqual(trend_data['growth_rate'], 1.0)
            self.assertIsInstance(trend_data['key_characteristics'], list)
            self.assertIsInstance(trend_data['winners'], list)
            self.assertGreaterEqual(trend_data['margin_profile'], 0.0)
            self.assertLessEqual(trend_data['margin_profile'], 1.0)
    
    def test_analyze_ai_semiconductor_stock(self):
        """Test analysis of AI semiconductor stock (NVIDIA)"""
        analysis = self.agent.analyze(self.ai_stock)
        
        self.assertEqual(analysis.agent_name, "secular_trend")
        self.assertEqual(analysis.stock_symbol, "NVDA")
        self.assertIsInstance(analysis.recommendation, InvestmentDecision)
        self.assertGreaterEqual(analysis.confidence_score, 0.0)
        self.assertLessEqual(analysis.confidence_score, 1.0)
        self.assertIn(analysis.risk_assessment, ["LOW", "MODERATE", "HIGH"])
        self.assertIsInstance(analysis.key_factors, list)
        self.assertIsInstance(analysis.concerns, list)
        self.assertIsInstance(analysis.reasoning, str)
    
    def test_analyze_cloud_stock(self):
        """Test analysis of cloud infrastructure stock (Microsoft)"""
        analysis = self.agent.analyze(self.cloud_stock)
        
        self.assertEqual(analysis.stock_symbol, "MSFT")
        self.assertIsInstance(analysis.recommendation, InvestmentDecision)
        # Microsoft should have strong trend positioning
        self.assertGreaterEqual(analysis.confidence_score, 0.5)
    
    def test_analyze_ev_stock(self):
        """Test analysis of electrification stock (Tesla)"""
        analysis = self.agent.analyze(self.ev_stock)
        
        self.assertEqual(analysis.stock_symbol, "TSLA")
        self.assertIsInstance(analysis.recommendation, InvestmentDecision)
    
    def test_analyze_traditional_stock(self):
        """Test analysis of traditional stock with limited trend exposure"""
        analysis = self.agent.analyze(self.traditional_stock)
        
        self.assertEqual(analysis.stock_symbol, "KO")
        self.assertIsInstance(analysis.recommendation, InvestmentDecision)
        # Traditional stock should have lower trend positioning
        # but analysis should still be valid
    
    def test_identify_relevant_trends(self):
        """Test trend identification for different stocks"""
        # AI semiconductor stock should identify AI trends
        ai_trends = self.agent._identify_relevant_trends(self.ai_stock)
        self.assertIsInstance(ai_trends, list)
        self.assertIn('ai_semiconductors', ai_trends)
        
        # Cloud stock should identify cloud trends
        cloud_trends = self.agent._identify_relevant_trends(self.cloud_stock)
        self.assertIn('agentic_ai', cloud_trends)
        self.assertIn('cloud_edge', cloud_trends)
        
        # EV stock should identify electrification trends
        ev_trends = self.agent._identify_relevant_trends(self.ev_stock)
        self.assertIn('electrification_ai', ev_trends)
        
        # Cybersecurity stock should identify security trends
        cyber_trends = self.agent._identify_relevant_trends(self.cyber_stock)
        self.assertIn('cybersecurity_agentic', cyber_trends)
        
        # Traditional stock should have fewer relevant trends
        traditional_trends = self.agent._identify_relevant_trends(self.traditional_stock)
        self.assertLessEqual(len(traditional_trends), 2)
    
    def test_trend_positioning_analysis(self):
        """Test trend positioning analysis"""
        relevant_trends = ['ai_semiconductors', 'cloud_edge']
        positioning = self.agent._analyze_trend_positioning(self.ai_stock, relevant_trends)
        
        self.assertIn('overall_positioning_score', positioning)
        self.assertIn('trend_alignment', positioning)
        self.assertIn('market_timing', positioning)
        self.assertIn('competitive_advantage', positioning)
        
        # Positioning score should be valid
        self.assertGreaterEqual(positioning['overall_positioning_score'], 0.0)
        self.assertLessEqual(positioning['overall_positioning_score'], 1.0)
        
        # Market timing should be valid
        valid_timings = ['early_leader', 'well_positioned', 'moderate', 'lagging']
        self.assertIn(positioning['market_timing'], valid_timings)
        
        # Trend alignment should contain trend data
        self.assertIsInstance(positioning['trend_alignment'], dict)
        for trend_key in relevant_trends:
            if trend_key in positioning['trend_alignment']:
                trend_info = positioning['trend_alignment'][trend_key]
                self.assertIn('score', trend_info)
                self.assertIn('trend_name', trend_info)
                self.assertIn('market_size', trend_info)
                self.assertIn('growth_rate', trend_info)
    
    def test_market_opportunity_assessment(self):
        """Test market opportunity assessment"""
        relevant_trends = ['agentic_ai', 'ai_semiconductors']
        opportunity = self.agent._assess_market_opportunity(self.ai_stock, relevant_trends)
        
        self.assertIn('total_addressable_market', opportunity)
        self.assertIn('weighted_growth_rate', opportunity)
        self.assertIn('opportunity_score', opportunity)
        self.assertIn('timeline_alignment', opportunity)
        
        # Market size should be substantial for AI trends
        self.assertGreater(opportunity['total_addressable_market'], 0)
        
        # Growth rate should be positive
        self.assertGreaterEqual(opportunity['weighted_growth_rate'], 0.0)
        
        # Opportunity score should be valid
        self.assertGreaterEqual(opportunity['opportunity_score'], 0.0)
        self.assertLessEqual(opportunity['opportunity_score'], 1.0)
        
        # Timeline alignment should be valid
        valid_timelines = ['near_term', 'medium_term', 'long_term']
        self.assertIn(opportunity['timeline_alignment'], valid_timelines)
    
    def test_competitive_position_analysis(self):
        """Test competitive position analysis"""
        relevant_trends = ['ai_semiconductors']
        competitive = self.agent._analyze_competitive_position(self.ai_stock, relevant_trends)
        
        self.assertIn('market_leadership', competitive)
        self.assertIn('differentiation', competitive)
        self.assertIn('execution_track_record', competitive)
        self.assertIn('competitive_moats', competitive)
        self.assertIn('overall_competitive_score', competitive)
        
        # All scores should be valid
        for key in ['market_leadership', 'differentiation', 'execution_track_record', 'overall_competitive_score']:
            self.assertGreaterEqual(competitive[key], 0.0)
            self.assertLessEqual(competitive[key], 1.0)
        
        # Competitive moats should be a list
        self.assertIsInstance(competitive['competitive_moats'], list)
    
    def test_execution_capability_assessment(self):
        """Test execution capability assessment"""
        relevant_trends = ['ai_semiconductors']
        execution = self.agent._assess_execution_capability(self.ai_stock, relevant_trends)
        
        self.assertIn('financial_resources', execution)
        self.assertIn('innovation_capability', execution)
        self.assertIn('market_access', execution)
        self.assertIn('talent_acquisition', execution)
        self.assertIn('execution_risk', execution)
        self.assertIn('overall_execution_score', execution)
        
        # All capability scores should be valid
        for key in ['financial_resources', 'innovation_capability', 'market_access', 'talent_acquisition', 'overall_execution_score']:
            self.assertGreaterEqual(execution[key], 0.0)
            self.assertLessEqual(execution[key], 1.0)
        
        # Execution risk should be valid
        self.assertIn(execution['execution_risk'], ['low', 'moderate', 'high'])
    
    def test_market_cap_influence(self):
        """Test market cap influence on trend analysis"""
        # Large cap should have different positioning than small cap
        large_cap_stock = Stock(
            symbol="LARGE",
            company_name="Large Tech Corp",
            sector="Technology",
            current_price=100.0,
            market_cap=500e9,  # Mega cap
            pe_ratio=25.0,
            dividend_yield=1.0,
            beta=1.0,
            volume=20000000
        )
        
        small_cap_stock = Stock(
            symbol="SMALL",
            company_name="Small Tech Corp",
            sector="Technology",
            current_price=50.0,
            market_cap=2e9,  # Small cap
            pe_ratio=40.0,
            dividend_yield=0.0,
            beta=1.8,
            volume=1000000
        )
        
        relevant_trends = ['agentic_ai']
        
        large_positioning = self.agent._analyze_trend_positioning(large_cap_stock, relevant_trends)
        small_positioning = self.agent._analyze_trend_positioning(small_cap_stock, relevant_trends)
        
        # Large cap should generally have better positioning
        self.assertGreaterEqual(large_positioning['overall_positioning_score'], 
                               small_positioning['overall_positioning_score'])
    
    def test_target_price_calculation(self):
        """Test target price calculation based on trends"""
        relevant_trends = ['ai_semiconductors']
        trend_positioning = {'overall_positioning_score': 0.8}
        
        target_price = self.agent._calculate_trend_target_price(
            self.ai_stock, relevant_trends, trend_positioning
        )
        
        if target_price is not None:
            self.assertGreater(target_price, 0)
            self.assertIsInstance(target_price, float)
            # Target should be reasonable relative to current price
            self.assertGreater(target_price, self.ai_stock.current_price * 0.5)
            self.assertLess(target_price, self.ai_stock.current_price * 5.0)
    
    def test_trend_risk_assessment(self):
        """Test trend-based risk assessment"""
        relevant_trends = ['ai_semiconductors']
        execution_capability = {'execution_risk': 'low', 'overall_execution_score': 0.8}
        
        risk = self.agent._assess_trend_risk(self.ai_stock, relevant_trends, execution_capability)
        self.assertIn(risk, ["LOW", "MODERATE", "HIGH"])
        
        # Test high risk scenario
        high_risk_execution = {'execution_risk': 'high', 'overall_execution_score': 0.3}
        high_risk_stock = Stock(
            symbol="RISKY",
            company_name="Risky Corp",
            sector="Technology",
            current_price=10.0,
            market_cap=500e6,  # Very small cap
            pe_ratio=100.0,
            dividend_yield=0.0,
            beta=2.5,
            volume=100000
        )
        
        high_risk = self.agent._assess_trend_risk(high_risk_stock, relevant_trends, high_risk_execution)
        self.assertEqual(high_risk, "HIGH")
    
    def test_key_factors_extraction(self):
        """Test key factors extraction"""
        relevant_trends = ['ai_semiconductors', 'agentic_ai']
        trend_positioning = {'overall_positioning_score': 0.8, 'market_timing': 'early_leader'}
        competitive_position = {'market_leadership': 0.9, 'overall_competitive_score': 0.8}
        
        factors = self.agent._extract_key_factors(relevant_trends, trend_positioning, competitive_position)
        
        self.assertIsInstance(factors, list)
        self.assertLessEqual(len(factors), 5)
        
        # Should identify strong positioning factors
        if len(factors) > 0:
            factor_text = ' '.join(factors)
            self.assertTrue(any(keyword in factor_text.lower() for keyword in 
                              ['trend', 'positioning', 'leadership', 'growth', 'exposure']))
    
    def test_concerns_identification(self):
        """Test concerns identification"""
        relevant_trends = []  # No trend exposure
        execution_capability = {'execution_risk': 'high', 'financial_resources': 0.3, 'innovation_capability': 0.4}
        
        small_cap_stock = Stock(
            symbol="CONCERN",
            company_name="Concerning Corp",
            sector="Technology",
            current_price=5.0,
            market_cap=500e6,  # Small cap
            pe_ratio=150.0,
            dividend_yield=0.0,
            beta=3.0,
            volume=50000
        )
        
        concerns = self.agent._identify_concerns(small_cap_stock, relevant_trends, execution_capability)
        
        self.assertIsInstance(concerns, list)
        self.assertLessEqual(len(concerns), 5)
        self.assertGreater(len(concerns), 0)  # Should identify concerns
    
    def test_fallback_analysis(self):
        """Test fallback analysis without LLM"""
        relevant_trends = ['ai_semiconductors']
        trend_positioning = {'overall_positioning_score': 0.8, 'market_timing': 'early_leader'}
        market_opportunity = {'opportunity_score': 0.8, 'weighted_growth_rate': 0.4, 'total_addressable_market': 500e9}
        competitive_position = {'overall_competitive_score': 0.8}
        execution_capability = {'overall_execution_score': 0.7, 'execution_risk': 'moderate'}
        
        recommendation, confidence, reasoning = self.agent._fallback_analysis(
            self.ai_stock, relevant_trends, trend_positioning, market_opportunity, 
            competitive_position, execution_capability
        )
        
        self.assertIsInstance(recommendation, InvestmentDecision)
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        self.assertIsInstance(reasoning, str)
        self.assertIn("Secular Technology Trends Analysis", reasoning)

if __name__ == '__main__':
    unittest.main()

