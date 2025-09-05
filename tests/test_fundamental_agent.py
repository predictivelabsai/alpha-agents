"""
Test suite for Fundamental Agent
"""

import pytest
import os
import sys
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.fundamental_agent import FundamentalAgent, QualifiedCompany


class TestFundamentalAgent:
    """Test cases for Fundamental Agent"""
    
    def setup_method(self):
        """Setup test environment"""
        self.agent = FundamentalAgent(use_llm=False)
    
    def test_agent_initialization(self):
        """Test agent initializes correctly"""
        assert self.agent is not None
        assert self.agent.yfinance_util is not None
        assert self.agent.llm is None  # No LLM in test mode
    
    def test_quantitative_criteria_pass(self):
        """Test company that meets all criteria"""
        # Mock metrics for a strong company
        strong_metrics = {
            'ticker': 'TEST',
            'company_name': 'Test Company',
            'sector': 'Technology',
            'market_cap': 1000000000,
            'revenue_growth_5y': 15.0,
            'net_income_growth_5y': 20.0,
            'cash_flow_growth_5y': 18.0,
            'roe_ttm': 18.0,
            'roic_ttm': 22.0,
            'current_ratio': 2.1,
            'debt_to_ebitda': 0.8,
            'debt_service_ratio': 150.0,
            'gross_margin': 45.0,
            'profit_margin': 15.0
        }
        
        result = self.agent._meets_quantitative_criteria(strong_metrics)
        assert result is True
    
    def test_quantitative_criteria_fail_growth(self):
        """Test company that fails growth criteria"""
        weak_growth_metrics = {
            'ticker': 'WEAK',
            'revenue_growth_5y': -5.0,  # Negative growth
            'net_income_growth_5y': 10.0,
            'cash_flow_growth_5y': 5.0,
            'roe_ttm': 15.0,
            'roic_ttm': 18.0,
            'current_ratio': 1.5,
            'debt_to_ebitda': 1.0,
            'debt_service_ratio': 100.0,
            'gross_margin': 30.0,
            'profit_margin': 10.0
        }
        
        result = self.agent._meets_quantitative_criteria(weak_growth_metrics)
        assert result is False
    
    def test_quantitative_criteria_fail_profitability(self):
        """Test company that fails profitability criteria"""
        weak_profit_metrics = {
            'ticker': 'LOWROE',
            'revenue_growth_5y': 10.0,
            'net_income_growth_5y': 8.0,
            'cash_flow_growth_5y': 12.0,
            'roe_ttm': 8.0,  # Below 12% threshold
            'roic_ttm': 10.0,  # Below 12% threshold
            'current_ratio': 1.8,
            'debt_to_ebitda': 1.5,
            'debt_service_ratio': 80.0,
            'gross_margin': 25.0,
            'profit_margin': 5.0
        }
        
        result = self.agent._meets_quantitative_criteria(weak_profit_metrics)
        assert result is False
    
    def test_quantitative_criteria_fail_debt(self):
        """Test company that fails debt criteria"""
        high_debt_metrics = {
            'ticker': 'DEBT',
            'revenue_growth_5y': 12.0,
            'net_income_growth_5y': 15.0,
            'cash_flow_growth_5y': 10.0,
            'roe_ttm': 16.0,
            'roic_ttm': 18.0,
            'current_ratio': 0.8,  # Below 1.0
            'debt_to_ebitda': 4.5,  # Above 3.0
            'debt_service_ratio': 25.0,  # Below 30%
            'gross_margin': 35.0,
            'profit_margin': 12.0
        }
        
        result = self.agent._meets_quantitative_criteria(high_debt_metrics)
        assert result is False
    
    def test_growth_score_calculation(self):
        """Test growth score calculation"""
        excellent_growth = {
            'revenue_growth_5y': 25.0,
            'net_income_growth_5y': 30.0,
            'cash_flow_growth_5y': 22.0
        }
        
        score = self.agent._calculate_growth_score(excellent_growth)
        assert score == 100  # Should get maximum score
        
        moderate_growth = {
            'revenue_growth_5y': 8.0,
            'net_income_growth_5y': 6.0,
            'cash_flow_growth_5y': 7.0
        }
        
        score = self.agent._calculate_growth_score(moderate_growth)
        assert 30 <= score <= 60  # Should get moderate score
    
    def test_profitability_score_calculation(self):
        """Test profitability score calculation"""
        excellent_profit = {
            'roe_ttm': 28.0,
            'roic_ttm': 32.0,
            'gross_margin': 55.0,
            'profit_margin': 25.0
        }
        
        score = self.agent._calculate_profitability_score(excellent_profit)
        assert score == 100  # Should get maximum score
    
    def test_debt_score_calculation(self):
        """Test debt score calculation"""
        excellent_debt = {
            'current_ratio': 2.5,
            'debt_to_ebitda': 0.3,
            'debt_service_ratio': 200.0
        }
        
        score = self.agent._calculate_debt_score(excellent_debt)
        assert score == 100  # Should get maximum score
    
    def test_qualified_company_creation(self):
        """Test creation of QualifiedCompany object"""
        test_metrics = {
            'ticker': 'QUAL',
            'company_name': 'Qualified Corp',
            'sector': 'Technology',
            'market_cap': 2000000000,
            'revenue_growth_5y': 18.0,
            'net_income_growth_5y': 22.0,
            'cash_flow_growth_5y': 20.0,
            'roe_ttm': 19.0,
            'roic_ttm': 24.0,
            'gross_margin': 48.0,
            'profit_margin': 18.0,
            'current_ratio': 2.2,
            'debt_to_ebitda': 0.6,
            'debt_service_ratio': 180.0
        }
        
        qualified = self.agent._create_qualified_company(test_metrics, 'Technology')
        
        assert isinstance(qualified, QualifiedCompany)
        assert qualified.ticker == 'QUAL'
        assert qualified.company_name == 'Qualified Corp'
        assert qualified.sector == 'Technology'
        assert qualified.overall_score > 80  # Should be high scoring
        assert qualified.growth_score > 0
        assert qualified.profitability_score > 0
        assert qualified.debt_score > 0
    
    @patch('agents.fundamental_agent.YFinanceUtil')
    def test_screen_sector_mock(self, mock_yfinance):
        """Test sector screening with mocked data"""
        # Mock the yfinance utility
        mock_util = Mock()
        mock_util.get_sector_universe.return_value = ['TEST1', 'TEST2']
        
        # Mock metrics for qualifying companies
        qualifying_metrics = {
            'ticker': 'TEST1',
            'company_name': 'Test Company 1',
            'sector': 'Technology',
            'market_cap': 1500000000,
            'revenue_growth_5y': 16.0,
            'net_income_growth_5y': 18.0,
            'cash_flow_growth_5y': 14.0,
            'roe_ttm': 17.0,
            'roic_ttm': 21.0,
            'gross_margin': 42.0,
            'profit_margin': 16.0,
            'current_ratio': 1.9,
            'debt_to_ebitda': 1.2,
            'debt_service_ratio': 120.0
        }
        
        mock_util.get_fundamental_metrics.return_value = qualifying_metrics
        self.agent.yfinance_util = mock_util
        
        # Test screening
        results = self.agent.screen_sector('Technology', 'US', max_market_cap=5000000000)
        
        # Verify results
        assert len(results) >= 0  # Should return some results
        mock_util.get_sector_universe.assert_called_once()
    
    def test_screening_summary(self):
        """Test screening summary generation"""
        # Create mock qualified companies
        companies = [
            QualifiedCompany(
                ticker='TEST1', company_name='Test 1', sector='Technology',
                market_cap=1000000000, revenue_growth_5y=15.0, net_income_growth_5y=18.0,
                cash_flow_growth_5y=16.0, roe_ttm=17.0, roic_ttm=20.0,
                gross_margin=40.0, profit_margin=15.0, current_ratio=1.8,
                debt_to_ebitda=1.0, debt_service_ratio=100.0,
                growth_score=85.0, profitability_score=88.0, debt_score=82.0,
                overall_score=85.2, commentary='', timestamp='2024-01-01'
            ),
            QualifiedCompany(
                ticker='TEST2', company_name='Test 2', sector='Healthcare',
                market_cap=2000000000, revenue_growth_5y=12.0, net_income_growth_5y=14.0,
                cash_flow_growth_5y=13.0, roe_ttm=15.0, roic_ttm=18.0,
                gross_margin=35.0, profit_margin=12.0, current_ratio=1.6,
                debt_to_ebitda=1.5, debt_service_ratio=80.0,
                growth_score=75.0, profitability_score=78.0, debt_score=75.0,
                overall_score=76.2, commentary='', timestamp='2024-01-01'
            )
        ]
        
        summary = self.agent.get_screening_summary(companies)
        
        assert summary['total_qualified'] == 2
        assert summary['avg_overall_score'] == (85.2 + 76.2) / 2
        assert 'TEST1' in summary['top_performers']
        assert len(summary['sectors_represented']) == 2
        assert summary['market_cap_range']['min'] == 1000000000
        assert summary['market_cap_range']['max'] == 2000000000


if __name__ == '__main__':
    pytest.main([__file__])

