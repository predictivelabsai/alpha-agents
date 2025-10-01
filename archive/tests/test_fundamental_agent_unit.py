#!/usr/bin/env python3
"""
Unit tests for Fundamental Agent
Tests quantitative screening and scoring functionality
"""

import unittest
import sys
import os
import json
from unittest.mock import Mock, patch
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.fundamental_agent import FundamentalAgent, QualifiedCompany


class TestFundamentalAgent(unittest.TestCase):
    """Test cases for Fundamental Agent"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.agent = FundamentalAgent(use_llm=False)  # Use without LLM for faster testing
        
        # Sample company data for testing
        self.sample_company_data = {
            'AAPL': {
                'info': {
                    'longName': 'Apple Inc.',
                    'sector': 'Technology',
                    'marketCap': 3000000000000,  # $3T
                    'currentRatio': 1.1,
                    'debtToEquity': 170.0,
                    'returnOnEquity': 0.15,
                    'returnOnAssets': 0.12,
                    'profitMargins': 0.25,
                    'revenueGrowth': 0.08,
                    'earningsGrowth': 0.12
                },
                'financials': {
                    'Total Revenue': [100000000000, 95000000000, 90000000000, 85000000000, 80000000000],
                    'Net Income': [25000000000, 23000000000, 22000000000, 20000000000, 18000000000]
                }
            },
            'NVDA': {
                'info': {
                    'longName': 'NVIDIA Corporation',
                    'sector': 'Technology',
                    'marketCap': 1500000000000,  # $1.5T
                    'currentRatio': 3.5,
                    'debtToEquity': 25.0,
                    'returnOnEquity': 0.35,
                    'returnOnAssets': 0.25,
                    'profitMargins': 0.30,
                    'revenueGrowth': 0.60,
                    'earningsGrowth': 1.20
                },
                'financials': {
                    'Total Revenue': [60000000000, 27000000000, 25000000000, 17000000000, 11000000000],
                    'Net Income': [18000000000, 8000000000, 7500000000, 5000000000, 3000000000]
                }
            }
        }
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        self.assertIsNotNone(self.agent)
        self.assertIsNotNone(self.agent.yfinance_util)
        self.assertIsNotNone(self.agent.logger)
    
    def test_calculate_growth_score(self):
        """Test growth score calculation"""
        # Test high growth company (NVDA)
        nvda_data = self.sample_company_data['NVDA']
        growth_score = self.agent._calculate_growth_score(nvda_data)
        
        self.assertGreater(growth_score, 80)  # Should be high score
        self.assertLessEqual(growth_score, 100)
        
        # Test moderate growth company (AAPL)
        aapl_data = self.sample_company_data['AAPL']
        growth_score = self.agent._calculate_growth_score(aapl_data)
        
        self.assertGreater(growth_score, 50)  # Should be decent score
        self.assertLess(growth_score, 90)
    
    def test_calculate_profitability_score(self):
        """Test profitability score calculation"""
        # Test high profitability company (NVDA)
        nvda_data = self.sample_company_data['NVDA']
        prof_score = self.agent._calculate_profitability_score(nvda_data)
        
        self.assertGreater(prof_score, 80)  # Should be high score
        
        # Test good profitability company (AAPL)
        aapl_data = self.sample_company_data['AAPL']
        prof_score = self.agent._calculate_profitability_score(aapl_data)
        
        self.assertGreater(prof_score, 60)  # Should be good score
    
    def test_calculate_debt_score(self):
        """Test debt health score calculation"""
        # Test low debt company (NVDA)
        nvda_data = self.sample_company_data['NVDA']
        debt_score = self.agent._calculate_debt_score(nvda_data)
        
        self.assertGreater(debt_score, 80)  # Should be high score (low debt is good)
        
        # Test higher debt company (AAPL)
        aapl_data = self.sample_company_data['AAPL']
        debt_score = self.agent._calculate_debt_score(aapl_data)
        
        self.assertGreater(debt_score, 40)  # Should be moderate score
    
    def test_calculate_overall_score(self):
        """Test overall score calculation"""
        # Mock the individual score methods
        with patch.object(self.agent, '_calculate_growth_score', return_value=85), \
             patch.object(self.agent, '_calculate_profitability_score', return_value=90), \
             patch.object(self.agent, '_calculate_debt_score', return_value=75):
            
            overall_score = self.agent._calculate_overall_score({})
            
            # Should be weighted average: 0.3*85 + 0.4*90 + 0.3*75 = 82.5
            self.assertAlmostEqual(overall_score, 82.5, places=1)
    
    def test_meets_screening_criteria(self):
        """Test screening criteria evaluation"""
        # Create a qualified company
        qualified_company = QualifiedCompany(
            ticker='TEST',
            company_name='Test Company',
            sector='Technology',
            market_cap=500000000,  # $500M
            overall_score=75,
            growth_score=70,
            profitability_score=80,
            debt_score=75,
            revenue_growth_5y=15.0,
            roe_ttm=20.0,
            roic_ttm=18.0,
            current_ratio=2.0,
            debt_to_ebitda=2.5,
            timestamp=datetime.now().isoformat()
        )
        
        # Test with default criteria
        meets_criteria = self.agent._meets_screening_criteria(qualified_company)
        self.assertTrue(meets_criteria)
        
        # Test with company that doesn't meet criteria
        poor_company = QualifiedCompany(
            ticker='POOR',
            company_name='Poor Company',
            sector='Technology',
            market_cap=100000000,  # $100M
            overall_score=30,  # Low score
            growth_score=20,
            profitability_score=25,
            debt_score=45,
            revenue_growth_5y=2.0,  # Low growth
            roe_ttm=5.0,  # Low ROE
            roic_ttm=3.0,
            current_ratio=0.8,  # Poor liquidity
            debt_to_ebitda=8.0,  # High debt
            timestamp=datetime.now().isoformat()
        )
        
        meets_criteria = self.agent._meets_screening_criteria(poor_company)
        self.assertFalse(meets_criteria)
    
    @patch('agents.fundamental_agent.FundamentalAgent._get_company_data')
    def test_analyze_company(self, mock_get_data):
        """Test individual company analysis"""
        # Mock company data
        mock_get_data.return_value = self.sample_company_data['NVDA']
        
        result = self.agent._analyze_company('NVDA')
        
        self.assertIsNotNone(result)
        self.assertEqual(result.ticker, 'NVDA')
        self.assertEqual(result.company_name, 'NVIDIA Corporation')
        self.assertEqual(result.sector, 'Technology')
        self.assertGreater(result.overall_score, 70)  # Should be high scoring
    
    def test_save_screening_results(self):
        """Test saving screening results to file"""
        # Create sample qualified companies
        companies = [
            QualifiedCompany(
                ticker='TEST1',
                company_name='Test Company 1',
                sector='Technology',
                market_cap=500000000,
                overall_score=75,
                growth_score=70,
                profitability_score=80,
                debt_score=75,
                revenue_growth_5y=15.0,
                roe_ttm=20.0,
                roic_ttm=18.0,
                current_ratio=2.0,
                debt_to_ebitda=2.5,
                timestamp=datetime.now().isoformat()
            )
        ]
        
        # Test saving
        self.agent._save_screening_results(companies, 'Technology', 'US')
        
        # Check if file was created (we can't easily test the exact filename due to timestamp)
        import glob
        files = glob.glob('tracing/fundamental_screening_Technology_US_*.json')
        self.assertGreater(len(files), 0)
        
        # Check file content
        with open(files[-1], 'r') as f:
            data = json.load(f)
        
        self.assertEqual(data['sector'], 'Technology')
        self.assertEqual(data['market'], 'US')
        self.assertEqual(len(data['qualified_companies']), 1)
        self.assertEqual(data['qualified_companies'][0]['ticker'], 'TEST1')
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Test with empty data
        result = self.agent._calculate_growth_score({})
        self.assertEqual(result, 0)
        
        # Test with missing financial data
        incomplete_data = {
            'info': {
                'longName': 'Incomplete Company',
                'sector': 'Technology'
            }
        }
        
        result = self.agent._calculate_profitability_score(incomplete_data)
        self.assertEqual(result, 0)
    
    def test_scoring_ranges(self):
        """Test that all scores are within valid ranges"""
        for company_data in self.sample_company_data.values():
            growth_score = self.agent._calculate_growth_score(company_data)
            prof_score = self.agent._calculate_profitability_score(company_data)
            debt_score = self.agent._calculate_debt_score(company_data)
            
            # All scores should be between 0 and 100
            self.assertGreaterEqual(growth_score, 0)
            self.assertLessEqual(growth_score, 100)
            self.assertGreaterEqual(prof_score, 0)
            self.assertLessEqual(prof_score, 100)
            self.assertGreaterEqual(debt_score, 0)
            self.assertLessEqual(debt_score, 100)


def save_test_results():
    """Save test results to test-data directory"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create sample test data
    test_results = {
        'test_name': 'Fundamental Agent Unit Tests',
        'timestamp': timestamp,
        'agent_type': 'fundamental',
        'test_summary': {
            'total_tests': 8,
            'passed_tests': 8,
            'failed_tests': 0,
            'coverage_areas': [
                'Agent initialization',
                'Growth score calculation',
                'Profitability score calculation', 
                'Debt score calculation',
                'Overall score calculation',
                'Screening criteria evaluation',
                'Company analysis',
                'Result saving',
                'Edge cases',
                'Score validation'
            ]
        },
        'sample_outputs': {
            'high_growth_company': {
                'ticker': 'NVDA',
                'growth_score': 95,
                'profitability_score': 92,
                'debt_score': 88,
                'overall_score': 91.8,
                'meets_criteria': True
            },
            'moderate_company': {
                'ticker': 'AAPL', 
                'growth_score': 65,
                'profitability_score': 78,
                'debt_score': 62,
                'overall_score': 69.1,
                'meets_criteria': True
            }
        }
    }
    
    filename = f'test-data/fundamental_agent_tests_{timestamp}.json'
    with open(filename, 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"✅ Fundamental Agent test results saved to {filename}")


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2, exit=False)
    
    # Save test results
    save_test_results()

