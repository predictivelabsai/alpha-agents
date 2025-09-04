#!/usr/bin/env python3
"""
Comprehensive test runner for Alpha Agents system.
Runs all unit tests and generates test data.
"""

import unittest
import sys
import os
import json
import csv
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents import (
    Stock, RiskTolerance, InvestmentDecision,
    FundamentalAgent, SentimentAgent, ValuationAgent, 
    RationaleAgent, SecularTrendAgent,
    create_multi_agent_portfolio_system
)

class AlphaAgentsTestRunner:
    """Comprehensive test runner for Alpha Agents system"""
    
    def __init__(self):
        self.test_results = {}
        self.test_data = []
        self.start_time = None
        self.end_time = None
        
        # Create test data directory
        self.test_data_dir = Path(__file__).parent.parent / "test-data"
        self.test_data_dir.mkdir(exist_ok=True)
        
        # Test stocks for comprehensive testing
        self.test_stocks = [
            Stock(
                symbol="AAPL",
                company_name="Apple Inc.",
                sector="Technology",
                current_price=150.0,
                market_cap=2500e9,
                pe_ratio=25.0,
                dividend_yield=0.5,
                beta=1.2,
                volume=50000000
            ),
            Stock(
                symbol="MSFT",
                company_name="Microsoft Corporation",
                sector="Technology",
                current_price=300.0,
                market_cap=2200e9,
                pe_ratio=28.0,
                dividend_yield=0.8,
                beta=0.9,
                volume=25000000
            ),
            Stock(
                symbol="NVDA",
                company_name="NVIDIA Corporation",
                sector="Technology",
                current_price=450.0,
                market_cap=1100e9,
                pe_ratio=65.0,
                dividend_yield=0.1,
                beta=1.7,
                volume=40000000
            ),
            Stock(
                symbol="TSLA",
                company_name="Tesla Inc.",
                sector="Consumer Discretionary",
                current_price=200.0,
                market_cap=650e9,
                pe_ratio=45.0,
                dividend_yield=0.0,
                beta=2.0,
                volume=80000000
            ),
            Stock(
                symbol="JPM",
                company_name="JPMorgan Chase & Co.",
                sector="Financial",
                current_price=140.0,
                market_cap=420e9,
                pe_ratio=12.0,
                dividend_yield=2.5,
                beta=1.1,
                volume=12000000
            ),
            Stock(
                symbol="JNJ",
                company_name="Johnson & Johnson",
                sector="Healthcare",
                current_price=160.0,
                market_cap=430e9,
                pe_ratio=15.0,
                dividend_yield=2.8,
                beta=0.7,
                volume=8000000
            ),
            Stock(
                symbol="XOM",
                company_name="Exxon Mobil Corporation",
                sector="Energy",
                current_price=110.0,
                market_cap=460e9,
                pe_ratio=14.0,
                dividend_yield=5.5,
                beta=1.3,
                volume=20000000
            ),
            Stock(
                symbol="AMZN",
                company_name="Amazon.com Inc.",
                sector="Consumer Discretionary",
                current_price=130.0,
                market_cap=1350e9,
                pe_ratio=50.0,
                dividend_yield=0.0,
                beta=1.4,
                volume=35000000
            )
        ]
    
    def run_unit_tests(self):
        """Run all unit tests"""
        print("🧪 Running Unit Tests")
        print("=" * 50)
        
        # Discover and run all unit tests
        test_loader = unittest.TestLoader()
        test_suite = test_loader.discover('unit', pattern='test_*.py')
        
        # Run tests with detailed output
        test_runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
        result = test_runner.run(test_suite)
        
        # Store results
        self.test_results['unit_tests'] = {
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'success_rate': (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0,
            'details': {
                'failures': [str(failure) for failure in result.failures],
                'errors': [str(error) for error in result.errors]
            }
        }
        
        print(f"\n✅ Unit Tests Complete: {result.testsRun} tests, {len(result.failures)} failures, {len(result.errors)} errors")
        return result.wasSuccessful()
    
    def test_individual_agents(self):
        """Test each agent individually with test stocks"""
        print("\n🤖 Testing Individual Agents")
        print("=" * 50)
        
        agents = {
            'fundamental': FundamentalAgent(RiskTolerance.MODERATE, None),
            'sentiment': SentimentAgent(RiskTolerance.MODERATE, None),
            'valuation': ValuationAgent(RiskTolerance.MODERATE, None),
            'rationale': RationaleAgent(RiskTolerance.MODERATE, None),
            'secular_trend': SecularTrendAgent(RiskTolerance.MODERATE, None)
        }
        
        agent_results = {}
        
        for agent_name, agent in agents.items():
            print(f"\n🔍 Testing {agent_name.title()} Agent...")
            agent_data = []
            
            for stock in self.test_stocks:
                try:
                    start_time = time.time()
                    analysis = agent.analyze(stock)
                    end_time = time.time()
                    
                    # Store analysis data
                    analysis_data = {
                        'timestamp': datetime.now().isoformat(),
                        'agent': agent_name,
                        'stock_symbol': stock.symbol,
                        'company_name': stock.company_name,
                        'sector': stock.sector,
                        'current_price': stock.current_price,
                        'market_cap': stock.market_cap,
                        'recommendation': analysis.recommendation.value,
                        'confidence_score': analysis.confidence_score,
                        'risk_assessment': analysis.risk_assessment,
                        'target_price': analysis.target_price,
                        'key_factors_count': len(analysis.key_factors),
                        'concerns_count': len(analysis.concerns),
                        'analysis_time_seconds': round(end_time - start_time, 3),
                        'reasoning_length': len(analysis.reasoning)
                    }
                    
                    agent_data.append(analysis_data)
                    self.test_data.append(analysis_data)
                    
                    print(f"  ✅ {stock.symbol}: {analysis.recommendation.value} (confidence: {analysis.confidence_score:.2f})")
                    
                except Exception as e:
                    print(f"  ❌ {stock.symbol}: Error - {e}")
                    agent_data.append({
                        'timestamp': datetime.now().isoformat(),
                        'agent': agent_name,
                        'stock_symbol': stock.symbol,
                        'error': str(e)
                    })
            
            # Calculate agent statistics
            successful_analyses = [d for d in agent_data if 'error' not in d]
            if successful_analyses:
                recommendations = [d['recommendation'] for d in successful_analyses]
                confidence_scores = [d['confidence_score'] for d in successful_analyses]
                
                agent_results[agent_name] = {
                    'total_stocks': len(self.test_stocks),
                    'successful_analyses': len(successful_analyses),
                    'success_rate': len(successful_analyses) / len(self.test_stocks),
                    'avg_confidence': sum(confidence_scores) / len(confidence_scores),
                    'recommendation_distribution': {
                        'buy': recommendations.count('buy'),
                        'hold': recommendations.count('hold'),
                        'sell': recommendations.count('sell'),
                        'avoid': recommendations.count('avoid')
                    },
                    'avg_analysis_time': sum(d['analysis_time_seconds'] for d in successful_analyses) / len(successful_analyses)
                }
            else:
                agent_results[agent_name] = {
                    'total_stocks': len(self.test_stocks),
                    'successful_analyses': 0,
                    'success_rate': 0.0
                }
        
        self.test_results['individual_agents'] = agent_results
        print(f"\n✅ Individual Agent Testing Complete")
        return True
    
    def test_multi_agent_system(self):
        """Test the multi-agent system integration"""
        print("\n🤝 Testing Multi-Agent System Integration")
        print("=" * 50)
        
        try:
            # Create multi-agent system (without API key for testing)
            mas = create_multi_agent_portfolio_system(
                openai_api_key="test-key",
                risk_tolerance="moderate",
                max_debate_rounds=1
            )
            
            print(f"✅ Multi-Agent System Created with {len(mas.agents)} agents")
            
            # Test system properties
            system_results = {
                'agents_count': len(mas.agents),
                'agent_names': list(mas.agents.keys()),
                'risk_tolerance': mas.risk_tolerance.value,
                'max_debate_rounds': mas.max_debate_rounds,
                'workflow_created': mas.workflow is not None
            }
            
            # Test with a subset of stocks (to avoid API calls)
            test_subset = self.test_stocks[:3]
            portfolio_results = []
            
            for stock in test_subset:
                try:
                    # Test individual agent analyses within the system
                    stock_analyses = {}
                    for agent_name, agent in mas.agents.items():
                        analysis = agent.analyze(stock)
                        stock_analyses[agent_name] = {
                            'recommendation': analysis.recommendation.value,
                            'confidence': analysis.confidence_score,
                            'risk': analysis.risk_assessment
                        }
                    
                    portfolio_results.append({
                        'stock_symbol': stock.symbol,
                        'agent_analyses': stock_analyses,
                        'consensus_possible': len(set(a['recommendation'] for a in stock_analyses.values())) <= 2
                    })
                    
                    print(f"  ✅ {stock.symbol}: All {len(mas.agents)} agents analyzed successfully")
                    
                except Exception as e:
                    print(f"  ❌ {stock.symbol}: Error - {e}")
            
            system_results['portfolio_analysis'] = portfolio_results
            self.test_results['multi_agent_system'] = system_results
            
            print(f"✅ Multi-Agent System Testing Complete")
            return True
            
        except Exception as e:
            print(f"❌ Multi-Agent System Error: {e}")
            self.test_results['multi_agent_system'] = {'error': str(e)}
            return False
    
    def generate_test_reports(self):
        """Generate comprehensive test reports"""
        print("\n📊 Generating Test Reports")
        print("=" * 50)
        
        # Save test results as JSON
        results_file = self.test_data_dir / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        print(f"✅ Test results saved to: {results_file}")
        
        # Save test data as CSV
        if self.test_data:
            csv_file = self.test_data_dir / f"agent_analysis_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            # Get all unique keys for CSV headers
            all_keys = set()
            for data in self.test_data:
                all_keys.update(data.keys())
            
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
                writer.writeheader()
                writer.writerows(self.test_data)
            
            print(f"✅ Test data saved to: {csv_file}")
        
        # Generate summary report
        summary_file = self.test_data_dir / f"test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(summary_file, 'w') as f:
            f.write("Alpha Agents System Test Summary\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Test Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {(self.end_time - self.start_time):.2f} seconds\n\n")
            
            # Unit test results
            if 'unit_tests' in self.test_results:
                ut = self.test_results['unit_tests']
                f.write(f"Unit Tests:\n")
                f.write(f"  Tests Run: {ut['tests_run']}\n")
                f.write(f"  Success Rate: {ut['success_rate']:.1%}\n")
                f.write(f"  Failures: {ut['failures']}\n")
                f.write(f"  Errors: {ut['errors']}\n\n")
            
            # Individual agent results
            if 'individual_agents' in self.test_results:
                f.write("Individual Agent Performance:\n")
                for agent_name, results in self.test_results['individual_agents'].items():
                    f.write(f"  {agent_name.title()} Agent:\n")
                    f.write(f"    Success Rate: {results['success_rate']:.1%}\n")
                    if 'avg_confidence' in results:
                        f.write(f"    Avg Confidence: {results['avg_confidence']:.2f}\n")
                        f.write(f"    Avg Analysis Time: {results['avg_analysis_time']:.3f}s\n")
                        rec_dist = results['recommendation_distribution']
                        f.write(f"    Recommendations: Buy({rec_dist['buy']}) Hold({rec_dist['hold']}) Sell({rec_dist['sell']}) Avoid({rec_dist['avoid']})\n")
                    f.write("\n")
            
            # Multi-agent system results
            if 'multi_agent_system' in self.test_results:
                mas = self.test_results['multi_agent_system']
                if 'error' not in mas:
                    f.write("Multi-Agent System:\n")
                    f.write(f"  Agents: {mas['agents_count']} ({', '.join(mas['agent_names'])})\n")
                    f.write(f"  Risk Tolerance: {mas['risk_tolerance']}\n")
                    f.write(f"  Workflow Created: {mas['workflow_created']}\n")
                    if 'portfolio_analysis' in mas:
                        consensus_count = sum(1 for r in mas['portfolio_analysis'] if r['consensus_possible'])
                        f.write(f"  Consensus Possible: {consensus_count}/{len(mas['portfolio_analysis'])} stocks\n")
        
        print(f"✅ Test summary saved to: {summary_file}")
    
    def run_all_tests(self):
        """Run all tests and generate reports"""
        self.start_time = time.time()
        
        print("🚀 Alpha Agents Comprehensive Testing")
        print("=" * 60)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Test Stocks: {len(self.test_stocks)}")
        print()
        
        # Run all test phases
        unit_success = self.run_unit_tests()
        agent_success = self.test_individual_agents()
        system_success = self.test_multi_agent_system()
        
        self.end_time = time.time()
        
        # Generate reports
        self.generate_test_reports()
        
        # Final summary
        print(f"\n🎯 Testing Complete!")
        print("=" * 60)
        print(f"Duration: {(self.end_time - self.start_time):.2f} seconds")
        print(f"Unit Tests: {'✅ PASS' if unit_success else '❌ FAIL'}")
        print(f"Agent Tests: {'✅ PASS' if agent_success else '❌ FAIL'}")
        print(f"System Tests: {'✅ PASS' if system_success else '❌ FAIL'}")
        print(f"Test Data Generated: {len(self.test_data)} records")
        print(f"Reports Saved: {self.test_data_dir}")
        
        return unit_success and agent_success and system_success

if __name__ == "__main__":
    runner = AlphaAgentsTestRunner()
    success = runner.run_all_tests()
    sys.exit(0 if success else 1)

