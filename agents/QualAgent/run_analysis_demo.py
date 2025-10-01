#!/usr/bin/env python3
"""
QualAgent Analysis Demonstration Script

This script demonstrates how to run qualitative analysis for technology companies
using the QualAgent system with various configurable options.

Features:
- Load example companies from JSON data
- Configure analysis parameters (models, themes, geographies)
- Run individual or batch analysis
- Save results to unified output file
- Display progress and cost tracking

Usage:
    python run_analysis_demo.py --help
    python run_analysis_demo.py --single NVDA
    python run_analysis_demo.py --batch --max-companies 3
    python run_analysis_demo.py --load-examples
"""

import json
import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from models.json_data_manager import JSONDataManager, Company
from engines.analysis_engine import AnalysisEngine, AnalysisConfig, AnalysisResult
from engines.llm_integration import LLMIntegration
from engines.tools_integration import ToolsIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QualAgentDemo:
    """Demonstration runner for QualAgent analysis system"""

    def __init__(self, output_file: str = None):
        """Initialize the demo system"""
        self.output_file = output_file or f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Initialize components
        self.db = JSONDataManager()
        self.engine = AnalysisEngine()
        self.llm = LLMIntegration()
        self.tools = ToolsIntegration()

        # Results storage
        self.batch_results = []

        logger.info(f"QualAgent Demo initialized. Results will be saved to: {self.output_file}")

    def load_example_companies(self) -> int:
        """Load example companies from JSON file into the system"""
        logger.info("Loading example companies...")

        example_file = Path(__file__).parent / "data" / "example_companies.json"

        if not example_file.exists():
            logger.error(f"Example companies file not found: {example_file}")
            return 0

        with open(example_file, 'r') as f:
            example_companies_data = json.load(f)

        added_count = 0
        for company_data in example_companies_data:
            try:
                # Check if company already exists
                existing = self.db.get_company_by_ticker(company_data['ticker'])
                if existing:
                    logger.info(f"Company {company_data['ticker']} already exists, skipping")
                    continue

                # Create Company object
                company = Company(
                    company_name=company_data['company_name'],
                    ticker=company_data['ticker'],
                    subsector=company_data['subsector'],
                    market_cap_usd=company_data.get('market_cap_usd'),
                    employees=company_data.get('employees'),
                    founded_year=company_data.get('founded_year'),
                    headquarters=company_data.get('headquarters'),
                    description=company_data.get('description'),
                    website=company_data.get('website')
                )

                # Add to database
                self.db.add_company(company)
                added_count += 1
                logger.info(f"Added: {company.company_name} ({company.ticker})")

            except Exception as e:
                logger.error(f"Failed to add company {company_data.get('ticker', 'Unknown')}: {e}")

        logger.info(f"Successfully loaded {added_count} example companies")
        return added_count

    def display_available_companies(self):
        """Display all available companies for analysis"""
        companies = self.db.list_companies()

        if not companies:
            logger.warning("No companies available. Run --load-examples first.")
            return

        print("\n" + "="*80)
        print("AVAILABLE COMPANIES FOR ANALYSIS")
        print("="*80)
        print(f"{'Ticker':<8} | {'Company Name':<35} | {'Subsector':<20} | {'Market Cap':<12}")
        print("-"*80)

        for company in companies:
            market_cap = f"${company.market_cap_usd/1e9:.1f}B" if company.market_cap_usd else "N/A"
            print(f"{company.ticker:<8} | {company.company_name[:35]:<35} | {company.subsector:<20} | {market_cap:<12}")

        print("-"*80)
        print(f"Total: {len(companies)} companies")

    def get_analysis_configuration(self,
                                 models: List[str] = None,
                                 focus_themes: List[str] = None,
                                 geographies: List[str] = None,
                                 enable_consensus: bool = True) -> AnalysisConfig:
        """Create analysis configuration with specified parameters"""

        # Default models if none specified
        if not models:
            available_models = self.llm.get_recommended_models()
            models = available_models[:1] if available_models else ['gpt-4o-mini']

        # Default focus themes
        if not focus_themes:
            focus_themes = []

        # Default geographies
        if not geographies:
            geographies = ['US', 'Global']

        config = AnalysisConfig(
            models_to_use=models,
            focus_themes=focus_themes,
            geographies_of_interest=geographies,
            enable_multi_model_consensus=enable_consensus,
            requested_by='QualAgent-Demo'
        )

        return config

    def estimate_analysis_cost(self, config: AnalysisConfig, num_companies: int = 1) -> Dict[str, float]:
        """Estimate costs for analysis configuration"""

        # LLM costs
        llm_cost_per_company = 0.0
        for model in config.models_to_use:
            model_info = self.llm.get_model_info(model)
            if model_info:
                # Rough estimate: 8000 tokens per analysis
                estimated_cost = (8000 / 1000) * model_info.cost_per_1k_tokens
                llm_cost_per_company += estimated_cost

        # Tools costs
        tools_cost_per_company = self.tools.estimate_tool_costs(
            self.tools.get_available_tools(),
            calls_per_tool=3  # Estimated 3 calls per tool per analysis
        )

        total_per_company = llm_cost_per_company + tools_cost_per_company
        total_batch_cost = total_per_company * num_companies

        return {
            'llm_cost_per_company': llm_cost_per_company,
            'tools_cost_per_company': tools_cost_per_company,
            'total_per_company': total_per_company,
            'total_batch_cost': total_batch_cost,
            'num_companies': num_companies,
            'models_used': len(config.models_to_use)
        }

    def run_single_analysis(self,
                          ticker: str,
                          models: List[str] = None,
                          focus_themes: List[str] = None,
                          geographies: List[str] = None,
                          enable_consensus: bool = True) -> Optional[AnalysisResult]:
        """Run analysis for a single company"""

        logger.info(f"Starting analysis for {ticker}")

        # Get company
        company = self.db.get_company_by_ticker(ticker.upper())
        if not company:
            logger.error(f"Company {ticker} not found in database")
            return None

        # Create configuration
        config = self.get_analysis_configuration(models, focus_themes, geographies, enable_consensus)

        # Estimate cost
        cost_estimate = self.estimate_analysis_cost(config)

        print(f"\n{'='*60}")
        print(f"ANALYSIS CONFIGURATION FOR {company.company_name} ({ticker})")
        print(f"{'='*60}")
        print(f"Models to use: {', '.join(config.models_to_use)}")
        print(f"Focus themes: {', '.join(config.focus_themes) if config.focus_themes else 'General analysis'}")
        print(f"Geographies: {', '.join(config.geographies_of_interest)}")
        print(f"Multi-model consensus: {config.enable_multi_model_consensus}")
        print(f"Estimated cost: ${cost_estimate['total_per_company']:.4f}")
        print(f"{'='*60}")

        # Get user confirmation for cost
        if cost_estimate['total_per_company'] > 0.10:  # Alert for costs over $0.10
            confirm = input(f"This analysis will cost approximately ${cost_estimate['total_per_company']:.4f}. Continue? (y/N): ")
            if confirm.lower() != 'y':
                logger.info("Analysis cancelled by user")
                return None

        try:
            # Run analysis
            start_time = time.time()
            result = self.engine.analyze_company(ticker.upper(), config)
            end_time = time.time()

            # Display results summary
            self._display_single_result(result, end_time - start_time)

            # Save to batch results
            self.batch_results.append({
                'timestamp': datetime.now().isoformat(),
                'company': result.company.__dict__,
                'configuration': config.__dict__,
                'result_summary': {
                    'request_id': result.request_id,
                    'success_rate': result.success_rate,
                    'total_cost_usd': result.total_cost_usd,
                    'processing_time': result.total_processing_time,
                    'models_used': len(result.llm_analyses),
                    'analyses_completed': len([a for a in result.llm_analyses if a.analysis_status == 'completed'])
                },
                'detailed_results': {
                    'llm_analyses': [a.__dict__ for a in result.llm_analyses],
                    'parsed_results': result.parsed_results,
                    'consensus_analysis': result.consensus_analysis
                }
            })

            # Save results to file
            self.save_batch_results()

            return result

        except Exception as e:
            logger.error(f"Analysis failed for {ticker}: {e}")
            return None

    def run_batch_analysis(self,
                         max_companies: int = 5,
                         models: List[str] = None,
                         focus_themes: List[str] = None,
                         geographies: List[str] = None,
                         subsector_filter: str = None) -> List[AnalysisResult]:
        """Run batch analysis for multiple companies"""

        # Get companies to analyze
        companies = self.db.list_companies(subsector=subsector_filter, limit=max_companies)

        if not companies:
            logger.error("No companies available for batch analysis")
            return []

        # Create configuration
        config = self.get_analysis_configuration(models, focus_themes, geographies, enable_consensus=False)  # Disable consensus for batch to save cost

        # Estimate total cost
        cost_estimate = self.estimate_analysis_cost(config, len(companies))

        print(f"\n{'='*70}")
        print(f"BATCH ANALYSIS CONFIGURATION")
        print(f"{'='*70}")
        print(f"Companies to analyze: {len(companies)}")
        print(f"Models per company: {', '.join(config.models_to_use)}")
        print(f"Focus themes: {', '.join(config.focus_themes) if config.focus_themes else 'General analysis'}")
        print(f"Geographies: {', '.join(config.geographies_of_interest)}")
        print(f"Subsector filter: {subsector_filter or 'All subsectors'}")
        print(f"Estimated total cost: ${cost_estimate['total_batch_cost']:.4f}")
        print(f"Average cost per company: ${cost_estimate['total_per_company']:.4f}")
        print(f"{'='*70}")

        # Show companies to be analyzed
        print("\\nCompanies to analyze:")
        for i, company in enumerate(companies, 1):
            print(f"  {i}. {company.ticker} - {company.company_name} ({company.subsector})")

        # Get user confirmation
        if cost_estimate['total_batch_cost'] > 0.50:  # Alert for batch costs over $0.50
            confirm = input(f"\\nThis batch analysis will cost approximately ${cost_estimate['total_batch_cost']:.4f}. Continue? (y/N): ")
            if confirm.lower() != 'y':
                logger.info("Batch analysis cancelled by user")
                return []

        # Run batch analysis
        results = []
        total_cost = 0.0
        start_time = time.time()

        for i, company in enumerate(companies, 1):
            print(f"\\n{'='*50}")
            print(f"ANALYZING {i}/{len(companies)}: {company.ticker}")
            print(f"{'='*50}")

            try:
                result = self.engine.analyze_company(company.ticker, config)
                results.append(result)
                total_cost += result.total_cost_usd

                # Brief result summary
                print(f"✓ {company.ticker} analysis completed: ${result.total_cost_usd:.4f}, {result.total_processing_time:.1f}s")

                # Add to batch results
                self.batch_results.append({
                    'timestamp': datetime.now().isoformat(),
                    'company': result.company.__dict__,
                    'configuration': config.__dict__,
                    'result_summary': {
                        'request_id': result.request_id,
                        'success_rate': result.success_rate,
                        'total_cost_usd': result.total_cost_usd,
                        'processing_time': result.total_processing_time,
                        'models_used': len(result.llm_analyses),
                        'analyses_completed': len([a for a in result.llm_analyses if a.analysis_status == 'completed'])
                    },
                    'detailed_results': {
                        'llm_analyses': [a.__dict__ for a in result.llm_analyses],
                        'parsed_results': result.parsed_results
                    }
                })

                # Small delay between analyses
                time.sleep(2)

            except Exception as e:
                logger.error(f"Analysis failed for {company.ticker}: {e}")

        # Display batch summary
        end_time = time.time()
        self._display_batch_summary(results, total_cost, end_time - start_time)

        # Save all results
        self.save_batch_results()

        return results

    def _display_single_result(self, result: AnalysisResult, processing_time: float):
        """Display summary of single analysis result"""

        print(f"\\n{'='*60}")
        print(f"ANALYSIS COMPLETED: {result.company.company_name}")
        print(f"{'='*60}")
        print(f"Request ID: {result.request_id}")
        print(f"Success Rate: {result.success_rate:.1%}")
        print(f"Total Cost: ${result.total_cost_usd:.4f}")
        print(f"Processing Time: {processing_time:.1f} seconds")
        print(f"Models Used: {len(result.llm_analyses)}")

        if result.llm_analyses:
            print("\\nModel Results:")
            for analysis in result.llm_analyses:
                status = "✓" if analysis.analysis_status == 'completed' else "✗"
                cost = analysis.cost_usd if analysis.cost_usd is not None else 0.0
                time_taken = analysis.processing_time_seconds if analysis.processing_time_seconds is not None else 0.0
                print(f"  {status} {analysis.llm_model}: ${cost:.4f}, {time_taken:.1f}s")

        if result.consensus_analysis:
            print("✓ Consensus analysis generated")

        print(f"\\nDetailed results saved to: {self.output_file}")
        print(f"{'='*60}")

    def _display_batch_summary(self, results: List[AnalysisResult], total_cost: float, total_time: float):
        """Display summary of batch analysis results"""

        successful = [r for r in results if r.success_rate > 0.5]

        print(f"\\n{'='*70}")
        print(f"BATCH ANALYSIS SUMMARY")
        print(f"{'='*70}")
        print(f"Companies analyzed: {len(results)}")
        print(f"Successful analyses: {len(successful)}")
        print(f"Total cost: ${total_cost:.4f}")
        print(f"Average cost per company: ${total_cost / len(results):.4f}")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Average time per company: {total_time / len(results):.1f} seconds")

        print("\\nResults by company:")
        for result in results:
            status = "✓" if result.success_rate > 0.5 else "✗"
            cost = result.total_cost_usd if result.total_cost_usd is not None else 0.0
            time_taken = result.total_processing_time if result.total_processing_time is not None else 0.0
            print(f"  {status} {result.company.ticker}: ${cost:.4f}, {time_taken:.1f}s")

        print(f"\\nAll results saved to: {self.output_file}")
        print(f"{'='*70}")

    def save_batch_results(self):
        """Save all batch results to unified output file"""
        output_data = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_analyses': len(self.batch_results),
                'qualagent_version': '2.0',
                'output_file': self.output_file
            },
            'analyses': self.batch_results
        }

        with open(self.output_file, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)

        logger.info(f"Results saved to {self.output_file}")

    def export_results_summary_csv(self) -> str:
        """Export results summary to CSV format"""
        if not self.batch_results:
            logger.warning("No results to export")
            return ""

        # Create summary data
        summary_data = []
        for result in self.batch_results:
            summary_data.append({
                'timestamp': result['timestamp'],
                'ticker': result['company']['ticker'],
                'company_name': result['company']['company_name'],
                'subsector': result['company']['subsector'],
                'success_rate': result['result_summary']['success_rate'],
                'total_cost_usd': result['result_summary']['total_cost_usd'],
                'processing_time': result['result_summary']['processing_time'],
                'models_used': result['result_summary']['models_used'],
                'analyses_completed': result['result_summary']['analyses_completed']
            })

        # Create DataFrame and save
        df = pd.DataFrame(summary_data)
        csv_file = self.output_file.replace('.json', '_summary.csv')
        df.to_csv(csv_file, index=False)

        logger.info(f"Summary exported to {csv_file}")
        return csv_file

    def display_system_status(self):
        """Display current system status and capabilities"""

        # Check API keys
        api_validation = self.llm.validate_api_keys()
        tools_validation = self.tools.validate_tool_access()

        # Get available models
        available_models = self.llm.get_available_models()
        recommended_models = self.llm.get_recommended_models()

        # Get companies count
        companies = self.db.list_companies()

        print(f"\\n{'='*70}")
        print(f"QUALAGENT SYSTEM STATUS")
        print(f"{'='*70}")

        print(f"Data Storage: JSON-based ({len(companies)} companies loaded)")
        print(f"Output File: {self.output_file}")

        print("\\nLLM Providers:")
        for provider, is_valid in api_validation.items():
            status = "✓ Connected" if is_valid else "○ Not configured"
            print(f"  {provider}: {status}")

        print(f"\\nAvailable Models: {len(available_models)}")
        for model in available_models:
            model_info = self.llm.get_model_info(model)
            cost = f"${model_info.cost_per_1k_tokens:.4f}/1k tokens" if model_info else "Cost unknown"
            print(f"  {model}: {cost}")

        print(f"\\nRecommended Models: {', '.join(recommended_models) if recommended_models else 'None available'}")

        print("\\nResearch Tools:")
        for tool_name, is_valid in tools_validation.items():
            status = "✓ Available" if is_valid else "○ Reference only"
            print(f"  {tool_name}: {status}")

        print(f"\\nCompanies by Subsector:")
        subsectors = {}
        for company in companies:
            subsectors[company.subsector] = subsectors.get(company.subsector, 0) + 1

        for subsector, count in sorted(subsectors.items()):
            print(f"  {subsector}: {count} companies")

        print(f"{'='*70}")

def main():
    """Main entry point for the demonstration script"""

    parser = argparse.ArgumentParser(
        description='QualAgent Analysis Demonstration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_analysis_demo.py --load-examples                    # Load example companies
  python run_analysis_demo.py --list                            # Show available companies
  python run_analysis_demo.py --status                          # Show system status
  python run_analysis_demo.py --single NVDA                     # Analyze NVIDIA
  python run_analysis_demo.py --single NVDA --models llama-3-70b # Use specific model
  python run_analysis_demo.py --batch --max-companies 3         # Batch analyze 3 companies
  python run_analysis_demo.py --batch --subsector Semiconductors # Analyze semiconductor companies
        """
    )

    # Main actions
    parser.add_argument('--load-examples', action='store_true',
                       help='Load example companies into the system')
    parser.add_argument('--list', action='store_true',
                       help='List all available companies')
    parser.add_argument('--status', action='store_true',
                       help='Show system status and capabilities')
    parser.add_argument('--single', type=str, metavar='TICKER',
                       help='Run analysis for a single company')
    parser.add_argument('--batch', action='store_true',
                       help='Run batch analysis for multiple companies')

    # Analysis configuration
    parser.add_argument('--models', nargs='+',
                       help='LLM models to use (e.g., llama-3-70b mixtral-8x7b)')
    parser.add_argument('--focus-themes', nargs='+',
                       help='Focus themes for analysis')
    parser.add_argument('--geographies', nargs='+',
                       help='Geographic markets to focus on')
    parser.add_argument('--no-consensus', action='store_true',
                       help='Disable multi-model consensus (for single analysis)')

    # Batch options
    parser.add_argument('--max-companies', type=int, default=5,
                       help='Maximum companies for batch analysis')
    parser.add_argument('--subsector', type=str,
                       help='Filter companies by subsector for batch analysis')

    # Output options
    parser.add_argument('--output', type=str,
                       help='Output file name (default: timestamped)')
    parser.add_argument('--export-csv', action='store_true',
                       help='Export results summary to CSV')

    args = parser.parse_args()

    # Initialize demo system
    demo = QualAgentDemo(output_file=args.output)

    try:
        if args.load_examples:
            count = demo.load_example_companies()
            print(f"Loaded {count} example companies into the system")

        elif args.list:
            demo.display_available_companies()

        elif args.status:
            demo.display_system_status()

        elif args.single:
            result = demo.run_single_analysis(
                ticker=args.single,
                models=args.models,
                focus_themes=args.focus_themes,
                geographies=args.geographies,
                enable_consensus=not args.no_consensus
            )

            if result and args.export_csv:
                demo.export_results_summary_csv()

        elif args.batch:
            results = demo.run_batch_analysis(
                max_companies=args.max_companies,
                models=args.models,
                focus_themes=args.focus_themes,
                geographies=args.geographies,
                subsector_filter=args.subsector
            )

            if results and args.export_csv:
                demo.export_results_summary_csv()

        else:
            parser.print_help()

    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()