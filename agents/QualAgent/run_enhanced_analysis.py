#!/usr/bin/env python3
"""
Enhanced QualAgent Analysis Script
Comprehensive qualitative analysis with multi-LLM support, enhanced scoring, and human feedback

Features:
- Multi-LLM analysis across 5 models
- Enhanced scoring system for all analysis components
- Weight approval system
- Human feedback integration
- Multiple output formats (JSON, CSV, PKL)
- Interactive weight configuration
- Cost estimation and tracking

Usage:
    python run_enhanced_analysis.py --help
    python run_enhanced_analysis.py --company NVDA --user-id analyst1
    python run_enhanced_analysis.py --company AAPL --expert-id expert1 --enable-feedback
    python run_enhanced_analysis.py --batch --companies NVDA,AAPL,MSFT --analysis-type comprehensive
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from engines.enhanced_analysis_controller import (
    EnhancedAnalysisController,
    EnhancedAnalysisConfig,
    EnhancedAnalysisResult
)
from engines.enhanced_scoring_system import WeightingScheme
from models.json_data_manager import JSONDataManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedQualAgentDemo:
    """Enhanced demonstration runner for QualAgent analysis system"""

    def __init__(self, auto_approve: bool = False):
        """Initialize the enhanced demo system"""
        self.controller = EnhancedAnalysisController()
        self.db = JSONDataManager()
        self.auto_approve = auto_approve

        logger.info("Enhanced QualAgent Demo initialized")

    def run_single_analysis(self, config: EnhancedAnalysisConfig) -> EnhancedAnalysisResult:
        """Run enhanced analysis for a single company"""
        logger.info(f"Starting enhanced analysis for {config.company_ticker}")

        # Estimate cost first
        cost_estimate = self.controller.estimate_analysis_cost(config)
        logger.info(f"Estimated cost: ${cost_estimate['total_estimated_cost']:.4f}")

        # Get user approval for cost
        if cost_estimate['total_estimated_cost'] > 0.50:  # Threshold for cost approval
            print(f"\nMONEY Cost Estimate: ${cost_estimate['total_estimated_cost']:.4f}")
            print(f"   - Multi-LLM analysis: ${cost_estimate['multi_llm_cost']:.4f}")
            print(f"   - Processing overhead: ${cost_estimate['overhead_cost']:.4f}")
            print(f"   - Models included: {cost_estimate['models_included']}")

            if self.auto_approve:
                print("   - Auto-approving cost (running from Streamlit)")
            else:
                if input("Proceed with analysis? (y/N): ").lower() != 'y':
                    logger.info("Analysis cancelled by user")
                    return None

        # Run the analysis
        result = self.controller.run_enhanced_analysis(config)

        # Display results
        self._display_results(result)

        return result

    def run_batch_analysis(self, companies: List[str], base_config: EnhancedAnalysisConfig) -> List[EnhancedAnalysisResult]:
        """Run enhanced analysis for multiple companies"""
        logger.info(f"Starting batch analysis for {len(companies)} companies")

        results = []
        total_estimated_cost = 0.0

        # Generate single human-readable timestamp for the entire batch
        batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"Batch timestamp: {batch_timestamp} - all companies will use this timestamp for grouping")

        # Estimate total cost
        for ticker in companies:
            config = EnhancedAnalysisConfig(**{**base_config.__dict__, 'company_ticker': ticker, 'batch_timestamp': batch_timestamp})
            cost_estimate = self.controller.estimate_analysis_cost(config)
            total_estimated_cost += cost_estimate['total_estimated_cost']

        print(f"\nMONEY Batch Analysis Cost Estimate: ${total_estimated_cost:.4f}")
        print(f"   - Companies: {len(companies)}")
        print(f"   - Average per company: ${total_estimated_cost/len(companies):.4f}")

        if total_estimated_cost > 2.00:  # Higher threshold for batch
            if self.auto_approve:
                print("   - Auto-approving batch cost (running from Streamlit)")
            else:
                if input("Proceed with batch analysis? (y/N): ").lower() != 'y':
                    logger.info("Batch analysis cancelled by user")
                    return []

        # Run analyses
        for i, ticker in enumerate(companies, 1):
            print(f"\nREFRESH Analyzing {ticker} ({i}/{len(companies)})")

            config = EnhancedAnalysisConfig(**{**base_config.__dict__, 'company_ticker': ticker, 'batch_timestamp': batch_timestamp})

            try:
                result = self.controller.run_enhanced_analysis(config)
                results.append(result)
                print(f"CHECK {ticker} completed: {result.final_composite_score:.2f}/5.0")

            except Exception as e:
                logger.error(f"X {ticker} failed: {str(e)}")
                continue

        # Generate batch summary
        self._display_batch_summary(results)

        return results

    def interactive_weight_configuration(self, user_id: str, company_ticker: str) -> WeightingScheme:
        """Interactive weight configuration"""
        print("\nTARGET WEIGHT CONFIGURATION")
        print("=" * 50)

        # Show default weights
        controller = self.controller
        default_weights_display = controller.scoring_system.get_default_weights_for_approval()

        print("Current default weights:")
        for category, weights in default_weights_display.items():
            print(f"\n{category}:")
            for name, weight in weights.items():
                print(f"  • {name}: {weight:.3f}")

        # Get user preference
        print("\nWeight Configuration Options:")
        print("1. Use default weights")
        print("2. Apply investment focus (Growth/Value/Quality/Risk)")
        print("3. Custom weight adjustment")

        choice = input("Select option (1-3): ").strip()

        if choice == "1":
            return controller.scoring_system.default_weights

        elif choice == "2":
            return self._apply_investment_focus()

        elif choice == "3":
            return self._custom_weight_adjustment()

        else:
            print("Invalid choice, using default weights")
            return controller.scoring_system.default_weights

    def _apply_investment_focus(self) -> WeightingScheme:
        """Apply investment focus to weights"""
        print("\nInvestment Focus Options:")
        print("1. Growth Focus - Emphasize growth drivers and innovation")
        print("2. Value Focus - Emphasize competitive moats and barriers")
        print("3. Quality Focus - Emphasize management and differentiation")
        print("4. Risk-Aware - Emphasize risk factors and red flags")

        focus = input("Select focus (1-4): ").strip()

        weights = WeightingScheme()

        if focus == "1":  # Growth Focus
            weights.key_growth_drivers = 0.20
            weights.transformation_potential = 0.15
            weights.platform_expansion = 0.10
            weights.barriers_to_entry = 0.12
            weights.network_effects = 0.12

        elif focus == "2":  # Value Focus
            weights.barriers_to_entry = 0.25
            weights.brand_monopoly = 0.20
            weights.switching_costs = 0.15
            weights.economies_of_scale = 0.15
            weights.key_growth_drivers = 0.08

        elif focus == "3":  # Quality Focus
            weights.competitive_differentiation = 0.20
            weights.management_quality = 0.15
            weights.technology_moats = 0.15
            weights.barriers_to_entry = 0.15
            weights.brand_monopoly = 0.12

        elif focus == "4":  # Risk-Aware
            weights.major_risk_factors = -0.15
            weights.red_flags = -0.10
            weights.barriers_to_entry = 0.20
            weights.switching_costs = 0.15
            weights.brand_monopoly = 0.15

        else:
            print("Invalid choice, using default weights")
            return self.controller.scoring_system.default_weights

        return weights.normalize_weights()

    def _custom_weight_adjustment(self) -> WeightingScheme:
        """Custom weight adjustment"""
        weights = WeightingScheme()

        print("\nCustom Weight Adjustment")
        print("Enter new weights (0.0-1.0) or press Enter to keep default:")

        weight_prompts = {
            "barriers_to_entry": "Barriers to Entry",
            "brand_monopoly": "Brand Monopoly",
            "economies_of_scale": "Economies of Scale",
            "network_effects": "Network Effects",
            "switching_costs": "Switching Costs",
            "key_growth_drivers": "Key Growth Drivers",
            "transformation_potential": "Transformation Potential",
            "competitive_differentiation": "Competitive Differentiation",
            "technology_moats": "Technology Moats"
        }

        for attr, name in weight_prompts.items():
            current = getattr(weights, attr)
            user_input = input(f"{name} [{current:.3f}]: ").strip()

            if user_input:
                try:
                    new_weight = float(user_input)
                    if 0.0 <= new_weight <= 1.0:
                        setattr(weights, attr, new_weight)
                    else:
                        print(f"Invalid weight {new_weight}, keeping default")
                except ValueError:
                    print(f"Invalid input '{user_input}', keeping default")

        return weights.normalize_weights()

    def _display_results(self, result: EnhancedAnalysisResult):
        """Display analysis results"""
        print("\n" + "="*60)
        print("TARGET ENHANCED ANALYSIS RESULTS")
        print("="*60)

        print(f"Company: {result.company.company_name} ({result.company.ticker})")
        print(f"Final Score: {result.final_composite_score:.2f}/5.0")
        print(f"Confidence: {result.final_confidence:.1%}")
        print(f"Recommendation: {result.recommendation}")

        print(f"\nCHART Analysis Details:")
        print(f"  • Models Used: {len(result.multi_llm_result.llm_results)}")
        print(f"  • Best Model: {result.multi_llm_result.best_model_recommendation}")
        print(f"  • Total Cost: ${result.total_cost_usd:.4f}")
        print(f"  • Processing Time: {result.total_time_seconds:.1f}s")

        print(f"\nTROPHY Top Consensus Scores:")
        sorted_scores = sorted(
            result.multi_llm_result.consensus_scores.items(),
            key=lambda x: x[1].score,
            reverse=True
        )

        for score_name, score_comp in sorted_scores[:5]:
            print(f"  • {score_comp.name}: {score_comp.score:.2f} (confidence: {score_comp.confidence:.1%})")

        print(f"\nDISK Files Generated:")
        for file_type, file_path in result.saved_files.items():
            print(f"  • {file_type.upper()}: {Path(file_path).name}")

    def _display_batch_summary(self, results: List[EnhancedAnalysisResult]):
        """Display batch analysis summary"""
        if not results:
            return

        print("\n" + "="*60)
        print("TRENDING_UP BATCH ANALYSIS SUMMARY")
        print("="*60)

        # Summary statistics
        scores = [r.final_composite_score for r in results]
        avg_score = sum(scores) / len(scores)
        total_cost = sum(r.total_cost_usd for r in results)
        total_time = sum(r.total_time_seconds for r in results)

        print(f"Companies Analyzed: {len(results)}")
        print(f"Average Score: {avg_score:.2f}/5.0")
        print(f"Total Cost: ${total_cost:.4f}")
        print(f"Total Time: {total_time:.1f}s")

        # Company rankings
        ranked_results = sorted(results, key=lambda r: r.final_composite_score, reverse=True)

        print(f"\nTROPHY Company Rankings:")
        for i, result in enumerate(ranked_results, 1):
            print(f"  {i}. {result.company.ticker}: {result.final_composite_score:.2f}/5.0 - {result.recommendation.split(' - ')[0]}")

    def create_analysis_config_from_args(self, args) -> EnhancedAnalysisConfig:
        """Create analysis config from command line arguments"""
        # Load custom weights if provided
        custom_weights = None
        if hasattr(args, 'custom_weights') and args.custom_weights:
            custom_weights = self._load_custom_weights(args.custom_weights)

        return EnhancedAnalysisConfig(
            user_id=args.user_id,
            company_ticker=args.company,
            analysis_type=args.analysis_type,
            models_to_use=args.models.split(',') if args.models else None,
            focus_themes=args.themes.split(',') if args.themes else None,
            geographies_of_interest=args.geographies.split(',') if args.geographies else ['US', 'Global'],
            lookback_window_months=args.lookback_months,
            enable_weight_approval=args.enable_weights,
            enable_human_feedback=args.enable_feedback,
            max_concurrent_models=args.max_concurrent,
            expert_id=args.expert_id,
            custom_weights=custom_weights
        )

    def _load_custom_weights(self, weights_file: str) -> WeightingScheme:
        """Load custom weights from JSON file"""
        try:
            with open(weights_file, 'r') as f:
                weight_data = json.load(f)

            if 'weights' in weight_data:
                weights_dict = weight_data['weights']
            else:
                weights_dict = weight_data

            # Create WeightingScheme object
            weights = WeightingScheme()

            # Load weights from file
            for attr, value in weights_dict.items():
                if hasattr(weights, attr):
                    setattr(weights, attr, float(value))

            print(f"[SUCCESS] Loaded custom weights from {weights_file}")
            if 'approved_by' in weight_data:
                print(f"   Approved by: {weight_data['approved_by']}")

            return weights.normalize_weights()

        except FileNotFoundError:
            print(f"[ERROR] Custom weights file not found: {weights_file}")
            print("   Using default weights instead")
            return None
        except Exception as e:
            print(f"[ERROR] Error loading custom weights: {e}")
            print("   Using default weights instead")
            return None

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Enhanced QualAgent Analysis System")

    # Required arguments
    parser.add_argument('--user-id', required=True, help='User ID for analysis tracking')

    # Analysis target
    parser.add_argument('--company', help='Single company ticker to analyze')
    parser.add_argument('--companies', help='Comma-separated list of companies for batch analysis')
    parser.add_argument('--batch', action='store_true', help='Run batch analysis')

    # Analysis configuration
    parser.add_argument('--analysis-type', default='comprehensive',
                       choices=['quick', 'comprehensive', 'expert_guided'],
                       help='Type of analysis to perform')
    parser.add_argument('--models', help='Comma-separated list of LLM models to use')
    parser.add_argument('--themes', help='Comma-separated focus themes for analysis')
    parser.add_argument('--geographies', default='US,Global', help='Geographic focus areas')
    parser.add_argument('--lookback-months', type=int, default=24, help='Lookback window in months')

    # Enhanced features
    parser.add_argument('--enable-weights', action='store_true', default=True,
                       help='Enable weight approval system')
    parser.add_argument('--enable-feedback', action='store_true', default=False,
                       help='Enable human feedback collection')
    parser.add_argument('--expert-id', help='Expert ID for feedback collection')
    parser.add_argument('--max-concurrent', type=int, default=3, help='Max concurrent LLM models')

    # Interactive features
    parser.add_argument('--interactive-weights', action='store_true',
                       help='Use interactive weight configuration')
    parser.add_argument('--custom-weights', type=str,
                       help='Path to custom weight configuration JSON file')
    parser.add_argument('--cost-estimate-only', action='store_true',
                       help='Only show cost estimate, do not run analysis')
    parser.add_argument('--auto-approve', action='store_true',
                       help='Automatically approve cost estimates without user input')

    args = parser.parse_args()

    # Validate arguments
    if not args.company and not args.companies and not args.batch:
        parser.error("Must specify either --company, --companies, or --batch")

    # Auto-enable feedback for expert-guided analysis (seamless workflow)
    if args.analysis_type == 'expert_guided':
        if not args.enable_feedback:
            args.enable_feedback = True
            logger.info("Expert-guided analysis: Auto-enabled feedback collection")

        # Enhanced user communication for expert-guided features
        print("EXPERT Expert-Guided Analysis Features:")
        print("   * Interactive weight approval")
        print("   * Multi-LLM consensus (5 models)")
        print("   * Human feedback collection")
        print("   * Detailed confidence metrics")
        print("   * Comprehensive file generation")

    # Smart defaulting: If feedback is enabled but no expert-id provided, default to user-id
    if args.enable_feedback and not args.expert_id:
        args.expert_id = args.user_id
        if args.analysis_type == 'expert_guided':
            print(f"   NOTE: Expert feedback will be collected as: {args.expert_id}")
        logger.info(f"Defaulting expert-id to user-id for feedback collection: {args.expert_id}")
    elif args.enable_feedback and args.analysis_type == 'expert_guided':
        print(f"   NOTE: Expert feedback will be collected as: {args.expert_id}")

    # Initialize demo
    demo = EnhancedQualAgentDemo(auto_approve=args.auto_approve)

    try:
        if args.company:
            # Single company analysis
            config = demo.create_analysis_config_from_args(args)

            # Interactive weight configuration if requested
            if args.interactive_weights:
                config.custom_weights = demo.interactive_weight_configuration(
                    args.user_id, args.company
                )

            # Cost estimate only
            if args.cost_estimate_only:
                cost_estimate = demo.controller.estimate_analysis_cost(config)
                print(f"Cost estimate for {args.company}: ${cost_estimate['total_estimated_cost']:.4f}")
                return

            # Run analysis
            result = demo.run_single_analysis(config)

            if result:
                # Generate report
                report = demo.controller.create_analysis_report(result, "markdown")
                report_file = f"analysis_report_{args.company}_{int(time.time())}.md"
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"\n[REPORT] Full report saved to: {report_file}")

        elif args.companies or args.batch:
            # Batch analysis
            if args.companies:
                companies = [c.strip() for c in args.companies.split(',')]
            else:
                # Get companies from database for batch
                companies = [c.ticker for c in demo.db.list_companies(limit=5)]

            if not companies:
                print("No companies found for batch analysis")
                return

            base_config = demo.create_analysis_config_from_args(args)
            results = demo.run_batch_analysis(companies, base_config)

            if results:
                # Save batch summary
                batch_file = f"batch_analysis_{int(time.time())}.json"
                batch_data = {
                    'timestamp': datetime.now().isoformat(),
                    'companies_analyzed': len(results),
                    'total_cost': sum(r.total_cost_usd for r in results),
                    'results': [
                        {
                            'ticker': r.company.ticker,
                            'score': r.final_composite_score,
                            'recommendation': r.recommendation,
                            'cost': r.total_cost_usd
                        } for r in results
                    ]
                }

                with open(batch_file, 'w', encoding='utf-8') as f:
                    json.dump(batch_data, f, indent=2)
                print(f"\n[REPORT] Batch summary saved to: {batch_file}")

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()