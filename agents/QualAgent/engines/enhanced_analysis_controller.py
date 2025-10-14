"""
Enhanced Analysis Controller for QualAgent
Orchestrates the complete enhanced workflow with multi-LLM, scoring, and human feedback
"""

import json
import logging
import time
import pandas as pd
import pickle
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime

from models.json_data_manager import JSONDataManager, Company
from engines.multi_llm_engine import MultiLLMEngine, MultiLLMResult
from engines.enhanced_scoring_system import EnhancedScoringSystem, WeightingScheme
from engines.weight_approval_system import WeightApprovalSystem, WeightApprovalSession
from engines.human_feedback_system import HumanFeedbackSystem
from engines.llm_integration import LLMIntegration

logger = logging.getLogger(__name__)

@dataclass
class EnhancedAnalysisConfig:
    """Configuration for enhanced analysis"""
    user_id: str
    company_ticker: str
    analysis_type: str = "comprehensive"  # "quick", "comprehensive", "expert_guided"
    models_to_use: List[str] = None  # If None, uses all available
    focus_themes: List[str] = None
    geographies_of_interest: List[str] = None
    lookback_window_months: int = 24
    enable_weight_approval: bool = True
    enable_human_feedback: bool = True
    max_concurrent_models: int = 3
    custom_weights: Optional[WeightingScheme] = None
    expert_id: Optional[str] = None
    batch_timestamp: Optional[str] = None  # For grouping analyses from the same batch run (format: YYYYMMDD_HHMMSS)

@dataclass
class EnhancedAnalysisResult:
    """Complete enhanced analysis result"""
    config: EnhancedAnalysisConfig
    company: Company
    multi_llm_result: MultiLLMResult
    weight_approval_session: Optional[WeightApprovalSession]
    human_feedback_id: Optional[str]
    final_composite_score: float
    final_confidence: float
    recommendation: str
    saved_files: Dict[str, str]
    execution_metadata: Dict
    total_cost_usd: float
    total_time_seconds: float

class EnhancedAnalysisController:
    """Main controller for enhanced QualAgent analysis workflow"""

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path.cwd() / "data"

        # Initialize all systems
        self.db = JSONDataManager(str(self.data_dir))
        self.multi_llm_engine = MultiLLMEngine(str(self.data_dir))
        self.scoring_system = EnhancedScoringSystem()
        self.weight_approval = WeightApprovalSystem(str(self.data_dir))
        self.human_feedback = HumanFeedbackSystem(str(self.data_dir))
        self.llm = LLMIntegration()

        # Results directory
        self.results_dir = self.data_dir.parent / "results"
        self.results_dir.mkdir(exist_ok=True)

        logger.info("Enhanced Analysis Controller initialized with all systems")

    def run_enhanced_analysis(self, config: EnhancedAnalysisConfig) -> EnhancedAnalysisResult:
        """Run complete enhanced analysis workflow"""
        start_time = time.time()
        logger.info(f"Starting enhanced analysis for {config.company_ticker}")

        try:
            # Step 1: Get company data
            company = self._get_company_data(config.company_ticker)

            # Step 2: Weight approval (if enabled)
            weight_session = None
            approved_weights = config.custom_weights

            if config.enable_weight_approval and not config.custom_weights:
                weight_session, approved_weights = self._handle_weight_approval(config, company)

            # Step 3: Multi-LLM analysis
            multi_llm_result = self._run_multi_llm_analysis(config, company, approved_weights)

            # Step 4: Human feedback collection (if enabled)
            feedback_id = None
            if config.enable_human_feedback and config.expert_id:
                feedback_id = self._collect_human_feedback(config, multi_llm_result)

            # Step 5: Generate final recommendation
            final_score, final_confidence, recommendation = self._generate_final_recommendation(
                multi_llm_result, weight_session, feedback_id
            )

            # Step 6: Save results in multiple formats
            saved_files = self._save_enhanced_results(config, multi_llm_result, weight_session)

            # Step 7: Create final result
            total_time = time.time() - start_time
            execution_metadata = {
                'analysis_completed_at': datetime.now().isoformat(),
                'total_execution_time': total_time,
                'systems_used': {
                    'multi_llm': True,
                    'weight_approval': weight_session is not None,
                    'human_feedback': feedback_id is not None
                },
                'models_successful': len(multi_llm_result.individual_scores),
                'models_failed': len(multi_llm_result.llm_results) - len(multi_llm_result.individual_scores)
            }

            result = EnhancedAnalysisResult(
                config=config,
                company=company,
                multi_llm_result=multi_llm_result,
                weight_approval_session=weight_session,
                human_feedback_id=feedback_id,
                final_composite_score=final_score,
                final_confidence=final_confidence,
                recommendation=recommendation,
                saved_files=saved_files,
                execution_metadata=execution_metadata,
                total_cost_usd=multi_llm_result.total_cost_usd,
                total_time_seconds=total_time
            )

            logger.info(f"Enhanced analysis completed for {config.company_ticker} in {total_time:.2f}s")
            return result

        except Exception as e:
            logger.error(f"Enhanced analysis failed for {config.company_ticker}: {str(e)}")
            raise

    def _get_company_data(self, ticker: str) -> Company:
        """Get or create company data"""
        company = self.db.get_company_by_ticker(ticker)
        if not company:
            # Try to create basic company entry
            logger.warning(f"Company {ticker} not found, creating basic entry")
            from models.json_data_manager import Company
            company = Company(
                company_name=f"{ticker} Inc.",
                ticker=ticker,
                subsector="Technology",
                description=f"{ticker} - Technology company"
            )
            company_id = self.db.add_company(company)
            company = self.db.get_company_by_id(company_id)

        return company

    def _handle_weight_approval(self, config: EnhancedAnalysisConfig,
                              company: Company) -> Tuple[WeightApprovalSession, WeightingScheme]:
        """Handle weight approval workflow"""
        logger.info(f"Starting weight approval for {config.user_id}")

        # Check if user has historical preferences
        user_prefs = self.weight_approval.get_user_weight_preferences(config.user_id)

        # Create approval session
        session = self.weight_approval.create_approval_session(
            config.user_id, config.company_ticker
        )

        # For now, return default weights (in real implementation, this would be interactive)
        # TODO: Implement interactive weight approval interface
        logger.info("Using default weights (interactive approval not implemented)")

        # Simulate user approving default weights
        user_response = {'response_type': 'approve_default'}
        session, approved = self.weight_approval.process_user_response(session, user_response)

        if approved:
            return session, session.modified_weights
        else:
            return session, self.scoring_system.default_weights

    def _run_multi_llm_analysis(self, config: EnhancedAnalysisConfig, company: Company,
                               weights: WeightingScheme) -> MultiLLMResult:
        """Run multi-LLM analysis"""
        logger.info(f"Running multi-LLM analysis with {config.max_concurrent_models} concurrent models")

        # Prepare analysis configuration
        analysis_config = {
            'focus_themes': config.focus_themes or [],
            'geographies_of_interest': config.geographies_of_interest or ['US', 'Global'],
            'lookback_window_months': config.lookback_window_months,
            'models_to_use': config.models_to_use,  # FIX: Pass through selected models
            'enable_multi_model_consensus': True,
            'save_intermediate_results': True,
            'requested_by': f"EnhancedAnalysis-{config.user_id}",
            'priority': 1
        }

        # Run multi-LLM analysis
        result = self.multi_llm_engine.run_multi_llm_analysis(
            company=company,
            analysis_config=analysis_config,
            user_weights=weights,
            max_concurrent=config.max_concurrent_models
        )

        return result

    def _collect_human_feedback(self, config: EnhancedAnalysisConfig,
                               multi_llm_result: MultiLLMResult) -> Optional[str]:
        """Collect human feedback on multi-LLM results"""
        if not config.expert_id:
            return None

        logger.info(f"Collecting human feedback from expert {config.expert_id}")

        try:
            # Present comparison to expert
            comparison_data = self.human_feedback.present_model_comparison(
                multi_llm_result, config.expert_id
            )

            # For now, simulate expert selection (in real implementation, this would be interactive)
            # TODO: Implement interactive expert feedback interface
            expert_selection = {
                'expert_id': config.expert_id,
                'selected_model': multi_llm_result.best_model_recommendation,
                'quality_ratings': {
                    model: 4 for model in multi_llm_result.individual_scores.keys()
                },
                'comments': 'Automated feedback collection',
                'reasoning': 'Selected system recommendation as best model'
            }

            # Collect feedback
            feedback_id = self.human_feedback.collect_expert_feedback(
                comparison_data, expert_selection
            )

            return feedback_id

        except Exception as e:
            logger.error(f"Failed to collect human feedback: {str(e)}")
            return None

    def _generate_final_recommendation(self, multi_llm_result: MultiLLMResult,
                                     weight_session: Optional[WeightApprovalSession],
                                     feedback_id: Optional[str]) -> Tuple[float, float, str]:
        """Generate final recommendation based on all analysis components"""

        score = multi_llm_result.composite_score
        confidence = multi_llm_result.composite_confidence

        # Generate textual recommendation
        if score >= 4.0:
            recommendation = "STRONG BUY - Company demonstrates excellent fundamentals across multiple dimensions"
        elif score >= 3.5:
            recommendation = "BUY - Company shows strong competitive position with good growth prospects"
        elif score >= 3.0:
            recommendation = "HOLD - Company has balanced strengths and weaknesses, monitor for improvements"
        elif score >= 2.5:
            recommendation = "WEAK HOLD - Company faces challenges but may have turnaround potential"
        else:
            recommendation = "SELL - Company demonstrates weak fundamentals and significant risks"

        # Adjust confidence based on feedback availability
        if feedback_id:
            confidence *= 1.1  # Increase confidence when expert feedback is available
        if weight_session and weight_session.approval_status == 'modified':
            confidence *= 1.05  # Slight increase for custom weights

        confidence = min(1.0, confidence)  # Cap at 1.0

        return score, confidence, recommendation

    def _save_enhanced_results(self, config: EnhancedAnalysisConfig,
                             multi_llm_result: MultiLLMResult,
                             weight_session: Optional[WeightApprovalSession]) -> Dict[str, str]:
        """Save results in multiple formats with enhanced metadata"""

        # Save multi-LLM results
        saved_files = self.multi_llm_engine.save_multi_format_results(
            multi_llm_result, str(self.results_dir), config.batch_timestamp
        )

        # Save enhanced metadata
        timestamp = config.batch_timestamp if config.batch_timestamp is not None else datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata_file = self.results_dir / f"enhanced_metadata_{config.company_ticker}_{timestamp}.json"

        metadata = {
            'analysis_config': asdict(config),
            'weight_approval_session': asdict(weight_session) if weight_session else None,
            'execution_summary': {
                'total_models_used': len(multi_llm_result.llm_results),
                'successful_models': len(multi_llm_result.individual_scores),
                'best_model': multi_llm_result.best_model_recommendation,
                'composite_score': multi_llm_result.composite_score,
                'confidence': multi_llm_result.composite_confidence
            }
        }

        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)

        saved_files['metadata'] = str(metadata_file)

        # Create summary CSV for easy viewing
        summary_csv = self.results_dir / f"analysis_summary_{config.company_ticker}_{timestamp}.csv"
        summary_data = {
            'metric': ['Composite Score', 'Confidence', 'Best Model', 'Total Cost', 'Processing Time'],
            'value': [
                multi_llm_result.composite_score,
                multi_llm_result.composite_confidence,
                multi_llm_result.best_model_recommendation,
                multi_llm_result.total_cost_usd,
                multi_llm_result.total_time_seconds
            ]
        }

        pd.DataFrame(summary_data).to_csv(summary_csv, index=False)
        saved_files['summary_csv'] = str(summary_csv)

        return saved_files

    def estimate_analysis_cost(self, config: EnhancedAnalysisConfig) -> Dict[str, float]:
        """Estimate cost for enhanced analysis"""
        # Get multi-LLM cost estimate
        model_count = len(config.models_to_use) if config.models_to_use else 5
        multi_llm_costs = self.multi_llm_engine.estimate_multi_llm_cost(
            config.company_ticker, model_count
        )

        # Add overhead for additional processing
        overhead_cost = multi_llm_costs['total_estimated'] * 0.1  # 10% overhead

        estimate = {
            'multi_llm_cost': multi_llm_costs['total_estimated'],
            'overhead_cost': overhead_cost,
            'total_estimated_cost': multi_llm_costs['total_estimated'] + overhead_cost,
            'models_included': multi_llm_costs['models_count'],
            'per_model_average': multi_llm_costs['total_estimated'] / multi_llm_costs['models_count']
        }

        return estimate

    def get_analysis_status(self, request_id: str) -> Dict:
        """Get status of ongoing analysis (for async implementations)"""
        # Placeholder for async status tracking
        return {
            'request_id': request_id,
            'status': 'completed',  # In current sync implementation
            'message': 'Analysis completed successfully'
        }

    def create_analysis_report(self, result: EnhancedAnalysisResult,
                             output_format: str = "markdown") -> str:
        """Create comprehensive analysis report"""

        if output_format == "markdown":
            return self._create_markdown_report(result)
        elif output_format == "html":
            return self._create_html_report(result)
        else:
            return self._create_text_report(result)

    def _create_markdown_report(self, result: EnhancedAnalysisResult) -> str:
        """Create markdown analysis report"""

        report = f"""# QualAgent Enhanced Analysis Report

## Company Overview
- **Ticker:** {result.company.ticker}
- **Company:** {result.company.company_name}
- **Subsector:** {result.company.subsector}
- **Analysis Date:** {result.execution_metadata['analysis_completed_at']}

## Executive Summary
- **Final Composite Score:** {result.final_composite_score:.2f}/5.0
- **Confidence Level:** {result.final_confidence:.1%}
- **Recommendation:** {result.recommendation}

## Multi-LLM Analysis Results
- **Models Used:** {len(result.multi_llm_result.llm_results)}
- **Successful Analyses:** {len(result.multi_llm_result.individual_scores)}
- **Best Performing Model:** {result.multi_llm_result.best_model_recommendation}
- **Total Cost:** ${result.total_cost_usd:.4f}
- **Processing Time:** {result.total_time_seconds:.1f} seconds

## Consensus Scores
"""

        # Add consensus scores table
        for score_name, score_comp in result.multi_llm_result.consensus_scores.items():
            report += f"- **{score_comp.name}:** {score_comp.score:.2f} (Confidence: {score_comp.confidence:.1%})\n"

        report += f"""
## Analysis Configuration
- **User ID:** {result.config.user_id}
- **Analysis Type:** {result.config.analysis_type}
- **Focus Themes:** {', '.join(result.config.focus_themes) if result.config.focus_themes else 'None'}
- **Weight Approval:** {'Enabled' if result.config.enable_weight_approval else 'Disabled'}
- **Human Feedback:** {'Enabled' if result.config.enable_human_feedback else 'Disabled'}

## Files Generated
"""
        for file_type, file_path in result.saved_files.items():
            report += f"- **{file_type.upper()}:** `{file_path}`\n"

        report += f"""
---
*Generated by QualAgent Enhanced Analysis System*
"""

        return report

    def _create_text_report(self, result: EnhancedAnalysisResult) -> str:
        """Create plain text analysis report"""
        return f"""
QualAgent Enhanced Analysis Report
==================================

Company: {result.company.company_name} ({result.company.ticker})
Final Score: {result.final_composite_score:.2f}/5.0
Confidence: {result.final_confidence:.1%}
Recommendation: {result.recommendation}

Models Used: {len(result.multi_llm_result.llm_results)}
Best Model: {result.multi_llm_result.best_model_recommendation}
Total Cost: ${result.total_cost_usd:.4f}
Processing Time: {result.total_time_seconds:.1f}s

Files saved in: {list(result.saved_files.values())}
"""

    def _create_html_report(self, result: EnhancedAnalysisResult) -> str:
        """Create HTML analysis report"""
        # Simplified HTML report
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>QualAgent Analysis Report - {result.company.ticker}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .score {{ font-size: 24px; color: #2E8B57; font-weight: bold; }}
        .recommendation {{ font-size: 18px; color: #4169E1; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{result.company.company_name} ({result.company.ticker})</h1>
        <div class="score">Final Score: {result.final_composite_score:.2f}/5.0</div>
        <div class="recommendation">{result.recommendation}</div>
    </div>

    <h2>Analysis Details</h2>
    <p><strong>Confidence:</strong> {result.final_confidence:.1%}</p>
    <p><strong>Models Used:</strong> {len(result.multi_llm_result.llm_results)}</p>
    <p><strong>Best Model:</strong> {result.multi_llm_result.best_model_recommendation}</p>
    <p><strong>Total Cost:</strong> ${result.total_cost_usd:.4f}</p>
    <p><strong>Processing Time:</strong> {result.total_time_seconds:.1f} seconds</p>

    <h2>Files Generated</h2>
    <ul>
        {''.join(f'<li><strong>{k}:</strong> {v}</li>' for k, v in result.saved_files.items())}
    </ul>

    <footer>
        <p><em>Generated by QualAgent Enhanced Analysis System</em></p>
    </footer>
</body>
</html>
"""