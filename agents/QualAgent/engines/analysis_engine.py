"""
Core Analysis Engine for QualAgent
Orchestrates the complete qualitative research workflow from input to database storage
"""

import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, asdict

from models.json_data_manager import JSONDataManager, Company, AnalysisRequest, LLMAnalysis
from engines.llm_integration import LLMIntegration, LLMResponse
from engines.prompt_adapter import PromptAdapter
from utils.result_parser import ResultParser

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class AnalysisConfig:
    """Configuration for analysis execution"""
    models_to_use: List[str]
    focus_themes: List[str] = None
    geographies_of_interest: List[str] = None
    lookback_window_months: int = 24
    enable_multi_model_consensus: bool = True
    save_intermediate_results: bool = True
    max_retries_per_model: int = 3
    requested_by: str = "system"
    priority: int = 1

@dataclass
class AnalysisResult:
    """Complete analysis result"""
    company: Company
    request_id: str
    llm_analyses: List[LLMAnalysis]
    parsed_results: List[Dict]
    consensus_analysis: Optional[Dict] = None
    execution_metadata: Optional[Dict] = None
    total_cost_usd: float = 0.0
    total_processing_time: float = 0.0
    success_rate: float = 0.0

class AnalysisEngine:
    """Core analysis engine orchestrating the complete workflow"""

    def __init__(self, data_dir: str = None):
        # Initialize components
        self.db = JSONDataManager(data_dir)
        self.llm = LLMIntegration()
        self.prompt_adapter = PromptAdapter()
        self.result_parser = ResultParser()

        # Validate system readiness
        self._validate_system()

        logger.info("Analysis engine initialized successfully")

    def _validate_system(self):
        """Validate that all system components are ready"""
        # Check API keys
        api_validation = self.llm.validate_api_keys()
        if not any(api_validation.values()):
            logger.warning("No valid API keys found - analysis will fail")

        # Check available models
        available_models = self.llm.get_available_models()
        if not available_models:
            logger.warning("No LLM models available")

        # Test database connection
        try:
            self.db.list_companies(limit=1)
            logger.info("Database connection validated")
        except Exception as e:
            logger.error(f"Database validation failed: {e}")

    def analyze_company(self, ticker: str, config: AnalysisConfig) -> AnalysisResult:
        """Run complete analysis for a company"""
        start_time = time.time()

        logger.info(f"Starting analysis for {ticker}")

        try:
            # 1. Get company data
            company = self._get_or_create_company(ticker)
            if not company:
                raise ValueError(f"Company {ticker} not found and could not be created")

            # 2. Create analysis request
            request_id = self._create_analysis_request(company, config)

            # 3. Update request status to in_progress
            self.db.update_analysis_request_status(request_id, 'in_progress')

            # 4. Generate multi-model analysis
            llm_analyses, parsed_results = self._execute_multi_model_analysis(
                company, request_id, config
            )

            # 5. Generate consensus if enabled
            consensus_analysis = None
            if config.enable_multi_model_consensus and len(parsed_results) > 1:
                consensus_analysis = self._generate_consensus_analysis(parsed_results)

            # 6. Calculate execution metadata
            total_time = time.time() - start_time
            total_cost = sum(analysis.cost_usd or 0 for analysis in llm_analyses)
            success_rate = len([a for a in llm_analyses if a.analysis_status == 'completed']) / len(llm_analyses)

            execution_metadata = {
                'total_processing_time_seconds': total_time,
                'total_cost_usd': total_cost,
                'success_rate': success_rate,
                'models_attempted': len(config.models_to_use),
                'models_succeeded': len([a for a in llm_analyses if a.analysis_status == 'completed']),
                'consensus_generated': consensus_analysis is not None
            }

            # 7. Update request status
            if success_rate > 0:
                self.db.update_analysis_request_status(
                    request_id, 'completed', processing_time=total_time
                )
            else:
                self.db.update_analysis_request_status(
                    request_id, 'failed', error_message="All model analyses failed"
                )

            # 8. Create result object
            result = AnalysisResult(
                company=company,
                request_id=request_id,
                llm_analyses=llm_analyses,
                parsed_results=parsed_results,
                consensus_analysis=consensus_analysis,
                execution_metadata=execution_metadata,
                total_cost_usd=total_cost,
                total_processing_time=total_time,
                success_rate=success_rate
            )

            logger.info(f"Analysis completed for {ticker}: {success_rate:.1%} success rate, ${total_cost:.4f} cost")

            return result

        except Exception as e:
            # Update request status to failed
            if 'request_id' in locals():
                self.db.update_analysis_request_status(
                    request_id, 'failed', error_message=str(e)
                )

            logger.error(f"Analysis failed for {ticker}: {e}")
            raise

    def _get_or_create_company(self, ticker: str) -> Optional[Company]:
        """Get company from database or create if not exists"""
        company = self.db.get_company_by_ticker(ticker)

        if not company:
            # Try to create company with basic information
            # In production, this would fetch from external data sources
            logger.warning(f"Company {ticker} not found in database - creating basic entry")

            basic_company = Company(
                company_name=f"Company {ticker}",
                ticker=ticker,
                subsector="Other Tech",
                description=f"Technology company with ticker {ticker}"
            )

            company_id = self.db.add_company(basic_company)
            basic_company.id = company_id
            return basic_company

        return company

    def _create_analysis_request(self, company: Company, config: AnalysisConfig) -> int:
        """Create analysis request in database"""
        request = AnalysisRequest(
            company_id=company.id,
            focus_themes=config.focus_themes,
            geographies_of_interest=config.geographies_of_interest,
            lookback_window_months=config.lookback_window_months,
            tools_available=["Tavily", "Twitter", "GuruFocus", "Reddit"],
            requested_by=config.requested_by,
            priority=config.priority
        )

        return self.db.create_analysis_request(request)

    def _execute_multi_model_analysis(self, company: Company, request_id: int,
                                    config: AnalysisConfig) -> Tuple[List[LLMAnalysis], List[Dict]]:
        """Execute analysis across multiple models"""

        llm_analyses = []
        parsed_results = []

        # Prepare company data for prompt adaptation
        company_data = {
            'company_name': company.company_name,
            'ticker': company.ticker,
            'subsector': company.subsector
        }

        # Create adapted prompts for each model
        model_prompts = self.prompt_adapter.create_multi_model_prompts(
            company_data,
            config.models_to_use,
            config.focus_themes,
            config.geographies_of_interest
        )

        # Execute analysis for each model
        for model_name in config.models_to_use:
            if model_name not in model_prompts:
                logger.error(f"Prompt adaptation failed for {model_name}")
                continue

            logger.info(f"Executing analysis with {model_name}")

            try:
                # Get adapted prompt
                system_prompt, user_messages = model_prompts[model_name]

                # Call LLM
                llm_response = self.llm.call_llm(
                    model_key=model_name,
                    messages=user_messages,
                    system_prompt=system_prompt,
                    max_retries=config.max_retries_per_model
                )

                # Create LLM analysis record
                analysis = LLMAnalysis(
                    request_id=request_id,
                    llm_model=model_name,
                    llm_provider=llm_response.provider,
                    input_prompt=system_prompt + "\n\n" + str(user_messages),
                    raw_output=llm_response.content,
                    analysis_status='completed' if not llm_response.error else 'failed',
                    tokens_used=llm_response.tokens_used,
                    cost_usd=llm_response.cost_usd,
                    processing_time_seconds=llm_response.processing_time_seconds,
                    error_details=llm_response.error
                )

                # Parse result if successful
                parsed_result = None
                if not llm_response.error:
                    try:
                        parsed_result = self.result_parser.parse_llm_output(
                            llm_response.content, company_data
                        )
                        analysis.parsed_output = parsed_result
                    except Exception as e:
                        logger.error(f"Failed to parse output from {model_name}: {e}")
                        analysis.analysis_status = 'partial'
                        analysis.error_details = f"Parsing error: {str(e)}"

                # Save to database
                analysis_id = self.db.save_llm_analysis(analysis)
                analysis.id = analysis_id

                # Save structured data if parsing was successful
                if parsed_result and config.save_intermediate_results:
                    self._save_structured_analysis_data(analysis_id, parsed_result)

                llm_analyses.append(analysis)
                if parsed_result:
                    parsed_results.append(parsed_result)

            except Exception as e:
                logger.error(f"Analysis failed for {model_name}: {e}")
                # Create failed analysis record
                failed_analysis = LLMAnalysis(
                    request_id=request_id,
                    llm_model=model_name,
                    llm_provider="unknown",
                    input_prompt="Failed before execution",
                    raw_output="",
                    analysis_status='failed',
                    error_details=str(e)
                )
                analysis_id = self.db.save_llm_analysis(failed_analysis)
                failed_analysis.id = analysis_id
                llm_analyses.append(failed_analysis)

        return llm_analyses, parsed_results

    def _save_structured_analysis_data(self, analysis_id: str, parsed_result: Dict):
        """Save structured analysis data to database"""
        try:
            # Save complete structured result using JSON storage
            self.db.save_structured_result(analysis_id, parsed_result)
            logger.info(f"Saved structured data for analysis {analysis_id}")

        except Exception as e:
            logger.error(f"Failed to save structured data for analysis {analysis_id}: {e}")

    def _generate_consensus_analysis(self, parsed_results: List[Dict]) -> Dict:
        """Generate consensus analysis from multiple model results"""
        if not parsed_results:
            return {}

        logger.info(f"Generating consensus from {len(parsed_results)} analyses")

        consensus = {
            'consensus_metadata': {
                'num_analyses': len(parsed_results),
                'generation_time': datetime.now().isoformat(),
                'method': 'simple_aggregation'
            }
        }

        # Aggregate dimensions scores
        if all('dimensions' in result for result in parsed_results):
            consensus['dimensions_consensus'] = self._aggregate_dimensions(parsed_results)

        # Aggregate moat assessments
        if all('moat_breakdown' in result for result in parsed_results):
            consensus['moat_consensus'] = self._aggregate_moat_breakdown(parsed_results)

        # Aggregate insights
        consensus['insights_consensus'] = self._aggregate_insights(parsed_results)

        return consensus

    def _aggregate_dimensions(self, results: List[Dict]) -> Dict:
        """Aggregate dimension scores across models"""
        dimension_scores = {}

        for result in results:
            for dim in result.get('dimensions', []):
                dim_name = dim.get('name')
                if dim_name not in dimension_scores:
                    dimension_scores[dim_name] = []

                score = dim.get('score')
                confidence = dim.get('confidence', 0.5)

                dimension_scores[dim_name].append({
                    'score': score,
                    'confidence': confidence,
                    'justification': dim.get('justification', '')
                })

        # Calculate consensus for each dimension
        consensus_dimensions = {}
        for dim_name, scores in dimension_scores.items():
            # Simple majority vote weighted by confidence
            score_weights = {}
            total_weight = 0

            for score_data in scores:
                score = score_data['score']
                weight = score_data['confidence']
                score_weights[score] = score_weights.get(score, 0) + weight
                total_weight += weight

            # Find consensus score
            consensus_score = max(score_weights.items(), key=lambda x: x[1])[0]
            consensus_confidence = score_weights[consensus_score] / total_weight

            consensus_dimensions[dim_name] = {
                'consensus_score': consensus_score,
                'consensus_confidence': consensus_confidence,
                'score_distribution': score_weights,
                'agreement_level': max(score_weights.values()) / total_weight
            }

        return consensus_dimensions

    def _aggregate_moat_breakdown(self, results: List[Dict]) -> Dict:
        """Aggregate moat breakdown across models"""
        moat_components = ['brand_monopoly', 'barriers_to_entry', 'economies_of_scale',
                          'network_effects', 'switching_costs']

        consensus_moat = {}

        for component in moat_components:
            labels = []
            for result in results:
                moat_data = result.get('moat_breakdown', {})
                component_data = moat_data.get(component, {})
                if 'label' in component_data:
                    labels.append(component_data['label'])

            if labels:
                # Simple majority vote
                label_counts = {}
                for label in labels:
                    label_counts[label] = label_counts.get(label, 0) + 1

                consensus_label = max(label_counts.items(), key=lambda x: x[1])[0]
                agreement = label_counts[consensus_label] / len(labels)

                consensus_moat[component] = {
                    'consensus_label': consensus_label,
                    'agreement_level': agreement,
                    'label_distribution': label_counts
                }

        return consensus_moat

    def _aggregate_insights(self, results: List[Dict]) -> Dict:
        """Aggregate insights across models"""
        all_tailwinds = []
        all_headwinds = []
        all_catalysts = []
        all_red_flags = []

        for result in results:
            all_tailwinds.extend(result.get('key_tailwinds', []))
            all_headwinds.extend(result.get('key_headwinds', []))
            all_catalysts.extend(result.get('catalysts_next_6_12m', []))
            all_red_flags.extend(result.get('red_flags', []))

        return {
            'most_common_tailwinds': self._get_most_common(all_tailwinds),
            'most_common_headwinds': self._get_most_common(all_headwinds),
            'most_common_catalysts': self._get_most_common(all_catalysts),
            'most_common_red_flags': self._get_most_common(all_red_flags)
        }

    def _get_most_common(self, items: List[str], top_n: int = 5) -> List[Dict]:
        """Get most common items from a list"""
        item_counts = {}
        for item in items:
            item_counts[item] = item_counts.get(item, 0) + 1

        sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
        return [{'item': item, 'frequency': count} for item, count in sorted_items[:top_n]]

    # Convenience methods for different use cases

    def quick_analysis(self, ticker: str, model: str = None) -> AnalysisResult:
        """Run quick single-model analysis"""
        if model is None:
            recommended = self.llm.get_recommended_models()
            model = recommended[0] if recommended else 'gpt-4o-mini'

        config = AnalysisConfig(
            models_to_use=[model],
            enable_multi_model_consensus=False,
            requested_by="quick_analysis"
        )

        return self.analyze_company(ticker, config)

    def comprehensive_analysis(self, ticker: str, focus_themes: List[str] = None) -> AnalysisResult:
        """Run comprehensive multi-model analysis"""
        recommended = self.llm.get_recommended_models()
        models = recommended[:3] if len(recommended) >= 3 else recommended

        config = AnalysisConfig(
            models_to_use=models,
            focus_themes=focus_themes,
            enable_multi_model_consensus=True,
            requested_by="comprehensive_analysis"
        )

        return self.analyze_company(ticker, config)

    def batch_analysis(self, tickers: List[str], config: AnalysisConfig) -> List[AnalysisResult]:
        """Run analysis on multiple companies"""
        results = []

        for ticker in tickers:
            try:
                result = self.analyze_company(ticker, config)
                results.append(result)
                logger.info(f"Completed batch analysis {len(results)}/{len(tickers)}: {ticker}")
            except Exception as e:
                logger.error(f"Batch analysis failed for {ticker}: {e}")

        return results

    def get_analysis_cost_estimate(self, config: AnalysisConfig) -> Dict[str, float]:
        """Estimate cost for analysis configuration"""
        estimated_tokens = 8000  # Rough estimate for TechQual analysis

        costs = {}
        total_cost = 0

        for model in config.models_to_use:
            model_cost = self.llm.get_cost_estimate(model, estimated_tokens)
            costs[model] = model_cost
            total_cost += model_cost

        costs['total_estimated_cost'] = total_cost
        return costs

def main():
    """Test analysis engine"""
    engine = AnalysisEngine()

    # Test quick analysis
    config = AnalysisConfig(
        models_to_use=['mixtral-8x7b'],
        focus_themes=['AI capabilities', 'Market position'],
        requested_by='test_user'
    )

    try:
        result = engine.analyze_company('NVDA', config)
        print(f"Analysis completed: {result.success_rate:.1%} success rate")
        print(f"Cost: ${result.total_cost_usd:.4f}")
        print(f"Processing time: {result.total_processing_time:.2f}s")

    except Exception as e:
        print(f"Analysis failed: {e}")

if __name__ == "__main__":
    main()