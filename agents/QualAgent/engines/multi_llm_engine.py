"""
Multi-LLM Engine for QualAgent
Executes analysis across multiple LLMs concurrently and provides consensus analysis
"""

import asyncio
import json
import logging
import time
import pandas as pd
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

from engines.llm_integration import LLMIntegration, LLMResponse
from engines.enhanced_scoring_system import EnhancedScoringSystem, ScoreComponent, WeightingScheme
from models.json_data_manager import JSONDataManager, Company

logger = logging.getLogger(__name__)

@dataclass
class MultiLLMResult:
    """Result from multi-LLM analysis"""
    company: Company
    request_id: str
    llm_results: Dict[str, Dict]  # model_name -> full result
    individual_scores: Dict[str, Dict[str, ScoreComponent]]  # model -> scores
    consensus_scores: Dict[str, ScoreComponent]
    composite_score: float
    composite_confidence: float
    best_model_recommendation: str
    execution_metadata: Dict
    total_cost_usd: float
    total_time_seconds: float

@dataclass
class LLMModelConfig:
    """Configuration for each LLM model"""
    model_name: str
    provider: str
    weight_in_consensus: float = 1.0
    enabled: bool = True
    description: str = ""

class MultiLLMEngine:
    """Engine for running analysis across multiple LLMs"""

    def __init__(self, data_dir: str = None):
        self.db = JSONDataManager(data_dir)
        self.llm = LLMIntegration()
        self.scoring_system = EnhancedScoringSystem()

        # Default TogetherAI models (5 LLMs as requested)
        self.default_models = [
            LLMModelConfig("mixtral-8x7b", "together", 1.0, True, "Excellent reasoning and analysis"),
            LLMModelConfig("llama-3-70b", "together", 1.0, True, "Strong general purpose model"),
            LLMModelConfig("qwen2-72b", "together", 1.0, True, "Strong analytical capabilities"),
            LLMModelConfig("llama-3.1-70b", "together", 0.8, True, "Good for creative analysis"),
            LLMModelConfig("deepseek-coder-33b", "together", 0.8, True, "Newest Llama model")
        ]

        logger.info(f"Multi-LLM Engine initialized with {len(self.default_models)} models")

    def get_available_models(self) -> List[LLMModelConfig]:
        """Get list of available and enabled models"""
        available_llm_models = self.llm.get_available_models()
        enabled_models = []

        for model_config in self.default_models:
            if model_config.enabled and model_config.model_name in available_llm_models:
                enabled_models.append(model_config)

        logger.info(f"Found {len(enabled_models)} enabled models: {[m.model_name for m in enabled_models]}")
        return enabled_models

    def run_multi_llm_analysis(self, company: Company, analysis_config: Dict,
                              user_weights: WeightingScheme = None,
                              max_concurrent: int = 3) -> MultiLLMResult:
        """Run analysis across multiple LLMs concurrently"""
        start_time = time.time()
        request_id = f"multi_llm_{int(time.time())}"

        logger.info(f"Starting multi-LLM analysis for {company.ticker}")

        # Get available models
        models = self.get_available_models()
        if not models:
            raise ValueError("No LLM models available for analysis")

        # Run analyses concurrently
        llm_results = self._run_concurrent_analysis(company, analysis_config, models, max_concurrent)

        # Extract scores from each model's results
        individual_scores = {}
        for model_name, result in llm_results.items():
            if result.get('error'):
                logger.warning(f"Model {model_name} failed: {result['error']}")
                continue

            parsed_output = result.get('parsed_output', {})
            logger.info(f"Processing {model_name}: parsed_output type: {type(parsed_output)}")

            try:
                scores = self.scoring_system.extract_all_scores(parsed_output)
                if not isinstance(scores, dict):
                    logger.error(f"Error: extract_all_scores returned {type(scores)}, expected dict")
                    scores = {}
                individual_scores[model_name] = scores
                logger.info(f"Extracted {len(scores)} scores for {model_name}")
            except Exception as e:
                logger.error(f"Error extracting scores for {model_name}: {e}")
                individual_scores[model_name] = {}

        # Generate consensus scores
        consensus_scores = self._generate_consensus_scores(individual_scores, models)

        # Calculate composite score
        weights = user_weights or self.scoring_system.default_weights
        composite_score, composite_confidence, scoring_metadata = \
            self.scoring_system.calculate_composite_score(consensus_scores, weights)

        # Determine best model
        best_model = self._determine_best_model(individual_scores, llm_results)

        # Calculate total cost and time
        total_cost = sum(result.get('cost_usd', 0) for result in llm_results.values())
        total_time = time.time() - start_time

        execution_metadata = {
            'models_used': list(llm_results.keys()),
            'successful_models': list(individual_scores.keys()),
            'failed_models': [name for name, result in llm_results.items() if result.get('error')],
            'consensus_method': 'weighted_average',
            'scoring_metadata': scoring_metadata,
            'execution_time_seconds': total_time,
            'concurrent_execution': True
        }

        return MultiLLMResult(
            company=company,
            request_id=request_id,
            llm_results=llm_results,
            individual_scores=individual_scores,
            consensus_scores=consensus_scores,
            composite_score=composite_score,
            composite_confidence=composite_confidence,
            best_model_recommendation=best_model,
            execution_metadata=execution_metadata,
            total_cost_usd=total_cost,
            total_time_seconds=total_time
        )

    def _run_concurrent_analysis(self, company: Company, analysis_config: Dict,
                               models: List[LLMModelConfig], max_concurrent: int) -> Dict[str, Dict]:
        """Run analysis concurrently across multiple models"""
        results = {}

        def run_single_model(model_config: LLMModelConfig) -> Tuple[str, Dict]:
            """Run analysis for a single model"""
            try:
                logger.info(f"Starting analysis with {model_config.model_name}")

                # Prepare the analysis configuration for this model
                config = analysis_config.copy()
                config['models_to_use'] = [model_config.model_name]

                # Run the analysis
                response = self.llm.run_analysis(
                    company=company,
                    config=config
                )

                if response.error:
                    return model_config.model_name, {'error': response.error}

                # Parse the response
                parsed_result = self._parse_llm_response(response)

                result = {
                    'model_name': model_config.model_name,
                    'provider': model_config.provider,
                    'raw_output': response.content,
                    'parsed_output': parsed_result,
                    'tokens_used': response.tokens_used,
                    'cost_usd': response.cost_usd,
                    'processing_time_seconds': response.processing_time_seconds,
                    'model_weight': model_config.weight_in_consensus
                }

                logger.info(f"Completed analysis with {model_config.model_name}")
                return model_config.model_name, result

            except Exception as e:
                logger.error(f"Error in {model_config.model_name}: {str(e)}")
                return model_config.model_name, {'error': str(e)}

        # Execute analyses concurrently
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            future_to_model = {executor.submit(run_single_model, model): model.model_name
                              for model in models}

            for future in as_completed(future_to_model):
                model_name, result = future.result()
                results[model_name] = result

        return results

    def _parse_llm_response(self, response: LLMResponse) -> Dict:
        """Parse LLM response into structured format"""
        try:
            # Try to extract JSON from response
            content = response.content.strip()

            # Remove markdown code blocks if present
            if content.startswith('```json'):
                content = content[7:]
            if content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]

            # Try to parse JSON
            return json.loads(content)

        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from {response.model_used}, attempting manual extraction")
            # Attempt manual extraction from narrative text
            manual_extraction = self._extract_from_narrative(response.content)
            return {
                'raw_text': response.content,
                'parsing_error': 'Failed to parse JSON - used manual extraction',
                'competitive_moat_analysis': manual_extraction.get('competitive_moat_analysis', {}),
                'strategic_insights': manual_extraction.get('strategic_insights', {}),
                'competitor_analysis': manual_extraction.get('competitor_analysis', {'competitors': []})
            }

    def _extract_from_narrative(self, text: str) -> Dict:
        """Extract structured information from narrative text"""
        import re

        result = {
            'competitive_moat_analysis': {},
            'strategic_insights': {
                'key_growth_drivers': [],
                'major_risk_factors': [],
                'red_flags': [],
                'transformation_potential': {'score': 3, 'confidence': 0.5, 'justification': 'Default score from narrative analysis'},
                'platform_expansion': {'score': 3, 'confidence': 0.5, 'justification': 'Default score from narrative analysis'},
                'competitive_differentiation': {'score': 3, 'confidence': 0.5, 'justification': 'Default score from narrative analysis'},
                'market_timing': {'score': 3, 'confidence': 0.5, 'justification': 'Default score from narrative analysis'},
                'management_quality': {'score': 3, 'confidence': 0.5, 'justification': 'Default score from narrative analysis'},
                'technology_moats': {'score': 3, 'confidence': 0.5, 'justification': 'Default score from narrative analysis'}
            },
            'competitor_analysis': {
                'competitors': []  # Expected format by enhanced scoring system
            }
        }

        # Try to extract scores from text patterns
        try:
            # Look for competitive moat related content
            moat_components = ['brand_monopoly', 'barriers_to_entry', 'economies_of_scale', 'network_effects', 'switching_costs']

            for component in moat_components:
                # Default scores for narrative analysis
                score = 3  # Default neutral score
                confidence = 0.6
                justification = f"Extracted from narrative analysis - {component.replace('_', ' ').title()}"

                # Simple scoring based on positive/negative language
                component_text = text.lower()
                if any(word in component_text for word in ['strong', 'dominant', 'leading', 'excellent', 'outstanding']):
                    score = 4
                elif any(word in component_text for word in ['very strong', 'exceptional', 'unmatched', 'superior']):
                    score = 5
                elif any(word in component_text for word in ['weak', 'limited', 'challenges', 'struggling']):
                    score = 2
                elif any(word in component_text for word in ['very weak', 'poor', 'failing', 'declining']):
                    score = 1

                result['competitive_moat_analysis'][component] = {
                    'score': score,
                    'confidence': confidence,
                    'justification': justification,
                    'sources': []
                }

            # Extract growth drivers from text (as simple strings for _score_list_component)
            growth_patterns = ['growth', 'expansion', 'opportunity', 'increasing demand', 'market trends']
            for pattern in growth_patterns:
                if pattern in text.lower():
                    result['strategic_insights']['key_growth_drivers'].append(
                        f"Growth driver identified: {pattern}"
                    )

            # Extract risk factors (as simple strings for _score_list_component)
            risk_patterns = ['risk', 'challenge', 'competition', 'threat', 'concern']
            for pattern in risk_patterns:
                if pattern in text.lower():
                    result['strategic_insights']['major_risk_factors'].append(
                        f"Risk factor identified: {pattern}"
                    )

            logger.info("Successfully extracted structured data from narrative text")

        except Exception as e:
            logger.warning(f"Error in narrative extraction: {e}")

        return result

    def _generate_consensus_scores(self, individual_scores: Dict[str, Dict[str, ScoreComponent]],
                                 models: List[LLMModelConfig]) -> Dict[str, ScoreComponent]:
        """Generate consensus scores from individual model scores"""
        if not individual_scores:
            return {}

        # Get model weights
        model_weights = {m.model_name: m.weight_in_consensus for m in models}

        # Find all score components across all models
        all_score_names = set()
        for scores in individual_scores.values():
            all_score_names.update(scores.keys())

        consensus_scores = {}

        for score_name in all_score_names:
            # Collect scores and weights for this component
            weighted_scores = []
            weighted_confidences = []
            all_justifications = []
            all_sources = []
            category = None

            for model_name, scores in individual_scores.items():
                if not isinstance(scores, dict):
                    logger.warning(f"Skipping {model_name}: scores is not a dict, got {type(scores)}")
                    continue
                if score_name in scores:
                    score_comp = scores[score_name]
                    weight = model_weights.get(model_name, 1.0)

                    weighted_scores.append(score_comp.score * weight)
                    weighted_confidences.append(score_comp.confidence * weight)
                    all_justifications.append(f"{model_name}: {score_comp.justification}")
                    all_sources.extend(score_comp.sources)
                    category = score_comp.category

            if weighted_scores:
                # Calculate consensus score and confidence
                total_weight = sum(model_weights.get(model, 1.0) for model in individual_scores.keys()
                                 if score_name in individual_scores[model])

                consensus_score = sum(weighted_scores) / total_weight if total_weight > 0 else 3.0
                consensus_confidence = sum(weighted_confidences) / total_weight if total_weight > 0 else 0.5

                # Combine justifications
                consensus_justification = f"Consensus from {len(weighted_scores)} models: " + \
                                        "; ".join(all_justifications[:3])  # Limit to first 3

                consensus_scores[score_name] = ScoreComponent(
                    name=score_name.replace('_', ' ').title(),
                    score=consensus_score,
                    confidence=consensus_confidence,
                    justification=consensus_justification,
                    sources=list(set(all_sources)),
                    category=category or 'consensus'
                )

        return consensus_scores

    def _determine_best_model(self, individual_scores: Dict[str, Dict[str, ScoreComponent]],
                            llm_results: Dict[str, Dict]) -> str:
        """Determine which model provided the best analysis"""
        if not individual_scores:
            return "none"

        model_quality_scores = {}

        for model_name, scores in individual_scores.items():
            if not scores:
                continue

            # Quality metrics
            num_scores = len(scores)
            avg_confidence = np.mean([score.confidence for score in scores.values()])

            # Check if the analysis has good justifications
            justification_quality = np.mean([
                len(score.justification) > 20 for score in scores.values()
            ])

            # Check for sources
            has_sources = any(len(score.sources) > 0 for score in scores.values())

            # Composite quality score
            quality_score = (
                num_scores * 0.3 +           # Number of scored components
                avg_confidence * 40 +        # Average confidence
                justification_quality * 20 + # Quality of justifications
                (10 if has_sources else 0)    # Presence of sources
            )

            model_quality_scores[model_name] = quality_score

        # Return model with highest quality score
        if not model_quality_scores:
            # Return first available model if no scores calculated
            return list(individual_scores.keys())[0] if individual_scores else "none"

        best_model = max(model_quality_scores.items(), key=lambda x: x[1])
        return best_model[0]

    def save_multi_format_results(self, result: MultiLLMResult, output_dir: str = None) -> Dict[str, str]:
        """Save results in JSON, CSV, and PKL formats"""
        output_dir = Path(output_dir) if output_dir else Path.cwd()
        timestamp = int(time.time())
        ticker = result.company.ticker

        saved_files = {}

        # 1. JSON format (detailed results)
        json_file = output_dir / f"multi_llm_analysis_{ticker}_{timestamp}.json"
        json_data = {
            'metadata': {
                'timestamp': timestamp,
                'company_ticker': ticker,
                'analysis_type': 'multi_llm',
                'models_used': result.execution_metadata['models_used']
            },
            'company': asdict(result.company),
            'composite_score': {
                'score': result.composite_score,
                'confidence': result.composite_confidence,
                'components_count': len(result.consensus_scores)
            },
            'consensus_scores': {name: asdict(score) for name, score in result.consensus_scores.items()},
            'individual_model_results': result.llm_results,
            'best_model': result.best_model_recommendation,
            'execution_metadata': result.execution_metadata,
            'total_cost_usd': result.total_cost_usd,
            'total_time_seconds': result.total_time_seconds
        }

        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        saved_files['json'] = str(json_file)

        # 2. CSV format (scores summary)
        csv_data = []
        for score_name, score_comp in result.consensus_scores.items():
            csv_data.append({
                'company_ticker': ticker,
                'score_component': score_name,
                'score': score_comp.score,
                'confidence': score_comp.confidence,
                'category': score_comp.category,
                'justification': score_comp.justification[:100] + '...' if len(score_comp.justification) > 100 else score_comp.justification
            })

        # Add composite score row
        csv_data.append({
            'company_ticker': ticker,
            'score_component': 'COMPOSITE_SCORE',
            'score': result.composite_score,
            'confidence': result.composite_confidence,
            'category': 'composite',
            'justification': f"Weighted composite from {len(result.consensus_scores)} components"
        })

        df = pd.DataFrame(csv_data)
        csv_file = output_dir / f"multi_llm_scores_{ticker}_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        saved_files['csv'] = str(csv_file)

        # 3. PKL format (full Python objects)
        pkl_file = output_dir / f"multi_llm_result_{ticker}_{timestamp}.pkl"
        with open(pkl_file, 'wb') as f:
            pickle.dump(result, f)
        saved_files['pkl'] = str(pkl_file)

        logger.info(f"Saved multi-LLM results in 3 formats: {list(saved_files.keys())}")
        return saved_files

    def estimate_multi_llm_cost(self, company_ticker: str, num_models: int = 5) -> Dict[str, float]:
        """Estimate cost for multi-LLM analysis"""
        # Get available models for cost estimation
        models = self.get_available_models()[:num_models]

        estimated_costs = {}
        total_cost = 0.0

        for model in models:
            # Estimate based on typical token usage (3000-5000 tokens per analysis)
            estimated_tokens = 4000
            model_cost_per_1k = self.llm.get_model_cost(model.model_name)
            model_cost = (estimated_tokens / 1000) * model_cost_per_1k

            estimated_costs[model.model_name] = model_cost
            total_cost += model_cost

        estimated_costs['total_estimated'] = total_cost
        estimated_costs['models_count'] = len(models)

        return estimated_costs