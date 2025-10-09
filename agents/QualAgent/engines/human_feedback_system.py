"""
Human Feedback Integration System for QualAgent
Enables expert selection and feedback collection to improve LLM performance
"""

import json
import logging
import time
import sqlite3
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class FeedbackEntry:
    """Individual feedback entry from expert"""
    feedback_id: str
    timestamp: str
    expert_id: str
    company_ticker: str
    model_comparison_type: str  # 'model_selection', 'score_adjustment', 'quality_rating'
    selected_model: Optional[str] = None
    model_rankings: Optional[List[str]] = None  # Ordered list of models by preference
    score_adjustments: Optional[Dict[str, float]] = None
    quality_ratings: Optional[Dict[str, int]] = None  # 1-5 rating for each model
    expert_comments: Optional[str] = None
    reasoning: Optional[str] = None

@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for a model based on feedback"""
    model_name: str
    selection_rate: float  # How often this model is selected as best
    average_ranking: float  # Average ranking (1=best, 5=worst)
    average_quality_rating: float  # Average 1-5 quality rating
    expert_preference_score: float  # Composite score
    feedback_count: int
    last_updated: str

class HumanFeedbackSystem:
    """System for collecting and managing human feedback on LLM analyses"""

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path.cwd() / "data"
        self.feedback_db_path = self.data_dir / "human_feedback.db"
        self.training_data_path = self.data_dir / "training_dataset.json"

        # Initialize database
        self._init_feedback_database()

        logger.info(f"Human Feedback System initialized with database: {self.feedback_db_path}")

    def _init_feedback_database(self):
        """Initialize SQLite database for feedback storage"""
        with sqlite3.connect(self.feedback_db_path) as conn:
            cursor = conn.cursor()

            # Feedback entries table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feedback_entries (
                    feedback_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    expert_id TEXT,
                    company_ticker TEXT,
                    model_comparison_type TEXT,
                    selected_model TEXT,
                    model_rankings TEXT,  -- JSON string
                    score_adjustments TEXT,  -- JSON string
                    quality_ratings TEXT,  -- JSON string
                    expert_comments TEXT,
                    reasoning TEXT
                )
            ''')

            # Model performance tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_performance (
                    model_name TEXT PRIMARY KEY,
                    selection_count INTEGER DEFAULT 0,
                    total_comparisons INTEGER DEFAULT 0,
                    ranking_sum REAL DEFAULT 0,
                    ranking_count INTEGER DEFAULT 0,
                    quality_sum REAL DEFAULT 0,
                    quality_count INTEGER DEFAULT 0,
                    last_updated TEXT
                )
            ''')

            # Expert profiles
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS expert_profiles (
                    expert_id TEXT PRIMARY KEY,
                    expert_name TEXT,
                    expertise_areas TEXT,  -- JSON string
                    feedback_count INTEGER DEFAULT 0,
                    reliability_score REAL DEFAULT 1.0,
                    created_at TEXT
                )
            ''')

            conn.commit()

    def present_model_comparison(self, multi_llm_result, expert_id: str,
                               comparison_type: str = "model_selection") -> Dict:
        """Present model results to expert for comparison and feedback"""

        presentation_data = {
            'comparison_id': f"comp_{int(time.time())}",
            'company': {
                'ticker': multi_llm_result.company.ticker,
                'name': multi_llm_result.company.company_name,
                'subsector': multi_llm_result.company.subsector
            },
            'comparison_type': comparison_type,
            'models': []
        }

        # Prepare model results for comparison
        for model_name, individual_scores in multi_llm_result.individual_scores.items():
            model_data = {
                'model_name': model_name,
                'composite_score': self._calculate_individual_composite_score(individual_scores),
                'key_scores': self._extract_key_scores_for_display(individual_scores),
                'analysis_quality': self._assess_analysis_quality(
                    multi_llm_result.llm_results.get(model_name, {})
                ),
                'cost': multi_llm_result.llm_results.get(model_name, {}).get('cost_usd', 0),
                'processing_time': multi_llm_result.llm_results.get(model_name, {}).get('processing_time_seconds', 0)
            }
            presentation_data['models'].append(model_data)

        # Sort models by composite score for easier comparison
        presentation_data['models'].sort(key=lambda x: x['composite_score'], reverse=True)

        # Add current system recommendation
        presentation_data['system_recommendation'] = {
            'best_model': multi_llm_result.best_model_recommendation,
            'consensus_score': multi_llm_result.composite_score,
            'consensus_confidence': multi_llm_result.composite_confidence
        }

        return presentation_data

    def collect_expert_feedback(self, comparison_data: Dict, expert_selection: Dict) -> str:
        """Collect and store expert feedback"""

        feedback_id = f"fb_{int(time.time())}"
        timestamp = datetime.now().isoformat()

        feedback = FeedbackEntry(
            feedback_id=feedback_id,
            timestamp=timestamp,
            expert_id=expert_selection['expert_id'],
            company_ticker=comparison_data['company']['ticker'],
            model_comparison_type=comparison_data['comparison_type'],
            selected_model=expert_selection.get('selected_model'),
            model_rankings=expert_selection.get('model_rankings'),
            score_adjustments=expert_selection.get('score_adjustments'),
            quality_ratings=expert_selection.get('quality_ratings'),
            expert_comments=expert_selection.get('comments'),
            reasoning=expert_selection.get('reasoning')
        )

        # Store in database
        self._store_feedback(feedback)

        # Update model performance metrics
        self._update_model_performance(feedback)

        # Update training dataset
        self._update_training_dataset(comparison_data, feedback)

        logger.info(f"Collected feedback {feedback_id} from expert {expert_selection['expert_id']}")
        return feedback_id

    def _store_feedback(self, feedback: FeedbackEntry):
        """Store feedback entry in database"""
        with sqlite3.connect(self.feedback_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO feedback_entries
                (feedback_id, timestamp, expert_id, company_ticker, model_comparison_type,
                 selected_model, model_rankings, score_adjustments, quality_ratings,
                 expert_comments, reasoning)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                feedback.feedback_id, feedback.timestamp, feedback.expert_id,
                feedback.company_ticker, feedback.model_comparison_type,
                feedback.selected_model,
                json.dumps(feedback.model_rankings) if feedback.model_rankings else None,
                json.dumps(feedback.score_adjustments) if feedback.score_adjustments else None,
                json.dumps(feedback.quality_ratings) if feedback.quality_ratings else None,
                feedback.expert_comments, feedback.reasoning
            ))
            conn.commit()

    def _update_model_performance(self, feedback: FeedbackEntry):
        """Update model performance metrics based on feedback"""
        with sqlite3.connect(self.feedback_db_path) as conn:
            cursor = conn.cursor()

            # Update for each model mentioned in feedback
            models_to_update = set()
            if feedback.selected_model:
                models_to_update.add(feedback.selected_model)
            if feedback.model_rankings:
                models_to_update.update(feedback.model_rankings)
            if feedback.quality_ratings:
                models_to_update.update(feedback.quality_ratings.keys())

            for model_name in models_to_update:
                # Get current metrics
                cursor.execute('SELECT * FROM model_performance WHERE model_name = ?', (model_name,))
                current = cursor.fetchone()

                if current is None:
                    # Initialize new model
                    cursor.execute('''
                        INSERT INTO model_performance
                        (model_name, selection_count, total_comparisons, ranking_sum, ranking_count,
                         quality_sum, quality_count, last_updated)
                        VALUES (?, 0, 0, 0, 0, 0, 0, ?)
                    ''', (model_name, feedback.timestamp))
                    current = (model_name, 0, 0, 0, 0, 0, 0, feedback.timestamp)

                # Update metrics
                selection_count = current[1]
                total_comparisons = current[2] + 1
                ranking_sum = current[3]
                ranking_count = current[4]
                quality_sum = current[5]
                quality_count = current[6]

                # Update selection count
                if feedback.selected_model == model_name:
                    selection_count += 1

                # Update ranking
                if feedback.model_rankings and model_name in feedback.model_rankings:
                    ranking = feedback.model_rankings.index(model_name) + 1  # 1-based ranking
                    ranking_sum += ranking
                    ranking_count += 1

                # Update quality rating
                if feedback.quality_ratings and model_name in feedback.quality_ratings:
                    quality_sum += feedback.quality_ratings[model_name]
                    quality_count += 1

                # Store updated metrics
                cursor.execute('''
                    UPDATE model_performance
                    SET selection_count = ?, total_comparisons = ?, ranking_sum = ?, ranking_count = ?,
                        quality_sum = ?, quality_count = ?, last_updated = ?
                    WHERE model_name = ?
                ''', (selection_count, total_comparisons, ranking_sum, ranking_count,
                      quality_sum, quality_count, feedback.timestamp, model_name))

            conn.commit()

    def _update_training_dataset(self, comparison_data: Dict, feedback: FeedbackEntry):
        """Update training dataset for future model improvement"""

        # Load existing training data
        training_data = []
        if self.training_data_path.exists():
            with open(self.training_data_path, 'r') as f:
                training_data = json.load(f)

        # Create training example
        training_example = {
            'id': feedback.feedback_id,
            'timestamp': feedback.timestamp,
            'input': {
                'company_ticker': feedback.company_ticker,
                'model_outputs': comparison_data['models'],
                'system_recommendation': comparison_data['system_recommendation']
            },
            'expert_feedback': {
                'expert_id': feedback.expert_id,
                'selected_model': feedback.selected_model,
                'model_rankings': feedback.model_rankings,
                'quality_ratings': feedback.quality_ratings,
                'score_adjustments': feedback.score_adjustments,
                'reasoning': feedback.reasoning
            },
            'metadata': {
                'comparison_type': feedback.model_comparison_type,
                'expert_comments': feedback.expert_comments
            }
        }

        training_data.append(training_example)

        # Save updated training data
        with open(self.training_data_path, 'w') as f:
            json.dump(training_data, f, indent=2)

        logger.info(f"Added training example {feedback.feedback_id} to dataset")

    def get_model_performance_report(self) -> Dict[str, ModelPerformanceMetrics]:
        """Generate performance report for all models"""
        with sqlite3.connect(self.feedback_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM model_performance')
            rows = cursor.fetchall()

        metrics = {}
        for row in rows:
            model_name, selection_count, total_comparisons, ranking_sum, ranking_count, \
            quality_sum, quality_count, last_updated = row

            # Calculate metrics
            selection_rate = selection_count / total_comparisons if total_comparisons > 0 else 0
            avg_ranking = ranking_sum / ranking_count if ranking_count > 0 else 3.0  # Neutral
            avg_quality = quality_sum / quality_count if quality_count > 0 else 3.0  # Neutral

            # Composite preference score (higher is better)
            # Combines selection rate, inverse ranking, and quality
            preference_score = (
                selection_rate * 0.4 +                    # 40% weight on selection
                ((6 - avg_ranking) / 5) * 0.3 +          # 30% weight on ranking (inverted)
                (avg_quality / 5) * 0.3                   # 30% weight on quality
            )

            metrics[model_name] = ModelPerformanceMetrics(
                model_name=model_name,
                selection_rate=selection_rate,
                average_ranking=avg_ranking,
                average_quality_rating=avg_quality,
                expert_preference_score=preference_score,
                feedback_count=total_comparisons,
                last_updated=last_updated
            )

        return metrics

    def get_feedback_insights(self, days_back: int = 30) -> Dict:
        """Get insights from recent feedback"""
        with sqlite3.connect(self.feedback_db_path) as conn:
            cursor = conn.cursor()

            # Get recent feedback
            cutoff_date = (datetime.now() - pd.Timedelta(days=days_back)).isoformat()
            cursor.execute('''
                SELECT * FROM feedback_entries
                WHERE timestamp > ?
                ORDER BY timestamp DESC
            ''', (cutoff_date,))

            recent_feedback = cursor.fetchall()

        insights = {
            'total_feedback_entries': len(recent_feedback),
            'expert_activity': {},
            'model_preferences': {},
            'common_adjustments': {},
            'feedback_trends': []
        }

        # Analyze expert activity
        for row in recent_feedback:
            expert_id = row[2]
            insights['expert_activity'][expert_id] = insights['expert_activity'].get(expert_id, 0) + 1

        # Analyze model preferences
        for row in recent_feedback:
            selected_model = row[5]
            if selected_model:
                insights['model_preferences'][selected_model] = \
                    insights['model_preferences'].get(selected_model, 0) + 1

        return insights

    def export_training_dataset(self, output_path: str = None) -> str:
        """Export training dataset for model fine-tuning"""
        output_path = output_path or str(self.data_dir / "exported_training_data.json")

        if not self.training_data_path.exists():
            return "No training data available"

        with open(self.training_data_path, 'r') as f:
            training_data = json.load(f)

        # Format for training (create positive/negative examples)
        formatted_data = []
        for example in training_data:
            if example['expert_feedback']['selected_model']:
                # Create positive example for selected model
                positive_example = {
                    'input': example['input'],
                    'output': {
                        'recommended_model': example['expert_feedback']['selected_model'],
                        'confidence': 'high',
                        'reasoning': example['expert_feedback']['reasoning']
                    },
                    'label': 'positive'
                }
                formatted_data.append(positive_example)

                # Create negative examples for non-selected models
                for model_data in example['input']['model_outputs']:
                    if model_data['model_name'] != example['expert_feedback']['selected_model']:
                        negative_example = {
                            'input': example['input'],
                            'output': {
                                'recommended_model': model_data['model_name'],
                                'confidence': 'low',
                                'reasoning': 'Not preferred by expert'
                            },
                            'label': 'negative'
                        }
                        formatted_data.append(negative_example)

        with open(output_path, 'w') as f:
            json.dump(formatted_data, f, indent=2)

        logger.info(f"Exported {len(formatted_data)} training examples to {output_path}")
        return output_path

    def _calculate_individual_composite_score(self, scores: Dict) -> float:
        """Calculate composite score for individual model result"""
        if not scores:
            return 3.0

        total_score = sum(score.score * score.confidence for score in scores.values())
        total_weight = sum(score.confidence for score in scores.values())

        return total_score / total_weight if total_weight > 0 else 3.0

    def _extract_key_scores_for_display(self, scores: Dict) -> Dict:
        """Extract key scores for expert review"""
        key_scores = {}
        important_components = [
            'moat_barriers_to_entry', 'moat_brand_monopoly', 'moat_switching_costs',
            'key_growth_drivers', 'competitive_positioning', 'transformation_potential'
        ]

        for component in important_components:
            if component in scores:
                score = scores[component]
                key_scores[component] = {
                    'score': score.score,
                    'confidence': score.confidence,
                    'justification': score.justification[:100] + '...' if len(score.justification) > 100 else score.justification
                }

        return key_scores

    def _assess_analysis_quality(self, llm_result: Dict) -> Dict:
        """Assess quality of individual LLM analysis"""
        quality_metrics = {
            'completeness': 0.0,
            'coherence': 0.0,
            'source_quality': 0.0,
            'overall': 0.0
        }

        if not llm_result or 'parsed_output' not in llm_result:
            return quality_metrics

        parsed = llm_result['parsed_output']

        # Completeness: presence of expected sections
        expected_sections = ['competitive_moat_analysis', 'strategic_insights', 'competitor_analysis']
        present_sections = sum(1 for section in expected_sections if section in parsed and parsed[section])
        quality_metrics['completeness'] = present_sections / len(expected_sections)

        # Source quality: presence of citations and sources
        has_sources = any('sources' in str(parsed).lower() for section in parsed.values())
        quality_metrics['source_quality'] = 1.0 if has_sources else 0.0

        # Overall quality
        quality_metrics['overall'] = (
            quality_metrics['completeness'] * 0.6 +
            quality_metrics['source_quality'] * 0.4
        )

        return quality_metrics