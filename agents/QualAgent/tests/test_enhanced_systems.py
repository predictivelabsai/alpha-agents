#!/usr/bin/env python3
"""
Comprehensive Test Suite for QualAgent Enhanced Systems
Tests all new enhanced features including multi-LLM, scoring, and human feedback
"""

import pytest
import json
import tempfile
import sqlite3
from pathlib import Path
import sys
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import pickle
from dataclasses import asdict

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import enhanced systems
from engines.enhanced_scoring_system import EnhancedScoringSystem, WeightingScheme, ScoreComponent
from engines.multi_llm_engine import MultiLLMEngine, MultiLLMResult, LLMModelConfig
from engines.human_feedback_system import HumanFeedbackSystem, FeedbackEntry
from engines.weight_approval_system import WeightApprovalSystem, WeightApprovalSession
from engines.enhanced_analysis_controller import EnhancedAnalysisController, EnhancedAnalysisConfig
from engines.workflow_optimizer import WorkflowOptimizer, WorkflowState
from models.json_data_manager import JSONDataManager, Company

class TestEnhancedScoringSystem:
    """Test the enhanced scoring system"""

    def setup_method(self):
        """Setup for each test"""
        self.scoring_system = EnhancedScoringSystem()

    def test_default_weights_initialization(self):
        """Test default weights are properly initialized"""
        weights = self.scoring_system.default_weights

        assert weights.barriers_to_entry > 0
        assert weights.brand_monopoly > 0
        assert weights.major_risk_factors < 0  # Risk factors should be negative
        assert weights.red_flags < 0

    def test_weight_normalization(self):
        """Test weight normalization works correctly"""
        weights = WeightingScheme(
            barriers_to_entry=0.5,
            brand_monopoly=0.3,
            key_growth_drivers=0.2
        )

        normalized = weights.normalize_weights()
        positive_weights = [v for v in asdict(normalized).values() if v > 0]

        # Should sum approximately to 1.0 (allowing for floating point precision)
        assert abs(sum(positive_weights) - 1.0) < 0.001

    def test_score_component_creation(self):
        """Test ScoreComponent creation and validation"""
        score = ScoreComponent(
            name="Test Score",
            score=4.5,
            confidence=0.8,
            justification="Test justification",
            sources=["source1", "source2"],
            category="test"
        )

        assert score.name == "Test Score"
        assert score.score == 4.5
        assert score.confidence == 0.8
        assert len(score.sources) == 2

    def test_extract_scores_from_analysis(self):
        """Test extracting scores from analysis results"""
        # Mock analysis result
        analysis_result = {
            'competitive_moat_analysis': {
                'Barriers to Entry': {
                    'score': 4,
                    'confidence': 0.8,
                    'justification': 'Strong technical barriers'
                },
                'Brand Monopoly': {
                    'score': 3,
                    'confidence': 0.7,
                    'justification': 'Moderate brand strength'
                }
            },
            'strategic_insights': {
                'key_growth_drivers': ['AI expansion', 'Market growth'],
                'major_risk_factors': ['Competition', 'Regulation'],
                'transformation_potential_assessment': 'Strong potential for AI transformation'
            },
            'competitor_analysis': {
                'competitors': [
                    {'company': 'Competitor A', 'competitive_positioning': 'Strong competitor'},
                    {'company': 'Competitor B', 'competitive_positioning': 'Weak position'}
                ]
            }
        }

        scores = self.scoring_system.extract_all_scores(analysis_result)

        # Should extract moat scores
        assert 'moat_barriers_to_entry' in scores
        assert 'moat_brand_monopoly' in scores

        # Should extract strategic insights
        assert 'key_growth_drivers' in scores
        assert 'major_risk_factors' in scores
        assert 'transformation_potential' in scores

        # Should extract competitor analysis
        assert 'competitive_positioning' in scores

    def test_composite_score_calculation(self):
        """Test composite score calculation with weights"""
        # Create sample scores
        scores = {
            'moat_barriers_to_entry': ScoreComponent('Barriers', 5.0, 0.9, 'Strong', [], 'moat'),
            'moat_brand_monopoly': ScoreComponent('Brand', 3.0, 0.7, 'Moderate', [], 'moat'),
            'key_growth_drivers': ScoreComponent('Growth', 4.0, 0.8, 'Good', [], 'growth'),
            'major_risk_factors': ScoreComponent('Risk', 2.0, 0.6, 'Low risk', [], 'risk')
        }

        weights = WeightingScheme(
            barriers_to_entry=0.3,
            brand_monopoly=0.2,
            key_growth_drivers=0.2,
            major_risk_factors=-0.1
        )

        composite_score, confidence, metadata = self.scoring_system.calculate_composite_score(
            scores, weights
        )

        assert 1.0 <= composite_score <= 5.0
        assert 0.0 <= confidence <= 1.0
        assert 'components_used' in metadata
        assert metadata['components_used'] == 4

    def test_get_default_weights_for_approval(self):
        """Test getting weights formatted for user approval"""
        weights_display = self.scoring_system.get_default_weights_for_approval()

        assert 'Core Competitive Moats (Higher Weight)' in weights_display
        assert 'Risk Factors (Negative Weight)' in weights_display

        # Check structure
        core_moats = weights_display['Core Competitive Moats (Higher Weight)']
        assert 'Barriers to Entry' in core_moats
        assert 'Brand Monopoly' in core_moats

class TestMultiLLMEngine:
    """Test the multi-LLM engine"""

    def setup_method(self):
        """Setup for each test"""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.temp_dir = temp_dir
            self.engine = MultiLLMEngine(temp_dir)

    @patch('engines.multi_llm_engine.LLMIntegration')
    def test_get_available_models(self, mock_llm_integration):
        """Test getting available models"""
        # Mock LLM integration
        mock_instance = Mock()
        mock_instance.get_available_models.return_value = [
            'mixtral-8x7b', 'llama-3-70b', 'qwen2-72b'
        ]
        mock_llm_integration.return_value = mock_instance

        # Create engine with mocked LLM
        engine = MultiLLMEngine()
        engine.llm = mock_instance

        models = engine.get_available_models()

        assert len(models) > 0
        assert all(isinstance(model, LLMModelConfig) for model in models)

    def test_model_config_creation(self):
        """Test LLM model configuration"""
        config = LLMModelConfig(
            model_name="test-model",
            provider="together",
            weight_in_consensus=1.0,
            enabled=True,
            description="Test model"
        )

        assert config.model_name == "test-model"
        assert config.provider == "together"
        assert config.weight_in_consensus == 1.0

    @patch('engines.multi_llm_engine.ThreadPoolExecutor')
    def test_concurrent_analysis_simulation(self, mock_executor):
        """Test concurrent analysis simulation"""
        # This is a simplified test due to complexity of mocking concurrent execution
        engine = MultiLLMEngine()

        # Mock company
        company = Company(
            company_name="Test Corp",
            ticker="TEST",
            subsector="Technology"
        )

        # Mock analysis config
        analysis_config = {
            'focus_themes': ['AI', 'Growth'],
            'geographies_of_interest': ['US'],
            'lookback_window_months': 24
        }

        # The actual test would require extensive mocking of the LLM responses
        # For now, we test that the method exists and can be called
        assert hasattr(engine, '_run_concurrent_analysis')
        assert hasattr(engine, 'run_multi_llm_analysis')

    def test_consensus_score_generation(self):
        """Test consensus score generation from individual model scores"""
        engine = MultiLLMEngine()

        # Mock individual scores from different models
        individual_scores = {
            'mixtral-8x7b': {
                'test_score': ScoreComponent('Test', 4.0, 0.8, 'Good', [], 'test')
            },
            'llama-3-70b': {
                'test_score': ScoreComponent('Test', 4.5, 0.9, 'Excellent', [], 'test')
            }
        }

        # Mock model configs
        models = [
            LLMModelConfig('mixtral-8x7b', 'together', 1.0),
            LLMModelConfig('llama-3-70b', 'together', 1.0)
        ]

        consensus_scores = engine._generate_consensus_scores(individual_scores, models)

        assert 'test_score' in consensus_scores
        consensus_score = consensus_scores['test_score']
        assert 4.0 <= consensus_score.score <= 4.5  # Should be average of 4.0 and 4.5

    def test_cost_estimation(self):
        """Test cost estimation for multi-LLM analysis"""
        engine = MultiLLMEngine()

        # Mock get_available_models to return test models
        engine.get_available_models = Mock(return_value=[
            LLMModelConfig('mixtral-8x7b', 'together', 1.0),
            LLMModelConfig('llama-3-70b', 'together', 1.0)
        ])

        # Mock LLM integration for cost estimation
        engine.llm.get_model_cost = Mock(return_value=0.0006)  # $0.0006 per 1K tokens

        cost_estimate = engine.estimate_multi_llm_cost("TEST", num_models=2)

        assert 'total_estimated' in cost_estimate
        assert cost_estimate['models_count'] == 2
        assert cost_estimate['total_estimated'] > 0

class TestHumanFeedbackSystem:
    """Test the human feedback system"""

    def setup_method(self):
        """Setup for each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.feedback_system = HumanFeedbackSystem(self.temp_dir)

    def test_database_initialization(self):
        """Test database is properly initialized"""
        db_path = Path(self.temp_dir) / "human_feedback.db"
        assert db_path.exists()

        # Check tables exist
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]

        assert 'feedback_entries' in tables
        assert 'model_performance' in tables
        assert 'expert_profiles' in tables

    def test_feedback_entry_creation(self):
        """Test feedback entry creation"""
        feedback = FeedbackEntry(
            feedback_id="test_feedback",
            timestamp="2023-10-01T12:00:00",
            expert_id="expert1",
            company_ticker="TEST",
            model_comparison_type="model_selection",
            selected_model="mixtral-8x7b",
            quality_ratings={"mixtral-8x7b": 5, "llama-3-70b": 4},
            expert_comments="Excellent analysis",
            reasoning="Comprehensive coverage of all aspects"
        )

        assert feedback.expert_id == "expert1"
        assert feedback.selected_model == "mixtral-8x7b"
        assert feedback.quality_ratings["mixtral-8x7b"] == 5

    def test_store_and_retrieve_feedback(self):
        """Test storing and retrieving feedback"""
        feedback = FeedbackEntry(
            feedback_id="test_feedback_123",
            timestamp="2023-10-01T12:00:00",
            expert_id="expert1",
            company_ticker="TEST",
            model_comparison_type="model_selection",
            selected_model="mixtral-8x7b"
        )

        # Store feedback
        self.feedback_system._store_feedback(feedback)

        # Verify it was stored
        with sqlite3.connect(self.feedback_system.feedback_db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM feedback_entries WHERE feedback_id = ?",
                          (feedback.feedback_id,))
            result = cursor.fetchone()

        assert result is not None
        assert result[0] == feedback.feedback_id  # feedback_id is first column

    def test_model_performance_tracking(self):
        """Test model performance tracking"""
        # Create feedback that should update performance
        feedback = FeedbackEntry(
            feedback_id="perf_test",
            timestamp="2023-10-01T12:00:00",
            expert_id="expert1",
            company_ticker="TEST",
            model_comparison_type="model_selection",
            selected_model="mixtral-8x7b",
            quality_ratings={"mixtral-8x7b": 5}
        )

        # Update performance
        self.feedback_system._update_model_performance(feedback)

        # Get performance report
        performance = self.feedback_system.get_model_performance_report()

        assert "mixtral-8x7b" in performance
        model_perf = performance["mixtral-8x7b"]
        assert model_perf.feedback_count == 1
        assert model_perf.average_quality_rating == 5.0

    def test_training_dataset_generation(self):
        """Test training dataset generation"""
        # Mock multi-LLM result
        mock_result = Mock()
        mock_result.individual_scores = {
            'mixtral-8x7b': {'test_score': ScoreComponent('Test', 4.0, 0.8, 'Good', [], 'test')},
            'llama-3-70b': {'test_score': ScoreComponent('Test', 3.5, 0.7, 'Fair', [], 'test')}
        }

        comparison_data = self.feedback_system.present_model_comparison(mock_result, "expert1")

        assert 'comparison_id' in comparison_data
        assert 'company' in comparison_data
        assert 'models' in comparison_data

class TestWeightApprovalSystem:
    """Test the weight approval system"""

    def setup_method(self):
        """Setup for each test"""
        self.temp_dir = tempfile.mkdtemp()
        self.weight_system = WeightApprovalSystem(self.temp_dir)

    def test_create_approval_session(self):
        """Test creating weight approval session"""
        session = self.weight_system.create_approval_session("user1", "TEST")

        assert session.user_id == "user1"
        assert session.company_ticker == "TEST"
        assert session.approval_status == "pending"
        assert session.original_weights is not None

    def test_present_weights_for_approval(self):
        """Test presenting weights for approval"""
        session = self.weight_system.create_approval_session("user1", "TEST")
        presentation = self.weight_system.present_weights_for_approval(session)

        assert 'session_info' in presentation
        assert 'weight_categories' in presentation
        assert 'approval_options' in presentation
        assert 'impact_examples' in presentation

    def test_process_user_response_approve(self):
        """Test processing user approval response"""
        session = self.weight_system.create_approval_session("user1", "TEST")

        user_response = {
            'response_type': 'approve_default'
        }

        updated_session, approved = self.weight_system.process_user_response(session, user_response)

        assert approved is True
        assert updated_session.approval_status == 'approved'
        assert updated_session.modified_weights is not None

    def test_process_user_response_modify(self):
        """Test processing user weight modification response"""
        session = self.weight_system.create_approval_session("user1", "TEST")

        user_response = {
            'response_type': 'modify_weights',
            'weight_changes': {
                'Barriers to Entry': 0.20,
                'Key Growth Drivers': 0.08
            }
        }

        updated_session, approved = self.weight_system.process_user_response(session, user_response)

        assert approved is True
        assert updated_session.approval_status == 'modified'
        assert updated_session.modified_weights.barriers_to_entry == 0.20

    def test_custom_focus_application(self):
        """Test applying custom focus to weights"""
        original_weights = WeightingScheme()
        focus_areas = ['competitive_moats', 'growth_potential']

        modified_weights = self.weight_system._apply_custom_focus(original_weights, focus_areas)

        # Moat-related weights should be increased
        assert modified_weights.barriers_to_entry > original_weights.barriers_to_entry
        assert modified_weights.key_growth_drivers > original_weights.key_growth_drivers

class TestEnhancedAnalysisController:
    """Test the enhanced analysis controller"""

    def setup_method(self):
        """Setup for each test"""
        self.temp_dir = tempfile.mkdtemp()
        # We'll use mocks for this since the controller integrates many components

    @patch('engines.enhanced_analysis_controller.MultiLLMEngine')
    @patch('engines.enhanced_analysis_controller.WeightApprovalSystem')
    @patch('engines.enhanced_analysis_controller.HumanFeedbackSystem')
    def test_controller_initialization(self, mock_feedback, mock_weights, mock_multi_llm):
        """Test controller initialization"""
        controller = EnhancedAnalysisController(self.temp_dir)

        assert controller.data_dir == Path(self.temp_dir)
        assert hasattr(controller, 'multi_llm_engine')
        assert hasattr(controller, 'scoring_system')
        assert hasattr(controller, 'weight_approval')
        assert hasattr(controller, 'human_feedback')

    def test_analysis_config_creation(self):
        """Test analysis configuration creation"""
        config = EnhancedAnalysisConfig(
            user_id="test_user",
            company_ticker="TEST",
            analysis_type="comprehensive",
            enable_weight_approval=True,
            enable_human_feedback=True,
            expert_id="expert1"
        )

        assert config.user_id == "test_user"
        assert config.company_ticker == "TEST"
        assert config.enable_weight_approval is True
        assert config.enable_human_feedback is True

    def test_cost_estimation(self):
        """Test cost estimation"""
        controller = EnhancedAnalysisController(self.temp_dir)

        # Mock the multi-LLM engine cost estimation
        controller.multi_llm_engine.estimate_multi_llm_cost = Mock(return_value={
            'total_estimated': 0.05,
            'models_count': 5
        })

        config = EnhancedAnalysisConfig(user_id="test", company_ticker="TEST")
        estimate = controller.estimate_analysis_cost(config)

        assert 'total_estimated_cost' in estimate
        assert estimate['total_estimated_cost'] > 0

class TestWorkflowOptimizer:
    """Test the workflow optimizer"""

    def test_workflow_state_creation(self):
        """Test workflow state creation"""
        state = WorkflowState(
            company_ticker="TEST",
            user_id="user1",
            current_step="initialize",
            analysis_data={}
        )

        assert state.company_ticker == "TEST"
        assert state.user_id == "user1"
        assert state.errors == []  # Should be initialized as empty list

    def test_task_optimization(self):
        """Test task execution order optimization"""
        optimizer = WorkflowOptimizer(enable_optimization=False)  # Use fallback

        tasks = [
            {'name': 'high_cost_task', 'estimated_cost': 2.0, 'estimated_time': 1.0, 'dependencies': []},
            {'name': 'low_cost_task', 'estimated_cost': 0.5, 'estimated_time': 0.5, 'dependencies': []},
            {'name': 'dependent_task', 'estimated_cost': 1.0, 'estimated_time': 1.0, 'dependencies': ['task1']}
        ]

        optimized = optimizer.optimize_execution_order(tasks)

        # Should return the tasks (optimization logic can be tested separately)
        assert len(optimized) == len(tasks)

class TestIntegration:
    """Integration tests for the complete enhanced system"""

    def setup_method(self):
        """Setup for integration tests"""
        self.temp_dir = tempfile.mkdtemp()

    @patch('engines.llm_integration.LLMIntegration')
    def test_end_to_end_workflow_simulation(self, mock_llm):
        """Test end-to-end workflow simulation"""
        # This is a simplified integration test
        # In a real scenario, this would test the complete workflow

        # Mock LLM responses
        mock_llm_instance = Mock()
        mock_llm_instance.get_available_models.return_value = ['mixtral-8x7b']
        mock_llm_instance.validate_api_keys.return_value = {'together': True}

        # Test scoring system integration
        scoring_system = EnhancedScoringSystem()
        weights = scoring_system.default_weights

        assert weights is not None

        # Test weight approval integration
        weight_system = WeightApprovalSystem(self.temp_dir)
        session = weight_system.create_approval_session("user1", "TEST")

        assert session.original_weights is not None

        # Test feedback system integration
        feedback_system = HumanFeedbackSystem(self.temp_dir)

        # Verify database was created
        assert feedback_system.feedback_db_path.exists()

    def test_data_format_consistency(self):
        """Test that all systems use consistent data formats"""
        # Test that WeightingScheme can be serialized/deserialized
        weights = WeightingScheme(barriers_to_entry=0.2, brand_monopoly=0.15)
        weights_dict = asdict(weights)
        reconstructed = WeightingScheme(**weights_dict)

        assert reconstructed.barriers_to_entry == weights.barriers_to_entry
        assert reconstructed.brand_monopoly == weights.brand_monopoly

        # Test ScoreComponent serialization
        score = ScoreComponent('Test', 4.0, 0.8, 'Good', ['source1'], 'test')
        score_dict = asdict(score)

        assert score_dict['score'] == 4.0
        assert score_dict['confidence'] == 0.8

# Test runner
if __name__ == "__main__":
    # Run specific test classes
    pytest.main([__file__, "-v", "--tb=short"])