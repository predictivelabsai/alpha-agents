"""
Workflow Optimizer using LangGraph and LangChain
Advanced workflow orchestration for complex multi-step analysis
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import json
import time

try:
    from langchain.schema import BaseMessage, HumanMessage, SystemMessage
    from langchain.memory import ConversationBufferMemory
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("LangChain/LangGraph not available. Install with: pip install langchain langgraph")

logger = logging.getLogger(__name__)

@dataclass
class WorkflowState:
    """State for workflow execution"""
    company_ticker: str
    user_id: str
    current_step: str
    analysis_data: Dict
    weights: Optional[Dict] = None
    user_preferences: Optional[Dict] = None
    llm_results: Optional[Dict] = None
    human_feedback: Optional[Dict] = None
    errors: List[str] = None
    metadata: Dict = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []
        if self.metadata is None:
            self.metadata = {}

class WorkflowOptimizer:
    """Advanced workflow optimizer using LangGraph"""

    def __init__(self, enable_optimization: bool = True):
        self.enable_optimization = enable_optimization and LANGCHAIN_AVAILABLE
        self.memory = ConversationBufferMemory() if LANGCHAIN_AVAILABLE else None
        self.checkpointer = MemorySaver() if LANGCHAIN_AVAILABLE else None

        if not LANGCHAIN_AVAILABLE:
            logger.warning("LangChain/LangGraph not available, using fallback workflow")

        logger.info(f"Workflow Optimizer initialized (optimization: {self.enable_optimization})")

    def create_enhanced_workflow(self) -> Optional['StateGraph']:
        """Create optimized workflow graph"""
        if not self.enable_optimization:
            return None

        try:
            # Define workflow steps
            workflow = StateGraph(WorkflowState)

            # Add workflow nodes
            workflow.add_node("initialize", self._initialize_step)
            workflow.add_node("weight_approval", self._weight_approval_step)
            workflow.add_node("multi_llm_analysis", self._multi_llm_step)
            workflow.add_node("scoring_calculation", self._scoring_step)
            workflow.add_node("human_feedback", self._feedback_step)
            workflow.add_node("result_generation", self._result_step)
            workflow.add_node("error_handling", self._error_handling_step)

            # Define workflow edges
            workflow.set_entry_point("initialize")

            workflow.add_conditional_edges(
                "initialize",
                self._should_require_weight_approval,
                {
                    "weight_approval": "weight_approval",
                    "multi_llm_analysis": "multi_llm_analysis"
                }
            )

            workflow.add_edge("weight_approval", "multi_llm_analysis")
            workflow.add_edge("multi_llm_analysis", "scoring_calculation")

            workflow.add_conditional_edges(
                "scoring_calculation",
                self._should_collect_feedback,
                {
                    "human_feedback": "human_feedback",
                    "result_generation": "result_generation"
                }
            )

            workflow.add_edge("human_feedback", "result_generation")
            workflow.add_edge("result_generation", END)

            # Add error handling
            workflow.add_conditional_edges(
                "error_handling",
                self._should_retry_or_fail,
                {
                    "retry": "initialize",
                    "fail": END
                }
            )

            return workflow.compile(checkpointer=self.checkpointer)

        except Exception as e:
            logger.error(f"Failed to create workflow: {str(e)}")
            return None

    def _initialize_step(self, state: WorkflowState) -> WorkflowState:
        """Initialize workflow step"""
        logger.info(f"Initializing workflow for {state.company_ticker}")

        state.current_step = "initialize"
        state.metadata["workflow_start"] = time.time()
        state.metadata["steps_completed"] = []

        # Load user preferences if available
        try:
            state.user_preferences = self._load_user_preferences(state.user_id)
        except Exception as e:
            state.errors.append(f"Failed to load user preferences: {str(e)}")

        state.metadata["steps_completed"].append("initialize")
        return state

    def _weight_approval_step(self, state: WorkflowState) -> WorkflowState:
        """Weight approval workflow step"""
        logger.info(f"Processing weight approval for {state.company_ticker}")

        state.current_step = "weight_approval"

        try:
            # Simulate weight approval (in real implementation, this would be interactive)
            if state.user_preferences and "weights" in state.user_preferences:
                state.weights = state.user_preferences["weights"]
            else:
                # Use default weights
                from engines.enhanced_scoring_system import EnhancedScoringSystem
                scoring_system = EnhancedScoringSystem()
                state.weights = scoring_system.default_weights.__dict__

            state.metadata["weight_approval_completed"] = True

        except Exception as e:
            error_msg = f"Weight approval failed: {str(e)}"
            state.errors.append(error_msg)
            logger.error(error_msg)

        state.metadata["steps_completed"].append("weight_approval")
        return state

    def _multi_llm_step(self, state: WorkflowState) -> WorkflowState:
        """Multi-LLM analysis workflow step"""
        logger.info(f"Running multi-LLM analysis for {state.company_ticker}")

        state.current_step = "multi_llm_analysis"

        try:
            # This would integrate with the actual multi-LLM engine
            # For now, simulate the process
            state.llm_results = {
                "models_used": ["mixtral-8x7b", "llama-3-70b", "qwen2-72b"],
                "analysis_completed": True,
                "processing_time": 30.0,
                "cost": 0.05
            }

            state.metadata["multi_llm_completed"] = True

        except Exception as e:
            error_msg = f"Multi-LLM analysis failed: {str(e)}"
            state.errors.append(error_msg)
            logger.error(error_msg)

        state.metadata["steps_completed"].append("multi_llm_analysis")
        return state

    def _scoring_step(self, state: WorkflowState) -> WorkflowState:
        """Scoring calculation workflow step"""
        logger.info(f"Calculating scores for {state.company_ticker}")

        state.current_step = "scoring_calculation"

        try:
            # Simulate scoring calculation
            state.analysis_data["composite_score"] = 3.8
            state.analysis_data["confidence"] = 0.85
            state.analysis_data["recommendation"] = "BUY"

            state.metadata["scoring_completed"] = True

        except Exception as e:
            error_msg = f"Scoring calculation failed: {str(e)}"
            state.errors.append(error_msg)
            logger.error(error_msg)

        state.metadata["steps_completed"].append("scoring_calculation")
        return state

    def _feedback_step(self, state: WorkflowState) -> WorkflowState:
        """Human feedback collection workflow step"""
        logger.info(f"Collecting human feedback for {state.company_ticker}")

        state.current_step = "human_feedback"

        try:
            # Simulate feedback collection
            state.human_feedback = {
                "feedback_collected": True,
                "expert_rating": 4,
                "feedback_notes": "Strong analysis quality"
            }

            state.metadata["feedback_completed"] = True

        except Exception as e:
            error_msg = f"Human feedback collection failed: {str(e)}"
            state.errors.append(error_msg)
            logger.error(error_msg)

        state.metadata["steps_completed"].append("human_feedback")
        return state

    def _result_step(self, state: WorkflowState) -> WorkflowState:
        """Result generation workflow step"""
        logger.info(f"Generating results for {state.company_ticker}")

        state.current_step = "result_generation"

        try:
            # Finalize results
            state.analysis_data["workflow_completed"] = True
            state.analysis_data["total_steps"] = len(state.metadata["steps_completed"])
            state.analysis_data["total_time"] = time.time() - state.metadata["workflow_start"]

            state.metadata["workflow_completed"] = True

        except Exception as e:
            error_msg = f"Result generation failed: {str(e)}"
            state.errors.append(error_msg)
            logger.error(error_msg)

        state.metadata["steps_completed"].append("result_generation")
        return state

    def _error_handling_step(self, state: WorkflowState) -> WorkflowState:
        """Error handling workflow step"""
        logger.warning(f"Handling errors for {state.company_ticker}: {state.errors}")

        state.current_step = "error_handling"
        state.metadata["error_handling_invoked"] = True

        # Log errors for analysis
        for error in state.errors:
            logger.error(f"Workflow error: {error}")

        return state

    def _should_require_weight_approval(self, state: WorkflowState) -> str:
        """Conditional logic for weight approval requirement"""
        # Check if user has custom preferences or if approval is required
        if state.user_preferences and "skip_weight_approval" in state.user_preferences:
            return "multi_llm_analysis"
        return "weight_approval"

    def _should_collect_feedback(self, state: WorkflowState) -> str:
        """Conditional logic for human feedback collection"""
        # Check if feedback is enabled and expert is available
        if state.metadata.get("feedback_enabled", False) and not state.errors:
            return "human_feedback"
        return "result_generation"

    def _should_retry_or_fail(self, state: WorkflowState) -> str:
        """Conditional logic for error retry"""
        # Simple retry logic - could be enhanced
        retry_count = state.metadata.get("retry_count", 0)
        if retry_count < 2 and len(state.errors) < 3:
            state.metadata["retry_count"] = retry_count + 1
            return "retry"
        return "fail"

    def _load_user_preferences(self, user_id: str) -> Dict:
        """Load user preferences for workflow optimization"""
        # In real implementation, this would load from database
        default_prefs = {
            "skip_weight_approval": False,
            "preferred_models": ["mixtral-8x7b", "llama-3-70b"],
            "feedback_enabled": True,
            "max_concurrent_models": 3
        }

        try:
            # Try to load user-specific preferences
            prefs_file = Path.cwd() / "data" / f"user_preferences_{user_id}.json"
            if prefs_file.exists():
                with open(prefs_file, 'r') as f:
                    user_prefs = json.load(f)
                return {**default_prefs, **user_prefs}
        except Exception as e:
            logger.warning(f"Failed to load user preferences: {str(e)}")

        return default_prefs

    def optimize_execution_order(self, tasks: List[Dict]) -> List[Dict]:
        """Optimize task execution order based on dependencies and resources"""
        if not self.enable_optimization:
            return tasks

        # Simple optimization: prioritize by cost/time ratio and dependencies
        try:
            # Sort by priority score
            def priority_score(task):
                cost = task.get('estimated_cost', 1.0)
                time = task.get('estimated_time', 1.0)
                dependencies = len(task.get('dependencies', []))
                return (1.0 / (cost * time)) - (dependencies * 0.1)

            optimized_tasks = sorted(tasks, key=priority_score, reverse=True)
            logger.info(f"Optimized execution order for {len(tasks)} tasks")
            return optimized_tasks

        except Exception as e:
            logger.error(f"Task optimization failed: {str(e)}")
            return tasks

    def create_dynamic_prompt(self, base_prompt: str, context: Dict) -> str:
        """Create dynamic prompts based on context and workflow state"""
        if not self.enable_optimization:
            return base_prompt

        try:
            # Enhance prompt with context
            context_additions = []

            if context.get('user_preferences'):
                context_additions.append("User has specific analysis preferences.")

            if context.get('previous_analyses'):
                context_additions.append("Consider insights from previous analyses.")

            if context.get('market_conditions'):
                context_additions.append(f"Current market condition: {context['market_conditions']}")

            if context_additions:
                enhanced_prompt = base_prompt + "\n\nAdditional Context:\n" + "\n".join(context_additions)
                return enhanced_prompt

        except Exception as e:
            logger.error(f"Dynamic prompt creation failed: {str(e)}")

        return base_prompt

    def analyze_workflow_performance(self, execution_data: Dict) -> Dict:
        """Analyze workflow performance for optimization"""
        analysis = {
            'total_execution_time': 0,
            'step_performance': {},
            'bottlenecks': [],
            'optimization_suggestions': []
        }

        try:
            steps = execution_data.get('steps_completed', [])
            step_times = execution_data.get('step_times', {})

            # Calculate performance metrics
            total_time = sum(step_times.values())
            analysis['total_execution_time'] = total_time

            # Identify bottlenecks (steps taking >30% of total time)
            for step, time_taken in step_times.items():
                if time_taken > total_time * 0.3:
                    analysis['bottlenecks'].append({
                        'step': step,
                        'time_taken': time_taken,
                        'percentage': (time_taken / total_time) * 100
                    })

            # Generate optimization suggestions
            if len(analysis['bottlenecks']) > 0:
                analysis['optimization_suggestions'].append(
                    "Consider parallel execution for bottleneck steps"
                )

            if total_time > 60:  # More than 1 minute
                analysis['optimization_suggestions'].append(
                    "Consider caching intermediate results"
                )

        except Exception as e:
            logger.error(f"Workflow performance analysis failed: {str(e)}")

        return analysis

class FallbackWorkflowManager:
    """Fallback workflow manager when LangGraph is not available"""

    def __init__(self):
        logger.info("Using fallback workflow manager")

    def execute_workflow(self, config: Dict, steps: List[Callable]) -> Dict:
        """Execute workflow steps sequentially"""
        results = {}
        start_time = time.time()

        for i, step in enumerate(steps):
            try:
                step_start = time.time()
                step_result = step(config)
                step_time = time.time() - step_start

                results[f'step_{i}'] = {
                    'result': step_result,
                    'execution_time': step_time,
                    'status': 'completed'
                }

            except Exception as e:
                results[f'step_{i}'] = {
                    'error': str(e),
                    'status': 'failed'
                }
                logger.error(f"Step {i} failed: {str(e)}")

        results['total_time'] = time.time() - start_time
        results['workflow_status'] = 'completed'

        return results

# Initialize the appropriate workflow manager
if LANGCHAIN_AVAILABLE:
    workflow_optimizer = WorkflowOptimizer()
else:
    workflow_optimizer = FallbackWorkflowManager()