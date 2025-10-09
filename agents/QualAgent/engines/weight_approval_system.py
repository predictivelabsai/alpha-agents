"""
Weight Approval System for QualAgent
Interactive system for users to review and approve scoring weights before analysis
"""

import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd

from engines.enhanced_scoring_system import WeightingScheme, EnhancedScoringSystem

logger = logging.getLogger(__name__)

@dataclass
class WeightApprovalSession:
    """Session for weight approval process"""
    session_id: str
    timestamp: str
    user_id: str
    company_ticker: str
    original_weights: WeightingScheme
    modified_weights: Optional[WeightingScheme] = None
    approval_status: str = "pending"  # pending, approved, modified, rejected
    user_modifications: Optional[Dict] = None
    session_notes: Optional[str] = None

class WeightApprovalSystem:
    """System for interactive weight approval and modification"""

    def __init__(self, data_dir: str = None):
        self.data_dir = Path(data_dir) if data_dir else Path.cwd() / "data"
        self.scoring_system = EnhancedScoringSystem()
        self.approval_history_file = self.data_dir / "weight_approval_history.json"

        # Load approval history
        self.approval_history = self._load_approval_history()

        logger.info("Weight Approval System initialized")

    def _load_approval_history(self) -> List[Dict]:
        """Load weight approval history from file"""
        if self.approval_history_file.exists():
            with open(self.approval_history_file, 'r') as f:
                return json.load(f)
        return []

    def _save_approval_history(self):
        """Save approval history to file"""
        with open(self.approval_history_file, 'w') as f:
            json.dump(self.approval_history, f, indent=2, default=str)

    def create_approval_session(self, user_id: str, company_ticker: str,
                              custom_weights: WeightingScheme = None) -> WeightApprovalSession:
        """Create a new weight approval session"""
        session_id = f"approval_{int(time.time())}"
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        weights = custom_weights or self.scoring_system.default_weights.normalize_weights()

        session = WeightApprovalSession(
            session_id=session_id,
            timestamp=timestamp,
            user_id=user_id,
            company_ticker=company_ticker,
            original_weights=weights
        )

        logger.info(f"Created weight approval session {session_id} for {company_ticker}")
        return session

    def present_weights_for_approval(self, session: WeightApprovalSession) -> Dict:
        """Present weights to user in a structured format for review"""

        # Get weights in user-friendly format
        weights_display = self.scoring_system.get_default_weights_for_approval()

        presentation = {
            'session_info': {
                'session_id': session.session_id,
                'company_ticker': session.company_ticker,
                'timestamp': session.timestamp,
                'user_id': session.user_id
            },
            'instructions': {
                'overview': "Please review the scoring weights below. These determine how much each component contributes to the final composite score.",
                'core_moats': "Core Competitive Moats have higher weights as they are part of the main 7-step analysis framework.",
                'strategic_factors': "Strategic factors provide important context with medium weights.",
                'growth_innovation': "Growth and innovation factors are valuable but weighted lower as they are more speculative.",
                'risk_factors': "Risk factors have negative weights and reduce the overall score.",
                'modification_options': [
                    "Approve as-is: Use the default weights",
                    "Modify weights: Adjust any weights to match your investment philosophy",
                    "Add custom focus: Emphasize specific factors for this analysis"
                ]
            },
            'weight_categories': weights_display,
            'weight_totals': self._calculate_weight_totals(session.original_weights),
            'impact_examples': self._generate_impact_examples(session.original_weights),
            'approval_options': {
                'approve_default': "Use these weights as-is",
                'modify_weights': "I want to modify some weights",
                'custom_focus': "Apply custom focus (increase weight for specific factors)",
                'postpone': "Review this later"
            }
        }

        return presentation

    def _calculate_weight_totals(self, weights: WeightingScheme) -> Dict:
        """Calculate weight category totals"""
        weight_dict = asdict(weights)

        totals = {
            'core_moats_total': sum([
                weight_dict['brand_monopoly'],
                weight_dict['barriers_to_entry'],
                weight_dict['economies_of_scale'],
                weight_dict['network_effects'],
                weight_dict['switching_costs']
            ]),
            'strategic_factors_total': sum([
                weight_dict['competitive_differentiation'],
                weight_dict['technology_moats'],
                weight_dict['market_timing'],
                weight_dict['management_quality']
            ]),
            'growth_factors_total': sum([
                weight_dict['key_growth_drivers'],
                weight_dict['transformation_potential'],
                weight_dict['platform_expansion']
            ]),
            'risk_factors_total': abs(sum([
                weight_dict['major_risk_factors'],
                weight_dict['red_flags']
            ]))
        }

        totals['positive_total'] = sum([
            totals['core_moats_total'],
            totals['strategic_factors_total'],
            totals['growth_factors_total']
        ])

        return totals

    def _generate_impact_examples(self, weights: WeightingScheme) -> Dict:
        """Generate examples of how weights affect scoring"""
        examples = {
            'high_moat_company': {
                'description': "Company with strong competitive moats but average growth",
                'example_scores': {
                    'Barriers to Entry': 5,
                    'Brand Monopoly': 4,
                    'Growth Drivers': 3,
                    'Risk Factors': 2
                },
                'estimated_composite': self._estimate_composite_score(weights, {
                    'barriers_to_entry': 5, 'brand_monopoly': 4,
                    'key_growth_drivers': 3, 'major_risk_factors': 2
                })
            },
            'high_growth_company': {
                'description': "High-growth company with moderate competitive advantages",
                'example_scores': {
                    'Barriers to Entry': 3,
                    'Brand Monopoly': 3,
                    'Growth Drivers': 5,
                    'Risk Factors': 2
                },
                'estimated_composite': self._estimate_composite_score(weights, {
                    'barriers_to_entry': 3, 'brand_monopoly': 3,
                    'key_growth_drivers': 5, 'major_risk_factors': 2
                })
            },
            'risky_company': {
                'description': "Company with potential but significant risk factors",
                'example_scores': {
                    'Barriers to Entry': 4,
                    'Brand Monopoly': 3,
                    'Growth Drivers': 4,
                    'Risk Factors': 4
                },
                'estimated_composite': self._estimate_composite_score(weights, {
                    'barriers_to_entry': 4, 'brand_monopoly': 3,
                    'key_growth_drivers': 4, 'major_risk_factors': 4
                })
            }
        }

        return examples

    def _estimate_composite_score(self, weights: WeightingScheme, sample_scores: Dict) -> float:
        """Estimate composite score for example scenarios"""
        weighted_sum = 0.0
        total_weight = 0.0

        weight_dict = asdict(weights)
        for component, score in sample_scores.items():
            if component in weight_dict:
                weight = weight_dict[component]
                if weight < 0:  # Risk factors
                    contribution = abs(weight) * (6.0 - score)  # Inverse for risks
                else:
                    contribution = weight * score

                weighted_sum += contribution
                total_weight += abs(weight)

        return weighted_sum / total_weight if total_weight > 0 else 3.0

    def process_user_response(self, session: WeightApprovalSession,
                            user_response: Dict) -> Tuple[WeightApprovalSession, bool]:
        """Process user response to weight approval"""

        response_type = user_response.get('response_type')
        session.user_modifications = user_response

        if response_type == 'approve_default':
            session.approval_status = 'approved'
            session.modified_weights = session.original_weights
            session.session_notes = "User approved default weights"
            approved = True

        elif response_type == 'modify_weights':
            # Process weight modifications
            modified_weights = self._apply_weight_modifications(
                session.original_weights, user_response.get('weight_changes', {})
            )
            session.modified_weights = modified_weights
            session.approval_status = 'modified'
            session.session_notes = f"User modified {len(user_response.get('weight_changes', {}))} weights"
            approved = True

        elif response_type == 'custom_focus':
            # Apply custom focus modifications
            focus_areas = user_response.get('focus_areas', [])
            modified_weights = self._apply_custom_focus(session.original_weights, focus_areas)
            session.modified_weights = modified_weights
            session.approval_status = 'custom_focus'
            session.session_notes = f"User applied custom focus to: {', '.join(focus_areas)}"
            approved = True

        elif response_type == 'postpone':
            session.approval_status = 'postponed'
            session.session_notes = "User postponed weight approval"
            approved = False

        else:
            session.approval_status = 'rejected'
            session.session_notes = "Invalid or rejected response"
            approved = False

        # Save to history
        self._save_session_to_history(session)

        logger.info(f"Processed user response for session {session.session_id}: {session.approval_status}")
        return session, approved

    def _apply_weight_modifications(self, original_weights: WeightingScheme,
                                  weight_changes: Dict) -> WeightingScheme:
        """Apply user-specified weight modifications"""
        modified_weights = WeightingScheme(**asdict(original_weights))

        # Map user-friendly names to attribute names
        name_mapping = {
            "Barriers to Entry": "barriers_to_entry",
            "Brand Monopoly": "brand_monopoly",
            "Economies of Scale": "economies_of_scale",
            "Network Effects": "network_effects",
            "Switching Costs": "switching_costs",
            "Competitive Differentiation": "competitive_differentiation",
            "Technology Moats": "technology_moats",
            "Market Timing": "market_timing",
            "Management Quality": "management_quality",
            "Key Growth Drivers": "key_growth_drivers",
            "Transformation Potential": "transformation_potential",
            "Platform Expansion": "platform_expansion",
            "Major Risk Factors": "major_risk_factors",
            "Red Flags": "red_flags"
        }

        for user_name, new_weight in weight_changes.items():
            attr_name = name_mapping.get(user_name)
            if attr_name and hasattr(modified_weights, attr_name):
                setattr(modified_weights, attr_name, float(new_weight))

        return modified_weights.normalize_weights()

    def _apply_custom_focus(self, original_weights: WeightingScheme,
                          focus_areas: List[str]) -> WeightingScheme:
        """Apply custom focus by increasing weights for specified areas"""
        modified_weights = WeightingScheme(**asdict(original_weights))

        # Focus area multipliers
        focus_multiplier = 1.3  # Increase focused areas by 30%

        focus_mapping = {
            'competitive_moats': ['barriers_to_entry', 'brand_monopoly', 'economies_of_scale',
                                'network_effects', 'switching_costs'],
            'growth_potential': ['key_growth_drivers', 'transformation_potential', 'platform_expansion'],
            'technology_strength': ['technology_moats', 'competitive_differentiation'],
            'management_execution': ['management_quality', 'market_timing'],
            'risk_assessment': ['major_risk_factors', 'red_flags']
        }

        for focus_area in focus_areas:
            if focus_area in focus_mapping:
                for attr_name in focus_mapping[focus_area]:
                    if hasattr(modified_weights, attr_name):
                        current_value = getattr(modified_weights, attr_name)
                        new_value = current_value * focus_multiplier
                        setattr(modified_weights, attr_name, new_value)

        return modified_weights.normalize_weights()

    def _save_session_to_history(self, session: WeightApprovalSession):
        """Save approval session to history"""
        session_dict = {
            'session_id': session.session_id,
            'timestamp': session.timestamp,
            'user_id': session.user_id,
            'company_ticker': session.company_ticker,
            'original_weights': asdict(session.original_weights),
            'modified_weights': asdict(session.modified_weights) if session.modified_weights else None,
            'approval_status': session.approval_status,
            'user_modifications': session.user_modifications,
            'session_notes': session.session_notes
        }

        self.approval_history.append(session_dict)
        self._save_approval_history()

    def get_user_weight_preferences(self, user_id: str) -> Dict:
        """Get historical weight preferences for a user"""
        user_sessions = [session for session in self.approval_history
                        if session['user_id'] == user_id and session['approval_status'] in ['approved', 'modified', 'custom_focus']]

        if not user_sessions:
            return {'message': 'No historical preferences found', 'use_defaults': True}

        # Analyze user's historical preferences
        common_modifications = {}
        focus_patterns = []

        for session in user_sessions[-5:]:  # Last 5 sessions
            if session['user_modifications']:
                mods = session['user_modifications']
                if mods.get('response_type') == 'modify_weights':
                    for weight_name, value in mods.get('weight_changes', {}).items():
                        if weight_name not in common_modifications:
                            common_modifications[weight_name] = []
                        common_modifications[weight_name].append(float(value))

                elif mods.get('response_type') == 'custom_focus':
                    focus_patterns.extend(mods.get('focus_areas', []))

        # Calculate average preferences
        avg_preferences = {}
        for weight_name, values in common_modifications.items():
            avg_preferences[weight_name] = sum(values) / len(values)

        return {
            'has_preferences': True,
            'average_modifications': avg_preferences,
            'common_focus_areas': list(set(focus_patterns)),
            'sessions_analyzed': len(user_sessions),
            'recommendation': 'Use historical preferences as starting point'
        }

    def generate_approval_summary(self, session: WeightApprovalSession) -> Dict:
        """Generate summary of approved weights for confirmation"""
        if not session.modified_weights:
            return {'error': 'No approved weights available'}

        summary = {
            'session_info': {
                'session_id': session.session_id,
                'company_ticker': session.company_ticker,
                'approval_status': session.approval_status,
                'timestamp': session.timestamp
            },
            'weight_comparison': self._compare_weights(session.original_weights, session.modified_weights),
            'impact_analysis': self._analyze_weight_impact(session.original_weights, session.modified_weights),
            'ready_for_analysis': session.approval_status in ['approved', 'modified', 'custom_focus']
        }

        return summary

    def _compare_weights(self, original: WeightingScheme, modified: WeightingScheme) -> Dict:
        """Compare original and modified weights"""
        orig_dict = asdict(original)
        mod_dict = asdict(modified)

        comparison = {}
        for key in orig_dict:
            comparison[key] = {
                'original': orig_dict[key],
                'modified': mod_dict[key],
                'change': mod_dict[key] - orig_dict[key],
                'change_percent': ((mod_dict[key] - orig_dict[key]) / orig_dict[key] * 100)
                                if orig_dict[key] != 0 else 0
            }

        return comparison

    def _analyze_weight_impact(self, original: WeightingScheme, modified: WeightingScheme) -> Dict:
        """Analyze the impact of weight changes"""
        # Test with sample scenarios to show impact
        test_scenarios = {
            'balanced_company': {'barriers_to_entry': 4, 'brand_monopoly': 3, 'key_growth_drivers': 4},
            'moat_heavy_company': {'barriers_to_entry': 5, 'brand_monopoly': 5, 'key_growth_drivers': 2},
            'growth_focused_company': {'barriers_to_entry': 2, 'brand_monopoly': 2, 'key_growth_drivers': 5}
        }

        impact_analysis = {}
        for scenario_name, scores in test_scenarios.items():
            original_score = self._estimate_composite_score(original, scores)
            modified_score = self._estimate_composite_score(modified, scores)

            impact_analysis[scenario_name] = {
                'original_score': original_score,
                'modified_score': modified_score,
                'score_change': modified_score - original_score,
                'impact_magnitude': abs(modified_score - original_score)
            }

        return impact_analysis

    def create_interactive_approval_interface(self, session: WeightApprovalSession) -> str:
        """Create an interactive interface for weight approval (returns HTML/text)"""
        presentation = self.present_weights_for_approval(session)

        # Create text-based interface (could be enhanced with HTML/Streamlit)
        interface_text = f"""
=== QUALAGENT SCORING WEIGHTS APPROVAL ===
Session: {session.session_id}
Company: {session.company_ticker}
User: {session.user_id}

{presentation['instructions']['overview']}

CURRENT WEIGHTS:

Core Competitive Moats (Primary Analysis - Higher Weight):
"""

        for category, weights in presentation['weight_categories'].items():
            interface_text += f"\n{category}:\n"
            if isinstance(weights, dict):
                for name, weight in weights.items():
                    interface_text += f"  • {name}: {weight:.3f}\n"

        interface_text += f"""

WEIGHT TOTALS:
{json.dumps(presentation['weight_totals'], indent=2)}

IMPACT EXAMPLES:
{json.dumps(presentation['impact_examples'], indent=2)}

APPROVAL OPTIONS:
1. approve_default - Use these weights as-is
2. modify_weights - Adjust specific weights
3. custom_focus - Apply focus to specific areas
4. postpone - Review later

To proceed, provide your response in this format:
{{
    "response_type": "approve_default" | "modify_weights" | "custom_focus" | "postpone",
    "weight_changes": {{"Weight Name": new_value}},  // if modifying
    "focus_areas": ["area1", "area2"],  // if using custom focus
    "notes": "Optional comments"
}}
"""

        return interface_text