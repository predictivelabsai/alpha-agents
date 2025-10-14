"""
Enhanced Scoring System for QualAgent
Comprehensive scoring framework for all analysis components with weighted composite scoring
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class ScoreComponent:
    """Individual score component with confidence"""
    name: str
    score: float  # 1-5 scale
    confidence: float  # 0.0-1.0 scale
    justification: str
    sources: List[str]
    category: str  # Main category this belongs to

@dataclass
class WeightingScheme:
    """Weighting scheme for composite scoring"""
    # Core moat components (from original 7-step analysis)
    brand_monopoly: float = 0.10
    barriers_to_entry: float = 0.15
    economies_of_scale: float = 0.10
    network_effects: float = 0.10
    switching_costs: float = 0.10

    # Strategic analysis components
    competitive_differentiation: float = 0.08
    market_timing: float = 0.06
    management_quality: float = 0.05
    technology_moats: float = 0.08

    # Forward-looking components
    key_growth_drivers: float = 0.05
    transformation_potential: float = 0.04
    platform_expansion: float = 0.03

    # Risk factors (negative weighting)
    major_risk_factors: float = -0.06
    red_flags: float = -0.04

    def normalize_weights(self) -> 'WeightingScheme':
        """
        Normalize ALL weights (positive and negative) to sum to 1.0

        This ensures risk factors are properly weighted relative to positive factors.
        After normalization: sum(all_weights) = 1.0
        """
        all_weights = asdict(self)
        total_weight_sum = sum(all_weights.values())

        if abs(total_weight_sum) < 1e-10:  # Avoid division by zero
            return self

        # Normalize ALL weights proportionally
        normalized = WeightingScheme()
        normalization_factor = 1.0 / total_weight_sum

        for key, value in all_weights.items():
            setattr(normalized, key, value * normalization_factor)

        return normalized

class EnhancedScoringSystem:
    """Enhanced scoring system for comprehensive QualAgent analysis"""

    def __init__(self):
        self.default_weights = WeightingScheme()
        self.score_components = {}

    def extract_all_scores(self, analysis_result: Dict) -> Dict[str, ScoreComponent]:
        """Extract scores from all components of analysis result"""
        scores = {}

        # 1. Existing competitive moat scores
        moat_analysis = analysis_result.get('competitive_moat_analysis', {})
        for component, data in moat_analysis.items():
            if isinstance(data, dict) and 'score' in data:
                scores[f"moat_{component.lower().replace(' ', '_')}"] = ScoreComponent(
                    name=component,
                    score=float(data['score']),
                    confidence=float(data.get('confidence', 0.7)),
                    justification=data.get('justification', ''),
                    sources=[],
                    category='competitive_moat'
                )

        # 2. Strategic insights components
        strategic_insights = analysis_result.get('strategic_insights', {})

        # Score growth drivers
        growth_drivers = strategic_insights.get('key_growth_drivers', [])
        if growth_drivers:
            scores['key_growth_drivers'] = self._score_list_component(
                growth_drivers, 'Growth Drivers', 'strategic_insights'
            )

        # Score risk factors (inverse scoring)
        risk_factors = strategic_insights.get('major_risk_factors', [])
        if risk_factors:
            scores['major_risk_factors'] = self._score_list_component(
                risk_factors, 'Risk Factors', 'strategic_insights', inverse=True
            )

        # Score catalysts - try different key names
        catalysts = strategic_insights.get('catalysts_for_next_6_12_months',
                                          strategic_insights.get('catalysts', []))
        if catalysts:
            scores['catalysts_6_12m'] = self._score_list_component(
                catalysts, 'Near-term Catalysts', 'strategic_insights'
            )

        # Score red flags (inverse scoring) - try different key names
        red_flags = strategic_insights.get('red_flags_and_warning_signals',
                                          strategic_insights.get('red_flags', []))
        if red_flags:
            scores['red_flags'] = self._score_list_component(
                red_flags, 'Red Flags', 'strategic_insights', inverse=True
            )

        # Transformation potential
        transformation = strategic_insights.get('transformation_potential_assessment', '')
        if transformation:
            scores['transformation_potential'] = self._score_text_component(
                transformation, 'Transformation Potential', 'strategic_insights'
            )

        # Platform expansion - handle both list and dict formats
        platform_expansion_data = strategic_insights.get('platform_expansion_opportunities',
                                                         strategic_insights.get('platform_expansion', {}))

        if isinstance(platform_expansion_data, list) and platform_expansion_data:
            scores['platform_expansion'] = self._score_list_component(
                platform_expansion_data, 'Platform Expansion', 'strategic_insights'
            )
        elif isinstance(platform_expansion_data, dict) and platform_expansion_data.get('score'):
            # Handle direct score format like {"score": 4, "confidence": 0.8, "justification": "..."}
            scores['platform_expansion'] = ScoreComponent(
                name='Platform Expansion',
                score=float(platform_expansion_data['score']),
                confidence=float(platform_expansion_data.get('confidence', 0.7)),
                justification=platform_expansion_data.get('justification', 'Platform expansion assessment'),
                sources=[],
                category='strategic_insights'
            )

        # Handle other strategic components in similar way
        strategic_components = [
            'transformation_potential', 'competitive_differentiation',
            'market_timing', 'management_quality', 'technology_moats'
        ]

        for component in strategic_components:
            component_data = strategic_insights.get(component, {})
            if isinstance(component_data, dict) and component_data.get('score'):
                scores[component] = ScoreComponent(
                    name=component.replace('_', ' ').title(),
                    score=float(component_data['score']),
                    confidence=float(component_data.get('confidence', 0.7)),
                    justification=component_data.get('justification', f'{component} assessment'),
                    sources=[],
                    category='strategic_insights'
                )


        # 3. Competitor analysis - handle both list and dict formats
        competitor_analysis = analysis_result.get('competitor_analysis', {})
        if isinstance(competitor_analysis, list):
            # Direct list format from JSON response
            competitors = competitor_analysis
        elif isinstance(competitor_analysis, dict):
            # Dict format with 'competitors' key
            competitors = competitor_analysis.get('competitors', [])
        else:
            competitors = []

        if competitors:
            scores['competitive_positioning'] = self._score_competitor_analysis(competitors)

        # 4. Dimensions analysis (if available from new schema)
        dimensions = analysis_result.get('dimensions', [])
        for dim_data in dimensions:
            if isinstance(dim_data, dict) and 'score' in dim_data:
                dim_name = dim_data.get('name', '').lower().replace(' ', '_')
                scores[f"dimension_{dim_name}"] = ScoreComponent(
                    name=dim_data.get('name', ''),
                    score=float(dim_data['score']),
                    confidence=float(dim_data.get('confidence', 0.7)),
                    justification=dim_data.get('justification', ''),
                    sources=dim_data.get('sources', []),
                    category='dimensions'
                )

        return scores

    def _score_list_component(self, items: List, name: str, category: str,
                             inverse: bool = False) -> ScoreComponent:
        """Score a list-based component (growth drivers, risks, etc.)"""
        if not items:
            return ScoreComponent(name, 3.0, 0.3, "No data available", [], category)

        # Handle both string lists and dictionary lists
        processed_items = []
        total_impact = 0
        impact_count = 0

        for item in items:
            if isinstance(item, str):
                # Manual extraction format: simple strings
                processed_items.append(item)
            elif isinstance(item, dict):
                # JSON format: structured dictionaries
                if 'driver' in item:
                    processed_items.append(item['driver'])
                    if 'impact' in item:
                        try:
                            impact_val = float(item.get('impact', 3))
                            total_impact += impact_val
                            impact_count += 1
                        except (ValueError, TypeError):
                            total_impact += 3  # Default impact
                            impact_count += 1
                elif 'risk' in item:
                    processed_items.append(item['risk'])
                    if 'severity' in item:
                        try:
                            severity_val = float(item.get('severity', 3))
                            total_impact += severity_val
                            impact_count += 1
                        except (ValueError, TypeError):
                            total_impact += 3  # Default severity
                            impact_count += 1
                elif 'flag' in item:
                    processed_items.append(item['flag'])
                    if 'severity' in item:
                        try:
                            severity_val = float(item.get('severity', 3))
                            total_impact += severity_val
                            impact_count += 1
                        except (ValueError, TypeError):
                            total_impact += 3  # Default severity
                            impact_count += 1
                else:
                    processed_items.append(str(item))
            else:
                processed_items.append(str(item))

        # Calculate score based on content and impact
        if impact_count > 0:
            # Use average impact from structured data
            avg_impact = total_impact / impact_count
            base_score = min(5.0, max(1.0, avg_impact))
            confidence = 0.7 + min(0.2, impact_count * 0.05)  # Higher confidence with structured data
            justification = f"Based on {len(items)} identified {name.lower()} with average impact {avg_impact:.1f}/5.0"
        else:
            # Fallback to length-based heuristic for string data
            total_length = sum(len(item) for item in processed_items)
            avg_length = total_length / len(processed_items) if processed_items else 0
            base_score = min(5.0, 2.0 + len(processed_items) * 0.5)

            # Adjust for quality (longer descriptions generally indicate more substance)
            if avg_length > 50:
                base_score += 0.5
            elif avg_length < 20:
                base_score -= 0.5

            confidence = 0.6 + min(0.3, len(processed_items) * 0.1)  # Higher confidence with more items
            justification = f"Based on {len(items)} identified {name.lower()}"
            if avg_length > 50:
                justification += " with detailed analysis"

        # Apply inverse scoring for risk factors
        if inverse:
            base_score = 6.0 - base_score  # Invert the scale

        score = max(1.0, min(5.0, base_score))

        return ScoreComponent(name, score, confidence, justification, [], category)

    def _score_text_component(self, text: str, name: str, category: str) -> ScoreComponent:
        """Score a text-based component"""
        if not text or len(text.strip()) < 10:
            return ScoreComponent(name, 3.0, 0.3, "Insufficient information", [], category)

        # Simple sentiment and length-based scoring
        positive_words = ['strong', 'growing', 'innovative', 'leading', 'advantage',
                         'opportunity', 'potential', 'expanding', 'successful']
        negative_words = ['weak', 'declining', 'limited', 'challenges', 'risks',
                         'threats', 'difficult', 'struggling']

        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        # Base score from sentiment
        sentiment_score = 3.0 + (positive_count - negative_count) * 0.3

        # Adjust for text length (more detailed = higher confidence)
        length_bonus = min(1.0, len(text) / 200)  # Up to 1 point for detailed analysis

        score = max(1.0, min(5.0, sentiment_score + length_bonus))
        confidence = 0.5 + min(0.4, len(text) / 300)  # Higher confidence for longer text

        return ScoreComponent(name, score, confidence,
                            f"Sentiment analysis of {len(text)} character assessment",
                            [], category)

    def _score_competitor_analysis(self, competitors: List[Dict]) -> ScoreComponent:
        """Score competitive positioning based on competitor analysis"""
        if not competitors:
            return ScoreComponent("Competitive Positioning", 3.0, 0.3,
                                "No competitor analysis available", [], "competitive")

        # Enhanced competitive positioning logic
        total_threat_level = 0
        market_position_indicators = 0
        num_competitors = len(competitors)

        for competitor in competitors:
            # Threat level analysis (lower threat = better positioning)
            threat_level = competitor.get('threat_level', 3)
            if isinstance(threat_level, (int, float)):
                total_threat_level += threat_level
            else:
                total_threat_level += 3  # Default if not numeric

            # Market position indicators
            positioning = competitor.get('competitive_position', '').lower()
            market_share = competitor.get('market_share', '').lower()

            # Look for indicators of strong competitive position
            strong_indicators = ['leader', 'dominant', 'strong position', 'market leader']
            weak_indicators = ['challenger', 'niche', 'small', 'limited', 'emerging']

            if any(indicator in positioning for indicator in strong_indicators):
                market_position_indicators += 1
            elif any(indicator in positioning for indicator in weak_indicators):
                market_position_indicators -= 0.5

        # Calculate average threat level (inverted for scoring)
        avg_threat_level = total_threat_level / num_competitors if num_competitors > 0 else 3

        # Convert threat level to competitive strength
        # Lower average threat from competitors = stronger competitive position
        threat_score = (6 - avg_threat_level)  # Invert scale: threat 1 = score 5, threat 5 = score 1

        # Market position adjustment
        position_adjustment = min(1.0, market_position_indicators * 0.3)

        # Final score calculation
        base_score = max(1.0, min(5.0, threat_score + position_adjustment))

        # Quality adjustment based on number of competitors analyzed
        if num_competitors >= 3:
            quality_bonus = 0.2
        elif num_competitors >= 2:
            quality_bonus = 0.1
        else:
            quality_bonus = 0

        final_score = min(5.0, base_score + quality_bonus)
        confidence = 0.6 + min(0.3, num_competitors * 0.1)

        justification = f"Analysis of {num_competitors} competitors, average threat level: {avg_threat_level:.1f}/5.0"

        return ScoreComponent("Competitive Positioning", final_score, confidence,
                            justification, [], "competitive")

    def calculate_composite_score(self, scores: Dict[str, ScoreComponent],
                                weights: WeightingScheme = None) -> Tuple[float, float, Dict]:
        """
        Calculate mathematically sound weighted composite score.

        Methodology:
        1. All weights (positive/negative) normalized to sum = 1.0
        2. Direct multiplication: weight * raw_score (no arbitrary inversions)
        3. Negative weights naturally reduce composite score
        4. Confidence tracked as metadata, not double-weighted
        5. Final score can extend beyond 1-5 range (financially meaningful)
        """
        if not scores:
            return 3.0, 0.3, {"error": "No scores available"}

        weights = weights or self.default_weights.normalize_weights()
        weight_dict = asdict(weights)

        # Verify weights sum to 1.0 (within tolerance)
        weight_sum = sum(weight_dict.values())
        if abs(weight_sum - 1.0) > 0.001:
            print(f"Warning: Weights sum to {weight_sum:.3f}, expected 1.000")

        composite_score = 0.0
        total_confidence = 0.0
        components_count = 0
        used_components = {}

        # Map score names to weight names
        score_to_weight_mapping = {
            'moat_brand_monopoly': 'brand_monopoly',
            'moat_barriers_to_entry': 'barriers_to_entry',
            'moat_economies_of_scale': 'economies_of_scale',
            'moat_network_effects': 'network_effects',
            'moat_switching_costs': 'switching_costs',
            'key_growth_drivers': 'key_growth_drivers',
            'major_risk_factors': 'major_risk_factors',
            'red_flags': 'red_flags',
            'transformation_potential': 'transformation_potential',
            'platform_expansion': 'platform_expansion'
        }

        for score_name, score_component in scores.items():
            weight_name = score_to_weight_mapping.get(score_name)
            if weight_name and weight_name in weight_dict:
                weight = weight_dict[weight_name]
                raw_score = score_component.score  # Keep 1-5 scale as-is
                confidence = score_component.confidence

                # Direct multiplication: weight * score
                # Negative weights naturally reduce the composite score
                contribution = weight * raw_score

                composite_score += contribution
                total_confidence += confidence
                components_count += 1

                used_components[score_name] = {
                    'raw_score': raw_score,
                    'confidence': confidence,
                    'weight': weight,
                    'contribution': contribution,
                    'interpretation': 'positive_factor' if weight > 0 else 'risk_factor'
                }

        if components_count == 0:
            return 3.0, 0.3, {"error": "No weighted components found"}

        # Average confidence as metadata
        average_confidence = total_confidence / components_count

        # Note: composite_score can extend beyond 1-5 range, which is financially meaningful
        # High positive weights with high scores can exceed 5.0
        # High risk scores with negative weights can push below 1.0

        metadata = {
            'weight_sum_check': weight_sum,
            'components_used': components_count,
            'component_details': used_components,
            'weights_applied': weight_dict,
            'calculation_method': 'mathematically_consistent_direct_weighting',
            'score_range_note': 'Score may extend beyond 1-5 range due to risk factor weighting'
        }

        return composite_score, average_confidence, metadata

    def get_default_weights_for_approval(self) -> Dict[str, float]:
        """Get default weights formatted for user approval"""
        weights = self.default_weights.normalize_weights()
        return {
            "Core Competitive Moats (Higher Weight)": {
                "Barriers to Entry": weights.barriers_to_entry,
                "Brand Monopoly": weights.brand_monopoly,
                "Economies of Scale": weights.economies_of_scale,
                "Network Effects": weights.network_effects,
                "Switching Costs": weights.switching_costs
            },
            "Strategic Factors (Medium Weight)": {
                "Competitive Differentiation": weights.competitive_differentiation,
                "Technology Moats": weights.technology_moats,
                "Market Timing": weights.market_timing,
                "Management Quality": weights.management_quality
            },
            "Growth & Innovation (Lower Weight)": {
                "Key Growth Drivers": weights.key_growth_drivers,
                "Transformation Potential": weights.transformation_potential,
                "Platform Expansion": weights.platform_expansion
            },
            "Risk Factors (Negative Weight)": {
                "Major Risk Factors": weights.major_risk_factors,
                "Red Flags": weights.red_flags
            }
        }

    def create_weights_from_user_input(self, user_weights: Dict) -> WeightingScheme:
        """Create WeightingScheme from user-provided weights"""
        weights = WeightingScheme()

        # Flatten user weights and map to our structure
        flat_weights = {}
        for category, items in user_weights.items():
            if isinstance(items, dict):
                flat_weights.update(items)
            else:
                flat_weights[category] = items

        # Map user-friendly names to our attribute names
        mapping = {
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

        for user_name, attr_name in mapping.items():
            if user_name in flat_weights:
                setattr(weights, attr_name, float(flat_weights[user_name]))

        return weights.normalize_weights()

# Initialize the scoring system
enhanced_scoring = EnhancedScoringSystem()