#!/usr/bin/env python3
"""
Test script for the new mathematically sound composite scoring system
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from engines.enhanced_scoring_system import WeightingScheme, ScoreComponent, EnhancedScoringSystem

def test_weight_normalization():
    """Test that weights are normalized correctly"""
    print("[TEST] Testing Weight Normalization")
    print("=" * 50)

    # Use default weights for testing
    weights = WeightingScheme()

    # Calculate original sum of ALL weights
    from dataclasses import asdict
    all_weights = asdict(weights)
    original_sum = sum(all_weights.values())

    print("Original weights (showing subset):")
    print(f"  barriers_to_entry: {weights.barriers_to_entry}")
    print(f"  brand_monopoly: {weights.brand_monopoly}")
    print(f"  major_risk_factors: {weights.major_risk_factors}")
    print(f"  red_flags: {weights.red_flags}")
    print(f"  ... and {len(all_weights)-4} other components")
    print(f"  ALL weights sum: {original_sum}")

    # Normalize
    normalized = weights.normalize_weights()

    print("\nNormalized weights (showing subset):")
    print(f"  barriers_to_entry: {normalized.barriers_to_entry:.3f}")
    print(f"  brand_monopoly: {normalized.brand_monopoly:.3f}")
    print(f"  major_risk_factors: {normalized.major_risk_factors:.3f}")
    print(f"  red_flags: {normalized.red_flags:.3f}")

    # Calculate sum of ALL normalized weights
    normalized_weights = asdict(normalized)
    normalized_sum = sum(normalized_weights.values())
    print(f"  ALL normalized weights sum: {normalized_sum:.3f}")

    assert abs(normalized_sum - 1.0) < 0.001, f"Weights should sum to 1.0, got {normalized_sum}"
    print("[PASS] Weight normalization test PASSED")

    return normalized

def test_composite_score_calculation():
    """Test composite score calculation with the new methodology"""
    print("\n[TEST] Testing Composite Score Calculation")
    print("=" * 50)

    # Get normalized weights from previous test
    normalized_weights = test_weight_normalization()

    # Create test scores
    scores = {
        'moat_barriers_to_entry': ScoreComponent('Barriers to Entry', 4.0, 0.8, 'Strong barriers', [], 'moat'),
        'moat_brand_monopoly': ScoreComponent('Brand Monopoly', 3.0, 0.7, 'Good brand', [], 'moat'),
        'major_risk_factors': ScoreComponent('Major Risk Factors', 3.0, 0.9, 'Moderate risks', [], 'risk'),
        'red_flags': ScoreComponent('Red Flags', 2.0, 0.6, 'Some concerns', [], 'risk')
    }

    print("Test scores:")
    for name, score in scores.items():
        print(f"  {name}: {score.score} (confidence: {score.confidence})")

    # Calculate composite score
    scoring_system = EnhancedScoringSystem()
    composite, confidence, metadata = scoring_system.calculate_composite_score(scores, normalized_weights)

    print(f"\nComposite Score: {composite:.3f}")
    print(f"Average Confidence: {confidence:.3f}")

    # Manual verification (using the same weights from normalization test)
    expected_composite = (
        normalized_weights.barriers_to_entry * 4.0 +
        normalized_weights.brand_monopoly * 3.0 +
        normalized_weights.major_risk_factors * 3.0 +
        normalized_weights.red_flags * 2.0
    )

    print(f"Expected (manual calculation): {expected_composite:.3f}")
    print(f"Difference: {abs(composite - expected_composite):.6f}")

    assert abs(composite - expected_composite) < 0.001, f"Composite score mismatch"
    print("[PASS] Composite score calculation test PASSED")

    # Show component contributions
    print("\nComponent contributions:")
    for name, details in metadata['component_details'].items():
        print(f"  {name}: {details['weight']:.3f} * {details['raw_score']:.1f} = {details['contribution']:.3f}")

    return composite, confidence, metadata

def test_financial_interpretation():
    """Test that the results make financial sense"""
    print("\n[TEST] Testing Financial Interpretation")
    print("=" * 50)

    composite, confidence, metadata = test_composite_score_calculation()

    print(f"Final composite score: {composite:.3f}")

    # Financial interpretation
    if composite > 4.0:
        interpretation = "Excellent investment opportunity"
    elif composite > 3.5:
        interpretation = "Good investment opportunity"
    elif composite > 2.5:
        interpretation = "Average investment opportunity"
    elif composite > 2.0:
        interpretation = "Below average, higher risk"
    else:
        interpretation = "High risk, avoid or deep value play"

    print(f"Financial interpretation: {interpretation}")

    # Check that risk factors properly reduced the score
    positive_contribution = sum(
        details['contribution'] for details in metadata['component_details'].values()
        if details['weight'] > 0
    )
    risk_contribution = sum(
        details['contribution'] for details in metadata['component_details'].values()
        if details['weight'] < 0
    )

    print(f"Positive factors contribution: {positive_contribution:.3f}")
    print(f"Risk factors contribution: {risk_contribution:.3f}")
    print(f"Net result: {positive_contribution + risk_contribution:.3f}")

    assert risk_contribution < 0, "Risk factors should have negative contribution"
    assert positive_contribution > 0, "Positive factors should have positive contribution"
    print("[PASS] Financial interpretation test PASSED")

if __name__ == "__main__":
    print("[START] Testing New Mathematically Sound Composite Scoring")
    print("=" * 70)

    try:
        test_weight_normalization()
        test_composite_score_calculation()
        test_financial_interpretation()

        print("\n" + "=" * 70)
        print("[SUCCESS] ALL TESTS PASSED! New scoring methodology is working correctly.")
        print("\n[SUMMARY] Key improvements:")
        print("  [+] All weights normalized together (sum = 1.0)")
        print("  [+] Risk factors directly reduce composite score")
        print("  [+] No arbitrary score inversions (6.0 - score)")
        print("  [+] Confidence as metadata, not double-weighted")
        print("  [+] Mathematically consistent and financially interpretable")

    except Exception as e:
        print(f"\n[ERROR] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()