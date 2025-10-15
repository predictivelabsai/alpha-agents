#!/usr/bin/env python3
"""
Test Weight Loading Mechanism
Quick verification that custom weights are loaded correctly
"""

import json
import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from run_enhanced_analysis import EnhancedQualAgentDemo
from engines.enhanced_scoring_system import WeightingScheme

def test_weight_loading():
    """Test the weight loading mechanism"""
    print("🔧 TESTING WEIGHT LOADING MECHANISM")
    print("=" * 50)

    # Create a test weight file
    test_weights = {
        'timestamp': 1234567890,
        'approved_by': 'test_user',
        'description': 'Test weight configuration',
        'weights': {
            'barriers_to_entry': 0.25,
            'brand_monopoly': 0.20,
            'economies_of_scale': 0.15,
            'network_effects': 0.10,
            'switching_costs': 0.08,
            'competitive_differentiation': 0.12,
            'technology_moats': 0.05,
            'management_quality': 0.03,
            'major_risk_factors': -0.08,
            'red_flags': -0.05
        }
    }

    test_file = 'test_weights.json'
    with open(test_file, 'w') as f:
        json.dump(test_weights, f, indent=2)

    print(f"✅ Created test weight file: {test_file}")

    # Test loading
    demo = EnhancedQualAgentDemo()

    try:
        loaded_weights = demo._load_custom_weights(test_file)

        if loaded_weights is None:
            print("❌ Failed to load weights - returned None")
            return False

        print("✅ Successfully loaded weights!")

        # Verify specific weights
        expected_barriers = 0.25
        actual_barriers = loaded_weights.barriers_to_entry

        print(f"\n🔍 VERIFICATION:")
        print(f"  Expected barriers_to_entry: {expected_barriers}")
        print(f"  Actual barriers_to_entry: {actual_barriers}")

        if abs(actual_barriers - expected_barriers) < 0.001:
            print("  ✅ Weight values match!")
        else:
            print("  ❌ Weight values don't match!")
            return False

        # Test normalization
        total_weight = sum([
            loaded_weights.barriers_to_entry,
            loaded_weights.brand_monopoly,
            loaded_weights.economies_of_scale,
            loaded_weights.network_effects,
            loaded_weights.switching_costs,
            loaded_weights.competitive_differentiation,
            loaded_weights.technology_moats,
            loaded_weights.management_quality,
            loaded_weights.major_risk_factors,
            loaded_weights.red_flags
        ])

        print(f"\n📊 NORMALIZATION CHECK:")
        print(f"  Total weight sum: {total_weight:.3f}")
        if abs(total_weight - 1.0) < 0.001:
            print("  ✅ Weights properly normalized!")
        else:
            print("  ⚠️  Weights not normalized (this is expected before normalization)")

        # Clean up
        Path(test_file).unlink()
        print(f"\n🧹 Cleaned up test file: {test_file}")

        return True

    except Exception as e:
        print(f"❌ Error during weight loading test: {e}")
        # Clean up
        if Path(test_file).exists():
            Path(test_file).unlink()
        return False

def test_command_line_args():
    """Test command line argument parsing for custom weights"""
    print(f"\n🔧 TESTING COMMAND LINE ARGUMENT PARSING")
    print("=" * 50)

    # Create mock args object
    class MockArgs:
        def __init__(self):
            self.user_id = 'test_user'
            self.company = 'TEST'
            self.analysis_type = 'expert_guided'
            self.models = None
            self.themes = None
            self.geographies = 'US,Global'
            self.lookback_months = 24
            self.enable_weights = True
            self.enable_feedback = True
            self.max_concurrent = 3
            self.expert_id = 'test_expert'
            self.custom_weights = 'approved_weights.json'  # This should be recognized

    # Test with custom weights
    demo = EnhancedQualAgentDemo()
    mock_args = MockArgs()

    # Create a dummy approved_weights.json file
    test_weights = {
        'timestamp': 1234567890,
        'approved_by': 'chenHX',
        'weights': {
            'barriers_to_entry': 0.30,
            'brand_monopoly': 0.25,
            'major_risk_factors': -0.10
        }
    }

    with open('approved_weights.json', 'w') as f:
        json.dump(test_weights, f, indent=2)

    try:
        config = demo.create_analysis_config_from_args(mock_args)

        if hasattr(config, 'custom_weights') and config.custom_weights is not None:
            print("✅ Custom weights loaded in config!")
            print(f"   barriers_to_entry: {config.custom_weights.barriers_to_entry:.3f}")
        else:
            print("❌ Custom weights not loaded in config")
            return False

        # Clean up
        Path('approved_weights.json').unlink()
        print("🧹 Cleaned up test file")

        return True

    except Exception as e:
        print(f"❌ Error during command line test: {e}")
        if Path('approved_weights.json').exists():
            Path('approved_weights.json').unlink()
        return False

if __name__ == "__main__":
    print("🚀 WEIGHT LOADING VERIFICATION TESTS")
    print("=" * 60)

    # Test 1: Direct weight loading
    test1_passed = test_weight_loading()

    # Test 2: Command line argument integration
    test2_passed = test_command_line_args()

    # Results
    print(f"\n📋 TEST RESULTS:")
    print(f"  Weight Loading: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"  Command Line Integration: {'✅ PASSED' if test2_passed else '❌ FAILED'}")

    if test1_passed and test2_passed:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"   Custom weight loading is working correctly.")
        print(f"   You can now use: --custom-weights approved_weights.json")
    else:
        print(f"\n❌ SOME TESTS FAILED!")
        print(f"   Custom weight loading needs debugging.")

    sys.exit(0 if (test1_passed and test2_passed) else 1)