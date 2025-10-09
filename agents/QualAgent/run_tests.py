#!/usr/bin/env python3
"""
Fixed Test Runner for QualAgent Enhanced Systems
Quick test runner to validate all enhanced features
"""

import sys
import subprocess
from pathlib import Path

def run_tests():
    """Run all tests for enhanced systems"""

    print("TEST QualAgent Enhanced Systems Test Runner")
    print("=" * 50)

    # Check if pytest is available
    try:
        import pytest
        print("PASS pytest found")
    except ImportError:
        print("FAIL pytest not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pytest"])
        import pytest
        print("PASS pytest installed")

    # Check test directory
    test_dir = Path(__file__).parent / "tests"
    if not test_dir.exists():
        test_dir.mkdir()
        print(f"FOLDER Created test directory: {test_dir}")

    test_file = test_dir / "test_enhanced_systems.py"
    if not test_file.exists():
        print(f"FAIL Test file not found: {test_file}")
        return False

    print(f"CLIPBOARD Running tests from: {test_file}")
    print("-" * 50)

    # Run tests
    try:
        result = pytest.main([
            str(test_file),
            "-v",
            "--tb=short",
            "--color=yes",
            "--durations=10"
        ])

        if result == 0:
            print("\nPASS All tests passed!")
            return True
        else:
            print(f"\nFAIL Tests failed with exit code: {result}")
            return False

    except Exception as e:
        print(f"FAIL Error running tests: {str(e)}")
        return False

def run_quick_system_check():
    """Run quick system check without full test suite"""

    print("\nSEARCH Quick System Check")
    print("-" * 30)

    try:
        # Test imports
        print("Testing imports...")

        from engines.enhanced_scoring_system import EnhancedScoringSystem
        print("PASS Enhanced Scoring System")

        from engines.multi_llm_engine import MultiLLMEngine
        print("PASS Multi-LLM Engine")

        from engines.human_feedback_system import HumanFeedbackSystem
        print("PASS Human Feedback System")

        from engines.weight_approval_system import WeightApprovalSystem
        print("PASS Weight Approval System")

        from engines.enhanced_analysis_controller import EnhancedAnalysisController
        print("PASS Enhanced Analysis Controller")

        try:
            from engines.workflow_optimizer import WorkflowOptimizer
            print("PASS Workflow Optimizer")
        except Exception as e:
            print(f"WARNING Workflow Optimizer (optional): {str(e)[:50]}...")
            print("   (Advanced workflow features may not be available)")

        # Test basic functionality
        print("\nTesting basic functionality...")

        scoring_system = EnhancedScoringSystem()
        weights = scoring_system.default_weights
        assert weights is not None
        print("PASS Scoring system initialization")

        weight_display = scoring_system.get_default_weights_for_approval()
        assert 'Core Competitive Moats (Higher Weight)' in weight_display
        print("PASS Weight display formatting")

        # Test database initialization
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_system = HumanFeedbackSystem(temp_dir)
            assert feedback_system.feedback_db_path.exists()
            print("PASS Database initialization")

        print("\nPASS Quick system check passed!")
        return True

    except ImportError as e:
        print(f"FAIL Import error: {str(e)}")
        print("TIP Run: pip install -r requirements_enhanced.txt")
        return False

    except Exception as e:
        print(f"FAIL System check error: {str(e)}")
        return False

def check_dependencies():
    """Check if all required dependencies are installed"""

    print("\nPACKAGE Dependency Check")
    print("-" * 20)

    # Package mapping: display_name -> import_name
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'requests': 'requests',
        'python-dotenv': 'dotenv',  # python-dotenv imports as 'dotenv'
        'openai': 'openai',
        'flask': 'flask',
        'pytest': 'pytest',
        'sqlite3': 'sqlite3'
    }

    # Enhanced packages (required for new features)
    enhanced_packages = {
        'langchain': 'langchain',
        'langgraph': 'langgraph',
        'langchain-core': 'langchain_core',
        'scipy': 'scipy',
        'scikit-learn': 'sklearn',  # scikit-learn imports as 'sklearn'
        'sqlalchemy': 'sqlalchemy'
    }

    # Optional packages
    optional_packages = {
        'plotly': 'plotly',
        'streamlit': 'streamlit',
        'psutil': 'psutil',
        'memory-profiler': 'memory_profiler',
        'black': 'black',
        'flake8': 'flake8'
    }

    missing_required = []
    missing_enhanced = []
    missing_optional = []

    print("Core Dependencies:")
    # Check required packages
    for display_name, import_name in required_packages.items():
        try:
            if import_name == 'sqlite3':
                import sqlite3
            else:
                __import__(import_name)
            print(f"  PASS {display_name}")
        except ImportError:
            missing_required.append(display_name)
            print(f"  FAIL {display_name}")

    print("\nEnhanced Features Dependencies:")
    # Check enhanced packages
    for display_name, import_name in enhanced_packages.items():
        try:
            __import__(import_name)
            print(f"  PASS {display_name}")
        except ImportError:
            missing_enhanced.append(display_name)
            print(f"  FAIL {display_name}")

    print("\nOptional Dependencies:")
    # Check optional packages
    for display_name, import_name in optional_packages.items():
        try:
            __import__(import_name)
            print(f"  PASS {display_name}")
        except ImportError:
            missing_optional.append(display_name)
            print(f"  WARNING  {display_name}")

    # Report results
    if missing_required:
        print(f"\nFAIL Missing REQUIRED packages: {', '.join(missing_required)}")
        print(f"TIP Install with: pip install {' '.join(missing_required)}")
        return False

    if missing_enhanced:
        print(f"\nWARNING  Missing ENHANCED FEATURES packages: {', '.join(missing_enhanced)}")
        print(f"TIP Install with: pip install {' '.join(missing_enhanced)}")
        print("   (Required for multi-LLM and advanced features)")

        # Allow continuation but warn user
        user_input = input("\nContinue with basic functionality only? (y/N): ")
        if user_input.lower() != 'y':
            return False

    if missing_optional:
        print(f"\nWARNING  Missing OPTIONAL packages: {', '.join(missing_optional)}")
        print(f"TIP Install with: pip install {' '.join(missing_optional)}")
        print("   (These enhance functionality but are not required)")

    print("\nPASS All required dependencies found!")
    return True

def test_api_keys():
    """Test API key configuration"""

    print("\nKEY API Key Check")
    print("-" * 20)

    try:
        from dotenv import load_dotenv
        import os

        # Load environment variables
        load_dotenv()

        # Check TogetherAI key
        together_key = os.getenv('TOGETHER_API_KEY')
        if together_key:
            print(f"PASS TogetherAI API Key: {together_key[:10]}...")
        else:
            print("FAIL TogetherAI API Key not found")

        # Check OpenAI key
        openai_key = os.getenv('OPENAI_API_KEY')
        if openai_key:
            print(f"PASS OpenAI API Key: {openai_key[:10]}...")
        else:
            print("WARNING  OpenAI API Key not found (optional)")

        # Check research tool keys
        tavily_key = os.getenv('TAVILY_API_KEY')
        if tavily_key:
            print(f"PASS Tavily API Key: {tavily_key[:10]}...")
        else:
            print("WARNING  Tavily API Key not found (optional)")

        return True

    except Exception as e:
        print(f"FAIL Error checking API keys: {str(e)}")
        return False

def main():
    """Main test runner"""

    print("QualAgent Enhanced Test Suite")
    print("=" * 40)

    # Check dependencies first
    if not check_dependencies():
        print("\nDependency check failed. Please install missing packages.")
        return 1

    # Check API keys
    test_api_keys()

    # Run quick system check
    if not run_quick_system_check():
        print("\nFAIL System check failed.")
        return 1

    # Ask user if they want to run full test suite
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        print("\n" + "=" * 50)
        if run_tests():
            print("\nSUCCESS All systems operational!")
            return 0
        else:
            print("\nFAIL Some tests failed.")
            return 1
    else:
        print("\nTIP Run with --full flag to execute complete test suite")
        print("   python run_tests_fixed.py --full")
        print("\nSUCCESS Quick checks passed! System appears to be working correctly.")
        return 0

if __name__ == "__main__":
    sys.exit(main())