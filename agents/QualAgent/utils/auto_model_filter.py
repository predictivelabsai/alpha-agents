#!/usr/bin/env python3
"""
Automatic Model Filtering for QualAgent
Discovers working LLM models and generates optimal model configuration
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.test_llm_api import LLMAPITester

class AutoModelFilter:
    """Automatically filter and recommend working LLM models"""

    def __init__(self):
        self.tester = LLMAPITester()
        self.working_models = []
        self.failed_models = []
        self.performance_metrics = {}

    def discover_working_models(self) -> Dict[str, List[str]]:
        """Discover all working models and categorize them"""
        print("🔍 DISCOVERING WORKING LLM MODELS")
        print("=" * 50)

        # Run comprehensive tests
        integration_results = self.tester.test_all_models()
        endpoint_results = self.tester.test_model_endpoints()

        # Categorize models by performance
        excellent_models = []  # Fast, cheap, reliable
        good_models = []       # Good balance of speed/cost/quality
        usable_models = []     # Works but slower/expensive

        for model, result in integration_results.items():
            if result['success']:
                # Calculate performance score
                speed_score = max(0, 5 - (result['time'] / 5))  # Faster = better
                cost_score = max(0, 5 - (result['cost'] * 1000))  # Cheaper = better
                reliability_score = 5 if result['json_parseable'] else 3

                total_score = (speed_score + cost_score + reliability_score) / 3

                model_info = {
                    'model': model,
                    'time': result['time'],
                    'cost': result['cost'],
                    'score': total_score,
                    'json_parseable': result['json_parseable']
                }

                if total_score >= 4.0:
                    excellent_models.append(model_info)
                elif total_score >= 3.0:
                    good_models.append(model_info)
                else:
                    usable_models.append(model_info)

                self.working_models.append(model)
                self.performance_metrics[model] = model_info
            else:
                self.failed_models.append(model)

        # Sort by performance
        excellent_models.sort(key=lambda x: x['score'], reverse=True)
        good_models.sort(key=lambda x: x['score'], reverse=True)
        usable_models.sort(key=lambda x: x['score'], reverse=True)

        return {
            'excellent': [m['model'] for m in excellent_models],
            'good': [m['model'] for m in good_models],
            'usable': [m['model'] for m in usable_models],
            'failed': self.failed_models
        }

    def generate_optimal_configurations(self, categorized_models: Dict[str, List[str]]) -> Dict[str, Dict]:
        """Generate optimal model configurations for different use cases"""
        configs = {}

        # High-performance configuration (best models only)
        high_perf_models = categorized_models['excellent'][:3]
        if len(high_perf_models) < 2:
            high_perf_models.extend(categorized_models['good'][:3-len(high_perf_models)])

        configs['high_performance'] = {
            'models': high_perf_models,
            'description': 'Fastest, most reliable models for high-frequency analysis',
            'use_case': 'Production environment, real-time analysis',
            'expected_time': '15-25 seconds',
            'expected_cost': '$0.004-0.008'
        }

        # Balanced configuration (mix of excellent and good)
        balanced_models = []
        balanced_models.extend(categorized_models['excellent'][:2])
        balanced_models.extend(categorized_models['good'][:2])
        if len(balanced_models) < 3:
            balanced_models.extend(categorized_models['usable'][:3-len(balanced_models)])

        configs['balanced'] = {
            'models': balanced_models,
            'description': 'Optimal balance of speed, cost, and consensus quality',
            'use_case': 'Standard analysis, expert-guided workflows',
            'expected_time': '20-35 seconds',
            'expected_cost': '$0.006-0.015'
        }

        # Cost-optimized configuration (cheapest working models)
        all_working = (categorized_models['excellent'] +
                      categorized_models['good'] +
                      categorized_models['usable'])

        cost_optimized = sorted(all_working,
                               key=lambda m: self.performance_metrics[m]['cost'])[:3]

        configs['cost_optimized'] = {
            'models': cost_optimized,
            'description': 'Cheapest working models while maintaining consensus quality',
            'use_case': 'Batch processing, large-scale analysis',
            'expected_time': '25-45 seconds',
            'expected_cost': '$0.002-0.008'
        }

        # Consensus-maximizing configuration (all working models)
        consensus_models = all_working[:5]  # Limit to 5 for practical reasons

        configs['consensus_maximizing'] = {
            'models': consensus_models,
            'description': 'Maximum number of models for highest consensus quality',
            'use_case': 'Critical decisions, research validation',
            'expected_time': '30-60 seconds',
            'expected_cost': '$0.010-0.025'
        }

        return configs

    def display_results(self, categorized_models: Dict[str, List[str]],
                       configs: Dict[str, Dict]):
        """Display comprehensive results"""
        print("\n📊 MODEL DISCOVERY RESULTS")
        print("=" * 60)

        # Summary statistics
        total_working = len(self.working_models)
        total_failed = len(self.failed_models)
        success_rate = (total_working / (total_working + total_failed)) * 100

        print(f"\n📈 SUMMARY:")
        print(f"  • Working Models: {total_working}")
        print(f"  • Failed Models: {total_failed}")
        print(f"  • Success Rate: {success_rate:.1f}%")

        # Model categories
        print(f"\n🏆 EXCELLENT MODELS ({len(categorized_models['excellent'])}):")
        for model in categorized_models['excellent']:
            metrics = self.performance_metrics[model]
            print(f"  • {model}")
            print(f"    └─ Score: {metrics['score']:.1f}/5.0, Time: {metrics['time']:.1f}s, Cost: ${metrics['cost']:.4f}")

        print(f"\n✅ GOOD MODELS ({len(categorized_models['good'])}):")
        for model in categorized_models['good']:
            metrics = self.performance_metrics[model]
            print(f"  • {model}")
            print(f"    └─ Score: {metrics['score']:.1f}/5.0, Time: {metrics['time']:.1f}s, Cost: ${metrics['cost']:.4f}")

        print(f"\n⚠️  USABLE MODELS ({len(categorized_models['usable'])}):")
        for model in categorized_models['usable']:
            metrics = self.performance_metrics[model]
            print(f"  • {model}")
            print(f"    └─ Score: {metrics['score']:.1f}/5.0, Time: {metrics['time']:.1f}s, Cost: ${metrics['cost']:.4f}")

        # Optimal configurations
        print(f"\n🎯 OPTIMAL CONFIGURATIONS:")
        for config_name, config in configs.items():
            print(f"\n  {config_name.upper()}:")
            print(f"    Models: {', '.join(config['models'][:3])}")
            print(f"    Use Case: {config['use_case']}")
            print(f"    Expected Time: {config['expected_time']}")
            print(f"    Expected Cost: {config['expected_cost']}")

    def save_configurations(self, configs: Dict[str, Dict]):
        """Save configurations to files for easy use"""
        timestamp = int(time.time())

        # Save model configurations
        config_file = f"optimal_model_configs_{timestamp}.json"
        with open(config_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'discovery_date': time.strftime('%Y-%m-%d %H:%M:%S'),
                'working_models': self.working_models,
                'failed_models': self.failed_models,
                'performance_metrics': self.performance_metrics,
                'configurations': configs
            }, f, indent=2)

        print(f"\n💾 Configurations saved to: {config_file}")

        # Generate command examples
        examples_file = f"model_usage_examples_{timestamp}.txt"
        with open(examples_file, 'w') as f:
            f.write("QualAgent Optimal Model Usage Examples\n")
            f.write("=" * 50 + "\n\n")

            for config_name, config in configs.items():
                models_str = ','.join(config['models'][:3])
                f.write(f"# {config_name.upper()} CONFIGURATION\n")
                f.write(f"# {config['description']}\n")
                f.write(f"python run_enhanced_analysis.py --user-id analyst1 --company MSFT --analysis-type expert_guided --models {models_str}\n\n")

        print(f"💾 Usage examples saved to: {examples_file}")

        # Generate working models list for quick copy-paste
        working_models_file = f"working_models_{timestamp}.txt"
        with open(working_models_file, 'w') as f:
            f.write("# Working LLM Models (copy-paste ready)\n\n")
            f.write("EXCELLENT_MODELS = [\n")
            for model in [m for m in self.working_models if self.performance_metrics[m]['score'] >= 4.0]:
                f.write(f"    '{model}',\n")
            f.write("]\n\n")
            f.write("ALL_WORKING_MODELS = [\n")
            for model in self.working_models:
                f.write(f"    '{model}',\n")
            f.write("]\n")

        print(f"💾 Working models list saved to: {working_models_file}")

def main():
    """Main discovery function"""
    print("🚀 QUALAGENT AUTOMATIC MODEL DISCOVERY")
    print("=" * 60)

    filter = AutoModelFilter()

    # Discover working models
    categorized_models = filter.discover_working_models()

    # Generate optimal configurations
    configs = filter.generate_optimal_configurations(categorized_models)

    # Display results
    filter.display_results(categorized_models, configs)

    # Save configurations
    filter.save_configurations(configs)

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if len(filter.working_models) >= 3:
        print(f"  ✅ Excellent model availability - use 'balanced' configuration")
        recommended_models = ','.join(configs['balanced']['models'][:3])
        print(f"  📋 Recommended command:")
        print(f"      python run_enhanced_analysis.py --user-id analyst1 --company MSFT --analysis-type expert_guided --models {recommended_models}")
    elif len(filter.working_models) >= 2:
        print(f"  ⚠️  Limited models - use all available working models")
        recommended_models = ','.join(filter.working_models[:3])
        print(f"  📋 Recommended command:")
        print(f"      python run_enhanced_analysis.py --user-id analyst1 --company MSFT --analysis-type expert_guided --models {recommended_models}")
    else:
        print(f"  ❌ Insufficient working models - check API connectivity")

if __name__ == "__main__":
    main()