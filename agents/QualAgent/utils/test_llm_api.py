#!/usr/bin/env python3
"""
TogetherAI API Testing Script
Comprehensive testing of LLM model accessibility and response parsing
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import requests

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from engines.llm_integration import LLMIntegration

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMAPITester:
    """Comprehensive LLM API testing and diagnostics"""

    def __init__(self):
        """Initialize the tester"""
        self.llm_integration = LLMIntegration()
        self.test_results = {}

        # Test prompt for consistent testing
        self.test_prompt = """
Please provide a brief financial analysis in JSON format:
{
  "company_name": "Test Company",
  "score": 4,
  "confidence": 0.8,
  "justification": "Brief test analysis"
}
"""

    def test_all_models(self) -> Dict[str, Dict]:
        """Test all configured LLM models"""
        print("🔬 TESTING CONFIGURED LLM MODELS")
        print("=" * 50)

        models = self.llm_integration.get_available_models()
        print(f"Found {len(models)} configured models: {models}")

        for model in models:
            print(f"\n📊 Testing {model}...")
            result = self.test_single_model(model)
            self.test_results[model] = result

            # Display immediate results
            if result['success']:
                print(f"✅ {model}: SUCCESS (cost: ${result['cost']:.4f}, time: {result['time']:.2f}s)")
            else:
                print(f"❌ {model}: FAILED - {result['error']}")

        return self.test_results

    def test_comprehensive_model_discovery(self) -> Dict[str, Dict]:
        """Test comprehensive model list for discovery"""
        print("\n🌐 COMPREHENSIVE MODEL DISCOVERY (RAW API)")
        print("=" * 60)

        api_key = os.getenv('TOGETHER_API_KEY')
        if not api_key:
            print("❌ TOGETHER_API_KEY not found in environment")
            return {}

        # Comprehensive model list for testing
        test_models = [
            # Current configured models
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "meta-llama/Llama-3-70b-chat-hf",
            "Qwen/Qwen2-72B-Instruct",
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "deepseek-ai/deepseek-coder-33b-instruct",

            # Additional popular models to test
            "meta-llama/Llama-3-8b-chat-hf",
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "google/gemma-2b-it",
            "google/gemma-7b-it",
            "Qwen/Qwen2-7B-Instruct",
            "Qwen/Qwen2-1.5B-Instruct",
            "deepseek-ai/deepseek-coder-6.7b-instruct",
            "codellama/CodeLlama-7b-Instruct-hf",
            "codellama/CodeLlama-13b-Instruct-hf",
            "codellama/CodeLlama-34b-Instruct-hf",
            "togethercomputer/RedPajama-INCITE-7B-Chat",
            "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
            "NousResearch/Nous-Hermes-2-Yi-34B",
            "teknium/OpenHermes-2.5-Mistral-7B",
            "mistralai/Mistral-7B-Instruct-v0.1",
            "upstage/SOLAR-10.7B-Instruct-v1.0",
            "garage-bAInd/Platypus2-70B-instruct",
            "openchat/openchat-3.5-1210",
            "Open-Orca/Mistral-7B-OpenOrca",
            "HuggingFaceH4/zephyr-7b-beta",

            # Chinese/International models to test
            "moonshot-ai/moonshot-v1-8k",  # Kimi K2 variant
            "moonshot-ai/moonshot-v1-32k", # Kimi K2 variant with longer context
            "moonshot-ai/moonshot-v1-128k", # Kimi K2 variant with very long context
            "THUDM/chatglm3-6b",
            "THUDM/chatglm2-6b",
            "baichuan-inc/Baichuan2-7B-Chat",
            "baichuan-inc/Baichuan2-13B-Chat",
            "01-ai/Yi-34B-Chat",
            "01-ai/Yi-6B-Chat",
            "alibaba/Qwen-7B-Chat",
            "alibaba/Qwen-14B-Chat",
            "internlm/internlm-chat-7b",
            "internlm/internlm-chat-20b",

            # Additional model variations to try
            "Qwen/Qwen1.5-7B-Chat",
            "Qwen/Qwen1.5-14B-Chat",
            "Qwen/Qwen1.5-72B-Chat",
            "microsoft/DialoGPT-large",
            "microsoft/DialoGPT-small",
            "facebook/blenderbot-400M-distill",
            "EleutherAI/gpt-j-6b",
            "EleutherAI/gpt-neox-20b",
            "bigscience/bloom-7b1",
            "bigscience/bloomz-7b1"
        ]

        discovery_results = {}
        print(f"Testing {len(test_models)} potential models...")

        for i, model in enumerate(test_models, 1):
            print(f"\n[{i}/{len(test_models)}] 🔍 Testing: {model}")

            try:
                start_time = time.time()
                response = requests.post(
                    "https://api.together.xyz/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": "You are a financial analyst. Respond in JSON format."},
                            {"role": "user", "content": self.test_prompt}
                        ],
                        "max_tokens": 100,
                        "temperature": 0.1
                    },
                    timeout=30
                )

                processing_time = time.time() - start_time

                if response.status_code == 200:
                    data = response.json()
                    content = data['choices'][0]['message']['content']
                    tokens_used = data.get('usage', {}).get('total_tokens', 0)

                    # Test JSON parsing
                    try:
                        if content.strip().startswith('```json'):
                            json_content = content.split('```json')[1].split('```')[0].strip()
                        elif content.strip().startswith('```'):
                            json_content = content.split('```')[1].strip()
                        else:
                            json_content = content.strip()

                        parsed_json = json.loads(json_content)
                        json_parseable = True
                    except:
                        json_parseable = False

                    discovery_results[model] = {
                        'success': True,
                        'time': processing_time,
                        'tokens': tokens_used,
                        'json_parseable': json_parseable,
                        'content_preview': content[:150] + "..." if len(content) > 150 else content,
                        'status_code': response.status_code
                    }

                    print(f"  ✅ SUCCESS! Time: {processing_time:.2f}s, Tokens: {tokens_used}, JSON: {json_parseable}")

                else:
                    discovery_results[model] = {
                        'success': False,
                        'error': f"HTTP {response.status_code}: {response.text[:100]}",
                        'time': processing_time,
                        'status_code': response.status_code
                    }
                    print(f"  ❌ FAILED: HTTP {response.status_code}")

            except requests.exceptions.Timeout:
                discovery_results[model] = {
                    'success': False,
                    'error': 'Request timeout (30s)',
                    'time': 30.0
                }
                print(f"  ⏰ TIMEOUT")

            except Exception as e:
                discovery_results[model] = {
                    'success': False,
                    'error': str(e),
                    'time': time.time() - start_time
                }
                print(f"  ❌ ERROR: {str(e)[:50]}")

        return discovery_results

    def test_single_model(self, model_name: str) -> Dict:
        """Test a single model with comprehensive diagnostics"""
        start_time = time.time()

        try:
            # Test basic API call
            response = self.llm_integration.call_llm(
                model_key=model_name,
                messages=[
                    {"role": "system", "content": "You are a financial analyst. Respond in JSON format."},
                    {"role": "user", "content": self.test_prompt}
                ]
            )

            processing_time = time.time() - start_time

            if response.error:
                return {
                    'success': False,
                    'error': response.error,
                    'time': processing_time,
                    'cost': 0.0
                }

            # Test JSON parsing
            try:
                if response.content.strip().startswith('```json'):
                    json_content = response.content.split('```json')[1].split('```')[0].strip()
                elif response.content.strip().startswith('```'):
                    json_content = response.content.split('```')[1].strip()
                else:
                    json_content = response.content.strip()

                parsed_json = json.loads(json_content)
                json_parseable = True

            except Exception as json_error:
                json_parseable = False
                parsed_json = None

            return {
                'success': True,
                'time': processing_time,
                'cost': response.cost_usd or 0.0,
                'tokens_used': response.tokens_used or 0,
                'content_length': len(response.content),
                'json_parseable': json_parseable,
                'response_preview': response.content[:200] + "..." if len(response.content) > 200 else response.content,
                'error': None
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'time': time.time() - start_time,
                'cost': 0.0
            }

    def test_model_endpoints(self) -> Dict[str, bool]:
        """Test raw TogetherAI endpoints directly"""
        print("\n🌐 TESTING RAW API ENDPOINTS")
        print("=" * 50)

        api_key = os.getenv('TOGETHER_API_KEY')
        if not api_key:
            print("❌ TOGETHER_API_KEY not found in environment")
            return {}

        # Comprehensive model list for testing
        test_models = [
            # Current configured models
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "meta-llama/Llama-3-70b-chat-hf",
            "Qwen/Qwen2-72B-Instruct",
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "deepseek-ai/deepseek-coder-33b-instruct",

            # Additional popular models to test
            "meta-llama/Llama-3-8b-chat-hf",
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
            "mistralai/Mistral-7B-Instruct-v0.3",
            "google/gemma-2b-it",
            "google/gemma-7b-it",
            "Qwen/Qwen2-7B-Instruct",
            "Qwen/Qwen2-1.5B-Instruct",
            "deepseek-ai/deepseek-coder-6.7b-instruct",
            "codellama/CodeLlama-7b-Instruct-hf",
            "codellama/CodeLlama-13b-Instruct-hf",
            "codellama/CodeLlama-34b-Instruct-hf",
            "togethercomputer/RedPajama-INCITE-7B-Chat",
            "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
            "NousResearch/Nous-Hermes-2-Yi-34B",
            "teknium/OpenHermes-2.5-Mistral-7B",
            "mistralai/Mistral-7B-Instruct-v0.1",
            "upstage/SOLAR-10.7B-Instruct-v1.0",
            "garage-bAInd/Platypus2-70B-instruct",
            "openchat/openchat-3.5-1210",
            "Open-Orca/Mistral-7B-OpenOrca",
            "HuggingFaceH4/zephyr-7b-beta",

            # Chinese/International models to test
            "moonshot-ai/moonshot-v1-8k",  # Kimi K2 variant
            "moonshot-ai/moonshot-v1-32k", # Kimi K2 variant with longer context
            "moonshot-ai/moonshot-v1-128k", # Kimi K2 variant with very long context
            "THUDM/chatglm3-6b",
            "THUDM/chatglm2-6b",
            "baichuan-inc/Baichuan2-7B-Chat",
            "baichuan-inc/Baichuan2-13B-Chat",
            "01-ai/Yi-34B-Chat",
            "01-ai/Yi-6B-Chat",
            "alibaba/Qwen-7B-Chat",
            "alibaba/Qwen-14B-Chat",
            "internlm/internlm-chat-7b",
            "internlm/internlm-chat-20b",

            # Additional model variations to try
            "Qwen/Qwen1.5-7B-Chat",
            "Qwen/Qwen1.5-14B-Chat",
            "Qwen/Qwen1.5-72B-Chat",
            "microsoft/DialoGPT-large",
            "microsoft/DialoGPT-small",
            "facebook/blenderbot-400M-distill",
            "EleutherAI/gpt-j-6b",
            "EleutherAI/gpt-neox-20b",
            "bigscience/bloom-7b1",
            "bigscience/bloomz-7b1"
        ]

        endpoint_results = {}

        for model in test_models:
            print(f"\n🔗 Testing endpoint: {model}")

            try:
                response = requests.post(
                    "https://api.together.xyz/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": "Hello, please respond with just 'OK'"}
                        ],
                        "max_tokens": 10,
                        "temperature": 0.1
                    },
                    timeout=30
                )

                if response.status_code == 200:
                    endpoint_results[model] = True
                    print(f"✅ {model}: ACCESSIBLE")
                else:
                    endpoint_results[model] = False
                    print(f"❌ {model}: HTTP {response.status_code} - {response.text[:100]}")

            except Exception as e:
                endpoint_results[model] = False
                print(f"❌ {model}: ERROR - {str(e)}")

        return endpoint_results

    def generate_report(self):
        """Generate comprehensive diagnostic report"""
        print("\n📋 COMPREHENSIVE DIAGNOSTIC REPORT")
        print("=" * 60)

        working_models = [m for m, r in self.test_results.items() if r['success']]
        failed_models = [m for m, r in self.test_results.items() if not r['success']]

        # Categorize working models by type
        chinese_models = [m for m in working_models if any(keyword in m.lower() for keyword in ['moonshot', 'kimi', 'chatglm', 'baichuan', 'yi-', 'qwen', 'internlm', 'thudm', 'alibaba'])]
        western_models = [m for m in working_models if m not in chinese_models]

        print(f"\n🌟 KIMI K2 / MOONSHOT MODELS:")
        kimi_models = [m for m in working_models if 'moonshot' in m.lower()]
        if kimi_models:
            for model in kimi_models:
                result = self.test_results[model]
                print(f"  ✅ {model}")
                print(f"    - Response time: {result['time']:.2f}s")
                if 'cost' in result:
                    print(f"    - Cost: ${result['cost']:.4f}")
                else:
                    print(f"    - Cost: N/A (raw API test)")
                print(f"    - JSON parseable: {result['json_parseable']}")
        else:
            print("  ❌ No Kimi K2/Moonshot models accessible")

        print(f"\n🇨🇳 CHINESE MODELS WORKING ({len(chinese_models)}):")
        for model in chinese_models:
            if 'moonshot' not in model.lower():  # Already shown above
                result = self.test_results[model]
                print(f"  • {model}")
                print(f"    - Response time: {result['time']:.2f}s")
                if 'cost' in result:
                    print(f"    - Cost: ${result['cost']:.4f}")
                else:
                    print(f"    - Cost: N/A (raw API test)")
                print(f"    - JSON parseable: {result['json_parseable']}")

        print(f"\n🌍 WESTERN MODELS WORKING ({len(western_models)}):")
        for model in western_models:
            result = self.test_results[model]
            print(f"  • {model}")
            print(f"    - Response time: {result['time']:.2f}s")
            if 'cost' in result:
                print(f"    - Cost: ${result['cost']:.4f}")
            else:
                print(f"    - Cost: N/A (raw API test)")
            print(f"    - JSON parseable: {result['json_parseable']}")

        print(f"\n❌ FAILED MODELS ({len(failed_models)}):")
        # Group failed models too
        failed_chinese = [m for m in failed_models if any(keyword in m.lower() for keyword in ['moonshot', 'kimi', 'chatglm', 'baichuan', 'yi-', 'qwen', 'internlm', 'thudm', 'alibaba'])]
        failed_western = [m for m in failed_models if m not in failed_chinese]

        if failed_chinese:
            print(f"  🇨🇳 Chinese models failed:")
            for model in failed_chinese:
                result = self.test_results[model]
                print(f"    • {model}: {result['error']}")

        if failed_western:
            print(f"  🌍 Western models failed:")
            for model in failed_western:
                result = self.test_results[model]
                print(f"    • {model}: {result['error']}")

        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        if len(working_models) >= 2:
            print(f"  ✅ System operational with {len(working_models)} working models")
            print(f"  📊 Recommended models for analysis: {', '.join(working_models[:3])}")
        else:
            print(f"  ⚠️  Only {len(working_models)} working model(s) - consensus quality reduced")

        if failed_models:
            print(f"  🔧 Failed models may need:")
            print(f"     - API key validation")
            print(f"     - Model name updates")
            print(f"     - Rate limit adjustments")

        # Cost analysis
        total_cost = sum(r.get('cost', 0) for r in self.test_results.values() if r['success'])
        avg_time = sum(r['time'] for r in self.test_results.values() if r['success']) / len(working_models) if working_models else 0

        print(f"\n💰 COST ANALYSIS:")
        print(f"  • Test cost: ${total_cost:.4f}")
        print(f"  • Average response time: {avg_time:.2f}s")
        print(f"  • Estimated full analysis cost: ${total_cost * 50:.4f}")  # Rough estimate

def main():
    """Main testing function"""
    print("🚀 QUALAGENT LLM API COMPREHENSIVE TESTING")
    print("=" * 60)

    tester = LLMAPITester()

    # Test 1: Configured models through integration layer
    print("Phase 1: Testing configured models...")
    configured_results = tester.test_all_models()

    # Test 2: Comprehensive model discovery
    print("\nPhase 2: Comprehensive model discovery...")
    discovery_results = tester.test_comprehensive_model_discovery()

    # Test 3: Generate comprehensive report
    print("\nPhase 3: Generating comprehensive report...")

    # Merge results for reporting
    tester.test_results.update(discovery_results)
    tester.generate_report()

    # Save results
    results_file = f"llm_test_results_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'configured_tests': configured_results,
            'discovery_tests': discovery_results,
            'timestamp': time.time(),
            'summary': {
                'configured_working': [m for m, r in configured_results.items() if r['success']],
                'configured_failed': [m for m, r in configured_results.items() if not r['success']],
                'discovery_working': [m for m, r in discovery_results.items() if r['success']],
                'discovery_failed': [m for m, r in discovery_results.items() if not r['success']],
                'total_tested': len(configured_results) + len(discovery_results),
                'total_working': len([m for m, r in {**configured_results, **discovery_results}.items() if r['success']])
            }
        }, f, indent=2)

    print(f"\n📁 Detailed results saved to: {results_file}")

    # Summary
    all_working = [m for m, r in {**configured_results, **discovery_results}.items() if r['success']]
    kimi_working = [m for m in all_working if 'moonshot' in m.lower()]

    print(f"\n🎯 FINAL SUMMARY:")
    print(f"  • Total models tested: {len(configured_results) + len(discovery_results)}")
    print(f"  • Working models found: {len(all_working)}")
    print(f"  • Kimi K2 models working: {len(kimi_working)}")

    if kimi_working:
        print(f"  🌟 Kimi K2 SUCCESS: {', '.join(kimi_working)}")
    else:
        print(f"  ❌ No Kimi K2 models found working")

    if len(all_working) > 3:
        print(f"\n💡 RECOMMENDED NEXT STEP:")
        print(f"  Run auto-discovery to find optimal combinations:")
        print(f"  python utils/auto_model_filter.py")

if __name__ == "__main__":
    main()