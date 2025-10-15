#!/usr/bin/env python3
"""
Focused Kimi K2 / Moonshot Testing Script
Specifically tests Kimi K2 variants on TogetherAI and other possible endpoints
"""

import os
import sys
import time
import json
import requests
from pathlib import Path
from typing import Dict, List

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

class KimiK2Tester:
    """Focused testing for Kimi K2 / Moonshot models"""

    def __init__(self):
        self.api_key = os.getenv('TOGETHER_API_KEY')
        self.test_results = {}

        # Comprehensive list of potential Kimi K2 model names
        self.kimi_variants = [
            # Most likely Moonshot/Kimi K2 names on TogetherAI
            "moonshot-ai/moonshot-v1-8k",
            "moonshot-ai/moonshot-v1-32k",
            "moonshot-ai/moonshot-v1-128k",
            "moonshot-ai/kimi-k2-8k",
            "moonshot-ai/kimi-k2-32k",
            "moonshot-ai/kimi-k2-128k",

            # Alternative naming patterns
            "moonshot/moonshot-v1-8k",
            "moonshot/moonshot-v1-32k",
            "moonshot/moonshot-v1-128k",
            "kimi/k2-8k",
            "kimi/k2-32k",
            "kimi/k2-128k",
            "Moonshot/moonshot-v1-8k",
            "Moonshot/moonshot-v1-32k",
            "Moonshot/moonshot-v1-128k",

            # Chinese naming variations
            "月之暗面/moonshot-v1-8k",
            "月之暗面/moonshot-v1-32k",
            "月之暗面/moonshot-v1-128k",
            "moonshot-ai/月之暗面-v1-8k",
            "moonshot-ai/月之暗面-v1-32k",

            # Other possible patterns
            "together/moonshot-v1-8k",
            "together/moonshot-v1-32k",
            "togethercomputer/moonshot-v1-8k",
            "togethercomputer/moonshot-v1-32k"
        ]

    def test_all_kimi_variants(self) -> Dict[str, Dict]:
        """Test all possible Kimi K2 model variants"""
        print("🌟 COMPREHENSIVE KIMI K2 / MOONSHOT TESTING")
        print("=" * 60)

        if not self.api_key:
            print("❌ TOGETHER_API_KEY not found in environment")
            return {}

        # Test simple prompt for financial analysis
        test_prompt = {
            "messages": [
                {"role": "system", "content": "You are a financial analyst. Respond in JSON format."},
                {"role": "user", "content": """Analyze Microsoft briefly:
{
  "company": "Microsoft",
  "score": 4,
  "confidence": 0.8,
  "reasoning": "brief analysis"
}"""}
            ],
            "max_tokens": 200,
            "temperature": 0.1
        }

        successful_models = []
        failed_models = []

        for i, model in enumerate(self.kimi_variants, 1):
            print(f"\n[{i}/{len(self.kimi_variants)}] 🔍 Testing: {model}")

            try:
                start_time = time.time()

                response = requests.post(
                    "https://api.together.xyz/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        **test_prompt,
                        "model": model
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

                        parsed = json.loads(json_content)
                        json_parseable = True
                    except:
                        json_parseable = False

                    result = {
                        'success': True,
                        'time': processing_time,
                        'tokens': tokens_used,
                        'json_parseable': json_parseable,
                        'content_preview': content[:150] + "..." if len(content) > 150 else content,
                        'status_code': response.status_code
                    }

                    successful_models.append(model)
                    print(f"  ✅ SUCCESS! Time: {processing_time:.2f}s, Tokens: {tokens_used}, JSON: {json_parseable}")

                else:
                    result = {
                        'success': False,
                        'error': f"HTTP {response.status_code}: {response.text[:100]}",
                        'time': processing_time,
                        'status_code': response.status_code
                    }
                    failed_models.append(model)
                    print(f"  ❌ FAILED: HTTP {response.status_code}")

            except requests.exceptions.Timeout:
                result = {
                    'success': False,
                    'error': 'Request timeout (30s)',
                    'time': 30.0,
                    'status_code': None
                }
                failed_models.append(model)
                print(f"  ⏰ TIMEOUT")

            except Exception as e:
                result = {
                    'success': False,
                    'error': str(e),
                    'time': time.time() - start_time,
                    'status_code': None
                }
                failed_models.append(model)
                print(f"  ❌ ERROR: {str(e)[:50]}")

            self.test_results[model] = result

        return {
            'successful': successful_models,
            'failed': failed_models,
            'results': self.test_results
        }

    def display_results(self, results: Dict):
        """Display comprehensive results"""
        successful = results['successful']
        failed = results['failed']

        print(f"\n📊 KIMI K2 TESTING RESULTS")
        print("=" * 60)

        print(f"\n📈 SUMMARY:")
        print(f"  • Total variants tested: {len(self.kimi_variants)}")
        print(f"  • Successful models: {len(successful)}")
        print(f"  • Failed models: {len(failed)}")
        print(f"  • Success rate: {len(successful)/len(self.kimi_variants)*100:.1f}%")

        if successful:
            print(f"\n🎉 WORKING KIMI K2 MODELS:")
            for model in successful:
                result = self.test_results[model]
                print(f"  ✅ {model}")
                print(f"    └─ Time: {result['time']:.2f}s, Tokens: {result.get('tokens', 'N/A')}, JSON: {result['json_parseable']}")
                print(f"    └─ Preview: {result['content_preview'][:100]}...")

            print(f"\n💡 RECOMMENDED USAGE:")
            best_model = successful[0]  # First working model
            print(f"  Add to LLM configuration:")
            print(f"  '{best_model}': LLMConfig(")
            print(f"      provider='together',")
            print(f"      model_name='{best_model}',")
            print(f"      cost_per_1k_tokens=0.0015  # Estimate")
            print(f"  )")

            print(f"\n🚀 TEST COMMAND:")
            print(f"python run_enhanced_analysis.py --user-id analyst1 --company MSFT --analysis-type expert_guided --models \"{best_model}\"")

        else:
            print(f"\n❌ NO WORKING KIMI K2 MODELS FOUND")
            print(f"   Possible reasons:")
            print(f"   • Kimi K2 not available on TogetherAI platform")
            print(f"   • Different model naming convention")
            print(f"   • Requires different API endpoint")
            print(f"   • Access restrictions or special permissions needed")

        print(f"\n🔍 SAMPLE FAILED ERRORS:")
        for model in failed[:3]:  # Show first 3 failures
            result = self.test_results[model]
            print(f"  • {model}: {result['error']}")

    def save_results(self, results: Dict):
        """Save results for analysis"""
        timestamp = int(time.time())
        filename = f"kimi_k2_test_results_{timestamp}.json"

        save_data = {
            'timestamp': timestamp,
            'test_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'total_tested': len(self.kimi_variants),
                'successful': len(results['successful']),
                'failed': len(results['failed']),
                'success_rate': len(results['successful'])/len(self.kimi_variants)*100
            },
            'successful_models': results['successful'],
            'failed_models': results['failed'],
            'detailed_results': self.test_results
        }

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Results saved to: {filename}")

def main():
    """Main testing function"""
    print("🚀 KIMI K2 / MOONSHOT MODEL DISCOVERY")
    print("=" * 60)
    print("This script tests various possible naming patterns for Kimi K2 on TogetherAI")

    tester = KimiK2Tester()

    # Run comprehensive tests
    results = tester.test_all_kimi_variants()

    # Display results
    tester.display_results(results)

    # Save results
    tester.save_results(results)

    # Additional recommendations
    if results['successful']:
        print(f"\n🎯 NEXT STEPS:")
        print(f"1. Add working Kimi K2 models to your LLM configuration")
        print(f"2. Test with actual financial analysis workflow")
        print(f"3. Compare performance with existing models")
        print(f"4. Integrate into auto_model_filter.py for automatic discovery")
    else:
        print(f"\n🔍 TROUBLESHOOTING:")
        print(f"1. Check if Kimi K2 is available on TogetherAI platform")
        print(f"2. Try alternative API endpoints (OpenAI, Anthropic, etc.)")
        print(f"3. Check model documentation for correct naming")
        print(f"4. Verify API key permissions")

if __name__ == "__main__":
    main()