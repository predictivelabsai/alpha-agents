"""
LLM Integration Engine for QualAgent
Provides unified interface for multiple LLM providers with emphasis on TogetherAI
"""

import os
import json
import time
import logging
import requests
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import openai
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    # Try alternative paths
    load_dotenv()  # Load from current directory or system environment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """Configuration for LLM models"""
    provider: str
    model_name: str
    max_tokens: int
    temperature: float
    api_endpoint: str
    supports_system_prompt: bool = True
    cost_per_1k_tokens: float = 0.0
    context_window: int = 4096

@dataclass
class LLMResponse:
    """Standardized LLM response"""
    content: str
    model_used: str
    provider: str
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None
    processing_time_seconds: Optional[float] = None
    error: Optional[str] = None
    raw_response: Optional[Dict] = None

class LLMIntegration:
    """Unified LLM integration for multiple providers"""

    def __init__(self):
        # Load API keys from environment
        self.together_api_key = os.getenv('TOGETHER_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')

        # Validate API keys
        if not self.together_api_key:
            logger.warning("TOGETHER_API_KEY not found in environment variables")

        # Initialize model configurations
        self.models = self._initialize_models()

        # Initialize clients
        self.together_client = self._init_together_client()
        self.openai_client = self._init_openai_client()

    def _initialize_models(self) -> Dict[str, LLMConfig]:
        """Initialize available model configurations"""
        models = {}

        # TogetherAI models (preferred for cost and variety)
        if self.together_api_key:
            models.update({
                'llama-3-70b': LLMConfig(
                    provider='together',
                    model_name='meta-llama/Llama-3-70b-chat-hf',
                    max_tokens=4000,
                    temperature=0.1,
                    api_endpoint='https://api.together.xyz/v1/chat/completions',
                    cost_per_1k_tokens=0.0009,
                    context_window=8192
                ),
                'mixtral-8x7b': LLMConfig(
                    provider='together',
                    model_name='mistralai/Mixtral-8x7B-Instruct-v0.1',
                    max_tokens=4000,
                    temperature=0.1,
                    api_endpoint='https://api.together.xyz/v1/chat/completions',
                    cost_per_1k_tokens=0.0006,
                    context_window=32768
                ),
                'qwen2-72b': LLMConfig(
                    provider='together',
                    model_name='Qwen/Qwen2-72B-Instruct',
                    max_tokens=4000,
                    temperature=0.1,
                    api_endpoint='https://api.together.xyz/v1/chat/completions',
                    cost_per_1k_tokens=0.0009,
                    context_window=32768
                ),
                'llama-3.1-70b': LLMConfig(
                    provider='together',
                    model_name='meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
                    max_tokens=4000,
                    temperature=0.1,
                    api_endpoint='https://api.together.xyz/v1/chat/completions',
                    cost_per_1k_tokens=0.0009,
                    context_window=131072
                ),
                'deepseek-coder-33b': LLMConfig(
                    provider='together',
                    model_name='deepseek-ai/deepseek-coder-33b-instruct',
                    max_tokens=4000,
                    temperature=0.1,
                    api_endpoint='https://api.together.xyz/v1/chat/completions',
                    cost_per_1k_tokens=0.0008,
                    context_window=16384
                )
            })

        # OpenAI models (backup)
        if self.openai_api_key:
            models.update({
                'gpt-4o': LLMConfig(
                    provider='openai',
                    model_name='gpt-4o',
                    max_tokens=4000,
                    temperature=0.1,
                    api_endpoint='https://api.openai.com/v1/chat/completions',
                    cost_per_1k_tokens=0.005,
                    context_window=128000
                ),
                'gpt-4o-mini': LLMConfig(
                    provider='openai',
                    model_name='gpt-4o-mini',
                    max_tokens=4000,
                    temperature=0.1,
                    api_endpoint='https://api.openai.com/v1/chat/completions',
                    cost_per_1k_tokens=0.00015,
                    context_window=128000
                )
            })

        logger.info(f"Initialized {len(models)} LLM models")
        return models

    def _init_together_client(self):
        """Initialize TogetherAI client"""
        if not self.together_api_key:
            return None

        session = requests.Session()
        session.headers.update({
            'Authorization': f'Bearer {self.together_api_key}',
            'Content-Type': 'application/json'
        })
        logger.info("TogetherAI client initialized")
        return session

    def _init_openai_client(self):
        """Initialize OpenAI client"""
        if not self.openai_api_key:
            return None

        try:
            client = openai.OpenAI(api_key=self.openai_api_key)
            logger.info("OpenAI client initialized")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            return None

    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return list(self.models.keys())

    def get_model_info(self, model_key: str) -> Optional[LLMConfig]:
        """Get model configuration"""
        return self.models.get(model_key)

    def get_recommended_models(self) -> List[str]:
        """Get recommended models for qualitative analysis"""
        # Prioritize models good for reasoning and analysis
        recommended = []

        # TogetherAI models (preferred)
        if 'llama-3.1-70b' in self.models:
            recommended.append('llama-3.1-70b')  # Best context window
        if 'mixtral-8x7b' in self.models:
            recommended.append('mixtral-8x7b')   # Good reasoning, cost-effective
        if 'qwen2-72b' in self.models:
            recommended.append('qwen2-72b')      # Strong analytical capabilities

        # OpenAI fallback
        if 'gpt-4o' in self.models:
            recommended.append('gpt-4o')         # High quality but expensive

        return recommended

    def call_llm(self, model_key: str, messages: List[Dict],
                 system_prompt: str = None, max_retries: int = 3) -> LLMResponse:
        """Call LLM with unified interface"""

        if model_key not in self.models:
            return LLMResponse(
                content="",
                model_used=model_key,
                provider="unknown",
                error=f"Model {model_key} not available"
            )

        config = self.models[model_key]
        start_time = time.time()

        # Prepare messages
        formatted_messages = self._format_messages(messages, system_prompt, config)

        # Call appropriate provider
        for attempt in range(max_retries):
            try:
                if config.provider == 'together':
                    response = self._call_together(config, formatted_messages)
                elif config.provider == 'openai':
                    response = self._call_openai(config, formatted_messages)
                else:
                    raise ValueError(f"Unsupported provider: {config.provider}")

                # Calculate processing time and cost
                processing_time = time.time() - start_time
                cost = self._calculate_cost(response.tokens_used or 0, config)

                response.processing_time_seconds = processing_time
                response.cost_usd = cost

                logger.info(f"LLM call successful: {model_key} in {processing_time:.2f}s")
                return response

            except Exception as e:
                logger.warning(f"LLM call attempt {attempt + 1} failed for {model_key}: {e}")
                if attempt == max_retries - 1:
                    return LLMResponse(
                        content="",
                        model_used=model_key,
                        provider=config.provider,
                        error=str(e),
                        processing_time_seconds=time.time() - start_time
                    )
                time.sleep(2 ** attempt)  # Exponential backoff

    def _format_messages(self, messages: List[Dict], system_prompt: str,
                        config: LLMConfig) -> List[Dict]:
        """Format messages for different providers"""
        formatted = []

        # Add system prompt if supported
        if system_prompt and config.supports_system_prompt:
            formatted.append({"role": "system", "content": system_prompt})

        # Add user messages
        for msg in messages:
            if not system_prompt or config.supports_system_prompt:
                formatted.append(msg)
            else:
                # Prepend system prompt to first user message if system role not supported
                if msg["role"] == "user" and not formatted:
                    content = f"{system_prompt}\n\n{msg['content']}"
                    formatted.append({"role": "user", "content": content})
                else:
                    formatted.append(msg)

        return formatted

    def _call_together(self, config: LLMConfig, messages: List[Dict]) -> LLMResponse:
        """Call TogetherAI API"""
        if not self.together_client:
            raise ValueError("TogetherAI client not initialized")

        payload = {
            "model": config.model_name,
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "stream": False
        }

        response = self.together_client.post(config.api_endpoint, json=payload)
        response.raise_for_status()

        data = response.json()

        # Extract response content
        content = data['choices'][0]['message']['content']
        tokens_used = data.get('usage', {}).get('total_tokens')

        return LLMResponse(
            content=content,
            model_used=config.model_name,
            provider='together',
            tokens_used=tokens_used,
            raw_response=data
        )

    def _call_openai(self, config: LLMConfig, messages: List[Dict]) -> LLMResponse:
        """Call OpenAI API"""
        if not self.openai_client:
            raise ValueError("OpenAI client not initialized")

        response = self.openai_client.chat.completions.create(
            model=config.model_name,
            messages=messages,
            max_tokens=config.max_tokens,
            temperature=config.temperature
        )

        content = response.choices[0].message.content
        tokens_used = response.usage.total_tokens if response.usage else None

        return LLMResponse(
            content=content,
            model_used=config.model_name,
            provider='openai',
            tokens_used=tokens_used,
            raw_response=response.model_dump() if hasattr(response, 'model_dump') else None
        )

    def _calculate_cost(self, tokens_used: int, config: LLMConfig) -> float:
        """Calculate API call cost"""
        if tokens_used and config.cost_per_1k_tokens:
            return (tokens_used / 1000) * config.cost_per_1k_tokens
        return 0.0

    def batch_analyze(self, prompts: List[Tuple[str, List[Dict]]], model_keys: List[str],
                     system_prompt: str = None) -> Dict[str, List[LLMResponse]]:
        """Run batch analysis across multiple models and prompts"""
        results = {}

        for i, (prompt_name, messages) in enumerate(prompts):
            results[prompt_name] = []

            for model_key in model_keys:
                logger.info(f"Processing {prompt_name} with {model_key} ({i+1}/{len(prompts)})")

                response = self.call_llm(
                    model_key=model_key,
                    messages=messages,
                    system_prompt=system_prompt
                )

                results[prompt_name].append(response)

                # Brief pause between calls to respect rate limits
                time.sleep(1)

        return results

    def get_cost_estimate(self, model_key: str, estimated_tokens: int) -> float:
        """Get cost estimate for analysis"""
        config = self.models.get(model_key)
        if config and config.cost_per_1k_tokens:
            return (estimated_tokens / 1000) * config.cost_per_1k_tokens
        return 0.0

    def validate_api_keys(self) -> Dict[str, bool]:
        """Validate API keys by making test calls"""
        validation_results = {}

        # Test TogetherAI
        if self.together_api_key:
            try:
                test_response = self.call_llm(
                    model_key='mixtral-8x7b',
                    messages=[{"role": "user", "content": "Hello, respond with just 'OK'"}]
                )
                validation_results['together'] = test_response.error is None
            except:
                validation_results['together'] = False
        else:
            validation_results['together'] = False

        # Test OpenAI
        if self.openai_api_key:
            try:
                test_response = self.call_llm(
                    model_key='gpt-4o-mini',
                    messages=[{"role": "user", "content": "Hello, respond with just 'OK'"}]
                )
                validation_results['openai'] = test_response.error is None
            except:
                validation_results['openai'] = False
        else:
            validation_results['openai'] = False

        return validation_results

def main():
    """Test LLM integration"""
    llm = LLMIntegration()

    print("Available models:")
    for model in llm.get_available_models():
        info = llm.get_model_info(model)
        print(f"  {model}: {info.provider}/{info.model_name}")

    print("\nRecommended models:")
    for model in llm.get_recommended_models():
        print(f"  {model}")

    # Test API key validation
    print("\nAPI key validation:")
    validation = llm.validate_api_keys()
    for provider, valid in validation.items():
        status = "✓" if valid else "✗"
        print(f"  {provider}: {status}")

if __name__ == "__main__":
    main()