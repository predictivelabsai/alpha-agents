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

        # Handle dynamic model creation for raw model names
        if model_key not in self.models:
            dynamic_config = self._create_dynamic_model_config(model_key)
            if dynamic_config:
                # Temporarily add the dynamic config to models
                self.models[model_key] = dynamic_config
                logger.info(f"Created dynamic model config for: {model_key}")
            else:
                return LLMResponse(
                    content="",
                    model_used=model_key,
                    provider="unknown",
                    error=f"Model {model_key} not available and cannot create dynamic config"
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


    def get_model_cost(self, model_key: str) -> float:
        """Get cost per 1K tokens for a model"""
        config = self.models.get(model_key)
        if config:
            return config.cost_per_1k_tokens
        return 0.0

    def run_analysis(self, company, config):
        """Run analysis using the structured prompt and JSON format"""
        try:
            # Get model from config
            models = config.get('models_to_use', ['mixtral-8x7b'])
            model = models[0] if models else 'mixtral-8x7b'

            # Create structured analysis prompt
            prompt = self._build_structured_prompt(company)

            # Call LLM with structured prompt
            response = self.call_llm(
                model_key=model,
                messages=[
                    {"role": "system", "content": "You are a senior equity research analyst. Provide analysis in valid JSON format only."},
                    {"role": "user", "content": prompt}
                ]
            )

            return response

        except Exception as e:
            return LLMResponse(
                content="",
                model_used=model,
                provider="unknown",
                error=str(e)
            )

    def _build_structured_prompt(self, company):
        """Build enhanced structured prompt for comprehensive qualitative analysis with JSON output"""
        return f"""
You are a senior equity research analyst conducting a comprehensive qualitative assessment of {company.company_name} ({company.ticker}) in the {company.subsector} sector. Your analysis will directly influence investment decisions, so maintain the highest standards of analytical rigor and objectivity.

ANALYTICAL FRAMEWORK:
You must evaluate the company using the specific dimensions outlined below. Apply consistent methodology, provide evidence-based analysis, and generate actionable investment insights. Consider the company's Total Addressable Market (TAM), Sales & Go-to-Market strategy, Unit Economics & Business Model, Financial & Capital Efficiency, and ESG & Regulatory factors as contextual information to inform your scoring of the required dimensions.

KEY ANALYTICAL PRINCIPLES:
1. Provide evidence-based analysis with specific source citations when available
2. Apply consistent scoring methodology across all dimensions (1-5 scale)
3. Include quantitative confidence scores (0.0-1.0) for all assessments
4. Identify and address contradictory evidence when present
5. Generate actionable investment insights
6. Maintain analytical independence and objectivity
7. Consider both quantitative metrics and qualitative factors
8. Think step-by-step and provide detailed chain-of-thought reasoning

ENHANCED SCORING METHODOLOGY:
- 1 = Very Weak/Poor: Significant structural disadvantages, poor execution, high risk
- 2 = Weak/Below Average: Below market standards, execution challenges, elevated risk
- 3 = Neutral/Average: Market-standard performance, mixed evidence, moderate positioning
- 4 = Strong/Above Average: Clear competitive advantages, good execution, strong positioning
- 5 = Very Strong/Excellent: Dominant position, exceptional execution, sustainable advantages

Confidence Levels:
- 0.9-1.0: High confidence with strong supporting data and multiple confirmatory sources
- 0.7-0.8: Good confidence with solid evidence but some data limitations
- 0.5-0.6: Moderate confidence with mixed or limited evidence
- 0.3-0.4: Low confidence due to limited data or contradictory evidence
- 0.0-0.2: Very low confidence, largely speculative assessment

=== COMPETITIVE MOAT ANALYSIS RUBRICS ===

**Brand Monopoly (Customer Loyalty & Pricing Power):**
Consider the brand's recognition, customer loyalty, pricing power, and willingness of customers to pay premium prices.
- Score 5: Iconic brand with unmatched loyalty, significant pricing power, customers choose brand over price
- Score 4: Strong brand recognition, good customer loyalty, some pricing flexibility vs competitors
- Score 3: Recognized brand with average loyalty, limited pricing power, moderate differentiation
- Score 2: Weak brand differentiation, price-sensitive customers, commodity-like positioning
- Score 1: No brand advantage, pure price competition, easily substitutable products/services

**Barriers to Entry (Market Protection):**
Evaluate regulatory barriers, capital requirements, technical complexity, distribution access, economies of scale needed for entry.
- Score 5: Extremely high barriers (regulatory moats, massive capital requirements, complex technology, exclusive partnerships)
- Score 4: High barriers requiring significant investment, expertise, or regulatory approval to overcome
- Score 3: Moderate barriers that can be overcome with substantial effort, capital, and time
- Score 2: Low barriers with some challenges for new entrants but achievable with moderate resources
- Score 1: Minimal barriers, easy market entry for competitors with basic capabilities

**Economies of Scale (Cost Advantages):**
Assess cost advantages from size, fixed cost leverage, purchasing power, operational efficiency at scale.
- Score 5: Massive scale advantages creating insurmountable cost gaps vs competitors
- Score 4: Significant scale benefits providing clear cost leadership and pricing advantages
- Score 3: Some scale advantages but competitors can achieve similar economics with effort
- Score 2: Limited scale benefits, easily matched by competitors of similar size
- Score 1: No scale advantages or actual disadvantages due to small size relative to market

**Network Effects (Value from User Base):**
Evaluate how user growth increases value for all participants, data advantages, platform effects.
- Score 5: Strong direct/indirect network effects creating exponential value growth with user additions
- Score 4: Clear network effects that increase switching costs and platform value significantly
- Score 3: Some network benefits but not critical to core value proposition or competitive position
- Score 2: Weak network effects with limited impact on competitiveness or user retention
- Score 1: No network effects or potential negative network externalities

**Switching Costs (Customer Retention):**
Assess integration complexity, data lock-in, retraining costs, contractual obligations, relationship switching barriers.
- Score 5: Extremely high switching costs (deep integration, proprietary data, long contracts, high retraining costs)
- Score 4: High switching costs creating strong customer stickiness and predictable revenue
- Score 3: Moderate switching costs providing some customer retention advantage
- Score 2: Low switching costs, customers can move to competitors with minimal friction
- Score 1: No switching costs, commoditized offering with easy substitution

=== STRATEGIC INSIGHTS ANALYSIS RUBRICS ===

**Competitive Differentiation (Unique Value Proposition):**
Assess sustainable competitive advantages, unique capabilities, defensibility of market position. Consider proprietary technology, unique business model, exclusive partnerships, specialized expertise.
- Score 5: Unique, defensible advantages that competitors cannot easily replicate
- Score 4: Strong differentiation with multiple competitive advantages
- Score 3: Some differentiation but advantages can be copied over time
- Score 2: Weak differentiation, easily replicated by competitors
- Score 1: No meaningful differentiation, commodity-like offering

**Market Timing (Positioning for Market Trends):**
Evaluate positioning for current/emerging trends, adoption cycles, technological shifts. Consider market readiness, competitive positioning timing, secular tailwinds.
- Score 5: Perfectly positioned for major secular trends with early mover advantage
- Score 4: Well-positioned for market trends with good timing
- Score 3: Reasonably positioned but timing advantages are unclear
- Score 2: Poorly positioned for market trends, late to market
- Score 1: Positioned against market trends or in declining markets

**Management Quality (Leadership Effectiveness):**
Assess leadership track record, strategic vision, execution capability, capital allocation, stakeholder management. Consider past performance, strategic decisions, communication, governance.
- Score 5: Exceptional leadership with proven track record of value creation and strategic execution
- Score 4: Strong management team with good execution and strategic vision
- Score 3: Competent management with mixed track record
- Score 2: Weak management with execution challenges or poor capital allocation
- Score 1: Poor management with history of value destruction or strategic failures

**Technology Moats (Technical Barriers):**
Evaluate proprietary technology, IP strength, R&D capabilities, technological sustainability. Consider patents, technical complexity, innovation pipeline, technological competitive advantages.
- Score 5: Proprietary technology with strong IP protection and sustainable technical advantages
- Score 4: Solid technology moats with good IP and technical capabilities
- Score 3: Some technology advantages but not sustainable long-term
- Score 2: Weak technology moats, easily copied or disrupted
- Score 1: No technology advantages or vulnerable to technological disruption

**Transformation Potential (Business Model Evolution):**
Evaluate ability to adapt business model, enter adjacent markets, leverage core capabilities for new opportunities. Consider innovation culture, strategic flexibility, platform extensibility.
- Score 5: High transformation potential with proven ability to evolve and enter new markets
- Score 4: Good transformation capabilities with some demonstrated flexibility
- Score 3: Moderate transformation potential but execution uncertainty
- Score 2: Limited transformation potential, rigid business model
- Score 1: No transformation potential, trapped in declining business model

**Platform Expansion (Growth Opportunities):**
Assess opportunities to expand platform, enter adjacent markets, leverage existing capabilities, create new revenue streams. Consider total addressable market expansion, cross-selling opportunities, geographic expansion.
- Score 5: Significant platform expansion opportunities with clear path to execution
- Score 4: Good expansion opportunities with reasonable execution probability
- Score 3: Some expansion opportunities but execution challenges exist
- Score 2: Limited expansion opportunities or high execution risk
- Score 1: No meaningful expansion opportunities or fundamental constraints

GROWTH DRIVERS ANALYSIS:
Identify specific, actionable growth drivers with impact assessment. Consider market expansion, product innovation, operational improvements, strategic initiatives.

RISK FACTORS ANALYSIS:
Identify major risks with severity and probability assessment. Consider competitive threats, regulatory risks, technology disruption, market risks, execution risks.

RED FLAGS ANALYSIS:
Identify specific warning signals or concerning developments. Consider governance issues, financial irregularities, competitive losses, strategic missteps.

COMPETITOR ANALYSIS FRAMEWORK:
CRITICAL: You MUST identify REAL, ACTUAL competitors by name and ticker symbol. Do NOT use placeholder names like "Competitor A", "Competitor B", "Company X", etc.

Research and identify 3-5 specific, real companies that compete directly or indirectly with the analyzed company. For each competitor:
1. Use the ACTUAL company name and stock ticker (or "Private" if not public)
2. Provide specific market data, revenue figures, or market positioning information
3. Identify real competitive advantages and business model differences
4. Assess actual threat level based on market presence and capabilities

Examples of ACCEPTABLE competitor identification:
- "Microsoft Corporation" (MSFT) - with specific details about their competing products
- "Amazon Web Services" (AMZN) - with actual market share data
- "Salesforce Inc." (CRM) - with real competitive positioning information

Examples of UNACCEPTABLE placeholder responses:
- "Competitor A" or "Competitor B"
- "Company X" or "Major Player"
- Generic descriptions without real company names

If you cannot identify specific real competitors, state "Unable to identify specific competitors with available information" rather than using placeholders.

IMPORTANT: Return your response as valid JSON in the exact format specified below. Do not include any text outside the JSON structure.

Required JSON Structure:
{{
  "competitive_moat_analysis": {{
    "brand_monopoly": {{"score": 1-5, "confidence": 0.0-1.0, "justification": "detailed evidence-based explanation with specific examples and reasoning", "sources": []}},
    "barriers_to_entry": {{"score": 1-5, "confidence": 0.0-1.0, "justification": "detailed evidence-based explanation with specific examples and reasoning", "sources": []}},
    "economies_of_scale": {{"score": 1-5, "confidence": 0.0-1.0, "justification": "detailed evidence-based explanation with specific examples and reasoning", "sources": []}},
    "network_effects": {{"score": 1-5, "confidence": 0.0-1.0, "justification": "detailed evidence-based explanation with specific examples and reasoning", "sources": []}},
    "switching_costs": {{"score": 1-5, "confidence": 0.0-1.0, "justification": "detailed evidence-based explanation with specific examples and reasoning", "sources": []}}
  }},
  "strategic_insights": {{
    "key_growth_drivers": [
      {{"driver": "specific growth driver with actionable detail", "impact": "1-5", "timeframe": "short/medium/long", "confidence": 0.0-1.0}}
    ],
    "major_risk_factors": [
      {{"risk": "specific risk with detailed explanation", "severity": "1-5", "probability": 0.0-1.0, "mitigation": "possible mitigation strategies and management responses"}}
    ],
    "red_flags": [
      {{"flag": "specific red flag or warning signal with context", "severity": "1-5", "evidence": "supporting evidence, data, and reasoning for concern"}}
    ],
    "transformation_potential": {{"score": 1-5, "confidence": 0.0-1.0, "justification": "detailed assessment of business model evolution capabilities, innovation culture, and strategic flexibility"}},
    "platform_expansion": {{"score": 1-5, "confidence": 0.0-1.0, "justification": "detailed assessment of growth opportunities, market expansion potential, and capability leverage"}},
    "competitive_differentiation": {{"score": 1-5, "confidence": 0.0-1.0, "justification": "detailed assessment of unique value propositions, sustainable advantages, and competitive positioning"}},
    "market_timing": {{"score": 1-5, "confidence": 0.0-1.0, "justification": "detailed assessment of market positioning, trend alignment, and timing advantages"}},
    "management_quality": {{"score": 1-5, "confidence": 0.0-1.0, "justification": "detailed assessment of leadership effectiveness, track record, and strategic execution capability"}},
    "technology_moats": {{"score": 1-5, "confidence": 0.0-1.0, "justification": "detailed assessment of technological barriers, IP strength, and innovation capabilities"}}
  }},
  "competitor_analysis": [
    {{
      "name": "REAL Company Name (e.g., Microsoft Corporation, Apple Inc., Amazon.com Inc.)",
      "ticker": "REAL ticker symbol (e.g., MSFT, AAPL, AMZN) or Private",
      "competitive_position": "detailed description of actual market position, business model, and competitive strategy with specific data",
      "market_share": "actual market share percentage, revenue figures, or specific market ranking with sources",
      "key_differentiators": ["specific real competitive advantage 1", "specific real competitive advantage 2", "specific real competitive advantage 3"],
      "threat_level": 1-5,
      "strategic_response": "specific strategic responses based on real competitive dynamics and market positioning"
    }}
  ]
}}

ANALYSIS REQUIREMENTS:
1. Provide comprehensive, evidence-based justifications for all scores with specific examples
2. Include quantitative metrics, market data, and financial performance indicators where available
3. Address potential contradictions or conflicting evidence in your assessment
4. Consider both current performance and future prospects in your scoring
5. Maintain analytical objectivity and highlight areas of uncertainty or data limitations
6. Focus on sustainable competitive advantages and long-term business model strength
7. Consider industry dynamics, market trends, competitive landscape, and regulatory environment
8. Use step-by-step reasoning and provide detailed chain-of-thought analysis

CRITICAL COMPETITOR ANALYSIS REQUIREMENT:
9. MANDATORY: Identify REAL competitors by actual company names and ticker symbols. Any use of placeholder names like "Competitor A", "Competitor B", "Company X", or similar generic terms will be considered analysis failure. Research actual market participants and provide specific company identifications.

Company to Analyze: {company.company_name}
Stock Ticker: {company.ticker}
Sector/Subsector: {company.subsector}

Begin your comprehensive qualitative analysis now. Think systematically through each dimension, provide detailed evidence-based reasoning, and ensure all scores are well-justified with specific examples and supporting evidence.
"""

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

    def _create_dynamic_model_config(self, model_key: str) -> Optional[LLMConfig]:
        """
        Create a dynamic LLMConfig for models not in the configured list.
        This allows the system to use models discovered during API testing.
        """
        # Check if this looks like a TogetherAI model path
        if '/' in model_key and any(provider in model_key for provider in [
            'meta-llama', 'mistralai', 'Qwen', 'deepseek-ai', 'google',
            'codellama', 'NousResearch', 'moonshot-ai', 'THUDM', 'baichuan-inc',
            '01-ai', 'alibaba', 'internlm', 'togethercomputer', 'teknium',
            'upstage', 'garage-bAInd', 'openchat', 'Open-Orca', 'HuggingFaceH4'
        ]):
            # This is a raw TogetherAI model name
            if not self.together_api_key:
                logger.warning(f"Cannot create dynamic config for {model_key}: TogetherAI API key not available")
                return None

            return LLMConfig(
                provider='together',
                model_name=model_key,  # Use the raw model name
                max_tokens=4000,
                temperature=0.1,
                api_endpoint='https://api.together.xyz/v1/chat/completions',
                cost_per_1k_tokens=0.001,  # Default estimate
                context_window=8192  # Default estimate
            )

        # Check if this is an OpenAI model
        if model_key.startswith('gpt-') or model_key in ['o1-preview', 'o1-mini']:
            if not self.openai_api_key:
                logger.warning(f"Cannot create dynamic config for {model_key}: OpenAI API key not available")
                return None

            return LLMConfig(
                provider='openai',
                model_name=model_key,
                max_tokens=4000,
                temperature=0.1,
                api_endpoint='https://api.openai.com/v1/chat/completions',
                cost_per_1k_tokens=0.005,  # Default estimate
                context_window=128000
            )

        logger.warning(f"Cannot create dynamic config for unknown model: {model_key}")
        return None

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