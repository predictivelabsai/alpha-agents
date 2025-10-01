"""
Prompt Adaptation Engine for QualAgent
Adapts TechQual prompts for different LLM models based on their capabilities
Includes tools integration for enhanced research capabilities
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
import sys

# Add engines path for tools integration
sys.path.append(str(Path(__file__).parent))
from tools_integration import ToolsIntegration

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PromptAdaptation:
    """Configuration for adapting prompts to specific models"""
    model_family: str  # llama, mixtral, qwen, gpt, etc.
    instruction_style: str  # direct, conversational, structured
    supports_json_mode: bool
    max_context_length: int
    prefers_explicit_instructions: bool
    requires_role_emphasis: bool
    output_formatting_preference: str  # json, markdown, plain

class PromptAdapter:
    """Adapts TechQual prompts for different LLM models"""

    def __init__(self, enhanced_prompt_path: str = None):
        if enhanced_prompt_path is None:
            enhanced_prompt_path = Path(__file__).parent.parent / "prompts" / "TechQual_Enhanced_WithTools_v2.json"

        self.base_prompt = self._load_base_prompt(enhanced_prompt_path)
        self.adaptations = self._initialize_adaptations()
        self.tools_integration = ToolsIntegration()

        logger.info(f"PromptAdapter initialized with tools-enhanced prompt from {enhanced_prompt_path}")
        logger.info(f"Available research tools: {len(self.tools_integration.get_available_tools())}")

    def _load_base_prompt(self, prompt_path: Path) -> Dict:
        """Load the enhanced TechQual prompt"""
        try:
            with open(prompt_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load enhanced prompt from {prompt_path}: {e}")
            # Fallback to basic prompt structure
            return self._create_fallback_prompt()

    def _create_fallback_prompt(self) -> Dict:
        """Create fallback prompt if enhanced prompt can't be loaded"""
        return {
            "agent_name": "TechQual-Fallback",
            "version": "Fallback-1.0",
            "instructions": "You are a senior equity research analyst specialized in technology companies...",
            "input_schema": {
                "type": "object",
                "required": ["company", "ticker", "subsector"],
                "properties": {
                    "company": {"type": "string"},
                    "ticker": {"type": "string"},
                    "subsector": {"type": "string"}
                }
            },
            "output_schema": {"type": "object"}
        }

    def _initialize_adaptations(self) -> Dict[str, PromptAdaptation]:
        """Initialize model-specific adaptations"""
        return {
            'llama': PromptAdaptation(
                model_family='llama',
                instruction_style='direct',
                supports_json_mode=True,
                max_context_length=8192,
                prefers_explicit_instructions=True,
                requires_role_emphasis=True,
                output_formatting_preference='json'
            ),
            'mixtral': PromptAdaptation(
                model_family='mixtral',
                instruction_style='structured',
                supports_json_mode=True,
                max_context_length=32768,
                prefers_explicit_instructions=True,
                requires_role_emphasis=False,
                output_formatting_preference='json'
            ),
            'qwen': PromptAdaptation(
                model_family='qwen',
                instruction_style='conversational',
                supports_json_mode=True,
                max_context_length=32768,
                prefers_explicit_instructions=False,
                requires_role_emphasis=True,
                output_formatting_preference='json'
            ),
            'gpt': PromptAdaptation(
                model_family='gpt',
                instruction_style='conversational',
                supports_json_mode=True,
                max_context_length=128000,
                prefers_explicit_instructions=False,
                requires_role_emphasis=False,
                output_formatting_preference='json'
            ),
            'deepseek': PromptAdaptation(
                model_family='deepseek',
                instruction_style='direct',
                supports_json_mode=True,
                max_context_length=16384,
                prefers_explicit_instructions=True,
                requires_role_emphasis=True,
                output_formatting_preference='json'
            )
        }

    def get_model_family(self, model_name: str) -> str:
        """Determine model family from model name"""
        model_name_lower = model_name.lower()

        if 'llama' in model_name_lower:
            return 'llama'
        elif 'mixtral' in model_name_lower or 'mistral' in model_name_lower:
            return 'mixtral'
        elif 'qwen' in model_name_lower:
            return 'qwen'
        elif 'gpt' in model_name_lower:
            return 'gpt'
        elif 'deepseek' in model_name_lower:
            return 'deepseek'
        else:
            logger.warning(f"Unknown model family for {model_name}, using default adaptations")
            return 'llama'  # Default fallback

    def adapt_prompt(self, model_name: str, company_data: Dict,
                    focus_themes: List[str] = None,
                    geographies: List[str] = None) -> Tuple[str, List[Dict]]:
        """Adapt prompt for specific model and company"""

        model_family = self.get_model_family(model_name)
        adaptation = self.adaptations[model_family]

        # Create system prompt
        system_prompt = self._create_system_prompt(adaptation, company_data)

        # Create user messages
        user_messages = self._create_user_messages(
            adaptation, company_data, focus_themes, geographies
        )

        return system_prompt, user_messages

    def _create_system_prompt(self, adaptation: PromptAdaptation, company_data: Dict) -> str:
        """Create adapted system prompt"""

        base_instructions = self.base_prompt.get('instructions', '')

        # Add model-specific adaptations
        if adaptation.requires_role_emphasis:
            role_emphasis = """
CRITICAL: You are a senior equity research analyst. Your analysis will directly influence investment decisions.
Maintain the highest standards of rigor and impartiality throughout your assessment.
"""
            base_instructions = role_emphasis + base_instructions

        # Add tools section
        tools_section = self.tools_integration.generate_tool_prompt_section(available_only=True)
        base_instructions = base_instructions + "\n\n" + tools_section

        # Add tool usage examples for the specific company
        if company_data.get('ticker'):
            tools_examples = self.tools_integration.generate_tool_usage_examples(company_data['ticker'])
            base_instructions = base_instructions + "\n" + tools_examples

        if adaptation.prefers_explicit_instructions:
            explicit_guidance = """

EXECUTION REQUIREMENTS:
1. Follow the exact dimension framework provided
2. Provide specific justifications with source citations
3. Apply consistent scoring methodology
4. Include quantitative confidence scores (0.0-1.0)
5. Generate detailed competitive analysis
6. Ensure JSON output matches required schema exactly
7. Utilize available research tools to gather current information
8. Cite specific sources and data points in your analysis

"""
            base_instructions = base_instructions + explicit_guidance

        # Add JSON formatting instructions based on model preference
        if adaptation.output_formatting_preference == 'json':
            json_instructions = """
OUTPUT FORMAT CRITICAL:
- Return ONLY valid JSON matching the output schema
- NO markdown formatting, NO code blocks, NO explanatory text
- Ensure all required fields are present
- Use exact enumeration values specified in schema
- Format dates as YYYY-MM-DD
- Double-quote all strings, no trailing commas
- Include specific source citations in your analysis
"""
            base_instructions = base_instructions + json_instructions

        # Add model-specific optimizations
        if adaptation.model_family == 'llama':
            base_instructions = self._optimize_for_llama(base_instructions)
        elif adaptation.model_family == 'mixtral':
            base_instructions = self._optimize_for_mixtral(base_instructions)
        elif adaptation.model_family == 'qwen':
            base_instructions = self._optimize_for_qwen(base_instructions)
        elif adaptation.model_family == 'gpt':
            base_instructions = self._optimize_for_gpt(base_instructions)

        return base_instructions

    def _create_user_messages(self, adaptation: PromptAdaptation, company_data: Dict,
                            focus_themes: List[str], geographies: List[str]) -> List[Dict]:
        """Create user messages with company-specific information"""

        # Get available tools from tools integration
        available_tools = self.tools_integration.get_all_tools()  # Include all tools for LLM awareness

        # Create analysis input
        analysis_input = {
            "company": company_data.get('company_name'),
            "ticker": company_data.get('ticker'),
            "subsector": company_data.get('subsector'),
            "focus_themes": focus_themes or [],
            "geographies_of_interest": geographies or ["US", "Global"],
            "lookback_window_months": 24,
            "tools_available": available_tools
        }

        # Create user message based on instruction style
        if adaptation.instruction_style == 'direct':
            user_content = self._create_direct_message(analysis_input, company_data)
        elif adaptation.instruction_style == 'conversational':
            user_content = self._create_conversational_message(analysis_input, company_data)
        else:  # structured
            user_content = self._create_structured_message(analysis_input, company_data)

        return [{"role": "user", "content": user_content}]

    def _create_direct_message(self, analysis_input: Dict, company_data: Dict) -> str:
        """Create direct instruction message"""
        return f"""Analyze the following company using the TechQual framework:

COMPANY: {analysis_input['company']}
TICKER: {analysis_input['ticker']}
SUBSECTOR: {analysis_input['subsector']}

ANALYSIS PARAMETERS:
- Focus themes: {', '.join(analysis_input.get('focus_themes', [])) or 'General technology analysis'}
- Geographic focus: {', '.join(analysis_input.get('geographies_of_interest', []))}
- Lookback window: {analysis_input['lookback_window_months']} months
- Available tools: {', '.join(analysis_input.get('tools_available', []))}

Execute complete TechQual analysis and return results in the specified JSON format."""

    def _create_conversational_message(self, analysis_input: Dict, company_data: Dict) -> str:
        """Create conversational message"""
        return f"""I need your expert analysis of {analysis_input['company']} ({analysis_input['ticker']}),
a company in the {analysis_input['subsector']} sector.

Please conduct a comprehensive TechQual assessment focusing on:
{chr(10).join(f"- {theme}" for theme in analysis_input.get('focus_themes', ['Overall business strength and competitive position']))}

Key considerations:
- Geographic markets: {', '.join(analysis_input.get('geographies_of_interest', []))}
- Analysis timeframe: {analysis_input['lookback_window_months']} months
- Research tools: {', '.join(analysis_input.get('tools_available', []))}

Please provide your analysis in the structured JSON format as specified."""

    def _create_structured_message(self, analysis_input: Dict, company_data: Dict) -> str:
        """Create structured analysis message"""
        return f"""# TechQual Analysis Request

## Target Company
- **Name**: {analysis_input['company']}
- **Ticker**: {analysis_input['ticker']}
- **Sector**: {analysis_input['subsector']}

## Analysis Scope
- **Focus Areas**: {', '.join(analysis_input.get('focus_themes', ['Comprehensive analysis']))}
- **Geographic Markets**: {', '.join(analysis_input.get('geographies_of_interest', []))}
- **Time Horizon**: {analysis_input['lookback_window_months']} months
- **Research Tools**: {', '.join(analysis_input.get('tools_available', []))}

## Required Output
Complete TechQual framework analysis in JSON format covering all required dimensions, moat breakdown, competitive landscape, and insights."""

    def _optimize_for_llama(self, instructions: str) -> str:
        """Optimize prompt for Llama models"""
        llama_optimization = """
LLAMA-SPECIFIC GUIDANCE:
- Use clear, step-by-step reasoning
- Provide explicit examples where helpful
- Structure analysis logically from general to specific
- Emphasize factual accuracy and source verification
"""
        return instructions + llama_optimization

    def _optimize_for_mixtral(self, instructions: str) -> str:
        """Optimize prompt for Mixtral models"""
        mixtral_optimization = """
ANALYSIS EXCELLENCE STANDARDS:
- Apply multi-faceted reasoning across all dimensions
- Synthesize insights from diverse sources and perspectives
- Maintain analytical rigor while highlighting key insights
- Balance depth with clarity in all assessments
"""
        return instructions + mixtral_optimization

    def _optimize_for_qwen(self, instructions: str) -> str:
        """Optimize prompt for Qwen models"""
        qwen_optimization = """
ANALYTICAL FRAMEWORK EMPHASIS:
- Follow systematic evaluation methodology
- Consider both quantitative indicators and qualitative factors
- Provide nuanced assessment of competitive dynamics
- Ensure comprehensive coverage of all required dimensions
"""
        return instructions + qwen_optimization

    def _optimize_for_gpt(self, instructions: str) -> str:
        """Optimize prompt for GPT models"""
        # GPT models typically need minimal additional optimization
        return instructions

    def create_multi_model_prompts(self, company_data: Dict, model_list: List[str],
                                 focus_themes: List[str] = None,
                                 geographies: List[str] = None) -> Dict[str, Tuple[str, List[Dict]]]:
        """Create adapted prompts for multiple models"""

        prompts = {}

        for model_name in model_list:
            try:
                system_prompt, user_messages = self.adapt_prompt(
                    model_name, company_data, focus_themes, geographies
                )
                prompts[model_name] = (system_prompt, user_messages)
                logger.info(f"Created adapted prompt for {model_name}")
            except Exception as e:
                logger.error(f"Failed to create prompt for {model_name}: {e}")

        return prompts

    def estimate_token_usage(self, system_prompt: str, user_messages: List[Dict]) -> int:
        """Estimate token usage for prompt (rough approximation)"""
        total_text = system_prompt
        for msg in user_messages:
            total_text += msg.get('content', '')

        # Rough estimate: ~4 characters per token
        estimated_tokens = len(total_text) // 4

        # Add buffer for response (estimated 4000 tokens for TechQual output)
        estimated_tokens += 4000

        return estimated_tokens

    def validate_prompt_adaptation(self, model_name: str, company_data: Dict) -> Dict[str, Any]:
        """Validate prompt adaptation for a model"""
        try:
            system_prompt, user_messages = self.adapt_prompt(model_name, company_data)

            model_family = self.get_model_family(model_name)
            adaptation = self.adaptations[model_family]

            token_estimate = self.estimate_token_usage(system_prompt, user_messages)

            return {
                "status": "valid",
                "model_family": model_family,
                "system_prompt_length": len(system_prompt),
                "user_message_length": sum(len(msg.get('content', '')) for msg in user_messages),
                "estimated_tokens": token_estimate,
                "within_context_limit": token_estimate <= adaptation.max_context_length,
                "adaptations_applied": [
                    "role_emphasis" if adaptation.requires_role_emphasis else None,
                    "explicit_instructions" if adaptation.prefers_explicit_instructions else None,
                    f"optimized_for_{adaptation.model_family}"
                ]
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

def main():
    """Test prompt adaptation"""
    adapter = PromptAdapter()

    # Test company data
    test_company = {
        'company_name': 'NVIDIA Corporation',
        'ticker': 'NVDA',
        'subsector': 'Semiconductors'
    }

    # Test different models
    test_models = ['llama-3-70b', 'mixtral-8x7b', 'qwen2-72b', 'gpt-4o']

    print("Prompt Adaptation Test Results:")
    print("=" * 50)

    for model in test_models:
        validation = adapter.validate_prompt_adaptation(model, test_company)
        print(f"\n{model}:")
        print(f"  Status: {validation['status']}")
        if validation['status'] == 'valid':
            print(f"  Model family: {validation['model_family']}")
            print(f"  Estimated tokens: {validation['estimated_tokens']:,}")
            print(f"  Within context limit: {validation['within_context_limit']}")
        else:
            print(f"  Error: {validation['error']}")

if __name__ == "__main__":
    main()