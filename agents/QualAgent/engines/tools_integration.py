"""
Tools Integration for QualAgent system
Provides access to external APIs and tools for enhanced LLM analysis
"""

import os
import requests
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()  # Load from current directory or system environment

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class ToolConfig:
    """Configuration for external tools"""
    name: str
    api_key: str
    base_url: str
    description: str
    available: bool = True
    cost_per_call: float = 0.0

class ToolsIntegration:
    """Integration manager for external research tools"""

    def __init__(self):
        """Initialize tools integration with API keys from environment"""
        self.tools = self._load_tool_configurations()
        logger.info(f"Tools integration initialized with {len(self.tools)} tools")

    def _load_tool_configurations(self) -> Dict[str, ToolConfig]:
        """Load tool configurations from environment variables"""
        tools = {}

        # Tavily Search API
        tavily_key = os.getenv('TAVILY_API_KEY')
        if tavily_key:
            tools['Tavily'] = ToolConfig(
                name='Tavily',
                api_key=tavily_key,
                base_url='https://api.tavily.com/search',
                description='Real-time web search and information retrieval',
                cost_per_call=0.001  # Estimated cost per search
            )

        # Polygon Financial Data API
        polygon_key = os.getenv('POLYGON_API_KEY')
        if polygon_key:
            tools['Polygon'] = ToolConfig(
                name='Polygon',
                api_key=polygon_key,
                base_url='https://api.polygon.io',
                description='Real-time and historical financial market data',
                cost_per_call=0.002  # Estimated cost per request
            )

        # Exa Search API
        exa_key = os.getenv('EXA_API_KEY')
        if exa_key:
            tools['Exa'] = ToolConfig(
                name='Exa',
                api_key=exa_key,
                base_url='https://api.exa.ai',
                description='AI-powered semantic search for high-quality content',
                cost_per_call=0.001  # Estimated cost per search
            )

        # Traditional tools (require separate API access - described for LLM awareness)
        tools['Twitter'] = ToolConfig(
            name='Twitter',
            api_key='',  # No key available, but LLM should know about capability
            base_url='https://api.twitter.com',
            description='Social media sentiment and discussions analysis',
            available=False,  # Not directly accessible but LLM can reference
            cost_per_call=0.0
        )

        tools['GuruFocus'] = ToolConfig(
            name='GuruFocus',
            api_key='',
            base_url='https://api.gurufocus.com',
            description='Fundamental financial data and metrics analysis',
            available=False,
            cost_per_call=0.0
        )

        tools['Reddit'] = ToolConfig(
            name='Reddit',
            api_key='',
            base_url='https://api.reddit.com',
            description='Community discussions and sentiment analysis',
            available=False,
            cost_per_call=0.0
        )

        return tools

    def get_available_tools(self) -> List[str]:
        """Get list of available tool names"""
        return [name for name, config in self.tools.items() if config.available]

    def get_all_tools(self) -> List[str]:
        """Get list of all tool names (including unavailable ones for LLM awareness)"""
        return list(self.tools.keys())

    def get_tool_descriptions(self) -> Dict[str, str]:
        """Get tool descriptions for prompt construction"""
        return {name: config.description for name, config in self.tools.items()}

    def estimate_tool_costs(self, tools_used: List[str], calls_per_tool: int = 5) -> float:
        """Estimate costs for tool usage"""
        total_cost = 0.0
        for tool_name in tools_used:
            if tool_name in self.tools:
                tool_cost = self.tools[tool_name].cost_per_call * calls_per_tool
                total_cost += tool_cost
        return total_cost

    def validate_tool_access(self) -> Dict[str, bool]:
        """Validate access to configured tools"""
        validation_results = {}

        for tool_name, config in self.tools.items():
            if not config.available or not config.api_key:
                validation_results[tool_name] = False
                continue

            try:
                if tool_name == 'Tavily':
                    validation_results[tool_name] = self._validate_tavily()
                elif tool_name == 'Polygon':
                    validation_results[tool_name] = self._validate_polygon()
                elif tool_name == 'Exa':
                    validation_results[tool_name] = self._validate_exa()
                else:
                    validation_results[tool_name] = False
            except Exception as e:
                logger.error(f"Validation failed for {tool_name}: {e}")
                validation_results[tool_name] = False

        return validation_results

    def _validate_tavily(self) -> bool:
        """Validate Tavily API access"""
        try:
            headers = {
                'Authorization': f'Bearer {self.tools["Tavily"].api_key}',
                'Content-Type': 'application/json'
            }

            # Simple test query
            payload = {
                'query': 'test',
                'max_results': 1
            }

            response = requests.post(
                self.tools["Tavily"].base_url,
                headers=headers,
                json=payload,
                timeout=10
            )

            return response.status_code == 200
        except Exception as e:
            logger.error(f"Tavily validation error: {e}")
            return False

    def _validate_polygon(self) -> bool:
        """Validate Polygon API access"""
        try:
            # Test with a simple endpoint
            url = f"{self.tools['Polygon'].base_url}/v2/aggs/ticker/AAPL/range/1/day/2023-01-01/2023-01-02"
            params = {'apikey': self.tools['Polygon'].api_key}

            response = requests.get(url, params=params, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Polygon validation error: {e}")
            return False

    def _validate_exa(self) -> bool:
        """Validate Exa API access"""
        try:
            headers = {
                'Authorization': f'Bearer {self.tools["Exa"].api_key}',
                'Content-Type': 'application/json'
            }

            # Simple test search
            payload = {
                'query': 'test',
                'num_results': 1
            }

            response = requests.post(
                f"{self.tools['Exa'].base_url}/search",
                headers=headers,
                json=payload,
                timeout=10
            )

            return response.status_code == 200
        except Exception as e:
            logger.error(f"Exa validation error: {e}")
            return False

    def generate_tool_prompt_section(self, available_only: bool = True) -> str:
        """Generate prompt section describing available tools"""
        tools_to_include = self.get_available_tools() if available_only else self.get_all_tools()

        if not tools_to_include:
            return "No external tools are currently available for this analysis."

        tool_descriptions = self.get_tool_descriptions()

        prompt_section = "## Available Research Tools\n\n"
        prompt_section += "You have access to the following research tools to enhance your analysis:\n\n"

        for tool_name in tools_to_include:
            config = self.tools[tool_name]
            status = "✓ Available" if config.available else "○ Reference only"
            prompt_section += f"**{tool_name}** ({status}): {tool_descriptions[tool_name]}\n"

        prompt_section += "\n### Tool Usage Guidelines:\n"
        prompt_section += "- Use tools to gather current market data, sentiment, and news\n"
        prompt_section += "- Cross-reference information from multiple sources when possible\n"
        prompt_section += "- Always cite specific sources and data points in your analysis\n"
        prompt_section += "- Be aware of data recency and relevance to the analysis timeframe\n"

        if available_only:
            available_tools = [t for t in tools_to_include if self.tools[t].available]
            if available_tools:
                prompt_section += f"- Currently active tools: {', '.join(available_tools)}\n"

        return prompt_section

    def generate_tool_usage_examples(self, company_ticker: str) -> str:
        """Generate examples of how to use tools for specific company analysis"""
        examples = f"\n### Tool Usage Examples for {company_ticker} Analysis:\n\n"

        available_tools = self.get_available_tools()

        if 'Tavily' in available_tools:
            examples += f"**Tavily Search Examples:**\n"
            examples += f"- '{company_ticker} quarterly earnings 2024'\n"
            examples += f"- '{company_ticker} competitive analysis market share'\n"
            examples += f"- '{company_ticker} product launches partnerships 2024'\n\n"

        if 'Polygon' in available_tools:
            examples += f"**Polygon Financial Data Examples:**\n"
            examples += f"- Current stock price and trading volume for {company_ticker}\n"
            examples += f"- Historical price performance and volatility metrics\n"
            examples += f"- Options flow and institutional trading patterns\n\n"

        if 'Exa' in available_tools:
            examples += f"**Exa Semantic Search Examples:**\n"
            examples += f"- '{company_ticker} technological innovation AI strategy'\n"
            examples += f"- '{company_ticker} competitive moats differentiation'\n"
            examples += f"- '{company_ticker} market opportunity growth potential'\n\n"

        # Add examples for reference tools
        examples += "**Reference Tool Context (for analysis depth):**\n"
        examples += f"- **Twitter/X**: Social sentiment, management communications, real-time reactions\n"
        examples += f"- **GuruFocus**: Fundamental ratios, insider trading, institutional holdings\n"
        examples += f"- **Reddit**: Retail investor sentiment, community discussions, product feedback\n"

        return examples

    def get_tool_configuration_summary(self) -> Dict[str, Any]:
        """Get summary of tool configuration for system status"""
        summary = {
            'total_tools': len(self.tools),
            'available_tools': len(self.get_available_tools()),
            'validation_status': self.validate_tool_access(),
            'estimated_cost_per_analysis': self.estimate_tool_costs(self.get_available_tools()),
            'tool_details': {}
        }

        for name, config in self.tools.items():
            summary['tool_details'][name] = {
                'available': config.available,
                'has_api_key': bool(config.api_key),
                'description': config.description,
                'cost_per_call': config.cost_per_call
            }

        return summary

if __name__ == "__main__":
    # Test tools integration
    tools = ToolsIntegration()

    print("Available tools:", tools.get_available_tools())
    print("All tools:", tools.get_all_tools())
    print("Validation results:", tools.validate_tool_access())
    print("\nTool prompt section:")
    print(tools.generate_tool_prompt_section())
    print("\nUsage examples for AAPL:")
    print(tools.generate_tool_usage_examples("AAPL"))