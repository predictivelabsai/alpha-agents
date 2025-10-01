"""
Result Parser for QualAgent
Parses and validates LLM outputs according to TechQual schema
"""

import json
import re
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ParseResult:
    """Result of parsing LLM output"""
    success: bool
    parsed_data: Optional[Dict] = None
    error_message: Optional[str] = None
    validation_issues: List[str] = None
    completeness_score: float = 0.0
    quality_metrics: Dict[str, Any] = None

class ResultParser:
    """Parser for TechQual LLM analysis results"""

    def __init__(self):
        self.required_fields = [
            'company', 'ticker', 'as_of_date', 'sector',
            'dimensions', 'moat_breakdown', 'competitors',
            'key_tailwinds', 'key_headwinds', 'catalysts_next_6_12m',
            'red_flags', 'queries_run', 'audit', 'step_alignment'
        ]

        self.required_dimensions = [
            'Moat', 'Network Effects & Ecosystem', 'Technological Leadership & IP',
            'Switching Costs & Integration Depth', 'Economies of Scale & Supply-Chain Power',
            'Secular Trend Exposure', 'Geopolitical/Regulatory Positioning',
            'Partnerships, Distribution & Go-to-Market', 'Management & Execution Signals',
            'Market/Sentiment Snapshot', 'Platform Strategy & Ecosystem Development'
        ]

        self.valid_scores = ['High', 'Moderate', 'Low', 'Insufficient Info']
        self.valid_moat_labels = ['WIDE', 'NARROW', 'NONE', 'INSUFFICIENT']

    def parse_llm_output(self, raw_output: str, company_data: Dict) -> Dict:
        """Parse LLM output into structured format"""

        logger.info("Parsing LLM output")

        # Clean and extract JSON
        cleaned_output = self._clean_output(raw_output)
        json_data = self._extract_json(cleaned_output)

        if not json_data:
            raise ValueError("No valid JSON found in LLM output")

        # Validate and enhance the parsed data
        validated_data = self._validate_and_enhance(json_data, company_data)

        # Add parsing metadata
        validated_data['_parsing_metadata'] = {
            'parsed_at': datetime.now().isoformat(),
            'original_length': len(raw_output),
            'cleaned_length': len(cleaned_output),
            'validation_passed': True
        }

        logger.info("LLM output parsed successfully")
        return validated_data

    def _clean_output(self, raw_output: str) -> str:
        """Clean raw LLM output to extract JSON"""

        # Remove common markdown formatting
        cleaned = re.sub(r'```json\s*', '', raw_output)
        cleaned = re.sub(r'```\s*$', '', cleaned)

        # Remove leading/trailing whitespace
        cleaned = cleaned.strip()

        # Try to find JSON object boundaries
        start_idx = cleaned.find('{')
        if start_idx == -1:
            return cleaned

        # Find the matching closing brace
        brace_count = 0
        end_idx = -1

        for i in range(start_idx, len(cleaned)):
            if cleaned[i] == '{':
                brace_count += 1
            elif cleaned[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break

        if end_idx > start_idx:
            cleaned = cleaned[start_idx:end_idx]

        return cleaned

    def _extract_json(self, cleaned_output: str) -> Optional[Dict]:
        """Extract and parse JSON from cleaned output"""

        try:
            # First attempt: direct parsing
            return json.loads(cleaned_output)
        except json.JSONDecodeError as e:
            logger.warning(f"Direct JSON parsing failed: {e}")

        # Second attempt: fix common JSON issues
        try:
            fixed_json = self._fix_common_json_issues(cleaned_output)
            return json.loads(fixed_json)
        except json.JSONDecodeError as e:
            logger.warning(f"Fixed JSON parsing failed: {e}")

        # Third attempt: find largest valid JSON object
        try:
            return self._extract_largest_valid_json(cleaned_output)
        except Exception as e:
            logger.error(f"All JSON extraction attempts failed: {e}")
            return None

    def _fix_common_json_issues(self, json_str: str) -> str:
        """Fix common JSON formatting issues"""

        # Remove trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)

        # Fix unescaped quotes in strings
        # This is complex and might break valid JSON, so be careful
        json_str = re.sub(r'(?<!\\)"(?=.*")', '\\"', json_str)

        # Fix single quotes to double quotes (be very careful here)
        # Only replace single quotes that are clearly intended as string delimiters
        json_str = re.sub(r"'([^']*)'(\s*:)", r'"\1"\2', json_str)

        return json_str

    def _extract_largest_valid_json(self, text: str) -> Optional[Dict]:
        """Extract the largest valid JSON object from text"""

        # Find all potential JSON objects
        potential_objects = []
        i = 0

        while i < len(text):
            if text[i] == '{':
                # Try to extract JSON object starting here
                brace_count = 0
                start = i

                for j in range(i, len(text)):
                    if text[j] == '{':
                        brace_count += 1
                    elif text[j] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            potential_json = text[start:j+1]
                            try:
                                parsed = json.loads(potential_json)
                                potential_objects.append((len(potential_json), parsed))
                            except:
                                pass
                            break
            i += 1

        # Return the largest valid JSON object
        if potential_objects:
            potential_objects.sort(key=lambda x: x[0], reverse=True)
            return potential_objects[0][1]

        return None

    def _validate_and_enhance(self, json_data: Dict, company_data: Dict) -> Dict:
        """Validate and enhance parsed JSON data"""

        # Ensure required top-level fields
        if 'company' not in json_data:
            json_data['company'] = company_data.get('company_name', 'Unknown')

        if 'ticker' not in json_data:
            json_data['ticker'] = company_data.get('ticker', 'Unknown')

        if 'as_of_date' not in json_data:
            json_data['as_of_date'] = datetime.now().strftime('%Y-%m-%d')

        if 'sector' not in json_data:
            json_data['sector'] = company_data.get('subsector', 'Technology')

        # Validate and fix dimensions
        json_data['dimensions'] = self._validate_dimensions(json_data.get('dimensions', []))

        # Validate and fix moat breakdown
        json_data['moat_breakdown'] = self._validate_moat_breakdown(json_data.get('moat_breakdown', {}))

        # Validate and fix competitors
        json_data['competitors'] = self._validate_competitors(json_data.get('competitors', []))

        # Ensure list fields exist
        list_fields = ['key_tailwinds', 'key_headwinds', 'catalysts_next_6_12m', 'red_flags']
        for field in list_fields:
            if field not in json_data or not isinstance(json_data[field], list):
                json_data[field] = []

        # Validate and fix metadata fields
        json_data['queries_run'] = self._validate_queries_run(json_data.get('queries_run', {}))
        json_data['audit'] = self._validate_audit(json_data.get('audit', {}))
        json_data['step_alignment'] = self._validate_step_alignment(json_data.get('step_alignment', {}))

        return json_data

    def _validate_dimensions(self, dimensions: List[Dict]) -> List[Dict]:
        """Validate and fix dimensions analysis"""

        if not isinstance(dimensions, list):
            return []

        validated_dimensions = []

        for dim in dimensions:
            if not isinstance(dim, dict):
                continue

            # Ensure required fields
            validated_dim = {
                'name': dim.get('name', 'Unknown'),
                'score': dim.get('score', 'Insufficient Info'),
                'justification': dim.get('justification', 'No justification provided'),
                'confidence': float(dim.get('confidence', 0.5)),
                'sources': self._validate_sources(dim.get('sources', [])),
                'reasoning_summaries': self._validate_reasoning_summaries(dim.get('reasoning_summaries', {}))
            }

            # Validate score
            if validated_dim['score'] not in self.valid_scores:
                validated_dim['score'] = 'Insufficient Info'

            # Validate confidence
            if not 0.0 <= validated_dim['confidence'] <= 1.0:
                validated_dim['confidence'] = 0.5

            validated_dimensions.append(validated_dim)

        # Ensure all required dimensions are present
        present_dimensions = {dim['name'] for dim in validated_dimensions}
        for required_dim in self.required_dimensions:
            if required_dim not in present_dimensions:
                validated_dimensions.append({
                    'name': required_dim,
                    'score': 'Insufficient Info',
                    'justification': 'Analysis not provided',
                    'confidence': 0.0,
                    'sources': [],
                    'reasoning_summaries': {
                        'supporting_evidence': [],
                        'counter_evidence': [],
                        'assumptions': [],
                        'contradictions': [],
                        'red_team_alternative': 'Analysis not provided'
                    }
                })

        return validated_dimensions

    def _validate_moat_breakdown(self, moat_breakdown: Dict) -> Dict:
        """Validate and fix moat breakdown"""

        if not isinstance(moat_breakdown, dict):
            moat_breakdown = {}

        moat_components = ['brand_monopoly', 'barriers_to_entry', 'economies_of_scale',
                          'network_effects', 'switching_costs']

        validated_moat = {}

        for component in moat_components:
            component_data = moat_breakdown.get(component, {})
            if not isinstance(component_data, dict):
                component_data = {}

            validated_component = {
                'label': component_data.get('label', 'INSUFFICIENT'),
                'notes': component_data.get('notes', 'No analysis provided'),
                'sources': self._validate_sources(component_data.get('sources', []))
            }

            # Validate label
            if validated_component['label'] not in self.valid_moat_labels:
                validated_component['label'] = 'INSUFFICIENT'

            validated_moat[component] = validated_component

        # Validate overall moat
        overall_moat = moat_breakdown.get('overall_moat', {})
        if not isinstance(overall_moat, dict):
            overall_moat = {}

        validated_moat['overall_moat'] = {
            'score': overall_moat.get('score', 'Insufficient Info'),
            'rollup_justification': overall_moat.get('rollup_justification', 'No justification provided'),
            'confidence': float(overall_moat.get('confidence', 0.5))
        }

        # Validate overall score
        if validated_moat['overall_moat']['score'] not in self.valid_scores:
            validated_moat['overall_moat']['score'] = 'Insufficient Info'

        # Validate overall confidence
        if not 0.0 <= validated_moat['overall_moat']['confidence'] <= 1.0:
            validated_moat['overall_moat']['confidence'] = 0.5

        return validated_moat

    def _validate_competitors(self, competitors: List[Dict]) -> List[Dict]:
        """Validate and fix competitors analysis"""

        if not isinstance(competitors, list):
            return []

        validated_competitors = []

        for comp in competitors:
            if not isinstance(comp, dict):
                continue

            validated_comp = {
                'name': comp.get('name', 'Unknown Competitor'),
                'ticker': comp.get('ticker', 'N/A'),
                'peer_role': comp.get('peer_role', 'Direct'),
                'comparability_score': int(comp.get('comparability_score', 5)),
                'business_model': comp.get('business_model', 'Not specified'),
                'target_segments': comp.get('target_segments', 'Not specified'),
                'differentiators': comp.get('differentiators', []) if isinstance(comp.get('differentiators'), list) else [],
                'moat_tags': self._validate_moat_tags(comp.get('moat_tags', {})),
                'geographies': comp.get('geographies', []) if isinstance(comp.get('geographies'), list) else [],
                'market_position': comp.get('market_position', 'Challenger'),
                'partnerships_or_design_wins': comp.get('partnerships_or_design_wins', []) if isinstance(comp.get('partnerships_or_design_wins'), list) else [],
                'peer_selection_rationale': comp.get('peer_selection_rationale', 'No rationale provided'),
                'sources': self._validate_sources(comp.get('sources', []))
            }

            # Validate peer role
            if validated_comp['peer_role'] not in ['Direct', 'Leader', 'Adjacent']:
                validated_comp['peer_role'] = 'Direct'

            # Validate comparability score
            if not 0 <= validated_comp['comparability_score'] <= 10:
                validated_comp['comparability_score'] = 5

            # Validate market position
            if validated_comp['market_position'] not in ['Leader', 'Challenger', 'Niche']:
                validated_comp['market_position'] = 'Challenger'

            validated_competitors.append(validated_comp)

        return validated_competitors

    def _validate_moat_tags(self, moat_tags: Dict) -> Dict:
        """Validate moat tags for competitors"""

        if not isinstance(moat_tags, dict):
            moat_tags = {}

        validated_tags = {}
        tag_names = ['brand', 'barriers', 'scale', 'network', 'switching']

        for tag_name in tag_names:
            tag_value = moat_tags.get(tag_name, 'INSUFFICIENT')
            if tag_value not in self.valid_moat_labels:
                tag_value = 'INSUFFICIENT'
            validated_tags[tag_name] = tag_value

        return validated_tags

    def _validate_sources(self, sources: List[Dict]) -> List[Dict]:
        """Validate sources list"""

        if not isinstance(sources, list):
            return []

        validated_sources = []

        for source in sources:
            if not isinstance(source, dict):
                continue

            validated_source = {
                'title': source.get('title', 'Unknown Source'),
                'url': source.get('url', 'http://example.com'),
                'publisher': source.get('publisher', 'Unknown Publisher'),
                'date': source.get('date', datetime.now().strftime('%Y-%m-%d')),
                'quote': source.get('quote', '')
            }

            # Validate date format
            try:
                datetime.strptime(validated_source['date'], '%Y-%m-%d')
            except ValueError:
                validated_source['date'] = datetime.now().strftime('%Y-%m-%d')

            validated_sources.append(validated_source)

        return validated_sources

    def _validate_reasoning_summaries(self, reasoning: Dict) -> Dict:
        """Validate reasoning summaries"""

        if not isinstance(reasoning, dict):
            reasoning = {}

        return {
            'supporting_evidence': reasoning.get('supporting_evidence', []) if isinstance(reasoning.get('supporting_evidence'), list) else [],
            'counter_evidence': reasoning.get('counter_evidence', []) if isinstance(reasoning.get('counter_evidence'), list) else [],
            'assumptions': reasoning.get('assumptions', []) if isinstance(reasoning.get('assumptions'), list) else [],
            'contradictions': reasoning.get('contradictions', []) if isinstance(reasoning.get('contradictions'), list) else [],
            'red_team_alternative': reasoning.get('red_team_alternative', 'No alternative perspective provided')
        }

    def _validate_queries_run(self, queries_run: Dict) -> Dict:
        """Validate queries run metadata"""

        if not isinstance(queries_run, dict):
            queries_run = {}

        return {
            'web': queries_run.get('web', []) if isinstance(queries_run.get('web'), list) else [],
            'twitter': queries_run.get('twitter', []) if isinstance(queries_run.get('twitter'), list) else [],
            'gurufocus': queries_run.get('gurufocus', []) if isinstance(queries_run.get('gurufocus'), list) else [],
            'reddit': queries_run.get('reddit', []) if isinstance(queries_run.get('reddit'), list) else []
        }

    def _validate_audit(self, audit: Dict) -> Dict:
        """Validate audit metadata"""

        if not isinstance(audit, dict):
            audit = {}

        return {
            'primary_source_used': bool(audit.get('primary_source_used', False)),
            'recency_ok': bool(audit.get('recency_ok', True)),
            'contradictions_noted': bool(audit.get('contradictions_noted', False)),
            'tool_limitations': audit.get('tool_limitations', 'No limitations noted')
        }

    def _validate_step_alignment(self, step_alignment: Dict) -> Dict:
        """Validate step alignment metadata"""

        if not isinstance(step_alignment, dict):
            step_alignment = {}

        return {
            'covered': step_alignment.get('covered', []) if isinstance(step_alignment.get('covered'), list) else [],
            'qualitative_proxies_only': step_alignment.get('qualitative_proxies_only', []) if isinstance(step_alignment.get('qualitative_proxies_only'), list) else [],
            'excluded': step_alignment.get('excluded', []) if isinstance(step_alignment.get('excluded'), list) else []
        }

    def validate_completeness(self, parsed_data: Dict) -> Tuple[float, List[str]]:
        """Calculate completeness score and identify missing elements"""

        issues = []
        total_score = 0
        max_score = 0

        # Check top-level required fields
        for field in self.required_fields:
            max_score += 1
            if field in parsed_data and parsed_data[field]:
                total_score += 1
            else:
                issues.append(f"Missing or empty field: {field}")

        # Check dimensions completeness
        if 'dimensions' in parsed_data:
            present_dims = {dim.get('name') for dim in parsed_data['dimensions']}
            for required_dim in self.required_dimensions:
                max_score += 1
                if required_dim in present_dims:
                    total_score += 1
                else:
                    issues.append(f"Missing dimension: {required_dim}")

        # Check moat breakdown completeness
        if 'moat_breakdown' in parsed_data:
            moat_components = ['brand_monopoly', 'barriers_to_entry', 'economies_of_scale',
                             'network_effects', 'switching_costs', 'overall_moat']
            for component in moat_components:
                max_score += 1
                if component in parsed_data['moat_breakdown']:
                    total_score += 1
                else:
                    issues.append(f"Missing moat component: {component}")

        completeness_score = total_score / max_score if max_score > 0 else 0.0

        return completeness_score, issues

def main():
    """Test result parser"""
    parser = ResultParser()

    # Test with sample JSON output
    sample_output = '''
    {
        "company": "Test Company",
        "ticker": "TEST",
        "as_of_date": "2025-01-01",
        "sector": "Technology",
        "dimensions": [
            {
                "name": "Moat",
                "score": "High",
                "justification": "Strong competitive advantages",
                "confidence": 0.8,
                "sources": [
                    {
                        "title": "Company Analysis",
                        "url": "http://example.com",
                        "publisher": "Analyst",
                        "date": "2025-01-01"
                    }
                ],
                "reasoning_summaries": {
                    "supporting_evidence": ["Strong brand"],
                    "counter_evidence": ["Some competition"],
                    "assumptions": ["Market remains stable"],
                    "contradictions": [],
                    "red_team_alternative": "Could face disruption"
                }
            }
        ],
        "moat_breakdown": {
            "brand_monopoly": {
                "label": "WIDE",
                "notes": "Strong brand recognition",
                "sources": []
            }
        },
        "competitors": [],
        "key_tailwinds": ["Growing market"],
        "key_headwinds": ["Competition"],
        "catalysts_next_6_12m": ["Product launch"],
        "red_flags": ["High valuation"],
        "queries_run": {"web": [], "twitter": [], "gurufocus": [], "reddit": []},
        "audit": {"primary_source_used": true, "recency_ok": true, "contradictions_noted": false, "tool_limitations": "None"},
        "step_alignment": {"covered": [], "qualitative_proxies_only": [], "excluded": []}
    }
    '''

    try:
        result = parser.parse_llm_output(sample_output, {"company_name": "Test Company", "ticker": "TEST"})
        print("Parsing successful!")
        print(f"Company: {result.get('company')}")
        print(f"Dimensions: {len(result.get('dimensions', []))}")

        completeness, issues = parser.validate_completeness(result)
        print(f"Completeness: {completeness:.2%}")
        if issues:
            print("Issues found:")
            for issue in issues[:5]:
                print(f"  - {issue}")

    except Exception as e:
        print(f"Parsing failed: {e}")

if __name__ == "__main__":
    main()