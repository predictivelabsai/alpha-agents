# Qualitative Analysis Database Integration

## Overview
This document explains how qualitative analysis JSON files are converted and stored in the PostgreSQL database table `qualitative_analysis`.

## Database Table Structure

### Qualitative Analysis Table (`qualitative_analysis`)

| Column | Type | MSB Example | AAPL Example | TSLA Example |
|--------|------|-------------|--------------|--------------|
| **id** | SERIAL | 1 | 2 | 3 |
| **run_id** | VARCHAR(50) | qual_analysis_1760222470 | qual_analysis_1760223000 | qual_analysis_1760223500 |
| **company_ticker** | VARCHAR(10) | MSB | AAPL | TSLA |
| **analysis_timestamp** | BIGINT | 1760222470 | 1760223000 | 1760223500 |
| **analysis_type** | VARCHAR(50) | multi_llm | multi_llm | multi_llm |
| **models_used** | TEXT[] | [qwen2-72b, llama-3-70b, deepseek-coder-33b] | [gpt-4, claude-3-opus, llama-3-70b] | [gpt-4, claude-3-sonnet, mixtral-8x7b] |
| **successful_models** | TEXT[] | [llama-3-70b, llama-3.1-70b, mixtral-8x7b] | [gpt-4, claude-3-opus, llama-3-70b] | [gpt-4, claude-3-sonnet, mixtral-8x7b] |
| **failed_models** | TEXT[] | [qwen2-72b, deepseek-coder-33b] | [] | [] |
| **company_name** | VARCHAR(255) | MSB Inc. | Apple Inc. | Tesla Inc. |
| **company_subsector** | VARCHAR(100) | Technology | Technology | Automotive |
| **company_market_cap_usd** | BIGINT | NULL | 3000000000000 | 800000000000 |
| **composite_score** | DECIMAL(5,3) | 2.781 | 4.250 | 3.750 |
| **composite_confidence** | DECIMAL(5,3) | 0.760 | 0.850 | 0.820 |
| **moat_brand_monopoly_score** | DECIMAL(5,3) | 2.714 | 4.500 | 4.000 |
| **moat_brand_monopoly_confidence** | DECIMAL(5,3) | 0.736 | 0.900 | 0.850 |
| **moat_network_effects_score** | DECIMAL(5,3) | 3.643 | 4.200 | 3.500 |
| **moat_network_effects_confidence** | DECIMAL(5,3) | 0.764 | 0.880 | 0.780 |
| **competitive_differentiation_score** | DECIMAL(5,3) | 3.357 | 4.800 | 4.200 |
| **technology_moats_score** | DECIMAL(5,3) | 4.000 | 4.600 | 4.500 |
| **management_quality_score** | DECIMAL(5,3) | 4.000 | 4.700 | 3.800 |
| **key_growth_drivers_score** | DECIMAL(5,3) | 4.357 | 4.300 | 4.100 |
| **major_risk_factors_score** | DECIMAL(5,3) | 2.357 | 2.100 | 3.200 |
| **red_flags_score** | DECIMAL(5,3) | 2.143 | 1.500 | 2.800 |
| **execution_time_seconds** | DECIMAL(10,3) | 14.495 | 12.350 | 15.200 |
| **total_cost_usd** | DECIMAL(10,6) | 0.005134 | 0.008750 | 0.012500 |
| **concurrent_execution** | BOOLEAN | true | true | true |
| **best_model** | VARCHAR(100) | llama-3-70b | gpt-4 | gpt-4 |
| **full_analysis_data** | JSONB | {complete JSON} | {complete JSON} | {complete JSON} |
| **individual_model_results** | JSONB | {LLM results} | {LLM results} | {LLM results} |
| **execution_metadata** | JSONB | {execution details} | {execution details} | {execution details} |

**Total Columns:** 25+ (including all 15 score components with their confidence and justification columns)

## JSON File Structure → Database Table Mapping

### Input: JSON File (`multi_llm_analysis_MSB_1760222470.json`)

```json
{
  "metadata": {
    "timestamp": 1760222470,
    "company_ticker": "MSB",
    "analysis_type": "multi_llm",
    "models_used": ["qwen2-72b", "llama-3-70b", "deepseek-coder-33b"]
  },
  "company": {
    "company_name": "MSB Inc.",
    "ticker": "MSB",
    "subsector": "Technology",
    "market_cap_usd": null,
    "employees": null
  },
  "composite_score": {
    "score": 2.7807639705096703,
    "confidence": 0.7602169035153329,
    "components_count": 15
  },
  "consensus_scores": {
    "moat_brand_monopoly": {
      "score": 2.7142857142857144,
      "confidence": 0.7357142857142858,
      "justification": "MSB has a recognizable brand..."
    },
    "moat_network_effects": {
      "score": 3.642857142857143,
      "confidence": 0.7642857142857143,
      "justification": "MSB's platform has a strong network effect..."
    }
  },
  "individual_model_results": {
    "llama-3-70b": {
      "model_name": "llama-3-70b",
      "provider": "together",
      "tokens_used": 1859,
      "cost_usd": 0.0016730999999999998
    }
  },
  "total_cost_usd": 0.0051342,
  "total_time_seconds": 14.495295763015747
}
```

### Output: Database Table Rows

#### Example 1: MSB Analysis
```sql
INSERT INTO qualitative_analysis (
    run_id, company_ticker, analysis_timestamp, analysis_type,
    models_used, successful_models, failed_models,
    company_name, company_subsector, company_market_cap_usd,
    composite_score, composite_confidence, composite_components_count,
    moat_brand_monopoly_score, moat_brand_monopoly_confidence, moat_brand_monopoly_justification,
    moat_network_effects_score, moat_network_effects_confidence, moat_network_effects_justification,
    execution_time_seconds, total_cost_usd, concurrent_execution,
    full_analysis_data, individual_model_results
) VALUES (
    'qual_analysis_1760222470', 'MSB', 1760222470, 'multi_llm',
    ARRAY['qwen2-72b', 'llama-3-70b', 'deepseek-coder-33b'], 
    ARRAY['llama-3-70b', 'llama-3.1-70b', 'mixtral-8x7b'], 
    ARRAY['qwen2-72b', 'deepseek-coder-33b'],
    'MSB Inc.', 'Technology', NULL,
    2.781, 0.760, 15,
    2.714, 0.736, 'MSB has a recognizable brand in the technology sector...',
    3.643, 0.764, 'MSB''s platform has a strong network effect...',
    14.495, 0.005134, true,
    '{"metadata": {...}, "company": {...}, "composite_score": {...}}',
    '{"llama-3-70b": {"model_name": "llama-3-70b", "provider": "together", ...}}'
);
```

#### Example 2: AAPL Analysis
```sql
INSERT INTO qualitative_analysis (
    run_id, company_ticker, analysis_timestamp, analysis_type,
    models_used, successful_models, failed_models,
    company_name, company_subsector, company_market_cap_usd,
    composite_score, composite_confidence, composite_components_count,
    moat_brand_monopoly_score, moat_brand_monopoly_confidence, moat_brand_monopoly_justification,
    moat_network_effects_score, moat_network_effects_confidence, moat_network_effects_justification,
    execution_time_seconds, total_cost_usd, concurrent_execution,
    full_analysis_data, individual_model_results
) VALUES (
    'qual_analysis_1760223000', 'AAPL', 1760223000, 'multi_llm',
    ARRAY['gpt-4', 'claude-3-opus', 'llama-3-70b'], 
    ARRAY['gpt-4', 'claude-3-opus', 'llama-3-70b'], 
    ARRAY[],
    'Apple Inc.', 'Technology', 3000000000000,
    4.250, 0.850, 15,
    4.500, 0.900, 'Apple has one of the strongest brand monopolies in the world...',
    4.200, 0.880, 'Apple''s ecosystem creates powerful network effects...',
    12.350, 0.008750, true,
    '{"metadata": {...}, "company": {...}, "composite_score": {...}}',
    '{"gpt-4": {"model_name": "gpt-4", "provider": "openai", ...}}'
);
```

#### Example 3: TSLA Analysis
```sql
INSERT INTO qualitative_analysis (
    run_id, company_ticker, analysis_timestamp, analysis_type,
    models_used, successful_models, failed_models,
    company_name, company_subsector, company_market_cap_usd,
    composite_score, composite_confidence, composite_components_count,
    moat_brand_monopoly_score, moat_brand_monopoly_confidence, moat_brand_monopoly_justification,
    moat_network_effects_score, moat_network_effects_confidence, moat_network_effects_justification,
    execution_time_seconds, total_cost_usd, concurrent_execution,
    full_analysis_data, individual_model_results
) VALUES (
    'qual_analysis_1760223500', 'TSLA', 1760223500, 'multi_llm',
    ARRAY['gpt-4', 'claude-3-sonnet', 'mixtral-8x7b'], 
    ARRAY['gpt-4', 'claude-3-sonnet', 'mixtral-8x7b'], 
    ARRAY[],
    'Tesla Inc.', 'Automotive', 800000000000,
    3.750, 0.820, 15,
    4.000, 0.850, 'Tesla has built a strong brand around innovation and sustainability...',
    3.500, 0.780, 'Tesla''s Supercharger network creates moderate network effects...',
    15.200, 0.012500, true,
    '{"metadata": {...}, "company": {...}, "composite_score": {...}}',
    '{"gpt-4": {"model_name": "gpt-4", "provider": "openai", ...}}'
);
```

## Data Transformation Process

### 1. **Metadata Extraction**
```python
# From JSON
metadata = analysis_data['metadata']
company = analysis_data['company']
composite_score = analysis_data['composite_score']

# To Database
run_id = f"qual_analysis_{metadata['timestamp']}"
company_ticker = company['ticker']
analysis_timestamp = metadata['timestamp']
analysis_type = metadata['analysis_type']
models_used = metadata['models_used']
```

### 2. **Score Component Mapping**
```python
# From JSON consensus_scores
consensus_scores = analysis_data['consensus_scores']

# To Database columns
for component, data in consensus_scores.items():
    db_column_score = f"{component}_score"
    db_column_confidence = f"{component}_confidence" 
    db_column_justification = f"{component}_justification"
    
    # Map values
    insert_data[db_column_score] = data.get('score')
    insert_data[db_column_confidence] = data.get('confidence')
    insert_data[db_column_justification] = data.get('justification', '')
```

### 3. **JSON Storage**
```python
# Store complete JSON for detailed analysis
insert_data['full_analysis_data'] = json.dumps(analysis_data)
insert_data['individual_model_results'] = json.dumps(analysis_data['individual_model_results'])
insert_data['execution_metadata'] = json.dumps(analysis_data['execution_metadata'])
```

## Database Schema Benefits

### **Structured Querying**
```sql
-- Find all companies with high composite scores
SELECT company_ticker, company_name, composite_score 
FROM qualitative_analysis 
WHERE composite_score > 4.0 
ORDER BY composite_score DESC;

-- Compare specific moat scores across companies
SELECT company_ticker, 
       moat_brand_monopoly_score, 
       moat_network_effects_score,
       moat_switching_costs_score
FROM qualitative_analysis 
WHERE company_ticker IN ('AAPL', 'MSFT', 'GOOGL');
```

### **JSONB Advanced Queries**
```sql
-- Query within the full JSON data
SELECT company_ticker, 
       full_analysis_data->'consensus_scores'->'technology_moats'->>'score' as tech_moat_score
FROM qualitative_analysis 
WHERE full_analysis_data->'metadata'->>'analysis_type' = 'multi_llm';

-- Find analyses with specific models
SELECT company_ticker, models_used
FROM qualitative_analysis 
WHERE 'gpt-4' = ANY(models_used);
```

## Usage in QualAgent

### **Saving Analysis Results**
```python
from utils.db_util import save_qualitative_analysis

# After running QualAgent analysis
with open('multi_llm_analysis_MSB_1760222470.json', 'r') as f:
    analysis_data = json.load(f)

success, message = save_qualitative_analysis(analysis_data, run_id="qual_analysis_20250115")
print(f"Save result: {success}, Message: {message}")
```

### **Loading Analysis Results**
```python
from utils.db_util import load_qualitative_analysis

# Load all analyses for a company
msb_analyses = load_qualitative_analysis(ticker="MSB")

# Load recent analyses
recent_analyses = load_qualitative_analysis(limit=10)

# Load specific run
specific_run = load_qualitative_analysis(run_id="qual_analysis_20250115")
```

## File Structure
```
alpha-agents/
├── sql/
│   └── create_qualitative_analysis_table.sql  # Database schema
├── utils/
│   └── db_util.py                             # Save/load functions
├── agents/QualAgent/results/
│   ├── multi_llm_analysis_MSB_1760222470.json
│   ├── multi_llm_analysis_AAPL_1760223000.json
│   └── multi_llm_analysis_TSLA_1760223500.json
└── docs/
    └── Qualitative_Analysis_Database_Integration.md
```

This integration allows for both structured querying of key metrics and detailed analysis of the complete JSON data using PostgreSQL's powerful JSONB capabilities.
