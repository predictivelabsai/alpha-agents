# QualAgent Enhanced Analysis System - Comprehensive User Guide

## Table of Contents
1. [System Overview](#system-overview)
2. [Quick Start Guide](#quick-start-guide)
3. [Adding New Companies](#adding-new-companies)
4. [Running Analysis](#running-analysis)
5. [Analysis Configuration Options](#analysis-configuration-options)
6. [Understanding Results](#understanding-results)
7. [Human Feedback Integration](#human-feedback-integration)
8. [Weight Management](#weight-management)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Features](#advanced-features)

## System Overview

The Enhanced QualAgent is a sophisticated financial analysis system that combines multiple LLMs to provide comprehensive company evaluations. It features:

- **Multi-LLM Analysis**: Concurrent execution across 5 LLM models for consensus-based insights
- **Enhanced Scoring**: Comprehensive scoring across 14+ components (vs original 5)
- **Weighted Composite Scoring**: User-configurable weights with confidence adjustments
- **Human Feedback Integration**: Expert feedback collection and model improvement
- **Multiple Output Formats**: JSON, CSV, PKL, and Markdown reports
- **Interactive Weight Approval**: Investment philosophy-based weight customization

### Key Components
- **Enhanced Scoring System**: Expands beyond competitive moats to all analysis aspects
- **Multi-LLM Engine**: Runs 5 models concurrently (mixtral-8x7b, llama-3-70b, qwen2-72b, llama-3.1-70b, deepseek-coder-33b)
- **Human Feedback System**: Collects expert preferences and builds training datasets
- **Weight Approval System**: Interactive weight configuration with investment philosophy presets

## Quick Start Guide

### Prerequisites
1. **Environment Setup**:
   ```bash
   # Activate your virtual environment
   conda activate personal_CRM

   # Install dependencies
   pip install -r requirements.txt
   ```

2. **API Configuration**:
   ```bash
   # Copy environment template
   cp .env.template .env

   # Edit .env file with your API keys
   TOGETHER_API_KEY=your_together_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here  # Optional
   ```

3. **Quick Test**:
   ```bash
   python run_tests.py
   ```

### 30-Second Demo
```bash
# Run enhanced analysis for NVIDIA with default settings
python run_enhanced_analysis.py --user-id analyst1 --company NVDA

# Check cost first
python run_enhanced_analysis.py --user-id analyst1 --company NVDA --cost-estimate-only
```

## Adding New Companies

### Method 1: Using the Simple Utility
```bash
# Add a company quickly
python utils/simple_add_company.py
# Follow the interactive prompts
```

### Method 2: Using the Advanced Utility
```bash
# Add with detailed information
python utils/add_company.py
```

### Method 3: Direct JSON Editing
Edit `data/companies.json`:
```json
{
  "AAPL": {
    "company_name": "Apple Inc.",
    "ticker": "AAPL",
    "subsector": "Technology Hardware",
    "market_cap": "Large Cap",
    "geography": "United States",
    "added_date": "2025-01-01",
    "status": "active"
  }
}
```

### Method 4: Programmatic Addition
```python
from models.json_data_manager import JSONDataManager, Company

data_manager = JSONDataManager()
company = Company(
    company_name="Tesla Inc.",
    ticker="TSLA",
    subsector="Electric Vehicles",
    market_cap="Large Cap",
    geography="United States"
)
data_manager.add_company("TSLA", company)
```

## Running Analysis

### Basic Analysis
```bash
# Standard comprehensive analysis
python run_enhanced_analysis.py --user-id analyst1 --company TSLA

# Quick analysis (faster, fewer components)
python run_enhanced_analysis.py --user-id analyst1 --company TSLA --analysis-type quick

# Expert-guided analysis (most comprehensive)
python run_enhanced_analysis.py --user-id analyst1 --company TSLA --analysis-type expert_guided
```

### Cost Estimation
```bash
# Get cost estimate before running
python run_enhanced_analysis.py --user-id analyst1 --company TSLA --cost-estimate-only
```

### Model Selection
```bash
# Use specific models
python run_enhanced_analysis.py --user-id analyst1 --company TSLA --models mixtral-8x7b,llama-3-70b

# Use single model for testing
python run_enhanced_analysis.py --user-id analyst1 --company TSLA --models mixtral-8x7b
```

### Batch Analysis
```bash
# Analyze multiple companies
python run_enhanced_analysis.py --user-id analyst1 --companies TSLA,NVDA,AAPL
```

## Analysis Configuration Options

### Command Line Parameters

| Parameter | Description | Options/Examples |
|-----------|-------------|------------------|
| `--user-id` | User identifier | `analyst1`, `researcher2` |
| `--company` | Single company ticker | `NVDA`, `TSLA`, `AAPL` |
| `--companies` | Multiple companies | `NVDA,TSLA,AAPL` |
| `--analysis-type` | Analysis depth | `quick`, `comprehensive`, `expert_guided` |
| `--models` | Specific LLM models | `mixtral-8x7b`, `llama-3-70b,qwen2-72b` |
| `--cost-estimate-only` | Cost preview | Flag, no value |
| `--disable-weight-approval` | Skip weight approval | Flag |
| `--disable-human-feedback` | Skip feedback collection | Flag |
| `--focus-themes` | Analysis focus areas | `growth,profitability,risk` |
| `--geography` | Regional focus | `US`, `Europe`, `Asia` |
| `--lookback-months` | Historical analysis period | `12`, `24`, `36` |

### Analysis Types

1. **Quick Analysis** (`quick`):
   - 1 concurrent model
   - Core scoring components
   - ~30 seconds execution
   - Basic output formats

2. **Comprehensive Analysis** (`comprehensive`):
   - 3 concurrent models
   - All 14+ scoring components
   - Weight approval process
   - Full output suite
   - ~2-3 minutes execution

3. **Expert-Guided Analysis** (`expert_guided`):
   - 5 concurrent models
   - Human feedback integration
   - Advanced consensus algorithms
   - Research-grade outputs
   - ~5-10 minutes execution

### Weight Configuration

The system uses investment philosophy presets:

- **Growth Focus**: Emphasizes growth drivers and market expansion
- **Value Focus**: Prioritizes financial metrics and valuation
- **Quality Focus**: Emphasizes competitive moats and sustainability
- **Risk-Aware**: Balances all factors with risk considerations

```bash
# System will prompt for weight approval during analysis
# Example output:
# Current weights for Growth philosophy:
# - Brand/Monopoly: 8%
# - Barriers to Entry: 12%
# - Growth Drivers: 15%
# - Risk Factors: -8%
#
# Approve weights? (y/n/modify):
```

## Understanding Results

### Output File Structure

After analysis, you'll find files in these locations:

```
results/
├── multi_llm_analysis_COMPANY_TIMESTAMP.json    # Complete analysis data
├── multi_llm_scores_COMPANY_TIMESTAMP.csv       # Structured scores
├── multi_llm_result_COMPANY_TIMESTAMP.pkl       # Binary Python object
├── enhanced_metadata_COMPANY_TIMESTAMP.json     # Execution metadata
└── analysis_summary_COMPANY_TIMESTAMP.csv       # Executive summary

analysis_report_COMPANY_TIMESTAMP.md             # Human-readable report
```

### Key Result Components

#### 1. Final Composite Score
- **Scale**: 1.0 to 5.0
- **Interpretation**:
  - 4.5-5.0: STRONG BUY
  - 3.5-4.4: BUY
  - 2.5-3.4: HOLD
  - 1.5-2.4: SELL
  - 1.0-1.4: STRONG SELL

#### 2. Confidence Level
- **Scale**: 0% to 100%
- **Factors**: Data availability, model consensus, analysis depth
- **Interpretation**:
  - 80%+: High confidence
  - 60-79%: Moderate confidence
  - 40-59%: Low confidence
  - <40%: Very low confidence

#### 3. Individual Component Scores

**Core Competitive Moat (Higher Weight)**:
- Brand/Monopoly Power
- Barriers to Entry
- Economies of Scale
- Network Effects
- Switching Costs

**Strategic Analysis (Medium Weight)**:
- Competitive Differentiation
- Market Timing
- Management Quality
- Technology Moats

**Growth & Transformation (Medium Weight)**:
- Key Growth Drivers
- Transformation Potential
- Platform Expansion

**Risk Assessment (Negative Weight)**:
- Major Risk Factors
- Red Flags

### Interpreting Multi-LLM Results

#### Model Consensus
- **High Consensus** (>80% agreement): Reliable, clear investment thesis
- **Moderate Consensus** (60-80%): Generally aligned with some differences
- **Low Consensus** (<60%): Mixed signals, requires deeper investigation

#### Best Model Selection
The system automatically selects the "best" model based on:
- Number of scored components
- Average confidence levels
- Quality of justifications
- Presence of supporting sources

### Example Result Interpretation

```json
{
  "final_composite_score": 4.2,
  "final_confidence": 0.78,
  "recommendation": "BUY - Strong competitive position with solid growth prospects",
  "top_scores": {
    "moat_brand_monopoly": {"score": 4.8, "confidence": 0.85},
    "growth_drivers": {"score": 4.5, "confidence": 0.80},
    "barriers_to_entry": {"score": 4.3, "confidence": 0.75}
  },
  "risk_factors": {
    "major_risks": {"score": 2.1, "confidence": 0.70},
    "red_flags": {"score": 1.8, "confidence": 0.65}
  }
}
```

**Interpretation**: Strong BUY recommendation with high confidence. Excellent brand power and growth prospects, with manageable risks.

## Human Feedback Integration

### Overview
The human feedback system allows experts to improve model performance by:
1. Comparing model outputs
2. Selecting preferred analyses
3. Building training datasets
4. Improving future analysis quality

### Feedback Collection Process

#### 1. Expert Comparison Interface
```bash
# Enable human feedback (included in expert_guided analysis)
python run_enhanced_analysis.py --user-id analyst1 --company NVDA --analysis-type expert_guided
```

During analysis, you'll see:
```
=== EXPERT FEEDBACK INTERFACE ===
Comparing analyses for NVIDIA Corporation (NVDA)

Model A (mixtral-8x7b):
- Composite Score: 4.2/5.0
- Key Strength: Strong competitive moats
- Key Risk: Market saturation

Model B (llama-3-70b):
- Composite Score: 3.8/5.0
- Key Strength: Technology leadership
- Key Risk: Regulatory concerns

Which analysis is more accurate/useful? (A/B/Both/Neither):
```

#### 2. Feedback Database
All feedback is stored in `data/human_feedback.db`:

```sql
-- View feedback history
SELECT expert_id, company_ticker, preferred_model, feedback_score
FROM feedback_entries
ORDER BY timestamp DESC;

-- View model performance trends
SELECT model_name, AVG(performance_score) as avg_score
FROM model_performance
GROUP BY model_name;
```

#### 3. Training Dataset Generation
```bash
# Generate training dataset from feedback
python -c "
from engines.human_feedback_system import HumanFeedbackSystem
hfs = HumanFeedbackSystem()
dataset = hfs.generate_training_dataset()
print(f'Generated dataset with {len(dataset)} examples')
"
```

### Expert Profile Management

#### Creating Expert Profiles
```python
from engines.human_feedback_system import HumanFeedbackSystem

hfs = HumanFeedbackSystem()
hfs.create_expert_profile(
    expert_id="senior_analyst_1",
    name="Jane Smith",
    expertise_areas=["Technology", "Growth Stocks"],
    experience_years=15,
    credentials=["CFA", "MBA"],
    bias_adjustments={"growth_bias": 0.1, "tech_bias": 0.2}
)
```

#### Feedback Quality Scoring
The system tracks expert feedback quality:
- **Consistency**: How consistent the expert's preferences are
- **Accuracy**: How well their preferences correlate with market outcomes
- **Coverage**: Breadth of companies/sectors they provide feedback on

## Weight Management

### Investment Philosophy Presets

#### 1. Growth Philosophy
```json
{
  "brand_monopoly": 0.08,
  "barriers_to_entry": 0.12,
  "economies_of_scale": 0.08,
  "network_effects": 0.08,
  "switching_costs": 0.08,
  "growth_drivers": 0.15,
  "transformation_potential": 0.12,
  "platform_expansion": 0.10,
  "major_risk_factors": -0.05,
  "red_flags": -0.03
}
```

#### 2. Value Philosophy
```json
{
  "brand_monopoly": 0.12,
  "barriers_to_entry": 0.15,
  "economies_of_scale": 0.12,
  "switching_costs": 0.12,
  "competitive_differentiation": 0.10,
  "management_quality": 0.08,
  "major_risk_factors": -0.08,
  "red_flags": -0.06
}
```

### Custom Weight Configuration

#### Interactive Weight Approval
During analysis, you can modify weights:
```
Current weights (Growth philosophy):
1. Brand/Monopoly: 8%
2. Barriers to Entry: 12%
3. Growth Drivers: 15%
...

Options:
(a) Approve current weights
(m) Modify weights
(p) Switch philosophy
(c) Use custom weights

Choice:
```

#### Programmatic Weight Setting
```python
from engines.enhanced_scoring_system import WeightingScheme

# Create custom weights
custom_weights = WeightingScheme(
    brand_monopoly=0.15,
    barriers_to_entry=0.20,
    growth_drivers=0.10,
    major_risk_factors=-0.05
)

# Use in analysis
config = EnhancedAnalysisConfig(
    user_id="analyst1",
    company_ticker="NVDA",
    custom_weights=custom_weights
)
```

### Weight Impact Analysis
The system shows how weight changes affect scoring:
```
Weight Impact Analysis:
- Increasing Growth Drivers from 10% to 15%:
  * NVDA score: 3.8 → 4.1 (+0.3)
  * TSLA score: 4.2 → 4.4 (+0.2)

- Reducing Risk Factors from -8% to -12%:
  * NVDA score: 4.1 → 3.9 (-0.2)
  * TSLA score: 4.4 → 4.0 (-0.4)
```

## Troubleshooting

### Common Issues

#### 1. Model Detection Failures
```bash
# Check model availability
python -c "
from engines.llm_integration import LLMIntegration
llm = LLMIntegration()
print('Available models:', list(llm.models.keys()))
"
```

**Solution**: Verify API keys and model names in configuration.

#### 2. API Rate Limiting
**Symptoms**: `429 Too Many Requests` errors
**Solutions**:
- Use fewer concurrent models: `--models mixtral-8x7b`
- Add delays between requests
- Check API usage limits

#### 3. Unicode Encoding Errors
**Symptoms**: `'charmap' codec can't encode character`
**Solution**: Results are saved correctly, only display issue. Use JSON/CSV files.

#### 4. Memory Issues with Large Analyses
**Symptoms**: System slowdown or crashes
**Solutions**:
- Use `quick` analysis type
- Reduce concurrent models
- Clear old results: `rm -rf archive/old_results/*`

#### 5. Database Lock Errors
**Symptoms**: SQLite database locked
**Solution**:
```bash
# Reset human feedback database
rm data/human_feedback.db
python -c "from engines.human_feedback_system import HumanFeedbackSystem; HumanFeedbackSystem()"
```

### Performance Optimization

#### 1. Cost Management
```bash
# Always check costs first
python run_enhanced_analysis.py --user-id analyst1 --company NVDA --cost-estimate-only

# Use single model for testing
python run_enhanced_analysis.py --user-id analyst1 --company NVDA --models mixtral-8x7b
```

#### 2. Speed Optimization
- **Quick Analysis**: ~30 seconds, essential components only
- **Single Model**: ~45 seconds, full analysis with one model
- **Comprehensive**: ~2-3 minutes, multi-model consensus
- **Expert-Guided**: ~5-10 minutes, includes human feedback

#### 3. Storage Management
```bash
# Clean old results periodically
find archive/old_results -name "*.json" -mtime +30 -delete

# Compress large datasets
tar -czf backup_$(date +%Y%m%d).tar.gz archive/
```

## Advanced Features

### 1. Batch Processing
```python
# Process multiple companies programmatically
from engines.enhanced_analysis_controller import EnhancedAnalysisController, EnhancedAnalysisConfig

controller = EnhancedAnalysisController()
companies = ["NVDA", "TSLA", "AAPL", "MSFT"]

for ticker in companies:
    config = EnhancedAnalysisConfig(
        user_id="batch_analyst",
        company_ticker=ticker,
        analysis_type="quick"
    )
    result = controller.run_enhanced_analysis(config)
    print(f"{ticker}: {result.final_composite_score:.2f}/5.0")
```

### 2. Custom Scoring Components
```python
# Add new scoring component
from engines.enhanced_scoring_system import ScoreComponent

custom_component = ScoreComponent(
    name="esg_score",
    score=4.2,
    confidence=0.75,
    justification="Strong ESG practices across all dimensions",
    sources=["ESG report 2024", "Third-party ratings"],
    category="Sustainability"
)
```

### 3. API Integration
```python
# Web API for external integrations
from api.web_api import app

# Run web server
app.run(host='0.0.0.0', port=5000)

# API endpoints:
# POST /analyze - Run analysis
# GET /results/{analysis_id} - Get results
# POST /feedback - Submit human feedback
```

### 4. Research Integration
```python
# Integration with research databases
from engines.tools_integration import ToolsIntegration

tools = ToolsIntegration()

# Get real-time data
financial_data = tools.get_financial_metrics("NVDA")
news_sentiment = tools.get_news_sentiment("NVDA")
analyst_estimates = tools.get_analyst_estimates("NVDA")
```

### 5. Workflow Automation
```python
# LangGraph workflow integration
from engines.workflow_optimizer import WorkflowOptimizer

optimizer = WorkflowOptimizer()

# Create automated workflow
workflow = optimizer.create_analysis_workflow([
    "data_collection",
    "multi_llm_analysis",
    "consensus_generation",
    "human_feedback",
    "final_scoring"
])

result = optimizer.execute_workflow(workflow, company="NVDA")
```

## Conclusion

The Enhanced QualAgent system provides institutional-quality financial analysis through:

- **Multi-LLM Consensus**: Reduces single-model bias
- **Comprehensive Scoring**: 14+ components vs original 5
- **Human Feedback Integration**: Continuous improvement through expert input
- **Flexible Configuration**: Customizable weights and analysis depth
- **Professional Outputs**: Multiple formats for different use cases

For additional support or feature requests, refer to the technical documentation or create an issue in the project repository.

---

*Last updated: 2025-10-07*
*Version: Enhanced QualAgent v2.0*