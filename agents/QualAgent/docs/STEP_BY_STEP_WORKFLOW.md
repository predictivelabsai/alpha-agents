# QualAgent Enhanced Analysis - Step-by-Step Workflow Guide

## Quick Reference Card

### Essential Commands
```bash
# 1. Add company
python utils/simple_add_company.py

# 2. Check cost
python run_enhanced_analysis.py --user-id analyst1 --company TICKER --cost-estimate-only

# 3. Run analysis
python run_enhanced_analysis.py --user-id analyst1 --company TICKER

# 4. Check results
ls results/
cat analysis_report_TICKER_*.md
```

## Workflow 1: First-Time Setup (5 minutes)

### Step 1: Environment Setup
```bash
# Activate your environment
conda activate personal_CRM

# Navigate to QualAgent directory
cd "D:\Oxford\Extra\Finance_NLP\alpha-agents\agents\QualAgent"

# Verify dependencies
python run_tests.py
```

**Expected Output**: All tests should pass
```
Testing Enhanced QualAgent Analysis
========================================
Cost estimate: $0.0026
Models included: 1
Enhanced system is ready!
```

### Step 2: Configure API Keys
```bash
# Check if .env exists
ls .env

# If not, copy template
cp .env.template .env

# Edit .env file (use notepad, vim, or any editor)
notepad .env
```

**Required in .env**:
```
TOGETHER_API_KEY=your_together_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # Optional but recommended
```

### Step 3: Verify System Status
```bash
# Check model availability
python -c "
from engines.llm_integration import LLMIntegration
llm = LLMIntegration()
print('Available models:', len(llm.models))
print('Models:', list(llm.models.keys())[:3], '...')
"
```

**Expected Output**:
```
Available models: 7
Models: ['mixtral-8x7b', 'llama-3-70b', 'qwen2-72b'] ...
```

## Workflow 2: Adding a New Company (2 minutes)

### Method A: Interactive Addition (Recommended for beginners)
```bash
python utils/simple_add_company.py
```

**Follow the prompts**:
```
Enter company ticker (e.g., AAPL): MSFT
Enter company name: Microsoft Corporation
Enter subsector: Cloud Software
Enter market cap (Large Cap/Mid Cap/Small Cap): Large Cap
Enter geography (US/Europe/Asia/Other): US
```

### Method B: Quick JSON Edit (For experienced users)
```bash
# Edit companies file directly
notepad data/companies.json
```

**Add entry**:
```json
{
  "MSFT": {
    "company_name": "Microsoft Corporation",
    "ticker": "MSFT",
    "subsector": "Cloud Software",
    "market_cap": "Large Cap",
    "geography": "US",
    "added_date": "2025-10-07",
    "status": "active"
  }
}
```

### Step 3: Verify Addition
```bash
# Check if company was added
python -c "
from models.json_data_manager import JSONDataManager
dm = JSONDataManager()
companies = dm.load_companies()
print('Available companies:', len(companies))
print('Last 3:', list(companies.keys())[-3:])
"
```

## Workflow 3: Running Your First Analysis (3-5 minutes)

### Step 1: Cost Check (Always do this first!)
```bash
python run_enhanced_analysis.py --user-id analyst1 --company MSFT --cost-estimate-only
```

**Expected Output**:
```
Cost estimate for MSFT: $0.0180
Enhanced system is ready!
```

**Cost Guidelines**:
- Under $0.05: Proceed freely
- $0.05-$0.20: Normal cost
- Over $0.20: Consider using fewer models

### Step 2: Quick Test Analysis
```bash
# Run with single model first (fastest, cheapest)
python run_enhanced_analysis.py --user-id analyst1 --company MSFT --analysis-type quick --models mixtral-8x7b
```

**What happens**:
1. System initializes (10-15 seconds)
2. Weight approval prompt appears
3. Analysis runs (30-60 seconds)
4. Results are saved

**Watch for**:
```
INFO - Starting enhanced analysis for MSFT
INFO - Starting weight approval for analyst1
```

### Step 3: Respond to Weight Approval
```
Current weights for MSFT analysis:
Investment Philosophy: Balanced

Core Competitive Moats:
- Brand/Monopoly: 10%
- Barriers to Entry: 15%
- Economies of Scale: 10%
...

Do you approve these weights? (y/n/modify): y
```

**Options**:
- `y`: Approve and continue
- `n`: Use different philosophy
- `modify`: Custom weight adjustment

### Step 4: Monitor Progress
```
INFO - Found 5 enabled models: ['mixtral-8x7b', 'llama-3-70b', ...]
INFO - Starting analysis with mixtral-8x7b
INFO - LLM call successful: mixtral-8x7b in 9.19s
INFO - Enhanced analysis completed for MSFT in 28.53s
```

**Success Indicators**:
- "Enhanced analysis completed"
- "Saved multi-LLM results in 3 formats"
- Files appear in results/

## Workflow 4: Understanding Your Results (5 minutes)

### Step 1: Quick Summary Check
```bash
# Find your latest analysis report
ls analysis_report_*MSFT*.md

# View the summary
head -20 analysis_report_MSFT_*.md
```

**Key Info to Look For**:
```markdown
## Executive Summary
- **Final Composite Score:** 4.2/5.0
- **Confidence Level:** 78%
- **Recommendation:** BUY - Strong competitive position
```

### Step 2: Detailed Results Exploration
```bash
# Check all generated files
ls results/*MSFT*

# View structured scores
head -10 results/multi_llm_scores_MSFT_*.csv
```

**File Types**:
- `.json`: Complete analysis data
- `.csv`: Structured scores (Excel-friendly)
- `.pkl`: Python binary (for programming)
- `.md`: Human-readable report

### Step 3: Score Interpretation

**Composite Score Scale**:
- 4.5-5.0: STRONG BUY ⭐⭐⭐⭐⭐
- 3.5-4.4: BUY ⭐⭐⭐⭐
- 2.5-3.4: HOLD ⭐⭐⭐
- 1.5-2.4: SELL ⭐⭐
- 1.0-1.4: STRONG SELL ⭐

**Confidence Levels**:
- 80%+: High confidence (reliable)
- 60-79%: Moderate confidence (good)
- 40-59%: Low confidence (uncertain)
- <40%: Very low confidence (investigate)

### Step 4: Component Analysis
```bash
# View detailed JSON for component scores
python -c "
import json
with open('results/multi_llm_analysis_MSFT_*.json') as f:
    data = json.load(f)
    print('Top scores:')
    for comp, score in data.get('consensus_scores', {}).items():
        if score.get('score', 0) > 4.0:
            print(f'  {comp}: {score[\"score\"]:.1f}/5.0')
"
```

## Workflow 5: Multi-LLM Comprehensive Analysis (5-10 minutes)

### Step 1: Full Multi-LLM Analysis
```bash
# Run comprehensive analysis with all models
python run_enhanced_analysis.py --user-id analyst1 --company MSFT --analysis-type comprehensive
```

**What's Different**:
- 3 concurrent models (vs 1 in quick)
- All 14+ scoring components
- Model consensus analysis
- Detailed confidence metrics

### Step 2: Compare Model Results
```bash
# Check which models succeeded
grep "LLM call successful" logs/latest.log

# View model comparison in JSON
python -c "
import json
with open('results/multi_llm_analysis_MSFT_*.json') as f:
    data = json.load(f)
    print('Model Performance:')
    for model, result in data.get('llm_results', {}).items():
        print(f'  {model}: Success={\"error\" not in result}')
"
```

### Step 3: Consensus Analysis
**Look for**:
- High consensus (>80%): Strong agreement across models
- Moderate consensus (60-80%): General agreement
- Low consensus (<60%): Mixed signals, needs investigation

## Workflow 6: Human Feedback Integration (10-15 minutes)

### Step 1: Enable Expert Feedback
```bash
# Run expert-guided analysis
python run_enhanced_analysis.py --user-id expert1 --company MSFT --analysis-type expert_guided
```

### Step 2: Provide Feedback
**During analysis, you'll see**:
```
=== EXPERT FEEDBACK INTERFACE ===
Comparing analyses for Microsoft Corporation (MSFT)

Model A (mixtral-8x7b):
- Composite Score: 4.2/5.0
- Key Strengths: Strong competitive moats, cloud dominance
- Key Risks: Regulatory scrutiny, market saturation

Model B (llama-3-70b):
- Composite Score: 3.9/5.0
- Key Strengths: Technology leadership, enterprise relationships
- Key Risks: Competition from AWS, execution risk

Which analysis provides better insights? (A/B/Both/Neither): A
Rate the quality of your preferred analysis (1-5): 4
```

### Step 3: View Feedback History
```bash
# Check feedback database
python -c "
from engines.human_feedback_system import HumanFeedbackSystem
hfs = HumanFeedbackSystem()
recent = hfs.get_recent_feedback(limit=5)
print('Recent feedback:')
for fb in recent:
    print(f'  {fb[\"company_ticker\"]}: preferred {fb[\"preferred_model\"]}')
"
```

### Step 4: Training Dataset Generation
```bash
# Generate training data from feedback
python -c "
from engines.human_feedback_system import HumanFeedbackSystem
hfs = HumanFeedbackSystem()
dataset = hfs.generate_training_dataset()
print(f'Training dataset: {len(dataset)} examples')
print('Sample entry:', dataset[0].keys() if dataset else 'No data')
"
```

## Workflow 7: Batch Processing Multiple Companies (15-30 minutes)

### Step 1: Prepare Company List
```bash
# Add multiple companies first
python utils/simple_add_company.py  # Repeat for each company

# Or edit companies.json with multiple entries
```

### Step 2: Estimate Batch Costs
```bash
# Check costs for multiple companies
for ticker in MSFT AAPL NVDA TSLA; do
    echo "Cost for $ticker:"
    python run_enhanced_analysis.py --user-id batch1 --company $ticker --cost-estimate-only
done
```

### Step 3: Run Batch Analysis
```bash
# Single command for multiple companies
python run_enhanced_analysis.py --user-id batch1 --companies MSFT,AAPL,NVDA,TSLA --analysis-type quick
```

### Step 4: Compare Results
```bash
# Generate comparison report
python -c "
import json, glob
results = {}
for file in glob.glob('results/multi_llm_analysis_*_*.json'):
    with open(file) as f:
        data = json.load(f)
        ticker = data.get('company', {}).get('ticker', 'Unknown')
        score = data.get('composite_score', 0)
        results[ticker] = score

print('Batch Analysis Results:')
for ticker, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f'  {ticker}: {score:.2f}/5.0')
"
```

## Workflow 8: Advanced Weight Management (10-15 minutes)

### Step 1: Explore Investment Philosophies
```bash
# View available philosophies
python -c "
from engines.weight_approval_system import WeightApprovalSystem
was = WeightApprovalSystem()
philosophies = was.get_available_philosophies()
print('Investment Philosophies:', list(philosophies.keys()))
"
```

### Step 2: Custom Weight Configuration
```bash
# Run analysis with custom weight selection
python run_enhanced_analysis.py --user-id custom1 --company MSFT
```

**During weight approval**:
```
Current philosophy: Balanced
Options:
(g) Growth-focused
(v) Value-focused
(q) Quality-focused
(r) Risk-aware
(c) Custom weights

Choice: g  # Select Growth-focused
```

### Step 3: Weight Impact Analysis
**System shows**:
```
Weight Impact Analysis for Growth Philosophy:
- Increasing Growth Drivers: 10% → 15%
- MSFT score change: 4.1 → 4.3 (+0.2)
- Technology weight increase affects scoring

Approve? (y/n): y
```

### Step 4: Save Weight Preferences
```bash
# View saved weight history
cat data/weight_approval_history.json
```

## Workflow 9: Troubleshooting Common Issues (5-10 minutes)

### Issue 1: Models Not Found
**Symptom**: "Found 0 enabled models"
**Solution**:
```bash
# Check API keys
python -c "
import os
print('TOGETHER_API_KEY set:', bool(os.getenv('TOGETHER_API_KEY')))
print('OPENAI_API_KEY set:', bool(os.getenv('OPENAI_API_KEY')))
"

# Test API connection
python -c "
from engines.llm_integration import LLMIntegration
llm = LLMIntegration()
validation = llm.validate_api_keys()
print('API validation:', validation)
"
```

### Issue 2: High Costs
**Symptom**: Cost estimate >$0.50
**Solutions**:
```bash
# Use fewer models
--models mixtral-8x7b

# Use quick analysis
--analysis-type quick

# Single company only
--company MSFT  # not --companies
```

### Issue 3: Analysis Fails
**Symptom**: "Analysis failed" errors
**Check**:
```bash
# View recent logs
tail -20 logs/latest.log

# Test basic functionality
python run_tests.py
```

### Issue 4: Unicode Display Issues
**Symptom**: Character encoding errors in console
**Solution**: Results are saved correctly, use files:
```bash
# View results in files instead of console
cat analysis_report_MSFT_*.md
notepad results/multi_llm_analysis_MSFT_*.json
```

## Workflow 10: Result Analysis Deep Dive (15-20 minutes)

### Step 1: Score Component Analysis
```bash
# Extract all component scores
python -c "
import json
with open('results/multi_llm_analysis_MSFT_*.json') as f:
    data = json.load(f)
    scores = data.get('consensus_scores', {})

    print('STRENGTHS (Score >= 4.0):')
    strong = {k: v for k, v in scores.items() if v.get('score', 0) >= 4.0}
    for comp, details in strong.items():
        print(f'  {comp}: {details[\"score\"]:.1f} (conf: {details[\"confidence\"]:.0%})')

    print('\nWEAKNESSES (Score <= 2.5):')
    weak = {k: v for k, v in scores.items() if v.get('score', 0) <= 2.5}
    for comp, details in weak.items():
        print(f'  {comp}: {details[\"score\"]:.1f} (conf: {details[\"confidence\"]:.0%})')
"
```

### Step 2: Confidence Analysis
```bash
# Analyze confidence patterns
python -c "
import json
with open('results/multi_llm_analysis_MSFT_*.json') as f:
    data = json.load(f)
    scores = data.get('consensus_scores', {})

    confidences = [v.get('confidence', 0) for v in scores.values()]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0

    print(f'Average confidence: {avg_conf:.1%}')
    print(f'High confidence components (>80%): {sum(1 for c in confidences if c > 0.8)}')
    print(f'Low confidence components (<60%): {sum(1 for c in confidences if c < 0.6)}')
"
```

### Step 3: Model Consensus Analysis
```bash
# Check model agreement
python -c "
import json
with open('results/multi_llm_analysis_MSFT_*.json') as f:
    data = json.load(f)

    print('Model Performance Summary:')
    print(f'  Best model: {data.get(\"best_model_recommendation\", \"Unknown\")}')
    print(f'  Total cost: ${data.get(\"total_cost_usd\", 0):.4f}')
    print(f'  Processing time: {data.get(\"total_time_seconds\", 0):.1f}s')

    models_used = len(data.get('llm_results', {}))
    print(f'  Successful models: {models_used}/5')
"
```

### Step 4: Investment Thesis Generation
```bash
# Generate investment summary
python -c "
import json
with open('results/multi_llm_analysis_MSFT_*.json') as f:
    data = json.load(f)

    score = data.get('composite_score', 0)
    conf = data.get('composite_confidence', 0)

    if score >= 4.0:
        thesis = 'STRONG INVESTMENT CASE'
    elif score >= 3.5:
        thesis = 'POSITIVE INVESTMENT CASE'
    elif score >= 2.5:
        thesis = 'NEUTRAL/HOLD POSITION'
    else:
        thesis = 'WEAK INVESTMENT CASE'

    print(f'Investment Thesis: {thesis}')
    print(f'Conviction Level: {\"High\" if conf > 0.7 else \"Medium\" if conf > 0.5 else \"Low\"}')
    print(f'Score: {score:.2f}/5.0 (Confidence: {conf:.1%})')
"
```

## Quick Reference: File Locations

### Input Files
- **Companies**: `data/companies.json`
- **Prompts**: `prompts/TechQual_Enhanced_WithTools_v2.json`
- **Config**: `.env`

### Output Files
- **Reports**: `analysis_report_TICKER_TIMESTAMP.md`
- **Full Data**: `results/multi_llm_analysis_TICKER_TIMESTAMP.json`
- **Scores**: `results/multi_llm_scores_TICKER_TIMESTAMP.csv`
- **Binary**: `results/multi_llm_result_TICKER_TIMESTAMP.pkl`
- **Metadata**: `results/enhanced_metadata_TICKER_TIMESTAMP.json`

### System Files
- **Feedback DB**: `data/human_feedback.db`
- **Weight History**: `data/weight_approval_history.json`
- **Logs**: System logs in console output

---

*Quick Start Summary: Setup → Add Company → Check Cost → Run Analysis → Review Results*