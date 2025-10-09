# QualAgent Enhanced User Guide

**Advanced Multi-LLM Financial Analysis System with Human Feedback Integration**

QualAgent Enhanced automatically runs analysis across 5 LLMs simultaneously, provides comprehensive scoring for all analysis components, enables expert feedback collection, and saves results in multiple formats for advanced analytics.

---

## 📋 Table of Contents

1. [What's New in Enhanced Version](#whats-new)
2. [Quick Start](#quick-start)
3. [Installation & Setup](#installation--setup)
4. [Enhanced Features](#enhanced-features)
5. [Command Line Usage](#command-line-usage)
6. [Interactive Weight Configuration](#weight-configuration)
7. [Human Feedback System](#human-feedback)
8. [Multi-Format Output](#multi-format-output)
9. [Cost Management & Optimization](#cost-management)
10. [Advanced Workflows](#advanced-workflows)
11. [API Reference](#api-reference)
12. [Troubleshooting](#troubleshooting)

---

## 🚀 What's New in Enhanced Version {#whats-new}

### **Multi-LLM Analysis Engine**
- **5 LLM Models**: Runs analysis across mixtral-8x7b, llama-3-70b, qwen2-72b, gemma-2-27b, and latest Llama models
- **Concurrent Execution**: Analyzes with multiple models simultaneously for speed
- **Consensus Scoring**: Intelligent consensus from all model outputs
- **Best Model Selection**: Automatically identifies the highest-quality analysis

### **Comprehensive Scoring System**
- **Expanded Components**: Scores ALL analysis elements, not just competitive moats
- **14 Score Categories**: Including growth drivers, risk factors, competitive positioning, transformation potential
- **Confidence Adjustments**: Each score includes confidence levels that affect final composite
- **Weighted Composite**: User-configurable weights for personalized investment philosophy

### **Human Feedback Integration**
- **Expert Selection**: Present model results to experts for quality rating
- **Training Dataset**: Automatically builds training data from expert selections
- **Performance Tracking**: Tracks which models perform best according to experts
- **Feedback Analytics**: Insights on model performance and expert preferences

### **Enhanced Data Management**
- **Multiple Formats**: JSON, CSV, and PKL output for different use cases
- **Structured Storage**: Organized results with metadata and lineage tracking
- **Export Options**: Easy export to spreadsheets and analysis tools
- **Version Control**: Track analysis versions and configurations

### **Advanced Workflow Features**
- **Weight Approval**: Interactive system for reviewing and customizing scoring weights
- **Cost Estimation**: Detailed cost breakdown before running analysis
- **Workflow Optimization**: LangGraph integration for intelligent execution planning
- **User Preferences**: Remembers user settings and preferences across sessions

---

## 🚀 Quick Start

### 1. Installation
```bash
# Navigate to QualAgent directory
cd D:\Oxford\Extra\Finance_NLP\alpha-agents\agents\QualAgent

# Install enhanced dependencies
pip install -r requirements_enhanced.txt

# Create .env file with API keys
cp .env.template .env
# Edit .env with your API keys
```

### 2. First Enhanced Analysis
```bash
# Run enhanced analysis with all features
python run_enhanced_analysis.py --user-id analyst1 --company NVDA --interactive-weights

# Quick analysis with defaults
python run_enhanced_analysis.py --user-id analyst1 --company AAPL

# Batch analysis with expert feedback
python run_enhanced_analysis.py --user-id analyst1 --companies NVDA,AAPL,MSFT --expert-id expert1 --enable-feedback
```

### 3. View Results
```bash
# Check generated files
ls -la *.json *.csv *.pkl *.md

# View summary
cat analysis_summary_NVDA_*.csv
```

---

## 💻 Installation & Setup

### Prerequisites
- **Python 3.8+**
- **TogetherAI API Key** (primary - supports 5 models)
- **OpenAI API Key** (backup)
- **Optional**: Tavily, Polygon, Exa API keys for enhanced research

### Step-by-Step Enhanced Installation

1. **Install Core Dependencies**
   ```bash
   pip install pandas numpy requests python-dotenv openai flask pytest jupyter
   ```

2. **Install Enhanced Features**
   ```bash
   pip install langchain langgraph langchain-core scipy scikit-learn sqlalchemy
   ```

3. **Install Optional Features**
   ```bash
   pip install plotly streamlit psutil memory-profiler black flake8
   ```

4. **Or Install Everything**
   ```bash
   pip install -r requirements_enhanced.txt
   ```

5. **Configure API Keys**
   ```bash
   # Copy template
   cp .env.template .env

   # Edit .env file
   TOGETHER_API_KEY=your_together_api_key_here
   OPENAI_API_KEY=your_openai_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here  # Optional
   POLYGON_API_KEY=your_polygon_api_key_here  # Optional
   EXA_API_KEY=your_exa_api_key_here  # Optional
   ```

6. **Test Installation**
   ```bash
   python run_enhanced_analysis.py --help
   ```

---

## ⭐ Enhanced Features

### **1. Multi-LLM Analysis Engine**

**Simultaneous 5-Model Analysis:**
```bash
# Runs across all 5 models by default
python run_enhanced_analysis.py --user-id analyst1 --company NVDA

# Specify specific models
python run_enhanced_analysis.py --user-id analyst1 --company AAPL --models mixtral-8x7b,llama-3-70b

# Control concurrency
python run_enhanced_analysis.py --user-id analyst1 --company MSFT --max-concurrent 2
```

**Model Selection & Consensus:**
- Automatic quality assessment of each model's output
- Weighted consensus scoring based on model reliability
- Best model recommendation with quality metrics
- Detailed comparison of model performance

### **2. Comprehensive Scoring System**

**All Components Scored (1-5 scale with confidence):**

**Core Competitive Moats:**
- Brand Monopoly
- Barriers to Entry
- Economies of Scale
- Network Effects
- Switching Costs

**Strategic Analysis:**
- Competitive Differentiation
- Technology Moats
- Market Timing
- Management Quality

**Growth & Innovation:**
- Key Growth Drivers
- Transformation Potential
- Platform Expansion Opportunities

**Risk Assessment:**
- Major Risk Factors (negative scoring)
- Red Flags (negative scoring)

**Competitive Intelligence:**
- Competitive Positioning
- Market Share Dynamics

### **3. Weight Approval System**

**Interactive Weight Configuration:**
```bash
# Interactive weight setup
python run_enhanced_analysis.py --user-id analyst1 --company NVDA --interactive-weights
```

**Investment Philosophy Presets:**
- **Growth Focus**: Emphasizes growth drivers and innovation (20% growth weights)
- **Value Focus**: Emphasizes competitive moats and barriers (25% moat weights)
- **Quality Focus**: Emphasizes differentiation and management (20% quality weights)
- **Risk-Aware**: Emphasizes risk factors and defensive characteristics

**Custom Weight Adjustment:**
- Modify any individual weight from 0.0-1.0
- Automatic normalization to ensure weights sum correctly
- Historical preference tracking and suggestions
- Impact analysis showing how weight changes affect scoring

### **4. Human Feedback Integration**

**Expert Feedback Collection:**
```bash
# Enable expert feedback
python run_enhanced_analysis.py --user-id analyst1 --company NVDA --expert-id expert1 --enable-feedback
```

**Feedback Features:**
- Model comparison presentation
- Quality rating collection (1-5 scale)
- Best model selection tracking
- Expert reasoning capture
- Training dataset generation

**Performance Analytics:**
- Model selection rates by experts
- Average quality ratings per model
- Expert preference patterns
- Feedback-based model improvement suggestions

### **5. Enhanced Data Management**

**Multiple Output Formats:**
- **JSON**: Complete detailed results with metadata
- **CSV**: Structured scores for spreadsheet analysis
- **PKL**: Python objects for programmatic use
- **Markdown**: Human-readable reports

**Data Organization:**
```
results/
├── multi_llm_analysis_NVDA_1696789123.json     # Complete results
├── multi_llm_scores_NVDA_1696789123.csv        # Scores summary
├── multi_llm_result_NVDA_1696789123.pkl        # Python objects
├── enhanced_metadata_NVDA_1696789123.json      # Enhanced metadata
├── analysis_summary_NVDA_1696789123.csv        # Quick summary
└── analysis_report_NVDA_1696789123.md          # Full report
```

---

## 📖 Command Line Usage

### **Enhanced Analysis Commands**

#### **Single Company Analysis**
```bash
# Comprehensive analysis with all features
python run_enhanced_analysis.py --user-id analyst1 --company NVDA \
  --analysis-type comprehensive --enable-feedback --expert-id expert1

# Quick analysis for screening
python run_enhanced_analysis.py --user-id analyst1 --company AAPL \
  --analysis-type quick --max-concurrent 2

# Themed analysis
python run_enhanced_analysis.py --user-id analyst1 --company MSFT \
  --themes "AI strategy,Cloud competition,Market expansion"
```

#### **Batch Analysis**
```bash
# Multiple companies
python run_enhanced_analysis.py --user-id analyst1 \
  --companies NVDA,AAPL,MSFT,GOOGL,AMZN

# Batch with expert feedback
python run_enhanced_analysis.py --user-id analyst1 --batch \
  --companies NVDA,AAPL,MSFT --expert-id expert1 --enable-feedback

# Large batch with cost control
python run_enhanced_analysis.py --user-id analyst1 \
  --companies NVDA,AAPL,MSFT,GOOGL,AMZN,META,TSLA \
  --max-concurrent 2 --analysis-type quick
```

#### **Cost Estimation**
```bash
# Estimate costs before running
python run_enhanced_analysis.py --user-id analyst1 --company NVDA --cost-estimate-only

# Batch cost estimation
python run_enhanced_analysis.py --user-id analyst1 \
  --companies NVDA,AAPL,MSFT --cost-estimate-only
```

### **Advanced Options**

```bash
# All available options
--user-id USER_ID              # Required: User identification
--company TICKER               # Single company analysis
--companies TICKER1,TICKER2    # Multiple companies
--batch                        # Enable batch mode

--analysis-type TYPE           # quick | comprehensive | expert_guided
--models MODEL1,MODEL2         # Specific models to use
--themes "theme1,theme2"       # Analysis focus themes
--geographies "US,EU,Global"   # Geographic focus

--enable-weights               # Enable weight approval (default: true)
--enable-feedback              # Enable human feedback collection
--expert-id EXPERT_ID          # Expert ID for feedback
--max-concurrent N             # Max concurrent models (default: 3)

--interactive-weights          # Interactive weight configuration
--cost-estimate-only           # Show cost estimate only
```

---

## ⚖️ Interactive Weight Configuration {#weight-configuration}

### **Weight Categories & Default Values**

```python
# Core Competitive Moats (Higher Weight - Main Framework)
Barriers to Entry: 0.150      # Technical/regulatory barriers
Brand Monopoly: 0.100         # Brand strength and loyalty
Economies of Scale: 0.100     # Cost advantages from scale
Network Effects: 0.100        # Value from user network
Switching Costs: 0.100        # Customer switching barriers

# Strategic Factors (Medium Weight)
Competitive Differentiation: 0.080  # Unique value propositions
Technology Moats: 0.080            # Technical advantages
Market Timing: 0.060               # Market readiness
Management Quality: 0.050          # Leadership effectiveness

# Growth & Innovation (Lower Weight - More Speculative)
Key Growth Drivers: 0.050          # Growth catalysts
Transformation Potential: 0.040    # Business model evolution
Platform Expansion: 0.030          # Expansion opportunities

# Risk Factors (Negative Weight - Reduce Score)
Major Risk Factors: -0.060         # Significant risks
Red Flags: -0.040                  # Warning signals
```

### **Interactive Configuration Process**

1. **Review Default Weights**
   ```
   Current default weights:
   Core Competitive Moats (52% total weight):
   • Barriers to Entry: 0.150
   • Brand Monopoly: 0.100
   • Economies of Scale: 0.100
   • Network Effects: 0.100
   • Switching Costs: 0.100
   ```

2. **Choose Configuration Method**
   ```
   Weight Configuration Options:
   1. Use default weights
   2. Apply investment focus (Growth/Value/Quality/Risk)
   3. Custom weight adjustment
   ```

3. **Apply Investment Focus** (Option 2)
   ```
   Investment Focus Options:
   1. Growth Focus - Emphasize growth drivers and innovation
   2. Value Focus - Emphasize competitive moats and barriers
   3. Quality Focus - Emphasize management and differentiation
   4. Risk-Aware - Emphasize risk factors and red flags
   ```

4. **Custom Adjustment** (Option 3)
   ```
   Custom Weight Adjustment
   Enter new weights (0.0-1.0) or press Enter to keep default:
   Barriers to Entry [0.150]: 0.200
   Brand Monopoly [0.100]:
   Key Growth Drivers [0.050]: 0.080
   ```

### **Weight Impact Analysis**

The system shows how weight changes affect scoring:

```
Impact Examples with Current Weights:

High Moat Company (Strong moats, average growth):
• Barriers: 5, Brand: 4, Growth: 3, Risk: 2
• Estimated Score: 4.2/5.0

High Growth Company (Moderate moats, strong growth):
• Barriers: 3, Brand: 3, Growth: 5, Risk: 2
• Estimated Score: 3.8/5.0

Risky Company (Good potential, high risks):
• Barriers: 4, Brand: 3, Growth: 4, Risk: 4
• Estimated Score: 3.3/5.0
```

---

## 👥 Human Feedback System {#human-feedback}

### **Expert Feedback Workflow**

1. **Enable Feedback Collection**
   ```bash
   python run_enhanced_analysis.py --user-id analyst1 --company NVDA \
     --expert-id expert1 --enable-feedback
   ```

2. **Model Comparison Presentation**
   ```
   Model Comparison for NVDA:

   Model: mixtral-8x7b
   • Composite Score: 4.2/5.0
   • Key Scores: Barriers(5), Growth(4), Risk(2)
   • Analysis Quality: 85%
   • Cost: $0.008, Time: 25s

   Model: llama-3-70b
   • Composite Score: 4.0/5.0
   • Key Scores: Barriers(4), Growth(5), Risk(2)
   • Analysis Quality: 82%
   • Cost: $0.012, Time: 30s
   ```

3. **Expert Selection Interface**
   ```
   Expert Selection Options:
   1. Select best overall model
   2. Rate model quality (1-5 scale)
   3. Rank models by preference
   4. Provide reasoning and comments
   ```

### **Feedback Data Collection**

**Feedback Types:**
- **Model Selection**: Which model provided the best analysis
- **Quality Ratings**: 1-5 rating for each model's output quality
- **Model Rankings**: Ordered preference of all models
- **Score Adjustments**: Expert modifications to specific scores
- **Reasoning**: Detailed explanation of expert decisions

**Training Dataset Generation:**
```json
{
  "feedback_id": "fb_1696789456",
  "expert_id": "expert1",
  "company_ticker": "NVDA",
  "selected_model": "mixtral-8x7b",
  "quality_ratings": {
    "mixtral-8x7b": 5,
    "llama-3-70b": 4,
    "qwen2-72b": 4
  },
  "reasoning": "Mixtral provided most comprehensive competitive analysis"
}
```

### **Performance Analytics**

**Model Performance Metrics:**
```python
# Get model performance report
from engines.human_feedback_system import HumanFeedbackSystem
feedback_system = HumanFeedbackSystem()
performance = feedback_system.get_model_performance_report()

# Results:
{
  "mixtral-8x7b": {
    "selection_rate": 0.65,        # Selected 65% of time
    "average_ranking": 1.3,        # Average rank (1=best)
    "average_quality": 4.2,        # Average quality rating
    "expert_preference_score": 0.85,  # Composite preference
    "feedback_count": 23
  }
}
```

**Feedback Insights:**
- Expert activity patterns
- Model preference trends
- Common score adjustments
- Quality improvement suggestions

---

## 💾 Multi-Format Output {#multi-format-output}

### **Output File Types**

#### **1. JSON Format** (Complete Results)
```json
{
  "metadata": {
    "timestamp": 1696789123,
    "company_ticker": "NVDA",
    "analysis_type": "multi_llm",
    "models_used": ["mixtral-8x7b", "llama-3-70b", "qwen2-72b"]
  },
  "composite_score": {
    "score": 4.2,
    "confidence": 0.88,
    "components_count": 14
  },
  "consensus_scores": {
    "moat_barriers_to_entry": {
      "score": 4.8,
      "confidence": 0.92,
      "justification": "Strong patent portfolio and technical complexity"
    }
  },
  "individual_model_results": {...},
  "execution_metadata": {...}
}
```

#### **2. CSV Format** (Structured Scores)
```csv
company_ticker,score_component,score,confidence,category,justification
NVDA,moat_barriers_to_entry,4.8,0.92,competitive_moat,"Strong patent portfolio..."
NVDA,key_growth_drivers,4.5,0.85,strategic_insights,"AI market expansion..."
NVDA,COMPOSITE_SCORE,4.2,0.88,composite,"Weighted composite from 14 components"
```

#### **3. PKL Format** (Python Objects)
```python
import pickle

# Load complete result object
with open('multi_llm_result_NVDA_1696789123.pkl', 'rb') as f:
    result = pickle.load(f)

# Access all data programmatically
print(f"Score: {result.composite_score}")
print(f"Best Model: {result.best_model_recommendation}")
for score_name, score_comp in result.consensus_scores.items():
    print(f"{score_name}: {score_comp.score}")
```

#### **4. Markdown Reports** (Human-Readable)
```markdown
# QualAgent Enhanced Analysis Report

## Company Overview
- **Ticker:** NVDA
- **Final Score:** 4.20/5.0
- **Confidence:** 88%
- **Recommendation:** STRONG BUY

## Multi-LLM Results
- **Models Used:** 5
- **Best Model:** mixtral-8x7b
- **Total Cost:** $0.045
```

### **File Organization**

```
D:/Oxford/Extra/Finance_NLP/alpha-agents/agents/QualAgent/
├── results/                           # Enhanced results directory
│   ├── multi_llm_analysis_NVDA_*.json     # Complete results
│   ├── multi_llm_scores_NVDA_*.csv        # Score summaries
│   ├── multi_llm_result_NVDA_*.pkl        # Python objects
│   ├── enhanced_metadata_NVDA_*.json      # Enhanced metadata
│   ├── analysis_summary_NVDA_*.csv        # Quick summaries
│   └── analysis_report_NVDA_*.md          # Full reports
├── data/                              # Enhanced data storage
│   ├── human_feedback.db                  # SQLite feedback database
│   ├── training_dataset.json             # Expert training data
│   ├── weight_approval_history.json      # Weight approval history
│   └── user_preferences_*.json           # User preference files
└── analysis_results_*.json           # Legacy results (preserved)
```

---

## 💰 Cost Management & Optimization {#cost-management}

### **Cost Estimation**

**Before Analysis:**
```bash
# Estimate single company cost
python run_enhanced_analysis.py --user-id analyst1 --company NVDA --cost-estimate-only

# Output:
Cost estimate for NVDA: $0.0485
  - Multi-LLM analysis: $0.0441
  - Processing overhead: $0.0044
  - Models included: 5
  - Per model average: $0.0088
```

**Batch Cost Estimation:**
```bash
# Estimate batch cost
python run_enhanced_analysis.py --user-id analyst1 \
  --companies NVDA,AAPL,MSFT --cost-estimate-only

# Output:
Batch Analysis Cost Estimate: $0.1455
  - Companies: 3
  - Average per company: $0.0485
```

### **Cost Control Strategies**

**1. Model Selection:**
```bash
# Use fewer models for screening
python run_enhanced_analysis.py --user-id analyst1 --company NVDA \
  --models mixtral-8x7b,llama-3-70b

# Quick analysis mode (reduced token usage)
python run_enhanced_analysis.py --user-id analyst1 --company NVDA \
  --analysis-type quick
```

**2. Concurrent Execution Control:**
```bash
# Reduce concurrent models to manage API rate limits
python run_enhanced_analysis.py --user-id analyst1 --company NVDA \
  --max-concurrent 2
```

**3. Focused Analysis:**
```bash
# Use specific themes to reduce analysis scope
python run_enhanced_analysis.py --user-id analyst1 --company NVDA \
  --themes "Competitive moats,Growth drivers"
```

### **Cost Breakdown by Model**

**TogetherAI Models (Primary - Recommended):**
- `mixtral-8x7b`: $0.0006/1K tokens (excellent value)
- `llama-3-70b`: $0.0009/1K tokens (best reasoning)
- `qwen2-72b`: $0.0009/1K tokens (strong analytics)
- `gemma-2-27b`: $0.0004/1K tokens (efficient)

**OpenAI Models (Backup):**
- `gpt-4o-mini`: $0.00015/1K tokens (cheapest GPT)
- `gpt-4o`: $0.005/1K tokens (premium quality)

**Typical Usage:**
- Single comprehensive analysis: 4,000-6,000 tokens per model
- Quick screening analysis: 2,000-3,000 tokens per model
- 5-model comprehensive analysis: ~$0.04-$0.06 per company

---

## 🔧 Advanced Workflows {#advanced-workflows}

### **LangGraph Workflow Optimization**

**Intelligent Execution Planning:**
```python
from engines.workflow_optimizer import WorkflowOptimizer

optimizer = WorkflowOptimizer(enable_optimization=True)
workflow = optimizer.create_enhanced_workflow()

# Automatic workflow optimization:
# - Parallel execution where possible
# - Dependency management
# - Error handling and retry logic
# - Resource optimization
```

**Workflow Steps:**
1. **Initialize** - Load user preferences and company data
2. **Weight Approval** - Interactive or automatic weight configuration
3. **Multi-LLM Analysis** - Parallel execution across models
4. **Scoring Calculation** - Consensus and composite scoring
5. **Human Feedback** - Expert evaluation (if enabled)
6. **Result Generation** - Multi-format output creation

### **Performance Optimization**

**Automatic Optimization Features:**
- **Task Prioritization**: Execute high-impact, low-cost tasks first
- **Resource Management**: Optimize API rate limits and concurrent execution
- **Caching**: Reuse results where appropriate
- **Error Recovery**: Intelligent retry logic with exponential backoff

**Performance Monitoring:**
```python
# Monitor execution performance
performance = optimizer.analyze_workflow_performance(execution_data)

# Results:
{
  "total_execution_time": 45.2,
  "bottlenecks": [
    {"step": "multi_llm_analysis", "time_taken": 30.1, "percentage": 66.6}
  ],
  "optimization_suggestions": [
    "Consider parallel execution for bottleneck steps",
    "Consider caching intermediate results"
  ]
}
```

### **User Preference Learning**

**Automatic Preference Detection:**
- Track weight modifications across sessions
- Learn expert feedback patterns
- Optimize model selection based on user preferences
- Suggest configuration improvements

**Preference Storage:**
```json
{
  "user_id": "analyst1",
  "weight_preferences": {
    "barriers_to_entry": 0.180,  # User consistently increases this
    "growth_drivers": 0.070      # User values growth more than default
  },
  "model_preferences": ["mixtral-8x7b", "llama-3-70b"],
  "analysis_settings": {
    "preferred_themes": ["AI strategy", "Competitive moats"],
    "max_concurrent": 3,
    "enable_feedback": true
  }
}
```

---

## 🔌 API Reference {#api-reference}

### **Enhanced Analysis Controller**

```python
from engines.enhanced_analysis_controller import (
    EnhancedAnalysisController,
    EnhancedAnalysisConfig
)

# Initialize controller
controller = EnhancedAnalysisController()

# Create configuration
config = EnhancedAnalysisConfig(
    user_id="analyst1",
    company_ticker="NVDA",
    analysis_type="comprehensive",
    enable_weight_approval=True,
    enable_human_feedback=True,
    expert_id="expert1"
)

# Run analysis
result = controller.run_enhanced_analysis(config)

# Access results
print(f"Score: {result.final_composite_score}")
print(f"Best Model: {result.multi_llm_result.best_model_recommendation}")
```

### **Multi-LLM Engine**

```python
from engines.multi_llm_engine import MultiLLMEngine

engine = MultiLLMEngine()

# Get available models
models = engine.get_available_models()

# Estimate costs
cost_estimate = engine.estimate_multi_llm_cost("NVDA", num_models=5)

# Run multi-LLM analysis
result = engine.run_multi_llm_analysis(company, analysis_config, user_weights)
```

### **Enhanced Scoring System**

```python
from engines.enhanced_scoring_system import EnhancedScoringSystem, WeightingScheme

scoring = EnhancedScoringSystem()

# Extract scores from analysis
scores = scoring.extract_all_scores(analysis_result)

# Calculate composite score with custom weights
weights = WeightingScheme(barriers_to_entry=0.20, growth_drivers=0.08)
composite_score, confidence, metadata = scoring.calculate_composite_score(scores, weights)
```

### **Human Feedback System**

```python
from engines.human_feedback_system import HumanFeedbackSystem

feedback_system = HumanFeedbackSystem()

# Present model comparison
comparison = feedback_system.present_model_comparison(multi_llm_result, "expert1")

# Collect feedback
feedback_id = feedback_system.collect_expert_feedback(comparison, expert_selection)

# Get performance metrics
performance = feedback_system.get_model_performance_report()
```

### **Weight Approval System**

```python
from engines.weight_approval_system import WeightApprovalSystem

weight_system = WeightApprovalSystem()

# Create approval session
session = weight_system.create_approval_session("analyst1", "NVDA")

# Present weights for approval
presentation = weight_system.present_weights_for_approval(session)

# Process user response
session, approved = weight_system.process_user_response(session, user_response)
```

---

## 🛠️ Troubleshooting

### **Installation Issues**

**LangChain/LangGraph Installation:**
```bash
# If LangGraph installation fails
pip install --upgrade pip
pip install langchain==0.1.0 langgraph==0.0.40 langchain-core==0.1.0

# Alternative installation
conda install -c conda-forge langchain
pip install langgraph
```

**SQLite Database Issues:**
```bash
# SQLite is built-in to Python, but if issues occur:
python -c "import sqlite3; print('SQLite OK')"

# If permission issues, check data directory permissions
chmod 755 data/
```

### **API and Model Issues**

**TogetherAI API Issues:**
```bash
# Test API connection
python -c "
import requests
headers = {'Authorization': 'Bearer YOUR_API_KEY'}
response = requests.get('https://api.together.xyz/models', headers=headers)
print(response.status_code)
"
```

**Model Availability:**
```python
# Check available models
from engines.multi_llm_engine import MultiLLMEngine
engine = MultiLLMEngine()
models = engine.get_available_models()
print(f"Available models: {[m.model_name for m in models]}")
```

**Cost Control:**
```bash
# If costs are too high
python run_enhanced_analysis.py --user-id analyst1 --company NVDA \
  --models mixtral-8x7b --analysis-type quick --max-concurrent 1
```

### **Performance Issues**

**Slow Execution:**
```bash
# Reduce concurrent models
--max-concurrent 1

# Use quick analysis
--analysis-type quick

# Use fewer models
--models mixtral-8x7b,llama-3-70b
```

**Memory Issues:**
```python
# Monitor memory usage
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")

# Reduce batch size
--companies NVDA,AAPL  # Instead of 5+ companies
```

### **Data Issues**

**Missing Results Files:**
```bash
# Check results directory
ls -la results/

# Verify data directory structure
ls -la data/

# Check permissions
chmod 755 results/ data/
```

**Database Corruption:**
```bash
# Reset human feedback database
rm data/human_feedback.db

# Reset training dataset
rm data/training_dataset.json
```

### **Common Error Messages**

**"No LLM models available":**
- Check API keys are correctly set in .env
- Verify API keys have sufficient credits
- Test with a single model first

**"Failed to parse JSON from model":**
- This is normal for some model outputs
- The system automatically handles parsing failures
- Check if analysis still completed successfully

**"Cost estimate too high":**
- Use `--cost-estimate-only` to check costs first
- Reduce number of models or use quick analysis
- Consider using only TogetherAI models (cheaper)

### **Getting Additional Help**

1. **Check Logs**: All operations are logged with timestamps
2. **Test Components**: Use individual components to isolate issues
3. **Validate Setup**: Use `--cost-estimate-only` to test configuration
4. **Monitor Resources**: Check memory and disk space usage
5. **API Status**: Check API provider status pages

---

## 📈 Performance Metrics & Analytics

### **Analysis Performance Tracking**

**Execution Metrics:**
```python
# Access performance data from results
execution_data = result.execution_metadata

metrics = {
    'total_time': execution_data['total_execution_time'],
    'models_used': len(result.multi_llm_result.llm_results),
    'success_rate': len(result.multi_llm_result.individual_scores) / len(result.multi_llm_result.llm_results),
    'cost_efficiency': result.final_composite_score / result.total_cost_usd,
    'confidence_level': result.final_confidence
}
```

**Quality Assessment:**
- Model consensus strength (agreement between models)
- Confidence score distribution across components
- Expert feedback alignment with system recommendations
- Historical performance trends

### **Cost Analytics**

**Cost Optimization Tracking:**
```python
# Track cost efficiency over time
cost_per_point = total_cost / composite_score
model_efficiency = {model: cost/quality for model, cost, quality in model_results}

# Identify most cost-effective models
best_value_models = sorted(model_efficiency.items(), key=lambda x: x[1])
```

---

**QualAgent Enhanced provides a comprehensive, production-ready system for advanced financial analysis with human feedback integration. The multi-LLM approach, comprehensive scoring, and expert feedback create a robust framework for institutional-quality investment research.**

For technical implementation details, see the [Technical Documentation Enhanced](TECHNICAL_DOCUMENTATION_ENHANCED.md).