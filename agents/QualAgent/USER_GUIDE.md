# QualAgent User Guide

**A comprehensive qualitative research system for analyzing technology companies using multiple Large Language Models**

QualAgent automatically gathers information using integrated research tools (Tavily, Polygon, Exa), runs multi-model analysis, and stores structured results in JSON format for easy access and deployment.

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Installation & Setup](#installation--setup)
3. [Configuration](#configuration)
4. [Command Line Usage](#command-line-usage)
5. [Jupyter Notebook Usage](#jupyter-notebook-usage)
6. [Data Management](#data-management)
7. [Cost Management](#cost-management)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Navigate to the project directory
cd D:\Oxford\Extra\Finance_NLP\alphaagent\QualAgent

# Install required packages
pip install -r requirements.txt

# Create .env file with your API keys (see Configuration section)
```

### 2. Test System
```bash
# Verify API keys and system functionality
python run_analysis_demo.py
```

### 3. Run Your First Analysis
```bash
# Interactive demo - select companies and configurations
python run_analysis_demo.py

# OR run directly
python run_analysis_demo.py --company NVDA --models mixtral-8x7b
```

### 4. Use Jupyter Notebook
1. Open `QualAgent_Demo_Notebook.ipynb`
2. Run the first cell to load environment
3. Follow the guided workflow

---

## 💻 Installation & Setup

### Prerequisites
- **Python 3.8+**
- **Internet connection** for API calls
- **API keys** for TogetherAI and/or OpenAI
- **Optional**: Research API keys (Tavily, Polygon, Exa)

### Step-by-Step Installation

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Create Environment File**
   Copy `.env.template` to `.env` and add your API keys

3. **Test Installation**
   ```bash
   python run_analysis_demo.py --help
   ```

---

## ⚙️ Configuration

### Required API Keys

Create a `.env` file in the project root:

```env
# LLM Providers (at least one required)
TOGETHER_API_KEY=your_together_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Research Tools (optional but recommended)
TAVILY_API_KEY=your_tavily_api_key_here
POLYGON_API_KEY=your_polygon_api_key_here
EXA_API_KEY=your_exa_api_key_here
```

### Getting API Keys

#### TogetherAI (Primary LLM Provider - Recommended)
1. Visit [together.ai](https://together.ai)
2. Sign up and create API key
3. Add $10-20 credits to start
4. **Cost**: $0.0006-0.0009 per 1K tokens

#### OpenAI (Backup LLM Provider)
1. Visit [platform.openai.com](https://platform.openai.com)
2. Create API key and add credits
3. **Cost**: $0.00015-0.005 per 1K tokens

#### Research Tools (Optional)
- **Tavily**: Real-time web search ([tavily.com](https://tavily.com))
- **Polygon**: Financial data ([polygon.io](https://polygon.io))
- **Exa**: Advanced web search ([exa.ai](https://exa.ai))

---

## 📖 Command Line Usage

### Interactive Demo
```bash
# Launch interactive menu
python run_analysis_demo.py
```

### Direct Commands

#### Single Company Analysis
```bash
# Quick analysis (single model, fast)
python run_analysis_demo.py --company AAPL --models mixtral-8x7b

# Multi-model analysis with themes
python run_analysis_demo.py --company NVDA --models mixtral-8x7b llama-3-70b --themes "AI market" "Competition"

# Save to specific file
python run_analysis_demo.py --company CRWD --output cybersecurity_analysis.json
```

#### Batch Analysis
```bash
# Pre-configured batch analysis
python run_analysis_demo.py --batch semiconductor_screening

# Custom batch with limits
python run_analysis_demo.py --batch cybersecurity_leaders --max-companies 3

# Full tech survey
python run_analysis_demo.py --batch full_tech_survey --max-companies 5
```

### Available Batch Configurations
- `semiconductor_screening` - AI/chip companies focus
- `cybersecurity_leaders` - Top cybersecurity firms
- `cloud_growth_stories` - High-growth cloud companies
- `full_tech_survey` - Comprehensive technology analysis

### Command Options
```bash
--company TICKER          # Analyze specific company
--models MODEL1 MODEL2     # Specify LLM models to use
--themes "theme1" "theme2" # Analysis focus themes
--batch CONFIG_NAME        # Use predefined batch configuration
--max-companies N          # Limit batch analysis size
--output filename.json     # Specify output file
--consensus               # Generate multi-model consensus
--help                    # Show all options
```

---

## 📓 Jupyter Notebook Usage

### Setup Environment (First Cell)
```python
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
env_path = Path.cwd() / '.env'
if env_path.exists():
    load_dotenv(env_path, override=True)
    print(f'✓ Environment loaded from: {env_path}')

# Add current directory to Python path
import sys
current_dir = str(Path.cwd())
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print("✓ Environment setup complete")
```

### Basic Usage
```python
from run_analysis_demo import QualAgentDemo

# Initialize demo
demo = QualAgentDemo("my_analysis_results.json")

# Load example companies
demo.load_example_companies()

# Display available companies
demo.display_available_companies()

# Run single analysis
result = demo.run_single_analysis(
    ticker="NVDA",
    models=["mixtral-8x7b"],
    themes=["AI market opportunity"]
)

# Run batch analysis
results = demo.run_batch_analysis(
    max_companies=2,
    models=["mixtral-8x7b"],
    subsector_filter="Semiconductors"
)
```

---

## 🗄️ Data Management

### File Structure
```
data/
├── companies.json              # Company database
├── analysis_requests.json      # Request history
├── llm_analyses.json          # LLM results
├── structured_results.json    # Processed data
└── example_companies.json     # Sample companies
```

### Adding Companies
```python
from models.json_data_manager import JSONDataManager, Company

db = JSONDataManager()

new_company = Company(
    company_name="Your Company Inc.",
    ticker="YCOM",
    subsector="Cloud/SaaS",
    market_cap_usd=5000000000,
    employees=1200,
    headquarters="San Francisco, CA",
    description="Company description"
)

company_id = db.add_company(new_company)
```

### Querying Data
```python
# Get company by ticker
company = db.get_company_by_ticker("AAPL")

# List companies by subsector
companies = db.list_companies(subsector="Semiconductors")

# Search companies
results = db.search_companies("nvidia")

# Get latest analysis
latest = db.get_company_latest_analysis("NVDA")
```

### Export Options
```python
# Export to DataFrame
df = db.export_to_dataframe('companies')
df.to_csv('companies_export.csv', index=False)

# Export analysis summary
summary = db.get_analysis_summary(limit=50)
```

### Data Backup
```bash
# Backup important files
cp data/companies.json backup/companies_backup_$(date +%Y%m%d).json
cp analysis_results_*.json backup/
```

---

## 💰 Cost Management

### Typical Analysis Costs
- **Quick analysis**: $0.01 - $0.05 per company
- **Comprehensive analysis**: $0.05 - $0.20 per company
- **Multi-model consensus**: $0.10 - $0.50 per company

### Cost-Effective Strategies
1. **Use TogetherAI models** - 3-5x cheaper than OpenAI
2. **Start with `mixtral-8x7b`** - Good quality/cost balance
3. **Use single model first** - Test before multi-model analysis
4. **Batch processing** - More efficient than individual analyses

### Model Costs (per 1K tokens)
**TogetherAI Models:**
- `mixtral-8x7b`: $0.0006 (recommended for most analyses)
- `llama-3-70b`: $0.0009 (best reasoning)
- `qwen2-72b`: $0.0009 (strong analytics)

**OpenAI Models:**
- `gpt-4o-mini`: $0.00015 (cheapest GPT)
- `gpt-4o`: $0.005 (highest quality but expensive)

### Cost Monitoring
```python
# Check before analysis
demo.estimate_batch_cost(
    companies=["AAPL", "NVDA"],
    models=["mixtral-8x7b"]
)

# View analysis costs
print(f"Analysis cost: ${result['cost']}")
```

---

## 🛠️ Troubleshooting

### Common Issues

#### "No API keys found"
```bash
# Check your .env file exists and has correct format
python run_analysis_demo.py --validate
```

#### "No LLM models available"
- Verify API keys are correct
- Check you have credits in your API account
- Try single model first: `--models mixtral-8x7b`

#### Jupyter notebook not loading environment
```python
# Add this to first cell
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path.cwd() / '.env', override=True)
```

#### High costs
- Use `mixtral-8x7b` instead of GPT models
- Start with single company analysis
- Use `--max-companies 1` for testing

#### Analysis fails
- Check internet connection
- Verify API keys haven't expired
- Try simpler model like `mixtral-8x7b`

### Getting Help
1. Check error messages in output
2. Verify API keys: `python run_analysis_demo.py --validate`
3. Test with minimal configuration first
4. Check API provider status pages

---

## 🎯 Best Practices

### Development Workflow
1. **Start small**: Test with single company analysis
2. **Validate setup**: Check API keys and models
3. **Monitor costs**: Use cost estimation before large batches
4. **Choose models wisely**: Balance quality vs cost
5. **Use focus themes**: Guide analysis to specific areas
6. **Regular backups**: Save important analysis results

### Analysis Strategy
1. **Quick analysis first**: Verify setup works
2. **Single model testing**: Before multi-model consensus
3. **Gradual scaling**: Start with 1-2 companies, then expand
4. **Theme-focused**: Use specific themes for better results
5. **Cost awareness**: Monitor spending, especially with GPT models

### Data Management
1. **Regular exports**: Export results to CSV/Excel
2. **Organized storage**: Use descriptive filenames
3. **Version control**: Keep track of analysis versions
4. **Clean data**: Remove test/incomplete analyses periodically

### Production Usage
1. **Environment isolation**: Use virtual environments
2. **API key security**: Never commit .env files
3. **Rate limiting**: Add delays for large batch jobs
4. **Error handling**: Plan for API failures and retries
5. **Cost monitoring**: Set spending alerts

---

## 📊 Example Workflows

### Research New Investment Opportunity
```bash
# 1. Quick screening
python run_analysis_demo.py --company NVDA --models mixtral-8x7b --themes "Market position"

# 2. Comprehensive analysis
python run_analysis_demo.py --company NVDA --models mixtral-8x7b llama-3-70b --themes "AI market" "Competition" "Growth prospects"

# 3. Competitive analysis
python run_analysis_demo.py --batch semiconductor_screening --max-companies 3
```

### Sector Screening
```bash
# Screen top cybersecurity companies
python run_analysis_demo.py --batch cybersecurity_leaders --max-companies 5

# Export results for spreadsheet analysis
# Results automatically saved to timestamped JSON files
```

### Deep Dive Analysis
```bash
# Multi-model consensus analysis
python run_analysis_demo.py --company AAPL --models mixtral-8x7b llama-3-70b qwen2-72b --consensus --themes "AI strategy" "Services growth" "Hardware innovation"
```

---

**QualAgent provides a powerful foundation for systematic technology company research. The modular design allows for easy customization and extension based on your specific research needs.**

For technical implementation details, see the Technical Documentation.