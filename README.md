# Alpha Agents - Financial Analysis Platform

## 🏛️ Overview

**Lohusalu Capital Management** - Advanced Financial Analysis Platform providing comprehensive quantitative and qualitative research tools for investment analysis.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- PostgreSQL database
- Required Python packages (see requirements.txt)

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd alpha-agents
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Setup FinProsso PostgreSQL database**
```bash
# Database is hosted on FinProsso platform
# No local database setup required
```

5. **Configure database connection**
Create `.env` file in project root:
```env
DATABASE_URL=postgresql://finprosso_username:finprosso_password@finprosso_host:5432/finprosso_database
```

6. **Run the application**
```bash
streamlit run Home.py
```

## 🎯 How to Use the Platform

### Step 1: Run Fundamental Screener
1. **Navigate to**: Fundamental Screener Agent
2. **Select filters**:
   - Region: US, Europe, Asia, etc.
   - Filter by: Sector or Industry
   - Market Cap: Min/Max values
3. **Click**: "Run Screen"
4. **Results**: View table with financial metrics and scores
5. **Save**: Click "💾 Store in PostgreSQL Database"

### Step 2: Run Qualitative Analysis
1. **Navigate to**: Qualitative Analysis Agent
2. **Data Input tab**:
   - **From Screener**: Use results from Step 1
   - **From Database**: Load previous screening results
   - **Upload CSV**: Upload your own company list
   - **Manual Input**: Enter tickers manually
3. **Select companies** to analyze
4. **Run Analysis tab**: Click "Run Analysis"
5. **Results & Downloads tab**: 
   - View results table
   - Click "💾 Save All Results to Database"

### Step 3: View and Export Results
- **Results & Downloads**: View all analysis results
- **Explore LLM Results**: Deep dive into individual company analysis
- **Download**: CSV, Excel, JSON files
- **Database**: All results automatically saved to PostgreSQL

## 📊 Application Structure

### Main Pages

#### 🏠 Home Page (`Home.py`)
- **Title**: Lohusalu Capital Management
- **Navigation**: Access to both analysis agents
- **Features**: 
  - Fundamental Screener Agent
  - Qualitative Analysis Agent

#### 📈 Fundamental Screener Agent (`pages/1_Fundamental_Screener_Agent.py`)
- **Purpose**: Quantitative financial analysis and stock screening
- **Features**:
  - Sector/Industry filtering
  - Market cap filtering
  - Comprehensive financial metrics calculation
  - Results export (CSV, Excel)
  - Database storage

#### 🧠 Qualitative Analysis Agent (`pages/2_Qualitative_Analysis_Agent.py`)
- **Purpose**: Qualitative analysis using multiple LLM models
- **Features**:
  - Multi-source data input (Screener, Database, CSV, Manual)
  - API testing and configuration
  - Interactive weight management
  - Batch analysis execution
  - Results exploration and download
  - Database integration

## 🗄️ Database Structure

The platform uses PostgreSQL with two main tables to store analysis results.

### Table 1: `fundamental_screen`
Stores quantitative screening results from the Fundamental Screener Agent.

**Key Columns:**
- `run_id`: Unique identifier for each screening run
- `ticker`: Stock ticker symbol
- `company_name`: Company name
- `sector`, `industry`: Business classification
- `market_cap`: Market capitalization
- **Financial Metrics**: TTM margins, ROA, ROE, ROIC, CAGR, consistency ratios
- **Risk Metrics**: Liquidity, leverage, debt servicing ratios
- `score`: Composite scoring (0-100)
- `created_at`: Timestamp

**Example Rows:**

| run_id | ticker | company_name | sector | industry | market_cap | gross_profit_margin_ttm | roa_ttm | roe_ttm | score | created_at |
|--------|--------|--------------|--------|----------|------------|------------------------|---------|---------|-------|------------|
| screen_20241201_143022 | AAPL | Apple Inc. | Technology | Consumer Electronics | 3000000000000 | 0.4500 | 0.1500 | 0.2500 | 85.5 | 2024-12-01 14:30:22 |
| screen_20241201_143022 | MSFT | Microsoft Corporation | Technology | Software | 2800000000000 | 0.4200 | 0.1800 | 0.2800 | 82.3 | 2024-12-01 14:30:22 |
| screen_20241201_143022 | GOOGL | Alphabet Inc. | Technology | Internet | 1800000000000 | 0.3800 | 0.1200 | 0.2200 | 78.9 | 2024-12-01 14:30:22 |

### Table 2: `qualitative_analysis`
Stores qualitative analysis results from the Qualitative Analysis Agent.

**Key Columns:**
- `run_id`: Unique identifier for each analysis run
- `company_ticker`: Stock ticker symbol
- `analysis_timestamp`: Unix timestamp of analysis
- `composite_score`: Overall qualitative score (0-5)
- `composite_confidence`: Confidence level (0-1)
- **Individual Component Scores**: 15 detailed analysis components
- `full_analysis_data`: Complete JSON analysis data
- `individual_model_results`: Results from each LLM model
- `execution_metadata`: Analysis execution details

**Example Rows:**

| run_id | company_ticker | analysis_timestamp | composite_score | composite_confidence | moat_brand_monopoly_score | management_quality_score | key_growth_drivers_score | created_at |
|--------|----------------|-------------------|-----------------|---------------------|---------------------------|-------------------------|-------------------------|------------|
| qual_batch_1760551582_AAPL | AAPL | 1760551582 | 4.2 | 0.85 | 4.5 | 4.0 | 3.8 | 2024-12-01 18:06:22 |
| qual_batch_1760551582_MSFT | MSFT | 1760551586 | 4.0 | 0.82 | 4.2 | 4.3 | 3.9 | 2024-12-01 18:06:26 |
| qual_batch_1760551582_GOOGL | GOOGL | 1760551590 | 3.8 | 0.78 | 3.9 | 3.7 | 4.1 | 2024-12-01 18:06:30 |

## 💾 Database Operations

### How Data is Saved to Database

#### 1. Fundamental Screener Results
**Process:**
1. User runs screening with filters (sector, market cap, etc.)
2. System calculates financial metrics for all companies
3. Results are displayed in a table
4. User clicks "💾 Store in PostgreSQL Database"
5. All results saved to FinProsso `fundamental_screen` table with unique `run_id`

**Code Example:**
```python
# Automatic saving after screening
from utils.db_util import save_fundamental_screen

# Save screening results
save_fundamental_screen(df_results, run_id="screen_20241201_143022")
```

#### 2. Qualitative Analysis Results
**Process:**
1. User selects companies for analysis (from screener, database, CSV, or manual)
2. System runs LLM analysis on selected companies
3. Results are parsed and displayed in table
4. User clicks "💾 Save All Results to Database"
5. Each company gets individual record in FinProsso `qualitative_analysis` table

**Code Example:**
```python
# Save individual analysis
from utils.db_util import save_qualitative_analysis

# Save analysis data with unique run_id per company
save_qualitative_analysis(analysis_data, run_id="qual_batch_1760551582_AAPL", company_ticker="AAPL")
```

### Database Schema Details

#### Fundamental Screen Table Structure
```sql
CREATE TABLE fundamental_screen (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(50) NOT NULL,
    ticker VARCHAR(10) NOT NULL,
    company_name VARCHAR(255),
    sector VARCHAR(100),
    industry VARCHAR(100),
    market_cap BIGINT,
    gross_profit_margin_ttm DECIMAL(5, 4),
    operating_profit_margin_ttm DECIMAL(5, 4),
    net_profit_margin_ttm DECIMAL(5, 4),
    roa_ttm DECIMAL(5, 4),
    roe_ttm DECIMAL(5, 4),
    roic_ttm DECIMAL(5, 4),
    score DECIMAL(5, 2),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

#### Qualitative Analysis Table Structure
```sql
CREATE TABLE qualitative_analysis (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(50) NOT NULL,
    company_ticker VARCHAR(10) NOT NULL,
    analysis_timestamp BIGINT NOT NULL,
    composite_score DECIMAL(5, 3),
    composite_confidence DECIMAL(5, 3),
    moat_brand_monopoly_score DECIMAL(5, 3),
    management_quality_score DECIMAL(5, 3),
    key_growth_drivers_score DECIMAL(5, 3),
    -- ... 12 more component scores ...
    full_analysis_data JSONB,
    individual_model_results JSONB,
    execution_metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

## 🔄 Complete Data Flow Process

### 1. Fundamental Screening Process
```
User Input (Sector/Industry/Market Cap) 
    ↓
Stock Universe Loading (from CSV files)
    ↓
Financial Data Fetching (yfinance API)
    ↓
Metrics Calculation (TTM margins, ROA, ROE, ROIC, CAGR, ratios)
    ↓
Percentile Scoring (0-100 scale)
    ↓
Results Display in Table
    ↓
User Clicks "Save to Database"
    ↓
Data Saved to FinProsso fundamental_screen table
```

### 2. Qualitative Analysis Process
```
Data Input (Screener/DB/CSV/Manual)
    ↓
Company Selection (multiselect with filters)
    ↓
LLM Analysis (GPT-4, Claude, Gemini)
    ↓
15 Component Scoring (Moat, Management, Growth, etc.)
    ↓
Weight Application & Composite Scoring
    ↓
Results Parsing & Table Display
    ↓
User Clicks "Save All Results to Database"
    ↓
Each Company Saved to FinProsso qualitative_analysis table
```

### 3. Database Integration Flow
```
Analysis Results
    ↓
Data Validation & Cleaning
    ↓
Unique Run ID Generation
    ↓
PostgreSQL Insert to FinProsso Database with Error Handling
    ↓
Success Confirmation
    ↓
Results Available for Future Analysis
```

## 🛠️ Configuration

### Database Setup
Create `.env` file in project root:
```env
DATABASE_URL=postgresql://finprosso_username:finprosso_password@finprosso_host:5432/finprosso_database
```

### API Keys Setup
Add to `.env` file:
```env
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
```

### Database Tables Creation
Tables are automatically created on FinProsso database when first saving data.

## 📁 Project Structure

```
alpha-agents/
├── Home.py                          # Main application entry point
├── pages/
│   ├── 1_Fundamental_Screener_Agent.py    # Quantitative analysis
│   └── 2_Qualitative_Analysis_Agent.py    # Qualitative analysis
├── utils/
│   ├── stock_screener.py           # Core screening logic
│   ├── db_util.py                  # Database operations
│   ├── test_llm_api.py             # API testing utilities
│   └── weight_manager.py           # Weight management system
├── engines/
│   └── enhanced_scoring_system.py   # Scoring algorithms
├── agents/
│   └── QualAgent/                  # Qualitative analysis engine
├── sql/
│   └── create_table.sql            # Database schema
├── docs/
│   └── Qualitative_Analysis_Database_Integration.md
└── requirements.txt
```

## 🎯 Key Features

### Fundamental Screener Agent
- **Comprehensive Metrics**: 20+ financial ratios and growth metrics
- **Sector Filtering**: Technology, Healthcare, Finance, etc.
- **Market Cap Filtering**: Customizable size criteria
- **Percentile Scoring**: Relative performance ranking
- **Export Options**: CSV, Excel, Database storage

### Qualitative Analysis Agent
- **Multi-LLM Support**: GPT-4, Claude, Gemini integration
- **Interactive Weights**: Customizable component importance
- **Batch Processing**: Analyze multiple companies simultaneously
- **Deep Analysis**: 15 qualitative components per company
- **Results Exploration**: Detailed LLM reasoning and justifications

## 🔧 Troubleshooting

### Common Issues

1. **Database Connection Error**
   - Check FinProsso database service is running
   - Verify credentials in `.env` file
   - Ensure DATABASE_URL is correctly formatted

2. **API Key Errors**
   - Verify API keys in `.env` file
   - Check API quotas and limits
   - Test API connectivity in API Testing tab

3. **Import Errors**
   - Ensure virtual environment is activated
   - Install all requirements: `pip install -r requirements.txt`
   - Check Python path configuration

4. **File Permission Errors**
   - Ensure write permissions for results directory
   - Check database user permissions

### Performance Tips

1. **Large Datasets**: Use sector/industry filtering to reduce processing time
2. **API Limits**: Monitor API usage in the API Testing tab
3. **Database**: Index frequently queried columns for better performance
4. **Memory**: Close unused browser tabs to free memory

## 📈 Usage Examples

### Example 1: Complete Analysis Workflow
```bash
# 1. Start the application
streamlit run Home.py

# 2. Run fundamental screening
# Navigate to: Fundamental Screener Agent
# Select: Technology sector, Market cap > $1B
# Click: "Run Screen"
# Click: "💾 Store in PostgreSQL Database"
# Result: 50+ companies saved to FinProsso fundamental_screen table

# 3. Run qualitative analysis
# Navigate to: Qualitative Analysis Agent
# Data Input tab: Select "From Screener" → Choose Technology companies
# Select: Top 5 companies (AAPL, MSFT, GOOGL, AMZN, TSLA)
# Run Analysis tab: Click "Run Analysis"
# Results & Downloads tab: Click "💾 Save All Results to Database"
# Result: 5 companies saved to FinProsso qualitative_analysis table
```

### Example 2: Database Query Examples
```sql
-- View all screening results
SELECT ticker, company_name, sector, score, created_at 
FROM fundamental_screen 
ORDER BY score DESC 
LIMIT 10;

-- View qualitative analysis results
SELECT company_ticker, composite_score, composite_confidence, created_at
FROM qualitative_analysis 
WHERE composite_score > 4.0
ORDER BY composite_score DESC;

-- Join both tables for comprehensive view
SELECT f.ticker, f.company_name, f.score as fundamental_score, 
       q.composite_score as qualitative_score
FROM fundamental_screen f
JOIN qualitative_analysis q ON f.ticker = q.company_ticker
WHERE f.score > 80 AND q.composite_score > 4.0;
```

### Example 3: Custom Weight Analysis
```python
# 1. Navigate to Weight Management tab
# 2. Adjust component weights:
#    - Moat Analysis: 30%
#    - Management Quality: 25%
#    - Growth Drivers: 20%
#    - Risk Factors: 15%
#    - Others: 10%
# 3. Run analysis with custom weights
# 4. Compare results with default weights
# 5. Save results to database for comparison
```

## 📞 Support

For technical support or feature requests:
- Check the troubleshooting section above
- Review the database integration documentation
- Verify all configuration files are properly set up

## 🔄 Updates and Maintenance

### Regular Maintenance
- Monitor database size and performance
- Update API keys before expiration
- Backup analysis results regularly
- Review and update weight configurations

### Version Updates
- Check for new requirements in `requirements.txt`
- Update database schema if needed
- Test API integrations after updates

---

**Lohusalu Capital Management** - Advanced Financial Analysis Platform
*Comprehensive quantitative and qualitative research tools for investment analysis*