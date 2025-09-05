# Lohusalu Capital Management - 3-Agent Investment Analysis System

## 🚀 Quick Deploy on Streamlit Cloud

This repository contains a streamlined 3-agent system for equity portfolio construction and investment analysis.

### 📋 Prerequisites for Streamlit Cloud Deployment

1. **Fork this repository** to your GitHub account
2. **Sign up at [streamlit.io](https://streamlit.io)** with your GitHub account
3. **Deploy** by connecting your forked repository

### 🔧 Environment Setup (Optional)

The system works in **fallback mode** without API keys, but for full functionality:

1. **OpenAI API Key** (optional): For enhanced LLM analysis
2. **Tavily API Key** (optional): For web research functionality

Add these as **secrets** in Streamlit Cloud:
- `OPENAI_API_KEY`
- `TAVILY_API_KEY`

### 🎯 System Architecture

**3-Agent Pipeline:**
1. **📊 Fundamental Agent** - Quantitative screening (growth, profitability, debt)
2. **🔍 Rationale Agent** - Qualitative analysis (moats, sentiment, trends)
3. **🏆 Ranker Agent** - Final scoring with investment recommendations

### 🌟 Key Features

- **No API Keys Required**: Works in fallback mode for testing
- **1-10 Scoring Scale**: Clear investment ratings with detailed reasoning
- **Portfolio Construction**: Automated allocation and position sizing
- **Export Capabilities**: CSV downloads and comprehensive reports
- **Comprehensive Logging**: All analysis traces saved for audit

### 📊 Sample Output

The system analyzes companies and provides:
- **Investment Scores**: 1-10 scale with detailed reasoning
- **Recommendations**: STRONG_BUY/BUY/HOLD/SELL with confidence levels
- **Portfolio Weights**: Optimized allocation percentages
- **Risk Assessment**: Position sizing based on confidence and market cap

### 🔄 Usage Flow

1. **Fundamental Agent**: Screen companies by sector and market cap
2. **Rationale Agent**: Analyze competitive advantages and market trends
3. **Ranker Agent**: Get final investment recommendations and portfolio

### 📁 Project Structure

```
├── Home.py                          # Main Streamlit app entry point
├── pages/
│   ├── 1_📊_Fundamental_Agent.py   # Quantitative screening interface
│   ├── 2_🔍_Rationale_Agent.py     # Qualitative analysis interface
│   └── 3_🏆_Ranker_Agent.py        # Final scoring interface
├── src/agents/                      # Core agent implementations
├── requirements.txt                 # Python dependencies
└── .env.example                     # Environment variables template
```

### 🚀 Local Development

```bash
# Clone the repository
git clone <your-fork-url>
cd alpha-agents

# Install dependencies
pip install -r requirements.txt

# Copy environment template (optional)
cp .env.example .env

# Run locally
streamlit run Home.py
```

### 🎉 Live Demo

Once deployed on Streamlit Cloud, the app provides:
- Interactive 3-agent analysis pipeline
- Real-time investment scoring
- Portfolio construction tools
- Comprehensive export capabilities

**Built with**: Streamlit, LangChain, OpenAI, Tavily, yfinance, Plotly

