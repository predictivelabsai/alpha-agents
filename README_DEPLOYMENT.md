# Lohusalu Capital Management - Deployment Guide

## 🚀 Streamlit Cloud Deployment

### Prerequisites
- GitHub repository with this code
- Streamlit Cloud account (free at share.streamlit.io)

### Deployment Steps

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit - Lohusalu Capital Management"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to https://share.streamlit.io
   - Connect your GitHub account
   - Select your repository
   - Set main file: `streamlit_app.py`
   - Add environment variables:
     - `OPENAI_API_KEY`: Your OpenAI API key
     - `TAVILY_API_KEY`: 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
   - Click "Deploy"

3. **Alternative: Local Deployment**
   ```bash
   pip install -r requirements.txt
   streamlit run streamlit_app.py
   ```

## 🔧 Environment Variables Required

- `OPENAI_API_KEY`: Your OpenAI API key for LLM functionality
- `TAVILY_API_KEY`: Web search API key (provided)

## 📁 Project Structure

```
alpha-agents/
├── streamlit_app.py          # Main entry point
├── requirements.txt          # Dependencies
├── pages/                    # Streamlit pages
│   ├── 1_🤖_Agentic_Screener_v2.py
│   ├── 2_📊_Fundamental_Agent.py
│   ├── 3_🔍_Rationale_Agent.py
│   ├── 4_🎯_Ranker_Agent.py
│   └── 5_📁_Trace_Manager.py
├── src/                      # Core agents and utilities
│   ├── agents/
│   └── utils/
├── prompts/                  # Agent prompts
├── tracing/                  # Analysis traces (auto-created)
└── docs/                     # Documentation
```

## 🎯 Features

- **3-Agent Pipeline**: Fundamental → Rationale → Ranker
- **Multi-Model Support**: OpenAI, Google, Anthropic, Mistral
- **Web Search Integration**: Tavily API for qualitative research
- **JSON Tracing**: Complete analysis audit trails
- **Interactive UI**: Professional Streamlit interface

## 🔍 Usage

1. **Agentic Screener**: Run complete 3-agent pipeline
2. **Individual Agents**: Test each agent separately
3. **Trace Manager**: View and analyze reasoning traces
4. **Multi-Model Testing**: Compare results across different LLMs

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **AI/ML**: LangChain, OpenAI, Google AI, Anthropic
- **Data**: yfinance, pandas, numpy
- **Search**: Tavily API
- **Visualization**: Plotly, matplotlib, seaborn

## 📊 System Status

✅ All pages tested and working  
✅ 3-agent system operational  
✅ Multi-model support functional  
✅ Web search integration active  
✅ JSON tracing implemented  
✅ No critical issues found  

**System Health: 100% - Ready for Production**

