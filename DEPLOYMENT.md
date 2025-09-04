# Alpha Agents - Deployment Guide

This document provides comprehensive instructions for deploying the Alpha Agents multi-agent equity portfolio system.

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- Git
- OpenAI API key

### Installation
```bash
# Clone the repository
git clone https://github.com/predictivelabsai/alpha-agents.git
cd alpha-agents

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### Running the Application
```bash
# Run tests (optional but recommended)
python tests/run_all_tests.py

# Start the Streamlit application
streamlit run Home.py
```

The application will be available at `http://localhost:8501`

## 📁 Project Structure

```
alpha-agents/
├── Home.py                 # Main Streamlit application entry point
├── pages/                  # Multi-page Streamlit pages
│   ├── 1_📊_Stock_Analysis.py
│   ├── 2_📊_Analytics.py
│   ├── 3_🎯_Portfolio_Builder.py
│   └── 4_ℹ️_About.py
├── src/                    # Source code
│   ├── agents/            # Multi-agent system
│   │   ├── base_agent.py
│   │   ├── fundamental_agent.py
│   │   ├── sentiment_agent.py
│   │   ├── valuation_agent.py
│   │   ├── rationale_agent.py
│   │   ├── secular_trend_agent.py
│   │   └── multi_agent_system.py
│   ├── database/          # Database schema and operations
│   └── visualizations.py  # Plotly visualization components
├── tests/                 # Comprehensive test suite
│   ├── unit/             # Unit tests for individual agents
│   ├── integration/      # Integration tests
│   └── run_all_tests.py  # Main test runner
├── test-data/            # Generated test data and results
├── docs/                 # Documentation
│   └── user_guide.md     # User guide with screenshots
├── requirements.txt      # Python dependencies
├── README.md            # Main documentation
└── DEPLOYMENT.md        # This file
```

## 🤖 Multi-Agent System

The system includes 5 specialized AI agents:

1. **Fundamental Agent**: Analyzes financial statements and fundamentals
2. **Sentiment Agent**: Processes news and market sentiment
3. **Valuation Agent**: Evaluates pricing and technical indicators
4. **Rationale Agent**: Assesses business quality using 7-step framework
5. **Secular Trend Agent**: Identifies technology trend positioning

## 📊 Features

- **Multi-page Streamlit Interface**: Clean, organized navigation
- **Interactive Visualizations**: Plotly charts and heatmaps
- **Comprehensive Testing**: Unit tests and integration tests
- **Database Integration**: SQLite for data persistence
- **Real-time Analysis**: Live stock analysis with agent collaboration

## 🧪 Testing

The project includes comprehensive testing:

```bash
# Run all tests
python tests/run_all_tests.py

# Test individual components
python -m pytest tests/unit/
```

Test results are saved in the `test-data/` directory.

## 🔧 Configuration

### Environment Variables
- `OPENAI_API_KEY`: Required for AI agent functionality

### Database
- SQLite database (`alpha_agents.db`) is created automatically
- No additional database setup required

## 📈 Performance

Based on testing:
- **40 test analyses** across 8 stocks and 5 agents
- **Average confidence**: 0.66
- **Success rate**: 100% for agent functionality
- **Response time**: < 1 second for fallback mode

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version: `python --version` (should be 3.10+)

2. **OpenAI API Errors**
   - Verify API key is set correctly in `.env` file
   - Check API key permissions and billing status

3. **Streamlit Issues**
   - Clear Streamlit cache: `streamlit cache clear`
   - Restart the application: `Ctrl+C` then `streamlit run Home.py`

4. **Database Issues**
   - Delete `alpha_agents.db` to reset database
   - Check file permissions in project directory

### Getting Help

- Check the [README.md](README.md) for detailed usage instructions
- Review the [User Guide](docs/user_guide.md) for feature explanations
- Open an issue on the GitHub repository for bugs or feature requests

## 🔄 Updates and Maintenance

### Updating the Application
```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

### Data Backup
- Database: `alpha_agents.db`
- Test results: `test-data/` directory
- User configurations: `.env` file

## 📞 Support

For technical support or questions:
- GitHub Repository: https://github.com/predictivelabsai/alpha-agents
- Issues: https://github.com/predictivelabsai/alpha-agents/issues

---

**Alpha Agents** - Advanced Multi-Agent System for Equity Portfolio Construction

