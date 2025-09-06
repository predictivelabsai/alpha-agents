# Lohusalu Capital Management - Agentic Equity Portfolio Construction

**Lohusalu Capital Management** is a sophisticated, multi-agent equity portfolio construction system designed to streamline the investment research process. This system leverages a powerful 3-agent architecture, combining quantitative screening, qualitative analysis, and intelligent ranking to identify high-potential investment opportunities.

![Lohusalu Capital Management](https://i.imgur.com/example.png)  <!-- Replace with actual screenshot -->

## 🚀 Key Features

- **3-Agent Architecture**: A robust pipeline featuring a Fundamental Agent, Rationale Agent, and Ranker Agent.
- **Quantitative & Qualitative Analysis**: Combines deep financial metric screening with qualitative analysis of moats, sentiment, and trends.
- **Multi-Model Support**: Seamlessly switch between top-tier LLMs (OpenAI, Google, Anthropic, Mistral) for analysis.
- **Comprehensive JSON Tracing**: In-depth tracing of each agent's reasoning and analysis, with a dedicated Trace Manager.
- **Interactive Streamlit UI**: A multi-page application for running agents, viewing results, and managing traces.
- **Dynamic Stock Discovery**: Agents dynamically discover and analyze stocks, with no hardcoded tickers.

## 🤖 3-Agent Pipeline Architecture

The core of Lohusalu Capital Management is its 3-agent pipeline, designed to mimic the workflow of a professional investment team:

### 1. **📊 Fundamental Agent**
- **Sector Analysis**: Identifies trending sectors and assigns investment weights.
- **Quantitative Screening**: Screens stocks against a comprehensive set of financial metrics (growth, profitability, valuation, etc.).
- **Intrinsic Value Calculation**: Performs DCF-based valuation to determine upside potential.

### 2. **🔍 Rationale Agent**
- **Qualitative Analysis**: Conducts in-depth analysis of economic moats, market sentiment, and secular trends.
- **Web Search Integration**: Leverages the Tavily search API for real-time information and citations.
- **Competitive Landscape**: Assesses the competitive position and market share trends of companies.

### 3. **🎯 Ranker Agent**
- **Composite Scoring**: Combines fundamental (60%) and qualitative (40%) scores into a single, comprehensive rating.
- **Investment Grading**: Assigns investment grades (A+ to D) with a detailed breakdown of scoring.
- **Portfolio Construction**: Generates a ranked list of investment opportunities and a sample portfolio.

## 📁 Project Structure

```
/alpha-agents
├── Home.py                   # Main Streamlit application
├── pages/                    # Streamlit pages for each agent and tool
│   ├── 1_🤖_Agentic_Screener_v2.py
│   ├── 2_📊_Fundamental_Agent.py
│   ├── 3_🔍_Rationale_Agent.py
│   ├── 4_🎯_Ranker_Agent.py
│   └── 5_📁_Trace_Manager.py
├── src/
│   ├── agents/               # Agent implementations
│   │   ├── fundamental_agent_v2.py
│   │   ├── rationale_agent_v2.py
│   │   └── ranker_agent_v2.py
│   └── utils/                # Utility functions
│       ├── trace_manager.py
│       └── yfinance_util.py
├── prompts/                  # Prompts for each agent
│   ├── fundamental_agent_v2.py
│   ├── rationale_agent_v2.py
│   └── ranker_agent_v2.py
├── tracing/                  # Directory for JSON trace files
├── logs/                     # Directory for application logs
├── docs/                     # Project documentation
│   ├── METHODOLOGY.md
│   └── user_guide.md
├── tests/                    # Test suite for agents
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🔧 Getting Started

### Prerequisites

- Python 3.8+
- Pip and Virtualenv
- An active Tavily API key (`TAVILY_API_KEY`)
- API keys for your chosen LLM providers (e.g., `OPENAI_API_KEY`)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/alpha-agents.git
    cd alpha-agents
    ```

2.  **Set up a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the root directory and add your API keys:
    ```
    TAVILY_API_KEY="your_tavily_api_key"
    OPENAI_API_KEY="your_openai_api_key"
    # Add other provider keys as needed
    ```

### Running the Application

To launch the Streamlit application, run:

```bash
streamlit run Home.py
```

The application will be accessible at `http://localhost:8501`.

## 📖 Usage

The application is divided into several pages, accessible from the sidebar:

- **Agentic Screener v2**: Run the complete 3-agent pipeline from start to finish.
- **Fundamental Agent**: Run the Fundamental Agent independently for sector analysis and stock screening.
- **Rationale Agent**: Perform deep qualitative analysis on a single stock.
- **Ranker Agent**: Score and rank stocks based on pre-computed or mock data.
- **Trace Manager**: View, search, and analyze the JSON traces generated by the agents.

## 🧪 Testing

To run the test suite for the agents, use:

```bash
pytest
```

## 📚 Documentation

- **[METHODOLOGY.md](docs/METHODOLOGY.md)**: A detailed explanation of the 3-agent architecture, scoring methodology, and tracing system.
- **[user_guide.md](docs/user_guide.md)**: A comprehensive guide to using the Streamlit application and its features.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs, feature requests, or improvements.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.


