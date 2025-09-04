"""
# Lohusalu Capital Management - Multi-Agent Equity Portfolio System

![Alpha Agents Banner](https://i.imgur.com/your-banner-image.png)  <!-- Replace with a real banner image -->

**Lohusalu Capital Management** is an advanced multi-agent system for equity portfolio construction, based on cutting-edge research in artificial intelligence and financial analysis. This Streamlit application demonstrates how specialized AI agents can collaborate, debate, and reach consensus to make sophisticated investment decisions.

[![Python Version](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29-orange.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1-green.svg)](https://www.langchain.com/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18-blue.svg)](https://plotly.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 Key Features

- **🤖 Multi-Agent Collaboration**: 5 specialized AI agents with domain expertise in fundamental analysis, sentiment analysis, valuation, business quality, and secular trends.
- **📈 Portfolio Construction**: Automated stock selection, risk-adjusted optimization, and diversification analysis.
- **📊 Advanced Analytics**: Interactive visualizations including heatmaps, performance charts, and portfolio optimization analysis.
- **🧪 Comprehensive Testing**: Extensive test suite with performance metrics, data generation, and validation of agent decision-making.
- **🌐 Web Interface**: User-friendly multi-page Streamlit application for easy interaction and analysis.

---

## 🏗️ System Architecture

The Lohusalu Capital Management system is built on a modern, modular architecture:

- **Multi-Agent Framework**: LangGraph-based workflow orchestration for structured debate and consensus.
- **AI Engine**: OpenAI GPT models for agent intelligence and reasoning.
- **Web Interface**: Streamlit for the interactive multi-page application.
- **Data Visualization**: Plotly for advanced, interactive charts and heatmaps.
- **Database**: SQLite for persistent storage of analyses and portfolios.

![System Architecture Diagram](https://i.imgur.com/your-architecture-diagram.png) <!-- Replace with a real diagram -->

---

## ⚙️ Installation

Follow these steps to set up and run the Alpha Agents system on your local machine.

### Prerequisites

- Python 3.10 or higher
- `pip` for package management
- `git` for cloning the repository

### 1. Clone the Repository

```bash
git clone https://github.com/predictivelabsai/alpha-agents.git
cd alpha-agents
```

### 2. Set Up a Virtual Environment

It is highly recommended to use a virtual environment to manage dependencies.

```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

Install all required Python packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory of the project and add your OpenAI API key.

```env
OPENAI_API_KEY="your-openai-api-key-here"
```

Replace `your-openai-api-key-here` with your actual OpenAI API key.

---

## 🚀 Running the Application

Once the installation is complete, you can run the Streamlit application.

### 1. Run the Test Suite (Optional but Recommended)

Before running the application, it is recommended to run the comprehensive test suite to ensure all components are working correctly and to generate initial analysis data.

```bash
python tests/run_all_tests.py
```

This will create a `test-data/` directory with CSV and JSON files containing agent analysis results.

### 2. Start the Streamlit Application

Run the following command to start the Streamlit server:

```bash
streamlit run Home.py
```

The application will be available at `http://localhost:8501` in your web browser.

---

## 📋 Usage Guide

The Alpha Agents application is organized into several pages, accessible from the sidebar navigation.

### 🏠 Home

The home page provides an overview of the system, its key features, and the specialized AI agents. It also shows a summary of system performance based on the latest test data.

### 📊 Stock Analysis

Analyze individual stocks by providing the stock symbol, company name, and other relevant details. The multi-agent system will perform a comprehensive analysis and provide recommendations, risk assessments, and detailed reasoning from each agent.

### 📊 Analytics

Explore advanced visualizations of the system's performance. This page includes:

- **Agent Consensus & Risk Heatmaps**: Visualize agent agreement and risk assessments across multiple stocks.
- **Performance Metrics**: View confidence score distributions and recommendation breakdowns.
- **Portfolio Optimization**: Analyze risk vs. expected return for a basket of stocks.
- **Sector Analysis**: Understand sector-wise performance and recommendations.

### 🎯 Portfolio Builder

Build and analyze diversified portfolios. Add multiple stocks to a portfolio and let the agents collaborate to provide investment decisions, sector allocation analysis, and overall portfolio recommendations.

### ℹ️ About

Learn more about the Alpha Agents system, its research foundation, system architecture, and technical implementation details.

---

## 🧪 Testing

The project includes a comprehensive testing framework to ensure reliability and validate agent performance.

- **Unit Tests**: Individual tests for each agent and core components.
- **Integration Tests**: Tests for the multi-agent system and its collaboration mechanisms.
- **Data Generation**: Scripts to generate test data for analysis and visualization.

To run all tests, use the following command:

```bash
python tests/run_all_tests.py
```

Test results and generated data are stored in the `test-data/` directory.

---

## 🤝 Contributing

Contributions are welcome! If you have ideas for new features, improvements, or bug fixes, please open an issue or submit a pull request.

1. **Fork the repository**
2. **Create a new branch** (`git checkout -b feature/your-feature-name`)
3. **Make your changes**
4. **Commit your changes** (`git commit -m 'Add some feature'`)
5. **Push to the branch** (`git push origin feature/your-feature-name`)
6. **Open a pull request**

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 📞 Contact

For any questions or inquiries, please contact the development team.

- **Project Lead**: [Your Name](mailto:your-email@example.com)
- **GitHub Repository**: [https://github.com/predictivelabsai/alpha-agents](https://github.com/predictivelabsai/alpha-agents)

"""

