# Lohusalu Capital Management - Multi-Agent Equity Analysis Methodology

## Executive Summary

The Lohusalu Capital Management system implements a sophisticated multi-agent framework for equity portfolio construction, based on cutting-edge research in AI-driven financial analysis. The system employs a **3-agent pipeline** that collaborates to provide comprehensive stock analysis and portfolio recommendations.

This methodology document outlines the systematic approach used by each agent, their collaboration mechanisms, and the final synthesis process that produces actionable investment recommendations.

## System Architecture

### Multi-Agent Framework
- **Architecture**: LangGraph-based collaborative system
- **Agent Count**: 6 specialized agents (5 analysis + 1 ranking)
- **Collaboration Method**: Sequential analysis with consensus building
- **Decision Aggregation**: Weighted confidence scoring with final synthesis
- **Quality Assurance**: Confidence scoring + risk assessment + fallback modes
- **Data Integration**: Real-time yfinance API integration

### Core Components
1. **Individual Agent Analysis**: Each agent provides specialized domain expertise
2. **Multi-Agent Collaboration**: Agents analyze independently then collaborate
3. **Ranking & Synthesis**: Final agent synthesizes all analyses into unified recommendations
4. **Portfolio Construction**: Optimized portfolio recommendations with risk management
5. **Performance Monitoring**: Continuous system performance tracking with reasoning traces
6. **Chain of Thought Logging**: Complete transparency in agent decision-making

## Agent Methodologies

### 1. **📊 Fundamental Agent**

**Primary Function**: Sector analysis and quantitative stock screening.

**Methodology**:
- **Sector Analysis**: Identifies trending sectors by analyzing market data and economic indicators. Assigns weights to sectors based on growth potential and momentum.
- **Quantitative Screening**: Screens stocks within selected sectors against a rigorous set of financial metrics, including:
    - **Growth**: Revenue and EPS growth (YoY and QoQ)
    - **Profitability**: ROE, ROA, and net profit margins
    - **Valuation**: P/E, P/S, and P/B ratios relative to industry peers
    - **Financial Health**: Debt-to-equity and current ratios
- **Intrinsic Value**: Calculates a DCF-based intrinsic value to determine upside potential.

### 2. **🔍 Rationale Agent**

**Primary Function**: Qualitative analysis of a company's competitive advantages and market position.

**Methodology**:
- **Economic Moat**: Assesses the strength and durability of a company's competitive advantages (e.g., network effects, brand loyalty, switching costs).
- **Sentiment Analysis**: Analyzes market sentiment through news, social media, and analyst ratings.
- **Secular Trends**: Identifies long-term secular trends and evaluates the company's alignment with them.
- **Tavily Search**: Conducts extensive web searches to gather qualitative data and support its analysis with citations.

### 3. **🎯 Ranker Agent**

**Primary Function**: Synthesizes the analysis from the Fundamental and Rationale agents to provide a final investment recommendation.

**Methodology**:
- **Composite Scoring**: Calculates a composite score based on a weighted average of the fundamental (60%) and qualitative (40%) scores.
- **Investment Grading**: Assigns an investment grade (A+ to D) based on the composite score.
- **Investment Thesis**: Generates a comprehensive investment thesis that outlines the key strengths, risks, and potential catalysts for the stock.
- **Portfolio Construction**: Provides a ranked list of investment opportunities and a sample portfolio allocation.

## Multi-Agent Collaboration Process

### Phase 1: Independent Analysis
1. Each of the 5 analysis agents processes the same stock independently
2. Agents use specialized prompts and methodologies
3. Real-time data integration from yfinance API
4. Individual recommendations and confidence scores generated

### Phase 2: Data Aggregation
1. All agent analyses collected by the Ranking Agent
2. Consistency checks and data validation performed
3. Agent-specific insights and concerns catalogued
4. Confidence scores and recommendations compiled

### Phase 3: Consensus Building
1. Agreement levels assessed across all agents
2. Conflicting viewpoints identified and analyzed
3. Confidence weighting applied to resolve disagreements
4. Consensus recommendation methodology applied

### Phase 4: Final Synthesis
1. Multi-factor composite scoring calculation
2. Risk-reward assessment integration
3. Investment thesis development
4. Final recommendation and price target determination

### Phase 5: Portfolio Integration
1. Individual stock rankings across all analyzed securities
2. Portfolio construction optimization
3. Risk management and diversification analysis
4. Position sizing recommendations

## Quality Assurance & Risk Management

### Confidence Scoring System
- Each agent provides confidence scores (0.0-1.0)
- Weighted averaging based on agent specialization
- Minimum confidence thresholds for recommendations
- Uncertainty quantification and communication

### Fallback Mechanisms
- Graceful degradation when API calls fail
- Default analysis modes for system reliability
- Error handling and logging for debugging
- Alternative data source integration capabilities

### Risk Assessment Framework
- Multi-dimensional risk evaluation across all agents
- Risk correlation analysis between different factors
- Risk mitigation strategy recommendations
- Continuous risk monitoring and alerting

### Performance Monitoring
- Agent performance tracking over time
- Recommendation accuracy measurement
- Confidence calibration assessment
- System performance optimization

## Technology Implementation

### LangGraph Framework
- State machine-based agent coordination
- Parallel processing capabilities for efficiency
- Robust error handling and recovery
- Scalable architecture for additional agents

### Data Integration
- Real-time yfinance API integration
- Comprehensive financial data coverage
- Historical data analysis capabilities
- Data quality validation and cleansing

### Logging & Monitoring
- Complete chain of thought logging
- Agent reasoning trace capture
- Performance metrics tracking
- System health monitoring

### User Interface
- Multi-page Streamlit application
- Interactive visualizations and charts
- Real-time analysis capabilities
- Professional portfolio management interface

## Conclusion

The Lohusalu Capital Management multi-agent system represents a sophisticated approach to equity analysis that combines the strengths of multiple analytical perspectives while mitigating individual agent limitations. Through systematic collaboration and synthesis, the system provides comprehensive, transparent, and actionable investment recommendations.

The 6-agent architecture ensures thorough coverage of all critical investment factors while maintaining the flexibility to adapt to changing market conditions and investment requirements. The systematic methodology, combined with real-time data integration and comprehensive monitoring, creates a robust platform for professional equity portfolio management.

---

*This methodology document reflects the current implementation of the Lohusalu Capital Management multi-agent system and is subject to continuous improvement and refinement based on performance feedback and market evolution.*

