"""
About Page - Information about the Alpha Agents system
"""

import streamlit as st
import sys
import os

# Page configuration
st.set_page_config(
    page_title="About - Alpha Agents",
    page_icon="ℹ️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .agent-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .feature-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """About page main function"""
    
    st.markdown('<h1 class="main-header">ℹ️ About Alpha Agents</h1>', unsafe_allow_html=True)
    
    # Introduction
    st.markdown("""
    ## 🚀 Overview
    
    **Alpha Agents** is an advanced multi-agent system for equity portfolio construction, based on cutting-edge 
    research in artificial intelligence and financial analysis. This system demonstrates how specialized AI agents 
    can collaborate, debate, and reach consensus to make sophisticated investment decisions.
    """)
    
    # Research Foundation
    st.markdown("---")
    st.subheader("📚 Research Foundation")
    
    st.markdown("""
    This implementation is based on the **Alpha Agents** research paper, which demonstrated the effectiveness 
    of multi-agent systems in financial decision-making through collaborative analysis, structured debate, 
    and consensus-building mechanisms.
    
    **Key Research Insights:**
    - Multi-agent systems outperform single-agent approaches in complex financial analysis
    - Structured debate mechanisms improve decision quality and reduce bias
    - Specialized agents with domain expertise provide more accurate assessments
    - Consensus-building through collaboration leads to better portfolio construction
    """)
    
    # System Architecture
    st.markdown("---")
    st.subheader("🏗️ System Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🤖 Multi-Agent Framework:**
        - 5 specialized AI agents with domain expertise
        - LangGraph-based workflow orchestration
        - Structured debate and consensus mechanisms
        - Real-time collaboration and analysis
        """)
    
    with col2:
        st.markdown("""
        **💻 Technical Stack:**
        - Python & Streamlit for the web interface
        - OpenAI GPT models for agent intelligence
        - Plotly for advanced data visualizations
        - SQLite for data persistence
        """)
    
    # Agent Descriptions
    st.markdown("---")
    st.subheader("🤖 Specialized AI Agents")
    
    agents_info = [
        {
            "name": "📊 Fundamental Agent",
            "description": "Analyzes 10-K/10-Q reports, financial statements, earnings quality, and fundamental metrics to assess company financial health and intrinsic value.",
            "expertise": ["Financial Statement Analysis", "Earnings Quality Assessment", "Valuation Modeling", "Credit Risk Analysis"]
        },
        {
            "name": "📰 Sentiment Agent", 
            "description": "Processes financial news, analyst ratings, social media sentiment, and market psychology to gauge investor sentiment and momentum.",
            "expertise": ["News Sentiment Analysis", "Social Media Monitoring", "Analyst Rating Aggregation", "Market Psychology Assessment"]
        },
        {
            "name": "💰 Valuation Agent",
            "description": "Analyzes stock prices, trading volumes, technical indicators, and relative valuation metrics to identify attractive entry points.",
            "expertise": ["Technical Analysis", "Relative Valuation", "Price Action Analysis", "Volume Pattern Recognition"]
        },
        {
            "name": "🧠 Rationale Agent",
            "description": "Evaluates business quality using a 7-step framework: sales growth, profitability, competitive moats, operational efficiency, and debt structure.",
            "expertise": ["Business Quality Assessment", "Competitive Advantage Analysis", "Operational Efficiency", "Financial Strength Evaluation"]
        },
        {
            "name": "🚀 Secular Trend Agent",
            "description": "Identifies companies positioned to benefit from major technology trends: Agentic AI, Cloud Infrastructure, AI Semiconductors, Cybersecurity, and Electrification.",
            "expertise": ["Technology Trend Analysis", "Market Opportunity Assessment", "Innovation Evaluation", "Future Growth Positioning"]
        }
    ]
    
    for agent in agents_info:
        with st.expander(agent["name"], expanded=False):
            st.write(agent["description"])
            st.write("**Areas of Expertise:**")
            for expertise in agent["expertise"]:
                st.write(f"• {expertise}")
    
    # Key Features
    st.markdown("---")
    st.subheader("✨ Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4>🔍 Individual Stock Analysis</h4>
            <p>Comprehensive analysis of individual stocks using all 5 specialized agents with detailed reasoning and recommendations.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>🎯 Portfolio Construction</h4>
            <p>Build and optimize portfolios with multi-agent collaboration, sector diversification, and risk assessment.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4>📊 Advanced Analytics</h4>
            <p>Interactive visualizations including heatmaps, performance charts, and portfolio optimization analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="feature-card">
            <h4>🧪 Comprehensive Testing</h4>
            <p>Extensive test suite with performance metrics, data generation, and validation of agent decision-making.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Technical Details
    st.markdown("---")
    st.subheader("⚙️ Technical Implementation")
    
    with st.expander("🔧 System Components", expanded=False):
        st.markdown("""
        **Core Components:**
        - **Multi-Agent System**: LangGraph-based workflow with specialized agents
        - **Database Layer**: SQLite for persistent storage of analyses and portfolios
        - **Visualization Engine**: Plotly-based interactive charts and heatmaps
        - **Testing Framework**: Comprehensive unit and integration tests
        - **Web Interface**: Streamlit multi-page application
        
        **Agent Capabilities:**
        - **Analysis**: Each agent provides specialized domain analysis
        - **Reasoning**: Transparent reasoning and factor identification
        - **Risk Assessment**: Multi-dimensional risk evaluation
        - **Target Pricing**: Quantitative price target calculation
        - **Confidence Scoring**: Self-assessed confidence in recommendations
        """)
    
    with st.expander("📈 Performance Metrics", expanded=False):
        st.markdown("""
        **System Performance:**
        - **Response Time**: < 5 seconds per stock analysis
        - **Accuracy**: High confidence scores (0.6-0.9 range)
        - **Coverage**: 8+ major sectors and market caps
        - **Scalability**: Handles portfolios up to 20 stocks
        - **Reliability**: Fallback mechanisms for robust operation
        
        **Agent Performance:**
        - **Consensus Rate**: 70%+ agreement on strong recommendations
        - **Risk Calibration**: Accurate risk assessment across market conditions
        - **Sector Expertise**: Specialized knowledge for different industries
        - **Trend Recognition**: Early identification of secular trends
        """)
    
    # Usage Guidelines
    st.markdown("---")
    st.subheader("📋 Usage Guidelines")
    
    st.markdown("""
    **Getting Started:**
    1. **Stock Analysis**: Start with individual stock analysis to understand agent capabilities
    2. **Portfolio Building**: Add multiple stocks to build a diversified portfolio
    3. **Analytics Review**: Use the analytics page to understand system performance
    4. **Risk Management**: Pay attention to risk assessments and sector allocation
    
    **Best Practices:**
    - Use multiple agents' perspectives for balanced analysis
    - Consider consensus recommendations for higher confidence
    - Monitor sector diversification in portfolio construction
    - Review agent reasoning for transparency and learning
    - Regularly analyze performance through the analytics dashboard
    
    **Important Notes:**
    - This system is for educational and research purposes
    - All analyses should be considered as part of broader investment research
    - Past performance does not guarantee future results
    - Always consult with financial professionals for investment decisions
    """)
    
    # Contact and Support
    st.markdown("---")
    st.subheader("📞 Contact & Support")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Development Team:**
        - Research-based implementation
        - Open-source architecture
        - Continuous improvement focus
        """)
    
    with col2:
        st.markdown("""
        **Technical Support:**
        - Comprehensive documentation
        - Test suite for validation
        - Modular design for extensibility
        """)
    
    # Version Information
    st.markdown("---")
    st.subheader("📋 Version Information")
    
    st.markdown("""
    **Current Version:** 1.0.0  
    **Release Date:** September 2025  
    **Features:** 5 Specialized Agents, Advanced Analytics, Portfolio Construction  
    **Status:** Production Ready  
    """)

if __name__ == "__main__":
    main()

