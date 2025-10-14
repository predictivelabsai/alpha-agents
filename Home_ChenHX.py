"""
Lohusalu Capital Management - Fundamental Screener
Simple landing page highlighting the screener and methodology.
"""

import streamlit as st

st.set_page_config(
    page_title="Lohusalu Capital Management",
    page_icon="🏛️",
    layout="wide",
)


def main():
    st.title("🏛️ Lohusalu Capital Management")
    st.markdown("**Advanced Financial Analysis Platform** – Comprehensive quantitative and qualitative research tools")

    # Create two main sections
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown(
            """
            ### 📊 Fundamental Screener
            A fast quantitative filter that ranks companies using public fundamentals.

            **🧠 Methodology (utils/stock_screener)**
            - Builds a universe from `utils/universe/us.csv` (ticker, sector, industry)
            - Pulls data via yfinance: financials, cash flow, balance sheet
            - Computes TTM margins (gross, operating, net, EBIT, EBITDA)
            - Calculates ROA/ROE/ROIC (average balance sheet denominators)
            - 4-year CAGR and consistency metrics for revenue, net income, OCF
            - Liquidity and leverage ratios: current, debt/EBITDA, debt service
            - Scores each company by percentile rank across metrics
            """
        )

        if st.button("📊 Open Fundamental Screener", type="primary"):
            st.switch_page("pages/1_Fundamental_Screener.py")

        st.info(
            "Universe file required: `utils/universe/us.csv`. If missing, create it with columns `ticker,sector,industry`."
        )

    with col2:
        st.markdown(
            """
            ### 🧠 QualAgent Enhanced Analysis
            Advanced qualitative analysis with multi-LLM support and interactive weight management.

            **🚀 Features**
            - Multi-LLM concurrent analysis (5+ models)
            - Enhanced scoring system (14 components)
            - Interactive weight configuration
            - Human feedback integration
            - Comprehensive risk assessment
            - Competitive moat analysis
            - Strategic insights evaluation
            - Multiple output formats (CSV, JSON, PKL, MD)
            """
        )

        if st.button("🧠 Open QualAgent Analysis", type="primary"):
            st.switch_page("pages/2_QualAgent_Analysis_ChenHX.py")

        st.info(
            "API keys required: TOGETHER_API_KEY, OPENAI_API_KEY. Add these to your .env file for full functionality."
        )

    st.markdown("---")
    st.caption("© Lohusalu Capital Management – quantitative research utilities")

if __name__ == "__main__":
    main()

