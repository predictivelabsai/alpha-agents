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
    st.markdown("**Fundamental Screener** – quantitative filtering powered by `utils/stock_screener`.")

    st.markdown(
        """
        ### 📊 What is the Fundamental Screener?
        A fast quantitative filter that ranks companies using public fundamentals.
        
        ### 🧠 Methodology (utils/stock_screener)
        - Builds a universe from `utils/universe/us.csv` (ticker, sector, industry)
        - Pulls data via yfinance: financials, cash flow, balance sheet
        - Computes TTM margins (gross, operating, net, EBIT, EBITDA)
        - Calculates ROA/ROE/ROIC (average balance sheet denominators)
        - 4-year CAGR and consistency metrics for revenue, net income, OCF
        - Liquidity and leverage ratios: current, debt/EBITDA, debt service
        - Scores each company by percentile rank across metrics
        
        ### 🚀 Get Started
        Run the screener with your chosen sector or industry and export results.
        """
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Open Fundamental Screener", type="primary"):
            st.switch_page("pages/1_Fundamental_Screener.py")

    with col2:
        st.info(
            "Universe file required: `utils/universe/us.csv`. If missing, create it with columns `ticker,sector,industry`."
        )

    st.markdown("---")
    st.caption("© Lohusalu Capital Management – quantitative research utilities")

if __name__ == "__main__":
    main()

