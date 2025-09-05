"""
Fundamental Agent - Quantitative Screening
Screens stocks based on strict quantitative criteria and intrinsic value analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sys
import os

# Add src to path
sys.path.append('src')
sys.path.append('.')

from agents.fundamental_agent import FundamentalAgent

st.set_page_config(
    page_title="Fundamental Agent - Lohusalu Capital",
    page_icon="📊",
    layout="wide"
)

def main():
    st.title("📊 Fundamental Agent")
    st.markdown("**Quantitative Screening & Intrinsic Value Analysis**")
    
    st.markdown("""
    The Fundamental Agent performs strict quantitative screening based on:
    - **Growth Metrics**: Revenue, earnings, and cash flow growth
    - **Profitability**: ROE, ROIC, margins, and efficiency ratios  
    - **Debt Health**: Leverage ratios and financial strength
    - **Intrinsic Value**: Valuation metrics and quality scores
    
    **Output**: List of qualified companies meeting quantitative criteria (not buy/sell recommendations)
    """)
    
    # Sidebar controls
    st.sidebar.header("Screening Parameters")
    
    # Market selection
    market = st.sidebar.selectbox(
        "Market",
        ["US", "China"],
        help="Select the stock market to screen"
    )
    
    # Sector selection
    sector = st.sidebar.selectbox(
        "Sector",
        [
            "Technology",
            "Healthcare", 
            "Consumer Discretionary",
            "Financials",
            "Communication Services",
            "Industrials",
            "Consumer Staples",
            "Energy",
            "Materials",
            "Real Estate",
            "Utilities"
        ],
        help="Select sector to screen"
    )
    
    # Market cap filter
    st.sidebar.subheader("Market Cap Filter")
    market_cap_option = st.sidebar.radio(
        "Focus",
        ["Microcap (<$1B)", "Small Cap ($1B-$5B)", "Mid Cap ($5B-$50B)", "Large Cap (>$50B)", "All Caps"]
    )
    
    # Map to actual values
    market_cap_limits = {
        "Microcap (<$1B)": (0, 1e9),
        "Small Cap ($1B-$5B)": (1e9, 5e9),
        "Mid Cap ($5B-$50B)": (5e9, 50e9),
        "Large Cap (>$50B)": (50e9, float('inf')),
        "All Caps": (0, float('inf'))
    }
    
    min_cap, max_cap = market_cap_limits[market_cap_option]
    
    # Advanced screening criteria
    with st.sidebar.expander("Advanced Criteria"):
        min_revenue_growth = st.slider("Min Revenue Growth (5Y %)", 0, 50, 10)
        min_roe = st.slider("Min ROE (%)", 0, 50, 15)
        min_roic = st.slider("Min ROIC (%)", 0, 50, 12)
        max_debt_ebitda = st.slider("Max Debt/EBITDA", 0.0, 10.0, 5.0)
        min_current_ratio = st.slider("Min Current Ratio", 0.5, 5.0, 1.0)
    
    # Number of companies to return
    top_n = st.sidebar.slider("Max Companies to Return", 5, 50, 20)
    
    # Run screening button
    if st.sidebar.button("🔍 Run Fundamental Screening", type="primary"):
        run_fundamental_screening(
            market, sector, min_cap, max_cap, top_n,
            min_revenue_growth, min_roe, min_roic, max_debt_ebitda, min_current_ratio
        )
    
    # Display recent results if available
    display_recent_results()

def run_fundamental_screening(market, sector, min_cap, max_cap, top_n, 
                            min_revenue_growth, min_roe, min_roic, max_debt_ebitda, min_current_ratio):
    """Run the fundamental screening process"""
    
    with st.spinner(f"🔍 Screening {sector} sector in {market} market..."):
        try:
            # Initialize agent
            agent = FundamentalAgent(use_llm=False)  # Fast mode without LLM
            
            # Custom screening criteria
            criteria = {
                'min_revenue_growth_5y': min_revenue_growth,
                'min_roe': min_roe,
                'min_roic': min_roic,
                'max_debt_to_ebitda': max_debt_ebitda,
                'min_current_ratio': min_current_ratio
            }
            
            # Run screening
            qualified_companies = agent.screen_sector(
                sector=sector,
                market=market,
                min_market_cap=min_cap,
                max_market_cap=max_cap,
                top_n=top_n,
                custom_criteria=criteria
            )
            
            if qualified_companies:
                st.success(f"✅ Found {len(qualified_companies)} qualified companies")
                display_screening_results(qualified_companies, sector, market)
            else:
                st.warning("⚠️ No companies found meeting the screening criteria. Try relaxing the parameters.")
                
        except Exception as e:
            st.error(f"❌ Error during screening: {str(e)}")
            st.info("💡 This might be due to data availability issues. Try a different sector or market.")

def display_screening_results(companies, sector, market):
    """Display the screening results"""
    
    st.header("🎯 Qualified Companies")
    
    # Convert to DataFrame for display
    df_data = []
    for company in companies:
        df_data.append({
            'Ticker': company.ticker,
            'Company': company.company_name,
            'Market Cap ($B)': company.market_cap / 1e9,
            'Overall Score': company.overall_score,
            'Growth Score': company.growth_score,
            'Profitability Score': company.profitability_score,
            'Debt Score': company.debt_score,
            'Revenue Growth (5Y %)': company.revenue_growth_5y,
            'ROE (%)': company.roe_ttm,
            'ROIC (%)': company.roic_ttm,
            'Current Ratio': company.current_ratio,
            'Debt/EBITDA': company.debt_to_ebitda
        })
    
    df = pd.DataFrame(df_data)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Companies Found", len(companies))
    
    with col2:
        avg_score = df['Overall Score'].mean()
        st.metric("Avg Overall Score", f"{avg_score:.1f}")
    
    with col3:
        avg_growth = df['Revenue Growth (5Y %)'].mean()
        st.metric("Avg Revenue Growth", f"{avg_growth:.1f}%")
    
    with col4:
        avg_roe = df['ROE (%)'].mean()
        st.metric("Avg ROE", f"{avg_roe:.1f}%")
    
    # Interactive table
    st.subheader("📋 Detailed Results")
    
    # Add filters
    col1, col2 = st.columns(2)
    
    with col1:
        min_score_filter = st.slider("Min Overall Score", 0, 100, 0)
    
    with col2:
        sort_by = st.selectbox("Sort By", 
                              ['Overall Score', 'Growth Score', 'Profitability Score', 'Market Cap ($B)'])
    
    # Apply filters
    filtered_df = df[df['Overall Score'] >= min_score_filter]
    filtered_df = filtered_df.sort_values(sort_by, ascending=False)
    
    # Display table
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Visualizations
    st.subheader("📊 Analysis Charts")
    
    tab1, tab2, tab3 = st.tabs(["Score Distribution", "Growth vs Profitability", "Risk Analysis"])
    
    with tab1:
        # Score distribution
        fig_scores = go.Figure()
        
        fig_scores.add_trace(go.Histogram(
            x=df['Overall Score'],
            name='Overall Score',
            opacity=0.7,
            nbinsx=10
        ))
        
        fig_scores.update_layout(
            title="Overall Score Distribution",
            xaxis_title="Score",
            yaxis_title="Count",
            showlegend=False
        )
        
        st.plotly_chart(fig_scores, use_container_width=True)
    
    with tab2:
        # Growth vs Profitability scatter
        fig_scatter = px.scatter(
            df,
            x='Growth Score',
            y='Profitability Score',
            size='Market Cap ($B)',
            color='Overall Score',
            hover_name='Ticker',
            hover_data=['Company', 'ROE (%)', 'Revenue Growth (5Y %)'],
            title="Growth vs Profitability Analysis"
        )
        
        fig_scatter.update_layout(
            xaxis_title="Growth Score",
            yaxis_title="Profitability Score"
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab3:
        # Risk analysis (debt metrics)
        fig_risk = px.scatter(
            df,
            x='Current Ratio',
            y='Debt/EBITDA',
            size='Market Cap ($B)',
            color='Debt Score',
            hover_name='Ticker',
            hover_data=['Company'],
            title="Financial Risk Analysis"
        )
        
        fig_risk.update_layout(
            xaxis_title="Current Ratio (Higher = Better Liquidity)",
            yaxis_title="Debt/EBITDA (Lower = Better)"
        )
        
        st.plotly_chart(fig_risk, use_container_width=True)
    
    # Export options
    st.subheader("💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📄 Download CSV",
            data=csv,
            file_name=f"fundamental_screening_{sector}_{market}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Prepare for next agent
        if st.button("➡️ Send to Rationale Agent", type="secondary"):
            st.session_state['qualified_companies'] = companies
            st.success("✅ Companies sent to Rationale Agent for qualitative analysis")
            st.info("👉 Go to the Rationale Agent page to continue the analysis")

def display_recent_results():
    """Display recent screening results if available"""
    
    if 'qualified_companies' in st.session_state and st.session_state['qualified_companies']:
        st.info("📋 Previous screening results are available in session state")
        
        if st.button("🔄 Show Previous Results"):
            display_screening_results(st.session_state['qualified_companies'], "Previous", "Session")

if __name__ == "__main__":
    main()

