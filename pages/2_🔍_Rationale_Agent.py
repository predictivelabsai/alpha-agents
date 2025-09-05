"""
Rationale Agent - Qualitative Analysis
Analyzes competitive moats, sentiment, and secular trends using web search
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

from agents.rationale_agent_updated import RationaleAgent

st.set_page_config(
    page_title="Rationale Agent - Lohusalu Capital",
    page_icon="🔍",
    layout="wide"
)

def main():
    st.title("🔍 Rationale Agent")
    st.markdown("**Qualitative Analysis with Web Research**")
    
    st.markdown("""
    The Rationale Agent performs comprehensive qualitative analysis using:
    - **🏰 Competitive Moats**: Brand strength, network effects, switching costs
    - **📈 Market Sentiment**: News analysis, analyst opinions, momentum indicators
    - **🌊 Secular Trends**: Industry trends, ESG factors, technological disruption
    - **🔍 Tavily Search**: Heavy web research with citations for each analysis
    
    **Output**: Qualitative scores (1-10 scale) with detailed reasoning and citations
    """)
    
    # Check for input from Fundamental Agent
    if 'qualified_companies' not in st.session_state or not st.session_state['qualified_companies']:
        st.warning("⚠️ No qualified companies found. Please run the Fundamental Agent first.")
        
        if st.button("👈 Go to Fundamental Agent"):
            st.switch_page("pages/1_📊_Fundamental_Agent.py")
        
        # Allow manual ticker input for testing
        st.subheader("🧪 Manual Testing")
        manual_ticker = st.text_input("Enter ticker for testing (e.g., AAPL, NVDA)")
        
        if manual_ticker and st.button("Analyze Single Company"):
            analyze_manual_ticker(manual_ticker)
        
        return
    
    qualified_companies = st.session_state['qualified_companies']
    
    # Display input companies
    st.subheader("📋 Input from Fundamental Agent")
    
    input_df = pd.DataFrame([{
        'Ticker': c.ticker,
        'Company': c.company_name,
        'Sector': c.sector,
        'Overall Score': c.overall_score,
        'Market Cap ($B)': c.market_cap / 1e9
    } for c in qualified_companies])
    
    st.dataframe(input_df, use_container_width=True, hide_index=True)
    
    # Analysis controls
    st.sidebar.header("Analysis Settings")
    
    # API key check
    tavily_key = st.sidebar.text_input("Tavily API Key (Optional)", type="password", 
                                      help="Enter Tavily API key for web search. Leave empty for fallback mode.")
    
    openai_key = st.sidebar.text_input("OpenAI API Key (Optional)", type="password",
                                      help="Enter OpenAI API key for LLM analysis. Leave empty for fallback mode.")
    
    # Analysis depth
    analysis_depth = st.sidebar.selectbox(
        "Analysis Depth",
        ["Quick (Fallback Mode)", "Standard (With LLM)", "Deep (LLM + Web Search)"],
        help="Choose analysis depth based on available API keys"
    )
    
    # Company selection
    selected_companies = st.sidebar.multiselect(
        "Select Companies to Analyze",
        options=[c.ticker for c in qualified_companies],
        default=[c.ticker for c in qualified_companies[:5]],  # Default to first 5
        help="Select which companies to analyze (limit for performance)"
    )
    
    # Run analysis button
    if st.sidebar.button("🔍 Run Qualitative Analysis", type="primary"):
        if selected_companies:
            run_rationale_analysis(qualified_companies, selected_companies, tavily_key, openai_key, analysis_depth)
        else:
            st.error("Please select at least one company to analyze")
    
    # Display recent results if available
    display_recent_results()

def analyze_manual_ticker(ticker):
    """Analyze a single ticker manually for testing"""
    
    with st.spinner(f"🔍 Analyzing {ticker}..."):
        try:
            # Create a mock qualified company for testing
            from agents.fundamental_agent import QualifiedCompany
            
            mock_company = QualifiedCompany(
                ticker=ticker,
                company_name=f"{ticker} Corporation",
                sector="Technology",
                market_cap=1000000000000,  # $1T
                revenue_growth_5y=15.0,
                net_income_growth_5y=20.0,
                cash_flow_growth_5y=18.0,
                roe_ttm=20.0,
                roic_ttm=15.0,
                gross_margin=40.0,
                profit_margin=25.0,
                current_ratio=2.0,
                debt_to_ebitda=2.5,
                debt_service_ratio=0.3,
                growth_score=75.0,
                profitability_score=80.0,
                debt_score=70.0,
                overall_score=75.0,
                commentary="Manual test entry",
                timestamp=datetime.now().isoformat()
            )
            
            # Initialize agent
            agent = RationaleAgent(api_key=None, tavily_api_key=None)
            
            # Analyze
            analysis = agent._analyze_individual_company(mock_company)
            
            # Display results
            display_single_analysis(analysis, mock_company)
            
        except Exception as e:
            st.error(f"❌ Error analyzing {ticker}: {str(e)}")

def run_rationale_analysis(qualified_companies, selected_tickers, tavily_key, openai_key, analysis_depth):
    """Run the rationale analysis process"""
    
    # Filter to selected companies
    companies_to_analyze = [c for c in qualified_companies if c.ticker in selected_tickers]
    
    with st.spinner(f"🔍 Performing qualitative analysis on {len(companies_to_analyze)} companies..."):
        try:
            # Initialize agent with API keys
            agent = RationaleAgent(
                api_key=openai_key if openai_key else None,
                tavily_api_key=tavily_key if tavily_key else None
            )
            
            # Set analysis mode based on depth
            if analysis_depth == "Quick (Fallback Mode)":
                agent.llm = None
                agent.tavily_client = None
            elif analysis_depth == "Standard (With LLM)" and not openai_key:
                st.warning("⚠️ OpenAI API key required for Standard mode. Using fallback mode.")
            elif analysis_depth == "Deep (LLM + Web Search)" and (not openai_key or not tavily_key):
                st.warning("⚠️ Both API keys required for Deep mode. Using available capabilities.")
            
            # Run analysis
            analyses = agent.analyze_companies(companies_to_analyze)
            
            if analyses:
                st.success(f"✅ Completed qualitative analysis for {len(analyses)} companies")
                
                # Store results in session state
                st.session_state['rationale_analyses'] = analyses
                
                display_analysis_results(analyses, companies_to_analyze)
            else:
                st.warning("⚠️ No analyses completed. Check the logs for errors.")
                
        except Exception as e:
            st.error(f"❌ Error during analysis: {str(e)}")

def display_single_analysis(analysis, company):
    """Display analysis for a single company"""
    
    st.subheader(f"📊 Analysis: {analysis.ticker}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Overall Qualitative Score", f"{analysis.overall_qualitative_score:.1f}/10")
    
    with col2:
        st.metric("Moat Score", f"{analysis.moat_score:.1f}/10", 
                 help=f"{analysis.moat_strength} {analysis.moat_type}")
    
    with col3:
        st.metric("Sentiment Score", f"{analysis.sentiment_score:.1f}/10",
                 help=analysis.sentiment_trend)
    
    with col4:
        st.metric("Trend Score", f"{analysis.trend_score:.1f}/10",
                 help=analysis.trend_alignment)
    
    # Key insights
    if analysis.key_insights:
        st.subheader("💡 Key Insights")
        for insight in analysis.key_insights:
            st.write(f"• {insight}")
    
    # Citations
    if analysis.citations:
        st.subheader("📚 Research Citations")
        for i, citation in enumerate(analysis.citations, 1):
            st.write(f"{i}. {citation}")

def display_analysis_results(analyses, companies):
    """Display the complete analysis results"""
    
    st.header("🎯 Qualitative Analysis Results")
    
    # Convert to DataFrame for display
    df_data = []
    for ticker, analysis in analyses.items():
        company = next((c for c in companies if c.ticker == ticker), None)
        
        df_data.append({
            'Ticker': ticker,
            'Company': analysis.company_name,
            'Overall Qualitative Score': analysis.overall_qualitative_score,
            'Moat Score': analysis.moat_score,
            'Moat Type': analysis.moat_type,
            'Moat Strength': analysis.moat_strength,
            'Sentiment Score': analysis.sentiment_score,
            'Sentiment Trend': analysis.sentiment_trend,
            'Trend Score': analysis.trend_score,
            'Trend Alignment': analysis.trend_alignment,
            'Key Insights Count': len(analysis.key_insights),
            'Citations Count': len(analysis.citations)
        })
    
    df = pd.DataFrame(df_data)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Companies Analyzed", len(analyses))
    
    with col2:
        avg_qual_score = df['Overall Qualitative Score'].mean()
        st.metric("Avg Qualitative Score", f"{avg_qual_score:.1f}/10")
    
    with col3:
        avg_moat_score = df['Moat Score'].mean()
        st.metric("Avg Moat Score", f"{avg_moat_score:.1f}/10")
    
    with col4:
        total_insights = df['Key Insights Count'].sum()
        st.metric("Total Insights", total_insights)
    
    # Interactive table
    st.subheader("📋 Detailed Results")
    
    # Sorting options
    sort_by = st.selectbox("Sort By", 
                          ['Overall Qualitative Score', 'Moat Score', 'Sentiment Score', 'Trend Score'])
    
    sorted_df = df.sort_values(sort_by, ascending=False)
    
    st.dataframe(
        sorted_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Detailed analysis for each company
    st.subheader("🔍 Detailed Company Analysis")
    
    selected_ticker = st.selectbox("Select Company for Details", 
                                  options=list(analyses.keys()))
    
    if selected_ticker:
        analysis = analyses[selected_ticker]
        company = next((c for c in companies if c.ticker == selected_ticker), None)
        
        display_single_analysis(analysis, company)
    
    # Visualizations
    st.subheader("📊 Analysis Charts")
    
    tab1, tab2, tab3 = st.tabs(["Score Comparison", "Moat Analysis", "Trend vs Sentiment"])
    
    with tab1:
        # Score comparison radar chart
        fig_radar = go.Figure()
        
        for _, row in df.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row['Moat Score'], row['Sentiment Score'], row['Trend Score'], row['Overall Qualitative Score']],
                theta=['Moat', 'Sentiment', 'Trends', 'Overall'],
                fill='toself',
                name=row['Ticker']
            ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )),
            showlegend=True,
            title="Qualitative Score Comparison"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with tab2:
        # Moat analysis
        fig_moat = px.scatter(
            df,
            x='Moat Score',
            y='Overall Qualitative Score',
            color='Moat Type',
            size='Key Insights Count',
            hover_name='Ticker',
            hover_data=['Moat Strength'],
            title="Competitive Moat Analysis"
        )
        
        st.plotly_chart(fig_moat, use_container_width=True)
    
    with tab3:
        # Trend vs Sentiment
        fig_trend = px.scatter(
            df,
            x='Trend Score',
            y='Sentiment Score',
            color='Overall Qualitative Score',
            size='Citations Count',
            hover_name='Ticker',
            hover_data=['Trend Alignment', 'Sentiment Trend'],
            title="Secular Trends vs Market Sentiment"
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
    
    # Export and next steps
    st.subheader("💾 Export & Next Steps")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📄 Download Analysis CSV",
            data=csv,
            file_name=f"rationale_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Send to Ranker Agent
        if st.button("➡️ Send to Ranker Agent", type="secondary"):
            st.success("✅ Analysis sent to Ranker Agent for final scoring")
            st.info("👉 Go to the Ranker Agent page to get final investment recommendations")

def display_recent_results():
    """Display recent analysis results if available"""
    
    if 'rationale_analyses' in st.session_state and st.session_state['rationale_analyses']:
        st.info("📋 Previous analysis results are available in session state")
        
        if st.button("🔄 Show Previous Results"):
            qualified_companies = st.session_state.get('qualified_companies', [])
            display_analysis_results(st.session_state['rationale_analyses'], qualified_companies)

if __name__ == "__main__":
    main()

