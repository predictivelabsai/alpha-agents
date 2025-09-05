"""
Ranker Agent - Final Scoring & Investment Recommendations
Takes inputs from Fundamental and Rationale agents to provide final 1-10 scoring
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

from agents.ranker_agent import RankerAgent

st.set_page_config(
    page_title="Ranker Agent - Lohusalu Capital",
    page_icon="🏆",
    layout="wide"
)

def main():
    st.title("🏆 Ranker Agent")
    st.markdown("**Final Scoring & Investment Recommendations**")
    
    st.markdown("""
    The Ranker Agent combines quantitative and qualitative analysis to provide:
    - **🎯 Final Investment Score**: 1-10 scale combining fundamental (50%) + qualitative (50%)
    - **💡 Investment Thesis**: Compelling investment case for each company
    - **🤔 Why Good Investment**: Detailed reasoning explaining the score across multiple dimensions
    - **📊 Recommendations**: STRONG_BUY/BUY/HOLD/SELL with confidence levels
    - **📈 Position Sizing**: LARGE/MEDIUM/SMALL based on score, confidence, and market cap
    - **🎯 Portfolio Construction**: Optimized portfolio recommendations with weights
    """)
    
    # Check for inputs from previous agents
    qualified_companies = st.session_state.get('qualified_companies', [])
    rationale_analyses = st.session_state.get('rationale_analyses', {})
    
    if not qualified_companies or not rationale_analyses:
        st.warning("⚠️ Missing inputs from previous agents. Please complete the analysis pipeline:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if not qualified_companies:
                st.error("❌ No qualified companies from Fundamental Agent")
                if st.button("👈 Go to Fundamental Agent"):
                    st.switch_page("pages/1_📊_Fundamental_Agent.py")
        
        with col2:
            if not rationale_analyses:
                st.error("❌ No qualitative analyses from Rationale Agent")
                if st.button("👈 Go to Rationale Agent"):
                    st.switch_page("pages/2_🔍_Rationale_Agent.py")
        
        # Allow manual testing
        st.subheader("🧪 Manual Testing")
        if st.button("Generate Sample Data for Testing"):
            generate_sample_data()
        
        return
    
    # Display pipeline inputs
    display_pipeline_inputs(qualified_companies, rationale_analyses)
    
    # Ranking controls
    st.sidebar.header("Ranking Settings")
    
    # API key for LLM analysis
    openai_key = st.sidebar.text_input("OpenAI API Key (Optional)", type="password",
                                      help="Enter OpenAI API key for detailed LLM analysis. Leave empty for fallback mode.")
    
    # Weighting scheme
    st.sidebar.subheader("Score Weighting")
    fundamental_weight = st.sidebar.slider("Fundamental Weight (%)", 0, 100, 50) / 100
    qualitative_weight = 1 - fundamental_weight
    
    st.sidebar.write(f"Qualitative Weight: {qualitative_weight:.0%}")
    
    # Portfolio settings
    st.sidebar.subheader("Portfolio Settings")
    max_positions = st.sidebar.slider("Max Portfolio Positions", 3, 20, 10)
    
    # Risk preferences
    risk_tolerance = st.sidebar.selectbox(
        "Risk Tolerance",
        ["Conservative", "Moderate", "Aggressive"],
        help="Affects position sizing and recommendation thresholds"
    )
    
    # Run ranking button
    if st.sidebar.button("🏆 Generate Final Rankings", type="primary"):
        run_final_ranking(qualified_companies, rationale_analyses, openai_key, 
                         fundamental_weight, qualitative_weight, max_positions, risk_tolerance)
    
    # Display recent results if available
    display_recent_results()

def generate_sample_data():
    """Generate sample data for testing"""
    
    from agents.fundamental_agent import QualifiedCompany
    from agents.rationale_agent_updated import RationaleAnalysis
    
    # Sample companies
    sample_companies = [
        QualifiedCompany(
            ticker='AAPL',
            company_name='Apple Inc.',
            sector='Technology',
            market_cap=3000000000000,
            revenue_growth_5y=8.0,
            net_income_growth_5y=10.0,
            cash_flow_growth_5y=9.0,
            roe_ttm=15.0,
            roic_ttm=12.0,
            gross_margin=40.0,
            profit_margin=25.0,
            current_ratio=1.1,
            debt_to_ebitda=2.5,
            debt_service_ratio=0.3,
            growth_score=70.0,
            profitability_score=80.0,
            debt_score=75.0,
            overall_score=75.0,
            commentary="Strong brand and ecosystem",
            timestamp=datetime.now().isoformat()
        ),
        QualifiedCompany(
            ticker='NVDA',
            company_name='NVIDIA Corporation',
            sector='Technology',
            market_cap=1500000000000,
            revenue_growth_5y=60.0,
            net_income_growth_5y=80.0,
            cash_flow_growth_5y=70.0,
            roe_ttm=35.0,
            roic_ttm=25.0,
            gross_margin=70.0,
            profit_margin=30.0,
            current_ratio=3.5,
            debt_to_ebitda=1.2,
            debt_service_ratio=0.1,
            growth_score=95.0,
            profitability_score=90.0,
            debt_score=85.0,
            overall_score=90.0,
            commentary="AI leadership and GPU dominance",
            timestamp=datetime.now().isoformat()
        )
    ]
    
    # Sample analyses
    sample_analyses = {
        'AAPL': RationaleAnalysis(
            ticker='AAPL',
            company_name='Apple Inc.',
            overall_qualitative_score=7.5,
            moat_score=8.5,
            moat_type='Brand',
            moat_strength='Strong',
            sentiment_score=7.0,
            sentiment_trend='Positive',
            trend_score=7.0,
            trend_alignment='Favorable',
            key_insights=['Strong ecosystem', 'Brand loyalty', 'Premium positioning'],
            citations=['Sample analysis'],
            timestamp=datetime.now().isoformat()
        ),
        'NVDA': RationaleAnalysis(
            ticker='NVDA',
            company_name='NVIDIA Corporation',
            overall_qualitative_score=9.0,
            moat_score=9.5,
            moat_type='Technology',
            moat_strength='Very Strong',
            sentiment_score=8.5,
            sentiment_trend='Very Positive',
            trend_score=9.5,
            trend_alignment='Highly Favorable',
            key_insights=['AI leadership', 'GPU dominance', 'Data center growth'],
            citations=['Sample analysis'],
            timestamp=datetime.now().isoformat()
        )
    }
    
    # Store in session state
    st.session_state['qualified_companies'] = sample_companies
    st.session_state['rationale_analyses'] = sample_analyses
    
    st.success("✅ Sample data generated! You can now run the ranking analysis.")
    st.rerun()

def display_pipeline_inputs(qualified_companies, rationale_analyses):
    """Display inputs from the analysis pipeline"""
    
    st.subheader("📋 Analysis Pipeline Inputs")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📊 Fundamental Agent Output**")
        fund_df = pd.DataFrame([{
            'Ticker': c.ticker,
            'Company': c.company_name,
            'Overall Score': c.overall_score,
            'Market Cap ($B)': c.market_cap / 1e9
        } for c in qualified_companies])
        
        st.dataframe(fund_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.write("**🔍 Rationale Agent Output**")
        rat_df = pd.DataFrame([{
            'Ticker': ticker,
            'Qualitative Score': analysis.overall_qualitative_score,
            'Moat Score': analysis.moat_score,
            'Sentiment Score': analysis.sentiment_score
        } for ticker, analysis in rationale_analyses.items()])
        
        st.dataframe(rat_df, use_container_width=True, hide_index=True)

def run_final_ranking(qualified_companies, rationale_analyses, openai_key, 
                     fundamental_weight, qualitative_weight, max_positions, risk_tolerance):
    """Run the final ranking process"""
    
    with st.spinner(f"🏆 Generating final rankings for {len(qualified_companies)} companies..."):
        try:
            # Initialize agent
            agent = RankerAgent(api_key=openai_key if openai_key else None)
            
            # Adjust weighting if needed (for future enhancement)
            # Currently uses 50/50 weighting in the agent
            
            # Run ranking
            ranked_companies = agent.rank_companies(
                qualified_companies=qualified_companies,
                rationale_analyses=rationale_analyses
            )
            
            if ranked_companies:
                st.success(f"✅ Generated final rankings for {len(ranked_companies)} companies")
                
                # Store results
                st.session_state['ranked_companies'] = ranked_companies
                
                # Display results
                display_ranking_results(ranked_companies, max_positions, risk_tolerance)
                
                # Generate portfolio recommendations
                portfolio_recs = agent.get_portfolio_recommendations(ranked_companies, max_positions)
                display_portfolio_recommendations(portfolio_recs)
                
            else:
                st.warning("⚠️ No rankings generated. Check the logs for errors.")
                
        except Exception as e:
            st.error(f"❌ Error during ranking: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

def display_ranking_results(ranked_companies, max_positions, risk_tolerance):
    """Display the final ranking results"""
    
    st.header("🎯 Final Investment Rankings")
    
    # Convert to DataFrame
    df_data = []
    for i, company in enumerate(ranked_companies, 1):
        df_data.append({
            'Rank': i,
            'Ticker': company.ticker,
            'Company': company.company_name,
            'Final Score': company.final_investment_score,
            'Fundamental Score': company.fundamental_score,
            'Qualitative Score': company.qualitative_score,
            'Recommendation': company.recommendation,
            'Confidence': company.confidence_level,
            'Position Size': company.position_size,
            'Market Cap ($B)': company.market_cap / 1e9,
            'Investment Thesis': company.investment_thesis[:100] + "..." if len(company.investment_thesis) > 100 else company.investment_thesis
        })
    
    df = pd.DataFrame(df_data)
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Companies Ranked", len(ranked_companies))
    
    with col2:
        avg_score = df['Final Score'].mean()
        st.metric("Avg Final Score", f"{avg_score:.1f}/10")
    
    with col3:
        strong_buys = len([c for c in ranked_companies if c.recommendation == 'STRONG_BUY'])
        st.metric("Strong Buy Count", strong_buys)
    
    with col4:
        high_confidence = len([c for c in ranked_companies if c.confidence_level == 'HIGH'])
        st.metric("High Confidence", high_confidence)
    
    # Interactive table
    st.subheader("📋 Detailed Rankings")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_score_filter = st.slider("Min Final Score", 1.0, 10.0, 1.0, 0.1)
    
    with col2:
        rec_filter = st.multiselect("Recommendations", 
                                   ['STRONG_BUY', 'BUY', 'HOLD', 'SELL'],
                                   default=['STRONG_BUY', 'BUY'])
    
    with col3:
        conf_filter = st.multiselect("Confidence Levels",
                                    ['HIGH', 'MEDIUM', 'LOW'],
                                    default=['HIGH', 'MEDIUM'])
    
    # Apply filters
    filtered_df = df[
        (df['Final Score'] >= min_score_filter) &
        (df['Recommendation'].isin(rec_filter)) &
        (df['Confidence'].isin(conf_filter))
    ]
    
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Detailed company analysis
    st.subheader("🔍 Detailed Investment Analysis")
    
    selected_ticker = st.selectbox("Select Company for Detailed Analysis", 
                                  options=[c.ticker for c in ranked_companies])
    
    if selected_ticker:
        company = next((c for c in ranked_companies if c.ticker == selected_ticker), None)
        if company:
            display_detailed_analysis(company)
    
    # Visualizations
    st.subheader("📊 Ranking Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["Score Breakdown", "Risk-Return", "Recommendations"])
    
    with tab1:
        # Score breakdown chart
        fig_scores = go.Figure()
        
        fig_scores.add_trace(go.Bar(
            name='Fundamental Score',
            x=df['Ticker'],
            y=df['Fundamental Score'],
            marker_color='lightblue'
        ))
        
        fig_scores.add_trace(go.Bar(
            name='Qualitative Score',
            x=df['Ticker'],
            y=df['Qualitative Score'],
            marker_color='lightcoral'
        ))
        
        fig_scores.add_trace(go.Scatter(
            name='Final Score',
            x=df['Ticker'],
            y=df['Final Score'],
            mode='lines+markers',
            line=dict(color='gold', width=3),
            marker=dict(size=8)
        ))
        
        fig_scores.update_layout(
            title="Score Breakdown by Company",
            xaxis_title="Company",
            yaxis_title="Score (1-10)",
            barmode='group'
        )
        
        st.plotly_chart(fig_scores, use_container_width=True)
    
    with tab2:
        # Risk-return scatter
        fig_risk = px.scatter(
            df,
            x='Final Score',
            y='Market Cap ($B)',
            color='Recommendation',
            size='Fundamental Score',
            hover_name='Ticker',
            hover_data=['Company', 'Confidence', 'Position Size'],
            title="Risk-Return Analysis",
            log_y=True
        )
        
        fig_risk.update_layout(
            xaxis_title="Final Investment Score",
            yaxis_title="Market Cap ($B, log scale)"
        )
        
        st.plotly_chart(fig_risk, use_container_width=True)
    
    with tab3:
        # Recommendation distribution
        rec_counts = df['Recommendation'].value_counts()
        
        fig_rec = px.pie(
            values=rec_counts.values,
            names=rec_counts.index,
            title="Recommendation Distribution"
        )
        
        st.plotly_chart(fig_rec, use_container_width=True)

def display_detailed_analysis(company):
    """Display detailed analysis for a selected company"""
    
    st.write(f"### 📊 {company.ticker} - {company.company_name}")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Final Investment Score", f"{company.final_investment_score:.1f}/10")
    
    with col2:
        st.metric("Recommendation", company.recommendation, 
                 help=f"Confidence: {company.confidence_level}")
    
    with col3:
        st.metric("Position Size", company.position_size)
    
    with col4:
        if company.price_target:
            st.metric("Price Target", f"${company.price_target:.2f}")
        else:
            st.metric("Price Target", "N/A")
    
    # Investment thesis
    st.subheader("💡 Investment Thesis")
    st.write(company.investment_thesis)
    
    # Why good investment
    st.subheader("🤔 Why Good Investment")
    st.write(company.why_good_investment)
    
    # Strengths and risks
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💪 Key Strengths")
        for strength in company.key_strengths:
            st.write(f"• {strength}")
    
    with col2:
        st.subheader("⚠️ Key Risks")
        for risk in company.key_risks:
            st.write(f"• {risk}")
    
    # Score breakdown
    st.subheader("📊 Score Breakdown")
    
    scores_df = pd.DataFrame({
        'Component': ['Growth', 'Profitability', 'Debt Health', 'Moat', 'Sentiment', 'Trends'],
        'Score': [
            company.growth_score,
            company.profitability_score,
            company.debt_health_score,
            company.moat_score,
            company.sentiment_score,
            company.trend_score
        ]
    })
    
    fig_breakdown = px.bar(
        scores_df,
        x='Component',
        y='Score',
        title=f"{company.ticker} - Component Scores",
        color='Score',
        color_continuous_scale='RdYlGn'
    )
    
    fig_breakdown.update_layout(yaxis_range=[0, 10])
    
    st.plotly_chart(fig_breakdown, use_container_width=True)

def display_portfolio_recommendations(portfolio_recs):
    """Display portfolio recommendations"""
    
    st.header("🎯 Portfolio Recommendations")
    
    if not portfolio_recs['portfolio']:
        st.warning("⚠️ No portfolio recommendations generated.")
        return
    
    # Portfolio summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Positions", portfolio_recs['total_positions'])
    
    with col2:
        st.metric("Avg Score", f"{portfolio_recs['avg_score']:.1f}/10")
    
    with col3:
        st.metric("Risk Profile", portfolio_recs['risk_profile'])
    
    with col4:
        top_position = portfolio_recs['portfolio'][0] if portfolio_recs['portfolio'] else None
        if top_position:
            st.metric("Top Position", f"{top_position['ticker']} ({top_position['weight']:.1f}%)")
    
    # Portfolio allocation table
    st.subheader("📋 Recommended Portfolio Allocation")
    
    portfolio_df = pd.DataFrame(portfolio_recs['portfolio'])
    
    st.dataframe(
        portfolio_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Portfolio visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Weight allocation pie chart
        fig_pie = px.pie(
            portfolio_df,
            values='weight',
            names='ticker',
            title="Portfolio Weight Allocation"
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Sector allocation if available
        if 'sector_allocation' in portfolio_recs and portfolio_recs['sector_allocation']:
            sector_df = pd.DataFrame([
                {'Sector': sector, 'Allocation': allocation}
                for sector, allocation in portfolio_recs['sector_allocation'].items()
            ])
            
            fig_sector = px.bar(
                sector_df,
                x='Sector',
                y='Allocation',
                title="Sector Allocation (%)"
            )
            
            st.plotly_chart(fig_sector, use_container_width=True)
    
    # Export portfolio
    st.subheader("💾 Export Portfolio")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = portfolio_df.to_csv(index=False)
        st.download_button(
            label="📄 Download Portfolio CSV",
            data=csv,
            file_name=f"portfolio_recommendations_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # Generate portfolio report
        if st.button("📊 Generate Portfolio Report"):
            generate_portfolio_report(portfolio_recs)

def generate_portfolio_report(portfolio_recs):
    """Generate a comprehensive portfolio report"""
    
    st.subheader("📊 Portfolio Investment Report")
    
    report = f"""
# Portfolio Investment Report
**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Portfolio Summary
- **Total Positions:** {portfolio_recs['total_positions']}
- **Average Score:** {portfolio_recs['avg_score']:.1f}/10
- **Risk Profile:** {portfolio_recs['risk_profile']}

## Investment Rationale
This portfolio represents the top investment opportunities identified through our 3-agent analysis system:

1. **Fundamental Agent** screened companies based on strict quantitative criteria
2. **Rationale Agent** analyzed competitive moats, sentiment, and secular trends
3. **Ranker Agent** combined both analyses for final scoring and recommendations

## Position Details
"""
    
    for position in portfolio_recs['portfolio']:
        report += f"""
### {position['ticker']} - {position['weight']:.1f}% Allocation
- **Recommendation:** {position['recommendation']}
- **Final Score:** {position['final_score']:.1f}/10
- **Investment Thesis:** {position['investment_thesis']}
"""
    
    st.markdown(report)
    
    # Download report
    st.download_button(
        label="📄 Download Full Report",
        data=report,
        file_name=f"portfolio_report_{datetime.now().strftime('%Y%m%d')}.md",
        mime="text/markdown"
    )

def display_recent_results():
    """Display recent ranking results if available"""
    
    if 'ranked_companies' in st.session_state and st.session_state['ranked_companies']:
        st.info("📋 Previous ranking results are available in session state")
        
        if st.button("🔄 Show Previous Results"):
            ranked_companies = st.session_state['ranked_companies']
            display_ranking_results(ranked_companies, 10, "Moderate")
            
            # Generate portfolio recommendations
            agent = RankerAgent()
            portfolio_recs = agent.get_portfolio_recommendations(ranked_companies, 10)
            display_portfolio_recommendations(portfolio_recs)

if __name__ == "__main__":
    main()

