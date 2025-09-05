"""
Lohusalu Capital Management - 3-Agent Equity Portfolio Construction System
Home page showcasing the new streamlined 3-agent architecture
"""

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="Lohusalu Capital Management",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Header
    st.title("🏛️ Lohusalu Capital Management")
    st.markdown("**Multi-Agent Equity Portfolio Construction System**")
    
    # Hero section
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); padding: 2rem; border-radius: 10px; margin: 1rem 0;">
        <h2 style="color: white; margin: 0;">Intelligent Investment Analysis Through AI Agents</h2>
        <p style="color: #e0e0e0; margin: 0.5rem 0 0 0;">Combining quantitative rigor with qualitative insights for superior investment decisions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # System overview
    st.header("🤖 3-Agent System Architecture")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 Fundamental Agent
        **Quantitative Screening**
        
        - Growth metrics analysis
        - Profitability assessment  
        - Debt health evaluation
        - Intrinsic value calculation
        - Strict filtering criteria
        
        **Output:** Qualified companies list
        """)
        
        if st.button("🚀 Start Fundamental Analysis", key="fund_btn"):
            st.switch_page("pages/1_📊_Fundamental_Agent.py")
    
    with col2:
        st.markdown("""
        ### 🔍 Rationale Agent
        **Qualitative Analysis**
        
        - Competitive moat analysis
        - Market sentiment evaluation
        - Secular trends assessment
        - Heavy web research (Tavily)
        - Citation-backed insights
        
        **Output:** 1-10 qualitative scores
        """)
        
        if st.button("🔍 Perform Qualitative Analysis", key="rat_btn"):
            st.switch_page("pages/2_🔍_Rationale_Agent.py")
    
    with col3:
        st.markdown("""
        ### 🏆 Ranker Agent
        **Final Scoring & Recommendations**
        
        - Combines quantitative + qualitative
        - 1-10 investment scoring
        - "Why good investment" reasoning
        - Position sizing recommendations
        - Portfolio optimization
        
        **Output:** Final investment rankings
        """)
        
        if st.button("🏆 Generate Final Rankings", key="rank_btn"):
            st.switch_page("pages/3_🏆_Ranker_Agent.py")
    
    # Process flow diagram
    st.header("🔄 Analysis Workflow")
    
    # Create flow diagram
    fig = go.Figure()
    
    # Add nodes
    nodes = [
        {"x": 1, "y": 3, "text": "📊<br>Fundamental<br>Agent", "color": "#3498db"},
        {"x": 3, "y": 3, "text": "🔍<br>Rationale<br>Agent", "color": "#e74c3c"},
        {"x": 5, "y": 3, "text": "🏆<br>Ranker<br>Agent", "color": "#f39c12"},
        {"x": 7, "y": 3, "text": "📈<br>Portfolio<br>Recommendations", "color": "#27ae60"}
    ]
    
    # Add arrows
    arrows = [
        {"x0": 1.5, "y0": 3, "x1": 2.5, "y1": 3},
        {"x0": 3.5, "y0": 3, "x1": 4.5, "y1": 3},
        {"x0": 5.5, "y0": 3, "x1": 6.5, "y1": 3}
    ]
    
    # Plot nodes
    for node in nodes:
        fig.add_trace(go.Scatter(
            x=[node["x"]],
            y=[node["y"]],
            mode='markers+text',
            marker=dict(size=80, color=node["color"]),
            text=node["text"],
            textposition="middle center",
            textfont=dict(color="white", size=12),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Add arrows
    for arrow in arrows:
        fig.add_annotation(
            x=arrow["x1"],
            y=arrow["y1"],
            ax=arrow["x0"],
            ay=arrow["y0"],
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            arrowhead=2,
            arrowsize=1,
            arrowwidth=3,
            arrowcolor="#34495e"
        )
    
    fig.update_layout(
        title="3-Agent Analysis Pipeline",
        xaxis=dict(range=[0, 8], showgrid=False, showticklabels=False),
        yaxis=dict(range=[2, 4], showgrid=False, showticklabels=False),
        height=300,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Key features
    st.header("✨ Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 **Quantitative Rigor**
        - Strict fundamental screening criteria
        - Growth, profitability, and debt analysis
        - Intrinsic value assessment
        - Market cap classification (microcap focus)
        
        ### 🔍 **Qualitative Insights**
        - Competitive moat evaluation
        - Market sentiment analysis
        - Secular trend alignment
        - Web research with citations
        """)
    
    with col2:
        st.markdown("""
        ### 🏆 **Investment Intelligence**
        - 1-10 scoring methodology
        - Detailed investment reasoning
        - Risk-adjusted recommendations
        - Portfolio optimization
        
        ### 🌍 **Global Coverage**
        - US and China markets
        - Multiple sector coverage
        - Microcap to large-cap analysis
        - Real-time data integration
        """)
    
    # Quick start guide
    st.header("🚀 Quick Start Guide")
    
    with st.expander("📖 How to Use the System", expanded=False):
        st.markdown("""
        ### Step 1: Fundamental Screening 📊
        1. Go to **Fundamental Agent** page
        2. Select market (US/China) and sector
        3. Set market cap preferences (microcap focus available)
        4. Adjust screening criteria (growth, profitability, debt)
        5. Run screening to get qualified companies
        
        ### Step 2: Qualitative Analysis 🔍
        1. Go to **Rationale Agent** page
        2. Input Tavily API key for web search (optional)
        3. Select companies from fundamental screening
        4. Choose analysis depth (Quick/Standard/Deep)
        5. Run qualitative analysis for moat, sentiment, trends
        
        ### Step 3: Final Ranking 🏆
        1. Go to **Ranker Agent** page
        2. Input OpenAI API key for detailed analysis (optional)
        3. Adjust weighting (fundamental vs qualitative)
        4. Set portfolio preferences
        5. Generate final rankings and portfolio recommendations
        
        ### Step 4: Portfolio Construction 📈
        - Review final investment scores (1-10 scale)
        - Read detailed "why good investment" reasoning
        - Analyze recommended position sizes
        - Export portfolio allocation and reports
        """)
    
    # System status
    st.header("📊 System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Check session state for pipeline progress
        fund_status = "✅ Complete" if st.session_state.get('qualified_companies') else "⏳ Pending"
        st.metric("Fundamental Analysis", fund_status)
    
    with col2:
        rat_status = "✅ Complete" if st.session_state.get('rationale_analyses') else "⏳ Pending"
        st.metric("Qualitative Analysis", rat_status)
    
    with col3:
        rank_status = "✅ Complete" if st.session_state.get('ranked_companies') else "⏳ Pending"
        st.metric("Final Rankings", rank_status)
    
    with col4:
        # Calculate completion percentage
        completed = sum([
            bool(st.session_state.get('qualified_companies')),
            bool(st.session_state.get('rationale_analyses')),
            bool(st.session_state.get('ranked_companies'))
        ])
        completion = f"{(completed/3)*100:.0f}%"
        st.metric("Pipeline Progress", completion)
    
    # Recent activity
    if any([st.session_state.get('qualified_companies'), 
            st.session_state.get('rationale_analyses'), 
            st.session_state.get('ranked_companies')]):
        
        st.header("📋 Recent Activity")
        
        if st.session_state.get('qualified_companies'):
            companies = st.session_state['qualified_companies']
            st.success(f"✅ {len(companies)} companies qualified from fundamental screening")
        
        if st.session_state.get('rationale_analyses'):
            analyses = st.session_state['rationale_analyses']
            st.success(f"✅ {len(analyses)} companies analyzed for qualitative factors")
        
        if st.session_state.get('ranked_companies'):
            rankings = st.session_state['ranked_companies']
            st.success(f"✅ {len(rankings)} companies ranked with final investment scores")
            
            # Show top 3 recommendations
            st.subheader("🏆 Top Investment Recommendations")
            
            for i, company in enumerate(rankings[:3], 1):
                with st.container():
                    col1, col2, col3, col4 = st.columns([1, 3, 2, 2])
                    
                    with col1:
                        st.write(f"**#{i}**")
                    
                    with col2:
                        st.write(f"**{company.ticker}** - {company.company_name}")
                    
                    with col3:
                        st.write(f"Score: **{company.final_investment_score:.1f}/10**")
                    
                    with col4:
                        st.write(f"**{company.recommendation}**")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p><strong>Lohusalu Capital Management</strong> | Multi-Agent Investment Analysis System</p>
        <p>Powered by AI • Built for Performance • Designed for Results</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

