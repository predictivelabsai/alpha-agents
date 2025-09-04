import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
import time
from datetime import datetime
import logging
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from database.schema import init_database, Stock, Analysis
from agents.multi_agent_system import AlphaAgentsSystem
from agents.ranking_agent import RankingAgent
from utils.dynamic_stock_discovery import dynamic_discovery
from database import DatabaseManager

# Page configuration
st.set_page_config(
    page_title="Stock Screener - Lohusalu Capital Management",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .screener-header {
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sector-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .stock-result {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        border-left: 4px solid #28a745;
    }
    .agent-consensus {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .metric-highlight {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def init_components():
    """Initialize database and multi-agent system"""
    db = DatabaseManager()
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        st.error("OpenAI API key not found. Please check your environment configuration.")
        return db, None, None
    
    mas = create_multi_agent_portfolio_system(
        openai_api_key=api_key,
        risk_tolerance="moderate",
        max_debate_rounds=2
    )
    
    ranking_agent = RankingAgent(
        openai_api_key=api_key,
        risk_tolerance=RiskTolerance.MODERATE
    )
    
    return db, mas, ranking_agent

def save_screening_results(results: list, sectors: list):
    """Save screening results to logs"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"logs/screening_results_{timestamp}.json"
        
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'sectors_screened': sectors,
            'total_stocks': len(results),
            'results': results
        }
        
        os.makedirs('logs', exist_ok=True)
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2, default=str)
        
        return log_file
    except Exception as e:
        logging.error(f"Error saving screening results: {e}")
        return None

def main():
    """Main Stock Screener page"""
    
    # Initialize components
    db, mas, ranking_agent = init_components()
    
    # Header
    st.markdown("""
    <div class="screener-header">
        <h1>📊 Stock Screener</h1>
        <p>Multi-sector stock screening powered by 6 specialized AI agents</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.title("🔍 Screening Configuration")
    
    # Sector selection
    available_sectors = [
        'Technology',
        'Healthcare', 
        'Financial Services',
        'Consumer Cyclical',
        'Communication Services',
        'Industrial',
        'Consumer Defensive',
        'Energy',
        'Utilities',
        'Real Estate',
        'Materials'
    ]
    
    selected_sectors = st.sidebar.multiselect(
        "Select Sectors to Screen",
        available_sectors,
        default=['Technology', 'Healthcare'],
        help="Choose one or more sectors for comprehensive screening"
    )
    
    # Screening parameters
    st.sidebar.subheader("📋 Screening Parameters")
    
    stocks_per_sector = st.sidebar.slider(
        "Stocks per Sector",
        min_value=5,
        max_value=20,
        value=10,
        help="Number of stocks to analyze per sector"
    )
    
    min_market_cap = st.sidebar.selectbox(
        "Minimum Market Cap",
        ["Any", "$1B+", "$10B+", "$50B+", "$100B+"],
        index=1,
        help="Filter by minimum market capitalization"
    )
    
    risk_tolerance = st.sidebar.selectbox(
        "Risk Tolerance",
        ["Conservative", "Moderate", "Aggressive"],
        index=1,
        help="Adjust analysis for risk preference"
    )
    
    # Main content area
    if not selected_sectors:
        st.warning("Please select at least one sector to begin screening.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"🎯 Screening {len(selected_sectors)} Sector(s)")
        
        # Display selected sectors
        for sector in selected_sectors:
            st.markdown(f"""
            <div class="sector-card">
                <h4>{sector}</h4>
                <p>Analyzing top {stocks_per_sector} stocks in this sector</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("⚙️ Agent Pipeline")
        st.markdown("""
        **Analysis Pipeline:**
        1. 📊 Fundamental Agent
        2. 📰 Sentiment Agent  
        3. 💰 Valuation Agent
        4. 🧠 Rationale Agent
        5. 🚀 Secular Trend Agent
        6. 🏆 Ranking Agent
        """)
    
    # Start screening button
    if st.button("🚀 Start Screening", type="primary", use_container_width=True):
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        all_results = []
        total_stocks = len(selected_sectors) * stocks_per_sector
        processed_stocks = 0
        
        # Process each sector
        for sector_idx, sector in enumerate(selected_sectors):
            status_text.text(f"Processing {sector} sector...")
            
            # Get stocks for this sector
            sector_stocks = yfinance_provider.get_sector_stocks(sector, stocks_per_sector)
            
            st.subheader(f"📈 {sector} Sector Results")
            
            sector_results = []
            
            # Process each stock in the sector
            for stock_idx, symbol in enumerate(sector_stocks):
                try:
                    status_text.text(f"Analyzing {symbol} ({sector})...")
                    
                    # Get stock data
                    stock_data = yfinance_provider.get_stock_data(symbol)
                    
                    # Create stock object
                    stock = Stock(
                        symbol=symbol,
                        company_name=stock_data['company_name'],
                        sector=stock_data['sector'],
                        market_cap=stock_data.get('market_cap', 0),
                        current_price=stock_data['current_price']
                    )
                    
                    # Run through agent pipeline
                    if mas and ranking_agent:
                        # Get analysis from all 5 agents
                        agent_analyses = {}
                        
                        for agent_name, agent in mas.agents.items():
                            try:
                                analysis = agent.analyze_stock(stock)
                                agent_analyses[agent_name] = analysis
                            except Exception as e:
                                logging.error(f"Error in {agent_name} analysis: {e}")
                                agent_analyses[agent_name] = {
                                    'recommendation': 'HOLD',
                                    'confidence_score': 0.5,
                                    'reasoning': f"Analysis failed: {str(e)}"
                                }
                        
                        # Get final ranking
                        final_analysis = ranking_agent.analyze_stock(stock, agent_analyses)
                        
                        # Add stock data to final analysis
                        final_analysis.update({
                            'sector': sector,
                            'market_cap': stock_data['market_cap'],
                            'current_price': stock_data['current_price'],
                            'pe_ratio': stock_data['calculated_metrics'].get('pe_ratio', 0),
                            'agent_analyses': agent_analyses
                        })
                        
                        sector_results.append(final_analysis)
                        all_results.append(final_analysis)
                    
                    processed_stocks += 1
                    progress_bar.progress(processed_stocks / total_stocks)
                    
                except Exception as e:
                    st.error(f"Error analyzing {symbol}: {str(e)}")
                    continue
            
            # Display sector results
            if sector_results:
                # Sort by composite score
                sector_results.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
                
                # Create results table
                results_df = pd.DataFrame([
                    {
                        'Symbol': r['stock_symbol'],
                        'Company': r['company_name'],
                        'Recommendation': r['recommendation'],
                        'Composite Score': r['composite_score'],
                        'Confidence': f"{r['confidence_score']:.2f}",
                        'Market Cap': f"${r['market_cap']/1e9:.1f}B" if r['market_cap'] > 0 else "N/A",
                        'P/E Ratio': f"{r['pe_ratio']:.1f}" if r['pe_ratio'] > 0 else "N/A"
                    }
                    for r in sector_results
                ])
                
                st.dataframe(results_df, use_container_width=True)
                
                # Show top 3 picks for this sector
                st.subheader(f"🏆 Top 3 Picks - {sector}")
                
                for i, result in enumerate(sector_results[:3]):
                    with st.expander(f"#{i+1}: {result['stock_symbol']} - {result['company_name']}"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Recommendation", result['recommendation'])
                            st.metric("Composite Score", f"{result['composite_score']}/100")
                        
                        with col2:
                            st.metric("Confidence", f"{result['confidence_score']:.2f}")
                            st.metric("Market Cap", f"${result['market_cap']/1e9:.1f}B" if result['market_cap'] > 0 else "N/A")
                        
                        with col3:
                            st.metric("P/E Ratio", f"{result['pe_ratio']:.1f}" if result['pe_ratio'] > 0 else "N/A")
                        
                        st.markdown("**Investment Thesis:**")
                        st.write(result.get('investment_thesis', 'Multi-agent analysis completed'))
                        
                        # Agent breakdown
                        st.markdown("**Agent Consensus:**")
                        agent_recs = []
                        for agent_name, analysis in result.get('agent_analyses', {}).items():
                            rec = analysis.get('recommendation', 'HOLD')
                            conf = analysis.get('confidence_score', 0.5)
                            agent_recs.append(f"**{agent_name.title()}**: {rec} ({conf:.2f})")
                        
                        st.markdown(" | ".join(agent_recs))
        
        # Overall screening summary
        if all_results:
            st.markdown("---")
            st.subheader("📊 Overall Screening Summary")
            
            # Rank all stocks across sectors
            all_results.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Stocks Analyzed", len(all_results))
            
            with col2:
                buy_count = sum(1 for r in all_results if r['recommendation'] in ['BUY', 'STRONG_BUY'])
                st.metric("Buy Recommendations", buy_count)
            
            with col3:
                avg_score = sum(r['composite_score'] for r in all_results) / len(all_results)
                st.metric("Average Score", f"{avg_score:.1f}/100")
            
            with col4:
                avg_confidence = sum(r['confidence_score'] for r in all_results) / len(all_results)
                st.metric("Average Confidence", f"{avg_confidence:.2f}")
            
            # Top 10 overall picks
            st.subheader("🏆 Top 10 Overall Picks")
            
            top_picks_df = pd.DataFrame([
                {
                    'Rank': i+1,
                    'Symbol': r['stock_symbol'],
                    'Company': r['company_name'],
                    'Sector': r['sector'],
                    'Recommendation': r['recommendation'],
                    'Score': r['composite_score'],
                    'Confidence': f"{r['confidence_score']:.2f}"
                }
                for i, r in enumerate(all_results[:10])
            ])
            
            st.dataframe(top_picks_df, use_container_width=True)
            
            # Visualization
            st.subheader("📈 Screening Results Visualization")
            
            # Score distribution by sector
            fig = px.box(
                pd.DataFrame(all_results),
                x='sector',
                y='composite_score',
                title="Composite Score Distribution by Sector",
                color='sector'
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendation distribution
            rec_counts = pd.DataFrame(all_results)['recommendation'].value_counts()
            fig2 = px.pie(
                values=rec_counts.values,
                names=rec_counts.index,
                title="Recommendation Distribution",
                color_discrete_map={
                    'STRONG_BUY': '#00cc44',
                    'BUY': '#28a745',
                    'HOLD': '#ffc107',
                    'SELL': '#fd7e14',
                    'STRONG_SELL': '#dc3545'
                }
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            # Save results
            log_file = save_screening_results(all_results, selected_sectors)
            if log_file:
                st.success(f"✅ Screening results saved to {log_file}")
        
        status_text.text("✅ Screening completed!")
        progress_bar.progress(1.0)

if __name__ == "__main__":
    main()

