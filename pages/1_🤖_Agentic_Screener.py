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
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from database.schema import DatabaseManager
from agents.multi_agent_system import MultiAgentPortfolioSystem
from agents.ranking_agent import RankingAgent
from utils.dynamic_stock_discovery import dynamic_discovery

# Simple Stock class for the screener
class Stock:
    def __init__(self, symbol: str, company_name: str, sector: str, market_cap: float, current_price: float):
        self.symbol = symbol
        self.company_name = company_name
        self.sector = sector
        self.market_cap = market_cap
        self.current_price = current_price

# Page configuration
st.set_page_config(
    page_title="Agentic Screener - Lohusalu Capital Management",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def init_components():
    """Initialize database and agents"""
    try:
        db = DatabaseManager()
        
        # Get OpenAI API key from environment
        import os
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            st.error("OpenAI API key not found in environment variables")
            return None, None, None
        
        mas = MultiAgentPortfolioSystem(openai_api_key=openai_api_key)
        ranking_agent = RankingAgent(openai_api_key=openai_api_key)
        return db, mas, ranking_agent
    except Exception as e:
        st.error(f"Error initializing components: {e}")
        return None, None, None

def main():
    """Stock Screener page main function"""
    
    # Initialize components
    db, mas, ranking_agent = init_components()
    
    # Page header
    st.markdown("""
    <div style="background: linear-gradient(90deg, #1f77b4, #17becf); padding: 2rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 2rem;">
        <h1>🤖 Agentic Screener</h1>
        <p>Multi-agent stock screening powered by 6 specialized AI agents</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("## 🔍 Screening Configuration")
        
        # Sector selection
        st.markdown("### Select Sectors to Screen")
        available_sectors = [
            'Technology', 'Healthcare', 'Financial Services', 
            'Consumer Cyclical', 'Communication Services',
            'Industrial', 'Consumer Defensive', 'Energy',
            'Utilities', 'Real Estate', 'Materials'
        ]
        
        selected_sectors = []
        for sector in available_sectors:
            if st.checkbox(sector, key=f"sector_{sector}"):
                selected_sectors.append(sector)
        
        if not selected_sectors:
            st.warning("Please select at least one sector to screen.")
            return
        
        st.markdown("### Market Cap Classification")
        
        # Market cap class selection
        market_cap_classes = {
            'Nano-cap': {'min': 0, 'max': 50e6, 'description': '$0 - $50M'},
            'Micro-cap': {'min': 0, 'max': 300e6, 'description': '$0 - $300M'},
            'Small-cap': {'min': 0, 'max': 1e9, 'description': '$0 - $1B'},
            'Sub-$2B': {'min': 0, 'max': 2e9, 'description': '$0 - $2B'},
            'Sub-$5B': {'min': 0, 'max': 5e9, 'description': '$0 - $5B'},
            'All Caps': {'min': 0, 'max': float('inf'), 'description': 'All Market Caps'}
        }
        
        selected_cap_class = st.selectbox(
            "Market Cap Class",
            options=list(market_cap_classes.keys()),
            index=2,  # Default to Small-cap (under $1B)
            format_func=lambda x: f"{x}: {market_cap_classes[x]['description']}",
            help="Select market capitalization range for stock discovery - all ranges start from $0"
        )
        
        # Custom range option
        if st.checkbox("Use Custom Market Cap Range"):
            col_min, col_max = st.columns(2)
            with col_min:
                custom_min = st.number_input(
                    "Min Market Cap ($M)",
                    min_value=0,
                    max_value=10000,
                    value=0,  # Start from $0
                    step=10
                ) * 1e6
            with col_max:
                custom_max = st.number_input(
                    "Max Market Cap ($M)",
                    min_value=1,
                    max_value=10000,
                    value=1000,  # Default to $1B
                    step=50
                ) * 1e6
            
            market_cap_range = {'min': custom_min, 'max': custom_max}
        else:
            market_cap_range = market_cap_classes[selected_cap_class]
        
        st.markdown("### Screening Parameters")
        
        # Stocks per sector
        stocks_per_sector = st.slider(
            "Stocks per Sector",
            min_value=5,
            max_value=50,
            value=15,
            help="Number of top stocks to discover per sector"
        )
        
        # Additional filters
        st.markdown("### Advanced Filters")
        
        # Minimum daily volume
        min_volume = st.selectbox(
            "Minimum Daily Volume",
            options=[0, 100000, 500000, 1000000, 5000000],
            index=1,
            format_func=lambda x: f"{x:,}" if x > 0 else "No minimum",
            help="Minimum average daily trading volume"
        )
        
        # Revenue growth filter
        min_revenue_growth = st.selectbox(
            "Minimum Revenue Growth",
            options=[0, 0.05, 0.10, 0.15, 0.20, 0.30],
            index=0,
            format_func=lambda x: f"{x:.0%}" if x > 0 else "No minimum",
            help="Minimum annual revenue growth rate"
        )
        
        # Profitability filter
        require_profitability = st.checkbox(
            "Require Profitability",
            value=False,
            help="Only include profitable companies"
        )
    
    with col2:
        st.markdown("## ⚙️ Agent Pipeline")
        
        st.markdown("""
        **Analysis Pipeline:**
        1. 📊 **Dynamic Stock Discovery** - Intelligent sector screening
        2. 🧠 **Fundamental Agent** - Financial analysis & DCF valuation
        3. 📰 **Sentiment Agent** - News & market sentiment processing  
        4. 💰 **Valuation Agent** - Technical analysis & relative valuation
        5. 🎯 **Rationale Agent** - Business quality assessment
        6. 🚀 **Secular Trend Agent** - Technology trend positioning
        7. 🏆 **Ranking Agent** - Final consensus & recommendation
        """)
    
    # Get screening criteria
    criteria = {
        'market_cap_range': market_cap_range,
        'min_volume': min_volume,
        'min_revenue_growth': min_revenue_growth,
        'require_profitability': require_profitability,
        'risk_tolerance': risk_tolerance.lower(),
        'stocks_per_sector': stocks_per_sector
    }
    
    # Display current screening criteria
    with col2:
        st.markdown("## ⚙️ Current Screening Criteria")
        
        st.markdown(f"""
        **Market Cap Range:** {market_cap_range['min']/1e6:.0f}M - {market_cap_range['max']/1e6:.0f}M
        **Sectors:** {', '.join(selected_sectors)}
        **Stocks per Sector:** {stocks_per_sector}
        **Min Daily Volume:** {min_volume:,} shares
        **Min Revenue Growth:** {min_revenue_growth:.0%}
        **Profitability Required:** {'Yes' if require_profitability else 'No'}
        **Risk Tolerance:** {risk_tolerance}
        """)
        
        st.markdown("## 🤖 Agent Pipeline")
        
        st.markdown("""
        **Analysis Pipeline:**
        1. 🔍 **Dynamic Stock Discovery** - Market cap & sector filtering
        2. 📊 **Fundamental Agent** - Financial analysis & DCF valuation
        3. 📰 **Sentiment Agent** - News & market sentiment processing  
        4. 💰 **Valuation Agent** - Technical analysis & relative valuation
        5. 🎯 **Rationale Agent** - Business quality assessment
        6. 🚀 **Secular Trend Agent** - Technology trend positioning
        7. 🏆 **Ranking Agent** - Final consensus & recommendation
        """)
        
        # Market cap class info
        st.markdown("### 📈 Market Cap Classifications")
        for cap_class, info in market_cap_classes.items():
            if cap_class == selected_cap_class:
                st.markdown(f"**🎯 {cap_class}:** {info['description']} ← *Selected*")
            else:
                st.markdown(f"• {cap_class}: {info['description']}")
    
    # Start screening button
    if st.button("🚀 Start Screening", type="primary", use_container_width=True):
        st.markdown("---")
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Results containers
        results_container = st.container()
        
        total_sectors = len(selected_sectors)
        all_results = []
        
        for i, sector in enumerate(selected_sectors):
            # Update progress
            progress = (i + 1) / total_sectors
            progress_bar.progress(progress)
            status_text.text(f"🔍 Discovering {selected_cap_class} stocks in {sector} sector...")
            
            # Discover stocks dynamically
            discovered_stocks = dynamic_discovery.discover_sector_stocks(
                sector=sector,
                criteria=criteria,
                limit=stocks_per_sector
            )
            
            if discovered_stocks:
                # Filter by market cap range and other criteria
                filtered_stocks = []
                for stock in discovered_stocks:
                    market_cap = stock.get('market_cap', 0)
                    
                    # Market cap filter
                    if not (market_cap_range['min'] <= market_cap <= market_cap_range['max']):
                        continue
                    
                    # Volume filter (if available)
                    avg_volume = stock.get('avg_volume', min_volume + 1)  # Default pass
                    if avg_volume < min_volume:
                        continue
                    
                    # Revenue growth filter
                    revenue_growth = stock.get('revenue_growth', 0)
                    if revenue_growth < min_revenue_growth:
                        continue
                    
                    # Profitability filter
                    if require_profitability:
                        profit_margin = stock.get('profit_margin', 0)
                        if profit_margin <= 0:
                            continue
                    
                    filtered_stocks.append(stock)
                
                # Display sector results
                with results_container:
                    st.markdown(f"## 📈 {sector} Sector Results ({selected_cap_class})")
                    
                    if filtered_stocks:
                        st.success(f"Found {len(filtered_stocks)} {selected_cap_class} stocks matching criteria")
                        
                        # Create results DataFrame
                        sector_df = pd.DataFrame(filtered_stocks)
                        
                        # Display top discoveries with reasoning
                        for idx, stock in enumerate(filtered_stocks[:5]):  # Show top 5
                            with st.expander(f"#{idx+1} {stock['symbol']} - {stock['company_name']} (Score: {stock['discovery_score']:.0f})"):
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Market Cap", f"${stock['market_cap']/1e6:.0f}M")
                                    st.metric("Current Price", f"${stock['current_price']:.2f}")
                                
                                with col2:
                                    st.metric("P/E Ratio", f"{stock.get('pe_ratio', 0):.1f}")
                                    st.metric("Revenue Growth", f"{stock.get('revenue_growth', 0):.1%}")
                                
                                with col3:
                                    st.metric("Profit Margin", f"{stock.get('profit_margin', 0):.1%}")
                                    st.metric("ROE", f"{stock.get('roe', 0):.1%}")
                                
                                # Discovery reasoning
                                st.markdown("**Discovery Reasons:**")
                                for reason in stock['discovery_reasons']:
                                    st.markdown(f"• {reason}")
                                
                                # Market cap classification
                                for cap_class, info in market_cap_classes.items():
                                    if info['min'] <= stock['market_cap'] <= info['max']:
                                        st.markdown(f"**Market Cap Class:** {cap_class}")
                                        break
                                
                                # Run agent analysis
                                if st.button(f"🤖 Run Agent Analysis", key=f"analyze_{stock['symbol']}"):
                                    with st.spinner(f"Analyzing {stock['symbol']} with all 6 agents..."):
                                        # Create stock object
                                        stock_obj = Stock(
                                            symbol=stock['symbol'],
                                            company_name=stock['company_name'],
                                            sector=stock['sector'],
                                            market_cap=stock['market_cap'],
                                            current_price=stock['current_price']
                                        )
                                        
                                        # Run agent analysis
                                        if mas and ranking_agent:
                                            try:
                                                # Run multi-agent analysis
                                                agent_results = mas.analyze_stock(stock_obj)
                                                
                                                # Get final ranking
                                                final_result = ranking_agent.analyze_stock(stock_obj, agent_results)
                                                
                                                # Display results
                                                st.success(f"✅ Analysis Complete!")
                                                st.markdown(f"**Final Recommendation:** {final_result.recommendation}")
                                                st.markdown(f"**Confidence:** {final_result.confidence:.2f}")
                                                st.markdown(f"**Target Price:** ${final_result.target_price:.2f}")
                                                
                                                # Show agent breakdown
                                                agent_breakdown = st.expander("Agent Breakdown")
                                                with agent_breakdown:
                                                    for agent_name, result in agent_results.items():
                                                        st.markdown(f"**{agent_name}:** {result.recommendation} (Confidence: {result.confidence:.2f})")
                                                        st.markdown(f"*Reasoning:* {result.reasoning[:200]}...")
                                                
                                            except Exception as e:
                                                st.error(f"Error in agent analysis: {str(e)}")
                        
                        # Sector summary chart
                        if len(filtered_stocks) > 1:
                            fig = px.scatter(
                                sector_df,
                                x='discovery_score',
                                y='market_cap',
                                size='current_price',
                                color='discovery_score',
                                hover_name='symbol',
                                title=f"{sector} Sector Discovery Map ({selected_cap_class})",
                                labels={
                                    'discovery_score': 'Discovery Score',
                                    'market_cap': 'Market Cap ($)',
                                    'current_price': 'Stock Price ($)'
                                }
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        all_results.extend(filtered_stocks)
                    
                    else:
                        st.warning(f"No {selected_cap_class} stocks found in {sector} matching the criteria")
                    
                    st.markdown("---")
            
            # Small delay to show progress
            time.sleep(0.5)
        
        # Final summary
        progress_bar.progress(1.0)
        status_text.text("✅ Screening Complete!")
        
        if all_results:
            st.markdown(f"## 🎯 Overall {selected_cap_class} Screening Summary")
            
            # Top discoveries across all sectors
            top_discoveries = sorted(all_results, key=lambda x: x['discovery_score'], reverse=True)[:10]
            
            summary_df = pd.DataFrame(top_discoveries)
            st.dataframe(
                summary_df[['symbol', 'company_name', 'sector', 'discovery_score', 'market_cap', 'current_price']],
                use_container_width=True
            )
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Discovered", len(all_results))
            with col2:
                st.metric("Avg Discovery Score", f"{np.mean([s['discovery_score'] for s in all_results]):.1f}")
            with col3:
                st.metric("Avg Market Cap", f"${np.mean([s['market_cap'] for s in all_results])/1e6:.0f}M")
            with col4:
                st.metric("Sectors Analyzed", len(selected_sectors))
        
        else:
            st.warning(f"No {selected_cap_class} stocks discovered matching the criteria. Try adjusting the parameters.")

if __name__ == "__main__":
    main()

