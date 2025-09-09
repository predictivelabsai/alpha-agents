"""
Advanced Visualizations for Alpha Agents System
Plotly-based charts, heatmaps, and interactive visualizations
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Dict, Any, Optional
import streamlit as st

class AlphaAgentsVisualizer:
    """Advanced visualization components for Alpha Agents system"""
    
    def __init__(self):
        self.color_palette = {
            'buy': '#28a745',      # Green
            'hold': '#ffc107',     # Yellow
            'sell': '#dc3545',     # Red
            'avoid': '#6c757d',    # Gray
            'low_risk': '#28a745',
            'moderate_risk': '#ffc107',
            'high_risk': '#dc3545',
            'agents': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        }
    
    def create_agent_consensus_heatmap(self, analysis_data: List[Dict]) -> go.Figure:
        """
        Create a heatmap showing agent consensus across stocks
        """
        if not analysis_data:
            return self._create_empty_figure("No data available for consensus heatmap")
        
        # Convert to DataFrame
        df = pd.DataFrame(analysis_data)
        
        # Create pivot table for heatmap
        pivot_data = df.pivot_table(
            index='stock_symbol', 
            columns='agent', 
            values='confidence_score',
            aggfunc='mean'
        )
        
        # Create recommendation matrix
        rec_pivot = df.pivot_table(
            index='stock_symbol',
            columns='agent',
            values='recommendation',
            aggfunc='first'
        )
        
        # Convert recommendations to numeric values for color mapping
        rec_numeric = rec_pivot.replace({
            'buy': 4, 'hold': 3, 'sell': 2, 'avoid': 1
        })
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=rec_numeric.values,
            x=rec_numeric.columns,
            y=rec_numeric.index,
            colorscale=[
                [0, self.color_palette['avoid']],
                [0.25, self.color_palette['sell']],
                [0.5, self.color_palette['hold']],
                [1.0, self.color_palette['buy']]
            ],
            text=rec_pivot.values,
            texttemplate="%{text}",
            textfont={"size": 12},
            hovertemplate="<b>%{y}</b><br>Agent: %{x}<br>Recommendation: %{text}<br>Confidence: %{customdata:.2f}<extra></extra>",
            customdata=pivot_data.values
        ))
        
        fig.update_layout(
            title={
                'text': "🤖 Agent Consensus Heatmap",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title="Agents",
            yaxis_title="Stocks",
            width=800,
            height=500,
            font=dict(size=12)
        )
        
        return fig
    
    def create_risk_assessment_heatmap(self, analysis_data: List[Dict]) -> go.Figure:
        """
        Create a heatmap showing risk assessment across agents and stocks
        """
        if not analysis_data:
            return self._create_empty_figure("No data available for risk heatmap")
        
        df = pd.DataFrame(analysis_data)
        
        # Create risk pivot table
        risk_pivot = df.pivot_table(
            index='stock_symbol',
            columns='agent',
            values='risk_assessment',
            aggfunc='first'
        )
        
        # Convert risk to numeric
        risk_numeric = risk_pivot.replace({
            'LOW': 1, 'MODERATE': 2, 'HIGH': 3
        })
        
        # Create confidence pivot for hover data
        conf_pivot = df.pivot_table(
            index='stock_symbol',
            columns='agent',
            values='confidence_score',
            aggfunc='mean'
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=risk_numeric.values,
            x=risk_numeric.columns,
            y=risk_numeric.index,
            colorscale=[
                [0, self.color_palette['low_risk']],
                [0.5, self.color_palette['moderate_risk']],
                [1.0, self.color_palette['high_risk']]
            ],
            text=risk_pivot.values,
            texttemplate="%{text}",
            textfont={"size": 12},
            hovertemplate="<b>%{y}</b><br>Agent: %{x}<br>Risk: %{text}<br>Confidence: %{customdata:.2f}<extra></extra>",
            customdata=conf_pivot.values
        ))
        
        fig.update_layout(
            title={
                'text': "⚠️ Risk Assessment Heatmap",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title="Agents",
            yaxis_title="Stocks",
            width=800,
            height=500,
            font=dict(size=12)
        )
        
        return fig
    
    def create_confidence_distribution_chart(self, analysis_data: List[Dict]) -> go.Figure:
        """
        Create a box plot showing confidence score distribution by agent
        """
        if not analysis_data:
            return self._create_empty_figure("No data available for confidence distribution")
        
        df = pd.DataFrame(analysis_data)
        
        fig = go.Figure()
        
        agents = df['agent'].unique()
        for i, agent in enumerate(agents):
            agent_data = df[df['agent'] == agent]
            
            fig.add_trace(go.Box(
                y=agent_data['confidence_score'],
                name=agent.title(),
                marker_color=self.color_palette['agents'][i % len(self.color_palette['agents'])],
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))
        
        fig.update_layout(
            title={
                'text': "📊 Agent Confidence Score Distribution",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            yaxis_title="Confidence Score",
            xaxis_title="Agents",
            width=800,
            height=500,
            showlegend=False
        )
        
        return fig
    
    def create_recommendation_distribution_pie(self, analysis_data: List[Dict]) -> go.Figure:
        """
        Create a pie chart showing overall recommendation distribution
        """
        if not analysis_data:
            return self._create_empty_figure("No data available for recommendation distribution")
        
        df = pd.DataFrame(analysis_data)
        
        # Count recommendations
        rec_counts = df['recommendation'].value_counts()
        
        colors = [self.color_palette.get(rec, '#cccccc') for rec in rec_counts.index]
        
        fig = go.Figure(data=[go.Pie(
            labels=rec_counts.index,
            values=rec_counts.values,
            marker_colors=colors,
            textinfo='label+percent+value',
            textfont_size=12,
            hole=0.3
        )])
        
        fig.update_layout(
            title={
                'text': "🎯 Overall Recommendation Distribution",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            width=600,
            height=500,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.01
            )
        )
        
        return fig
    
    def create_agent_performance_radar(self, analysis_data: List[Dict]) -> go.Figure:
        """
        Create a radar chart showing agent performance metrics
        """
        if not analysis_data:
            return self._create_empty_figure("No data available for performance radar")
        
        df = pd.DataFrame(analysis_data)
        
        # Calculate metrics by agent
        agent_metrics = df.groupby('agent').agg({
            'confidence_score': 'mean',
            'recommendation': lambda x: (x == 'buy').sum() / len(x),  # Buy ratio
            'risk_assessment': lambda x: (x == 'LOW').sum() / len(x),  # Low risk ratio
            'target_price': lambda x: x.notna().sum() / len(x),  # Target price coverage
            'key_factors_count': 'mean',
            'concerns_count': 'mean'
        }).round(3)
        
        # Normalize metrics to 0-1 scale
        agent_metrics['buy_ratio'] = agent_metrics['recommendation']
        agent_metrics['low_risk_ratio'] = agent_metrics['risk_assessment']
        agent_metrics['target_coverage'] = agent_metrics['target_price']
        agent_metrics['avg_factors'] = agent_metrics['key_factors_count'] / agent_metrics['key_factors_count'].max()
        agent_metrics['concern_ratio'] = 1 - (agent_metrics['concerns_count'] / agent_metrics['concerns_count'].max())
        
        categories = ['Confidence', 'Buy Ratio', 'Low Risk', 'Target Coverage', 'Key Factors', 'Low Concerns']
        
        fig = go.Figure()
        
        for i, (agent, metrics) in enumerate(agent_metrics.iterrows()):
            values = [
                metrics['confidence_score'],
                metrics['buy_ratio'],
                metrics['low_risk_ratio'],
                metrics['target_coverage'],
                metrics['avg_factors'],
                metrics['concern_ratio']
            ]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name=agent.title(),
                line_color=self.color_palette['agents'][i % len(self.color_palette['agents'])]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title={
                'text': "🎯 Agent Performance Radar",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            width=700,
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_stock_analysis_comparison(self, analysis_data: List[Dict], stock_symbol: str) -> go.Figure:
        """
        Create a comparison chart for a specific stock across all agents
        """
        if not analysis_data:
            return self._create_empty_figure("No data available for stock comparison")
        
        df = pd.DataFrame(analysis_data)
        stock_data = df[df['stock_symbol'] == stock_symbol]
        
        if stock_data.empty:
            return self._create_empty_figure(f"No data available for {stock_symbol}")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Confidence Scores', 'Target Prices', 'Risk Assessment', 'Key Metrics'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "table"}]]
        )
        
        # Confidence scores
        fig.add_trace(
            go.Bar(
                x=stock_data['agent'],
                y=stock_data['confidence_score'],
                name='Confidence',
                marker_color=self.color_palette['agents'][:len(stock_data)]
            ),
            row=1, col=1
        )
        
        # Target prices
        target_data = stock_data.dropna(subset=['target_price'])
        if not target_data.empty:
            fig.add_trace(
                go.Bar(
                    x=target_data['agent'],
                    y=target_data['target_price'],
                    name='Target Price',
                    marker_color=self.color_palette['agents'][:len(target_data)]
                ),
                row=1, col=2
            )
        
        # Risk assessment (convert to numeric)
        risk_numeric = stock_data['risk_assessment'].map({'LOW': 1, 'MODERATE': 2, 'HIGH': 3})
        fig.add_trace(
            go.Bar(
                x=stock_data['agent'],
                y=risk_numeric,
                name='Risk Level',
                marker_color=[self.color_palette['low_risk'] if r == 1 else 
                             self.color_palette['moderate_risk'] if r == 2 else 
                             self.color_palette['high_risk'] for r in risk_numeric]
            ),
            row=2, col=1
        )
        
        # Summary table
        summary_data = stock_data[['agent', 'recommendation', 'confidence_score', 'risk_assessment']].round(2)
        fig.add_trace(
            go.Table(
                header=dict(values=['Agent', 'Recommendation', 'Confidence', 'Risk']),
                cells=dict(values=[summary_data[col] for col in summary_data.columns])
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title={
                'text': f"📈 {stock_symbol} - Multi-Agent Analysis",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            width=1000,
            height=700,
            showlegend=False
        )
        
        return fig
    
    def create_portfolio_optimization_chart(self, analysis_data: List[Dict]) -> go.Figure:
        """
        Create a scatter plot for portfolio optimization (Risk vs Return)
        """
        if not analysis_data:
            return self._create_empty_figure("No data available for portfolio optimization")
        
        df = pd.DataFrame(analysis_data)
        
        # Calculate average metrics per stock
        stock_metrics = df.groupby('stock_symbol').agg({
            'confidence_score': 'mean',
            'target_price': 'mean',
            'current_price': 'first',
            'risk_assessment': lambda x: x.mode().iloc[0] if not x.empty else 'MODERATE',
            'recommendation': lambda x: x.mode().iloc[0] if not x.empty else 'hold',
            'company_name': 'first',
            'sector': 'first'
        }).reset_index()
        
        # Calculate expected return (target price vs current price)
        stock_metrics['expected_return'] = (
            (stock_metrics['target_price'] - stock_metrics['current_price']) / 
            stock_metrics['current_price'] * 100
        ).fillna(0)
        
        # Map risk to numeric
        risk_map = {'LOW': 1, 'MODERATE': 2, 'HIGH': 3}
        stock_metrics['risk_numeric'] = stock_metrics['risk_assessment'].map(risk_map)
        
        # Create scatter plot
        fig = go.Figure()
        
        for rec in stock_metrics['recommendation'].unique():
            rec_data = stock_metrics[stock_metrics['recommendation'] == rec]
            
            fig.add_trace(go.Scatter(
                x=rec_data['risk_numeric'],
                y=rec_data['expected_return'],
                mode='markers+text',
                name=rec.title(),
                text=rec_data['stock_symbol'],
                textposition="top center",
                marker=dict(
                    size=rec_data['confidence_score'] * 30,  # Size based on confidence
                    color=self.color_palette.get(rec, '#cccccc'),
                    opacity=0.7,
                    line=dict(width=2, color='white')
                ),
                hovertemplate="<b>%{text}</b><br>" +
                            "Company: %{customdata[0]}<br>" +
                            "Sector: %{customdata[1]}<br>" +
                            "Expected Return: %{y:.1f}%<br>" +
                            "Risk Level: %{customdata[2]}<br>" +
                            "Confidence: %{customdata[3]:.2f}<br>" +
                            "<extra></extra>",
                customdata=rec_data[['company_name', 'sector', 'risk_assessment', 'confidence_score']].values
            ))
        
        fig.update_layout(
            title={
                'text': "🎯 Portfolio Optimization: Risk vs Expected Return",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title="Risk Level (1=Low, 2=Moderate, 3=High)",
            yaxis_title="Expected Return (%)",
            width=900,
            height=600,
            xaxis=dict(tickmode='array', tickvals=[1, 2, 3], ticktext=['Low', 'Moderate', 'High']),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Add quadrant lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_vline(x=2, line_dash="dash", line_color="gray", opacity=0.5)
        
        return fig
    
    def create_sector_analysis_chart(self, analysis_data: List[Dict]) -> go.Figure:
        """
        Create a sector-wise analysis chart
        """
        if not analysis_data:
            return self._create_empty_figure("No data available for sector analysis")
        
        df = pd.DataFrame(analysis_data)
        
        # Calculate sector metrics
        sector_metrics = df.groupby('sector').agg({
            'confidence_score': 'mean',
            'recommendation': lambda x: (x == 'buy').sum() / len(x),
            'risk_assessment': lambda x: (x == 'LOW').sum() / len(x),
            'stock_symbol': 'nunique'
        }).round(3)
        
        sector_metrics.columns = ['Avg Confidence', 'Buy Ratio', 'Low Risk Ratio', 'Stock Count']
        
        # Create grouped bar chart
        fig = go.Figure()
        
        metrics = ['Avg Confidence', 'Buy Ratio', 'Low Risk Ratio']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        for i, metric in enumerate(metrics):
            fig.add_trace(go.Bar(
                name=metric,
                x=sector_metrics.index,
                y=sector_metrics[metric],
                marker_color=colors[i],
                yaxis='y',
                offsetgroup=i
            ))
        
        fig.update_layout(
            title={
                'text': "🏭 Sector Analysis Overview",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis_title="Sectors",
            yaxis_title="Score (0-1)",
            width=900,
            height=500,
            barmode='group',
            showlegend=True
        )
        
        return fig
    
    def _create_empty_figure(self, message: str) -> go.Figure:
        """Create an empty figure with a message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            xanchor='center', yanchor='middle',
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            width=800,
            height=400
        )
        return fig
    
    def create_performance_summary_table(self, analysis_data: List[Dict]) -> pd.DataFrame:
        """
        Create a comprehensive performance summary table
        """
        if not analysis_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(analysis_data)
        
        # Agent performance summary
        agent_summary = df.groupby('agent').agg({
            'stock_symbol': 'count',
            'confidence_score': ['mean', 'std'],
            'recommendation': lambda x: pd.Series({
                'buy_count': (x == 'buy').sum(),
                'hold_count': (x == 'hold').sum(),
                'sell_count': (x == 'sell').sum(),
                'avoid_count': (x == 'avoid').sum()
            }),
            'risk_assessment': lambda x: pd.Series({
                'low_risk': (x == 'LOW').sum(),
                'moderate_risk': (x == 'MODERATE').sum(),
                'high_risk': (x == 'HIGH').sum()
            }),
            'target_price': lambda x: x.notna().sum(),
            'analysis_time_seconds': 'mean'
        }).round(3)
        
        # Flatten multi-level columns
        agent_summary.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                                for col in agent_summary.columns.values]
        
        return agent_summary.reset_index()

# Utility functions for Streamlit integration
def load_test_data(file_path: str) -> List[Dict]:
    """Load test data from CSV file"""
    try:
        df = pd.read_csv(file_path)
        return df.to_dict('records')
    except Exception as e:
        st.error(f"Error loading test data: {e}")
        return []

def display_visualization_page(visualizer: AlphaAgentsVisualizer, analysis_data: List[Dict]):
    """Display the visualization page in Streamlit"""
    st.markdown('<h1 class="main-header">📊 Advanced Analytics & Visualizations</h1>', unsafe_allow_html=True)
    
    if not analysis_data:
        st.warning("No analysis data available. Please run the test suite first.")
        return
    
    # Sidebar for visualization options
    st.sidebar.subheader("📊 Visualization Options")
    
    viz_options = {
        "Agent Consensus Heatmap": "consensus_heatmap",
        "Risk Assessment Heatmap": "risk_heatmap", 
        "Confidence Distribution": "confidence_dist",
        "Recommendation Distribution": "rec_pie",
        "Agent Performance Radar": "performance_radar",
        "Portfolio Optimization": "portfolio_opt",
        "Sector Analysis": "sector_analysis",
        "Stock Comparison": "stock_comparison"
    }
    
    selected_viz = st.sidebar.selectbox(
        "Select Visualization",
        list(viz_options.keys())
    )
    
    # Display selected visualization
    if viz_options[selected_viz] == "consensus_heatmap":
        st.plotly_chart(visualizer.create_agent_consensus_heatmap(analysis_data), use_container_width=True)
        
    elif viz_options[selected_viz] == "risk_heatmap":
        st.plotly_chart(visualizer.create_risk_assessment_heatmap(analysis_data), use_container_width=True)
        
    elif viz_options[selected_viz] == "confidence_dist":
        st.plotly_chart(visualizer.create_confidence_distribution_chart(analysis_data), use_container_width=True)
        
    elif viz_options[selected_viz] == "rec_pie":
        st.plotly_chart(visualizer.create_recommendation_distribution_pie(analysis_data), use_container_width=True)
        
    elif viz_options[selected_viz] == "performance_radar":
        st.plotly_chart(visualizer.create_agent_performance_radar(analysis_data), use_container_width=True)
        
    elif viz_options[selected_viz] == "portfolio_opt":
        st.plotly_chart(visualizer.create_portfolio_optimization_chart(analysis_data), use_container_width=True)
        
    elif viz_options[selected_viz] == "sector_analysis":
        st.plotly_chart(visualizer.create_sector_analysis_chart(analysis_data), use_container_width=True)
        
    elif viz_options[selected_viz] == "stock_comparison":
        df = pd.DataFrame(analysis_data)
        available_stocks = df['stock_symbol'].unique()
        selected_stock = st.sidebar.selectbox("Select Stock", available_stocks)
        st.plotly_chart(visualizer.create_stock_analysis_comparison(analysis_data, selected_stock), use_container_width=True)
    
    # Performance summary table
    st.subheader("📋 Performance Summary")
    summary_df = visualizer.create_performance_summary_table(analysis_data)
    if not summary_df.empty:
        st.dataframe(summary_df, use_container_width=True)

