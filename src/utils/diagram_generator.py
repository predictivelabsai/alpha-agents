"""
Diagram Generator for Multi-Agent Collaboration Visualization
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Any

class MultiAgentDiagramGenerator:
    """
    Generates interactive diagrams showing multi-agent collaboration
    """
    
    def __init__(self):
        self.agent_colors = {
            'fundamental': '#1f77b4',  # Blue
            'sentiment': '#ff7f0e',    # Orange
            'valuation': '#2ca02c',    # Green
            'rationale': '#d62728',    # Red
            'secular_trend': '#9467bd', # Purple
            'ranking': '#8c564b'       # Brown
        }
        
        self.agent_positions = {
            'fundamental': (0, 2),
            'sentiment': (2, 2),
            'valuation': (4, 2),
            'rationale': (0, 0),
            'secular_trend': (2, 0),
            'ranking': (2, 1)
        }
    
    def create_collaboration_diagram(self) -> go.Figure:
        """
        Create interactive multi-agent collaboration diagram
        """
        fig = go.Figure()
        
        # Add agent nodes
        for agent, (x, y) in self.agent_positions.items():
            color = self.agent_colors[agent]
            
            # Agent circle
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode='markers+text',
                marker=dict(
                    size=80,
                    color=color,
                    line=dict(width=3, color='white')
                ),
                text=agent.replace('_', '<br>').title(),
                textposition='middle center',
                textfont=dict(size=10, color='white', family='Arial Black'),
                name=agent.title(),
                hovertemplate=f"<b>{agent.title()} Agent</b><br>" +
                            f"Specialization: {self._get_agent_description(agent)}<br>" +
                            "<extra></extra>",
                showlegend=False
            ))
        
        # Add collaboration arrows
        self._add_collaboration_arrows(fig)
        
        # Add data flow indicators
        self._add_data_flow(fig)
        
        # Customize layout
        fig.update_layout(
            title={
                'text': "🤖 Multi-Agent Collaboration Architecture",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Arial Black'}
            },
            xaxis=dict(
                range=[-0.5, 4.5],
                showgrid=False,
                showticklabels=False,
                zeroline=False
            ),
            yaxis=dict(
                range=[-0.5, 2.5],
                showgrid=False,
                showticklabels=False,
                zeroline=False
            ),
            plot_bgcolor='rgba(240,248,255,0.8)',
            paper_bgcolor='white',
            height=500,
            margin=dict(l=20, r=20, t=60, b=20),
            annotations=self._create_annotations()
        )
        
        return fig
    
    def create_agent_workflow_diagram(self) -> go.Figure:
        """
        Create workflow diagram showing agent processing sequence
        """
        fig = go.Figure()
        
        # Workflow steps
        steps = [
            ('Data Input', 'yfinance API', '#17a2b8'),
            ('Fundamental', 'Financial Analysis', '#1f77b4'),
            ('Sentiment', 'News & Market Psychology', '#ff7f0e'),
            ('Valuation', 'Technical & Price Analysis', '#2ca02c'),
            ('Rationale', '7-Step Business Quality', '#d62728'),
            ('Secular Trend', 'Technology Positioning', '#9467bd'),
            ('Ranking', 'Final Synthesis', '#8c564b'),
            ('Output', 'Investment Decision', '#28a745')
        ]
        
        # Create workflow boxes
        for i, (step, description, color) in enumerate(steps):
            x = i
            y = 0
            
            # Box
            fig.add_shape(
                type="rect",
                x0=x-0.4, y0=y-0.3,
                x1=x+0.4, y1=y+0.3,
                fillcolor=color,
                line=dict(color="white", width=2),
                opacity=0.8
            )
            
            # Text
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode='text',
                text=f"<b>{step}</b><br>{description}",
                textfont=dict(size=10, color='white', family='Arial'),
                showlegend=False,
                hovertemplate=f"<b>{step}</b><br>{description}<extra></extra>"
            ))
            
            # Arrow to next step
            if i < len(steps) - 1:
                fig.add_annotation(
                    x=x+0.5, y=y,
                    ax=x+0.4, ay=y,
                    xref='x', yref='y',
                    axref='x', ayref='y',
                    arrowhead=2,
                    arrowsize=1.5,
                    arrowwidth=2,
                    arrowcolor='#333333'
                )
        
        fig.update_layout(
            title={
                'text': "📊 Agent Processing Workflow",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'family': 'Arial Black'}
            },
            xaxis=dict(
                range=[-0.8, len(steps)-0.2],
                showgrid=False,
                showticklabels=False,
                zeroline=False
            ),
            yaxis=dict(
                range=[-0.8, 0.8],
                showgrid=False,
                showticklabels=False,
                zeroline=False
            ),
            plot_bgcolor='rgba(248,249,250,0.8)',
            paper_bgcolor='white',
            height=200,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        return fig
    
    def create_agent_performance_radar(self, agent_scores: Dict[str, float]) -> go.Figure:
        """
        Create radar chart showing agent performance/confidence
        """
        agents = list(agent_scores.keys())
        scores = list(agent_scores.values())
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=scores + [scores[0]],  # Close the polygon
            theta=agents + [agents[0]],
            fill='toself',
            fillcolor='rgba(31, 119, 180, 0.3)',
            line=dict(color='rgba(31, 119, 180, 1)', width=2),
            marker=dict(size=8, color='rgba(31, 119, 180, 1)'),
            name='Agent Performance'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickmode='linear',
                    tick0=0,
                    dtick=0.2
                )
            ),
            title={
                'text': "🎯 Agent Performance Radar",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16, 'family': 'Arial Black'}
            },
            height=400,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        return fig
    
    def _add_collaboration_arrows(self, fig: go.Figure):
        """Add arrows showing agent collaboration"""
        
        # Arrows from analysis agents to ranking agent
        analysis_agents = ['fundamental', 'sentiment', 'valuation', 'rationale', 'secular_trend']
        ranking_pos = self.agent_positions['ranking']
        
        for agent in analysis_agents:
            start_pos = self.agent_positions[agent]
            
            # Calculate arrow direction
            dx = ranking_pos[0] - start_pos[0]
            dy = ranking_pos[1] - start_pos[1]
            
            # Normalize and adjust for node size
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                dx_norm = dx / length * 0.3
                dy_norm = dy / length * 0.3
                
                fig.add_annotation(
                    x=ranking_pos[0] - dx_norm,
                    y=ranking_pos[1] - dy_norm,
                    ax=start_pos[0] + dx_norm,
                    ay=start_pos[1] + dy_norm,
                    xref='x', yref='y',
                    axref='x', ayref='y',
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=self.agent_colors[agent],
                    opacity=0.7
                )
    
    def _add_data_flow(self, fig: go.Figure):
        """Add data flow indicators"""
        
        # Data input flow
        fig.add_trace(go.Scatter(
            x=[-0.3, 0.3, 2.3, 4.3],
            y=[2.3, 2.3, 2.3, 2.3],
            mode='lines',
            line=dict(color='#17a2b8', width=3, dash='dot'),
            name='Data Flow',
            showlegend=False,
            hovertemplate="yfinance Data Input<extra></extra>"
        ))
        
        # Add data source annotation
        fig.add_annotation(
            x=-0.3, y=2.3,
            text="📊 yfinance<br>Data",
            showarrow=False,
            font=dict(size=10, color='#17a2b8'),
            bgcolor='rgba(23, 162, 184, 0.1)',
            bordercolor='#17a2b8',
            borderwidth=1
        )
    
    def _create_annotations(self) -> List[Dict]:
        """Create annotations for the diagram"""
        return [
            dict(
                x=2, y=-0.3,
                text="<b>🏆 Final Investment Decision</b>",
                showarrow=False,
                font=dict(size=12, color='#28a745'),
                bgcolor='rgba(40, 167, 69, 0.1)',
                bordercolor='#28a745',
                borderwidth=1
            ),
            dict(
                x=4.2, y=1.5,
                text="<b>Collaboration Flow</b><br>• Data sharing<br>• Cross-validation<br>• Consensus building",
                showarrow=False,
                font=dict(size=10, color='#6c757d'),
                bgcolor='rgba(248, 249, 250, 0.9)',
                bordercolor='#dee2e6',
                borderwidth=1,
                align='left'
            )
        ]
    
    def _get_agent_description(self, agent: str) -> str:
        """Get description for each agent"""
        descriptions = {
            'fundamental': 'Financial statement analysis & DCF valuation',
            'sentiment': 'News analysis & market psychology',
            'valuation': 'Technical analysis & price momentum',
            'rationale': '7-step business quality framework',
            'secular_trend': 'Technology trend positioning',
            'ranking': 'Multi-agent synthesis & final decision'
        }
        return descriptions.get(agent, 'Specialized analysis')

# Global instance
diagram_generator = MultiAgentDiagramGenerator()

