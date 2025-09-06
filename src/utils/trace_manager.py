"""
Trace Manager - Lohusalu Capital Management
Comprehensive JSON tracing and reasoning trace management system
"""

import os
import json
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import logging
from pathlib import Path
import glob

class TraceManager:
    """
    Comprehensive trace management system for agent reasoning and analysis
    
    Features:
    - JSON trace storage and retrieval
    - Expandable reasoning traces
    - DataFrame conversion for analysis
    - Trace comparison and analytics
    - Search and filtering capabilities
    """
    
    def __init__(self, trace_dir: str = "tracing", logs_dir: str = "logs"):
        self.trace_dir = Path(trace_dir)
        self.logs_dir = Path(logs_dir)
        self.logger = logging.getLogger(__name__)
        
        # Create directories if they don't exist
        self.trace_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Trace file patterns
        self.trace_patterns = {
            'fundamental': 'fundamental_agent_trace_*.json',
            'rationale': 'rationale_agent_trace_*.json',
            'ranker': 'ranker_agent_trace_*.json',
            'agentic_screener': 'agentic_screener_trace_*.json'
        }
    
    def save_trace(self, trace_data: Dict[str, Any], agent_type: str, 
                   custom_filename: Optional[str] = None) -> str:
        """
        Save trace data to JSON file
        
        Args:
            trace_data: Dictionary containing trace information
            agent_type: Type of agent ('fundamental', 'rationale', 'ranker', 'agentic_screener')
            custom_filename: Optional custom filename
            
        Returns:
            Path to saved trace file
        """
        try:
            # Add metadata
            trace_data.update({
                'timestamp': datetime.now().isoformat(),
                'agent_type': agent_type,
                'trace_version': '2.0'
            })
            
            # Generate filename
            if custom_filename:
                filename = custom_filename
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{agent_type}_agent_trace_{timestamp}.json"
            
            filepath = self.trace_dir / filename
            
            # Save trace
            with open(filepath, 'w') as f:
                json.dump(trace_data, f, indent=2, default=str)
            
            self.logger.info(f"Trace saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"Error saving trace: {e}")
            return ""
    
    def load_trace(self, filepath: str) -> Optional[Dict[str, Any]]:
        """Load trace data from JSON file"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading trace {filepath}: {e}")
            return None
    
    def get_traces_by_agent(self, agent_type: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get traces for specific agent type"""
        try:
            pattern = self.trace_patterns.get(agent_type, f"{agent_type}_agent_trace_*.json")
            trace_files = list(self.trace_dir.glob(pattern))
            
            # Sort by modification time (newest first)
            trace_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            if limit:
                trace_files = trace_files[:limit]
            
            traces = []
            for filepath in trace_files:
                trace_data = self.load_trace(filepath)
                if trace_data:
                    trace_data['filepath'] = str(filepath)
                    trace_data['filename'] = filepath.name
                    traces.append(trace_data)
            
            return traces
            
        except Exception as e:
            self.logger.error(f"Error getting traces for {agent_type}: {e}")
            return []
    
    def get_all_traces(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all traces across all agents"""
        try:
            trace_files = list(self.trace_dir.glob("*_agent_trace_*.json"))
            
            # Sort by modification time (newest first)
            trace_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            if limit:
                trace_files = trace_files[:limit]
            
            traces = []
            for filepath in trace_files:
                trace_data = self.load_trace(filepath)
                if trace_data:
                    trace_data['filepath'] = str(filepath)
                    trace_data['filename'] = filepath.name
                    traces.append(trace_data)
            
            return traces
            
        except Exception as e:
            self.logger.error(f"Error getting all traces: {e}")
            return []
    
    def search_traces(self, query: str, agent_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search traces by content"""
        try:
            if agent_type:
                traces = self.get_traces_by_agent(agent_type)
            else:
                traces = self.get_all_traces()
            
            matching_traces = []
            query_lower = query.lower()
            
            for trace in traces:
                # Search in trace content
                trace_str = json.dumps(trace, default=str).lower()
                if query_lower in trace_str:
                    matching_traces.append(trace)
            
            return matching_traces
            
        except Exception as e:
            self.logger.error(f"Error searching traces: {e}")
            return []
    
    def traces_to_dataframe(self, traces: List[Dict[str, Any]], agent_type: str) -> pd.DataFrame:
        """Convert traces to DataFrame for analysis"""
        try:
            if not traces:
                return pd.DataFrame()
            
            if agent_type == 'fundamental':
                return self._fundamental_traces_to_df(traces)
            elif agent_type == 'rationale':
                return self._rationale_traces_to_df(traces)
            elif agent_type == 'ranker':
                return self._ranker_traces_to_df(traces)
            elif agent_type == 'agentic_screener':
                return self._agentic_screener_traces_to_df(traces)
            else:
                return self._generic_traces_to_df(traces)
                
        except Exception as e:
            self.logger.error(f"Error converting traces to DataFrame: {e}")
            return pd.DataFrame()
    
    def _fundamental_traces_to_df(self, traces: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert fundamental agent traces to DataFrame"""
        rows = []
        
        for trace in traces:
            base_info = {
                'timestamp': trace.get('timestamp'),
                'model_provider': trace.get('model_provider'),
                'model_name': trace.get('model_name'),
                'filename': trace.get('filename')
            }
            
            # Sector analyses
            sector_analyses = trace.get('sector_analyses', [])
            for sector in sector_analyses:
                row = base_info.copy()
                row.update({
                    'type': 'sector_analysis',
                    'sector': sector.get('sector'),
                    'weight': sector.get('weight'),
                    'momentum_score': sector.get('momentum_score'),
                    'growth_potential': sector.get('growth_potential')
                })
                rows.append(row)
            
            # Stock screenings
            stock_screenings = trace.get('stock_screenings', [])
            for stock in stock_screenings:
                row = base_info.copy()
                row.update({
                    'type': 'stock_screening',
                    'ticker': stock.get('ticker'),
                    'sector': stock.get('sector'),
                    'fundamental_score': stock.get('fundamental_score'),
                    'market_cap': stock.get('market_cap'),
                    'upside_potential': stock.get('upside_potential')
                })
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _rationale_traces_to_df(self, traces: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert rationale agent traces to DataFrame"""
        rows = []
        
        for trace in traces:
            analysis = trace.get('analysis', {})
            
            row = {
                'timestamp': trace.get('timestamp'),
                'model_provider': trace.get('model_provider'),
                'model_name': trace.get('model_name'),
                'filename': trace.get('filename'),
                'ticker': analysis.get('ticker'),
                'company_name': analysis.get('company_name'),
                'qualitative_score': analysis.get('qualitative_score'),
                'moat_score': analysis.get('moat_analysis', {}).get('moat_score'),
                'moat_strength': analysis.get('moat_analysis', {}).get('moat_strength'),
                'sentiment_score': analysis.get('sentiment_analysis', {}).get('sentiment_score'),
                'sentiment': analysis.get('sentiment_analysis', {}).get('overall_sentiment'),
                'trends_score': analysis.get('secular_trends', {}).get('trend_score'),
                'trend_alignment': analysis.get('secular_trends', {}).get('trend_alignment'),
                'competitive_score': analysis.get('competitive_position', {}).get('competitive_score'),
                'market_position': analysis.get('competitive_position', {}).get('market_position')
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _ranker_traces_to_df(self, traces: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert ranker agent traces to DataFrame"""
        rows = []
        
        for trace in traces:
            investment_scores = trace.get('investment_scores', [])
            
            for score in investment_scores:
                row = {
                    'timestamp': trace.get('timestamp'),
                    'model_provider': trace.get('model_provider'),
                    'model_name': trace.get('model_name'),
                    'filename': trace.get('filename'),
                    'ticker': score.get('ticker'),
                    'company_name': score.get('company_name'),
                    'sector': score.get('sector'),
                    'composite_score': score.get('composite_score'),
                    'investment_grade': score.get('investment_grade'),
                    'fundamental_score': score.get('component_scores', {}).get('fundamental_score'),
                    'qualitative_score': score.get('component_scores', {}).get('qualitative_score'),
                    'upside_potential': score.get('upside_potential'),
                    'risk_rating': score.get('risk_rating'),
                    'conviction_level': score.get('conviction_level'),
                    'time_horizon': score.get('time_horizon')
                }
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _agentic_screener_traces_to_df(self, traces: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert agentic screener traces to DataFrame"""
        rows = []
        
        for trace in traces:
            pipeline_results = trace.get('pipeline_results', {})
            
            row = {
                'timestamp': trace.get('timestamp'),
                'model_provider': trace.get('model_provider'),
                'model_name': trace.get('model_name'),
                'filename': trace.get('filename'),
                'total_stocks_analyzed': len(pipeline_results.get('final_recommendations', [])),
                'avg_composite_score': self._calculate_avg_score(pipeline_results.get('final_recommendations', [])),
                'execution_time': trace.get('execution_time'),
                'sectors_analyzed': len(set([stock.get('sector', '') for stock in pipeline_results.get('final_recommendations', [])]))
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _generic_traces_to_df(self, traces: List[Dict[str, Any]]) -> pd.DataFrame:
        """Convert generic traces to DataFrame"""
        rows = []
        
        for trace in traces:
            row = {
                'timestamp': trace.get('timestamp'),
                'agent_type': trace.get('agent_type'),
                'model_provider': trace.get('model_provider'),
                'model_name': trace.get('model_name'),
                'filename': trace.get('filename'),
                'trace_version': trace.get('trace_version')
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _calculate_avg_score(self, recommendations: List[Dict]) -> float:
        """Calculate average composite score from recommendations"""
        if not recommendations:
            return 0.0
        
        scores = [rec.get('composite_score', 0) for rec in recommendations]
        return sum(scores) / len(scores) if scores else 0.0
    
    def compare_traces(self, trace_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple traces"""
        try:
            traces = []
            for trace_id in trace_ids:
                trace_path = self.trace_dir / trace_id
                trace_data = self.load_trace(trace_path)
                if trace_data:
                    traces.append(trace_data)
            
            if len(traces) < 2:
                return {'error': 'Need at least 2 traces for comparison'}
            
            comparison = {
                'trace_count': len(traces),
                'agent_types': [trace.get('agent_type') for trace in traces],
                'models_used': [f"{trace.get('model_provider')}-{trace.get('model_name')}" for trace in traces],
                'timestamps': [trace.get('timestamp') for trace in traces]
            }
            
            # Add specific comparisons based on agent type
            if all(trace.get('agent_type') == 'ranker' for trace in traces):
                comparison.update(self._compare_ranker_traces(traces))
            elif all(trace.get('agent_type') == 'fundamental' for trace in traces):
                comparison.update(self._compare_fundamental_traces(traces))
            elif all(trace.get('agent_type') == 'rationale' for trace in traces):
                comparison.update(self._compare_rationale_traces(traces))
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing traces: {e}")
            return {'error': str(e)}
    
    def _compare_ranker_traces(self, traces: List[Dict]) -> Dict[str, Any]:
        """Compare ranker agent traces"""
        comparison = {}
        
        # Compare average scores
        avg_scores = []
        for trace in traces:
            scores = [score.get('composite_score', 0) for score in trace.get('investment_scores', [])]
            avg_scores.append(sum(scores) / len(scores) if scores else 0)
        
        comparison['average_scores'] = avg_scores
        comparison['score_difference'] = max(avg_scores) - min(avg_scores) if avg_scores else 0
        
        return comparison
    
    def _compare_fundamental_traces(self, traces: List[Dict]) -> Dict[str, Any]:
        """Compare fundamental agent traces"""
        comparison = {}
        
        # Compare number of stocks found
        stocks_found = [len(trace.get('stock_screenings', [])) for trace in traces]
        comparison['stocks_found'] = stocks_found
        comparison['stocks_difference'] = max(stocks_found) - min(stocks_found) if stocks_found else 0
        
        return comparison
    
    def _compare_rationale_traces(self, traces: List[Dict]) -> Dict[str, Any]:
        """Compare rationale agent traces"""
        comparison = {}
        
        # Compare qualitative scores
        qual_scores = [trace.get('analysis', {}).get('qualitative_score', 0) for trace in traces]
        comparison['qualitative_scores'] = qual_scores
        comparison['score_difference'] = max(qual_scores) - min(qual_scores) if qual_scores else 0
        
        return comparison
    
    def get_trace_analytics(self, agent_type: Optional[str] = None, 
                           days_back: int = 30) -> Dict[str, Any]:
        """Get analytics for traces"""
        try:
            # Get traces from specified time period
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            if agent_type:
                traces = self.get_traces_by_agent(agent_type)
            else:
                traces = self.get_all_traces()
            
            # Filter by date
            recent_traces = []
            for trace in traces:
                trace_date = datetime.fromisoformat(trace.get('timestamp', ''))
                if trace_date >= cutoff_date:
                    recent_traces.append(trace)
            
            analytics = {
                'total_traces': len(recent_traces),
                'date_range': f"Last {days_back} days",
                'agent_breakdown': {},
                'model_usage': {},
                'daily_activity': {}
            }
            
            # Agent breakdown
            for trace in recent_traces:
                agent = trace.get('agent_type', 'unknown')
                analytics['agent_breakdown'][agent] = analytics['agent_breakdown'].get(agent, 0) + 1
            
            # Model usage
            for trace in recent_traces:
                model = f"{trace.get('model_provider', 'unknown')}-{trace.get('model_name', 'unknown')}"
                analytics['model_usage'][model] = analytics['model_usage'].get(model, 0) + 1
            
            # Daily activity
            for trace in recent_traces:
                date = trace.get('timestamp', '')[:10]  # Get date part
                analytics['daily_activity'][date] = analytics['daily_activity'].get(date, 0) + 1
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"Error getting trace analytics: {e}")
            return {}
    
    def cleanup_old_traces(self, days_to_keep: int = 90) -> int:
        """Clean up old trace files"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            deleted_count = 0
            
            for trace_file in self.trace_dir.glob("*_agent_trace_*.json"):
                file_time = datetime.fromtimestamp(trace_file.stat().st_mtime)
                if file_time < cutoff_date:
                    trace_file.unlink()
                    deleted_count += 1
            
            self.logger.info(f"Cleaned up {deleted_count} old trace files")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Error cleaning up traces: {e}")
            return 0
    
    def export_traces_to_csv(self, agent_type: str, output_path: Optional[str] = None) -> str:
        """Export traces to CSV file"""
        try:
            traces = self.get_traces_by_agent(agent_type)
            df = self.traces_to_dataframe(traces, agent_type)
            
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"{agent_type}_traces_{timestamp}.csv"
            
            df.to_csv(output_path, index=False)
            self.logger.info(f"Traces exported to {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Error exporting traces: {e}")
            return ""

# Streamlit integration functions
def display_trace_manager_interface():
    """Display trace manager interface in Streamlit"""
    
    st.markdown("## 📁 Trace Management System")
    
    trace_manager = TraceManager()
    
    # Sidebar controls
    with st.sidebar:
        st.markdown("### 🔧 Trace Controls")
        
        agent_filter = st.selectbox(
            "Filter by Agent",
            options=["All", "fundamental", "rationale", "ranker", "agentic_screener"],
            index=0
        )
        
        days_back = st.slider(
            "Days to Show",
            min_value=1,
            max_value=90,
            value=30
        )
        
        search_query = st.text_input(
            "Search Traces",
            placeholder="Enter search term..."
        )
    
    # Main interface tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Recent Traces", "📈 Analytics", "🔍 Search", "⚙️ Management"])
    
    with tab1:
        display_recent_traces_tab(trace_manager, agent_filter, days_back)
    
    with tab2:
        display_analytics_tab(trace_manager, agent_filter, days_back)
    
    with tab3:
        display_search_tab(trace_manager, search_query, agent_filter)
    
    with tab4:
        display_management_tab(trace_manager)

def display_recent_traces_tab(trace_manager: TraceManager, agent_filter: str, days_back: int):
    """Display recent traces tab"""
    
    # Get traces
    if agent_filter == "All":
        traces = trace_manager.get_all_traces(limit=50)
    else:
        traces = trace_manager.get_traces_by_agent(agent_filter, limit=50)
    
    if not traces:
        st.info("No traces found for the selected criteria.")
        return
    
    # Filter by date
    cutoff_date = datetime.now() - timedelta(days=days_back)
    recent_traces = [
        trace for trace in traces
        if datetime.fromisoformat(trace.get('timestamp', '')) >= cutoff_date
    ]
    
    st.markdown(f"### Found {len(recent_traces)} traces in the last {days_back} days")
    
    # Display traces
    for trace in recent_traces[:20]:  # Show first 20
        with st.expander(f"📄 {trace.get('filename', 'Unknown')} - {trace.get('timestamp', 'Unknown')[:19]}", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Trace Info:**")
                st.markdown(f"- Agent: {trace.get('agent_type', 'Unknown')}")
                st.markdown(f"- Model: {trace.get('model_provider', 'Unknown')} - {trace.get('model_name', 'Unknown')}")
                st.markdown(f"- Version: {trace.get('trace_version', 'Unknown')}")
            
            with col2:
                st.markdown("**Actions:**")
                if st.button(f"📥 Download", key=f"download_{trace.get('filename')}"):
                    with open(trace.get('filepath'), 'r') as f:
                        st.download_button(
                            label="Download JSON",
                            data=f.read(),
                            file_name=trace.get('filename'),
                            mime="application/json"
                        )
            
            # Show expandable content
            if st.checkbox(f"Show content", key=f"show_{trace.get('filename')}"):
                st.json(trace)

def display_analytics_tab(trace_manager: TraceManager, agent_filter: str, days_back: int):
    """Display analytics tab"""
    
    # Get analytics
    analytics = trace_manager.get_trace_analytics(
        agent_type=None if agent_filter == "All" else agent_filter,
        days_back=days_back
    )
    
    if not analytics:
        st.info("No analytics data available.")
        return
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Traces", analytics.get('total_traces', 0))
    
    with col2:
        st.metric("Date Range", analytics.get('date_range', 'Unknown'))
    
    with col3:
        agent_count = len(analytics.get('agent_breakdown', {}))
        st.metric("Agent Types", agent_count)
    
    # Agent breakdown
    if analytics.get('agent_breakdown'):
        st.markdown("### 🤖 Agent Usage")
        agent_df = pd.DataFrame([
            {'Agent': agent, 'Traces': count}
            for agent, count in analytics['agent_breakdown'].items()
        ])
        
        fig = px.pie(agent_df, values='Traces', names='Agent', title="Traces by Agent Type")
        st.plotly_chart(fig, use_container_width=True)
    
    # Model usage
    if analytics.get('model_usage'):
        st.markdown("### 🧠 Model Usage")
        model_df = pd.DataFrame([
            {'Model': model, 'Usage': count}
            for model, count in analytics['model_usage'].items()
        ])
        
        fig = px.bar(model_df, x='Model', y='Usage', title="Model Usage Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Daily activity
    if analytics.get('daily_activity'):
        st.markdown("### 📅 Daily Activity")
        daily_df = pd.DataFrame([
            {'Date': date, 'Traces': count}
            for date, count in analytics['daily_activity'].items()
        ])
        daily_df['Date'] = pd.to_datetime(daily_df['Date'])
        daily_df = daily_df.sort_values('Date')
        
        fig = px.line(daily_df, x='Date', y='Traces', title="Daily Trace Activity")
        st.plotly_chart(fig, use_container_width=True)

def display_search_tab(trace_manager: TraceManager, search_query: str, agent_filter: str):
    """Display search tab"""
    
    if not search_query:
        st.info("Enter a search term to find traces.")
        return
    
    # Perform search
    agent_type = None if agent_filter == "All" else agent_filter
    results = trace_manager.search_traces(search_query, agent_type)
    
    st.markdown(f"### Found {len(results)} traces matching '{search_query}'")
    
    if not results:
        st.info("No traces found matching your search criteria.")
        return
    
    # Display results
    for result in results[:10]:  # Show first 10 results
        with st.expander(f"📄 {result.get('filename', 'Unknown')}", expanded=False):
            st.markdown(f"**Agent:** {result.get('agent_type', 'Unknown')}")
            st.markdown(f"**Timestamp:** {result.get('timestamp', 'Unknown')}")
            st.markdown(f"**Model:** {result.get('model_provider', 'Unknown')} - {result.get('model_name', 'Unknown')}")
            
            if st.button(f"View Details", key=f"view_{result.get('filename')}"):
                st.json(result)

def display_management_tab(trace_manager: TraceManager):
    """Display management tab"""
    
    st.markdown("### 🛠️ Trace Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Export Traces")
        
        export_agent = st.selectbox(
            "Select Agent for Export",
            options=["fundamental", "rationale", "ranker", "agentic_screener"]
        )
        
        if st.button("📤 Export to CSV"):
            output_path = trace_manager.export_traces_to_csv(export_agent)
            if output_path:
                st.success(f"Traces exported to {output_path}")
                
                # Provide download
                with open(output_path, 'r') as f:
                    st.download_button(
                        label="Download CSV",
                        data=f.read(),
                        file_name=output_path,
                        mime="text/csv"
                    )
    
    with col2:
        st.markdown("#### Cleanup")
        
        days_to_keep = st.slider(
            "Days to Keep",
            min_value=7,
            max_value=365,
            value=90
        )
        
        if st.button("🗑️ Cleanup Old Traces"):
            deleted_count = trace_manager.cleanup_old_traces(days_to_keep)
            st.success(f"Deleted {deleted_count} old trace files")
    
    # DataFrame conversion
    st.markdown("#### Convert to DataFrame")
    
    df_agent = st.selectbox(
        "Select Agent for DataFrame",
        options=["fundamental", "rationale", "ranker", "agentic_screener"],
        key="df_agent"
    )
    
    if st.button("📊 Convert to DataFrame"):
        traces = trace_manager.get_traces_by_agent(df_agent, limit=100)
        if traces:
            df = trace_manager.traces_to_dataframe(traces, df_agent)
            st.dataframe(df, use_container_width=True)
            
            # Download option
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download DataFrame as CSV",
                data=csv,
                file_name=f"{df_agent}_traces_dataframe.csv",
                mime="text/csv"
            )
        else:
            st.info(f"No traces found for {df_agent} agent")

if __name__ == "__main__":
    # Test the trace manager
    trace_manager = TraceManager()
    
    # Test saving a trace
    test_trace = {
        'test_data': 'This is a test trace',
        'agent_analysis': {'score': 85, 'reasoning': 'Test reasoning'}
    }
    
    filepath = trace_manager.save_trace(test_trace, 'test')
    print(f"Test trace saved to: {filepath}")
    
    # Test loading
    loaded_trace = trace_manager.load_trace(filepath)
    print(f"Loaded trace: {loaded_trace}")

