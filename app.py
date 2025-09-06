#!/usr/bin/env python3
"""
Main application entry point for Lohusalu Capital Management
"""

import os
import sys
import streamlit as st

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Set environment variables for deployment
os.environ.setdefault('OPENAI_API_KEY', os.getenv('OPENAI_API_KEY', ''))
os.environ.setdefault('TAVILY_API_KEY', os.getenv('TAVILY_API_KEY', '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'))

# Import and run the main Home application
if __name__ == "__main__":
    # Configure Streamlit
    st.set_page_config(
        page_title="Lohusalu Capital Management",
        page_icon="🏛️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Import and run Home.py
    exec(open('Home.py').read())

