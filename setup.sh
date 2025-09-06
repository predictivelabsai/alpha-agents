#!/bin/bash

# Setup script for Lohusalu Capital Management deployment

echo "🚀 Setting up Lohusalu Capital Management..."

# Create necessary directories
mkdir -p tracing logs .streamlit

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo "✅ Setup complete!"
echo "🌐 Run with: streamlit run Home.py"

