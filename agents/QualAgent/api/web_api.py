"""
Simple Web API for QualAgent
Provides REST endpoints for frontend integration
"""

import json
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Any
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.json_data_manager import JSONDataManager, Company
from engines.analysis_engine import AnalysisEngine, AnalysisConfig
from engines.llm_integration import LLMIntegration

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Initialize components
db = JSONDataManager()
engine = AnalysisEngine()
llm = LLMIntegration()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        companies = db.list_companies(limit=1)

        # Test LLM availability
        available_models = llm.get_available_models()

        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'database': 'connected',
            'available_models': len(available_models),
            'models': available_models[:3]  # Show first 3 models
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/companies', methods=['GET'])
def list_companies():
    """List companies in database"""
    try:
        subsector = request.args.get('subsector')
        limit = int(request.args.get('limit', 50))

        companies = db.list_companies(subsector=subsector, limit=limit)

        return jsonify({
            'companies': [
                {
                    'id': c.id,
                    'company_name': c.company_name,
                    'ticker': c.ticker,
                    'subsector': c.subsector,
                    'market_cap_usd': c.market_cap_usd,
                    'description': c.description
                } for c in companies
            ],
            'count': len(companies)
        }), 200
    except Exception as e:
        logger.error(f"Error listing companies: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/companies/search', methods=['GET'])
def search_companies():
    """Search companies by name or ticker"""
    try:
        query = request.args.get('q', '').strip()
        subsector = request.args.get('subsector')

        if not query:
            return jsonify({'error': 'Query parameter "q" is required'}), 400

        results = db.search_companies(query, subsector)

        return jsonify({
            'results': results,
            'count': len(results),
            'query': query
        }), 200
    except Exception as e:
        logger.error(f"Error searching companies: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/companies/<ticker>', methods=['GET'])
def get_company(ticker):
    """Get company details by ticker"""
    try:
        company = db.get_company_by_ticker(ticker.upper())

        if not company:
            return jsonify({'error': f'Company {ticker} not found'}), 404

        # Get latest analysis if available
        latest_analysis = db.get_company_latest_analysis(ticker.upper())

        return jsonify({
            'company': {
                'id': company.id,
                'company_name': company.company_name,
                'ticker': company.ticker,
                'subsector': company.subsector,
                'market_cap_usd': company.market_cap_usd,
                'employees': company.employees,
                'founded_year': company.founded_year,
                'headquarters': company.headquarters,
                'description': company.description,
                'website': company.website
            },
            'latest_analysis': latest_analysis
        }), 200
    except Exception as e:
        logger.error(f"Error getting company {ticker}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/companies', methods=['POST'])
def add_company():
    """Add new company to database"""
    try:
        data = request.get_json()

        required_fields = ['company_name', 'ticker', 'subsector']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Check if company already exists
        existing = db.get_company_by_ticker(data['ticker'].upper())
        if existing:
            return jsonify({'error': f'Company {data["ticker"]} already exists'}), 409

        company = Company(
            company_name=data['company_name'],
            ticker=data['ticker'].upper(),
            subsector=data['subsector'],
            market_cap_usd=data.get('market_cap_usd'),
            employees=data.get('employees'),
            founded_year=data.get('founded_year'),
            headquarters=data.get('headquarters'),
            description=data.get('description'),
            website=data.get('website')
        )

        company_id = db.add_company(company)

        return jsonify({
            'message': f'Company {data["ticker"]} added successfully',
            'company_id': company_id
        }), 201
    except Exception as e:
        logger.error(f"Error adding company: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/analysis/models', methods=['GET'])
def get_available_models():
    """Get available LLM models and their information"""
    try:
        available_models = llm.get_available_models()
        recommended_models = llm.get_recommended_models()

        model_info = []
        for model_key in available_models:
            info = llm.get_model_info(model_key)
            if info:
                model_info.append({
                    'key': model_key,
                    'provider': info.provider,
                    'model_name': info.model_name,
                    'cost_per_1k_tokens': info.cost_per_1k_tokens,
                    'context_window': info.context_window,
                    'recommended': model_key in recommended_models
                })

        return jsonify({
            'available_models': model_info,
            'recommended_models': recommended_models
        }), 200
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/analysis/cost-estimate', methods=['POST'])
def estimate_analysis_cost():
    """Estimate cost for analysis configuration"""
    try:
        data = request.get_json()

        if 'models' not in data:
            return jsonify({'error': 'Models list is required'}), 400

        config = AnalysisConfig(
            models_to_use=data['models'],
            focus_themes=data.get('focus_themes'),
            enable_multi_model_consensus=data.get('enable_consensus', True)
        )

        cost_estimate = engine.get_analysis_cost_estimate(config)

        return jsonify({
            'cost_estimate': cost_estimate,
            'configuration': {
                'models': data['models'],
                'focus_themes': data.get('focus_themes', []),
                'enable_consensus': data.get('enable_consensus', True)
            }
        }), 200
    except Exception as e:
        logger.error(f"Error estimating cost: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/analysis/run', methods=['POST'])
def run_analysis():
    """Run analysis for a company"""
    try:
        data = request.get_json()

        # Validate required fields
        if 'ticker' not in data:
            return jsonify({'error': 'Ticker is required'}), 400

        ticker = data['ticker'].upper()

        # Validate company exists
        company = db.get_company_by_ticker(ticker)
        if not company:
            return jsonify({'error': f'Company {ticker} not found'}), 404

        # Create analysis configuration
        config = AnalysisConfig(
            models_to_use=data.get('models', llm.get_recommended_models()[:1]),
            focus_themes=data.get('focus_themes'),
            geographies_of_interest=data.get('geographies', ['US', 'Global']),
            lookback_window_months=data.get('lookback_window_months', 24),
            enable_multi_model_consensus=data.get('enable_consensus', True),
            requested_by=data.get('requested_by', 'api_user'),
            priority=data.get('priority', 1)
        )

        # Run analysis
        logger.info(f"Starting analysis for {ticker} via API")
        result = engine.analyze_company(ticker, config)

        # Format response
        response = {
            'ticker': ticker,
            'request_id': result.request_id,
            'status': 'completed' if result.success_rate > 0 else 'failed',
            'success_rate': result.success_rate,
            'total_cost_usd': result.total_cost_usd,
            'processing_time_seconds': result.total_processing_time,
            'models_used': [analysis.llm_model for analysis in result.llm_analyses],
            'analyses_completed': len([a for a in result.llm_analyses if a.analysis_status == 'completed']),
            'consensus_generated': result.consensus_analysis is not None,
            'execution_metadata': result.execution_metadata
        }

        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Error running analysis: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/analysis/quick/<ticker>', methods=['POST'])
def quick_analysis(ticker):
    """Run quick single-model analysis"""
    try:
        data = request.get_json() if request.is_json else {}

        ticker = ticker.upper()
        model = data.get('model', llm.get_recommended_models()[0] if llm.get_recommended_models() else 'gpt-4o-mini')

        # Validate company exists
        company = db.get_company_by_ticker(ticker)
        if not company:
            return jsonify({'error': f'Company {ticker} not found'}), 404

        logger.info(f"Starting quick analysis for {ticker} with {model}")
        result = engine.quick_analysis(ticker, model)

        response = {
            'ticker': ticker,
            'model_used': model,
            'request_id': result.request_id,
            'status': 'completed' if result.success_rate > 0 else 'failed',
            'cost_usd': result.total_cost_usd,
            'processing_time_seconds': result.total_processing_time,
            'analysis_id': result.llm_analyses[0].id if result.llm_analyses else None
        }

        return jsonify(response), 200
    except Exception as e:
        logger.error(f"Error in quick analysis for {ticker}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/analysis/<int:analysis_id>', methods=['GET'])
def get_analysis_details(analysis_id):
    """Get detailed analysis results"""
    try:
        details = db.get_analysis_details(analysis_id)

        if not details:
            return jsonify({'error': f'Analysis {analysis_id} not found'}), 404

        return jsonify(details), 200
    except Exception as e:
        logger.error(f"Error getting analysis {analysis_id}: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/analysis/summary', methods=['GET'])
def get_analysis_summary():
    """Get analysis summary for dashboard"""
    try:
        limit = int(request.args.get('limit', 20))
        summary = db.get_analysis_summary(limit)

        return jsonify({
            'analyses': summary,
            'count': len(summary)
        }), 200
    except Exception as e:
        logger.error(f"Error getting analysis summary: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/subsectors', methods=['GET'])
def get_subsectors():
    """Get available subsectors"""
    try:
        subsectors = [
            'Semiconductors', 'Cloud/SaaS', 'Cybersecurity',
            'Consumer/Devices', 'Infrastructure', 'Electronic Components', 'Other Tech'
        ]

        return jsonify({'subsectors': subsectors}), 200
    except Exception as e:
        logger.error(f"Error getting subsectors: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

def create_app(config_name='default'):
    """Application factory"""
    return app

def main():
    """Run the API server"""
    print("Starting QualAgent Web API...")
    print("Available endpoints:")
    print("  GET  /health - Health check")
    print("  GET  /companies - List companies")
    print("  GET  /companies/search?q=<query> - Search companies")
    print("  GET  /companies/<ticker> - Get company details")
    print("  POST /companies - Add new company")
    print("  GET  /analysis/models - Get available models")
    print("  POST /analysis/cost-estimate - Estimate analysis cost")
    print("  POST /analysis/run - Run comprehensive analysis")
    print("  POST /analysis/quick/<ticker> - Run quick analysis")
    print("  GET  /analysis/<id> - Get analysis details")
    print("  GET  /analysis/summary - Get analysis summary")
    print("  GET  /subsectors - Get available subsectors")
    print()
    print("API will be available at: http://localhost:5000")
    print("API documentation: http://localhost:5000/health")

    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    main()