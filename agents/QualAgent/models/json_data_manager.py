"""
JSON-based Data Manager for QualAgent system
Replaces SQLite database with JSON file storage for easier deployment
"""

import json
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Company:
    """Company data model"""
    company_name: str
    ticker: str
    subsector: str
    market_cap_usd: Optional[float] = None
    employees: Optional[int] = None
    founded_year: Optional[int] = None
    headquarters: Optional[str] = None
    description: Optional[str] = None
    website: Optional[str] = None
    status: str = 'active'
    id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

@dataclass
class AnalysisRequest:
    """Analysis request data model"""
    company_id: str
    focus_themes: List[str] = None
    geographies_of_interest: List[str] = None
    lookback_window_months: int = 24
    tools_available: List[str] = None
    requested_by: Optional[str] = None
    priority: int = 1
    id: Optional[str] = None
    request_date: Optional[str] = None
    status: str = 'pending'
    error_message: Optional[str] = None
    processing_time_seconds: Optional[float] = None

@dataclass
class LLMAnalysis:
    """LLM analysis result data model"""
    request_id: str
    llm_model: str
    llm_provider: str
    input_prompt: str
    raw_output: str
    parsed_output: Optional[Dict] = None
    analysis_status: str = 'completed'
    tokens_used: Optional[int] = None
    cost_usd: Optional[float] = None
    processing_time_seconds: Optional[float] = None
    error_details: Optional[str] = None
    id: Optional[str] = None
    analysis_date: Optional[str] = None

class JSONDataManager:
    """JSON-based data manager for QualAgent system"""

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data"

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)

        # Data file paths
        self.companies_file = self.data_dir / "companies.json"
        self.analysis_requests_file = self.data_dir / "analysis_requests.json"
        self.llm_analyses_file = self.data_dir / "llm_analyses.json"
        self.structured_results_file = self.data_dir / "structured_results.json"

        # Initialize data files
        self._init_data_files()
        logger.info(f"JSON Data Manager initialized at: {self.data_dir}")

    def _init_data_files(self):
        """Initialize JSON data files if they don't exist"""
        for file_path in [self.companies_file, self.analysis_requests_file,
                         self.llm_analyses_file, self.structured_results_file]:
            if not file_path.exists():
                with open(file_path, 'w') as f:
                    json.dump([], f)

    def _load_json(self, file_path: Path) -> List[Dict]:
        """Load JSON data from file"""
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_json(self, file_path: Path, data: List[Dict]):
        """Save JSON data to file"""
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def _generate_id(self) -> str:
        """Generate unique ID"""
        return str(uuid.uuid4())

    def _add_timestamps(self, record: Dict, update: bool = False) -> Dict:
        """Add timestamp fields to record"""
        now = datetime.now().isoformat()
        if not update:
            record['created_at'] = now
        record['updated_at'] = now
        return record

    # Company operations
    def add_company(self, company: Company) -> str:
        """Add new company to storage"""
        companies = self._load_json(self.companies_file)

        # Check if company already exists
        existing = next((c for c in companies if c['ticker'] == company.ticker), None)
        if existing:
            raise ValueError(f"Company with ticker {company.ticker} already exists")

        # Add new company
        company_dict = asdict(company)
        company_dict['id'] = self._generate_id()
        company_dict = self._add_timestamps(company_dict)

        companies.append(company_dict)
        self._save_json(self.companies_file, companies)

        logger.info(f"Added company: {company.company_name} ({company.ticker}) with ID: {company_dict['id']}")
        return company_dict['id']

    def get_company_by_ticker(self, ticker: str) -> Optional[Company]:
        """Get company by ticker symbol"""
        companies = self._load_json(self.companies_file)
        company_dict = next((c for c in companies if c['ticker'] == ticker and c['status'] == 'active'), None)

        if company_dict:
            return Company(**company_dict)
        return None

    def get_company_by_id(self, company_id: str) -> Optional[Company]:
        """Get company by ID"""
        companies = self._load_json(self.companies_file)
        company_dict = next((c for c in companies if c['id'] == company_id), None)

        if company_dict:
            return Company(**company_dict)
        return None

    def list_companies(self, subsector: str = None, limit: int = 100) -> List[Company]:
        """List companies with optional subsector filter"""
        companies = self._load_json(self.companies_file)

        # Filter active companies
        filtered = [c for c in companies if c['status'] == 'active']

        # Filter by subsector if specified
        if subsector:
            filtered = [c for c in filtered if c['subsector'] == subsector]

        # Sort by company name and limit
        filtered = sorted(filtered, key=lambda x: x['company_name'])[:limit]

        return [Company(**c) for c in filtered]

    def search_companies(self, query: str, subsector: str = None) -> List[Dict]:
        """Search companies by name or ticker"""
        companies = self._load_json(self.companies_file)
        query = query.lower()

        # Filter active companies
        filtered = [c for c in companies if c['status'] == 'active']

        # Search in company name and ticker
        filtered = [c for c in filtered if
                   query in c['company_name'].lower() or
                   query in c['ticker'].lower()]

        # Filter by subsector if specified
        if subsector:
            filtered = [c for c in filtered if c['subsector'] == subsector]

        # Sort by relevance (exact ticker match first, then company name)
        def sort_key(company):
            if company['ticker'].lower() == query:
                return (0, company['company_name'])
            elif query in company['ticker'].lower():
                return (1, company['company_name'])
            else:
                return (2, company['company_name'])

        filtered = sorted(filtered, key=sort_key)[:20]
        return filtered

    # Analysis request operations
    def create_analysis_request(self, request: AnalysisRequest) -> str:
        """Create new analysis request"""
        requests = self._load_json(self.analysis_requests_file)

        request_dict = asdict(request)
        request_dict['id'] = self._generate_id()
        request_dict['request_date'] = datetime.now().isoformat()
        request_dict = self._add_timestamps(request_dict)

        requests.append(request_dict)
        self._save_json(self.analysis_requests_file, requests)

        logger.info(f"Created analysis request ID: {request_dict['id']} for company ID: {request.company_id}")
        return request_dict['id']

    def get_analysis_request(self, request_id: str) -> Optional[AnalysisRequest]:
        """Get analysis request by ID"""
        requests = self._load_json(self.analysis_requests_file)
        request_dict = next((r for r in requests if r['id'] == request_id), None)

        if request_dict:
            return AnalysisRequest(**request_dict)
        return None

    def update_analysis_request_status(self, request_id: str, status: str,
                                     error_message: str = None, processing_time: float = None):
        """Update analysis request status"""
        requests = self._load_json(self.analysis_requests_file)

        for request in requests:
            if request['id'] == request_id:
                request['status'] = status
                request['error_message'] = error_message
                request['processing_time_seconds'] = processing_time
                request = self._add_timestamps(request, update=True)
                break

        self._save_json(self.analysis_requests_file, requests)

    # LLM analysis operations
    def save_llm_analysis(self, analysis: LLMAnalysis) -> str:
        """Save LLM analysis result"""
        analyses = self._load_json(self.llm_analyses_file)

        analysis_dict = asdict(analysis)
        analysis_dict['id'] = self._generate_id()
        analysis_dict['analysis_date'] = datetime.now().isoformat()
        analysis_dict = self._add_timestamps(analysis_dict)

        analyses.append(analysis_dict)
        self._save_json(self.llm_analyses_file, analyses)

        logger.info(f"Saved LLM analysis ID: {analysis_dict['id']} for request ID: {analysis.request_id}")
        return analysis_dict['id']

    def get_llm_analysis(self, analysis_id: str) -> Optional[LLMAnalysis]:
        """Get LLM analysis by ID"""
        analyses = self._load_json(self.llm_analyses_file)
        analysis_dict = next((a for a in analyses if a['id'] == analysis_id), None)

        if analysis_dict:
            return LLMAnalysis(**analysis_dict)
        return None

    def get_analyses_by_request(self, request_id: str) -> List[LLMAnalysis]:
        """Get all LLM analyses for a request"""
        analyses = self._load_json(self.llm_analyses_file)
        request_analyses = [a for a in analyses if a['request_id'] == request_id]

        return [LLMAnalysis(**a) for a in request_analyses]

    # Structured analysis data operations
    def save_structured_result(self, analysis_id: str, structured_data: Dict):
        """Save structured analysis result"""
        results = self._load_json(self.structured_results_file)

        result_record = {
            'id': self._generate_id(),
            'analysis_id': analysis_id,
            'structured_data': structured_data,
            'created_at': datetime.now().isoformat()
        }

        results.append(result_record)
        self._save_json(self.structured_results_file, results)

        logger.info(f"Saved structured result for analysis ID: {analysis_id}")

    def get_structured_result(self, analysis_id: str) -> Optional[Dict]:
        """Get structured result by analysis ID"""
        results = self._load_json(self.structured_results_file)
        result = next((r for r in results if r['analysis_id'] == analysis_id), None)

        if result:
            return result['structured_data']
        return None

    # Query operations for frontend
    def get_company_latest_analysis(self, ticker: str) -> Optional[Dict]:
        """Get latest analysis for a company"""
        company = self.get_company_by_ticker(ticker)
        if not company:
            return None

        # Get all requests for this company
        requests = self._load_json(self.analysis_requests_file)
        company_requests = [r for r in requests if r['company_id'] == company.id and r['status'] == 'completed']

        if not company_requests:
            return None

        # Get latest request
        latest_request = max(company_requests, key=lambda x: x['request_date'])

        # Get analyses for this request
        analyses = self.get_analyses_by_request(latest_request['id'])
        successful_analyses = [a for a in analyses if a.analysis_status == 'completed']

        if not successful_analyses:
            return None

        # Return summary
        return {
            'company': asdict(company),
            'request': latest_request,
            'analyses_count': len(successful_analyses),
            'total_cost': sum(a.cost_usd or 0 for a in successful_analyses),
            'latest_analysis_date': max(a.analysis_date for a in successful_analyses)
        }

    def get_analysis_summary(self, limit: int = 50) -> List[Dict]:
        """Get analysis summary for dashboard"""
        requests = self._load_json(self.analysis_requests_file)
        analyses = self._load_json(self.llm_analyses_file)
        companies = self._load_json(self.companies_file)

        # Create company lookup
        company_lookup = {c['id']: c for c in companies}

        # Get recent completed requests
        completed_requests = [r for r in requests if r['status'] == 'completed']
        completed_requests = sorted(completed_requests, key=lambda x: x['request_date'], reverse=True)[:limit]

        summaries = []
        for request in completed_requests:
            company = company_lookup.get(request['company_id'])
            if not company:
                continue

            # Get analyses for this request
            request_analyses = [a for a in analyses if a['request_id'] == request['id']]
            successful_analyses = [a for a in request_analyses if a['analysis_status'] == 'completed']

            if successful_analyses:
                summaries.append({
                    'request_id': request['id'],
                    'company_name': company['company_name'],
                    'ticker': company['ticker'],
                    'subsector': company['subsector'],
                    'request_date': request['request_date'],
                    'analyses_count': len(successful_analyses),
                    'total_cost': sum(a.get('cost_usd', 0) or 0 for a in successful_analyses),
                    'processing_time': request.get('processing_time_seconds', 0)
                })

        return summaries

    # Export/Import operations
    def export_to_dataframe(self, table: str) -> pd.DataFrame:
        """Export data to pandas DataFrame"""
        file_map = {
            'companies': self.companies_file,
            'analysis_requests': self.analysis_requests_file,
            'llm_analyses': self.llm_analyses_file,
            'structured_results': self.structured_results_file
        }

        if table not in file_map:
            raise ValueError(f"Unknown table: {table}")

        data = self._load_json(file_map[table])
        return pd.DataFrame(data)

    def import_from_dataframe(self, table: str, df: pd.DataFrame):
        """Import data from pandas DataFrame"""
        file_map = {
            'companies': self.companies_file,
            'analysis_requests': self.analysis_requests_file,
            'llm_analyses': self.llm_analyses_file,
            'structured_results': self.structured_results_file
        }

        if table not in file_map:
            raise ValueError(f"Unknown table: {table}")

        # Convert DataFrame to list of dicts
        data = df.to_dict('records')
        self._save_json(file_map[table], data)
        logger.info(f"Imported {len(data)} records to {table}")

    def create_sample_database(self):
        """Create sample database with example companies"""
        logger.info("Creating sample database with technology companies...")

        # Sample technology companies across different subsectors
        sample_companies = [
            # Semiconductors
            Company(
                company_name="NVIDIA Corporation",
                ticker="NVDA",
                subsector="Semiconductors",
                market_cap_usd=1200000000000,  # $1.2T
                employees=29600,
                founded_year=1993,
                headquarters="Santa Clara, CA",
                description="Designs graphics processing units for gaming and AI markets",
                website="https://www.nvidia.com"
            ),
            Company(
                company_name="Advanced Micro Devices Inc.",
                ticker="AMD",
                subsector="Semiconductors",
                market_cap_usd=220000000000,  # $220B
                employees=26000,
                founded_year=1969,
                headquarters="Santa Clara, CA",
                description="Designs and manufactures microprocessors and graphics processors",
                website="https://www.amd.com"
            ),
            Company(
                company_name="Marvell Technology Inc.",
                ticker="MRVL",
                subsector="Semiconductors",
                market_cap_usd=50000000000,  # $50B
                employees=7000,
                founded_year=1995,
                headquarters="Wilmington, DE",
                description="Semiconductor solutions for data infrastructure, 5G, and automotive",
                website="https://www.marvell.com"
            ),

            # Cloud/SaaS
            Company(
                company_name="Snowflake Inc.",
                ticker="SNOW",
                subsector="Cloud/SaaS",
                market_cap_usd=45000000000,  # $45B
                employees=6800,
                founded_year=2012,
                headquarters="Bozeman, MT",
                description="Cloud-based data platform company",
                website="https://www.snowflake.com"
            ),
            Company(
                company_name="Datadog Inc.",
                ticker="DDOG",
                subsector="Cloud/SaaS",
                market_cap_usd=35000000000,  # $35B
                employees=5000,
                founded_year=2010,
                headquarters="New York, NY",
                description="Monitoring and analytics platform for cloud applications",
                website="https://www.datadoghq.com"
            ),
            Company(
                company_name="MongoDB Inc.",
                ticker="MDB",
                subsector="Cloud/SaaS",
                market_cap_usd=18000000000,  # $18B
                employees=4500,
                founded_year=2007,
                headquarters="New York, NY",
                description="Document database platform for modern applications",
                website="https://www.mongodb.com"
            ),

            # Cybersecurity
            Company(
                company_name="CrowdStrike Holdings Inc.",
                ticker="CRWD",
                subsector="Cybersecurity",
                market_cap_usd=75000000000,  # $75B
                employees=8500,
                founded_year=2011,
                headquarters="Austin, TX",
                description="Cloud-delivered endpoint protection platform",
                website="https://www.crowdstrike.com"
            ),
            Company(
                company_name="Palo Alto Networks Inc.",
                ticker="PANW",
                subsector="Cybersecurity",
                market_cap_usd=110000000000,  # $110B
                employees=14000,
                founded_year=2005,
                headquarters="Santa Clara, CA",
                description="Enterprise cybersecurity platform and services",
                website="https://www.paloaltonetworks.com"
            ),
            Company(
                company_name="Zscaler Inc.",
                ticker="ZS",
                subsector="Cybersecurity",
                market_cap_usd=25000000000,  # $25B
                employees=6000,
                founded_year=2008,
                headquarters="San Jose, CA",
                description="Cloud-native security platform",
                website="https://www.zscaler.com"
            ),

            # Consumer/Devices
            Company(
                company_name="Apple Inc.",
                ticker="AAPL",
                subsector="Consumer/Devices",
                market_cap_usd=3400000000000,  # $3.4T
                employees=164000,
                founded_year=1976,
                headquarters="Cupertino, CA",
                description="Consumer electronics and digital services",
                website="https://www.apple.com"
            ),
            Company(
                company_name="Tesla Inc.",
                ticker="TSLA",
                subsector="Consumer/Devices",
                market_cap_usd=800000000000,  # $800B
                employees=140000,
                founded_year=2003,
                headquarters="Austin, TX",
                description="Electric vehicles and energy storage systems",
                website="https://www.tesla.com"
            ),

            # Infrastructure
            Company(
                company_name="Palantir Technologies Inc.",
                ticker="PLTR",
                subsector="Infrastructure",
                market_cap_usd=20000000000,  # $20B
                employees=4000,
                founded_year=2003,
                headquarters="Denver, CO",
                description="Big data analytics platform for government and commercial clients",
                website="https://www.palantir.com"
            ),
            Company(
                company_name="Cloudflare Inc.",
                ticker="NET",
                subsector="Infrastructure",
                market_cap_usd=30000000000,  # $30B
                employees=3500,
                founded_year=2009,
                headquarters="San Francisco, CA",
                description="Web infrastructure and website security services",
                website="https://www.cloudflare.com"
            ),

            # Electronic Components
            Company(
                company_name="Analog Devices Inc.",
                ticker="ADI",
                subsector="Electronic Components",
                market_cap_usd=90000000000,  # $90B
                employees=25000,
                founded_year=1965,
                headquarters="Wilmington, MA",
                description="Analog, mixed-signal, and digital signal processing technologies",
                website="https://www.analog.com"
            ),
            Company(
                company_name="Microchip Technology Inc.",
                ticker="MCHP",
                subsector="Electronic Components",
                market_cap_usd=45000000000,  # $45B
                employees=20000,
                founded_year=1989,
                headquarters="Chandler, AZ",
                description="Microcontroller, mixed-signal, analog and Flash-IP solutions",
                website="https://www.microchip.com"
            )
        ]

        # Add sample companies
        added_count = 0
        for company in sample_companies:
            try:
                existing = self.get_company_by_ticker(company.ticker)
                if not existing:
                    self.add_company(company)
                    added_count += 1
                    logger.info(f"Added sample company: {company.company_name} ({company.ticker})")
                else:
                    logger.info(f"Company {company.ticker} already exists, skipping")
            except Exception as e:
                logger.error(f"Error adding company {company.ticker}: {e}")

        logger.info(f"Sample database creation complete. Added {added_count} new companies")
        return added_count

def create_sample_database() -> JSONDataManager:
    """Create sample database with example companies (standalone function)"""
    db = JSONDataManager()
    db.create_sample_database()
    return db

if __name__ == "__main__":
    # Test database setup
    db = create_sample_database()
    companies = db.list_companies()
    print(f"Created database with {len(companies)} companies:")
    for company in companies:
        print(f"  {company.ticker}: {company.company_name}")