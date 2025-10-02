#!/usr/bin/env python3
"""
Simple Company Addition for QualAgent
Quick Python functions to add companies with minimal info
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.json_data_manager import JSONDataManager, Company

def add_company_simple(ticker: str, company_name: str, market_cap: float = None, subsector: str = None):
    """
    Add a company with minimal information

    Args:
        ticker: Stock ticker (e.g., "AAPL")
        company_name: Full company name (e.g., "Apple Inc.")
        market_cap: Market cap in USD (optional)
        subsector: Technology subsector (optional, will auto-guess)

    Returns:
        bool: True if added successfully, False if already exists or error
    """
    db = JSONDataManager()

    # Check if company already exists
    existing = db.get_company_by_ticker(ticker.upper())
    if existing:
        print(f"Company {ticker.upper()} already exists: {existing.company_name}")
        return False

    # Auto-guess subsector if not provided
    if not subsector:
        subsector = guess_subsector(company_name)

    # Create company
    company = Company(
        company_name=company_name,
        ticker=ticker.upper(),
        subsector=subsector,
        market_cap_usd=market_cap,
        description=f"{company_name} - Technology company"
    )

    try:
        company_id = db.add_company(company)
        print(f"+ Added: {company_name} ({ticker.upper()}) - {subsector}")
        if market_cap:
            print(f"  Market Cap: ${market_cap:,.0f}")
        return True
    except Exception as e:
        print(f"X Error adding {ticker}: {e}")
        return False

def guess_subsector(company_name: str) -> str:
    """Guess subsector based on company name"""
    name_lower = company_name.lower()

    # Subsector keyword mapping
    if any(word in name_lower for word in ['cloud', 'software', 'saas', 'data']):
        return 'Cloud/SaaS'
    elif any(word in name_lower for word in ['cyber', 'security']):
        return 'Cybersecurity'
    elif any(word in name_lower for word in ['semiconductor', 'chip', 'micro', 'intel', 'nvidia', 'amd']):
        return 'Semiconductors'
    elif any(word in name_lower for word in ['auto', 'car', 'electric', 'tesla', 'device', 'phone', 'apple']):
        return 'Consumer/Devices'
    elif any(word in name_lower for word in ['network', 'infrastructure', 'cloud']):
        return 'Infrastructure'
    elif any(word in name_lower for word in ['component', 'analog', 'sensor']):
        return 'Electronic Components'
    else:
        return 'Technology'

def add_multiple_companies(companies_list):
    """
    Add multiple companies from a list

    Args:
        companies_list: List of tuples (ticker, name) or (ticker, name, market_cap)

    Example:
        add_multiple_companies([
            ("PLTR", "Palantir Technologies"),
            ("SNOW", "Snowflake Inc.", 45000000000),
            ("DDOG", "Datadog Inc.")
        ])
    """
    added_count = 0

    for item in companies_list:
        if len(item) == 2:
            ticker, name = item
            market_cap = None
        elif len(item) == 3:
            ticker, name, market_cap = item
        else:
            print(f"Invalid format for {item}, skipping")
            continue

        if add_company_simple(ticker, name, market_cap):
            added_count += 1

    print(f"\n✓ Successfully added {added_count} companies")

def show_all_companies():
    """Display all companies in the database"""
    db = JSONDataManager()
    companies = db.list_companies()

    print(f"\n=== All Companies ({len(companies)}) ===")
    print(f"{'Ticker':<8} | {'Company Name':<40} | {'Subsector':<20} | Market Cap")
    print("-" * 80)

    for company in companies:
        market_cap_str = f"${company.market_cap_usd/1e9:.1f}B" if company.market_cap_usd else "N/A"
        print(f"{company.ticker:<8} | {company.company_name[:40]:<40} | {company.subsector:<20} | {market_cap_str}")

# Example usage functions
def example_add_single():
    """Example: Add a single company"""
    add_company_simple(
        ticker="CRM",
        company_name="Salesforce Inc.",
        market_cap=220000000000,
        subsector="Cloud/SaaS"
    )

def example_add_multiple():
    """Example: Add multiple companies at once"""
    companies_to_add = [
        ("UBER", "Uber Technologies Inc.", 120000000000),
        ("LYFT", "Lyft Inc.", 15000000000),
        ("RBLX", "Roblox Corporation"),  # No market cap
        ("UNITY", "Unity Software Inc.")
    ]

    add_multiple_companies(companies_to_add)

def example_quick_add():
    """Example: Quick add with minimal info"""
    # Just ticker and name - everything else auto-guessed
    add_company_simple("SHOP", "Shopify Inc.")
    add_company_simple("SQ", "Block Inc.")
    add_company_simple("PYPL", "PayPal Holdings Inc.")

if __name__ == "__main__":
    # Show usage examples
    print("QualAgent Simple Company Addition")
    print("=================================")
    print()
    print("Quick Examples:")
    print('add_company_simple("SHOP", "Shopify Inc.")')
    print('add_company_simple("CRM", "Salesforce Inc.", 220000000000)')
    print()
    print("To see all companies:")
    print('show_all_companies()')
    print()
    print("Current companies in database:")
    show_all_companies()