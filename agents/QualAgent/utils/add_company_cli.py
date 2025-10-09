#!/usr/bin/env python3
"""
Command-Line Interface for Adding Companies to QualAgent
Simple CLI tool to add companies with proper argument parsing
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.simple_add_company import add_company_simple, show_all_companies

def main():
    parser = argparse.ArgumentParser(
        description='Add companies to QualAgent database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add company with ticker and name only
  python utils/add_company_cli.py --ticker SHOP --name "Shopify Inc."

  # Add company with market cap
  python utils/add_company_cli.py --ticker CRM --name "Salesforce Inc." --market-cap 220000000000

  # Add company with custom subsector
  python utils/add_company_cli.py --ticker PLTR --name "Palantir Technologies" --subsector "Analytics/AI"

  # Show all companies
  python utils/add_company_cli.py --show-all
        """
    )

    # Add arguments
    parser.add_argument('--ticker', required=False, help='Stock ticker (e.g., AAPL)')
    parser.add_argument('--name', required=False, help='Full company name (e.g., "Apple Inc.")')
    parser.add_argument('--market-cap', type=float, help='Market cap in USD (e.g., 3000000000000)')
    parser.add_argument('--subsector', help='Technology subsector (auto-guessed if not provided)')
    parser.add_argument('--show-all', action='store_true', help='Show all companies in database')

    args = parser.parse_args()

    # Show all companies if requested
    if args.show_all:
        show_all_companies()
        return

    # Validate required arguments for adding
    if not args.ticker or not args.name:
        parser.error("Both --ticker and --name are required for adding a company")

    # Add the company
    success = add_company_simple(
        ticker=args.ticker,
        company_name=args.name,
        market_cap=args.market_cap,
        subsector=args.subsector
    )

    if success:
        print(f"\n✅ Successfully added {args.name} ({args.ticker.upper()})")
        print("\nTo run analysis on this company:")
        print(f"python run_enhanced_analysis.py --user-id analyst1 --company {args.ticker.upper()}")
    else:
        print(f"\n❌ Failed to add {args.name} ({args.ticker.upper()})")
        sys.exit(1)

if __name__ == "__main__":
    main()