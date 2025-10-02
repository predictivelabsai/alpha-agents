#!/usr/bin/env python3
"""
Easy Company Addition Tool for QualAgent

This script provides multiple ways to add new companies to the QualAgent database
with minimal required information (just ticker and company name).

Usage Examples:
    # Single company - minimal info
    python utils/add_company.py --ticker MSFT --name "Microsoft Corporation"

    # Single company - with market cap
    python utils/add_company.py --ticker AAPL --name "Apple Inc." --market-cap 3400000000000

    # Single company - interactive mode
    python utils/add_company.py --interactive

    # Batch from CSV file
    python utils/add_company.py --from-csv companies_to_add.csv

    # Quick add multiple companies
    python utils/add_company.py --quick-add "TSLA,Tesla Inc." "NVDA,NVIDIA Corporation"
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional, Dict, Any

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.json_data_manager import JSONDataManager, Company

class CompanyAdder:
    """Helper class for adding companies with minimal information"""

    def __init__(self):
        self.db = JSONDataManager()

        # Common subsector mappings (you can customize these)
        self.subsector_keywords = {
            'cloud': 'Cloud/SaaS',
            'saas': 'Cloud/SaaS',
            'software': 'Cloud/SaaS',
            'cyber': 'Cybersecurity',
            'security': 'Cybersecurity',
            'chip': 'Semiconductors',
            'semiconductor': 'Semiconductors',
            'gpu': 'Semiconductors',
            'cpu': 'Semiconductors',
            'auto': 'Consumer/Devices',
            'car': 'Consumer/Devices',
            'electric': 'Consumer/Devices',
            'phone': 'Consumer/Devices',
            'device': 'Consumer/Devices',
            'infrastructure': 'Infrastructure',
            'network': 'Infrastructure',
            'component': 'Electronic Components',
            'analog': 'Electronic Components'
        }

    def guess_subsector(self, company_name: str, description: str = "") -> str:
        """Guess subsector based on company name and description"""
        text = (company_name + " " + description).lower()

        for keyword, subsector in self.subsector_keywords.items():
            if keyword in text:
                return subsector

        # Default fallback
        return "Technology"

    def add_company_minimal(self, ticker: str, company_name: str,
                           market_cap: Optional[float] = None,
                           subsector: Optional[str] = None,
                           description: Optional[str] = None) -> bool:
        """Add company with minimal information"""

        # Check if company already exists
        existing = self.db.get_company_by_ticker(ticker.upper())
        if existing:
            print(f"Company {ticker.upper()} already exists: {existing.company_name}")
            return False

        # Guess subsector if not provided
        if not subsector:
            subsector = self.guess_subsector(company_name, description or "")

        # Create company with available information
        company = Company(
            company_name=company_name,
            ticker=ticker.upper(),
            subsector=subsector,
            market_cap_usd=market_cap,
            description=description or f"{company_name} - Technology company"
        )

        try:
            company_id = self.db.add_company(company)
            print(f"+ Added: {company_name} ({ticker.upper()}) - {subsector}")
            if market_cap:
                print(f"  Market Cap: ${market_cap:,.0f}")
            return True
        except Exception as e:
            print(f"X Error adding {ticker}: {e}")
            return False

    def interactive_add(self):
        """Interactive mode for adding companies"""
        print("=== Interactive Company Addition ===")
        print("Enter company information (press Enter to skip optional fields)")
        print()

        while True:
            # Required fields
            ticker = input("Ticker symbol (required): ").strip().upper()
            if not ticker:
                print("Ticker is required!")
                continue

            company_name = input("Company name (required): ").strip()
            if not company_name:
                print("Company name is required!")
                continue

            # Check if exists
            existing = self.db.get_company_by_ticker(ticker)
            if existing:
                print(f"Company {ticker} already exists: {existing.company_name}")
                continue

            # Optional fields
            print("\nOptional fields (press Enter to skip):")

            market_cap_str = input("Market cap in USD (e.g., 1000000000): ").strip()
            market_cap = None
            if market_cap_str:
                try:
                    market_cap = float(market_cap_str)
                except ValueError:
                    print("Invalid market cap, skipping")

            subsector = input("Subsector (or press Enter to auto-guess): ").strip()
            description = input("Description: ").strip()

            # Add the company
            success = self.add_company_minimal(
                ticker=ticker,
                company_name=company_name,
                market_cap=market_cap,
                subsector=subsector if subsector else None,
                description=description if description else None
            )

            if success:
                print("Company added successfully!")

            # Continue?
            continue_adding = input("\nAdd another company? (y/n): ").strip().lower()
            if continue_adding != 'y':
                break

        print("\nInteractive addition completed!")

    def add_from_csv(self, csv_file: str):
        """Add companies from CSV file"""
        try:
            with open(csv_file, 'r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                added_count = 0

                for row in reader:
                    ticker = row.get('ticker', '').strip().upper()
                    company_name = row.get('company_name', '').strip()

                    if not ticker or not company_name:
                        print(f"Skipping row with missing ticker or name: {row}")
                        continue

                    market_cap = None
                    market_cap_str = row.get('market_cap', '').strip()
                    if market_cap_str:
                        try:
                            market_cap = float(market_cap_str)
                        except ValueError:
                            pass

                    subsector = row.get('subsector', '').strip() or None
                    description = row.get('description', '').strip() or None

                    if self.add_company_minimal(ticker, company_name, market_cap, subsector, description):
                        added_count += 1

                print(f"\nAdded {added_count} companies from {csv_file}")

        except FileNotFoundError:
            print(f"File not found: {csv_file}")
        except Exception as e:
            print(f"Error reading CSV file: {e}")

    def create_sample_csv(self, filename: str = "companies_template.csv"):
        """Create a sample CSV template for bulk import"""
        with open(filename, 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['ticker', 'company_name', 'market_cap', 'subsector', 'description'])
            writer.writerow(['EXAMPLE', 'Example Corp', '1000000000', 'Cloud/SaaS', 'Example technology company'])
            writer.writerow(['TEST', 'Test Inc', '', '', 'Another example (optional fields empty)'])

        print(f"Created sample CSV template: {filename}")
        print("Edit this file with your companies and run:")
        print(f"python utils/add_company.py --from-csv {filename}")

    def quick_add_multiple(self, company_strings: list):
        """Quick add from "TICKER,Company Name" format"""
        added_count = 0

        for company_str in company_strings:
            try:
                parts = company_str.split(',', 1)
                if len(parts) != 2:
                    print(f"Invalid format: {company_str} (expected: TICKER,Company Name)")
                    continue

                ticker = parts[0].strip().upper()
                company_name = parts[1].strip()

                if self.add_company_minimal(ticker, company_name):
                    added_count += 1

            except Exception as e:
                print(f"Error processing {company_str}: {e}")

        print(f"\nQuick-added {added_count} companies")

def main():
    parser = argparse.ArgumentParser(description='Add companies to QualAgent database')

    # Single company mode
    parser.add_argument('--ticker', type=str, help='Company ticker symbol')
    parser.add_argument('--name', type=str, help='Company name')
    parser.add_argument('--market-cap', type=float, help='Market capitalization in USD')
    parser.add_argument('--subsector', type=str, help='Technology subsector')
    parser.add_argument('--description', type=str, help='Company description')

    # Interactive mode
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode for adding companies')

    # Batch modes
    parser.add_argument('--from-csv', type=str, metavar='FILE',
                       help='Add companies from CSV file')
    parser.add_argument('--create-csv-template', type=str, nargs='?',
                       const='companies_template.csv', metavar='FILE',
                       help='Create a CSV template for bulk import')
    parser.add_argument('--quick-add', nargs='+', metavar='TICKER,NAME',
                       help='Quick add companies in "TICKER,Company Name" format')

    # Utility
    parser.add_argument('--list', action='store_true', help='List current companies')

    args = parser.parse_args()

    if not any(vars(args).values()):
        parser.print_help()
        return

    adder = CompanyAdder()

    # List companies
    if args.list:
        companies = adder.db.list_companies()
        print(f"\n=== Current Companies ({len(companies)}) ===")
        for company in companies:
            market_cap_str = f"${company.market_cap_usd/1e9:.1f}B" if company.market_cap_usd else "N/A"
            print(f"{company.ticker:<6} | {company.company_name:<40} | {company.subsector:<20} | {market_cap_str}")
        return

    # Create CSV template
    if args.create_csv_template:
        adder.create_sample_csv(args.create_csv_template)
        return

    # Single company mode
    if args.ticker and args.name:
        adder.add_company_minimal(
            ticker=args.ticker,
            company_name=args.name,
            market_cap=args.market_cap,
            subsector=args.subsector,
            description=args.description
        )
        return

    # Interactive mode
    if args.interactive:
        adder.interactive_add()
        return

    # CSV import
    if args.from_csv:
        adder.add_from_csv(args.from_csv)
        return

    # Quick add
    if args.quick_add:
        adder.quick_add_multiple(args.quick_add)
        return

    # If ticker provided without name, or vice versa
    if args.ticker or args.name:
        print("Error: Both --ticker and --name are required for single company mode")
        return

if __name__ == "__main__":
    main()