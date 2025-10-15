"""
Parallel Stock Screener - Full Feature Version
Uses parallel processing and caching while maintaining all original calculations and features
"""

import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from pathlib import Path
import pickle
from datetime import datetime, timedelta
import os
from difflib import get_close_matches
from collections.abc import Collection
from tqdm import tqdm

class ParallelStockScreener:
    """Parallel stock screener with full feature set and caching"""
    
    # Same sectors and industries as original StockScreener
    SECTORS_INDUSTRIES = {
        "Technology": [
            "Information Technology Services",
            "Software - Application",
            "Software - Infrastructure",
            "Communication Equipment",
            "Computer Hardware",
            "Consumer Electronics",
            "Electronic Components",
            "Electronics & Computer Distribution",
            "Scientific & Technical Instruments",
            "Semiconductor Equipment & Materials",
            "Semiconductors",
            "Solar",
        ],
        "Basic Materials": [
            "Agricultural Inputs",
            "Building Materials",
            "Chemicals",
            "Specialty Chemicals",
            "Lumber & Wood Production",
            "Paper & Paper Products",
            "Aluminum",
            "Copper",
            "Other Industrials Metals & Mining",
            "Gold",
            "Silver",
            "Other Precious Metals & Mining",
            "Coking Coal",
            "Steel",
        ],
        "Consumer Cycical": [
            "Auto & Truck Dealerships",
            "Auto Manufacturers",
            "Auto Parts",
            "Recreational Vehicles",
            "Furnishings, Fixtures & Appliances",
            "Residential Construction",
            "Textile Manufacturing",
            "Apparel Manufacturing",
            "Footwear & Accessories",
            "Packaging & Containers",
            "Personal Services",
            "Restaurants",
            "Apparel Retail",
            "Department Stores",
            "Home Improvement Retail",
            "Luxury Goods",
            "Internet Retail",
            "Specialty Retail",
            "Gambling",
            "Leisure",
            "Lodging",
            "Resorts & Casinos",
            "Travel Services",
        ],
        "Financial Services": [
            "Asset Management",
            "Banks - Diversified",
            "Banks - Regional",
            "Mortgage Finance",
            "Capital Markets",
            "Financial Data & Stock Exchanges",
            "Insurance - Life",
            "Insurance - Property & Casualty",
            "Insurance - Reinsurance",
            "Insurance - Specialty",
            "Insurance Brokers",
            "Insurance - Diversified",
            "Shell Companies",
            "Financial Conglomerates",
            "Credit Services",
        ],
        "Real Estate": [
            "Real Estate - Development",
            "Real Estate Services",
            "Real Estate - Diversified",
            "REIT - Healthcare Facilities",
            "REIT - Hotel & Motel",
            "REIT - Industrial",
            "REIT - Office",
            "REIT - Residential",
            "REIT - Retail",
            "REIT - Mortgage",
            "REIT - Specialty",
            "REIT - Diversified",
        ],
        "Consumer Defensive": [
            "Beverages - Brewers",
            "Beverages - Wineries & Distilleries",
            "Beverages - Non-Alcoholic",
            "Confectioners",
            "Farm Products",
            "Household & Personal Products",
            "Packaged Foods",
            "Education & Training Services",
            "Discount Stores",
            "Food Distribution",
            "Grocery Stores",
            "Tobacco",
        ],
        "Healthcare": [
            "Biotechnology",
            "Drug Manufacturers - General",
            "Drug Manufacturers - Specialty & Generic",
            "Healthcare Plans",
            "Medical Care Facilities",
            "Pharmaceutical Retailers",
            "Health Information Services",
            "Medical Services",
            "Medical Instruments & Supplies",
            "Diagnostics & Research",
            "Medical Distribution",
        ],
        "Utilities": [
            "Utilities - Independent Power Producers",
            "Utilities - Renewable",
            "Utilities - Regulated Water",
            "Utilities - Regulated Electronic",
            "Utilities - Regulated Gas",
            "Utilities - Diversified",
        ],
        "Communication Services": [
            "Telecom Services",
            "Advertising Agencies",
            "Publishing",
            "Broadcasting",
            "Entertainment",
            "Internet Content & Information",
            "Electroning Gaming and Multimedia",
        ],
        "Energy": [
            "Oil & Gas Drilling",
            "Oil & Gas E&P",
            "Oil & Gas Integrated",
            "Oil & Gas Midstream",
            "Oil & Gas Refining & Marketing",
            "Oil & Gas Equipment & Services",
            "Thermal Coal",
            "Uranium",
        ],
        "Industrials": [
            "Aerospace & Defense",
            "Specialty Business Services",
            "Consulting Services",
            "Rental & Leasing Services",
            "Security & Protection Services",
            "Staffing & Employment Services",
            "Conglomerates",
            "Engineering & Construction",
            "Infrastructure Operations",
            "Building Products & Equipment",
            "Farm & Heavy Construction Machinery",
            "Industrial Distribution",
            "Business Equipment & Supplies",
            "Specialty Industrial Machinery",
            "Metal Fabrication",
            "Pollution & Treatment Controls",
            "Tools & Accessories",
            "Electrical Equipment & Parts",
            "Airports & Air Services",
            "Airlines",
            "Railroads",
            "Marine Shipping",
            "Trucking",
            "Integrated Freight & Logistics",
            "Waste Management",
        ],
    }
    
    SECTORS = list(SECTORS_INDUSTRIES.keys())
    INDUSTRIES = [industry for industries in SECTORS_INDUSTRIES.values() for industry in industries]
    
    def __init__(self, max_workers=10, cache_hours=24):
        self.max_workers = max_workers
        self.cache_hours = cache_hours
        self.cache_file = Path("utils/cache/parallel_screener_cache.pkl")
        self.cache_file.parent.mkdir(exist_ok=True)
        
        # Initialize data containers
        self.data = pd.DataFrame()
        self.yearly_calc_data = pd.DataFrame()
        self.quarterly_calc_data = pd.DataFrame()
        
        # Set region and industries (will be set in screen method)
        self.region = "US"
        self.industries = None
    
    def get_cached_data(self):
        """Load cached data if available and not expired"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                cache_time = cache_data.get('timestamp')
                if cache_time and datetime.now() - cache_time < timedelta(hours=self.cache_hours):
                    print(f"✅ Using cached data from {cache_time}")
                    return cache_data.get('data')
            except Exception as e:
                print(f"⚠️ Cache load failed: {e}")
        
        return None
    
    def save_cache(self, data):
        """Save data to cache"""
        try:
            cache_data = {
                'data': data,
                'timestamp': datetime.now()
            }
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print("💾 Data cached successfully")
        except Exception as e:
            print(f"⚠️ Cache save failed: {e}")
    
    def _get_stock_universe(self, path="utils/universe") -> pd.DataFrame:
        """Get stock universe from CSV file - same logic as original screener"""
        base = Path(path)
        base.mkdir(exist_ok=True)
        csv_path = base / f"{self.region.lower()}.csv"
        
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Stock universe file for region '{self.region}' not found at {csv_path}."
            )
        
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        # Filter by industries if specified (same as original screener)
        if self.industries:
            out = df[df["industry"].isin(self.industries)].copy().set_index("ticker")
        else:
            out = df.set_index("ticker")
        return out
    
    def _extract_section_data(self, stock: yf.Ticker, section: str, subset: list[str] | None = None, period: str = "Y") -> pd.Series:
        """Extract financial data section - same as original"""
        section = section.lower()
        if period == "Q":
            section = "quarterly_" + section
        data: pd.DataFrame = getattr(stock, section).copy()
        if subset is not None:
            data = data.reindex(subset)
        # keep only the most recent 5 periods and rename columns
        if period == "Y":
            data = data.iloc[:, :4]
            column_names = [f"0{period}"] + [f"-{i}{period}" for i in range(1, 4)]
        else:
            data = data.iloc[:, :5]
            column_names = [f"0{period}"] + [f"-{i}{period}" for i in range(1, 5)]
        data.columns = column_names[: data.shape[1]]
        data = data.reindex(columns=column_names)
        data_stacked: pd.Series = data.stack(future_stack=True)
        data_stacked.name = stock.ticker
        return data_stacked
    
    def fetch_ticker_data(self, ticker):
        """Fetch comprehensive data for a single ticker - same calculations as original"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            market_cap = info.get("marketCap", -float("inf"))
            if market_cap <= 0:
                return None
            
            company_name = info.get("longName") or info.get("shortName")
            
            # Extract all financial data sections - same as original
            financials = self._extract_section_data(
                stock,
                section="financials",
                subset=[
                    "Total Revenue",
                    "Net Income",
                    "Operating Income",
                    "Gross Profit",
                    "Cost Of Revenue",
                    "EBIT",
                    "EBITDA",
                    "Tax Rate For Calcs",
                    "Interest Expense",
                ],
            )
            quarterly_financials = self._extract_section_data(
                stock,
                section="financials",
                subset=[
                    "Total Revenue",
                    "Net Income",
                    "Operating Income",
                    "Gross Profit",
                    "Cost Of Revenue",
                    "EBIT",
                    "EBITDA",
                    "Normalized Income",
                    "Tax Rate For Calcs",
                    "Interest Expense",
                ],
                period="Q",
            )
            cash_flow = self._extract_section_data(
                stock,
                section="cash_flow",
                subset=["Operating Cash Flow", "Repayment of Debt"],
            )
            quarterly_cash_flow = self._extract_section_data(
                stock,
                section="cash_flow",
                subset=["Operating Cash Flow", "Repayment Of Debt"],
                period="Q",
            )
            balance_sheet = self._extract_section_data(
                stock,
                section="balance_sheet",
                subset=[
                    "Stockholders Equity",
                    "Invested Capital",
                    "Accounts Receivable",
                    "Accounts Payable",
                    "Inventory",
                    "Current Assets",
                    "Current Liabilities",
                    "Total Debt",
                    "Total Assets",
                ],
            )
            quarterly_balance_sheet = self._extract_section_data(
                stock,
                section="balance_sheet",
                subset=[
                    "Stockholders Equity",
                    "Invested Capital",
                    "Accounts Receivable",
                    "Accounts Payable",
                    "Inventory",
                    "Current Assets",
                    "Current Liabilities",
                    "Total Debt",
                    "Total Assets",
                ],
                period="Q",
            )
            
            return {
                'ticker': ticker,
                'company_name': company_name,
                'market_cap': market_cap,
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'financials': financials,
                'quarterly_financials': quarterly_financials,
                'cash_flow': cash_flow,
                'quarterly_cash_flow': quarterly_cash_flow,
                'balance_sheet': balance_sheet,
                'quarterly_balance_sheet': quarterly_balance_sheet,
            }
            
        except Exception as e:
            print(f"❌ Error fetching {ticker}: {e}")
            return None
    
    def fetch_data_parallel(self, tickers):
        """Fetch data for multiple tickers in parallel"""
        print(f"🚀 Fetching data for {len(tickers)} tickers using {self.max_workers} workers...")
        
        results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_ticker = {
                executor.submit(self.fetch_ticker_data, ticker): ticker 
                for ticker in tickers
            }
            
            # Process completed tasks with progress bar
            for future in tqdm(as_completed(future_to_ticker), total=len(tickers), desc="Fetching data"):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    print(f"❌ {ticker}: {e}")
        
        end_time = time.time()
        print(f"⏱️ Data fetching completed in {end_time - start_time:.1f} seconds")
        
        return results
    
    def _compute_ttm_ratio(self, dividend: str, divisor: str) -> pd.Series:
        """Compute TTM ratio - same as original"""
        dividend_sum, divisor_sum = (
            self.quarterly_calc_data[dividend].iloc[:, :4].sum(axis=1),
            self.quarterly_calc_data[divisor].iloc[:, :4].sum(axis=1),
        )
        return pd.to_numeric(dividend_sum) / pd.to_numeric(divisor_sum)
    
    def _compute_ttm_average_ratio(self, dividend: str, divisor: str) -> pd.Series:
        """Compute TTM average ratio - same as original"""
        dividend_sum = self.quarterly_calc_data[dividend].iloc[:, :4].sum(axis=1)
        divisor_df = self.quarterly_calc_data[divisor]
        divisor_avg = (
            divisor_df.iloc[:, [0, 4]].sum(axis=1, skipna=False)
            + (divisor_df.iloc[:, 1:4] * 2).sum(axis=1, skipna=False)
        ) / 8
        return dividend_sum / divisor_avg
    
    def _compute_4y_growth(self, metric: str) -> pd.Series:
        """Compute 4-year growth - same as original"""
        return (
            self.yearly_calc_data[metric].iloc[:, -1]
            / self.yearly_calc_data[metric].iloc[:, 0]
        ) ** (1 / 5) - 1
    
    def _compute_consistency_ratio(self, metric: str) -> pd.Series:
        """Compute consistency ratio - same as original"""
        pct_change = self.yearly_calc_data[metric].iloc[:, ::-1].pct_change(axis=1)
        return pct_change.mean(axis=1) / pct_change.std(axis=1)
    
    def _compute_metrics(self):
        """Compute all metrics - same as original StockScreener"""
        print("📊 Computing financial metrics...")
        
        # TTM Margins
        self.data["gross_profit_margin_ttm"] = self._compute_ttm_ratio("Gross Profit", "Total Revenue")
        self.data["operating_profit_margin_ttm"] = self._compute_ttm_ratio("Operating Income", "Total Revenue")
        self.data["net_profit_margin_ttm"] = self._compute_ttm_ratio("Net Income", "Total Revenue")
        self.data["ebit_margin_ttm"] = self._compute_ttm_ratio("EBIT", "Total Revenue")
        self.data["ebitda_margin_ttm"] = self._compute_ttm_ratio("EBITDA", "Total Revenue")
        
        # ROA and ROE
        self.data["roa_ttm"] = self._compute_ttm_average_ratio("Net Income", "Total Assets")
        self.data["roe_ttm"] = self._compute_ttm_average_ratio("Net Income", "Stockholders Equity")
        
        # NOPAT and ROIC
        yearly_nopat = (1 - self.yearly_calc_data["Tax Rate For Calcs"]) * self.yearly_calc_data["EBIT"]
        self.yearly_calc_data = pd.concat([self.yearly_calc_data, pd.concat({"NOPAT": yearly_nopat}, axis=1)], axis=1)
        
        quarterly_nopat = (1 - self.quarterly_calc_data["Tax Rate For Calcs"]) * self.quarterly_calc_data["EBIT"]
        self.quarterly_calc_data = pd.concat([self.quarterly_calc_data, pd.concat({"NOPAT": quarterly_nopat}, axis=1)], axis=1)
        
        self.data["roic_ttm"] = self._compute_ttm_average_ratio("NOPAT", "Invested Capital")
        
        # Growth metrics
        self.data["total_revenue_4y_cagr"] = self._compute_4y_growth("Total Revenue")
        self.data["net_income_4y_cagr"] = self._compute_4y_growth("Net Income")
        self.data["operating_cash_flow_4y_cagr"] = self._compute_4y_growth("Operating Cash Flow")
        
        # Consistency metrics
        self.data["total_revenue_4y_consistency"] = self._compute_consistency_ratio("Total Revenue")
        self.data["net_income_4y_consistency"] = self._compute_consistency_ratio("Net Income")
        self.data["operating_cash_flow_4y_consistency"] = self._compute_consistency_ratio("Operating Cash Flow")
        
        # Liquidity and leverage ratios
        self.data["current_ratio"] = (
            self.quarterly_calc_data["Current Assets"]["0Q"]
            / self.quarterly_calc_data["Current Liabilities"]["0Q"]
        )
        self.data["debt_to_ebitda_ratio"] = (
            self.quarterly_calc_data["Total Debt"]["0Q"]
            / self.quarterly_calc_data["EBITDA"]["0Q"]
        )
        self.data["debt_servicing_ratio"] = (
            self.quarterly_calc_data["Interest Expense"]["0Q"]
            + self.quarterly_calc_data["Repayment Of Debt"]["0Q"].abs()
        ) / self.quarterly_calc_data["EBITDA"]["0Q"]
    
    def score(self, profitability_weight: float = 1, growth_weight: float = 1, debt_weight: float = 1):
        """Score using percentile ranking - same as original"""
        print("🎯 Computing scores using percentile ranking...")
        score = self.data.rank(pct=True).mean(axis=1)
        self.data.loc[:, "score"] = score * 100
    
    def screen(self, region="US", sectors=None, industries=None, min_cap=0, max_cap=float('inf'), max_companies=500):
        """Run parallel screening with full feature set"""
        
        # Set region and industries for universe loading
        self.region = region
        self.industries = industries
        
        print(f"🔍 Starting parallel screening...")
        print(f"📊 Parameters: region={region}, min_cap={min_cap}, max_cap={max_cap}")
        print(f"📊 Sectors: {sectors}, Industries: {industries}")
        print(f"📊 Max companies: {max_companies}")
        
        # Try to use cached data first
        cached_data = self.get_cached_data()
        
        if cached_data is not None:
            print(f"📋 Using cached data: {len(cached_data)} companies")
            # Reconstruct data structures from cache
            self._reconstruct_from_cache(cached_data)
        else:
            print("🔄 Cache not available, fetching fresh data...")
            
            # Get universe
            universe_df = self._get_stock_universe()
            
            # Apply sector/industry filters
            if sectors or industries:
                if sectors:
                    if isinstance(sectors, str):
                        sectors = [sectors]
                    sector_industries = {ind for s in sectors for ind in self.SECTORS_INDUSTRIES[s]}
                    universe_df = universe_df[universe_df['industry'].isin(sector_industries)]
                
                if industries:
                    if isinstance(industries, str):
                        industries = [industries]
                    universe_df = universe_df[universe_df['industry'].isin(industries)]
            
            # Limit to max_companies
            tickers = universe_df.index[:max_companies].tolist()
            
            # Fetch data in parallel
            results = self.fetch_data_parallel(tickers)
            
            if not results:
                print("❌ No data fetched")
                return pd.DataFrame()
            
            # Process results
            self._process_results(results)
            
            # Cache the processed data - ensure ticker is included
            cache_data = {
                'data': self.data.reset_index().to_dict('records'),  # Include ticker in data
                'yearly_calc_data': self.yearly_calc_data.to_dict(),
                'quarterly_calc_data': self.quarterly_calc_data.to_dict()
            }
            self.save_cache(cache_data)
        
        # Apply market cap filter
        print(f"🔍 Applying market cap filter...")
        self.data = self.data[self.data["market_cap"].between(min_cap, max_cap)]
        
        # Sort by score
        self.data = self.data.sort_values(by="score", ascending=False)
        
        # Ensure ticker is the index (same format as original screener)
        if 'ticker' in self.data.columns:
            self.data = self.data.set_index('ticker')
        
        print(f"✅ Parallel screening completed: {len(self.data)} companies found")
        return self.data
    
    def _process_results(self, results):
        """Process fetched results into data structures"""
        print("📊 Processing financial data...")
        
        # Initialize data structures
        self.data = pd.DataFrame()
        self.yearly_calc_data = pd.DataFrame()
        self.quarterly_calc_data = pd.DataFrame()
        
        for result in results:
            ticker = result['ticker']
            
            # Add basic info to main data - ensure ticker is included as column initially
            self.data.loc[ticker, 'ticker'] = ticker
            self.data.loc[ticker, 'company_name'] = result['company_name']
            self.data.loc[ticker, 'market_cap'] = result['market_cap']
            self.data.loc[ticker, 'sector'] = result['sector']
            self.data.loc[ticker, 'industry'] = result['industry']
            
            # Add financial data
            yearly_combined = pd.concat([
                result['financials'], 
                result['cash_flow'], 
                result['balance_sheet']
            ])
            quarterly_combined = pd.concat([
                result['quarterly_financials'], 
                result['quarterly_cash_flow'], 
                result['quarterly_balance_sheet']
            ])
            
            self.yearly_calc_data.loc[yearly_combined.name, yearly_combined.index] = yearly_combined
            self.quarterly_calc_data.loc[quarterly_combined.name, quarterly_combined.index] = quarterly_combined
        
        # Compute metrics and score
        self._compute_metrics()
        self.score()
    
    def _reconstruct_from_cache(self, cached_data):
        """Reconstruct data structures from cached data"""
        # Reconstruct main data - handle ticker column properly
        if 'data' in cached_data:
            self.data = pd.DataFrame(cached_data['data'])
            # Set ticker as index if it exists, otherwise use default index
            if 'ticker' in self.data.columns:
                self.data = self.data.set_index('ticker')
            elif 'index' in self.data.columns:
                self.data = self.data.set_index('index')
        
        # Reconstruct calculation data
        if 'yearly_calc_data' in cached_data:
            self.yearly_calc_data = pd.DataFrame(cached_data['yearly_calc_data'])
        if 'quarterly_calc_data' in cached_data:
            self.quarterly_calc_data = pd.DataFrame(cached_data['quarterly_calc_data'])

# Usage example
def run_parallel_screen():
    """Run parallel screening example"""
    screener = ParallelStockScreener(max_workers=15, cache_hours=6)
    
    # Run screening with same parameters as original
    df = screener.screen(
        region="US",
        sectors=["Technology"],
        min_cap=1000000000,  # 1B market cap
        max_cap=100000000000,  # 100B market cap
        max_companies=100
    )
    
    print(f"\n📊 Results: {len(df)} companies")
    if not df.empty:
        print(df[['company_name', 'sector', 'score']].head(10))
    
    return df

if __name__ == "__main__":
    run_parallel_screen()