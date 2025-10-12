"""
Optimized Stock Screener - Fast Version
Uses parallel processing and caching to reduce execution time
"""

import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from pathlib import Path
import pickle
from datetime import datetime, timedelta
import os

class FastStockScreener:
    """Optimized stock screener with parallel processing and caching"""
    
    def __init__(self, max_workers=10, cache_hours=24):
        self.max_workers = max_workers
        self.cache_hours = cache_hours
        self.cache_file = Path("utils/cache/screener_cache.pkl")
        self.cache_file.parent.mkdir(exist_ok=True)
    
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
    
    def fetch_ticker_data(self, ticker):
        """Fetch data for a single ticker"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Get financial data
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            cashflow = stock.cashflow
            
            # Calculate metrics
            market_cap = info.get('marketCap', 0)
            if market_cap == 0:
                return None
            
            # Revenue and income data
            try:
                revenue = financials.loc['Total Revenue'].iloc[0] if 'Total Revenue' in financials.index else 0
                net_income = financials.loc['Net Income'].iloc[0] if 'Net Income' in financials.index else 0
                gross_profit = financials.loc['Gross Profit'].iloc[0] if 'Gross Profit' in financials.index else 0
                operating_income = financials.loc['Operating Income'].iloc[0] if 'Operating Income' in financials.index else 0
            except:
                revenue = net_income = gross_profit = operating_income = 0
            
            # Calculate ratios (TTM - same as original StockScreener)
            gross_profit_margin = (gross_profit / revenue) if revenue > 0 else 0
            operating_profit_margin = (operating_income / revenue) if revenue > 0 else 0
            net_profit_margin = (net_income / revenue) if revenue > 0 else 0
            
            # Additional margins (same as original)
            try:
                ebit = financials.loc['EBIT'].iloc[0] if 'EBIT' in financials.index else 0
                ebitda = financials.loc['EBITDA'].iloc[0] if 'EBITDA' in financials.index else 0
            except:
                ebit = ebitda = 0
            
            ebit_margin = (ebit / revenue) if revenue > 0 else 0
            ebitda_margin = (ebitda / revenue) if revenue > 0 else 0
            
            # ROE, ROA calculations (TTM average - same as original)
            try:
                total_equity = balance_sheet.loc['Stockholders Equity'].iloc[0] if 'Stockholders Equity' in balance_sheet.index else 1
                total_assets = balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else 1
            except:
                total_equity = total_assets = 1
            
            roe = (net_income / total_equity) if total_equity > 0 else 0
            roa = (net_income / total_assets) if total_assets > 0 else 0
            
            # Additional ratios (same as original)
            try:
                current_assets = balance_sheet.loc['Current Assets'].iloc[0] if 'Current Assets' in balance_sheet.index else 0
                current_liabilities = balance_sheet.loc['Current Liabilities'].iloc[0] if 'Current Liabilities' in balance_sheet.index else 1
                total_debt = balance_sheet.loc['Total Debt'].iloc[0] if 'Total Debt' in balance_sheet.index else 0
            except:
                current_assets = current_liabilities = total_debt = 0
            
            current_ratio = (current_assets / current_liabilities) if current_liabilities > 0 else 0
            debt_to_ebitda_ratio = (total_debt / ebitda) if ebitda > 0 else 0
            
            # Simple scoring
            score = (
                gross_profit_margin * 0.3 +
                operating_profit_margin * 0.3 +
                net_profit_margin * 0.2 +
                roe * 0.1 +
                roa * 0.1
            ) * 100
            
            return {
                'ticker': ticker,
                'company_name': info.get('longName', ticker),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': market_cap,
                'gross_profit_margin_ttm': gross_profit_margin,
                'operating_profit_margin_ttm': operating_profit_margin,
                'net_profit_margin_ttm': net_profit_margin,
                'ebit_margin_ttm': ebit_margin,
                'ebitda_margin_ttm': ebitda_margin,
                'roe_ttm': roe,
                'roa_ttm': roa,
                'current_ratio': current_ratio,
                'debt_to_ebitda_ratio': debt_to_ebitda_ratio,
                'score': score
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
            
            # Process completed tasks
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        print(f"✅ {ticker}: {result['company_name']} (Score: {result['score']:.1f})")
                except Exception as e:
                    print(f"❌ {ticker}: {e}")
        
        end_time = time.time()
        print(f"⏱️ Data fetching completed in {end_time - start_time:.1f} seconds")
        
        return results
    
    def screen(self, region="US", min_cap=0, max_cap=float('inf'), 
               sectors=None, industries=None, max_companies=100):
        """Run optimized screening"""
        
        print(f"🔍 Starting optimized screening...")
        print(f"📊 Parameters: region={region}, min_cap={min_cap}, max_cap={max_cap}")
        print(f"📊 Sectors: {sectors}, Industries: {industries}")
        print(f"📊 Max companies: {max_companies}")
        
        # Try to use cached data first
        cached_data = self.get_cached_data()
        
        if cached_data is not None:
            print(f"📋 Using cached data: {len(cached_data)} companies")
            df = pd.DataFrame(cached_data)
        else:
            print("🔄 Cache not available, fetching fresh data...")
            
            # Get ticker list (simplified for speed)
            tickers = self.get_sample_tickers(max_companies)
            
            # Fetch data in parallel
            results = self.fetch_data_parallel(tickers)
            
            if not results:
                print("❌ No data fetched")
                return pd.DataFrame()
            
            df = pd.DataFrame(results)
            
            # Cache the data
            self.save_cache(results)
        
        # Apply filters
        print(f"🔍 Applying filters to {len(df)} companies...")
        
        # Market cap filter (convert to billions for easier comparison)
        if min_cap > 0:
            df = df[df['market_cap'] >= min_cap]
        if max_cap < float('inf'):
            df = df[df['market_cap'] <= max_cap]
        
        print(f"📊 After market cap filter: {len(df)} companies")
        
        # Sector filter
        if sectors:
            if isinstance(sectors, str):
                sectors = [sectors]
            df = df[df['sector'].isin(sectors)]
            print(f"📊 After sector filter: {len(df)} companies")
        
        # Industry filter
        if industries:
            if isinstance(industries, str):
                industries = [industries]
            df = df[df['industry'].isin(industries)]
            print(f"📊 After industry filter: {len(df)} companies")
        
        # Sort by score
        df = df.sort_values('score', ascending=False)
        
        print(f"✅ Screening completed: {len(df)} companies found")
        return df
    
    def get_sample_tickers(self, max_companies=100):
        """Get sample tickers for testing (faster than full universe)"""
        # Popular tech stocks for quick testing
        tech_tickers = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'AMD', 'INTC', 'CRM', 'ORCL', 'ADBE', 'PYPL', 'UBER', 'LYFT',
            'SQ', 'ROKU', 'ZM', 'DOCU', 'SNOW', 'PLTR', 'CRWD', 'OKTA',
            'NET', 'DDOG', 'MDB', 'TWLO', 'SPOT', 'SHOP', 'SQ', 'ROKU'
        ]
        
        # Add some random tickers for variety
        random_tickers = [
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'V', 'MA',
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR',
            'KO', 'PEP', 'WMT', 'PG', 'JNJ', 'UNH', 'HD', 'MA'
        ]
        
        all_tickers = tech_tickers + random_tickers
        return all_tickers[:max_companies]

# Usage example
def run_fast_screen():
    """Run fast screening example"""
    screener = FastStockScreener(max_workers=15, cache_hours=6)
    
    # Run screening
    df = screener.screen(
        region="US",
        min_cap=1000000000,  # 1B market cap
        max_cap=100000000000,  # 100B market cap
        sectors=["Technology"],
        max_companies=50
    )
    
    print(f"\n📊 Results: {len(df)} companies")
    if not df.empty:
        print(df[['ticker', 'company_name', 'sector', 'score']].head(10))
    
    return df

if __name__ == "__main__":
    run_fast_screen()
