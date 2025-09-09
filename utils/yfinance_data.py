"""
yfinance Data Utility Module
Provides standardized data fetching and processing for all agents
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json

class YFinanceDataProvider:
    """
    Centralized data provider using yfinance API
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 300  # 5 minutes cache
    
    def get_stock_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive stock data for a single symbol
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary containing all relevant stock data
        """
        try:
            # Check cache first
            cache_key = f"{symbol}_full_data"
            if self._is_cached(cache_key):
                return self.cache[cache_key]['data']
            
            # Fetch data from yfinance
            ticker = yf.Ticker(symbol)
            
            # Get basic info
            info = ticker.info
            
            # Get historical data
            hist_1y = ticker.history(period="1y")
            hist_5y = ticker.history(period="5y")
            
            # Get financial statements
            financials = ticker.financials
            balance_sheet = ticker.balance_sheet
            cash_flow = ticker.cashflow
            
            # Compile comprehensive data
            stock_data = {
                'symbol': symbol,
                'company_name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'current_price': info.get('currentPrice', 0),
                'basic_info': info,
                'price_history': {
                    '1y': hist_1y,
                    '5y': hist_5y
                },
                'financials': {
                    'income_statement': financials,
                    'balance_sheet': balance_sheet,
                    'cash_flow': cash_flow
                },
                'calculated_metrics': self._calculate_metrics(info, financials, balance_sheet, cash_flow, hist_1y),
                'timestamp': datetime.now()
            }
            
            # Cache the data
            self._cache_data(cache_key, stock_data)
            
            return stock_data
            
        except Exception as e:
            logging.error(f"Error fetching data for {symbol}: {e}")
            return self._get_fallback_data(symbol)
    
    def get_sector_stocks(self, sector: str, limit: int = 20) -> List[str]:
        """
        Get list of stocks for a given sector
        
        Args:
            sector: Sector name
            limit: Maximum number of stocks to return
            
        Returns:
            List of stock symbols
        """
        # Predefined sector mappings for demo purposes
        # In production, this would query a financial database
        sector_stocks = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX', 'ADBE', 'CRM', 'ORCL', 'IBM', 'INTC', 'AMD', 'QCOM'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'TMO', 'DHR', 'BMY', 'AMGN', 'GILD', 'BIIB', 'REGN', 'VRTX', 'ISRG', 'ZTS', 'CVS'],
            'Financial Services': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SCHW', 'USB', 'PNC', 'TFC', 'COF', 'BK', 'STT'],
            'Consumer Cyclical': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TJX', 'F', 'GM', 'MAR', 'HLT', 'YUM', 'CMG', 'ORLY'],
            'Communication Services': ['GOOGL', 'META', 'NFLX', 'DIS', 'CMCSA', 'VZ', 'T', 'CHTR', 'TMUS', 'ATVI', 'EA', 'TTWO', 'SNAP', 'PINS', 'TWTR'],
            'Industrial': ['BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'RTX', 'LMT', 'NOC', 'GD', 'FDX', 'UNP', 'CSX', 'NSC', 'DAL'],
            'Consumer Defensive': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'KMB', 'GIS', 'K', 'CPB', 'CAG', 'SJM', 'HSY', 'MKC', 'CHD'],
            'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'OXY', 'BKR', 'HAL', 'DVN', 'FANG', 'APA', 'HES'],
            'Utilities': ['NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'XEL', 'SRE', 'PEG', 'ED', 'ETR', 'WEC', 'ES', 'FE', 'AWK'],
            'Real Estate': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'WELL', 'SPG', 'O', 'SBAC', 'DLR', 'REG', 'BXP', 'ARE', 'VTR', 'ESS'],
            'Materials': ['LIN', 'APD', 'SHW', 'FCX', 'NEM', 'DOW', 'DD', 'PPG', 'ECL', 'IFF', 'ALB', 'CE', 'VMC', 'MLM', 'PKG']
        }
        
        stocks = sector_stocks.get(sector, [])
        return stocks[:limit]
    
    def get_batch_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get data for multiple stocks efficiently
        
        Args:
            symbols: List of stock symbols
            
        Returns:
            Dictionary mapping symbols to their data
        """
        batch_data = {}
        
        for symbol in symbols:
            try:
                batch_data[symbol] = self.get_stock_data(symbol)
            except Exception as e:
                logging.error(f"Error fetching batch data for {symbol}: {e}")
                batch_data[symbol] = self._get_fallback_data(symbol)
        
        return batch_data
    
    def _calculate_metrics(self, info: Dict, financials: pd.DataFrame, 
                          balance_sheet: pd.DataFrame, cash_flow: pd.DataFrame, 
                          price_history: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate derived financial metrics
        """
        metrics = {}
        
        try:
            # Basic valuation metrics
            metrics['pe_ratio'] = info.get('trailingPE', 0)
            metrics['pb_ratio'] = info.get('priceToBook', 0)
            metrics['ps_ratio'] = info.get('priceToSalesTrailing12Months', 0)
            metrics['peg_ratio'] = info.get('pegRatio', 0)
            
            # Profitability metrics
            metrics['roe'] = info.get('returnOnEquity', 0)
            metrics['roa'] = info.get('returnOnAssets', 0)
            metrics['profit_margin'] = info.get('profitMargins', 0)
            metrics['operating_margin'] = info.get('operatingMargins', 0)
            
            # Financial health metrics
            metrics['debt_to_equity'] = info.get('debtToEquity', 0)
            metrics['current_ratio'] = info.get('currentRatio', 0)
            metrics['quick_ratio'] = info.get('quickRatio', 0)
            
            # Growth metrics
            metrics['revenue_growth'] = info.get('revenueGrowth', 0)
            metrics['earnings_growth'] = info.get('earningsGrowth', 0)
            
            # Price performance
            if not price_history.empty:
                current_price = price_history['Close'].iloc[-1]
                price_1m_ago = price_history['Close'].iloc[-22] if len(price_history) >= 22 else current_price
                price_3m_ago = price_history['Close'].iloc[-66] if len(price_history) >= 66 else current_price
                price_1y_ago = price_history['Close'].iloc[0]
                
                metrics['price_performance'] = {
                    '1m': ((current_price - price_1m_ago) / price_1m_ago * 100) if price_1m_ago > 0 else 0,
                    '3m': ((current_price - price_3m_ago) / price_3m_ago * 100) if price_3m_ago > 0 else 0,
                    '1y': ((current_price - price_1y_ago) / price_1y_ago * 100) if price_1y_ago > 0 else 0
                }
                
                # Technical indicators
                metrics['rsi'] = self._calculate_rsi(price_history['Close'])
                metrics['moving_averages'] = {
                    'ma_20': price_history['Close'].rolling(20).mean().iloc[-1] if len(price_history) >= 20 else current_price,
                    'ma_50': price_history['Close'].rolling(50).mean().iloc[-1] if len(price_history) >= 50 else current_price,
                    'ma_200': price_history['Close'].rolling(200).mean().iloc[-1] if len(price_history) >= 200 else current_price
                }
            
        except Exception as e:
            logging.error(f"Error calculating metrics: {e}")
        
        return metrics
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not rsi.empty else 50.0
        except:
            return 50.0
    
    def _is_cached(self, cache_key: str) -> bool:
        """Check if data is cached and still valid"""
        if cache_key not in self.cache:
            return False
        
        cache_time = self.cache[cache_key]['timestamp']
        return (datetime.now() - cache_time).seconds < self.cache_duration
    
    def _cache_data(self, cache_key: str, data: Any):
        """Cache data with timestamp"""
        self.cache[cache_key] = {
            'data': data,
            'timestamp': datetime.now()
        }
    
    def _get_fallback_data(self, symbol: str) -> Dict[str, Any]:
        """Return fallback data when API fails"""
        return {
            'symbol': symbol,
            'company_name': f"{symbol} Inc.",
            'sector': 'Unknown',
            'industry': 'Unknown',
            'market_cap': 0,
            'current_price': 100.0,
            'basic_info': {},
            'price_history': {'1y': pd.DataFrame(), '5y': pd.DataFrame()},
            'financials': {
                'income_statement': pd.DataFrame(),
                'balance_sheet': pd.DataFrame(),
                'cash_flow': pd.DataFrame()
            },
            'calculated_metrics': {
                'pe_ratio': 15.0,
                'pb_ratio': 2.0,
                'ps_ratio': 3.0,
                'roe': 0.15,
                'roa': 0.08,
                'debt_to_equity': 0.3,
                'revenue_growth': 0.05,
                'price_performance': {'1m': 0, '3m': 0, '1y': 0}
            },
            'timestamp': datetime.now(),
            'fallback': True
        }

# Global instance
yfinance_provider = YFinanceDataProvider()

