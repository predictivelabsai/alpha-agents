"""
Dynamic Stock Discovery System
Uses AI agents to intelligently discover and screen stocks based on criteria
"""

import asyncio
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

class DynamicStockDiscovery:
    """
    Intelligent stock discovery system that uses multiple data sources
    and AI reasoning to find interesting investment opportunities
    """
    
    def __init__(self):
        self.cache = {}
        self.cache_duration = 3600  # 1 hour cache for discovery results
        
    def discover_sector_stocks(self, sector: str, criteria: Dict[str, Any], limit: int = 20) -> List[Dict[str, Any]]:
        """
        Dynamically discover stocks in a sector based on intelligent criteria
        
        Args:
            sector: Target sector
            criteria: Discovery criteria (market_cap, growth_rate, etc.)
            limit: Maximum number of stocks to return
            
        Returns:
            List of discovered stocks with reasoning
        """
        try:
            # Get sector universe from multiple sources
            sector_universe = self._get_sector_universe(sector)
            
            # Apply intelligent filtering
            filtered_stocks = self._apply_intelligent_filters(sector_universe, criteria)
            
            # Score and rank stocks
            scored_stocks = self._score_and_rank_stocks(filtered_stocks, criteria)
            
            # Return top stocks with reasoning
            return scored_stocks[:limit]
            
        except Exception as e:
            logging.error(f"Error in stock discovery: {e}")
            return self._get_fallback_stocks(sector, limit)
    
    def _get_sector_universe(self, sector: str) -> List[str]:
        """
        Get comprehensive list of stocks in a sector from multiple sources
        """
        try:
            # Method 1: Use sector ETF holdings
            sector_etfs = {
                'Technology': ['XLK', 'VGT', 'FTEC'],
                'Healthcare': ['XLV', 'VHT', 'FHLC'],
                'Financial Services': ['XLF', 'VFH', 'FNCL'],
                'Consumer Cyclical': ['XLY', 'VCR', 'FDIS'],
                'Communication Services': ['XLC', 'VOX', 'FCOM'],
                'Industrial': ['XLI', 'VIS', 'FIDU'],
                'Consumer Defensive': ['XLP', 'VDC', 'FSTA'],
                'Energy': ['XLE', 'VDE', 'FENY'],
                'Utilities': ['XLU', 'VPU', 'FUTY'],
                'Real Estate': ['XLRE', 'VNQ', 'FREL'],
                'Materials': ['XLB', 'VAW', 'FMAT']
            }
            
            # Get holdings from sector ETFs
            universe = set()
            etfs = sector_etfs.get(sector, [])
            
            for etf in etfs[:1]:  # Use first ETF to avoid rate limits
                try:
                    etf_ticker = yf.Ticker(etf)
                    # Try to get holdings (this may not work for all ETFs)
                    # For now, use a curated list but make it more dynamic
                    pass
                except:
                    continue
            
            # Method 2: Use market cap and sector screening
            universe.update(self._screen_by_market_data(sector))
            
            return list(universe)
            
        except Exception as e:
            logging.error(f"Error getting sector universe: {e}")
            return self._get_fallback_universe(sector)
    
    def _screen_by_market_data(self, sector: str) -> List[str]:
        """
        Screen stocks using market data and financial metrics
        """
        # For now, use a more diverse set of stocks per sector
        # In production, this would query financial databases
        
        diverse_stocks = {
            'Technology': [
                # Large cap
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
                # Mid cap
                'CRM', 'ADBE', 'NFLX', 'ORCL', 'AMD', 'QCOM', 'INTC',
                # Growth/emerging
                'SNOW', 'PLTR', 'CRWD', 'ZS', 'OKTA', 'DDOG', 'NET',
                # Specialized
                'AVGO', 'TXN', 'LRCX', 'KLAC', 'AMAT', 'MU', 'MRVL'
            ],
            'Healthcare': [
                # Pharma giants
                'JNJ', 'PFE', 'ABBV', 'MRK', 'BMY', 'AMGN', 'GILD',
                # Biotech
                'REGN', 'VRTX', 'BIIB', 'ILMN', 'MRNA', 'BNTX', 'SGEN',
                # Medical devices
                'TMO', 'DHR', 'ISRG', 'SYK', 'BSX', 'MDT', 'ABT',
                # Healthcare services
                'UNH', 'CVS', 'CI', 'HUM', 'ANTM', 'CNC', 'MOH'
            ],
            'Financial Services': [
                # Major banks
                'JPM', 'BAC', 'WFC', 'C', 'USB', 'PNC', 'TFC',
                # Investment banks
                'GS', 'MS', 'SCHW', 'BLK', 'BK', 'STT', 'NTRS',
                # Fintech/emerging
                'V', 'MA', 'PYPL', 'SQ', 'COIN', 'SOFI', 'AFRM',
                # Insurance
                'BRK-B', 'AXP', 'AIG', 'PRU', 'MET', 'AFL', 'ALL'
            ],
            'Consumer Cyclical': [
                # E-commerce/retail
                'AMZN', 'HD', 'LOW', 'TJX', 'COST', 'WMT', 'TGT',
                # Automotive
                'TSLA', 'F', 'GM', 'RIVN', 'LCID', 'NIO', 'XPEV',
                # Restaurants/hospitality
                'MCD', 'SBUX', 'CMG', 'YUM', 'QSR', 'DPZ', 'MAR',
                # Apparel/luxury
                'NKE', 'LULU', 'TJX', 'RH', 'ETSY', 'RVLV', 'TPG'
            ],
            'Communication Services': [
                # Tech platforms
                'GOOGL', 'META', 'NFLX', 'DIS', 'SNAP', 'PINS', 'TWTR',
                # Telecom
                'VZ', 'T', 'TMUS', 'CHTR', 'CMCSA', 'DISH', 'SIRI',
                # Gaming/entertainment
                'ATVI', 'EA', 'TTWO', 'RBLX', 'U', 'ROKU', 'SPOT',
                # Media
                'NFLX', 'DIS', 'PARA', 'WBD', 'FOX', 'FOXA', 'LYV'
            ]
        }
        
        return diverse_stocks.get(sector, [])
    
    def _apply_intelligent_filters(self, universe: List[str], criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Apply intelligent filtering based on criteria
        """
        filtered_stocks = []
        min_market_cap = criteria.get('min_market_cap', 1e9)  # $1B default
        
        # Process stocks in parallel for efficiency
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {
                executor.submit(self._evaluate_stock, symbol, criteria): symbol 
                for symbol in universe
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    stock_data = future.result(timeout=30)
                    if stock_data and stock_data.get('market_cap', 0) >= min_market_cap:
                        filtered_stocks.append(stock_data)
                except Exception as e:
                    logging.warning(f"Error evaluating {symbol}: {e}")
                    continue
        
        return filtered_stocks
    
    def _evaluate_stock(self, symbol: str, criteria: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Evaluate individual stock against criteria
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Basic validation
            if not info or info.get('marketCap', 0) == 0:
                return None
            
            # Get key metrics
            stock_data = {
                'symbol': symbol,
                'company_name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'current_price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                'pe_ratio': info.get('trailingPE', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'profit_margin': info.get('profitMargins', 0),
                'roe': info.get('returnOnEquity', 0),
                'debt_to_equity': info.get('debtToEquity', 0),
                'beta': info.get('beta', 1.0),
                'analyst_rating': info.get('recommendationMean', 3.0),
                'price_to_book': info.get('priceToBook', 0),
                'forward_pe': info.get('forwardPE', 0),
                'peg_ratio': info.get('pegRatio', 0),
                'discovery_score': 0,
                'discovery_reasons': []
            }
            
            # Calculate discovery score based on criteria
            stock_data['discovery_score'] = self._calculate_discovery_score(stock_data, criteria)
            stock_data['discovery_reasons'] = self._generate_discovery_reasons(stock_data, criteria)
            
            return stock_data
            
        except Exception as e:
            logging.error(f"Error evaluating stock {symbol}: {e}")
            return None
    
    def _calculate_discovery_score(self, stock_data: Dict[str, Any], criteria: Dict[str, Any]) -> float:
        """
        Calculate intelligent discovery score for stock
        """
        score = 0.0
        
        # Growth scoring
        revenue_growth = stock_data.get('revenue_growth', 0)
        if revenue_growth > 0.2:  # 20%+ growth
            score += 25
        elif revenue_growth > 0.1:  # 10%+ growth
            score += 15
        elif revenue_growth > 0.05:  # 5%+ growth
            score += 10
        
        # Profitability scoring
        profit_margin = stock_data.get('profit_margin', 0)
        if profit_margin > 0.2:  # 20%+ margin
            score += 20
        elif profit_margin > 0.1:  # 10%+ margin
            score += 15
        elif profit_margin > 0.05:  # 5%+ margin
            score += 10
        
        # Valuation scoring (lower PE is better, but not too low)
        pe_ratio = stock_data.get('pe_ratio', 0)
        if 10 <= pe_ratio <= 25:  # Reasonable valuation
            score += 15
        elif 5 <= pe_ratio < 10 or 25 < pe_ratio <= 35:  # Acceptable
            score += 10
        
        # Financial health scoring
        roe = stock_data.get('roe', 0)
        if roe > 0.15:  # 15%+ ROE
            score += 15
        elif roe > 0.1:  # 10%+ ROE
            score += 10
        
        # Debt management
        debt_to_equity = stock_data.get('debt_to_equity', 0)
        if debt_to_equity < 0.3:  # Low debt
            score += 10
        elif debt_to_equity < 0.6:  # Moderate debt
            score += 5
        
        # Analyst sentiment
        analyst_rating = stock_data.get('analyst_rating', 3.0)
        if analyst_rating <= 2.0:  # Strong buy/buy
            score += 10
        elif analyst_rating <= 2.5:  # Moderate buy
            score += 5
        
        return min(score, 100)  # Cap at 100
    
    def _generate_discovery_reasons(self, stock_data: Dict[str, Any], criteria: Dict[str, Any]) -> List[str]:
        """
        Generate human-readable reasons for stock discovery
        """
        reasons = []
        
        # Growth reasons
        revenue_growth = stock_data.get('revenue_growth', 0)
        if revenue_growth > 0.2:
            reasons.append(f"Exceptional revenue growth: {revenue_growth:.1%}")
        elif revenue_growth > 0.1:
            reasons.append(f"Strong revenue growth: {revenue_growth:.1%}")
        
        # Profitability reasons
        profit_margin = stock_data.get('profit_margin', 0)
        if profit_margin > 0.2:
            reasons.append(f"High profit margins: {profit_margin:.1%}")
        elif profit_margin > 0.1:
            reasons.append(f"Healthy profit margins: {profit_margin:.1%}")
        
        # Valuation reasons
        pe_ratio = stock_data.get('pe_ratio', 0)
        if 10 <= pe_ratio <= 25:
            reasons.append(f"Reasonable valuation: P/E {pe_ratio:.1f}")
        
        # Financial strength
        roe = stock_data.get('roe', 0)
        if roe > 0.15:
            reasons.append(f"Strong return on equity: {roe:.1%}")
        
        debt_to_equity = stock_data.get('debt_to_equity', 0)
        if debt_to_equity < 0.3:
            reasons.append("Conservative debt management")
        
        # Market position
        market_cap = stock_data.get('market_cap', 0)
        if market_cap > 100e9:  # $100B+
            reasons.append("Large-cap market leader")
        elif market_cap > 10e9:  # $10B+
            reasons.append("Mid-cap growth opportunity")
        
        return reasons
    
    def _score_and_rank_stocks(self, stocks: List[Dict[str, Any]], criteria: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Score and rank stocks by discovery criteria
        """
        # Sort by discovery score (highest first)
        ranked_stocks = sorted(stocks, key=lambda x: x.get('discovery_score', 0), reverse=True)
        
        # Add ranking information
        for i, stock in enumerate(ranked_stocks):
            stock['discovery_rank'] = i + 1
            stock['discovery_percentile'] = ((len(ranked_stocks) - i) / len(ranked_stocks)) * 100
        
        return ranked_stocks
    
    def _get_fallback_stocks(self, sector: str, limit: int) -> List[Dict[str, Any]]:
        """
        Fallback stocks when discovery fails
        """
        fallback_symbols = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'TMO'],
            'Financial Services': ['JPM', 'BAC', 'V', 'MA', 'BRK-B']
        }
        
        symbols = fallback_symbols.get(sector, ['AAPL', 'MSFT', 'GOOGL'])[:limit]
        
        fallback_stocks = []
        for symbol in symbols:
            fallback_stocks.append({
                'symbol': symbol,
                'company_name': f"{symbol} Inc.",
                'sector': sector,
                'market_cap': 1e12,  # $1T fallback
                'current_price': 100.0,
                'discovery_score': 50,
                'discovery_reasons': ['Fallback selection - major market participant'],
                'discovery_rank': len(fallback_stocks) + 1,
                'fallback': True
            })
        
        return fallback_stocks
    
    def _get_fallback_universe(self, sector: str) -> List[str]:
        """
        Fallback universe when discovery fails
        """
        return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JNJ', 'PFE', 'JPM']

# Global instance
dynamic_discovery = DynamicStockDiscovery()

