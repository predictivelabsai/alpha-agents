"""
Enhanced yfinance utility for fundamental analysis
Supports US and China markets with comprehensive financial metrics
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")


class YFinanceUtil:
    """Enhanced yfinance utility for fundamental stock analysis"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Market mappings
        self.us_exchanges = [".", "-"]  # No suffix for US stocks
        self.china_exchanges = {
            "Shanghai": ".SS",
            "Shenzhen": ".SZ",
            "Hong Kong": ".HK",
        }

        # Sector classifications
        self.sectors = [
            "Technology",
            "Healthcare",
            "Financial Services",
            "Consumer Cyclical",
            "Communication Services",
            "Industrial",
            "Consumer Defensive",
            "Energy",
            "Utilities",
            "Real Estate",
            "Materials",
        ]

    def get_sector_universe(
        self, sector: str, market: str = "US", max_market_cap: float = None
    ) -> List[str]:
        """Get stock universe for a specific sector and market"""
        try:
            if market.upper() == "US":
                return self._get_us_sector_stocks(sector, max_market_cap)
            elif market.upper() == "CHINA":
                return self._get_china_sector_stocks(sector, max_market_cap)
            else:
                raise ValueError(f"Unsupported market: {market}")
        except Exception as e:
            self.logger.error(f"Error getting sector universe: {e}")
            return []

    def _get_us_sector_stocks(
        self, sector: str, max_market_cap: float = None
    ) -> List[str]:
        """Get US stocks for a specific sector"""
        # Sector-specific stock mappings (expanded)
        sector_stocks = {
            "Technology": [
                "AAPL",
                "MSFT",
                "GOOGL",
                "AMZN",
                "META",
                "NVDA",
                "ADBE",
                "CRM",
                "ORCL",
                "IBM",
                "INTC",
                "AMD",
                "QCOM",
                "CSCO",
                "AVGO",
                "NOW",
                "INTU",
                "TXN",
                "MU",
                "AMAT",
                "LRCX",
                "KLAC",
                "MCHP",
                "FTNT",
                "PANW",
                "CRWD",
                "ZS",
                "OKTA",
                "DDOG",
                "NET",
            ],
            "Healthcare": [
                "JNJ",
                "UNH",
                "PFE",
                "ABT",
                "TMO",
                "MDT",
                "DHR",
                "BMY",
                "ABBV",
                "MRK",
                "LLY",
                "GILD",
                "AMGN",
                "CVS",
                "CI",
                "ANTM",
                "HUM",
                "CNC",
                "MOH",
                "ELV",
                "ISRG",
                "SYK",
                "BSX",
                "ZBH",
                "BAX",
                "BDX",
                "EW",
                "HOLX",
                "DXCM",
                "VEEV",
            ],
            "Financial Services": [
                "JPM",
                "BAC",
                "WFC",
                "GS",
                "MS",
                "C",
                "AXP",
                "BLK",
                "SCHW",
                "USB",
                "PNC",
                "TFC",
                "COF",
                "CME",
                "ICE",
                "SPGI",
                "MCO",
                "AON",
                "MMC",
                "AJG",
                "CB",
                "TRV",
                "PGR",
                "ALL",
                "MET",
                "PRU",
                "AIG",
                "AFL",
                "HIG",
                "FIS",
            ],
            "Consumer Cyclical": [
                "AMZN",
                "TSLA",
                "HD",
                "NKE",
                "MCD",
                "SBUX",
                "LOW",
                "TJX",
                "F",
                "GM",
                "BKNG",
                "MAR",
                "RCL",
                "CCL",
                "NCLH",
                "LVS",
                "MGM",
                "WYNN",
                "CZR",
                "PENN",
                "EBAY",
                "ETSY",
                "W",
                "CHWY",
                "RVLV",
                "LULU",
                "ULTA",
                "TPG",
                "KMX",
                "AN",
            ],
            "Communication Services": [
                "GOOGL",
                "META",
                "DIS",
                "NFLX",
                "CMCSA",
                "VZ",
                "T",
                "TMUS",
                "CHTR",
                "DISH",
                "TWTR",
                "SNAP",
                "PINS",
                "ROKU",
                "SPOT",
                "ZM",
                "DOCU",
                "TEAM",
                "WORK",
                "PTON",
                "MTCH",
                "BMBL",
                "RBLX",
                "U",
                "FUBO",
                "SIRI",
                "LBRDK",
                "LBRDA",
                "FWONK",
                "BATRK",
            ],
            "Industrial": [
                "BA",
                "CAT",
                "GE",
                "MMM",
                "HON",
                "UPS",
                "RTX",
                "LMT",
                "NOC",
                "GD",
                "FDX",
                "UNP",
                "CSX",
                "NSC",
                "DAL",
                "UAL",
                "AAL",
                "LUV",
                "JBLU",
                "ALK",
                "DE",
                "EMR",
                "ETN",
                "PH",
                "ITW",
                "ROK",
                "DOV",
                "XYL",
                "FTV",
                "GNRC",
            ],
            "Consumer Defensive": [
                "WMT",
                "PG",
                "KO",
                "PEP",
                "COST",
                "WBA",
                "CVS",
                "TGT",
                "KR",
                "CL",
                "GIS",
                "K",
                "CPB",
                "SJM",
                "HSY",
                "MDLZ",
                "MNST",
                "KDP",
                "STZ",
                "BF-B",
                "PM",
                "MO",
                "BTI",
                "EL",
                "CLX",
                "CHD",
                "CAG",
                "HRL",
                "MKC",
                "LW",
            ],
            "Energy": [
                "XOM",
                "CVX",
                "COP",
                "EOG",
                "SLB",
                "PSX",
                "VLO",
                "MPC",
                "OXY",
                "BKR",
                "HAL",
                "DVN",
                "FANG",
                "APA",
                "EQT",
                "CTRA",
                "MRO",
                "HES",
                "KMI",
                "OKE",
                "WMB",
                "EPD",
                "ET",
                "MPLX",
                "PAA",
                "TRGP",
                "AM",
                "SUN",
                "DINO",
                "SM",
            ],
            "Utilities": [
                "NEE",
                "DUK",
                "SO",
                "D",
                "AEP",
                "EXC",
                "XEL",
                "SRE",
                "PEG",
                "ED",
                "ETR",
                "WEC",
                "ES",
                "FE",
                "AWK",
                "ATO",
                "CMS",
                "DTE",
                "NI",
                "LNT",
                "EVRG",
                "CNP",
                "AES",
                "PPL",
                "PNW",
                "IDA",
                "UGI",
                "NJR",
                "SWX",
                "OGE",
            ],
            "Real Estate": [
                "AMT",
                "PLD",
                "CCI",
                "EQIX",
                "PSA",
                "WELL",
                "DLR",
                "O",
                "SBAC",
                "EXR",
                "AVB",
                "EQR",
                "VTR",
                "ESS",
                "MAA",
                "UDR",
                "CPT",
                "FRT",
                "BXP",
                "VNO",
                "KIM",
                "REG",
                "SPG",
                "MAC",
                "SLG",
                "HST",
                "RHP",
                "PK",
                "APLE",
                "AIV",
            ],
            "Materials": [
                "LIN",
                "APD",
                "SHW",
                "FCX",
                "NEM",
                "DOW",
                "DD",
                "PPG",
                "ECL",
                "IFF",
                "ALB",
                "CE",
                "VMC",
                "MLM",
                "PKG",
                "IP",
                "WRK",
                "SON",
                "SEE",
                "AVY",
                "CF",
                "FMC",
                "LYB",
                "EMN",
                "RPM",
                "SUM",
                "KRA",
                "SLVM",
                "HUN",
                "OLN",
            ],
        }

        stocks = sector_stocks.get(sector, [])

        # Filter by market cap if specified
        if max_market_cap:
            filtered_stocks = []
            for ticker in stocks:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    market_cap = info.get("marketCap", 0)
                    if market_cap and market_cap <= max_market_cap:
                        filtered_stocks.append(ticker)
                except:
                    continue
            return filtered_stocks

        return stocks

    def _get_china_sector_stocks(
        self, sector: str, max_market_cap: float = None
    ) -> List[str]:
        """Get China stocks for a specific sector"""
        # Sample China stocks by sector (would need expansion)
        china_sector_stocks = {
            "Technology": [
                "BABA",
                "9988.HK",
                "JD",
                "9618.HK",
                "BIDU",
                "9888.HK",
                "TCEHY",
                "0700.HK",
                "PDD",
                "NTES",
                "9999.HK",
                "TME",
                "BILI",
                "IQ",
                "DOYU",
                "HUYA",
            ],
            "Healthcare": [
                "WuXi AppTec",
                "2359.HK",
                "BeiGene",
                "6160.HK",
                "Innovent Bio",
                "1801.HK",
            ],
            "Financial Services": [
                "ICBC",
                "1398.HK",
                "CCB",
                "0939.HK",
                "ABC",
                "1288.HK",
                "BOC",
                "3988.HK",
            ],
            "Consumer Cyclical": [
                "NIO",
                "XPEV",
                "LI",
                "BEKE",
                "EDU",
                "TAL",
                "YMM",
                "VIPS",
            ],
            "Energy": [
                "PetroChina",
                "0857.HK",
                "Sinopec",
                "0386.HK",
                "CNOOC",
                "0883.HK",
            ],
        }

        return china_sector_stocks.get(sector, [])

    def get_fundamental_metrics(self, ticker: str) -> Dict:
        """Get comprehensive fundamental metrics for a stock"""
        try:
            stock = yf.Ticker(ticker)

            # Get basic info
            info = stock.info

            # Get financial statements
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow

            # Get historical data
            hist = stock.history(period="5y")

            metrics = {
                "ticker": ticker,
                "company_name": info.get("longName", "N/A"),
                "sector": info.get("sector"),
                "industry": info.get("industry", "N/A"),
                "market_cap": info.get("marketCap", 0),
                "current_price": info.get("currentPrice", 0),
                # Growth metrics
                "revenue_growth_5y": self._calculate_growth_rate(
                    financials, "Total Revenue", 5
                ),
                "net_income_growth_5y": self._calculate_growth_rate(
                    financials, "Net Income", 5
                ),
                "cash_flow_growth_5y": self._calculate_growth_rate(
                    cash_flow, "Operating Cash Flow", 5
                ),
                # Profitability metrics
                "roe_ttm": info.get("returnOnEquity", 0) * 100
                if info.get("returnOnEquity")
                else 0,
                "roic_ttm": self._calculate_roic(financials, balance_sheet),
                "gross_margin": info.get("grossMargins", 0) * 100
                if info.get("grossMargins")
                else 0,
                "profit_margin": info.get("profitMargins", 0) * 100
                if info.get("profitMargins")
                else 0,
                # Debt metrics
                "current_ratio": self._calculate_current_ratio(balance_sheet),
                "debt_to_ebitda": self._calculate_debt_to_ebitda(
                    financials, balance_sheet
                ),
                "debt_service_ratio": self._calculate_debt_service_ratio(
                    financials, cash_flow
                ),
                # Additional metrics
                "pe_ratio": info.get("trailingPE", 0),
                "pb_ratio": info.get("priceToBook", 0),
                "dividend_yield": info.get("dividendYield", 0) * 100
                if info.get("dividendYield")
                else 0,
                # Quality scores
                "fundamental_score": 0,  # Will be calculated
                "growth_consistency": self._check_growth_consistency(
                    financials, cash_flow
                ),
                "profitability_consistency": self._check_profitability_consistency(
                    financials
                ),
                "debt_health": self._assess_debt_health(balance_sheet, financials),
            }

            # Calculate overall fundamental score
            metrics["fundamental_score"] = self._calculate_fundamental_score(metrics)

            return metrics

        except Exception as e:
            self.logger.error(f"Error getting fundamental metrics for {ticker}: {e}")
            return {"ticker": ticker, "error": str(e)}

    def _calculate_growth_rate(
        self, df: pd.DataFrame, metric: str, years: int
    ) -> float:
        """Calculate compound annual growth rate for a metric"""
        try:
            if df.empty or metric not in df.index:
                return 0

            values = df.loc[metric].dropna()
            if len(values) < 2:
                return 0

            # Get the most recent and oldest values
            recent_value = values.iloc[0]  # Most recent year
            old_value = values.iloc[-1]  # Oldest year

            if old_value <= 0:
                return 0

            # Calculate CAGR
            periods = min(len(values) - 1, years)
            if periods <= 0:
                return 0

            cagr = ((recent_value / old_value) ** (1 / periods) - 1) * 100
            return round(cagr, 2)

        except Exception:
            return 0

    def _calculate_roic(
        self, financials: pd.DataFrame, balance_sheet: pd.DataFrame
    ) -> float:
        """Calculate Return on Invested Capital"""
        try:
            if financials.empty or balance_sheet.empty:
                return 0

            # Get EBIT (Operating Income)
            ebit = (
                financials.loc["Operating Income"].iloc[0]
                if "Operating Income" in financials.index
                else 0
            )

            # Get invested capital (Total Assets - Current Liabilities)
            total_assets = (
                balance_sheet.loc["Total Assets"].iloc[0]
                if "Total Assets" in balance_sheet.index
                else 0
            )
            current_liabilities = (
                balance_sheet.loc["Current Liabilities"].iloc[0]
                if "Current Liabilities" in balance_sheet.index
                else 0
            )

            invested_capital = total_assets - current_liabilities

            if invested_capital <= 0:
                return 0

            roic = (ebit / invested_capital) * 100
            return round(roic, 2)

        except Exception:
            return 0

    def _calculate_current_ratio(self, balance_sheet: pd.DataFrame) -> float:
        """Calculate current ratio"""
        try:
            if balance_sheet.empty:
                return 0

            current_assets = (
                balance_sheet.loc["Current Assets"].iloc[0]
                if "Current Assets" in balance_sheet.index
                else 0
            )
            current_liabilities = (
                balance_sheet.loc["Current Liabilities"].iloc[0]
                if "Current Liabilities" in balance_sheet.index
                else 0
            )

            if current_liabilities <= 0:
                return 0

            return round(current_assets / current_liabilities, 2)

        except Exception:
            return 0

    def _calculate_debt_to_ebitda(
        self, financials: pd.DataFrame, balance_sheet: pd.DataFrame
    ) -> float:
        """Calculate debt to EBITDA ratio"""
        try:
            if financials.empty or balance_sheet.empty:
                return 0

            # Calculate EBITDA
            ebit = (
                financials.loc["Operating Income"].iloc[0]
                if "Operating Income" in financials.index
                else 0
            )
            depreciation = (
                financials.loc["Depreciation And Amortization"].iloc[0]
                if "Depreciation And Amortization" in financials.index
                else 0
            )
            ebitda = ebit + depreciation

            # Get total debt
            total_debt = (
                balance_sheet.loc["Total Debt"].iloc[0]
                if "Total Debt" in balance_sheet.index
                else 0
            )

            if ebitda <= 0:
                return float("inf")

            return round(total_debt / ebitda, 2)

        except Exception:
            return 0

    def _calculate_debt_service_ratio(
        self, financials: pd.DataFrame, cash_flow: pd.DataFrame
    ) -> float:
        """Calculate debt service coverage ratio"""
        try:
            if financials.empty or cash_flow.empty:
                return 0

            # Get operating cash flow
            operating_cf = (
                cash_flow.loc["Operating Cash Flow"].iloc[0]
                if "Operating Cash Flow" in cash_flow.index
                else 0
            )

            # Get interest expense
            interest_expense = (
                financials.loc["Interest Expense"].iloc[0]
                if "Interest Expense" in financials.index
                else 0
            )

            if interest_expense <= 0:
                return 100  # No debt service required

            debt_service_ratio = (operating_cf / abs(interest_expense)) * 100
            return round(debt_service_ratio, 2)

        except Exception:
            return 0

    def _check_growth_consistency(
        self, financials: pd.DataFrame, cash_flow: pd.DataFrame
    ) -> Dict:
        """Check consistency of growth metrics"""
        try:
            revenue_growth = self._calculate_growth_rate(financials, "Total Revenue", 5)
            income_growth = self._calculate_growth_rate(financials, "Net Income", 5)
            cf_growth = self._calculate_growth_rate(cash_flow, "Operating Cash Flow", 5)

            return {
                "revenue_consistent": revenue_growth > 0,
                "income_consistent": income_growth > 0,
                "cash_flow_consistent": cf_growth > 0,
                "overall_consistent": all(
                    [revenue_growth > 0, income_growth > 0, cf_growth > 0]
                ),
            }
        except Exception:
            return {"overall_consistent": False}

    def _check_profitability_consistency(self, financials: pd.DataFrame) -> Dict:
        """Check consistency of profitability metrics"""
        try:
            # Check if margins are improving or stable over time
            if financials.empty or "Total Revenue" not in financials.index:
                return {"overall_consistent": False}

            revenues = financials.loc["Total Revenue"].dropna()
            net_incomes = (
                financials.loc["Net Income"].dropna()
                if "Net Income" in financials.index
                else pd.Series()
            )

            if len(revenues) < 2 or len(net_incomes) < 2:
                return {"overall_consistent": False}

            # Calculate profit margins for each year
            margins = []
            for i in range(min(len(revenues), len(net_incomes))):
                if revenues.iloc[i] > 0:
                    margin = (net_incomes.iloc[i] / revenues.iloc[i]) * 100
                    margins.append(margin)

            if len(margins) < 2:
                return {"overall_consistent": False}

            # Check if margins are generally stable or improving
            margin_trend = np.polyfit(range(len(margins)), margins, 1)[0]  # Slope

            return {
                "margin_trend": margin_trend,
                "margins_stable": margin_trend >= -1,  # Allow small decline
                "overall_consistent": margin_trend >= -1
                and all(m > 0 for m in margins[-3:]),  # Recent margins positive
            }
        except Exception:
            return {"overall_consistent": False}

    def _assess_debt_health(
        self, balance_sheet: pd.DataFrame, financials: pd.DataFrame
    ) -> Dict:
        """Assess overall debt health"""
        try:
            current_ratio = self._calculate_current_ratio(balance_sheet)
            debt_to_ebitda = self._calculate_debt_to_ebitda(financials, balance_sheet)

            return {
                "current_ratio_healthy": current_ratio > 1.0,
                "debt_to_ebitda_healthy": debt_to_ebitda < 3.0,
                "overall_healthy": current_ratio > 1.0 and debt_to_ebitda < 3.0,
            }
        except Exception:
            return {"overall_healthy": False}

    def _calculate_fundamental_score(self, metrics: Dict) -> float:
        """Calculate overall fundamental score (0-100)"""
        try:
            score = 0
            max_score = 100

            # Growth score (30 points)
            growth_score = 0
            if metrics.get("revenue_growth_5y", 0) > 5:
                growth_score += 10
            if metrics.get("net_income_growth_5y", 0) > 5:
                growth_score += 10
            if metrics.get("cash_flow_growth_5y", 0) > 5:
                growth_score += 10

            # Profitability score (30 points)
            profitability_score = 0
            if metrics.get("roe_ttm", 0) > 12:
                profitability_score += 10
            if metrics.get("roic_ttm", 0) > 12:
                profitability_score += 10
            if metrics.get("profit_margin", 0) > 5:
                profitability_score += 10

            # Debt health score (20 points)
            debt_score = 0
            if metrics.get("current_ratio", 0) > 1.0:
                debt_score += 10
            if metrics.get("debt_to_ebitda", float("inf")) < 3.0:
                debt_score += 10

            # Valuation score (20 points)
            valuation_score = 0
            pe_ratio = metrics.get("pe_ratio", 0)
            if 10 < pe_ratio < 25:  # Reasonable P/E range
                valuation_score += 10
            pb_ratio = metrics.get("pb_ratio", 0)
            if 1 < pb_ratio < 3:  # Reasonable P/B range
                valuation_score += 10

            total_score = (
                growth_score + profitability_score + debt_score + valuation_score
            )
            return round(total_score, 1)

        except Exception:
            return 0

    def screen_stocks(
        self, sector: str, market: str = "US", criteria: Dict = None
    ) -> List[Dict]:
        """Screen stocks based on fundamental criteria"""
        try:
            # Get stock universe
            tickers = self.get_sector_universe(
                sector, market, criteria.get("max_market_cap")
            )

            results = []
            for ticker in tickers[:50]:  # Limit to 50 stocks for performance
                metrics = self.get_fundamental_metrics(ticker)

                if "error" not in metrics:
                    # Apply screening criteria
                    if self._meets_criteria(metrics, criteria):
                        results.append(metrics)

            # Sort by fundamental score
            results.sort(key=lambda x: x.get("fundamental_score", 0), reverse=True)

            return results

        except Exception as e:
            self.logger.error(f"Error screening stocks: {e}")
            return []

    def _meets_criteria(self, metrics: Dict, criteria: Dict) -> bool:
        """Check if stock meets screening criteria"""
        if not criteria:
            return True

        try:
            # Revenue growth criteria
            if criteria.get("min_revenue_growth", 0) > metrics.get(
                "revenue_growth_5y", 0
            ):
                return False

            # ROE criteria
            if criteria.get("min_roe", 0) > metrics.get("roe_ttm", 0):
                return False

            # Current ratio criteria
            if criteria.get("min_current_ratio", 0) > metrics.get("current_ratio", 0):
                return False

            # Debt to EBITDA criteria
            if criteria.get("max_debt_to_ebitda", float("inf")) < metrics.get(
                "debt_to_ebitda", 0
            ):
                return False

            # Market cap criteria
            if criteria.get("max_market_cap") and metrics.get(
                "market_cap", 0
            ) > criteria.get("max_market_cap"):
                return False

            return True

        except Exception:
            return False
