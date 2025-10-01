"""
Fundamental Agent - Quantitative Filtering
Filters stocks based on strict quantitative criteria without recommendations
"""

import json
import logging
from datetime import datetime
from collections.abc import Collection
from typing import Dict, List
from dataclasses import dataclass, asdict
import pandas as pd
import pathlib

# import utils.metric as m
import argparse

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

from tqdm import tqdm

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from utils.yfinance_util import YFinanceUtil
from utils.data import StockScreener


@dataclass
class QualifiedCompany:
    """Data class for companies that meet quantitative criteria"""

    ticker: str
    company_name: str
    sector: str
    industry: str
    market_cap: float

    # Growth metrics
    revenue_growth_5y: float
    net_income_growth_5y: float
    cash_flow_growth_5y: float

    # Profitability metrics
    roe_ttm: float
    roic_ttm: float
    gross_margin: float
    profit_margin: float

    # Debt metrics
    current_ratio: float
    debt_to_ebitda: float
    debt_service_ratio: float

    # Quality scores
    growth_score: float
    profitability_score: float
    debt_score: float
    overall_score: float

    # Optional LLM commentary
    # commentary: str

    # timestamp: str


class FundamentalAgent:
    """Agent for quantitative fundamental filtering"""

    def __init__(self, api_key: str = None, use_llm: bool = False):
        self.yfinance_util = YFinanceUtil()
        self.use_llm = use_llm

        # Initialize LLM only if requested
        if use_llm and api_key:
            try:
                self.llm = ChatOpenAI(
                    model="gpt-4", temperature=0.1, openai_api_key=api_key
                )
            except Exception as e:
                logger.warning(f"Failed to initialize LLM: {e}")
                self.llm = None
        else:
            self.llm = None

    def screen(
        self,
        sector: str,
        region: str,
        min_cap: float = 0,
        max_cap: float = float("inf"),
        n_return: int = 20,
    ) -> pd.DataFrame:
        orig_data = get_data_deprecated(
            region, sector, min_cap=min_cap, max_cap=max_cap
        )
        data = pd.DataFrame(index=orig_data.index)
        data["revenue_growth"] = m.calculate_growth(orig_data, prefix="total_revenue")
        data["net_income_growth"] = m.calculate_growth(orig_data, prefix="net_income")
        data["cash_flow_growth"] = m.calculate_growth(orig_data, prefix="cash_flow")
        data["stockholders_equity_avg"] = m.calculate_average(
            orig_data, prefix="stockholders_equity"
        )
        data

    def screen_sector(
        self,
        region: str,
        sectors: str | Collection[str] | None = None,
        industries: str | Collection[str] | None = None,
        min_cap: float = 0,
        max_cap: float = float("inf"),
        verbose: bool = True,
    ) -> pd.DataFrame:
        """Screen sector for companies meeting quantitative criteria"""

        # Get stock universe
        tickers = (
            StockScreener(
                region=region,
                sectors=sectors,
                industries=industries,
                min_cap=min_cap,
                max_cap=max_cap,
            )
            ._get_stock_universe()
            .index.tolist()
        )
        qualified_companies = []
        if verbose:
            tickers = tqdm(tickers)
        for ticker in tickers:
            try:
                # Get financial metrics
                metrics = self.yfinance_util.get_fundamental_metrics(ticker)

                if "error" in metrics:
                    continue

                # Apply quantitative filters
                qualified_company = self._create_qualified_company(
                    metrics, metrics["sector"]
                )
                qualified_companies.append(qualified_company)

            # print(
            #     f"✓ {ticker} qualified with score {qualified_company.overall_score:.1f}"
            # )

            except Exception as e:
                logger.error(f"Error analyzing {ticker}: {e}")
                continue

        # Sort by overall score
        qualified_companies.sort(key=lambda x: x.overall_score, reverse=True)

        # Limit results

        qualified_companies_list = [asdict(s) for s in qualified_companies]
        return pd.DataFrame(qualified_companies_list)

    def _meets_quantitative_criteria(self, metrics: Dict) -> bool:
        """Check if company meets strict quantitative criteria"""
        try:
            # 1. Growth Consistency Criteria
            revenue_growth = metrics.get("revenue_growth_5y", 0)
            income_growth = metrics.get("net_income_growth_5y", 0)
            cf_growth = metrics.get("cash_flow_growth_5y", 0)

            # Must have consistent positive growth
            if revenue_growth <= 0 or cf_growth <= 0:
                return False

            # If net income declining, check operating income (simplified check)
            if income_growth <= 0:
                # For now, we'll be strict and require positive net income growth
                return False

            # 2. Profitability and Efficiency Criteria
            roe = metrics.get("roe_ttm", 0)
            roic = metrics.get("roic_ttm", 0)

            # ROE and ROIC must be >= 12%
            if roe < 12 or roic < 12:
                return False

            # 3. Conservative Debt Criteria
            current_ratio = metrics.get("current_ratio", 0)
            debt_to_ebitda = metrics.get("debt_to_ebitda", float("inf"))
            debt_service_ratio = metrics.get("debt_service_ratio", 0)

            # Current ratio > 1.0
            if current_ratio <= 1.0:
                return False

            # Debt to EBITDA < 3.0
            if debt_to_ebitda >= 3.0:
                return False

            # Debt service ratio < 30% (we calculate as percentage)
            if debt_service_ratio > 0 and debt_service_ratio < 30:
                return False

            # 4. Additional Quality Checks
            gross_margin = metrics.get("gross_margin", 0)
            profit_margin = metrics.get("profit_margin", 0)

            # Must have reasonable margins
            if gross_margin <= 0 or profit_margin <= 0:
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking criteria: {e}")
            return False

    def _create_qualified_company(self, metrics: Dict, sector: str) -> QualifiedCompany:
        """Create QualifiedCompany object from metrics"""

        # Calculate component scores (0-100)
        growth_score = self._calculate_growth_score(metrics)
        profitability_score = self._calculate_profitability_score(metrics)
        debt_score = self._calculate_debt_score(metrics)

        # Overall score (weighted average)
        overall_score = (
            growth_score * 0.4 + profitability_score * 0.4 + debt_score * 0.2
        )

        # Generate commentary if LLM available
        commentary = ""
        if self.llm:
            commentary = self._generate_commentary(metrics)

        qualified_company = QualifiedCompany(
            ticker=metrics.get("ticker", ""),
            company_name=metrics.get("company_name", "N/A"),
            sector=metrics.get("sector", sector),
            industry=metrics.get("industry"),
            market_cap=metrics.get("market_cap", 0),
            # Growth metrics
            revenue_growth_5y=metrics.get("revenue_growth_5y", 0),
            net_income_growth_5y=metrics.get("net_income_growth_5y", 0),
            cash_flow_growth_5y=metrics.get("cash_flow_growth_5y", 0),
            # Profitability metrics
            roe_ttm=metrics.get("roe_ttm", 0),
            roic_ttm=metrics.get("roic_ttm", 0),
            gross_margin=metrics.get("gross_margin", 0),
            profit_margin=metrics.get("profit_margin", 0),
            # Debt metrics
            current_ratio=metrics.get("current_ratio", 0),
            debt_to_ebitda=metrics.get("debt_to_ebitda", 0),
            debt_service_ratio=metrics.get("debt_service_ratio", 0),
            # Scores
            growth_score=growth_score,
            profitability_score=profitability_score,
            debt_score=debt_score,
            overall_score=overall_score,
        )

        return qualified_company

    def _calculate_growth_score(self, metrics: Dict) -> float:
        """Calculate growth quality score (0-100)"""
        score = 0

        # Revenue growth (0-40 points)
        revenue_growth = metrics.get("revenue_growth_5y", 0)
        if revenue_growth > 20:
            score += 40
        elif revenue_growth > 10:
            score += 30
        elif revenue_growth > 5:
            score += 20
        elif revenue_growth > 0:
            score += 10

        # Income growth (0-30 points)
        income_growth = metrics.get("net_income_growth_5y", 0)
        if income_growth > 20:
            score += 30
        elif income_growth > 10:
            score += 20
        elif income_growth > 5:
            score += 15
        elif income_growth > 0:
            score += 10

        # Cash flow growth (0-30 points)
        cf_growth = metrics.get("cash_flow_growth_5y", 0)
        if cf_growth > 20:
            score += 30
        elif cf_growth > 10:
            score += 20
        elif cf_growth > 5:
            score += 15
        elif cf_growth > 0:
            score += 10

        return min(score, 100)

    def _calculate_profitability_score(self, metrics: Dict) -> float:
        """Calculate profitability quality score (0-100)"""
        score = 0

        # ROE (0-40 points)
        roe = metrics.get("roe_ttm", 0)
        if roe > 25:
            score += 40
        elif roe > 20:
            score += 35
        elif roe > 15:
            score += 30
        elif roe > 12:
            score += 20

        # ROIC (0-40 points)
        roic = metrics.get("roic_ttm", 0)
        if roic > 25:
            score += 40
        elif roic > 20:
            score += 35
        elif roic > 15:
            score += 30
        elif roic > 12:
            score += 20

        # Margins (0-20 points)
        gross_margin = metrics.get("gross_margin", 0)
        profit_margin = metrics.get("profit_margin", 0)

        if gross_margin > 50 and profit_margin > 20:
            score += 20
        elif gross_margin > 30 and profit_margin > 10:
            score += 15
        elif gross_margin > 20 and profit_margin > 5:
            score += 10
        elif gross_margin > 0 and profit_margin > 0:
            score += 5

        return min(score, 100)

    def _calculate_debt_score(self, metrics: Dict) -> float:
        """Calculate debt health score (0-100)"""
        score = 0

        # Current ratio (0-40 points)
        current_ratio = metrics.get("current_ratio", 0)
        if current_ratio > 2.0:
            score += 40
        elif current_ratio > 1.5:
            score += 30
        elif current_ratio > 1.2:
            score += 20
        elif current_ratio > 1.0:
            score += 10

        # Debt to EBITDA (0-40 points)
        debt_to_ebitda = metrics.get("debt_to_ebitda", float("inf"))
        if debt_to_ebitda < 0.5:
            score += 40
        elif debt_to_ebitda < 1.0:
            score += 35
        elif debt_to_ebitda < 2.0:
            score += 25
        elif debt_to_ebitda < 3.0:
            score += 15

        # Debt service (0-20 points)
        debt_service = metrics.get("debt_service_ratio", 0)
        if debt_service > 100:  # Very strong coverage
            score += 20
        elif debt_service > 50:
            score += 15
        elif debt_service > 30:
            score += 10
        elif debt_service > 0:
            score += 5

        return min(score, 100)

    def _generate_commentary(self, metrics: Dict) -> str:
        """Generate LLM commentary on the qualified company"""
        try:
            if not self.llm:
                return ""

            prompt = f"""
Provide brief commentary on this qualified company's financial strength:

Company: {metrics.get("company_name", "N/A")}
Sector: {metrics.get("sector", "N/A")}

Key Metrics:
- Revenue Growth (5Y): {metrics.get("revenue_growth_5y", 0):.1f}%
- ROE: {metrics.get("roe_ttm", 0):.1f}%
- ROIC: {metrics.get("roic_ttm", 0):.1f}%
- Current Ratio: {metrics.get("current_ratio", 0):.2f}
- Debt/EBITDA: {metrics.get("debt_to_ebitda", 0):.2f}

Provide 2-3 sentences highlighting the key financial strengths that make this a qualified candidate.
"""

            messages = [
                SystemMessage(
                    content="You are a financial analyst providing brief commentary on qualified companies."
                ),
                HumanMessage(content=prompt),
            ]

            response = self.llm.invoke(messages)
            return response.content.strip()

        except Exception as e:
            logger.error(f"Error generating commentary: {e}")
            return ""

    def _save_screening_results(
        self, companies: List[QualifiedCompany], sector: str, market: str
    ):
        """Save screening results to tracing folder"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = (
                f"tracing/fundamental_screening_{sector}_{market}_{timestamp}.json"
            )

            results = {
                "sector": sector,
                "market": market,
                "timestamp": timestamp,
                "total_qualified": len(companies),
                "companies": [asdict(company) for company in companies],
            }

            with open(filename, "w") as f:
                json.dump(results, f, indent=2, default=str)

            print(f"Saved screening results to {filename}")

        except Exception as e:
            logger.error(f"Error saving screening results: {e}")

    def get_screening_summary(self, companies: List[QualifiedCompany]) -> Dict:
        """Get summary statistics for screening results"""
        if not companies:
            return {}

        return {
            "total_qualified": len(companies),
            "avg_overall_score": sum(c.overall_score for c in companies)
            / len(companies),
            "avg_growth_score": sum(c.growth_score for c in companies) / len(companies),
            "avg_profitability_score": sum(c.profitability_score for c in companies)
            / len(companies),
            "avg_debt_score": sum(c.debt_score for c in companies) / len(companies),
            "top_performers": [c.ticker for c in companies[:5]],
            "sectors_represented": list(set(c.sector for c in companies)),
            "market_cap_range": {
                "min": min(c.market_cap for c in companies),
                "max": max(c.market_cap for c in companies),
                "avg": sum(c.market_cap for c in companies) / len(companies),
            },
        }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the fundamental agent.")
    parser.add_argument("region")
    parser.add_argument("--sector", help="Sector to filter by.")
    parser.add_argument("--industry", help="Industry to filter by.")
    parser.add_argument(
        "--min-cap",
        dest="min_cap",
        type=int,
        default=0,
        help="Minimum market capitalization in USD.",
    )
    parser.add_argument(
        "--max-cap",
        dest="max_cap",
        type=int,
        default=float("inf"),
        help="Maximum market capitalization in USD.",
    )
    parser.add_argument("--excel", action="store_true")
    args = parser.parse_args(argv)
    if (args.sector is not None) ^ (args.industry is not None):
        sector_industry: str = args.sector or args.industry
    else:
        parser.error("Please provide exactly one of --sector or --industry.")
    agent = FundamentalAgent()
    results = agent.screen_sector(
        region=args.region,
        sectors=args.sector,
        industries=args.industry,
        min_cap=args.min_cap,
        max_cap=args.max_cap,
    )
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    pathlib.Path("test-data/screener").mkdir(exist_ok=True)
    if args.excel:
        results.to_excel(
            f"test-data/screener/{sector_industry.replace(' - ', '_').lower()}_{timestamp}.xlsx"
        )
    else:
        results.to_csv(
            f"test-data/screener/{sector_industry.replace(' - ', '_').lower()}_{timestamp}.csv"
        )


if __name__ == "__main__":
    main()
