import pandas as pd
from difflib import get_close_matches
from collections.abc import Collection
import yfinance as yf
from tqdm import tqdm
from pathlib import Path


def _create_us_stock_universe():
    nasdaq = pd.read_csv(
        "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt",
        sep="|",
        usecols=["Symbol", "Security Name", "ETF"],
        keep_default_na=False,
    ).iloc[:-1]
    nasdaq = nasdaq[nasdaq["ETF"] == "N"]
    nasdaq.loc[:, "Security Name"] = nasdaq["Security Name"].str.split(" - ").str[0]
    nasdaq = (
        nasdaq.drop_duplicates(subset=["Security Name"], keep="first").rename(
            {"Symbol": "ticker"}, axis=1
        )
    )[["ticker"]]
    others = pd.read_csv(
        "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt",
        sep="|",
        usecols=["ACT Symbol", "Security Name", "Exchange", "ETF"],
    ).iloc[:-1]
    others = others[others["Exchange"].isin(["N", "A"]) & (others["ETF"] == "N")]
    others = others[~others["Security Name"].str.contains("%")]
    others = others[others["ACT Symbol"].str.match(r"^[A-Za-z0-9]+$")]
    others = others.rename({"ACT Symbol": "ticker"}, axis=1)[["ticker"]]
    all_tickers = (
        pd.concat([nasdaq, others]).sort_values("ticker").reset_index(drop=True)
    )
    sectors = []
    industries = []
    for t in tqdm(all_tickers["ticker"]):
        try:
            info = yf.Ticker(t).info
            sectors.append(info.get("sector", None))
            industries.append(info.get("industry", None))
        except Exception:
            sectors.append(None)
            industries.append(None)
    all_tickers["sector"] = sectors
    all_tickers["industry"] = industries
    all_tickers.to_csv("utils/universe/us.csv", index=False)


class StockScreener:
    REGIONS = ["US"]
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

    SECTORS: list[str] = list(SECTORS_INDUSTRIES.keys())
    INDUSTRIES: list[str] = [
        industry
        for industries in SECTORS_INDUSTRIES.values()
        for industry in industries
    ]

    def __init__(
        self,
        region: str,
        sectors: str | Collection[str] | None = None,
        industries: str | Collection[str] | None = None,
        min_cap: float = 0,
        max_cap: float = float("inf"),
    ) -> None:
        self._validate_region(region)
        self.region: str = region
        self.industries: set[str]
        self.min_cap: float = min_cap
        self.max_cap: float = max_cap
        if sectors is None and industries is None:
            self.industries = set(self.INDUSTRIES)
        elif sectors is None and industries is not None:
            self.industries = self._normalize_industries(industries)
        elif sectors is not None and industries is None:
            sectors = self._normalize_sectors(sectors)
            self.industries = {
                ind for s in sectors for ind in self.SECTORS_INDUSTRIES[s]
            }
        else:
            sectors = self._normalize_sectors(sectors)
            self.industries = self._normalize_industries(industries)
            self.industries |= {
                ind for s in sectors for ind in self.SECTORS_INDUSTRIES[s]
            }

    @staticmethod
    def _as_set(x: str | Collection[str]) -> set[str]:
        return {x} if isinstance(x, str) else set(x)

    @classmethod
    def _validate_region(cls, region: str) -> None:
        if region not in cls.REGIONS:
            raise ValueError(f"Invalid region '{region}'.")

    @classmethod
    def _normalize_sectors(
        cls,
        sectors: str | Collection[str],
    ) -> set[str]:
        result = cls._as_set(sectors)
        for s in result:
            if s not in cls.SECTORS:
                close = get_close_matches(s, cls.SECTORS, n=1)
                msg = f"Invalid sector '{s}'."
                if close:
                    msg += f" Did you mean '{close[0]}'?"
                raise ValueError(msg)
        return result

    @classmethod
    def _normalize_industries(
        cls,
        industries: str | Collection[str],
    ) -> set[str]:
        result = cls._as_set(industries)
        for ind in result:
            if ind not in cls.INDUSTRIES:
                close = get_close_matches(ind, cls.INDUSTRIES, n=1)
                msg = f"Invalid industry '{ind}'."
                if close:
                    msg += f" Did you mean '{close[0]}'?"
                raise ValueError(msg)
        return result

    def _get_stock_universe(self, path="utils/universe") -> pd.DataFrame:
        base = Path(path)
        base.mkdir(exist_ok=True)
        csv_path = base / f"{self.region.lower()}.csv"

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Stock universe file for region '{self.region}' not found at {csv_path}."
            )

        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
        out = df[df["industry"].isin(self.industries)].copy().set_index("ticker")
        return out

    @staticmethod
    def _extract_section_data(
        stock: yf.Ticker,
        section: str,
        subset: list[str] | None = None,
        period: str = "Y",
    ) -> pd.Series:
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

    def _get_data(self, verbose: bool = True) -> None:
        self.data = self._get_stock_universe()
        self.yearly_calc_data = pd.DataFrame()
        self.quarterly_calc_data = pd.DataFrame()
        tickers = tqdm(self.data.index) if verbose else self.data.index
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
            except Exception:
                continue
            market_cap = info.get("marketCap", -float("inf"))

            company_name = info.get("longName") or info.get("shortName")
            self.data.loc[ticker, "market_cap"] = market_cap
            self.data.loc[ticker, "company_name"] = company_name
            financials = StockScreener._extract_section_data(
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
            quarterly_financials = StockScreener._extract_section_data(
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
            cash_flow = StockScreener._extract_section_data(
                stock,
                section="cash_flow",
                subset=["Operating Cash Flow", "Repayment of Debt"],
            )
            quarterly_cash_flow = StockScreener._extract_section_data(
                stock,
                section="cash_flow",
                subset=["Operating Cash Flow", "Repayment Of Debt"],
                period="Q",
            )
            balance_sheet = StockScreener._extract_section_data(
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
            quarterly_balance_sheet = StockScreener._extract_section_data(
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
            yearly_combined = pd.concat([financials, cash_flow, balance_sheet])
            quarterly_combined = pd.concat(
                [quarterly_financials, quarterly_cash_flow, quarterly_balance_sheet]
            )
            self.yearly_calc_data.loc[yearly_combined.name, yearly_combined.index] = (
                yearly_combined
            )
            self.quarterly_calc_data.loc[
                quarterly_combined.name, quarterly_combined.index
            ] = quarterly_combined

    def _compute_metric(self) -> pd.Series:
        self.data["gross_profit_margin_ttm"] = self._compute_ttm_ratio(
            "Gross Profit", "Total Revenue"
        )
        self.data["operating_profit_margin_ttm"] = self._compute_ttm_ratio(
            "Operating Income", "Total Revenue"
        )
        self.data["net_profit_margin_ttm"] = self._compute_ttm_ratio(
            "Net Income", "Total Revenue"
        )
        self.data["ebit_margin_ttm"] = self._compute_ttm_ratio("EBIT", "Total Revenue")
        self.data["ebitda_margin_ttm"] = self._compute_ttm_ratio(
            "EBITDA", "Total Revenue"
        )
        self.data["roa_ttm"] = self._compute_ttm_average_ratio(
            "Net Income", "Total Assets"
        )
        self.data["roe_ttm"] = self._compute_ttm_average_ratio(
            "Net Income", "Stockholders Equity"
        )
        yearly_nopat = (
            1 - self.yearly_calc_data["Tax Rate For Calcs"]
        ) * self.yearly_calc_data["EBIT"]
        self.yearly_calc_data = pd.concat(
            [self.yearly_calc_data, pd.concat({"NOPAT": yearly_nopat}, axis=1)], axis=1
        )
        quarterly_nopat = (
            1 - self.quarterly_calc_data["Tax Rate For Calcs"]
        ) * self.quarterly_calc_data["EBIT"]
        self.quarterly_calc_data = pd.concat(
            [self.quarterly_calc_data, pd.concat({"NOPAT": quarterly_nopat}, axis=1)],
            axis=1,
        )
        self.data["roic_ttm"] = self._compute_ttm_average_ratio(
            "NOPAT", "Invested Capital"
        )
        self.data["total_revenue_4y_cagr"] = self._compute_4y_growth("Total Revenue")
        self.data["net_income_4y_cagr"] = self._compute_4y_growth("Net Income")
        self.data["operating_cash_flow_4y_cagr"] = self._compute_4y_growth(
            "Operating Cash Flow"
        )
        self.data["total_revenue_4y_consistency"] = self._compute_consistency_ratio(
            "Total Revenue"
        )
        self.data["net_income_4y_consistency"] = self._compute_consistency_ratio(
            "Net Income"
        )
        self.data["operating_cash_flow_4y_consistency"] = (
            self._compute_consistency_ratio("Operating Cash Flow")
        )
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

    def _compute_4y_growth(self, metric: str) -> pd.Series:
        return (
            self.yearly_calc_data[metric].iloc[:, -1]
            / self.yearly_calc_data[metric].iloc[:, 0]
        ) ** (1 / 5) - 1

    def _compute_ttm_ratio(self, dividend: str, divisor: str) -> pd.Series:
        dividend_sum, divisor_sum = (
            self.quarterly_calc_data[dividend].iloc[:, :4].sum(axis=1),
            self.quarterly_calc_data[divisor].iloc[:, :4].sum(axis=1),
        )
        return pd.to_numeric(dividend_sum) / pd.to_numeric(divisor_sum)

    def _compute_ttm_average_ratio(self, dividend: str, divisor: str) -> pd.Series:
        dividend_sum = self.quarterly_calc_data[dividend].iloc[:, :4].sum(axis=1)
        divisor_df = self.quarterly_calc_data[divisor]
        divisor_avg = (
            divisor_df.iloc[:, [0, 4]].sum(axis=1, skipna=False)
            + (divisor_df.iloc[:, 1:4] * 2).sum(axis=1, skipna=False)
        ) / 8
        return dividend_sum / divisor_avg

    def _compute_latest_ratio(self, dividend: str, divisor: str) -> pd.Series:
        return self.yearly_calc_data[dividend] / self.yearly_calc_data[divisor]

    def _compute_consistency_ratio(self, metric: str) -> pd.Series:
        pct_change = self.yearly_calc_data[metric].iloc[:, ::-1].pct_change(axis=1)
        return pct_change.mean(axis=1) / pct_change.std(axis=1)

    def screen(self):
        self._get_data()
        self._compute_metric()
        self.score()
        return self.data[
            self.data["market_cap"].between(self.min_cap, self.max_cap)
        ].sort_values(by="score", ascending=False)

    def score(
        self,
        profitability_weight: float = 1,
        growth_weight: float = 1,
        debt_weight: float = 1,
    ):
        score = self.data.rank(pct=True).mean(axis=1)
        self.data.loc[:, "score"] = score * 100


def get_data_deprecated(
    region: str,
    sectors: str | Collection[str] | None = None,
    industries: str | Collection[str] | None = None,
    min_cap: float = 0,
    max_cap: float = float("inf"),
    verbose: bool = True,
) -> list[str]:
    """
    DEPRECATED. Use StockScreener class instead.
    """
    screener = StockScreener(
        region=region,
        sectors=sectors,
        industries=industries,
        min_cap=min_cap,
        max_cap=max_cap,
    )
    screener._get_data(verbose=verbose)
    info = screener._get_stock_universe()
    info = info.loc[
        screener.financials.index.intersection(
            screener.cash_flow.index.intersection(screener.balance_sheet.index)
        )
    ]
    info = info[["company_name", "market_cap"]]
    return info, screener.financials, screener.cash_flow, screener.balance_sheet


if __name__ == "__main__":
    screener = StockScreener("US", industries="Semiconductors")
    screener._get_data(verbose=True)
    screener._compute_metric()
    print(screener.data)
    print(screener.data.isna().sum())
