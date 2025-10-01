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
        self.financials: pd.DataFrame = pd.DataFrame()
        self.cash_flow: pd.DataFrame = pd.DataFrame()
        self.balance_sheet: pd.DataFrame = pd.DataFrame()

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
        data = data.iloc[:, :5]
        column_names = [f"0{period}"] + [f"-{i}{period}" for i in range(1, 5)]
        data.columns = column_names[: data.shape[1]]
        data = data.reindex(columns=column_names)
        data_stacked: pd.Series = data.stack(future_stack=True)
        data_stacked.name = stock.ticker
        return data_stacked

    def _get_data(self, verbose: bool = True) -> None:
        self.data = self._get_stock_universe()
        tickers = tqdm(self.data.index) if verbose else self.data.index
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
            except Exception:
                continue
            market_cap = info.get("marketCap", -float("inf"))
            if self.min_cap <= market_cap <= self.max_cap:
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
                        "EBITDA",
                    ],
                )
                self.financials.loc[financials.name, financials.index] = financials
                cash_flow = StockScreener._extract_section_data(
                    stock,
                    section="cash_flow",
                    subset=[
                        "Operating Cash Flow",
                    ],
                )
                self.cash_flow.loc[cash_flow.name, cash_flow.index] = cash_flow
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
                    ],
                )
                self.balance_sheet.loc[balance_sheet.name, balance_sheet.index] = (
                    balance_sheet
                )


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
