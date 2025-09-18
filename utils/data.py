import pandas as pd
from difflib import get_close_matches
from typing import Callable, ParamSpec, TypeVar, Concatenate, cast, overload
from collections.abc import Collection
import yfinance as yf
from tqdm import tqdm
from pathlib import Path
import requests
from functools import wraps

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

REGIONS = ["US"]
SECTORS: list[str] = list(SECTORS_INDUSTRIES.keys())
_INDUSTRY_TO_SECTOR = {
    industry: sector for sector, inds in SECTORS_INDUSTRIES.items() for industry in inds
}


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


P = ParamSpec("P")
R = TypeVar("R")


def _as_set(x: str | Collection[str]) -> list[str]:
    return {x} if isinstance(x, str) else set(x)


def _validate_region(region: str) -> None:
    if region not in REGIONS:
        raise ValueError(
            f"Invalid region '{region}'. Must be one of: {', '.join(sorted(REGIONS))}"
        )


def _normalize_sectors(
    sectors: str | Collection[str],
) -> set[str]:
    result = _as_set(sectors)
    for s in result:
        if s not in SECTORS_INDUSTRIES:
            close = get_close_matches(s, list(SECTORS_INDUSTRIES.keys()), n=1)
            msg = f"Invalid sector '{s}'."
            if close:
                msg += f" Did you mean '{close[0]}'?"
            raise ValueError(msg)
    return result


def _normalize_industries(
    industries: str | Collection[str],
) -> set[str]:
    result = _as_set(industries)
    for ind in result:
        if ind not in _INDUSTRY_TO_SECTOR:
            close = get_close_matches(ind, list(_INDUSTRY_TO_SECTOR.keys()), n=1)
            msg = f"Invalid industry '{ind}'."
            if close:
                msg += f" Did you mean '{close[0]}'?"
            raise ValueError(msg)
    return result


def validate_region_sector(
    func: Callable[Concatenate[str, Collection[str], Collection[str], P], R],
) -> Callable[
    Concatenate[str, str | Collection[str] | None, str | Collection[str] | None, P], R
]:
    @wraps(func)
    def wrapper(
        region: str,
        sectors: str | Collection[str] | None,
        industries: str | Collection[str] | None,
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        _validate_region(region)
        if sectors is None and industries is None:
            industries = set().union(*SECTORS_INDUSTRIES.values())
            sectors = set()
        elif sectors is None and industries is not None:
            industries = _normalize_industries(industries)
            sectors = set()
        elif sectors is not None and industries is None:
            sectors = _normalize_sectors(sectors)
            industries = {ind for s in sectors for ind in SECTORS_INDUSTRIES[s]}
        else:
            sectors = _normalize_sectors(sectors)
            industries = _normalize_industries(industries)
            industries |= {ind for s in sectors for ind in SECTORS_INDUSTRIES[s]}
        return func(
            region,
            cast(Collection[str], sectors),
            cast(Collection[str], industries),
            *args,
            **kwargs,
        )

    return wrapper


@overload
def get_stock_universe(
    region: str,
    sectors: str | Collection[str] | None,
    industries: str | Collection[str] | None,
) -> list[str]: ...


@validate_region_sector
def get_stock_universe(
    region: str,
    sectors: set[str],
    industries: set[str],
) -> list[str]:
    return list(_get_stock_universe(region, sectors, industries).index)


def _get_stock_universe(
    region: str,
    sectors: set[str],
    industries: set[str],
) -> pd.DataFrame:
    base = Path("utils/universe")
    base.mkdir(exist_ok=True)
    csv_path = base / f"{region.lower()}.csv"

    if not csv_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    out = df[df["industry"].isin(industries)].copy().set_index("ticker")
    return out


def _extract_section_data(
    stock: yf.Ticker,
    section: str,
    subset: list[str] | None = None,
    period: str = "Y",
) -> pd.Series:
    """Extracts standardized time-indexed data for a given section.
    period:  "Y" for yearly, anything else treated as quarterly ("Q")
    """
    section = section.lower()
    if period == "Q":
        section = "quarterly_" + section
    data: pd.DataFrame = getattr(stock, section).copy()

    if subset is not None:
        data = data.reindex(subset)
    else:
        subset = data.index

    data.columns = [f"0{period}"] + [f"-{i}{period}" for i in range(1, data.shape[1])]
    data = data.stack(future_stack=True).to_frame().T
    data.columns = [
        f"{metric.lower().replace(' ', '_')}_{per}" for metric, per in data.columns
    ]

    data = data.reindex(
        columns=[
            f"{metric.lower().replace(' ', '_')}_{y}{period}"
            for metric in subset
            for y in [0, -1, -2, -3, -4]
        ]
    )
    return pd.Series(data.iloc[0], name=stock.ticker)


@validate_region_sector
def get_data(
    region: str,
    sectors: str | Collection[str] | None,
    industries: str | Collection[str] | None,
    min_cap: float = 0,
    max_cap: float = float("inf"),
) -> pd.DataFrame:
    data = _get_stock_universe(region, sectors, industries)
    for ticker in tqdm(data.index):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
        except Exception:
            continue
        market_cap = info.get("marketCap", -float("inf"))
        if min_cap <= market_cap <= max_cap:
            company_name = info.get("longName") or info.get("shortName")
            data.loc[ticker, "market_cap"] = market_cap
            data.loc[ticker, "company_name"] = company_name
            financials = _extract_section_data(
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
            data.loc[ticker, financials.index] = financials
            cash_flow = _extract_section_data(
                stock,
                section="cash_flow",
                subset=[
                    "Operating Cash Flow",
                ],
            )
            data.loc[ticker, cash_flow.index] = cash_flow
            balance_sheet = _extract_section_data(
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
            data.loc[ticker, balance_sheet.index] = balance_sheet
    return data


def get_data_deprecated(
    region: str,
    sectors: str | Collection[str] | None,
    industries: str | Collection[str] | None,
    min_cap: float = 0,
    max_cap: float = float("inf"),
) -> pd.DataFrame:
    tickers = get_stock_universe(region, sectors, industries)
    tickers_obj = yf.Tickers(" ".join(tickers))
    for ticker in tqdm(tickers):
        try:
            stock = tickers_obj.tickers[ticker]
            info = stock.info
        except Exception:
            tickers.remove(ticker)
            continue
        market_cap = info.get("marketCap", -float("inf"))
        if not (min_cap <= market_cap <= max_cap):
            tickers.remove(ticker)
    return tickers
