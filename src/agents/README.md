# Fundamental Agent CLI

The `fundamental_agent` module provides a quantitative stock screener that can be run from the command line. It pulls fundamentals via `YFinanceUtil`, applies the agent's filters, and writes the screened tickers to a CSV file for further analysis.

## Running the CLI

Run the module directly with Python:

```bash
python -m src.agents.fundamental_agent REGION [options]
```

The command requires a `REGION` positional argument and at least one of `--sector` or `--industry`.

## Arguments

| Flag | Required | Description |
|------|----------|-------------|
| `REGION` | Yes | Currently only `US` is supported. |
| `--sector SECTOR [SECTOR ...]` | One of sector/industry must be set | Filter results to one sector. |
| `--industry INDUSTRY [INDUSTRY ...]` | One of sector/industry must be set | Filter results to one industry |
| `--min-cap N` | No (defaults to `0`) | Minimum market capitalization in USD. |
| `--max-cap N` | No (defaults to unlimited) | Maximum market capitalization in USD. |

## Output

Each run saves a CSV file in `test-data/screener`. The filename combines the current timestamp and the sector or industry label. For example:

```
test-data/screnner/technology_20250918-104132.csv
```

## Examples

Screen US technology names above $50B:

```bash
python -m src.agents.fundamental_agent US --sector "Technology" --min-cap 50000000000
```

Screen US semiconductor stocks between $5B and $30B in market cap:

```bash
python -m src.agents.fundamental_agent US --industry "Semiconductors" --min-cap 5000000000 --max-cap 30000000000
```


## Sectors and Industries Reference List

| Sector              | Industry                              |
|--------------------|--------------------------------------|
| Technology         | Information Technology Services       |
|                   | Software – Application               |
|                   | Software – Infrastructure            |
|                   | Communication Equipment              |
|                   | Computer Hardware                    |
|                   | Consumer Electronics                 |
|                   | Electronic Components                |
|                   | Electronics & Computer Distribution  |
|                   | Scientific & Technical Instruments   |
|                   | Semiconductor Equipment & Materials  |
|                   | Semiconductors                       |
|                   | Solar                                |
| Basic Materials   | Agricultural Inputs                  |
|                   | Building Materials                   |
|                   | Chemicals                            |
|                   | Specialty Chemicals                  |
|                   | Lumber & Wood Production             |
|                   | Paper & Paper Products               |
|                   | Aluminum                             |
|                   | Copper                               |
|                   | Other Industrials Metals & Mining    |
|                   | Gold                                 |
|                   | Silver                               |
|                   | Other Precious Metals & Mining       |
|                   | Coking Coal                          |
|                   | Steel                                |
| Consumer Cyclical | Auto & Truck Dealerships             |
|                   | Auto Manufacturers                   |
|                   | Auto Parts                           |
|                   | Recreational Vehicles                |
|                   | Furnishings, Fixtures & Appliances  |
|                   | Residential Construction             |
|                   | Textile Manufacturing                |
|                   | Apparel Manufacturing                |
|                   | Footwear & Accessories               |
|                   | Packaging & Containers               |
|                   | Personal Services                    |
|                   | Restaurants                          |
|                   | Apparel Retail                       |
|                   | Department Stores                    |
|                   | Home Improvement Retail              |
|                   | Luxury Goods                         |
|                   | Internet Retail                      |
|                   | Specialty Retail                     |
|                   | Gambling                             |
|                   | Leisure                              |
|                   | Lodging                              |
|                   | Resorts & Casinos                    |
|                   | Travel Services                      |
| Financial Services | Asset Management                    |
|                   | Banks – Diversified                  |
|                   | Banks – Regional                     |
|                   | Mortgage Finance                     |
|                   | Capital Markets                      |
|                   | Financial Data & Stock Exchanges     |
|                   | Insurance – Life                     |
|                   | Insurance – Property & Casualty      |
|                   | Insurance – Reinsurance              |
|                   | Insurance – Specialty                |
|                   | Insurance Brokers                    |
|                   | Insurance – Diversified              |
|                   | Shell Companies                      |
|                   | Financial Conglomerates              |
|                   | Credit Services                      |
| Real Estate       | Real Estate – Development            |
|                   | Real Estate Services                 |
|                   | Real Estate – Diversified            |
|                   | REIT – Healthcare Facilities         |
|                   | REIT – Hotel & Motel                 |
|                   | REIT – Industrial                    |
|                   | REIT – Office                        |
|                   | REIT – Residential                   |
|                   | REIT – Retail                        |
|                   | REIT – Mortgage                      |
|                   | REIT – Specialty                     |
|                   | REIT – Diversified                   |
| Consumer Defensive| Beverages – Brewers                  |
|                   | Beverages – Wineries & Distilleries  |
|                   | Beverages – Non-Alcoholic            |
|                   | Confectioners                        |
|                   | Farm Products                        |
|                   | Household & Personal Products        |
|                   | Packaged Foods                       |
|                   | Education & Training Services        |
|                   | Discount Stores                      |
|                   | Food Distribution                    |
|                   | Grocery Stores                       |
|                   | Tobacco                              |
| Healthcare        | Biotechnology                        |
|                   | Drug Manufacturers – General         |
|                   | Drug Manufacturers – Specialty & Generic |
|                   | Healthcare Plans                     |
|                   | Medical Care Facilities              |
|                   | Pharmaceutical Retailers             |
|                   | Health Information Services          |
|                   | Medical Services                     |
|                   | Medical Instruments & Supplies       |
|                   | Diagnostics & Research               |
|                   | Medical Distribution                 |
| Utilities         | Utilities – Independent Power Producers |
|                   | Utilities – Renewable                |
|                   | Utilities – Regulated Water          |
|                   | Utilities – Regulated Electronic     |
|                   | Utilities – Regulated Gas            |
|                   | Utilities – Diversified              |
| Communication Services | Telecom Services                |
|                   | Advertising Agencies                 |
|                   | Publishing                           |
|                   | Broadcasting                         |
|                   | Entertainment                       |
|                   | Internet Content & Information       |
|                   | Electroning Gaming and Multimedia    |
| Energy           | Oil & Gas Drilling                    |
|                   | Oil & Gas E&P                        |
|                   | Oil & Gas Integrated                 |
|                   | Oil & Gas Midstream                  |
|                   | Oil & Gas Refining & Marketing       |
|                   | Oil & Gas Equipment & Services       |
|                   | Thermal Coal                         |
|                   | Uranium                              |
| Industrials      | Aerospace & Defense                   |
|                   | Specialty Business Services          |
|                   | Consulting Services                  |
|                   | Rental & Leasing Services            |
|                   | Security & Protection Services       |
|                   | Staffing & Employment Services       |
|                   | Conglomerates                        |
|                   | Engineering & Construction           |
|                   | Infrastructure Operations           |
|                   | Building Products & Equipment        |
|                   | Farm & Heavy Construction Machinery |
|                   | Industrial Distribution             |
|                   | Business Equipment & Supplies        |
|                   | Specialty Industrial Machinery      |
|                   | Metal Fabrication                   |
|                   | Pollution & Treatment Controls      |
|                   | Tools & Accessories                 |
|                   | Electrical Equipment & Parts        |
|                   | Airports & Air Services             |
|                   | Airlines                            |
|                   | Railroads                           |
|                   | Marine Shipping                     |
|                   | Trucking                            |
|                   | Integrated Freight & Logistics      |
|                   | Waste Management                    |
