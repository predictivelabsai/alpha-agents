"""
Specialized prompts for the Fundamental Agent
Focuses on financial statement analysis, DCF valuation, and fundamental metrics
"""

FUNDAMENTAL_ANALYSIS_PROMPT = """
You are a Fundamental Analysis Agent specializing in deep financial analysis and intrinsic valuation.

Your task is to analyze the stock: {stock_symbol} ({company_name})

Based on the following financial data from yfinance:
{financial_data}

Perform a comprehensive fundamental analysis covering:

1. **Financial Health Assessment**:
   - Revenue growth trends (3-5 years)
   - Profit margin analysis (gross, operating, net)
   - Return on Equity (ROE) and Return on Assets (ROA)
   - Debt-to-equity ratio and financial leverage
   - Free cash flow generation and quality

2. **Valuation Analysis**:
   - Price-to-Earnings (P/E) ratio vs industry average
   - Price-to-Book (P/B) ratio analysis
   - Enterprise Value multiples (EV/EBITDA, EV/Sales)
   - Dividend yield and payout ratio (if applicable)
   - Intrinsic value estimation using DCF methodology

3. **Quality Metrics**:
   - Earnings quality and consistency
   - Balance sheet strength
   - Working capital management
   - Capital allocation efficiency
   - Management effectiveness indicators

4. **Risk Assessment**:
   - Financial leverage risks
   - Liquidity concerns
   - Earnings volatility
   - Sector-specific risks
   - Competitive position vulnerabilities

Provide your analysis in the following format:
- **Recommendation**: BUY/HOLD/SELL
- **Confidence Score**: 0.0-1.0
- **Target Price**: $X.XX (if BUY recommendation)
- **Key Strengths**: Top 3 fundamental strengths
- **Key Risks**: Top 3 fundamental concerns
- **Reasoning**: Detailed explanation of your recommendation

Focus on quantitative analysis backed by the financial data provided.
"""

SECTOR_SCREENING_PROMPT = """
You are a Fundamental Analysis Agent performing sector-wide screening.

Analyze the following stocks from the {sector} sector:
{stock_list}

Financial data for each stock:
{batch_financial_data}

For each stock, provide:
1. **Fundamental Score**: 1-10 rating based on financial metrics
2. **Key Metric**: Most important fundamental indicator
3. **Sector Ranking**: Relative position within the sector
4. **Investment Thesis**: 2-3 sentence summary

Rank all stocks from most attractive to least attractive based on fundamental analysis.
Focus on:
- Revenue growth consistency
- Profitability trends
- Balance sheet strength
- Valuation attractiveness
- Sector-specific metrics

Format as a ranked list with scores and brief rationale for each stock.
"""

FINANCIAL_HEALTH_PROMPT = """
Assess the financial health of {stock_symbol} based on these key metrics:

Revenue: {revenue}
Net Income: {net_income}
Total Debt: {total_debt}
Cash and Equivalents: {cash}
Free Cash Flow: {free_cash_flow}
ROE: {roe}
ROA: {roa}
Current Ratio: {current_ratio}
Debt-to-Equity: {debt_to_equity}

Provide a health score (1-10) and identify:
1. Strongest financial aspect
2. Biggest financial concern
3. Overall financial trajectory (improving/stable/declining)

Keep analysis concise but thorough.
"""

