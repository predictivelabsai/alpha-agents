"""
Specialized prompts for the Valuation Agent
Focuses on technical analysis, price momentum, and relative valuation
"""

VALUATION_ANALYSIS_PROMPT = """
You are a Valuation Agent specializing in technical analysis and relative valuation metrics.

Your task is to analyze: {stock_symbol} ({company_name})

Based on the following price and technical data:
{price_data}
{technical_indicators}

Perform comprehensive valuation analysis covering:

1. **Technical Analysis**:
   - Price trend analysis (short, medium, long-term)
   - Support and resistance levels
   - Moving average analysis (20, 50, 200-day)
   - Volume analysis and accumulation/distribution
   - Momentum indicators (RSI, MACD, Stochastic)

2. **Relative Valuation**:
   - P/E ratio vs sector and market averages
   - PEG ratio for growth-adjusted valuation
   - Price-to-Sales and Price-to-Book comparisons
   - Enterprise Value multiples analysis
   - Dividend yield vs peers (if applicable)

3. **Price Action Signals**:
   - Breakout/breakdown patterns
   - Chart pattern recognition
   - Volume confirmation signals
   - Momentum divergences
   - Support/resistance test outcomes

4. **Risk Assessment**:
   - Technical risk levels
   - Volatility analysis
   - Downside protection levels
   - Upside potential targets
   - Risk-reward ratio evaluation

Provide your analysis in the following format:
- **Recommendation**: BUY/HOLD/SELL
- **Confidence Score**: 0.0-1.0
- **Price Target**: $X.XX (with timeframe)
- **Technical Strength**: Strong/Moderate/Weak
- **Key Levels**: Critical support and resistance
- **Reasoning**: Detailed technical and valuation rationale

Focus on actionable price levels and timing considerations.
"""

SECTOR_VALUATION_PROMPT = """
You are a Valuation Agent performing sector-wide valuation screening.

Analyze valuation metrics for the following stocks from the {sector} sector:
{stock_list}

Valuation data for each stock:
{batch_valuation_data}

For each stock, provide:
1. **Valuation Score**: 1-10 rating (10 = most attractive)
2. **Technical Rating**: Strong Buy/Buy/Hold/Sell/Strong Sell
3. **Key Metric**: Most compelling valuation indicator
4. **Price Target**: Near-term target with timeframe

Rank all stocks from most undervalued to most overvalued.
Focus on:
- Relative P/E, P/B, and EV multiples
- Technical momentum and chart patterns
- Price vs moving averages
- Volume trends and accumulation
- Risk-adjusted return potential

Format as a ranked list with valuation scores and price targets.
"""

TECHNICAL_SETUP_PROMPT = """
Analyze the technical setup for {stock_symbol}:

Current Price: ${current_price}
52-Week Range: ${week_52_low} - ${week_52_high}

Moving Averages:
- 20-day: ${ma_20}
- 50-day: ${ma_50}
- 200-day: ${ma_200}

Technical Indicators:
- RSI: {rsi}
- MACD: {macd}
- Volume (10-day avg): {volume_avg}

Recent Performance:
- 1 Week: {perf_1w}%
- 1 Month: {perf_1m}%
- 3 Months: {perf_3m}%

Identify:
1. **Current Trend**: Uptrend/Downtrend/Sideways
2. **Key Levels**: Next support and resistance
3. **Signal Strength**: Strong/Moderate/Weak
4. **Entry Strategy**: Buy now/Wait for pullback/Avoid

Provide concise technical assessment with specific price levels.
"""

