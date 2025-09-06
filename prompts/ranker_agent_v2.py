"""
Ranker Agent Prompts - Lohusalu Capital Management
Specialized prompts for final scoring and investment recommendations
"""

INVESTMENT_THESIS_PROMPT = """
You are an expert investment analyst providing comprehensive investment thesis and recommendations.

Generate a comprehensive investment thesis for {company_name} ({ticker}) based on the following analysis:

FUNDAMENTAL ANALYSIS:
- Overall Fundamental Score: {fundamental_score:.1f}/100
- Growth Score: {growth_score:.1f}/100 (Revenue Growth: {revenue_growth:.1f}%, Earnings Growth: {earnings_growth:.1f}%)
- Profitability Score: {profitability_score:.1f}/100 (ROE: {roe:.1f}%, ROIC: {roic:.1f}%, Margins: {profit_margin:.1f}%)
- Financial Strength Score: {financial_strength_score:.1f}/100 (Current Ratio: {current_ratio:.2f}, Debt/Equity: {debt_to_equity:.2f})
- Valuation Score: {valuation_score:.1f}/100 (P/E: {pe_ratio:.1f}, Upside Potential: {upside_potential:.1f}%)

QUALITATIVE ANALYSIS:
- Overall Qualitative Score: {qualitative_score:.1f}/100
- Economic Moat: {moat_strength} ({moat_score:.1f}/100)
- Market Sentiment: {sentiment} ({sentiment_score:.1f}/100)
- Secular Trends: {trend_alignment} ({trends_score:.1f}/100)
- Competitive Position: {market_position} ({competitive_score:.1f}/100)

COMPOSITE SCORE: {composite_score:.1f}/100
INVESTMENT GRADE: {investment_grade}

Provide a comprehensive investment thesis that includes:

## Investment Strengths
Identify 2-3 key strengths that make this an attractive investment:
- Focus on the highest-scoring areas from the analysis
- Highlight sustainable competitive advantages
- Emphasize strong financial metrics and market position

## Investment Risks
Identify 2-3 main risks that could impact the investment:
- Consider areas with lower scores
- Assess market, operational, and financial risks
- Evaluate competitive threats and industry challenges

## Key Catalysts
Identify factors that could drive outperformance:
- Growth drivers and expansion opportunities
- Product launches or strategic initiatives
- Market trends and secular tailwinds
- Potential re-rating opportunities

## Investment Recommendation
Provide a clear conclusion based on the analysis:
- Synthesize the quantitative and qualitative factors
- Explain why the composite score and grade are justified
- Recommend appropriate position sizing and time horizon
- Address the risk-reward profile

Write a professional, balanced analysis suitable for investment decision-making. Focus on the most material factors and provide specific reasoning based on the quantitative and qualitative analysis. Ensure the recommendation aligns with the composite score and investment grade.

Structure your response with clear sections and bullet points for easy readability.
"""

KEY_FACTORS_EXTRACTION_PROMPT = """
You are an expert at extracting key investment factors from analysis.

Extract key investment factors from the following investment thesis:

{investment_thesis}

Identify and categorize the most important factors mentioned in the thesis:

**Key Strengths**: The main competitive advantages and positive factors that support the investment case
**Key Risks**: The primary concerns and potential negative factors that could impact performance  
**Catalysts**: Specific events, trends, or developments that could drive outperformance

Provide the output in JSON format:
{{
    "key_strengths": ["<strength1>", "<strength2>", "<strength3>"],
    "key_risks": ["<risk1>", "<risk2>", "<risk3>"],
    "catalysts": ["<catalyst1>", "<catalyst2>", "<catalyst3>"]
}}

Guidelines:
- Focus on the most important and actionable factors
- Keep descriptions concise but specific
- Prioritize factors that are most likely to impact investment performance
- Ensure factors are distinct and non-overlapping
- Extract exactly 3 items for each category when possible

Base your extraction on the specific content and analysis provided in the investment thesis.
"""

PORTFOLIO_THESIS_PROMPT = """
You are an expert portfolio manager creating comprehensive portfolio recommendations.

Generate a portfolio-level investment thesis based on the following selected investments:

PORTFOLIO COMPOSITION:
{portfolio_details}

PORTFOLIO METRICS:
- Number of Positions: {num_positions}
- Average Composite Score: {avg_score:.1f}/100
- Expected Return Potential: {expected_return:.1f}%
- Risk Profile: {risk_profile}
- Diversification Score: {diversification_score:.1f}/100
- Overall Conviction: {overall_conviction}

SECTOR ALLOCATION:
{sector_allocation}

TOP HOLDINGS:
{top_holdings}

Provide a comprehensive portfolio thesis that addresses:

## Portfolio Strategy
Explain the overall investment strategy and approach:
- Investment philosophy and selection criteria
- Risk-return objectives
- Time horizon and investment style

## Sector Allocation Rationale
Justify the sector allocation decisions:
- Why these sectors were emphasized
- How the allocation aligns with market opportunities
- Diversification benefits and risk management

## Key Portfolio Themes
Identify 3-4 major investment themes:
- Secular trends driving the portfolio
- Common characteristics of selected companies
- Thematic exposure and growth drivers

## Risk Management
Address portfolio-level risks and mitigation:
- Concentration risks and diversification
- Market sensitivity and correlation
- Downside protection measures

## Expected Outcomes
Provide realistic expectations:
- Return potential and time horizon
- Key performance drivers
- Scenarios for outperformance and underperformance

## Implementation Considerations
Practical aspects of portfolio construction:
- Position sizing recommendations
- Monitoring and rebalancing approach
- Entry and exit strategies

Write a professional portfolio thesis suitable for institutional investors. Focus on the strategic rationale and risk-adjusted return potential.
"""

SCORING_METHODOLOGY_EXPLANATION_PROMPT = """
Explain the scoring methodology used for {ticker} investment analysis:

SCORING BREAKDOWN:
- Composite Score: {composite_score:.1f}/100
- Investment Grade: {investment_grade}

COMPONENT ANALYSIS:
Fundamental Analysis ({fundamental_weight}% weight):
- Growth Score: {growth_score:.1f}/100 ({growth_weight}% of fundamental)
- Profitability Score: {profitability_score:.1f}/100 ({profitability_weight}% of fundamental)  
- Financial Strength Score: {financial_strength_score:.1f}/100 ({financial_strength_weight}% of fundamental)
- Valuation Score: {valuation_score:.1f}/100 ({valuation_weight}% of fundamental)

Qualitative Analysis ({qualitative_weight}% weight):
- Economic Moat Score: {moat_score:.1f}/100 ({moat_weight}% of qualitative)
- Sentiment Score: {sentiment_score:.1f}/100 ({sentiment_weight}% of qualitative)
- Secular Trends Score: {trends_score:.1f}/100 ({trends_weight}% of qualitative)
- Competitive Position Score: {competitive_score:.1f}/100 ({competitive_weight}% of qualitative)

Explain:
1. **Methodology Overview**: How the composite score is calculated
2. **Component Weighting**: Rationale for the weighting scheme
3. **Score Interpretation**: What each score level means
4. **Grade Assignment**: How investment grades are determined
5. **Strengths and Limitations**: What the scoring captures and what it might miss

Provide a clear, educational explanation of how the quantitative scoring translates to investment recommendations.
"""

RISK_ASSESSMENT_PROMPT = """
Conduct a comprehensive risk assessment for {company_name} ({ticker}):

FINANCIAL RISK FACTORS:
- Debt-to-Equity Ratio: {debt_to_equity:.2f}
- Current Ratio: {current_ratio:.2f}
- Interest Coverage: {interest_coverage:.1f}x
- Free Cash Flow: ${free_cash_flow:,.0f}

MARKET RISK FACTORS:
- Market Capitalization: ${market_cap:,.0f}
- Beta: {beta:.2f}
- Average Daily Volume: {avg_volume:,.0f}
- Price Volatility: {volatility:.1f}%

BUSINESS RISK FACTORS:
- Economic Moat Strength: {moat_strength}
- Competitive Position: {competitive_position}
- Industry Attractiveness: {industry_attractiveness}
- Regulatory Environment: {regulatory_environment}

Assess risk across multiple dimensions:

## Financial Risk Assessment
Evaluate balance sheet strength and financial stability:
- Leverage and debt service capability
- Liquidity and working capital management
- Cash flow generation and sustainability
- Financial flexibility and covenant compliance

## Market Risk Assessment  
Analyze market-related risks:
- Size and liquidity considerations
- Volatility and correlation factors
- Market sentiment and momentum risks
- Sector and style factor exposure

## Business Risk Assessment
Evaluate operational and strategic risks:
- Competitive threats and market share erosion
- Technology disruption and obsolescence
- Regulatory changes and compliance costs
- Management execution and governance

## Overall Risk Rating
Provide an overall risk assessment:
- Primary risk factors and their likelihood
- Risk mitigation factors and strengths
- Appropriate risk rating (Low/Medium/High)
- Risk-adjusted return considerations

Focus on material risks that could significantly impact investment performance.
"""

CONVICTION_LEVEL_ASSESSMENT_PROMPT = """
Assess the conviction level for {company_name} ({ticker}) investment recommendation:

SCORE CONSISTENCY ANALYSIS:
- Composite Score: {composite_score:.1f}/100
- Score Standard Deviation: {score_std:.1f}
- Score Range: {score_min:.1f} - {score_max:.1f}

COMPONENT SCORES:
- Growth: {growth_score:.1f}/100
- Profitability: {profitability_score:.1f}/100
- Financial Strength: {financial_strength_score:.1f}/100
- Valuation: {valuation_score:.1f}/100
- Economic Moat: {moat_score:.1f}/100
- Sentiment: {sentiment_score:.1f}/100
- Secular Trends: {trends_score:.1f}/100
- Competitive Position: {competitive_score:.1f}/100

QUALITY INDICATORS:
- Data Quality: {data_quality}
- Analysis Depth: {analysis_depth}
- Source Reliability: {source_reliability}

Determine conviction level based on:

## Score Consistency
Evaluate how consistent scores are across components:
- High consistency suggests strong conviction
- Mixed scores may indicate moderate conviction
- Highly variable scores suggest lower conviction

## Analytical Confidence
Assess confidence in the underlying analysis:
- Quality and depth of fundamental analysis
- Reliability of qualitative research
- Completeness of information available

## Risk-Adjusted Attractiveness
Consider the risk-reward profile:
- Upside potential relative to downside risk
- Probability of achieving expected returns
- Margin of safety in the investment case

## Conviction Recommendation
Provide final conviction assessment:
- High Conviction: Strong, consistent case with high confidence
- Medium Conviction: Good case with some uncertainties
- Low Conviction: Weak or highly uncertain case

Explain the rationale for the conviction level and how it should influence position sizing.
"""

TIME_HORIZON_ASSESSMENT_PROMPT = """
Determine the appropriate investment time horizon for {company_name} ({ticker}):

BUSINESS CHARACTERISTICS:
- Economic Moat Strength: {moat_strength}
- Moat Sustainability: {moat_sustainability}
- Competitive Position: {competitive_position}
- Industry Life Cycle: {industry_lifecycle}

GROWTH PROFILE:
- Growth Stage: {growth_stage}
- Revenue Growth Rate: {revenue_growth:.1f}%
- Market Opportunity: {market_opportunity}
- Scalability: {scalability}

SECULAR TRENDS:
- Primary Trends: {primary_trends}
- Trend Time Horizon: {trend_horizon}
- Trend Strength: {trend_strength}
- Disruption Risk: {disruption_risk}

VALUATION CONSIDERATIONS:
- Current Valuation: {current_valuation}
- Upside Potential: {upside_potential:.1f}%
- Catalyst Timeline: {catalyst_timeline}
- Mean Reversion Risk: {mean_reversion_risk}

Assess appropriate time horizon based on:

## Business Durability
Consider how long competitive advantages can be sustained:
- Wide moats suggest longer investment horizons
- Narrow or no moats may require shorter horizons
- Industry dynamics and disruption potential

## Growth Trajectory
Evaluate the growth profile and sustainability:
- High-growth companies may need time to compound
- Mature companies may offer shorter-term opportunities
- Cyclical considerations and timing

## Secular Trend Alignment
Consider the timeline for secular trends to play out:
- Long-term trends support longer holding periods
- Short-term trends may require tactical timing
- Trend inflection points and adoption curves

## Valuation and Catalysts
Assess when value recognition might occur:
- Deeply undervalued stocks may need time for recognition
- Catalyst-driven situations may have shorter timelines
- Market cycle considerations

## Time Horizon Recommendation
Provide specific time horizon guidance:
- Long-term (5+ years): Strong moats, secular tailwinds, patient capital
- Medium-term (2-5 years): Good fundamentals, moderate catalysts
- Short-term (1-2 years): Tactical opportunities, catalyst-driven

Explain the rationale and key factors driving the time horizon assessment.
"""

