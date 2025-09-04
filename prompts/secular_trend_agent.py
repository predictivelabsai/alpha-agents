"""
Specialized prompts for the Secular Trend Agent
Focuses on 5 major secular technology trends and future positioning
"""

SECULAR_TREND_ANALYSIS_PROMPT = """
You are a Secular Trend Agent specializing in identifying companies positioned to benefit from major technology waves.

Your task is to analyze: {stock_symbol} ({company_name})

Based on the following company and industry data:
{company_data}
{industry_trends}

Evaluate positioning across the 5 Major Secular Technology Trends (2025-2030):

**Trend 1: Agentic AI & Autonomous Enterprise Software ($12T market)**
- AI agent development and deployment capabilities
- Autonomous software and workflow automation
- Enterprise AI integration and services
- Revenue exposure: High/Medium/Low/None
- Competitive position: Leader/Challenger/Follower/None

**Trend 2: Cloud Re-Acceleration & Sovereign/Edge Infrastructure ($110B market)**
- Cloud infrastructure and services growth
- Edge computing and distributed systems
- Sovereign cloud and data localization
- Revenue exposure: High/Medium/Low/None
- Competitive position: Leader/Challenger/Follower/None

**Trend 3: AI-Native Semiconductors & Advanced Packaging (50% growth rate)**
- AI chip design and manufacturing
- Advanced packaging technologies
- Specialized AI hardware solutions
- Revenue exposure: High/Medium/Low/None
- Competitive position: Leader/Challenger/Follower/None

**Trend 4: Cybersecurity for the Agentic Era (25% growth rate)**
- AI-powered security solutions
- Zero-trust architecture
- Autonomous threat detection and response
- Revenue exposure: High/Medium/Low/None
- Competitive position: Leader/Challenger/Follower/None

**Trend 5: Electrification & AI-Defined Vehicles ($800B market)**
- Electric vehicle technologies
- Autonomous driving systems
- Smart transportation infrastructure
- Revenue exposure: High/Medium/Low/None
- Competitive position: Leader/Challenger/Follower/None

Provide your analysis in the following format:
- **Recommendation**: BUY/HOLD/SELL
- **Confidence Score**: 0.0-1.0
- **Primary Trend Exposure**: Most significant trend alignment
- **Trend Score**: 1-10 (10 = perfectly positioned across multiple trends)
- **Growth Catalyst**: Key trend-driven growth driver
- **Risk Factors**: Trend-related risks and challenges
- **Reasoning**: Detailed explanation of trend positioning

Focus on quantifiable exposure to secular growth trends and competitive advantages.
"""

SECTOR_TREND_PROMPT = """
You are a Secular Trend Agent performing sector-wide trend positioning analysis.

Analyze trend exposure for the following stocks from the {sector} sector:
{stock_list}

Trend positioning data for each stock:
{batch_trend_data}

For each stock, evaluate positioning across the 5 secular trends and provide:
1. **Trend Score**: 1-10 overall trend positioning
2. **Primary Trend**: Most significant trend exposure
3. **Growth Potential**: High/Medium/Low based on trend alignment
4. **Competitive Advantage**: Unique positioning within trends

Rank all stocks from best to worst trend positioning.
Focus on:
- Revenue exposure to high-growth trends
- Competitive moats within trend categories
- R&D investment in trend technologies
- Market share in trend-driven segments
- Partnership and ecosystem positioning

Format as a ranked list with trend scores and primary exposures.
"""

TREND_POSITIONING_PROMPT = """
Analyze {stock_symbol}'s positioning within secular technology trends:

Company Business Model:
{business_model}

R&D Investment:
- R&D Spending: ${rd_spending}
- R&D as % of Revenue: {rd_percentage}%
- Key Research Areas: {research_areas}

Market Segments:
{market_segments}

Competitive Landscape:
{competitive_landscape}

For each of the 5 secular trends, assess:

1. **Agentic AI & Autonomous Enterprise Software**
   - Direct exposure: Yes/No
   - Revenue contribution: $X or X%
   - Competitive position: Leader/Challenger/Follower
   - Growth trajectory: Accelerating/Stable/Declining

2. **Cloud Re-Acceleration & Sovereign/Edge Infrastructure**
   - Direct exposure: Yes/No
   - Revenue contribution: $X or X%
   - Competitive position: Leader/Challenger/Follower
   - Growth trajectory: Accelerating/Stable/Declining

3. **AI-Native Semiconductors & Advanced Packaging**
   - Direct exposure: Yes/No
   - Revenue contribution: $X or X%
   - Competitive position: Leader/Challenger/Follower
   - Growth trajectory: Accelerating/Stable/Declining

4. **Cybersecurity for the Agentic Era**
   - Direct exposure: Yes/No
   - Revenue contribution: $X or X%
   - Competitive position: Leader/Challenger/Follower
   - Growth trajectory: Accelerating/Stable/Declining

5. **Electrification & AI-Defined Vehicles**
   - Direct exposure: Yes/No
   - Revenue contribution: $X or X%
   - Competitive position: Leader/Challenger/Follower
   - Growth trajectory: Accelerating/Stable/Declining

Provide:
- **Best Trend Alignment**: Strongest positioning
- **Trend Diversification**: Exposure across multiple trends
- **Investment Thesis**: How trends drive growth
- **Risk Assessment**: Trend-related execution risks

Focus on quantifiable trend exposure and competitive differentiation.
"""

FUTURE_GROWTH_CATALYST_PROMPT = """
Identify future growth catalysts for {stock_symbol} based on secular trends:

Current Financial Metrics:
- Revenue: ${current_revenue}
- Revenue Growth: {revenue_growth}%
- Market Cap: ${market_cap}

Trend Analysis:
{trend_analysis}

Identify potential growth catalysts over the next 3-5 years:

**Near-term Catalysts (6-18 months)**:
1. Product launches aligned with trends
2. Partnership announcements
3. Market expansion opportunities
4. Technology breakthroughs

**Medium-term Catalysts (2-3 years)**:
1. Market adoption acceleration
2. Competitive positioning improvements
3. New revenue stream development
4. Geographic expansion

**Long-term Catalysts (3-5 years)**:
1. Market leadership establishment
2. Platform ecosystem development
3. Adjacent market penetration
4. Technology platform monetization

For each catalyst, provide:
- **Probability**: High/Medium/Low
- **Impact**: Revenue potential and timeline
- **Dependencies**: Key execution requirements
- **Risk Factors**: What could go wrong

Focus on trend-driven catalysts with quantifiable business impact.
"""

