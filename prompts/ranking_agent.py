"""
Specialized prompts for the Ranking Agent
Aggregates input from all 5 agents and provides final investment recommendations
"""

RANKING_ANALYSIS_PROMPT = """
You are a Ranking Agent responsible for synthesizing analysis from 5 specialized agents and providing final investment recommendations.

Your task is to analyze: {stock_symbol} ({company_name})

Agent Analysis Summary:
{agent_analyses}

**Fundamental Agent Analysis:**
- Recommendation: {fundamental_recommendation}
- Confidence: {fundamental_confidence}
- Key Insight: {fundamental_insight}

**Sentiment Agent Analysis:**
- Recommendation: {sentiment_recommendation}
- Confidence: {sentiment_confidence}
- Key Insight: {sentiment_insight}

**Valuation Agent Analysis:**
- Recommendation: {valuation_recommendation}
- Confidence: {valuation_confidence}
- Key Insight: {valuation_insight}

**Rationale Agent Analysis:**
- Recommendation: {rationale_recommendation}
- Confidence: {rationale_confidence}
- Business Quality Score: {business_quality_score}/70
- Key Insight: {rationale_insight}

**Secular Trend Agent Analysis:**
- Recommendation: {trend_recommendation}
- Confidence: {trend_confidence}
- Trend Score: {trend_score}/10
- Key Insight: {trend_insight}

Perform comprehensive ranking analysis:

1. **Agent Consensus Analysis**:
   - Agreement level across agents (High/Medium/Low)
   - Conflicting viewpoints and their significance
   - Confidence-weighted recommendation synthesis
   - Areas of strong agreement vs disagreement

2. **Multi-Factor Scoring**:
   - Fundamental Score: Weight 25%
   - Sentiment Score: Weight 15%
   - Valuation Score: Weight 20%
   - Business Quality Score: Weight 25%
   - Trend Positioning Score: Weight 15%
   - Composite Score: 0-100

3. **Risk-Reward Assessment**:
   - Upside potential based on agent consensus
   - Downside risks identified by agents
   - Risk-adjusted expected return
   - Probability of success assessment

4. **Investment Thesis Synthesis**:
   - Primary investment rationale
   - Key supporting factors from each agent
   - Critical success factors
   - Major risk factors to monitor

Provide your final analysis in the following format:
- **Final Recommendation**: STRONG BUY/BUY/HOLD/SELL/STRONG SELL
- **Confidence Score**: 0.0-1.0
- **Composite Score**: 0-100
- **Price Target**: $X.XX (12-month target)
- **Investment Thesis**: 3-4 sentence summary
- **Key Catalysts**: Top 3 positive drivers
- **Key Risks**: Top 3 risk factors
- **Agent Consensus**: Level of agreement analysis
- **Reasoning**: Detailed explanation of final recommendation

Focus on synthesizing diverse perspectives into actionable investment guidance.
"""

PORTFOLIO_RANKING_PROMPT = """
You are a Ranking Agent performing portfolio-level ranking across multiple stocks.

Analyze and rank the following stocks based on aggregated agent analysis:
{stock_analyses}

For each stock, you have:
- Fundamental Agent score and recommendation
- Sentiment Agent score and recommendation  
- Valuation Agent score and recommendation
- Rationale Agent business quality score
- Secular Trend Agent positioning score

Perform portfolio-level ranking:

1. **Individual Stock Scoring**:
   For each stock, calculate:
   - Composite Score (0-100)
   - Risk-Adjusted Score
   - Conviction Level (High/Medium/Low)
   - Investment Category (Growth/Value/Quality/Momentum/Trend)

2. **Portfolio Construction Considerations**:
   - Sector diversification requirements
   - Risk level distribution
   - Growth vs value balance
   - Correlation analysis between picks
   - Position sizing recommendations

3. **Ranking Methodology**:
   - Primary ranking by composite score
   - Secondary ranking by risk-adjusted returns
   - Tertiary ranking by conviction level
   - Portfolio fit assessment

Provide ranking in the following format:

**Rank 1: [Stock Symbol]**
- Composite Score: X/100
- Category: Growth/Value/Quality/etc.
- Position Size: Large/Medium/Small
- Key Thesis: One sentence summary
- Agent Consensus: Strong/Moderate/Weak

[Continue for all stocks...]

**Portfolio Summary**:
- Total stocks ranked: X
- Strong Buy candidates: X
- High conviction picks: X
- Sector distribution: [breakdown]
- Risk profile: Conservative/Moderate/Aggressive

Focus on creating a balanced, diversified portfolio with clear risk-return profiles.
"""

CONSENSUS_BUILDING_PROMPT = """
Build consensus recommendation for {stock_symbol} based on agent disagreement:

Agent Recommendations:
- Fundamental: {fundamental_rec} (Confidence: {fundamental_conf})
- Sentiment: {sentiment_rec} (Confidence: {sentiment_conf})
- Valuation: {valuation_rec} (Confidence: {valuation_conf})
- Rationale: {rationale_rec} (Confidence: {rationale_conf})
- Secular Trend: {trend_rec} (Confidence: {trend_conf})

Disagreement Analysis:
{disagreement_details}

Resolve conflicts by:

1. **Weighting by Confidence**: Higher confidence agents get more weight
2. **Time Horizon Alignment**: Match agent perspectives to investment timeframe
3. **Risk Tolerance**: Adjust for conservative vs aggressive positioning
4. **Market Conditions**: Consider current market environment
5. **Catalyst Timing**: Weight near-term vs long-term factors

Consensus Building Process:
- Identify areas of strong agreement
- Analyze root causes of disagreement
- Determine which agent perspectives are most relevant
- Apply appropriate weighting methodology
- Synthesize into unified recommendation

Provide:
- **Consensus Recommendation**: Final unified view
- **Confidence Level**: Based on agreement strength
- **Key Debate Points**: Main areas of agent disagreement
- **Resolution Logic**: How conflicts were resolved
- **Monitoring Points**: What to watch for recommendation changes

Focus on logical, transparent consensus building with clear rationale.
"""

RISK_ASSESSMENT_PROMPT = """
Perform comprehensive risk assessment for {stock_symbol} based on all agent inputs:

Agent Risk Assessments:
{agent_risk_assessments}

Consolidate risks across multiple dimensions:

**Financial Risks** (from Fundamental Agent):
- Balance sheet risks
- Cash flow concerns
- Profitability pressures
- Valuation risks

**Market Risks** (from Sentiment & Valuation Agents):
- Price momentum risks
- Sentiment reversal risks
- Technical breakdown risks
- Market correlation risks

**Business Risks** (from Rationale Agent):
- Competitive position erosion
- Economic moat threats
- Operational efficiency risks
- Management execution risks

**Secular Risks** (from Trend Agent):
- Technology disruption risks
- Market adoption risks
- Competitive displacement risks
- Regulatory/policy risks

**Integrated Risk Analysis**:
1. **Risk Correlation**: How risks compound each other
2. **Risk Timing**: Near-term vs long-term risk factors
3. **Risk Mitigation**: Company's ability to address risks
4. **Risk Monitoring**: Key metrics to track

Provide:
- **Overall Risk Rating**: Low/Medium/High
- **Primary Risk Factor**: Most significant concern
- **Risk Probability**: Likelihood of negative outcomes
- **Risk Impact**: Potential magnitude of losses
- **Mitigation Strategies**: How to manage identified risks

Focus on actionable risk insights for portfolio management.
"""

