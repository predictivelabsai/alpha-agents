# Human Feedback Integration System - Complete Guide

## Overview

The Human Feedback Integration System is a sophisticated mechanism that allows financial experts to improve QualAgent's analysis quality through:

1. **Model Comparison Interface** - Compare outputs from multiple LLMs
2. **Expert Preference Collection** - Capture which analyses experts find most valuable
3. **Training Dataset Generation** - Build datasets for model fine-tuning
4. **Continuous Improvement Loop** - Use feedback to enhance future analyses

## System Architecture

### Components
```
Human Feedback System
├── Expert Profile Management
├── Model Comparison Interface
├── Feedback Collection Engine
├── Training Dataset Generator
├── Performance Analytics
└── Improvement Loop Controller
```

### Database Schema
```sql
-- Expert profiles with bias adjustments
CREATE TABLE expert_profiles (
    expert_id TEXT PRIMARY KEY,
    name TEXT,
    expertise_areas TEXT,  -- JSON array
    experience_years INTEGER,
    credentials TEXT,      -- JSON array
    bias_adjustments TEXT, -- JSON object
    created_date TEXT
);

-- Individual feedback entries
CREATE TABLE feedback_entries (
    feedback_id TEXT PRIMARY KEY,
    expert_id TEXT,
    company_ticker TEXT,
    analysis_session_id TEXT,
    preferred_model TEXT,
    feedback_score INTEGER,  -- 1-5 rating
    feedback_text TEXT,
    timestamp TEXT,
    confidence_level REAL
);

-- Model performance tracking
CREATE TABLE model_performance (
    model_name TEXT,
    company_ticker TEXT,
    expert_id TEXT,
    performance_score REAL,
    feedback_count INTEGER,
    last_updated TEXT,
    PRIMARY KEY (model_name, company_ticker, expert_id)
);
```

## How Human Feedback Improves the Agent

### 1. Model Selection Optimization

**Current Process**:
The system automatically selects the "best" model based on:
- Number of scored components
- Average confidence levels
- Quality of justifications
- Presence of supporting sources

**With Human Feedback**:
```python
# Expert feedback influences model selection
def determine_best_model_with_feedback(self, models_results, company_ticker):
    # Get historical expert preferences for this company/sector
    expert_preferences = self.get_expert_model_preferences(company_ticker)

    # Weight automatic scoring with expert preferences
    for model_name, auto_score in automatic_scores.items():
        expert_weight = expert_preferences.get(model_name, 0.5)
        final_score = (auto_score * 0.7) + (expert_weight * 0.3)
        weighted_scores[model_name] = final_score

    return max(weighted_scores, key=weighted_scores.get)
```

### 2. Consensus Generation Enhancement

**Without Feedback**: Simple averaging of model outputs
**With Feedback**: Weighted consensus based on expert trust scores

```python
def generate_expert_weighted_consensus(self, model_results, company_ticker):
    expert_weights = self.get_model_trust_scores(company_ticker)

    weighted_consensus = {}
    for component in score_components:
        weighted_sum = 0
        total_weight = 0

        for model, results in model_results.items():
            model_weight = expert_weights.get(model, 1.0)
            component_score = results.get(component, {}).get('score', 3.0)
            weighted_sum += component_score * model_weight
            total_weight += model_weight

        weighted_consensus[component] = weighted_sum / total_weight

    return weighted_consensus
```

### 3. Continuous Learning Loop

**Training Dataset Generation**:
```python
def generate_training_dataset(self):
    """Generate training data from expert feedback"""
    dataset = []

    # Get all feedback entries with model comparisons
    feedback_entries = self.get_comparative_feedback()

    for entry in feedback_entries:
        training_example = {
            'input': {
                'company_data': entry['company_info'],
                'analysis_prompt': entry['prompt_used']
            },
            'outputs': {
                'model_a': entry['model_a_result'],
                'model_b': entry['model_b_result']
            },
            'expert_preference': entry['preferred_model'],
            'expert_reasoning': entry['feedback_text'],
            'quality_score': entry['feedback_score']
        }
        dataset.append(training_example)

    return dataset
```

## Expert Feedback Collection Process

### Step 1: Expert Profile Creation

```bash
# Create expert profile
python -c "
from engines.human_feedback_system import HumanFeedbackSystem

hfs = HumanFeedbackSystem()
hfs.create_expert_profile(
    expert_id='senior_analyst_tech',
    name='Sarah Johnson',
    expertise_areas=['Technology', 'Software', 'Semiconductors'],
    experience_years=12,
    credentials=['CFA', 'MBA Finance'],
    bias_adjustments={
        'growth_bias': 0.15,      # Slightly favor growth metrics
        'tech_familiarity': 0.25, # Strong tech sector knowledge
        'risk_aversion': -0.10     # Less risk-averse than average
    }
)
print('Expert profile created successfully')
"
```

### Step 2: Model Comparison Interface

During expert-guided analysis (`--analysis-type expert_guided`):

```
=== EXPERT FEEDBACK INTERFACE ===
Analyzing: Microsoft Corporation (MSFT)
Expert: senior_analyst_tech

Model Comparison Round 1/3

MODEL A: mixtral-8x7b
┌─────────────────────────────────────────────────────────┐
│ Composite Score: 4.2/5.0                               │
│ Confidence: 78%                                         │
│                                                         │
│ KEY STRENGTHS:                                          │
│ • Dominant cloud platform (Azure) with 23% market     │
│   share and growing enterprise adoption                │
│ • Strong competitive moats in enterprise software      │
│ • Excellent management execution under Nadella         │
│                                                         │
│ KEY RISKS:                                              │
│ • Intense competition from AWS and Google Cloud        │
│ • Regulatory scrutiny on market dominance              │
│ • Execution risk on AI integration initiatives         │
│                                                         │
│ COMPONENT SCORES:                                       │
│ • Brand/Monopoly: 4.8/5.0                             │
│ • Network Effects: 4.5/5.0                            │
│ • Growth Drivers: 4.2/5.0                             │
│ • Major Risks: 2.3/5.0                                │
└─────────────────────────────────────────────────────────┘

MODEL B: llama-3-70b
┌─────────────────────────────────────────────────────────┐
│ Composite Score: 3.9/5.0                               │
│ Confidence: 71%                                         │
│                                                         │
│ KEY STRENGTHS:                                          │
│ • Deep enterprise relationships and high switching      │
│   costs create sustainable competitive advantages      │
│ • AI integration across Office 365 suite provides      │
│   differentiation and pricing power                    │
│ • Strong financial position with consistent cash flow  │
│                                                         │
│ KEY RISKS:                                              │
│ • Market saturation in traditional software segments   │
│ • Cloud infrastructure requires massive ongoing capex  │
│ • Talent retention challenges in competitive market    │
│                                                         │
│ COMPONENT SCORES:                                       │
│ • Switching Costs: 4.7/5.0                            │
│ • Technology Moats: 4.3/5.0                           │
│ • Barriers to Entry: 4.1/5.0                          │
│ • Growth Drivers: 3.8/5.0                             │
└─────────────────────────────────────────────────────────┘

Questions:
1. Which analysis provides more actionable insights? (A/B/Both/Neither): A

2. Rate the quality of your preferred analysis (1-5): 4

3. What makes the preferred analysis better?
   > Model A provides more specific market share data and clearer
   > competitive positioning. The risk assessment is more concrete
   > and actionable for investment decisions.

4. Any concerns with the preferred analysis?
   > Could use more detail on AI monetization timeline and specific
   > competitive threats from smaller cloud providers.

5. Confidence in your assessment (1-5): 4

[Continue to next comparison] [Skip remaining] [Finish feedback]
```

### Step 3: Feedback Processing and Storage

```python
# Feedback processing pipeline
def process_expert_feedback(self, feedback_data):
    """Process and store expert feedback"""

    # 1. Store individual feedback entry
    feedback_entry = {
        'feedback_id': generate_uuid(),
        'expert_id': feedback_data['expert_id'],
        'company_ticker': feedback_data['company_ticker'],
        'preferred_model': feedback_data['preferred_model'],
        'feedback_score': feedback_data['quality_rating'],
        'feedback_text': feedback_data['reasoning'],
        'timestamp': datetime.now().isoformat(),
        'confidence_level': feedback_data['confidence'] / 5.0
    }
    self.store_feedback_entry(feedback_entry)

    # 2. Update model performance metrics
    self.update_model_performance(
        model_name=feedback_data['preferred_model'],
        company_ticker=feedback_data['company_ticker'],
        expert_id=feedback_data['expert_id'],
        performance_boost=0.1  # Increase trust score
    )

    # 3. Update non-preferred model (slight decrease)
    for model in feedback_data['compared_models']:
        if model != feedback_data['preferred_model']:
            self.update_model_performance(
                model_name=model,
                expert_id=feedback_data['expert_id'],
                performance_boost=-0.05
            )

    # 4. Trigger retraining if enough feedback collected
    if self.get_feedback_count() % 50 == 0:
        self.trigger_model_retraining()
```

## How Feedback Improves Future Analyses

### 1. Dynamic Model Weighting

**Before Feedback** (Equal weighting):
```python
model_weights = {
    'mixtral-8x7b': 1.0,
    'llama-3-70b': 1.0,
    'qwen2-72b': 1.0,
    'llama-3.1-70b': 1.0,
    'deepseek-coder-33b': 1.0
}
```

**After Expert Feedback** (Performance-based weighting):
```python
# Weights adjusted based on expert preferences
model_weights = {
    'mixtral-8x7b': 1.3,      # Preferred by experts 65% of time
    'llama-3-70b': 1.1,      # Preferred by experts 55% of time
    'qwen2-72b': 0.8,        # Preferred by experts 35% of time
    'llama-3.1-70b': 1.2,    # Preferred by experts 60% of time
    'deepseek-coder-33b': 0.7 # Preferred by experts 30% of time
}
```

### 2. Sector-Specific Model Selection

```python
def get_sector_optimized_models(self, company_sector):
    """Select best models based on sector-specific feedback"""

    sector_performance = self.get_sector_model_performance(company_sector)

    # For Technology companies
    if company_sector in ['Technology', 'Software', 'Semiconductors']:
        return [
            'mixtral-8x7b',    # Best for tech analysis based on feedback
            'llama-3.1-70b',   # Strong technical depth
            'llama-3-70b'      # Good business model analysis
        ]

    # For Healthcare companies
    elif company_sector in ['Healthcare', 'Pharmaceuticals', 'Biotech']:
        return [
            'llama-3-70b',     # Best regulatory analysis
            'qwen2-72b',       # Strong R&D assessment
            'mixtral-8x7b'     # Good risk evaluation
        ]
```

### 3. Expert-Guided Component Emphasis

```python
def adjust_scoring_weights_by_feedback(self, base_weights, company_ticker):
    """Adjust component weights based on expert feedback patterns"""

    expert_emphasis = self.analyze_expert_focus_areas(company_ticker)

    adjusted_weights = base_weights.copy()

    # If experts consistently highlight certain components for this sector
    for component, emphasis_factor in expert_emphasis.items():
        if emphasis_factor > 1.2:  # Strong expert emphasis
            adjusted_weights[component] *= emphasis_factor
        elif emphasis_factor < 0.8:  # Expert de-emphasis
            adjusted_weights[component] *= emphasis_factor

    return adjusted_weights.normalize_weights()
```

## Advanced Feedback Analytics

### 1. Expert Reliability Scoring

```python
def calculate_expert_reliability(self, expert_id):
    """Calculate expert reliability based on prediction accuracy"""

    expert_feedback = self.get_expert_feedback_history(expert_id)
    correct_predictions = 0
    total_predictions = 0

    for feedback in expert_feedback:
        if self.has_market_outcome(feedback['company_ticker'], feedback['timestamp']):
            market_outcome = self.get_market_outcome(
                feedback['company_ticker'],
                feedback['timestamp']
            )

            predicted_direction = self.score_to_direction(feedback['preferred_score'])
            actual_direction = self.market_to_direction(market_outcome)

            if predicted_direction == actual_direction:
                correct_predictions += 1
            total_predictions += 1

    reliability_score = correct_predictions / total_predictions if total_predictions > 0 else 0.5
    return reliability_score
```

### 2. Consensus Drift Detection

```python
def detect_consensus_drift(self, company_ticker):
    """Detect when expert opinions diverge from model consensus"""

    recent_analyses = self.get_recent_analyses(company_ticker, days=90)
    expert_scores = []
    model_scores = []

    for analysis in recent_analyses:
        if analysis.get('expert_feedback'):
            expert_scores.append(analysis['expert_feedback']['adjusted_score'])
            model_scores.append(analysis['original_composite_score'])

    if len(expert_scores) >= 5:
        drift = np.mean(expert_scores) - np.mean(model_scores)

        if abs(drift) > 0.5:  # Significant drift
            self.trigger_model_recalibration(company_ticker, drift)

    return drift
```

### 3. Training Dataset Quality Assessment

```python
def assess_training_quality(self):
    """Assess quality of generated training dataset"""

    dataset = self.generate_training_dataset()

    quality_metrics = {
        'size': len(dataset),
        'expert_diversity': len(set(d['expert_id'] for d in dataset)),
        'sector_coverage': len(set(d['company_sector'] for d in dataset)),
        'consensus_rate': sum(1 for d in dataset if d['high_consensus']) / len(dataset),
        'avg_confidence': np.mean([d['expert_confidence'] for d in dataset])
    }

    # Quality gates
    is_high_quality = (
        quality_metrics['size'] >= 100 and
        quality_metrics['expert_diversity'] >= 3 and
        quality_metrics['sector_coverage'] >= 5 and
        quality_metrics['avg_confidence'] >= 0.7
    )

    return quality_metrics, is_high_quality
```

## Feedback-Driven Model Improvement Process

### Phase 1: Feedback Collection (Ongoing)
1. **Expert Interface**: Collect preferences during analyses
2. **Feedback Storage**: Store in structured database
3. **Quality Validation**: Verify feedback consistency

### Phase 2: Pattern Analysis (Weekly)
1. **Model Performance**: Track preference patterns
2. **Sector Analysis**: Identify sector-specific strengths
3. **Expert Reliability**: Update expert trust scores

### Phase 3: System Updates (Monthly)
1. **Weight Adjustments**: Update model consensus weights
2. **Selection Logic**: Optimize model selection algorithms
3. **Component Emphasis**: Adjust scoring component weights

### Phase 4: Model Retraining (Quarterly)
1. **Dataset Generation**: Create training data from feedback
2. **Fine-tuning**: Improve model prompt engineering
3. **Validation**: Test improvements on held-out dataset

## Implementation Examples

### Getting Feedback History
```bash
# View recent expert feedback
python -c "
from engines.human_feedback_system import HumanFeedbackSystem
hfs = HumanFeedbackSystem()

recent = hfs.get_recent_feedback(limit=10)
print('Recent Expert Feedback:')
for feedback in recent:
    print(f'  {feedback[\"timestamp\"][:10]}: {feedback[\"company_ticker\"]} - '
          f'Preferred {feedback[\"preferred_model\"]} (Score: {feedback[\"feedback_score\"]}/5)')
"
```

### Analyzing Model Performance Trends
```bash
# Check model performance by sector
python -c "
from engines.human_feedback_system import HumanFeedbackSystem
hfs = HumanFeedbackSystem()

performance = hfs.get_model_performance_by_sector('Technology')
print('Technology Sector Model Performance:')
for model, score in performance.items():
    print(f'  {model}: {score:.2f}/5.0')
"
```

### Generating Training Dataset
```bash
# Create training dataset for model improvement
python -c "
from engines.human_feedback_system import HumanFeedbackSystem
hfs = HumanFeedbackSystem()

dataset = hfs.generate_training_dataset()
print(f'Training dataset generated: {len(dataset)} examples')

# Save dataset
import json
with open('training_dataset.json', 'w') as f:
    json.dump(dataset, f, indent=2)
print('Dataset saved to training_dataset.json')
"
```

## Best Practices for Expert Feedback

### 1. Expert Guidelines
- **Consistency**: Provide feedback based on clear, consistent criteria
- **Specificity**: Include specific reasoning for preferences
- **Objectivity**: Focus on analytical quality, not personal investment views
- **Coverage**: Provide feedback across different sectors and market conditions

### 2. Feedback Quality
- **High-Quality Feedback**: Specific, actionable, consistent with expert expertise
- **Medium-Quality Feedback**: Clear preference but limited reasoning
- **Low-Quality Feedback**: Inconsistent or contradictory preferences

### 3. System Calibration
- **Regular Reviews**: Monthly review of feedback patterns and system performance
- **Expert Validation**: Quarterly expert review sessions
- **Market Validation**: Annual comparison of expert preferences vs. market outcomes

## Continuous Improvement Cycle

```
Expert Feedback → Pattern Analysis → System Updates → Performance Measurement
       ↑                                                        ↓
Market Validation ← Quality Assessment ← Model Retraining ← Dataset Generation
```

The human feedback system creates a continuous improvement loop that makes QualAgent increasingly accurate and valuable for investment decision-making.

---

*This system transforms QualAgent from a static analysis tool into a continuously learning, expert-guided investment research platform.*