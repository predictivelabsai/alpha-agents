-- Create qualitative_analysis table for storing QualAgent results
CREATE TABLE IF NOT EXISTS qualitative_analysis (
    -- Primary key and identification
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(50) NOT NULL,
    company_ticker VARCHAR(10) NOT NULL,
    analysis_timestamp BIGINT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Metadata
    analysis_type VARCHAR(50),
    models_used TEXT[], -- Array of model names
    successful_models TEXT[], -- Array of successful model names
    failed_models TEXT[], -- Array of failed model names
    consensus_method VARCHAR(50),
    execution_time_seconds DECIMAL(10,3),
    total_cost_usd DECIMAL(10,6),
    concurrent_execution BOOLEAN,
    
    -- Company information
    company_name VARCHAR(255),
    company_subsector VARCHAR(100),
    company_market_cap_usd BIGINT,
    company_employees INTEGER,
    company_founded_year INTEGER,
    company_headquarters VARCHAR(255),
    company_description TEXT,
    company_website VARCHAR(255),
    company_status VARCHAR(50),
    company_id VARCHAR(100),
    company_created_at TIMESTAMP,
    company_updated_at TIMESTAMP,
    
    -- Composite scores
    composite_score DECIMAL(5,3),
    composite_confidence DECIMAL(5,3),
    composite_components_count INTEGER,
    
    -- Competitive Moats Scores
    moat_brand_monopoly_score DECIMAL(5,3),
    moat_brand_monopoly_confidence DECIMAL(5,3),
    moat_brand_monopoly_justification TEXT,
    
    moat_barriers_to_entry_score DECIMAL(5,3),
    moat_barriers_to_entry_confidence DECIMAL(5,3),
    moat_barriers_to_entry_justification TEXT,
    
    moat_economies_of_scale_score DECIMAL(5,3),
    moat_economies_of_scale_confidence DECIMAL(5,3),
    moat_economies_of_scale_justification TEXT,
    
    moat_network_effects_score DECIMAL(5,3),
    moat_network_effects_confidence DECIMAL(5,3),
    moat_network_effects_justification TEXT,
    
    moat_switching_costs_score DECIMAL(5,3),
    moat_switching_costs_confidence DECIMAL(5,3),
    moat_switching_costs_justification TEXT,
    
    -- Strategic Insights Scores
    competitive_differentiation_score DECIMAL(5,3),
    competitive_differentiation_confidence DECIMAL(5,3),
    competitive_differentiation_justification TEXT,
    
    technology_moats_score DECIMAL(5,3),
    technology_moats_confidence DECIMAL(5,3),
    technology_moats_justification TEXT,
    
    market_timing_score DECIMAL(5,3),
    market_timing_confidence DECIMAL(5,3),
    market_timing_justification TEXT,
    
    management_quality_score DECIMAL(5,3),
    management_quality_confidence DECIMAL(5,3),
    management_quality_justification TEXT,
    
    key_growth_drivers_score DECIMAL(5,3),
    key_growth_drivers_confidence DECIMAL(5,3),
    key_growth_drivers_justification TEXT,
    
    transformation_potential_score DECIMAL(5,3),
    transformation_potential_confidence DECIMAL(5,3),
    transformation_potential_justification TEXT,
    
    platform_expansion_score DECIMAL(5,3),
    platform_expansion_confidence DECIMAL(5,3),
    platform_expansion_justification TEXT,
    
    -- Risk Factors Scores
    major_risk_factors_score DECIMAL(5,3),
    major_risk_factors_confidence DECIMAL(5,3),
    major_risk_factors_justification TEXT,
    
    red_flags_score DECIMAL(5,3),
    red_flags_confidence DECIMAL(5,3),
    red_flags_justification TEXT,
    
    -- Additional metadata
    best_model VARCHAR(100),
    total_weight_used DECIMAL(10,6),
    components_used INTEGER,
    
    -- Full JSON data for detailed analysis
    full_analysis_data JSONB,
    
    -- Individual model results (JSONB for flexibility)
    individual_model_results JSONB,
    
    -- Execution metadata
    execution_metadata JSONB,
    
    -- Indexes for better performance
    CONSTRAINT unique_analysis_run UNIQUE (run_id, company_ticker, analysis_timestamp)
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_qualitative_analysis_ticker ON qualitative_analysis(company_ticker);
CREATE INDEX IF NOT EXISTS idx_qualitative_analysis_timestamp ON qualitative_analysis(analysis_timestamp);
CREATE INDEX IF NOT EXISTS idx_qualitative_analysis_composite_score ON qualitative_analysis(composite_score);
CREATE INDEX IF NOT EXISTS idx_qualitative_analysis_created_at ON qualitative_analysis(created_at);
CREATE INDEX IF NOT EXISTS idx_qualitative_analysis_run_id ON qualitative_analysis(run_id);

-- Create GIN index for JSONB full_analysis_data for better JSON queries
CREATE INDEX IF NOT EXISTS idx_qualitative_analysis_full_data_gin ON qualitative_analysis USING GIN (full_analysis_data);

-- Add comments for documentation
COMMENT ON TABLE qualitative_analysis IS 'Stores comprehensive qualitative analysis results from QualAgent multi-LLM analysis';
COMMENT ON COLUMN qualitative_analysis.run_id IS 'Unique identifier for the analysis run';
COMMENT ON COLUMN qualitative_analysis.company_ticker IS 'Stock ticker symbol';
COMMENT ON COLUMN qualitative_analysis.analysis_timestamp IS 'Unix timestamp when analysis was performed';
COMMENT ON COLUMN qualitative_analysis.composite_score IS 'Overall composite score (0-5 scale)';
COMMENT ON COLUMN qualitative_analysis.full_analysis_data IS 'Complete JSON analysis data for detailed review';
COMMENT ON COLUMN qualitative_analysis.individual_model_results IS 'Individual LLM model results and outputs';
