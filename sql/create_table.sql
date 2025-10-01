-- DDL for table: fundamental_screen (PostgreSQL)

CREATE TABLE IF NOT EXISTS fundamental_screen (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL,
    ticker TEXT NOT NULL,
    company_name TEXT,
    sector TEXT,
    industry TEXT,

    market_cap DOUBLE PRECISION,

    gross_profit_margin_ttm DOUBLE PRECISION,
    operating_profit_margin_ttm DOUBLE PRECISION,
    net_profit_margin_ttm DOUBLE PRECISION,
    ebit_margin_ttm DOUBLE PRECISION,
    ebitda_margin_ttm DOUBLE PRECISION,

    roa_ttm DOUBLE PRECISION,
    roe_ttm DOUBLE PRECISION,
    roic_ttm DOUBLE PRECISION,

    total_revenue_4y_cagr DOUBLE PRECISION,
    net_income_4y_cagr DOUBLE PRECISION,
    operating_cash_flow_4y_cagr DOUBLE PRECISION,

    total_revenue_4y_consistency DOUBLE PRECISION,
    net_income_4y_consistency DOUBLE PRECISION,
    operating_cash_flow_4y_consistency DOUBLE PRECISION,

    current_ratio DOUBLE PRECISION,
    debt_to_ebitda_ratio DOUBLE PRECISION,
    debt_servicing_ratio DOUBLE PRECISION,

    score DOUBLE PRECISION,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT uq_fundamental_screen_run_ticker UNIQUE (run_id, ticker)
);
