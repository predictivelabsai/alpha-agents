"""Streamlit app for interacting with the FundamentalAgent screener."""

import math
from pathlib import Path
import io
from typing import Dict
import pandas as pd

import streamlit as st

try:
    from .fundamental_agent import FundamentalAgent
except ImportError:  # pragma: no cover - fallback when run as a script
    import sys

    src_dir = Path(__file__).resolve().parents[2]
    repo_root = src_dir.parent
    for path in (src_dir, repo_root):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    from agents.fundamental_agent.fundamental_agent import FundamentalAgent

from agents.fundamental_agent.stock_screener import StockScreener

SECTORS = sorted(StockScreener.SECTORS)
INDUSTRIES = sorted(StockScreener.INDUSTRIES)
REGIONS = sorted(StockScreener.REGIONS)
DEFAULT_REGION = "US"


@st.cache_resource(show_spinner=False)
def get_agent() -> FundamentalAgent:
    """Create a single FundamentalAgent instance for the app lifecycle."""

    return FundamentalAgent()


@st.cache_data(show_spinner=False)
def run_screen(
    region: str,
    filter_by: str,
    selection: str,
    min_cap: float,
    max_cap: float,
) -> "pd.DataFrame":
    """Execute the screen with the provided parameters."""

    options: Dict[str, object] = {
        "region": region,
        "min_cap": min_cap,
        "max_cap": max_cap,
        "sectors": None,
        "industries": None,
    }
    if filter_by == "Sector":
        options["sectors"] = selection
    else:
        options["industries"] = selection
    agent = StockScreener(**options)
    return agent.screen()


def main() -> None:
    st.set_page_config(page_title="Fundamental Screener", layout="wide")
    st.title("Fundamental Agent Screener")

    st.sidebar.header("Filters")
    filter_by = st.sidebar.radio("Filter By", ("Sector", "Industry"))

    if filter_by == "Sector":
        selection = st.sidebar.selectbox("Sector", SECTORS)
    else:
        selection = st.sidebar.selectbox("Industry", INDUSTRIES)

    region = st.sidebar.selectbox(
        "Region", REGIONS, index=REGIONS.index(DEFAULT_REGION)
    )
    min_cap = st.sidebar.number_input(
        "Minimum market cap (USD)", min_value=0.0, value=0.0, step=1e9
    )
    max_cap_value = st.sidebar.number_input(
        "Maximum market cap (USD, 0 for unlimited)", min_value=0.0, value=0.0, step=1e9
    )
    max_cap = float("inf") if max_cap_value == 0 else max_cap_value

    run_requested = st.sidebar.button("Run screen", type="primary")

    if "screen_params" not in st.session_state:
        st.info("Adjust the filters and click 'Run screen' to fetch results.")

    if run_requested:
        if not math.isinf(max_cap) and max_cap < min_cap:
            st.sidebar.error(
                "Maximum market cap must be greater than minimum market cap."
            )
        else:
            st.session_state["screen_params"] = {
                "region": region,
                "filter_by": filter_by,
                "selection": selection,
                "min_cap": float(min_cap),
                "max_cap": float(max_cap),
            }

    params = st.session_state.get("screen_params")
    if params:
        with st.spinner("Screening companies..."):
            df = run_screen(**params)
        if df.empty:
            st.warning("No companies matched the current filters.")
        else:
            st.caption(
                f"Showing {len(df)} companies filtered by {params['filter_by'].lower()} '{params['selection']}'."
            )
            st.dataframe(df, width="stretch")
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results as CSV",
                csv,
                file_name="fundamental_screen_results.csv",
                mime="text/csv",
            )
            excel_buffer = io.BytesIO()
            df.to_excel(excel_buffer, index=False, sheet_name="Results")
            excel_buffer.seek(0)
            st.download_button(
                "Download results as Excel",
                data=excel_buffer.getvalue(),
                file_name="fundamental_screen_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )


if __name__ == "__main__":
    main()
