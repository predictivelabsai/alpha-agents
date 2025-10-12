"""Streamlit app for interacting with the StockScreener."""

import math
import io
import sys
import os
from typing import Dict
import pandas as pd  # type: ignore

import streamlit as st  # type: ignore

# Add the parent directory to Python path to find utils module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.stock_screener import StockScreener
from utils.db_util import save_fundamental_screen

SECTORS = sorted(StockScreener.SECTORS)
INDUSTRIES = sorted(StockScreener.INDUSTRIES)
REGIONS = sorted(StockScreener.REGIONS)
DEFAULT_REGION = "US"

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
    screener = StockScreener(**options)
    return screener.screen()


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
            
            # Store results in session state for QualAgent
            st.session_state["screener_results"] = df
            st.success(f"✅ Results stored for QualAgent analysis!")
            
            # Ensure ticker column is included in exports
            df_export = df.reset_index(drop=False)
            if "ticker" not in df_export.columns and "index" in df_export.columns:
                df_export = df_export.rename(columns={"index": "ticker"})
            csv = df_export.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download results as CSV",
                csv,
                file_name="fundamental_screen_results.csv",
                mime="text/csv",
            )
            excel_buffer = io.BytesIO()
            df_export.to_excel(excel_buffer, index=False, sheet_name="Results")
            excel_buffer.seek(0)
            st.download_button(
                "Download results as Excel",
                data=excel_buffer.getvalue(),
                file_name="fundamental_screen_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
            if st.button("Store in Database"):
                with st.spinner("Saving results to database..."):
                    run_id = save_fundamental_screen(df.reset_index(drop=False))
                st.success(f"Saved {len(df)} rows to database (run_id={run_id}).")


if __name__ == "__main__":
    main()
