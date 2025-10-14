"""Streamlit app for interacting with the StockScreener."""

import math
import io
import sys
import os
from typing import Dict
import pandas as pd  # type: ignore

import streamlit as st  # type: ignore

# Add the parent directory to Python path to find utils module
# More robust path resolution for Streamlit
script_dir = os.path.dirname(os.path.abspath(__file__))  # pages directory
parent_dir = os.path.dirname(script_dir)  # alpha-agents directory
utils_dir = os.path.join(parent_dir, 'utils')

# Add parent directory to Python path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# # Debug path information
# st.write(f"🔍 Debug Info:")
# st.write(f"Script directory: {script_dir}")
# st.write(f"Parent directory: {parent_dir}")
# st.write(f"Utils directory: {utils_dir}")
# st.write(f"Utils exists: {os.path.exists(utils_dir)}")
# st.write(f"Python path contains parent: {parent_dir in sys.path}")

# # Verify utils directory exists
# if not os.path.exists(utils_dir):
#     st.error(f"Utils directory not found at: {utils_dir}")
#     st.stop()

# # Verify stock_screener.py exists
# stock_screener_file = os.path.join(utils_dir, 'stock_screener.py')
# if not os.path.exists(stock_screener_file):
#     st.error(f"stock_screener.py not found at: {stock_screener_file}")
#     st.stop()

# st.write(f"Stock screener file exists: {os.path.exists(stock_screener_file)}")

try:
    from utils.stock_screener import StockScreener
    # st.success("✅ Successfully imported StockScreener!")
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error(f"Current sys.path: {sys.path}")
    st.stop()
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
    st.set_page_config(page_title="Fundamental Screener Agent", layout="wide")
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
            csv = df_export.to_csv().encode("utf-8")
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
