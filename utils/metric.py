import pandas as pd
from typing import List


def calculate_growth(df: pd.DataFrame, prefix: str) -> pd.Series:
    cols = [c for c in df.columns if c.startswith(prefix)]
    sub = df[cols]

    def _calculate_growth(row):
        non_na = row.dropna()
        if len(non_na) < 2:
            return None
        old = non_na.iloc[-1]
        new = non_na.iloc[0]
        return (new / old - 1) * 100

    return sub.apply(_calculate_growth, axis=1)


def is_increasing(df: pd.DataFrame, prefix: str) -> pd.Series:
    cols = [c for c in df.columns if c.startswith(prefix)]
    sub = df[cols]

    def _is_increasing(row):
        non_na = row.dropna()
        if len(non_na) < 2:
            return False
        ascending = non_na.iloc[::-1]
        return ascending.is_monotonic_increasing

    return sub.apply(_is_increasing, axis=1)


def is_decreasing(df: pd.DataFrame, prefix: str) -> pd.Series:
    return ~is_increasing(df, prefix)


def calculate_ratio(
    df: pd.DataFrame, dividend_prefix: str, divisor_prefix: str, ratio_prefix: str
) -> pd.Series:
    dividend_cols = [c for c in df.columns if c.startswith(dividend_prefix)]
    divisor_cols = [c for c in df.columns if c.startswith(divisor_prefix)]

    def get_suffix(col: str) -> str:
        return (
            col.split(dividend_prefix)[-1]
            if col.startswith(dividend_prefix)
            else col.split(divisor_prefix)[-1]
        )

    suffixes = sorted(
        set(map(get_suffix, dividend_cols)) & set(map(get_suffix, divisor_cols))
    )

    out = pd.DataFrame(index=df.index)

    for s in suffixes:
        d_col = f"{dividend_prefix}{s}"
        v_col = f"{divisor_prefix}{s}"
        out[f"{ratio_prefix}{s}"] = df[d_col] / df[v_col]

    return out


def calculate_average(df: pd.DataFrame, prefix: str, rolling_window: int = 2):
    cols = [c for c in df.columns if c.startswith(prefix)]

    def extract_offset(col: str) -> int:
        suffix = col.split(prefix)[-1]
        return 0 if suffix == "0Y" else -int(suffix.replace("Y", ""))

    sorted_cols = sorted(cols, key=extract_offset, reverse=True)
    sub = df[sorted_cols[::-1]]
    avg = sub.T.rolling(rolling_window, axis=0).mean().T
    avg = avg.iloc[:, (rolling_window - 1) :]
    avg.columns = [f"{prefix}avg_{c.split(prefix)[-1]}" for c in avg.columns]
    return avg
