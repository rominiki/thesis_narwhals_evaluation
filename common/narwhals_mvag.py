"""
Moving window aggregation using Narwhals for cross-backend compatibility.

Uses Narwhals' native rolling operations for backward-only windows on all backends.
Supports backward, forward, and centered windows across pandas, Polars, Modin, and Ibis.
"""
import narwhals as nw
import numpy as np
from typing import Any

SUPPORTED_AGGREGATION_METHODS = ["mean", "sum", "std", "var"]


def compute_moving_aggregation_narwhals(
    data: Any,
    window_before: int,
    window_end: int,
    aggregation_method: str,
    input_col: str = "INPUT",
    output_col: str = "VALUE",
) -> Any:
    """
    Compute moving window aggregations across multiple DataFrame backends.

    Supports pandas, Polars, Modin, and Ibis with a unified interface.
    Handles backward, forward, and centered windows with group-aware computation.

    Window Types:
        - Backward-only: window_before > 0, window_end = 0
        - Forward-only: window_before = 0, window_end > 0
        - Centered: window_before > 0, window_end > 0

    Args:
        data: DataFrame in any supported format (pandas, Polars, Modin, Ibis)
        window_before: Number of rows to include before current row
        window_end: Number of rows to include after current row
        aggregation_method: One of "mean", "sum", "std", "var"
        input_col: Name of column to aggregate
        output_col: Name of output column for results

    Returns:
        DataFrame in the same format as input, with output_col added

    Raises:
        ValueError: If aggregation_method is not supported
    """
    if aggregation_method not in SUPPORTED_AGGREGATION_METHODS:
        raise ValueError(f"Unsupported method: {aggregation_method}")

    if window_before < 0 or window_end < 0:
        raise ValueError(
            f"Window sizes must be non-negative: "
            f"window_before={window_before}, window_end={window_end}"
        )

    def compute_window_values(vals: np.ndarray) -> np.ndarray:
        """Core window computation logic for forward/centered windows."""
        agg_funcs = {
            "mean": np.mean,
            "sum": np.sum,
            "std": lambda x: np.std(x, ddof=1),
            "var": lambda x: np.var(x, ddof=1),
        }
        agg_func = agg_funcs[aggregation_method]
        n = len(vals)
        result = np.empty(n, dtype=float)

        # For std/var, we need at least 2 values for sample statistics (ddof=1)
        requires_min_2 = aggregation_method in ("std", "var")

        for i in range(n):
            start = max(0, i - window_before)
            end = min(n, i + window_end + 1)
            window_vals = vals[start:end]

            # Return NaN for std/var when window has < 2 values
            if requires_min_2 and len(window_vals) < 2:
                result[i] = np.nan
            else:
                result[i] = agg_func(window_vals)
        return result

    df = nw.from_native(data)

    # PATH 1: Backward-Only Window (window_end = 0)
    # Uses Narwhals rolling operations for ALL backends including Ibis
    if window_end == 0:

        df = df.sort("LISTING_ID", "DATE")

        window_size = window_before + 1
        method = {
            "mean": "rolling_mean",
            "sum": "rolling_sum",
            "std": "rolling_std",
            "var": "rolling_var",
        }[aggregation_method]

        # Build the Narwhals rolling expression
        expr = getattr(nw.col(input_col), method)(window_size, min_samples=1)

        if df.implementation.is_ibis():
            rolled = expr.over("LISTING_ID", order_by="DATE")
        else:
            rolled = expr.over("LISTING_ID")

        result_df = df.with_columns(rolled.alias(output_col))
        return nw.to_native(result_df)

    # PATH 2: Forward or Centered Window On Ibis (window_end > 0)
    # Uses raw Ibis window functions because Narwhals doesn't support forward windows
    if df.implementation.is_ibis():
        import ibis

        native_df = nw.to_native(df)

        win = ibis.window(
            group_by="LISTING_ID",
            order_by="DATE",
            preceding=window_before,
            following=window_end,
        )

        agg_expr = {
            "mean": native_df[input_col].mean(),
            "sum": native_df[input_col].sum(),
            "std": native_df[input_col].std(),
            "var": native_df[input_col].var(),
        }[aggregation_method]

        return native_df.mutate(**{output_col: agg_expr.over(win)})

    # Eager backends(pandas, Polars, Modin)
    # For forward or centered windows (window_end > 0)
    df = df.sort("LISTING_ID", "DATE")

    # PATH 3: Forward/Centered Window On Polars
    if df.implementation.is_polars():
        import polars as pl

        native_df = nw.to_native(df)

        def polars_apply_window(group_df: pl.DataFrame) -> pl.DataFrame:
            vals = group_df[input_col].to_numpy()
            result = compute_window_values(vals)
            return group_df.with_columns(pl.Series(output_col, result))

        return native_df.group_by(
            "LISTING_ID", maintain_order=True
        ).map_groups(polars_apply_window)

    # PATH 4: Forward/Centered Window On Pandas/Modin
    native_df = nw.to_native(df)

    def pandas_apply_window(group):
        vals = group[input_col].values
        result = compute_window_values(vals)
        group = group.copy()
        group[output_col] = result
        return group

    result = native_df.groupby("LISTING_ID", group_keys=False).apply(
        pandas_apply_window, include_groups=False
    )

    result["LISTING_ID"] = native_df["LISTING_ID"].values

    return result
