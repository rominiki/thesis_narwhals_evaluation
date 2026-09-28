"""
Unified ratio calculation using Narwhals for cross-backend compatibility.

Provides multiple ratio calculation methods with safe handling of invalid
denominators and logarithm inputs.
"""

import narwhals as nw
import math
from typing import Any, Literal

RatioMethod = Literal["ratio", "discrete_pct", "continuous_pct", "robust"]


def calculate_ratio_narwhals(
    data: Any,
    *,
    numerator_col: str,
    denominator_col: str,
    result_col: str = "VALUE",
    method: RatioMethod = "ratio",
) -> Any:
    """
    Compute a ratio between two columns using Narwhals.

    Args:
        data: DataFrame in any supported format
        numerator_col: Name of the numerator column
        denominator_col: Name of the denominator column
        result_col: Name of the output column
        method: Ratio calculation method:
            - "ratio": a / b
            - "discrete_pct": (a / b) - 1
            - "continuous_pct": ln(a / b)
            - "robust": (a - b) / ((|a| + |b|) / 2)

    Returns:
        DataFrame in the same native format as the input, with result_col added.

    Raises:
        ValueError: If method is not supported.
    """
    if method not in ("ratio", "discrete_pct", "continuous_pct", "robust"):
        raise ValueError(
            f"Unknown method: {method}. Must be one of: "
            "ratio, discrete_pct, continuous_pct, robust"
        )

    df = nw.from_native(data)

    num = nw.col(numerator_col)
    den = nw.col(denominator_col)

    safe_den = (
        nw.when(den != 0)
        .then(den)
        .otherwise(None)
    )

    if method == "ratio":
        expr = num / safe_den

    elif method == "discrete_pct":
        expr = (num / safe_den) - 1

    elif method == "continuous_pct":
        ratio = num / den
        expr = (
            nw.when((den != 0) & (ratio > 0))
            .then(ratio.log(base=math.e))
            .otherwise(None)
        )

    else:  # robust
        abs_sum = num.abs() + den.abs()
        expr = (
            nw.when(abs_sum != 0)
            .then((num - den) / (abs_sum / 2))
            .otherwise(None)
        )

    result_df = df.with_columns(expr.alias(result_col))
    return nw.to_native(result_df)
