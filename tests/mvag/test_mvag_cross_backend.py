"""
Cross-backend consistency tests for compute_moving_aggregation_narwhals.
Verifies that pandas, Polars, Modin, and Ibis all produce identical results
for the same input data and window configurations.

Works with both Ray and Dask backends (auto-detected via conftest.py).
"""

import unittest
import numpy as np
import pandas as pd
import polars as pl
import modin.pandas as mpd
import ibis
from common.narwhals_mvag import compute_moving_aggregation_narwhals


class TestMvagCrossBackend(unittest.TestCase):
    def setUp(self):
        self.raw_df = pd.DataFrame(
            {
                "LISTING_ID": [1, 1, 1, 1, 1, 2, 2, 2],
                "DATE": pd.to_datetime(
                    [
                        "2021-01-01",
                        "2021-01-02",
                        "2021-01-03",
                        "2021-01-04",
                        "2021-01-05",
                        "2021-01-01",
                        "2021-01-02",
                        "2021-01-03",
                    ]
                ),
                "INPUT": [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0],
            }
        )
        self.pandas_df = self.raw_df
        self.polars_df = pl.from_pandas(self.raw_df)
        self.modin_df = mpd.DataFrame(self.raw_df)
        self.ibis_table = ibis.memtable(self.raw_df)

    def _extract(self, res, lid):
        """Return sorted VALUE array for a given LISTING_ID from any backend result."""
        if isinstance(res, pl.DataFrame):
            return res.filter(pl.col("LISTING_ID") == lid).sort("DATE")["VALUE"].to_numpy()
        if hasattr(res, "execute"):  # ibis lazy table
            df = res.execute()
        elif hasattr(res, "_to_pandas"):  # modin
            df = res._to_pandas()
        elif hasattr(res, "to_pandas"):
            df = res.to_pandas()
        else:
            df = pd.DataFrame(res)
        return df[df["LISTING_ID"] == lid].sort_values("DATE")["VALUE"].to_numpy(dtype=float)

    def _compare_all(self, window_before, window_end, method):
        """Run on all four backends and assert pairwise equality."""
        kwargs = dict(
            window_before=window_before,
            window_end=window_end,
            aggregation_method=method,
            input_col="INPUT",
            output_col="VALUE",
        )
        results = {
            "pandas": compute_moving_aggregation_narwhals(self.pandas_df, **kwargs),
            "polars": compute_moving_aggregation_narwhals(self.polars_df, **kwargs),
            "modin": compute_moving_aggregation_narwhals(self.modin_df, **kwargs),
            "ibis": compute_moving_aggregation_narwhals(self.ibis_table, **kwargs),
        }

        backends = list(results.keys())
        for lid in self.raw_df["LISTING_ID"].unique():
            ref_vals = self._extract(results[backends[0]], lid)
            for backend in backends[1:]:
                other_vals = self._extract(results[backend], lid)
                np.testing.assert_array_equal(
                    np.isnan(ref_vals),
                    np.isnan(other_vals),
                    err_msg=f"NaN mismatch: {backends[0]} vs {backend}, "
                    f"LISTING_ID={lid}, window=({window_before},{window_end}), method={method}",
                )
                mask = ~np.isnan(ref_vals)
                np.testing.assert_array_almost_equal(
                    ref_vals[mask],
                    other_vals[mask],
                    decimal=5,
                    err_msg=f"Value mismatch: {backends[0]} vs {backend}, "
                    f"LISTING_ID={lid}, window=({window_before},{window_end}), method={method}",
                )

    # mean / sum

    def test_backward_mean(self):
        self._compare_all(2, 0, "mean")

    def test_backward_sum(self):
        self._compare_all(1, 0, "sum")

    def test_forward_mean(self):
        self._compare_all(0, 1, "mean")

    def test_forward_sum(self):
        self._compare_all(0, 1, "sum")

    def test_centered_mean(self):
        self._compare_all(1, 1, "mean")

    def test_centered_sum(self):
        self._compare_all(1, 1, "sum")

    def test_identity_mean(self):
        self._compare_all(0, 0, "mean")

    def test_identity_sum(self):
        self._compare_all(0, 0, "sum")

    def test_large_backward_mean(self):
        self._compare_all(10, 0, "mean")

    def test_large_forward_sum(self):
        self._compare_all(0, 10, "sum")

    # std / var

    def test_backward_std(self):
        self._compare_all(2, 0, "std")

    def test_backward_var(self):
        self._compare_all(2, 0, "var")

    def test_centered_std(self):
        self._compare_all(1, 1, "std")

    def test_centered_var(self):
        self._compare_all(1, 1, "var")

    def test_forward_std(self):
        self._compare_all(0, 1, "std")

    def test_forward_var(self):
        self._compare_all(0, 1, "var")


if __name__ == "__main__":
    unittest.main()
