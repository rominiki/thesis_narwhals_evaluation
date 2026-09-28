"""
Cross-backend consistency tests for calculate_ratio_narwhals.
Verifies that pandas, Polars, Modin, and Ibis all produce identical results
for the same input data and all four ratio methods.

Works with both Ray and Dask backends (auto-detected via conftest.py).
"""

import unittest
import numpy as np
import pandas as pd
import polars as pl
import modin.pandas as mpd
import ibis
from common.narwhals_ratio import calculate_ratio_narwhals


class TestRatioAllBackends(unittest.TestCase):
    """
    For each ratio method, create the same data in all four backend formats,
    run the same function, and assert identical results.
    """

    def setUp(self):
        self.num = [10.0, 20.0, 0.0, -5.0, 100.0, 7.0]
        self.den = [2.0, 4.0, 3.0, 5.0, 25.0, 7.0]

        # Same data, four formats
        self.pandas_df = pd.DataFrame({"NUM": self.num, "DEN": self.den})
        self.polars_df = pl.DataFrame({"NUM": self.num, "DEN": self.den})
        self.modin_df = mpd.DataFrame({"NUM": self.num, "DEN": self.den})
        self.ibis_table = ibis.memtable(pd.DataFrame({"NUM": self.num, "DEN": self.den}))

    def _extract_values(self, result):
        """Convert any backend result to a numpy array for comparison."""
        if isinstance(result, pl.DataFrame):
            return np.array([v if v is not None else float("nan") for v in result["VALUE"].to_list()])
        if hasattr(result, "execute"):
            return result.execute()["VALUE"].to_numpy(dtype=float, na_value=float("nan"))
        if hasattr(result, "_to_pandas"):
            return result._to_pandas()["VALUE"].to_numpy(dtype=float, na_value=float("nan"))
        if hasattr(result, "to_pandas"):
            return result.to_pandas()["VALUE"].to_numpy(dtype=float, na_value=float("nan"))
        return result["VALUE"].to_numpy(dtype=float, na_value=float("nan"))

    def _run_all_backends(self, method):
        """Run the same ratio method on all four backends and return results."""
        kwargs = dict(numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method=method)

        results = {
            "pandas": self._extract_values(calculate_ratio_narwhals(self.pandas_df, **kwargs)),
            "polars": self._extract_values(calculate_ratio_narwhals(self.polars_df, **kwargs)),
            "modin": self._extract_values(calculate_ratio_narwhals(self.modin_df, **kwargs)),
            "ibis": self._extract_values(calculate_ratio_narwhals(self.ibis_table, **kwargs)),
        }
        return results

    def _assert_all_equal(self, results, method):
        """Assert that all four backends produce identical results."""
        backends = list(results.keys())
        reference = results[backends[0]]

        for backend in backends[1:]:
            other = results[backend]

            # NaN positions must match
            np.testing.assert_array_equal(
                np.isnan(reference),
                np.isnan(other),
                err_msg=f"NaN positions differ: {backends[0]} vs {backend} for method={method}",
            )

            # Finite values must match
            mask = ~np.isnan(reference)
            np.testing.assert_array_almost_equal(
                reference[mask],
                other[mask],
                decimal=5,
                err_msg=f"Values differ: {backends[0]} vs {backend} for method={method}",
            )

    # One test per method: all four backends must agree
    def test_ratio_all_backends(self):
        results = self._run_all_backends("ratio")
        self._assert_all_equal(results, "ratio")

    def test_discrete_pct_all_backends(self):
        results = self._run_all_backends("discrete_pct")
        self._assert_all_equal(results, "discrete_pct")

    def test_continuous_pct_all_backends(self):
        results = self._run_all_backends("continuous_pct")
        self._assert_all_equal(results, "continuous_pct")

    def test_robust_all_backends(self):
        results = self._run_all_backends("robust")
        self._assert_all_equal(results, "robust")

    # Zero denominator: all backends must return NaN in same positions
    def test_zero_denominator_all_backends(self):
        num = [10.0, 5.0, 0.0]
        den = [0.0, 2.0, 0.0]

        pandas_df = pd.DataFrame({"NUM": num, "DEN": den})
        polars_df = pl.DataFrame({"NUM": num, "DEN": den})
        modin_df = mpd.DataFrame({"NUM": num, "DEN": den})
        ibis_table = ibis.memtable(pd.DataFrame({"NUM": num, "DEN": den}))

        kwargs = dict(numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="ratio")

        results = {
            "pandas": self._extract_values(calculate_ratio_narwhals(pandas_df, **kwargs)),
            "polars": self._extract_values(calculate_ratio_narwhals(polars_df, **kwargs)),
            "modin": self._extract_values(calculate_ratio_narwhals(modin_df, **kwargs)),
            "ibis": self._extract_values(calculate_ratio_narwhals(ibis_table, **kwargs)),
        }
        self._assert_all_equal(results, "ratio with zeros")

    # Return type: each backend gets its own type back
    def test_pandas_returns_pandas(self):
        res = calculate_ratio_narwhals(self.pandas_df, numerator_col="NUM", denominator_col="DEN", method="ratio")
        self.assertIsInstance(res, pd.DataFrame)

    def test_polars_returns_polars(self):
        res = calculate_ratio_narwhals(self.polars_df, numerator_col="NUM", denominator_col="DEN", method="ratio")
        self.assertIsInstance(res, pl.DataFrame)

    def test_modin_returns_modin(self):
        res = calculate_ratio_narwhals(self.modin_df, numerator_col="NUM", denominator_col="DEN", method="ratio")
        # Modin DataFrames are instances of modin.pandas.DataFrame
        self.assertTrue(type(res).__module__.startswith("modin"))

    def test_ibis_returns_ibis(self):
        res = calculate_ratio_narwhals(self.ibis_table, numerator_col="NUM", denominator_col="DEN", method="ratio")
        self.assertTrue(hasattr(res, "execute"), "Ibis result should have .execute()")


if __name__ == "__main__":
    unittest.main()
