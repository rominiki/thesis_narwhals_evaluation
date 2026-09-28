"""
Tests for compute_moving_aggregation_narwhals on Polars DataFrames.
"""

import unittest
import numpy as np
import polars as pl
from common.narwhals_mvag import compute_moving_aggregation_narwhals


class TestMvagPolars(unittest.TestCase):
    def setUp(self):
        from datetime import date
        self.df = pl.DataFrame(
            {
                "LISTING_ID": [1, 1, 1, 1, 1, 2, 2, 2],
                "DATE": pl.Series([
                    date(2021, 1, 1), date(2021, 1, 2), date(2021, 1, 3),
                    date(2021, 1, 4), date(2021, 1, 5),
                    date(2021, 1, 1), date(2021, 1, 2), date(2021, 1, 3),
                ]),
                "INPUT": [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0],
            }
        )

    def _g1(self, res):
        return res.filter(pl.col("LISTING_ID") == 1).sort("DATE")["VALUE"].to_numpy()

    def _g2(self, res):
        return res.filter(pl.col("LISTING_ID") == 2).sort("DATE")["VALUE"].to_numpy()

    def test_backward_mean(self):
        res = compute_moving_aggregation_narwhals(self.df, 2, 0, "mean", "INPUT", "VALUE")
        np.testing.assert_array_almost_equal(self._g1(res), [1.0, 1.5, 2.0, 3.0, 4.0])

    def test_backward_sum(self):
        res = compute_moving_aggregation_narwhals(self.df, 1, 0, "sum", "INPUT", "VALUE")
        np.testing.assert_array_almost_equal(self._g1(res), [1.0, 3.0, 5.0, 7.0, 9.0])
        np.testing.assert_array_almost_equal(self._g2(res), [10.0, 30.0, 50.0])

    def test_forward_mean(self):
        res = compute_moving_aggregation_narwhals(self.df, 0, 1, "mean", "INPUT", "VALUE")
        np.testing.assert_array_almost_equal(self._g1(res), [1.5, 2.5, 3.5, 4.5, 5.0])

    def test_centered_mean(self):
        res = compute_moving_aggregation_narwhals(self.df, 1, 1, "mean", "INPUT", "VALUE")
        np.testing.assert_array_almost_equal(self._g1(res), [1.5, 2.0, 3.0, 4.0, 4.5])

    def test_std_backward_first_row_nan(self):
        res = compute_moving_aggregation_narwhals(self.df, 1, 0, "std", "INPUT", "VALUE")
        vals = self._g1(res)
        self.assertTrue(np.isnan(vals[0]))
        np.testing.assert_almost_equal(vals[1], np.std([1, 2], ddof=1), decimal=5)

    def test_var_equals_std_squared(self):
        s = self._g1(compute_moving_aggregation_narwhals(self.df, 2, 0, "std", "INPUT", "VALUE"))
        v = self._g1(compute_moving_aggregation_narwhals(self.df, 2, 0, "var", "INPUT", "VALUE"))
        mask = ~np.isnan(s)
        np.testing.assert_array_almost_equal(s[mask] ** 2, v[mask], decimal=5)

    def test_returns_polars_dataframe(self):
        res = compute_moving_aggregation_narwhals(self.df, 1, 0, "mean", "INPUT", "VALUE")
        self.assertIsInstance(res, pl.DataFrame)

    def test_unsupported_method_raises(self):
        with self.assertRaises(ValueError):
            compute_moving_aggregation_narwhals(self.df, 1, 0, "median", "INPUT", "VALUE")


if __name__ == "__main__":
    unittest.main()
