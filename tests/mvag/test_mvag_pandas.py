"""
Tests for compute_moving_aggregation_narwhals on raw pandas DataFrames.
"""

import unittest
import numpy as np
import pandas as pd
from common.narwhals_mvag import compute_moving_aggregation_narwhals


class TestMvagPandas(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
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

    def test_backward_mean(self):
        res = compute_moving_aggregation_narwhals(self.df, 2, 0, "mean", "INPUT", "VALUE")
        np.testing.assert_array_almost_equal(
            res[res["LISTING_ID"] == 1].sort_values("DATE")["VALUE"].values, [1.0, 1.5, 2.0, 3.0, 4.0]
        )

    def test_backward_sum(self):
        res = compute_moving_aggregation_narwhals(self.df, 1, 0, "sum", "INPUT", "VALUE")
        np.testing.assert_array_almost_equal(
            res[res["LISTING_ID"] == 1].sort_values("DATE")["VALUE"].values, [1.0, 3.0, 5.0, 7.0, 9.0]
        )

    def test_forward_mean(self):
        res = compute_moving_aggregation_narwhals(self.df, 0, 1, "mean", "INPUT", "VALUE")
        np.testing.assert_array_almost_equal(
            res[res["LISTING_ID"] == 1].sort_values("DATE")["VALUE"].values, [1.5, 2.5, 3.5, 4.5, 5.0]
        )

    def test_centered_mean(self):
        res = compute_moving_aggregation_narwhals(self.df, 1, 1, "mean", "INPUT", "VALUE")
        np.testing.assert_array_almost_equal(
            res[res["LISTING_ID"] == 1].sort_values("DATE")["VALUE"].values, [1.5, 2.0, 3.0, 4.0, 4.5]
        )

    def test_std_backward(self):
        res = compute_moving_aggregation_narwhals(self.df, 1, 0, "std", "INPUT", "VALUE")
        vals = res[res["LISTING_ID"] == 1].sort_values("DATE")["VALUE"].values
        self.assertTrue(np.isnan(vals[0]))
        np.testing.assert_almost_equal(vals[1], np.std([1, 2], ddof=1), decimal=5)

    def test_var_equals_std_squared(self):
        s = (
            compute_moving_aggregation_narwhals(self.df, 2, 0, "std", "INPUT", "VALUE")[self.df["LISTING_ID"] == 1]
            .sort_values("DATE")["VALUE"]
            .values
        )
        v = (
            compute_moving_aggregation_narwhals(self.df, 2, 0, "var", "INPUT", "VALUE")[self.df["LISTING_ID"] == 1]
            .sort_values("DATE")["VALUE"]
            .values
        )
        mask = ~np.isnan(s)
        np.testing.assert_array_almost_equal(s[mask] ** 2, v[mask], decimal=5)

    def test_returns_pandas_dataframe(self):
        res = compute_moving_aggregation_narwhals(self.df, 1, 0, "mean", "INPUT", "VALUE")
        self.assertIsInstance(res, pd.DataFrame)

    def test_unsupported_method_raises(self):
        with self.assertRaises(ValueError):
            compute_moving_aggregation_narwhals(self.df, 1, 0, "median", "INPUT", "VALUE")


if __name__ == "__main__":
    unittest.main()
