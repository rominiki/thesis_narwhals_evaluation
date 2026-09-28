"""
Tests for calculate_ratio_narwhals on the Modin path.
Covers all four methods: ratio, discrete_pct, continuous_pct, robust.

Works with both Ray and Dask backends (auto-detected via conftest.py).
"""

import unittest
import numpy as np
import modin.pandas as mpd
from common.narwhals_ratio import calculate_ratio_narwhals


def to_pandas(res):
    if hasattr(res, "_to_pandas"):
        return res._to_pandas()
    if hasattr(res, "to_pandas"):
        return res.to_pandas()
    return res


def make_df(num, den):
    return mpd.DataFrame({"NUM": num, "DEN": den})


class TestRatioModin(unittest.TestCase):

    # ratio
    def test_ratio_basic(self):
        df = make_df([10.0, 20.0, 30.0], [2.0, 4.0, 5.0])
        res = to_pandas(
            calculate_ratio_narwhals(df, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="ratio")
        )
        np.testing.assert_array_almost_equal(res["VALUE"].values, [5.0, 5.0, 6.0])

    def test_ratio_zero_denominator_is_null(self):
        df = make_df([10.0, 5.0], [0.0, 2.0])
        res = to_pandas(
            calculate_ratio_narwhals(df, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="ratio")
        )
        self.assertTrue(np.isnan(res["VALUE"].values[0]))
        self.assertAlmostEqual(res["VALUE"].values[1], 2.5)

    def test_ratio_negative_values(self):
        df = make_df([-10.0, 10.0], [2.0, -5.0])
        res = to_pandas(
            calculate_ratio_narwhals(df, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="ratio")
        )
        np.testing.assert_array_almost_equal(res["VALUE"].values, [-5.0, -2.0])

    # discrete_pct
    def test_discrete_pct_basic(self):
        # (num/den) - 1
        df = make_df([110.0, 90.0], [100.0, 100.0])
        res = to_pandas(
            calculate_ratio_narwhals(
                df, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="discrete_pct"
            )
        )
        np.testing.assert_array_almost_equal(res["VALUE"].values, [0.10, -0.10])

    def test_discrete_pct_zero_denominator_is_null(self):
        df = make_df([10.0], [0.0])
        res = to_pandas(
            calculate_ratio_narwhals(
                df, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="discrete_pct"
            )
        )
        self.assertTrue(np.isnan(res["VALUE"].values[0]))

    def test_discrete_pct_no_change(self):
        """num == den : discrete_pct = 0."""
        df = make_df([5.0, 10.0], [5.0, 10.0])
        res = to_pandas(
            calculate_ratio_narwhals(
                df, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="discrete_pct"
            )
        )
        np.testing.assert_array_almost_equal(res["VALUE"].values, [0.0, 0.0])

    # continuous_pct
    def test_continuous_pct_basic(self):
        # log(num/den)
        df = make_df([np.e, 1.0], [1.0, 1.0])
        res = to_pandas(
            calculate_ratio_narwhals(
                df, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="continuous_pct"
            )
        )
        np.testing.assert_array_almost_equal(res["VALUE"].values, [1.0, 0.0])

    def test_continuous_pct_zero_denominator_is_null(self):
        df = make_df([10.0], [0.0])
        res = to_pandas(
            calculate_ratio_narwhals(
                df, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="continuous_pct"
            )
        )
        self.assertTrue(np.isnan(res["VALUE"].values[0]))

    def test_continuous_pct_negative_ratio_is_null(self):
        """ratio <= 0 : log undefined : None."""
        df = make_df([-5.0], [2.0])
        res = to_pandas(
            calculate_ratio_narwhals(
                df, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="continuous_pct"
            )
        )
        self.assertTrue(np.isnan(res["VALUE"].values[0]))

    def test_continuous_pct_no_change(self):
        """num == den : log(1) = 0."""
        df = make_df([7.0], [7.0])
        res = to_pandas(
            calculate_ratio_narwhals(
                df, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="continuous_pct"
            )
        )
        np.testing.assert_almost_equal(res["VALUE"].values[0], 0.0)

    # robust
    def test_robust_basic(self):
        # (num - den) / ((|num| + |den|) / 2)
        df = make_df([3.0], [1.0])
        res = to_pandas(
            calculate_ratio_narwhals(
                df, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="robust"
            )
        )
        expected = (3.0 - 1.0) / ((3.0 + 1.0) / 2)  # = 1.0
        np.testing.assert_almost_equal(res["VALUE"].values[0], expected)

    def test_robust_both_zero_is_null(self):
        df = make_df([0.0], [0.0])
        res = to_pandas(
            calculate_ratio_narwhals(
                df, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="robust"
            )
        )
        self.assertTrue(np.isnan(res["VALUE"].values[0]))

    def test_robust_symmetric(self):
        """robust(a, b) == -robust(b, a)."""
        df_ab = make_df([4.0], [2.0])
        df_ba = make_df([2.0], [4.0])
        r_ab = to_pandas(
            calculate_ratio_narwhals(
                df_ab, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="robust"
            )
        )["VALUE"].values[0]
        r_ba = to_pandas(
            calculate_ratio_narwhals(
                df_ba, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="robust"
            )
        )["VALUE"].values[0]
        np.testing.assert_almost_equal(r_ab, -r_ba)

    def test_robust_bounded(self):
        """robust result is always in [-2, 2]."""
        import random

        random.seed(42)
        nums = [random.uniform(-100, 100) for _ in range(50)]
        dens = [random.uniform(-100, 100) for _ in range(50)]
        df = make_df(nums, dens)
        res = to_pandas(
            calculate_ratio_narwhals(
                df, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="robust"
            )
        )
        vals = res["VALUE"].dropna().values
        self.assertTrue(all(-2.0 <= v <= 2.0 for v in vals))

    # General
    def test_unknown_method_raises(self):
        df = make_df([1.0], [1.0])
        with self.assertRaises(ValueError):
            calculate_ratio_narwhals(df, numerator_col="NUM", denominator_col="DEN", method="unknown")

    def test_result_col_name(self):
        df = make_df([2.0], [1.0])
        res = to_pandas(
            calculate_ratio_narwhals(
                df, numerator_col="NUM", denominator_col="DEN", result_col="MY_RATIO", method="ratio"
            )
        )
        self.assertIn("MY_RATIO", res.columns)

    def test_original_columns_preserved(self):
        df = make_df([2.0, 4.0], [1.0, 2.0])
        res = to_pandas(
            calculate_ratio_narwhals(df, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="ratio")
        )
        self.assertIn("NUM", res.columns)
        self.assertIn("DEN", res.columns)


if __name__ == "__main__":
    unittest.main()
