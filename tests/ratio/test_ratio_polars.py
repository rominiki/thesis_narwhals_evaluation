"""
Tests for calculate_ratio_narwhals on Polars DataFrames.
Proves that the Narwhals function works on Polars without any changes.
"""

import unittest
import numpy as np
import polars as pl
from common.narwhals_ratio import calculate_ratio_narwhals


def make_df(num, den):
    return pl.DataFrame({"NUM": num, "DEN": den})


def to_list(res, col):
    """Extract column values as a Python list, handling nulls as NaN."""
    return [v if v is not None else float("nan") for v in res[col].to_list()]


class TestRatioPolars(unittest.TestCase):

    # ratio
    def test_ratio_basic(self):
        df = make_df([10.0, 20.0, 30.0], [2.0, 4.0, 5.0])
        res = calculate_ratio_narwhals(
            df, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="ratio"
        )
        np.testing.assert_array_almost_equal(to_list(res, "VALUE"), [5.0, 5.0, 6.0])

    def test_ratio_zero_denominator_is_null(self):
        df = make_df([10.0, 5.0], [0.0, 2.0])
        res = calculate_ratio_narwhals(
            df, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="ratio"
        )
        vals = to_list(res, "VALUE")
        self.assertTrue(np.isnan(vals[0]))
        self.assertAlmostEqual(vals[1], 2.5)

    def test_ratio_negative_values(self):
        df = make_df([-10.0, 10.0], [2.0, -5.0])
        res = calculate_ratio_narwhals(
            df, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="ratio"
        )
        np.testing.assert_array_almost_equal(to_list(res, "VALUE"), [-5.0, -2.0])

    # discrete_pct
    def test_discrete_pct_basic(self):
        df = make_df([110.0, 90.0], [100.0, 100.0])
        res = calculate_ratio_narwhals(
            df, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="discrete_pct"
        )
        np.testing.assert_array_almost_equal(to_list(res, "VALUE"), [0.10, -0.10])

    def test_discrete_pct_zero_denominator_is_null(self):
        df = make_df([10.0], [0.0])
        res = calculate_ratio_narwhals(
            df, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="discrete_pct"
        )
        self.assertTrue(np.isnan(to_list(res, "VALUE")[0]))

    def test_discrete_pct_no_change(self):
        df = make_df([5.0, 10.0], [5.0, 10.0])
        res = calculate_ratio_narwhals(
            df, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="discrete_pct"
        )
        np.testing.assert_array_almost_equal(to_list(res, "VALUE"), [0.0, 0.0])

    # continuous_pct
    def test_continuous_pct_basic(self):
        df = make_df([np.e, 1.0], [1.0, 1.0])
        res = calculate_ratio_narwhals(
            df, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="continuous_pct"
        )
        np.testing.assert_array_almost_equal(to_list(res, "VALUE"), [1.0, 0.0])

    def test_continuous_pct_zero_denominator_is_null(self):
        df = make_df([10.0], [0.0])
        res = calculate_ratio_narwhals(
            df, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="continuous_pct"
        )
        self.assertTrue(np.isnan(to_list(res, "VALUE")[0]))

    def test_continuous_pct_negative_ratio_is_null(self):
        df = make_df([-5.0], [2.0])
        res = calculate_ratio_narwhals(
            df, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="continuous_pct"
        )
        self.assertTrue(np.isnan(to_list(res, "VALUE")[0]))

    def test_continuous_pct_no_change(self):
        df = make_df([7.0], [7.0])
        res = calculate_ratio_narwhals(
            df, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="continuous_pct"
        )
        np.testing.assert_almost_equal(to_list(res, "VALUE")[0], 0.0)

    # robust
    def test_robust_basic(self):
        df = make_df([3.0], [1.0])
        res = calculate_ratio_narwhals(
            df, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="robust"
        )
        expected = (3.0 - 1.0) / ((3.0 + 1.0) / 2)
        np.testing.assert_almost_equal(to_list(res, "VALUE")[0], expected)

    def test_robust_both_zero_is_null(self):
        df = make_df([0.0], [0.0])
        res = calculate_ratio_narwhals(
            df, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="robust"
        )
        self.assertTrue(np.isnan(to_list(res, "VALUE")[0]))

    def test_robust_symmetric(self):
        df_ab = make_df([4.0], [2.0])
        df_ba = make_df([2.0], [4.0])
        r_ab = to_list(
            calculate_ratio_narwhals(
                df_ab, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="robust"
            ),
            "VALUE",
        )[0]
        r_ba = to_list(
            calculate_ratio_narwhals(
                df_ba, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="robust"
            ),
            "VALUE",
        )[0]
        np.testing.assert_almost_equal(r_ab, -r_ba)

    def test_robust_bounded(self):
        import random

        random.seed(42)
        nums = [random.uniform(-100, 100) for _ in range(50)]
        dens = [random.uniform(-100, 100) for _ in range(50)]
        df = make_df(nums, dens)
        res = calculate_ratio_narwhals(
            df, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="robust"
        )
        vals = [v for v in to_list(res, "VALUE") if not np.isnan(v)]
        self.assertTrue(all(-2.0 <= v <= 2.0 for v in vals))

    # General
    def test_unknown_method_raises(self):
        df = make_df([1.0], [1.0])
        with self.assertRaises(ValueError):
            calculate_ratio_narwhals(df, numerator_col="NUM", denominator_col="DEN", method="unknown")

    def test_result_col_name(self):
        df = make_df([2.0], [1.0])
        res = calculate_ratio_narwhals(
            df, numerator_col="NUM", denominator_col="DEN", result_col="MY_RATIO", method="ratio"
        )
        self.assertIn("MY_RATIO", res.columns)

    def test_original_columns_preserved(self):
        df = make_df([2.0, 4.0], [1.0, 2.0])
        res = calculate_ratio_narwhals(
            df, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="ratio"
        )
        self.assertIn("NUM", res.columns)
        self.assertIn("DEN", res.columns)

    def test_returns_polars_dataframe(self):
        """Verify that passing Polars in gives Polars back."""
        df = make_df([10.0], [2.0])
        res = calculate_ratio_narwhals(df, numerator_col="NUM", denominator_col="DEN", method="ratio")
        self.assertIsInstance(res, pl.DataFrame)


if __name__ == "__main__":
    unittest.main()
