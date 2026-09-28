"""
Tests for calculate_ratio_narwhals on the Ibis path.
Covers all four methods: ratio, discrete_pct, continuous_pct, robust.
"""

import unittest
import numpy as np
import pandas as pd
import ibis
from common.narwhals_ratio import calculate_ratio_narwhals


def to_pandas(res):
    if isinstance(res, pd.DataFrame):
        return res
    if hasattr(res, "execute"):
        return res.execute()
    return pd.DataFrame(res)


def make_table(num, den):
    df = pd.DataFrame({"NUM": num, "DEN": den})
    return ibis.memtable(df)


class TestRatioIbis(unittest.TestCase):
    # ratio
    def test_ratio_basic(self):
        t = make_table([10.0, 20.0, 30.0], [2.0, 4.0, 5.0])
        res = to_pandas(
            calculate_ratio_narwhals(t, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="ratio")
        )
        np.testing.assert_array_almost_equal(res["VALUE"].values, [5.0, 5.0, 6.0])

    def test_ratio_zero_denominator_is_null(self):
        t = make_table([10.0, 5.0], [0.0, 2.0])
        res = to_pandas(
            calculate_ratio_narwhals(t, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="ratio")
        )
        self.assertTrue(res["VALUE"].isna().any())
        non_null = res["VALUE"].dropna().values
        np.testing.assert_almost_equal(non_null[0], 2.5)

    def test_ratio_negative_values(self):
        t = make_table([-10.0, 10.0], [2.0, -5.0])
        res = to_pandas(
            calculate_ratio_narwhals(t, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="ratio")
        )
        np.testing.assert_array_almost_equal(res["VALUE"].values, [-5.0, -2.0])

    # discrete_pct
    def test_discrete_pct_basic(self):
        t = make_table([110.0, 90.0], [100.0, 100.0])
        res = to_pandas(
            calculate_ratio_narwhals(
                t, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="discrete_pct"
            )
        )
        np.testing.assert_array_almost_equal(res["VALUE"].values, [0.10, -0.10])

    def test_discrete_pct_zero_denominator_is_null(self):
        t = make_table([10.0], [0.0])
        res = to_pandas(
            calculate_ratio_narwhals(
                t, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="discrete_pct"
            )
        )
        self.assertTrue(res["VALUE"].isna().all())

    def test_discrete_pct_no_change(self):
        t = make_table([5.0, 10.0], [5.0, 10.0])
        res = to_pandas(
            calculate_ratio_narwhals(
                t, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="discrete_pct"
            )
        )
        np.testing.assert_array_almost_equal(res["VALUE"].values, [0.0, 0.0])

    # continuous_pct
    def test_continuous_pct_basic(self):
        t = make_table([np.e, 1.0], [1.0, 1.0])
        res = to_pandas(
            calculate_ratio_narwhals(
                t, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="continuous_pct"
            )
        )
        np.testing.assert_array_almost_equal(res["VALUE"].values, [1.0, 0.0])

    def test_continuous_pct_zero_denominator_is_null(self):
        t = make_table([10.0], [0.0])
        res = to_pandas(
            calculate_ratio_narwhals(
                t, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="continuous_pct"
            )
        )
        self.assertTrue(res["VALUE"].isna().all())

    def test_continuous_pct_negative_ratio_is_null(self):
        t = make_table([-5.0], [2.0])
        res = to_pandas(
            calculate_ratio_narwhals(
                t, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="continuous_pct"
            )
        )
        self.assertTrue(res["VALUE"].isna().all())

    def test_continuous_pct_no_change(self):
        t = make_table([7.0], [7.0])
        res = to_pandas(
            calculate_ratio_narwhals(
                t, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="continuous_pct"
            )
        )
        np.testing.assert_almost_equal(res["VALUE"].values[0], 0.0)

    # robust
    def test_robust_basic(self):
        t = make_table([3.0], [1.0])
        res = to_pandas(
            calculate_ratio_narwhals(t, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="robust")
        )
        expected = (3.0 - 1.0) / ((3.0 + 1.0) / 2)
        np.testing.assert_almost_equal(res["VALUE"].values[0], expected)

    def test_robust_both_zero_is_null(self):
        t = make_table([0.0], [0.0])
        res = to_pandas(
            calculate_ratio_narwhals(t, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="robust")
        )
        self.assertTrue(res["VALUE"].isna().all())

    def test_robust_symmetric(self):
        t_ab = make_table([4.0], [2.0])
        t_ba = make_table([2.0], [4.0])
        r_ab = to_pandas(
            calculate_ratio_narwhals(
                t_ab, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="robust"
            )
        )["VALUE"].values[0]
        r_ba = to_pandas(
            calculate_ratio_narwhals(
                t_ba, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="robust"
            )
        )["VALUE"].values[0]
        np.testing.assert_almost_equal(r_ab, -r_ba)

    # General
    def test_unknown_method_raises(self):
        t = make_table([1.0], [1.0])
        with self.assertRaises(ValueError):
            calculate_ratio_narwhals(t, numerator_col="NUM", denominator_col="DEN", method="unknown")

    def test_result_col_name(self):
        t = make_table([2.0], [1.0])
        res = to_pandas(
            calculate_ratio_narwhals(
                t, numerator_col="NUM", denominator_col="DEN", result_col="MY_RATIO", method="ratio"
            )
        )
        self.assertIn("MY_RATIO", res.columns)

    def test_original_columns_preserved(self):
        t = make_table([2.0, 4.0], [1.0, 2.0])
        res = to_pandas(
            calculate_ratio_narwhals(t, numerator_col="NUM", denominator_col="DEN", result_col="VALUE", method="ratio")
        )
        self.assertIn("NUM", res.columns)
        self.assertIn("DEN", res.columns)


if __name__ == "__main__":
    unittest.main()
