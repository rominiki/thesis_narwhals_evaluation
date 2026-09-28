"""
Tests for compute_moving_aggregation_narwhals on the Modin.

Works with both Ray and Dask backends (auto-detected via conftest.py).
"""

import unittest
import numpy as np
import modin.pandas as mpd
from common.narwhals_mvag import compute_moving_aggregation_narwhals


def to_pandas(res):
    if hasattr(res, "_to_pandas"):
        return res._to_pandas()
    if hasattr(res, "to_pandas"):
        return res.to_pandas()
    return res


class TestMovingAggregationModin(unittest.TestCase):
    def setUp(self):
        self.df = mpd.DataFrame(
            {
                "LISTING_ID": [1, 1, 1, 1, 1, 2, 2, 2],
                "DATE": mpd.to_datetime(
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

    # Backward window
    def test_backward_sum_window1(self):
        res = compute_moving_aggregation_narwhals(self.df, 1, 0, "sum", "INPUT", "VALUE")
        df = to_pandas(res)
        np.testing.assert_array_equal(df[df["LISTING_ID"] == 1]["VALUE"].values, [1.0, 3.0, 5.0, 7.0, 9.0])
        np.testing.assert_array_equal(df[df["LISTING_ID"] == 2]["VALUE"].values, [10.0, 30.0, 50.0])

    def test_backward_mean_window1(self):
        res = compute_moving_aggregation_narwhals(self.df, 1, 0, "mean", "INPUT", "VALUE")
        df = to_pandas(res)
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 1]["VALUE"].values, [1.0, 1.5, 2.5, 3.5, 4.5])
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 2]["VALUE"].values, [10.0, 15.0, 25.0])

    def test_backward_mean_window2(self):
        res = compute_moving_aggregation_narwhals(self.df, 2, 0, "mean", "INPUT", "VALUE")
        df = to_pandas(res)
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 1]["VALUE"].values, [1.0, 1.5, 2.0, 3.0, 4.0])
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 2]["VALUE"].values, [10.0, 15.0, 20.0])

    def test_backward_sum_window2(self):
        res = compute_moving_aggregation_narwhals(self.df, 2, 0, "sum", "INPUT", "VALUE")
        df = to_pandas(res)
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 1]["VALUE"].values, [1.0, 3.0, 6.0, 9.0, 12.0])

    def test_window_larger_than_group(self):
        res = compute_moving_aggregation_narwhals(self.df, 10, 0, "mean", "INPUT", "VALUE")
        df = to_pandas(res)
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 2]["VALUE"].values, [10.0, 15.0, 20.0])

    # Forward window
    def test_forward_sum_window1(self):
        res = compute_moving_aggregation_narwhals(self.df, 0, 1, "sum", "INPUT", "VALUE")
        df = to_pandas(res)
        np.testing.assert_array_equal(df[df["LISTING_ID"] == 1]["VALUE"].values, [3.0, 5.0, 7.0, 9.0, 5.0])
        np.testing.assert_array_equal(df[df["LISTING_ID"] == 2]["VALUE"].values, [30.0, 50.0, 30.0])

    def test_forward_mean_window1(self):
        res = compute_moving_aggregation_narwhals(self.df, 0, 1, "mean", "INPUT", "VALUE")
        df = to_pandas(res)
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 1]["VALUE"].values, [1.5, 2.5, 3.5, 4.5, 5.0])
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 2]["VALUE"].values, [15.0, 25.0, 30.0])

    def test_forward_sum_window2(self):
        res = compute_moving_aggregation_narwhals(self.df, 0, 2, "sum", "INPUT", "VALUE")
        df = to_pandas(res)
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 1]["VALUE"].values, [6.0, 9.0, 12.0, 9.0, 5.0])
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 2]["VALUE"].values, [60.0, 50.0, 30.0])

    # Centered window
    def test_centered_mean(self):
        res = compute_moving_aggregation_narwhals(self.df, 1, 1, "mean", "INPUT", "VALUE")
        df = to_pandas(res)
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 1]["VALUE"].values, [1.5, 2.0, 3.0, 4.0, 4.5])
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 2]["VALUE"].values, [15.0, 20.0, 25.0])

    def test_centered_sum(self):
        res = compute_moving_aggregation_narwhals(self.df, 1, 1, "sum", "INPUT", "VALUE")
        df = to_pandas(res)
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 1]["VALUE"].values, [3.0, 6.0, 9.0, 12.0, 9.0])

    # Identity window (window_before=0, window_end=0)
    def test_identity_mean(self):
        res = compute_moving_aggregation_narwhals(self.df, 0, 0, "mean", "INPUT", "VALUE")
        df = to_pandas(res)
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 1]["VALUE"].values, [1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 2]["VALUE"].values, [10.0, 20.0, 30.0])

    def test_identity_sum(self):
        res = compute_moving_aggregation_narwhals(self.df, 0, 0, "sum", "INPUT", "VALUE")
        df = to_pandas(res)
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 1]["VALUE"].values, [1.0, 2.0, 3.0, 4.0, 5.0])

    # Single listing
    def test_single_listing_mean_backward(self):
        single_df = mpd.DataFrame(
            {
                "LISTING_ID": [1, 1, 1, 1, 1],
                "DATE": mpd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04", "2021-01-05"]),
                "INPUT": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        res = compute_moving_aggregation_narwhals(single_df, 2, 0, "mean", "INPUT", "VALUE")
        df = to_pandas(res)
        np.testing.assert_array_almost_equal(df["VALUE"].values, [1.0, 1.5, 2.0, 3.0, 4.0])

    def test_single_listing_sum_backward(self):
        single_df = mpd.DataFrame(
            {
                "LISTING_ID": [1, 1, 1, 1, 1],
                "DATE": mpd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04", "2021-01-05"]),
                "INPUT": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        res = compute_moving_aggregation_narwhals(single_df, 2, 0, "sum", "INPUT", "VALUE")
        df = to_pandas(res)
        np.testing.assert_array_almost_equal(df["VALUE"].values, [1.0, 3.0, 6.0, 9.0, 12.0])

    def test_single_row_group(self):
        single_row_df = mpd.DataFrame({"LISTING_ID": [1], "DATE": mpd.to_datetime(["2021-01-01"]), "INPUT": [42.0]})
        res = compute_moving_aggregation_narwhals(single_row_df, 2, 0, "mean", "INPUT", "VALUE")
        df = to_pandas(res)
        np.testing.assert_array_almost_equal(df["VALUE"].values, [42.0])

    # Groups with non-overlapping date ranges
    def test_different_dates_across_groups(self):
        """Groups whose dates do not overlap — partitioning must be date-independent."""
        diff_df = mpd.DataFrame(
            {
                "LISTING_ID": [1, 1, 1, 1, 1, 2, 2, 2],
                "DATE": mpd.to_datetime(
                    [
                        "2021-01-01",
                        "2021-01-02",
                        "2021-01-03",
                        "2021-01-04",
                        "2021-01-05",
                        "2021-01-07",
                        "2021-01-08",
                        "2021-01-09",
                    ]
                ),
                "INPUT": [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0],
            }
        )
        res = compute_moving_aggregation_narwhals(diff_df, 1, 0, "mean", "INPUT", "VALUE")
        df = to_pandas(res)
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 1]["VALUE"].values, [1.0, 1.5, 2.5, 3.5, 4.5])
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 2]["VALUE"].values, [10.0, 15.0, 25.0])

    # Unsorted input
    def test_unsorted_input_backward(self):
        """Input in random order — function must sort internally."""
        shuffled_df = mpd.DataFrame(
            {
                "LISTING_ID": [2, 1, 1, 2, 1, 2, 1, 1],
                "DATE": mpd.to_datetime(
                    [
                        "2021-01-02",
                        "2021-01-04",
                        "2021-01-01",
                        "2021-01-03",
                        "2021-01-05",
                        "2021-01-01",
                        "2021-01-03",
                        "2021-01-02",
                    ]
                ),
                "INPUT": [20.0, 4.0, 1.0, 30.0, 5.0, 10.0, 3.0, 2.0],
            }
        )
        res = compute_moving_aggregation_narwhals(shuffled_df, 1, 0, "sum", "INPUT", "VALUE")
        df = to_pandas(res)
        np.testing.assert_array_equal(df[df["LISTING_ID"] == 1]["VALUE"].values, [1.0, 3.0, 5.0, 7.0, 9.0])
        np.testing.assert_array_equal(df[df["LISTING_ID"] == 2]["VALUE"].values, [10.0, 30.0, 50.0])

    def test_unsorted_input_forward(self):
        """Unsorted input with forward window - both sort and forward path exercised."""
        shuffled_df = mpd.DataFrame(
            {
                "LISTING_ID": [2, 1, 1, 2, 1, 2, 1, 1],
                "DATE": mpd.to_datetime(
                    [
                        "2021-01-02",
                        "2021-01-04",
                        "2021-01-01",
                        "2021-01-03",
                        "2021-01-05",
                        "2021-01-01",
                        "2021-01-03",
                        "2021-01-02",
                    ]
                ),
                "INPUT": [20.0, 4.0, 1.0, 30.0, 5.0, 10.0, 3.0, 2.0],
            }
        )
        res = compute_moving_aggregation_narwhals(shuffled_df, 0, 1, "sum", "INPUT", "VALUE")
        df = to_pandas(res)
        np.testing.assert_array_equal(df[df["LISTING_ID"] == 1]["VALUE"].values, [3.0, 5.0, 7.0, 9.0, 5.0])
        np.testing.assert_array_equal(df[df["LISTING_ID"] == 2]["VALUE"].values, [30.0, 50.0, 30.0])

    # Edge cases
    def test_negative_and_zero_values(self):
        neg_df = mpd.DataFrame(
            {
                "LISTING_ID": [1, 1, 1, 1],
                "DATE": mpd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"]),
                "INPUT": [-3.0, 0.0, 5.0, -2.0],
            }
        )
        res = compute_moving_aggregation_narwhals(neg_df, 1, 0, "sum", "INPUT", "VALUE")
        df = to_pandas(res)
        np.testing.assert_array_almost_equal(df["VALUE"].values, [-3.0, -3.0, 5.0, 3.0])

    def test_many_groups(self):
        many_df = mpd.DataFrame(
            {
                "LISTING_ID": [1, 1, 2, 2, 3, 3, 4, 4],
                "DATE": mpd.to_datetime(
                    [
                        "2021-01-01",
                        "2021-01-02",
                        "2021-01-01",
                        "2021-01-02",
                        "2021-01-01",
                        "2021-01-02",
                        "2021-01-01",
                        "2021-01-02",
                    ]
                ),
                "INPUT": [1.0, 2.0, 10.0, 20.0, 100.0, 200.0, 1000.0, 2000.0],
            }
        )
        res = compute_moving_aggregation_narwhals(many_df, 1, 0, "mean", "INPUT", "VALUE")
        df = to_pandas(res)
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 1]["VALUE"].values, [1.0, 1.5])
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 2]["VALUE"].values, [10.0, 15.0])
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 3]["VALUE"].values, [100.0, 150.0])
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 4]["VALUE"].values, [1000.0, 1500.0])

    def test_unsupported_method_raises(self):
        with self.assertRaises(ValueError):
            compute_moving_aggregation_narwhals(self.df, 1, 0, "median", "INPUT", "VALUE")

    # std / var - backward window
    def test_std_backward_first_row_nan(self):
        """With window_size=2, first row of each group has only 1 sample : NaN."""
        res = compute_moving_aggregation_narwhals(self.df, 1, 0, "std", "INPUT", "VALUE")
        df = to_pandas(res)
        g1 = df[df["LISTING_ID"] == 1].sort_values("DATE")["VALUE"].values
        self.assertTrue(np.isnan(g1[0]))

    def test_std_backward_known_values(self):
        res = compute_moving_aggregation_narwhals(self.df, 1, 0, "std", "INPUT", "VALUE")
        df = to_pandas(res)
        g1 = df[df["LISTING_ID"] == 1].sort_values("DATE")["VALUE"].values
        expected = np.std([[1, 2], [2, 3], [3, 4], [4, 5]], axis=1, ddof=1)
        np.testing.assert_array_almost_equal(g1[1:], expected, decimal=5)

    def test_var_equals_std_squared_backward(self):
        res_std = compute_moving_aggregation_narwhals(self.df, 2, 0, "std", "INPUT", "VALUE")
        res_var = compute_moving_aggregation_narwhals(self.df, 2, 0, "var", "INPUT", "VALUE")
        std_df = to_pandas(res_std)
        var_df = to_pandas(res_var)
        for lid in [1, 2]:
            s = std_df[std_df["LISTING_ID"] == lid].sort_values("DATE")["VALUE"].values
            v = var_df[var_df["LISTING_ID"] == lid].sort_values("DATE")["VALUE"].values
            mask = ~np.isnan(s)
            np.testing.assert_array_almost_equal(s[mask] ** 2, v[mask], decimal=5)

    def test_std_groups_independent_backward(self):
        res = compute_moving_aggregation_narwhals(self.df, 1, 0, "std", "INPUT", "VALUE")
        df = to_pandas(res)
        g2 = df[df["LISTING_ID"] == 2].sort_values("DATE")["VALUE"].values
        self.assertTrue(np.isnan(g2[0]))
        np.testing.assert_array_almost_equal(g2[1], np.std([10, 20], ddof=1), decimal=5)
        np.testing.assert_array_almost_equal(g2[2], np.std([20, 30], ddof=1), decimal=5)

    # std / var - forward / centered window (numpy groupby path)
    def test_std_centered_ddof1(self):
        """Centered window std must use ddof=1 to match narwhals/pandas convention."""
        res = compute_moving_aggregation_narwhals(self.df, 1, 1, "std", "INPUT", "VALUE")
        df = to_pandas(res)
        g1 = df[df["LISTING_ID"] == 1].sort_values("DATE")["VALUE"].values
        np.testing.assert_almost_equal(g1[1], np.std([1, 2, 3], ddof=1), decimal=5)
        np.testing.assert_almost_equal(g1[2], np.std([2, 3, 4], ddof=1), decimal=5)

    def test_var_centered_ddof1(self):
        res = compute_moving_aggregation_narwhals(self.df, 1, 1, "var", "INPUT", "VALUE")
        df = to_pandas(res)
        g1 = df[df["LISTING_ID"] == 1].sort_values("DATE")["VALUE"].values
        np.testing.assert_almost_equal(g1[1], np.var([1, 2, 3], ddof=1), decimal=5)

    def test_var_equals_std_squared_forward(self):
        res_std = compute_moving_aggregation_narwhals(self.df, 1, 1, "std", "INPUT", "VALUE")
        res_var = compute_moving_aggregation_narwhals(self.df, 1, 1, "var", "INPUT", "VALUE")
        std_df = to_pandas(res_std)
        var_df = to_pandas(res_var)
        s = std_df[std_df["LISTING_ID"] == 1].sort_values("DATE")["VALUE"].values
        v = var_df[var_df["LISTING_ID"] == 1].sort_values("DATE")["VALUE"].values
        mask = ~np.isnan(s)
        np.testing.assert_array_almost_equal(s[mask] ** 2, v[mask], decimal=5)

    def test_std_single_element_window_is_nan(self):
        """window_before=0, window_end=0 : each row is its own window : std is NaN (ddof=1)."""
        res = compute_moving_aggregation_narwhals(self.df, 0, 0, "std", "INPUT", "VALUE")
        df = to_pandas(res)
        g1 = df[df["LISTING_ID"] == 1].sort_values("DATE")["VALUE"].values
        self.assertTrue(all(np.isnan(v) for v in g1))


if __name__ == "__main__":
    unittest.main()
