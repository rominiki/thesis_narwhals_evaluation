import unittest
import numpy as np
import pandas as pd
import ibis
from common.narwhals_mvag import compute_moving_aggregation_narwhals


class TestMovingAggregationNarwhalsIbis(unittest.TestCase):
    def setUp(self):
        df = pd.DataFrame(
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
        self.ibis_table = ibis.memtable(df)

    def test_simple_rolling_sum(self):
        res = compute_moving_aggregation_narwhals(self.ibis_table, 1, 0, "sum", "INPUT", "VALUE")
        df = res if isinstance(res, pd.DataFrame) else res.to_pandas()
        expected_1 = [1.0, 3.0, 5.0, 7.0, 9.0]
        expected_2 = [10.0, 30.0, 50.0]
        np.testing.assert_array_equal(df[df["LISTING_ID"] == 1]["VALUE"].values, expected_1)
        np.testing.assert_array_equal(df[df["LISTING_ID"] == 2]["VALUE"].values, expected_2)

    def test_rolling_sum_with_future(self):
        res = compute_moving_aggregation_narwhals(self.ibis_table, 0, 1, "sum", "INPUT", "VALUE")
        df = res if isinstance(res, pd.DataFrame) else res.to_pandas()
        expected_1 = [3.0, 5.0, 7.0, 9.0, 5.0]
        expected_2 = [30.0, 50.0, 30.0]
        np.testing.assert_array_equal(df[df["LISTING_ID"] == 1]["VALUE"].values, expected_1)
        np.testing.assert_array_equal(df[df["LISTING_ID"] == 2]["VALUE"].values, expected_2)

    def test_rolling_mean_centered(self):
        res = compute_moving_aggregation_narwhals(self.ibis_table, 1, 1, "mean", "INPUT", "VALUE")
        df = res if isinstance(res, pd.DataFrame) else res.to_pandas()
        expected_1 = [1.5, 2.0, 3.0, 4.0, 4.5]
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 1]["VALUE"].values, expected_1)
        expected_2 = [15.0, 20.0, 25.0]
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 2]["VALUE"].values, expected_2)

    def test_rolling_mean_backward(self):
        res = compute_moving_aggregation_narwhals(self.ibis_table, 2, 0, "mean", "INPUT", "VALUE")
        df = res if isinstance(res, pd.DataFrame) else res.to_pandas()
        expected_1 = [1.0, 1.5, 2.0, 3.0, 4.0]
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 1]["VALUE"].values, expected_1)
        expected_2 = [10.0, 15.0, 20.0]
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 2]["VALUE"].values, expected_2)

    def test_invalid_method(self):
        with self.assertRaises(ValueError):
            compute_moving_aggregation_narwhals(self.ibis_table, 1, 0, "median", "INPUT", "VALUE")

    def test_identity_window(self):
        res = compute_moving_aggregation_narwhals(self.ibis_table, 0, 0, "mean", "INPUT", "VALUE")
        df = res if isinstance(res, pd.DataFrame) else res.to_pandas()
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 1]["VALUE"].values, [1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 2]["VALUE"].values, [10.0, 20.0, 30.0])

    def test_single_row_group(self):
        single_df = pd.DataFrame(
            {
                "LISTING_ID": [1],
                "DATE": pd.to_datetime(["2021-01-01"]),
                "INPUT": [42.0],
            }
        )
        single_table = ibis.memtable(single_df)
        res = compute_moving_aggregation_narwhals(single_table, 2, 0, "mean", "INPUT", "VALUE")
        df = res if isinstance(res, pd.DataFrame) else res.to_pandas()
        np.testing.assert_array_almost_equal(df["VALUE"].values, [42.0])

    def test_window_larger_than_group(self):
        res = compute_moving_aggregation_narwhals(self.ibis_table, 10, 0, "mean", "INPUT", "VALUE")
        df = res if isinstance(res, pd.DataFrame) else res.to_pandas()
        expected_2 = [10.0, 15.0, 20.0]
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 2]["VALUE"].values, expected_2)

    def test_forward_only_sum(self):
        res = compute_moving_aggregation_narwhals(self.ibis_table, 0, 2, "sum", "INPUT", "VALUE")
        df = res if isinstance(res, pd.DataFrame) else res.to_pandas()
        expected_1 = [6.0, 9.0, 12.0, 9.0, 5.0]
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 1]["VALUE"].values, expected_1)
        expected_2 = [60.0, 50.0, 30.0]
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 2]["VALUE"].values, expected_2)

    def test_forward_only_mean(self):
        res = compute_moving_aggregation_narwhals(self.ibis_table, 0, 1, "mean", "INPUT", "VALUE")
        df = res if isinstance(res, pd.DataFrame) else res.to_pandas()
        expected_1 = [1.5, 2.5, 3.5, 4.5, 5.0]
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 1]["VALUE"].values, expected_1)
        expected_2 = [15.0, 25.0, 30.0]
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 2]["VALUE"].values, expected_2)

    def test_backward_1_mean(self):
        res = compute_moving_aggregation_narwhals(self.ibis_table, 1, 0, "mean", "INPUT", "VALUE")
        df = res if isinstance(res, pd.DataFrame) else res.to_pandas()
        expected_1 = [1.0, 1.5, 2.5, 3.5, 4.5]
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 1]["VALUE"].values, expected_1)
        expected_2 = [10.0, 15.0, 25.0]
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 2]["VALUE"].values, expected_2)

    def test_different_dates_across_groups(self):
        diff_df = pd.DataFrame(
            {
                "LISTING_ID": [1, 1, 1, 2, 2, 2],
                "DATE": pd.to_datetime(
                    [
                        "2021-01-01",
                        "2021-01-02",
                        "2021-01-03",
                        "2021-01-07",
                        "2021-01-08",
                        "2021-01-09",
                    ]
                ),
                "INPUT": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            }
        )
        diff_table = ibis.memtable(diff_df)
        res = compute_moving_aggregation_narwhals(diff_table, 1, 0, "mean", "INPUT", "VALUE")
        df = res if isinstance(res, pd.DataFrame) else res.to_pandas()
        expected_1 = [1.0, 1.5, 2.5]
        expected_2 = [10.0, 15.0, 25.0]
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 1]["VALUE"].values, expected_1)
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 2]["VALUE"].values, expected_2)

    def test_negative_and_zero_values(self):
        neg_df = pd.DataFrame(
            {
                "LISTING_ID": [1, 1, 1, 1],
                "DATE": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"]),
                "INPUT": [-3.0, 0.0, 5.0, -2.0],
            }
        )
        neg_table = ibis.memtable(neg_df)
        res = compute_moving_aggregation_narwhals(neg_table, 1, 0, "sum", "INPUT", "VALUE")
        df = res if isinstance(res, pd.DataFrame) else res.to_pandas()
        expected = [-3.0, -3.0, 5.0, 3.0]
        np.testing.assert_array_almost_equal(df["VALUE"].values, expected)

    def test_many_groups(self):
        many_df = pd.DataFrame(
            {
                "LISTING_ID": [1, 1, 2, 2, 3, 3, 4, 4],
                "DATE": pd.to_datetime(
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
        many_table = ibis.memtable(many_df)
        res = compute_moving_aggregation_narwhals(many_table, 1, 0, "mean", "INPUT", "VALUE")
        df = res if isinstance(res, pd.DataFrame) else res.to_pandas()
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 1]["VALUE"].values, [1.0, 1.5])
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 2]["VALUE"].values, [10.0, 15.0])
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 3]["VALUE"].values, [100.0, 150.0])
        np.testing.assert_array_almost_equal(df[df["LISTING_ID"] == 4]["VALUE"].values, [1000.0, 1500.0])

    def test_extra_columns_preserved(self):
        extra_df = pd.DataFrame(
            {
                "LISTING_ID": [1, 1, 1],
                "DATE": pd.to_datetime(["2021-01-01", "2021-01-02", "2021-01-03"]),
                "INPUT": [10.0, 20.0, 30.0],
                "EXTRA": ["a", "b", "c"],
            }
        )
        extra_table = ibis.memtable(extra_df)
        res = compute_moving_aggregation_narwhals(extra_table, 1, 0, "sum", "INPUT", "VALUE")
        df = res if isinstance(res, pd.DataFrame) else res.to_pandas()
        self.assertIn("EXTRA", df.columns)
        self.assertEqual(list(df["EXTRA"].values), ["a", "b", "c"])

    # std / var - backward window
    def test_std_backward_first_row_nan(self):
        res = compute_moving_aggregation_narwhals(self.ibis_table, 1, 0, "std", "INPUT", "VALUE")
        df = res if isinstance(res, pd.DataFrame) else res.to_pandas()
        g1 = df[df["LISTING_ID"] == 1].sort_values("DATE")["VALUE"].values
        self.assertTrue(np.isnan(g1[0]))

    def test_std_backward_known_values(self):
        res = compute_moving_aggregation_narwhals(self.ibis_table, 1, 0, "std", "INPUT", "VALUE")
        df = res if isinstance(res, pd.DataFrame) else res.to_pandas()
        g1 = df[df["LISTING_ID"] == 1].sort_values("DATE")["VALUE"].values
        self.assertTrue(np.isnan(g1[0]))
        expected = np.std([[1, 2], [2, 3], [3, 4], [4, 5]], axis=1, ddof=1)
        np.testing.assert_array_almost_equal(g1[1:], expected, decimal=5)

    def test_var_equals_std_squared_backward(self):
        res_std = compute_moving_aggregation_narwhals(self.ibis_table, 2, 0, "std", "INPUT", "VALUE")
        res_var = compute_moving_aggregation_narwhals(self.ibis_table, 2, 0, "var", "INPUT", "VALUE")
        std_df = res_std if isinstance(res_std, pd.DataFrame) else res_std.to_pandas()
        var_df = res_var if isinstance(res_var, pd.DataFrame) else res_var.to_pandas()
        for lid in [1, 2]:
            s = std_df[std_df["LISTING_ID"] == lid].sort_values("DATE")["VALUE"].values
            v = var_df[var_df["LISTING_ID"] == lid].sort_values("DATE")["VALUE"].values
            mask = ~np.isnan(s)
            np.testing.assert_array_almost_equal(s[mask] ** 2, v[mask], decimal=5)

    def test_std_groups_independent(self):
        res = compute_moving_aggregation_narwhals(self.ibis_table, 1, 0, "std", "INPUT", "VALUE")
        df = res if isinstance(res, pd.DataFrame) else res.to_pandas()
        g2 = df[df["LISTING_ID"] == 2].sort_values("DATE")["VALUE"].values
        self.assertTrue(np.isnan(g2[0]))
        np.testing.assert_almost_equal(g2[1], np.std([10, 20], ddof=1), decimal=5)
        np.testing.assert_almost_equal(g2[2], np.std([20, 30], ddof=1), decimal=5)

    # std / var - forward / centered window
    def test_std_forward_window(self):
        res = compute_moving_aggregation_narwhals(self.ibis_table, 1, 1, "std", "INPUT", "VALUE")
        df = res if isinstance(res, pd.DataFrame) else res.to_pandas()
        g1 = df[df["LISTING_ID"] == 1].sort_values("DATE")["VALUE"].values
        np.testing.assert_almost_equal(g1[1], np.std([1, 2, 3], ddof=1), decimal=5)

    def test_var_forward_window(self):
        res = compute_moving_aggregation_narwhals(self.ibis_table, 1, 1, "var", "INPUT", "VALUE")
        df = res if isinstance(res, pd.DataFrame) else res.to_pandas()
        g1 = df[df["LISTING_ID"] == 1].sort_values("DATE")["VALUE"].values
        np.testing.assert_almost_equal(g1[1], np.var([1, 2, 3], ddof=1), decimal=5)

    def test_var_equals_std_squared_forward(self):
        res_std = compute_moving_aggregation_narwhals(self.ibis_table, 1, 1, "std", "INPUT", "VALUE")
        res_var = compute_moving_aggregation_narwhals(self.ibis_table, 1, 1, "var", "INPUT", "VALUE")
        std_df = res_std if isinstance(res_std, pd.DataFrame) else res_std.to_pandas()
        var_df = res_var if isinstance(res_var, pd.DataFrame) else res_var.to_pandas()
        s = std_df[std_df["LISTING_ID"] == 1].sort_values("DATE")["VALUE"].values
        v = var_df[var_df["LISTING_ID"] == 1].sort_values("DATE")["VALUE"].values
        mask = ~np.isnan(s)
        np.testing.assert_array_almost_equal(s[mask] ** 2, v[mask], decimal=5)


if __name__ == "__main__":
    unittest.main()
