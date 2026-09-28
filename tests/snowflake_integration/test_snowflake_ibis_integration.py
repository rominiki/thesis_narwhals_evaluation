"""
Integration tests for Ibis Approach: Snowflake + Ibis backend.

Uses mocking to test the integration without requiring actual Snowflake credentials.
Tests verify that:
1. Backend correctly initializes and manages connections
2. Data loading produces correct Ibis table types
3. Business logic functions work with Ibis tables
4. SQL generation and execution paths are correct
"""

import unittest
from unittest.mock import MagicMock
import pandas as pd
import ibis

from ibis_based_snowflake_integration.snowflake_ibis_backend import SnowflakeIbisBackend
from common.narwhals_mvag import compute_moving_aggregation_narwhals
from common.narwhals_ratio import calculate_ratio_narwhals


def _make_backend_with_mock_conn(mock_table):
    """
    Create a SnowflakeIbisBackend whose _get_connection is patched so that
    conn.sql(...) returns mock_table.  
    No real Snowflake call is made.
    """
    mock_conn = MagicMock()
    mock_conn.sql.return_value = mock_table

    backend = SnowflakeIbisBackend()

    backend._get_connection = MagicMock(return_value=mock_conn)
    return backend


class TestSnowflakeIbisIntegration(unittest.TestCase):
    """Test Snowflake Ibis backend integration with mocked connections."""

    def setUp(self):
        """Set up test data that simulates Snowflake query results."""
        self.test_mvag_data = pd.DataFrame({
            "LISTING_ID": [1, 1, 1, 1, 1, 2, 2, 2],
            "DATE": pd.to_datetime([
                "2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04",
                "2021-01-05", "2021-01-01", "2021-01-02", "2021-01-03"
            ]),
            "INPUT": [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0],
        })

        self.test_ratio_data = pd.DataFrame({
            "LISTING_ID": [1, 1, 1, 2, 2, 2],
            "DATE": pd.to_datetime([
                "2021-01-01", "2021-01-02", "2021-01-03",
                "2021-01-01", "2021-01-02", "2021-01-03"
            ]),
            "NUMERATOR": [10.0, 20.0, 30.0, 5.0, 10.0, 15.0],
            "DENOMINATOR": [5.0, 10.0, 0.0, 2.0, 5.0, 3.0],
        })

    def test_backend_initialization(self):
        """Test that backend initializes without actual Snowflake connection."""
        backend = SnowflakeIbisBackend()

        self.assertIsNotNone(backend.connection_params)
        self.assertIsNone(backend.connection)  # Connection not created until needed
        self.assertIsNotNone(backend._ibis)    # Ibis should be available
        self.assertIn('account', backend.connection_params)
        self.assertIn('user', backend.connection_params)

    def test_mvag_backward_window(self):
        """Test moving aggregation with backward window on mocked Ibis table."""
        mock_table = ibis.memtable(self.test_mvag_data)
        backend = _make_backend_with_mock_conn(mock_table)

        ibis_table = backend.load_data_from_snowflake("SELECT * FROM test_table")
        backend._get_connection.assert_called_once()

        result = compute_moving_aggregation_narwhals(
            data=ibis_table,
            window_before=2,
            window_end=0,
            aggregation_method="mean",
            input_col="INPUT",
            output_col="VALUE"
        )

        result_df = result.execute()

        self.assertIn("VALUE", result_df.columns)
        self.assertEqual(len(result_df), len(self.test_mvag_data))

        first_val = result_df[result_df["LISTING_ID"] == 1].iloc[0]["VALUE"]
        self.assertAlmostEqual(first_val, 1.0, places=5)

    def test_mvag_forward_window(self):
        """Test moving aggregation with forward window on mocked Ibis table."""
        mock_table = ibis.memtable(self.test_mvag_data)
        backend = _make_backend_with_mock_conn(mock_table)

        ibis_table = backend.load_data_from_snowflake("SELECT * FROM test_table")

        result = compute_moving_aggregation_narwhals(
            data=ibis_table,
            window_before=0,
            window_end=1,
            aggregation_method="sum",
            input_col="INPUT",
            output_col="VALUE"
        )

        result_df = result.execute()

        self.assertIn("VALUE", result_df.columns)
        # First value for LISTING_ID=1 should be sum of [1.0, 2.0] = 3.0
        first_val = result_df[result_df["LISTING_ID"] == 1].iloc[0]["VALUE"]
        self.assertAlmostEqual(first_val, 3.0, places=5)

    def test_mvag_centered_window(self):
        """Test moving aggregation with centered window on mocked Ibis table."""
        mock_table = ibis.memtable(self.test_mvag_data)
        backend = _make_backend_with_mock_conn(mock_table)

        ibis_table = backend.load_data_from_snowflake("SELECT * FROM test_table")

        result = compute_moving_aggregation_narwhals(
            data=ibis_table,
            window_before=1,
            window_end=1,
            aggregation_method="mean",
            input_col="INPUT",
            output_col="VALUE"
        )

        result_df = result.execute()

        self.assertIn("VALUE", result_df.columns)
        # Second value for LISTING_ID=1 should be mean of [1.0, 2.0, 3.0] = 2.0
        second_val = result_df[result_df["LISTING_ID"] == 1].iloc[1]["VALUE"]
        self.assertAlmostEqual(second_val, 2.0, places=5)

    def test_mvag_std_var(self):
        """Test std and var aggregations on mocked Ibis table."""
        mock_table = ibis.memtable(self.test_mvag_data)
        backend = _make_backend_with_mock_conn(mock_table)

        ibis_table = backend.load_data_from_snowflake("SELECT * FROM test_table")

        result_std = compute_moving_aggregation_narwhals(
            data=ibis_table,
            window_before=2,
            window_end=0,
            aggregation_method="std",
            input_col="INPUT",
            output_col="VALUE"
        )
        self.assertIn("VALUE", result_std.execute().columns)

        result_var = compute_moving_aggregation_narwhals(
            data=ibis_table,
            window_before=2,
            window_end=0,
            aggregation_method="var",
            input_col="INPUT",
            output_col="VALUE"
        )
        self.assertIn("VALUE", result_var.execute().columns)

    def test_ratio_calculation(self):
        """Test ratio calculation on mocked Ibis table."""
        mock_table = ibis.memtable(self.test_ratio_data)
        backend = _make_backend_with_mock_conn(mock_table)

        ibis_table = backend.load_data_from_snowflake("SELECT * FROM test_table")

        result = calculate_ratio_narwhals(
            data=ibis_table,
            numerator_col="NUMERATOR",
            denominator_col="DENOMINATOR",
            result_col="VALUE",
            method="ratio"
        )

        result_df = result.execute()

        self.assertIn("VALUE", result_df.columns)
        # First ratio: 10.0 / 5.0 = 2.0
        first_val = result_df.iloc[0]["VALUE"]
        self.assertAlmostEqual(first_val, 2.0, places=5)

    def test_ratio_methods(self):
        """Test all ratio calculation methods on mocked Ibis table."""
        mock_table = ibis.memtable(self.test_ratio_data)
        backend = _make_backend_with_mock_conn(mock_table)

        ibis_table = backend.load_data_from_snowflake("SELECT * FROM test_table")

        for method in ["ratio", "discrete_pct", "continuous_pct", "robust"]:
            with self.subTest(method=method):
                result = calculate_ratio_narwhals(
                    data=ibis_table,
                    numerator_col="NUMERATOR",
                    denominator_col="DENOMINATOR",
                    result_col="VALUE",
                    method=method
                )
                self.assertIn("VALUE", result.execute().columns)

    def test_ratio_division_by_zero(self):
        """Test that division by zero is handled correctly."""
        mock_table = ibis.memtable(self.test_ratio_data)
        backend = _make_backend_with_mock_conn(mock_table)

        ibis_table = backend.load_data_from_snowflake("SELECT * FROM test_table")

        result = calculate_ratio_narwhals(
            data=ibis_table,
            numerator_col="NUMERATOR",
            denominator_col="DENOMINATOR",
            result_col="VALUE",
            method="ratio"
        )

        result_df = result.execute()

        # Third row has denominator=0, should produce NaN
        third_val = result_df.iloc[2]["VALUE"]
        self.assertTrue(pd.isna(third_val))

    def test_connection_validation(self):
        """Test that connection parameter validation works."""
        backend = SnowflakeIbisBackend()

        # Should have required params from config
        is_valid = backend._validate_connection_params()
        self.assertTrue(is_valid)

        # Test with missing params
        backend.connection_params = {"user": "test"}
        is_valid = backend._validate_connection_params()
        self.assertFalse(is_valid)

    def test_lazy_execution(self):
        """Test that Ibis operations are lazy until execute() is called."""
        mock_table = ibis.memtable(self.test_mvag_data)
        backend = _make_backend_with_mock_conn(mock_table)

        ibis_table = backend.load_data_from_snowflake("SELECT * FROM test_table")

        result = compute_moving_aggregation_narwhals(
            data=ibis_table,
            window_before=1,
            window_end=0,
            aggregation_method="mean",
            input_col="INPUT",
            output_col="VALUE"
        )

        # Result should be an Ibis expression, not a DataFrame
        self.assertTrue(hasattr(result, 'execute'))

        result_df = result.execute()
        self.assertIsInstance(result_df, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
