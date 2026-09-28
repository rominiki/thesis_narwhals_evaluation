"""
Integration tests for Modin Approach: Snowflake + Modin backend.

Uses mocking to test the integration without requiring actual Snowflake credentials.
Tests verify that:
1. Backend correctly initializes and manages connections
2. Data loading produces correct DataFrame types
3. Business logic functions work with Snowflake-backed DataFrames
4. Results are computed correctly
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

from modin_based_snowflake_integration.snowflake_modin_backend import SnowflakeModinBackend
from common.narwhals_mvag import compute_moving_aggregation_narwhals
from common.narwhals_ratio import calculate_ratio_narwhals


class TestSnowflakeModinIntegration(unittest.TestCase):
    """Test Snowflake Modin backend integration with mocked connections."""

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

    @patch('modin_based_snowflake_integration.snowflake_modin_backend.Session')
    def test_backend_initialization(self, mock_session_class):
        """Test that backend initializes without actual Snowflake connection."""
        backend = SnowflakeModinBackend()

        self.assertIsNotNone(backend.connection_params)
        self.assertIsNone(backend.session)  # Session not created until needed
        self.assertIn('account', backend.connection_params)
        self.assertIn('user', backend.connection_params)

    @patch('modin_based_snowflake_integration.snowflake_modin_backend.Session')
    @patch('modin_based_snowflake_integration.snowflake_modin_backend.pd.read_snowflake', create=True)
    def test_mvag_backward_window(self, mock_read_snowflake, mock_session_class):
        """Test moving aggregation with backward window on mocked Snowflake data."""
        # Mock the Snowpark session
        mock_session = MagicMock()
        mock_session_class.builder.configs.return_value.create.return_value = mock_session

        # Mock read_snowflake to return pandas DataFrame (will be converted to Modin internally)
        mock_read_snowflake.return_value = pd.DataFrame(self.test_mvag_data)

        # Initialize backend and load data
        backend = SnowflakeModinBackend()
        modin_df = backend.load_data_from_snowflake("SELECT * FROM test_table")

        # Verify read_snowflake was called
        mock_read_snowflake.assert_called_once()

        # Run moving aggregation
        result = compute_moving_aggregation_narwhals(
            data=modin_df,
            window_before=2,
            window_end=0,
            aggregation_method="mean",
            input_col="INPUT",
            output_col="VALUE"
        )

        # Convert to pandas for verification
        result_pd = result.to_pandas() if hasattr(result, 'to_pandas') else pd.DataFrame(result)

        # Verify result structure
        self.assertIn("VALUE", result_pd.columns)
        self.assertEqual(len(result_pd), len(self.test_mvag_data))

        # Verify first value for LISTING_ID=1 (mean of [1.0])
        first_val = result_pd[result_pd["LISTING_ID"] == 1].iloc[0]["VALUE"]
        self.assertAlmostEqual(first_val, 1.0, places=5)

    @patch('modin_based_snowflake_integration.snowflake_modin_backend.Session')
    @patch('modin_based_snowflake_integration.snowflake_modin_backend.pd.read_snowflake', create=True)
    def test_mvag_forward_window(self, mock_read_snowflake, mock_session_class):
        """Test moving aggregation with forward window on mocked Snowflake data."""
        mock_session = MagicMock()
        mock_session_class.builder.configs.return_value.create.return_value = mock_session
        mock_read_snowflake.return_value = pd.DataFrame(self.test_mvag_data)

        backend = SnowflakeModinBackend()
        modin_df = backend.load_data_from_snowflake("SELECT * FROM test_table")

        # Run with forward window
        result = compute_moving_aggregation_narwhals(
            data=modin_df,
            window_before=0,
            window_end=1,
            aggregation_method="sum",
            input_col="INPUT",
            output_col="VALUE"
        )

        result_pd = result.to_pandas() if hasattr(result, 'to_pandas') else pd.DataFrame(result)

        self.assertIn("VALUE", result_pd.columns)
        # First value for LISTING_ID=1 should be sum of [1.0, 2.0] = 3.0
        first_val = result_pd[result_pd["LISTING_ID"] == 1].iloc[0]["VALUE"]
        self.assertAlmostEqual(first_val, 3.0, places=5)

    @patch('modin_based_snowflake_integration.snowflake_modin_backend.Session')
    @patch('modin_based_snowflake_integration.snowflake_modin_backend.pd.read_snowflake', create=True)
    def test_mvag_std_var(self, mock_read_snowflake, mock_session_class):
        """Test std and var aggregations on mocked Snowflake data."""
        mock_session = MagicMock()
        mock_session_class.builder.configs.return_value.create.return_value = mock_session
        mock_read_snowflake.return_value = pd.DataFrame(self.test_mvag_data)

        backend = SnowflakeModinBackend()
        modin_df = backend.load_data_from_snowflake("SELECT * FROM test_table")

        # Test std
        result_std = compute_moving_aggregation_narwhals(
            data=modin_df,
            window_before=2,
            window_end=0,
            aggregation_method="std",
            input_col="INPUT",
            output_col="VALUE"
        )

        result_std_pd = result_std.to_pandas() if hasattr(result_std, 'to_pandas') else pd.DataFrame(result_std)
        self.assertIn("VALUE", result_std_pd.columns)

        # Test var
        result_var = compute_moving_aggregation_narwhals(
            data=modin_df,
            window_before=2,
            window_end=0,
            aggregation_method="var",
            input_col="INPUT",
            output_col="VALUE"
        )

        result_var_pd = result_var.to_pandas() if hasattr(result_var, 'to_pandas') else pd.DataFrame(result_var)
        self.assertIn("VALUE", result_var_pd.columns)

    @patch('modin_based_snowflake_integration.snowflake_modin_backend.Session')
    @patch('modin_based_snowflake_integration.snowflake_modin_backend.pd.read_snowflake', create=True)
    def test_ratio_calculation(self, mock_read_snowflake, mock_session_class):
        """Test ratio calculation on mocked Snowflake data."""
        mock_session = MagicMock()
        mock_session_class.builder.configs.return_value.create.return_value = mock_session
        mock_read_snowflake.return_value = pd.DataFrame(self.test_ratio_data)

        backend = SnowflakeModinBackend()
        modin_df = backend.load_data_from_snowflake("SELECT * FROM test_table")

        # Test basic ratio
        result = calculate_ratio_narwhals(
            data=modin_df,
            numerator_col="NUMERATOR",
            denominator_col="DENOMINATOR",
            result_col="VALUE",
            method="ratio"
        )

        result_pd = result.to_pandas() if hasattr(result, 'to_pandas') else pd.DataFrame(result)

        self.assertIn("VALUE", result_pd.columns)
        # First ratio: 10.0 / 5.0 = 2.0
        first_val = result_pd.iloc[0]["VALUE"]
        self.assertAlmostEqual(first_val, 2.0, places=5)

    @patch('modin_based_snowflake_integration.snowflake_modin_backend.Session')
    @patch('modin_based_snowflake_integration.snowflake_modin_backend.pd.read_snowflake', create=True)
    def test_ratio_methods(self, mock_read_snowflake, mock_session_class):
        """Test all ratio calculation methods on mocked Snowflake data."""
        mock_session = MagicMock()
        mock_session_class.builder.configs.return_value.create.return_value = mock_session
        mock_read_snowflake.return_value = pd.DataFrame(self.test_ratio_data)

        backend = SnowflakeModinBackend()
        modin_df = backend.load_data_from_snowflake("SELECT * FROM test_table")

        methods = ["ratio", "discrete_pct", "continuous_pct", "robust"]

        for method in methods:
            with self.subTest(method=method):
                result = calculate_ratio_narwhals(
                    data=modin_df,
                    numerator_col="NUMERATOR",
                    denominator_col="DENOMINATOR",
                    result_col="VALUE",
                    method=method
                )

                result_pd = result.to_pandas() if hasattr(result, 'to_pandas') else pd.DataFrame(result)
                self.assertIn("VALUE", result_pd.columns)

    @patch('modin_based_snowflake_integration.snowflake_modin_backend.Session')
    def test_connection_validation(self, mock_session_class):
        """Test that connection parameter validation works."""
        backend = SnowflakeModinBackend()

        # Should have required params from config
        is_valid = backend._validate_connection_params()
        self.assertTrue(is_valid)

        # Test with missing params
        backend.connection_params = {"user": "test"}
        is_valid = backend._validate_connection_params()
        self.assertFalse(is_valid)


if __name__ == "__main__":
    unittest.main()
