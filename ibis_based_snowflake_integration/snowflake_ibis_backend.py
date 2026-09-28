"""
Snowflake Ibis backend integration for Approach 2.
Handles connection management and data loading using Ibis with Snowflake.
"""

from typing import Dict, Any, Optional

try:
    import ibis
    IBIS_AVAILABLE = True
    try:
        ibis.snowflake
        SNOWFLAKE_BACKEND_AVAILABLE = True
    except AttributeError:
        SNOWFLAKE_BACKEND_AVAILABLE = False
except ImportError:
    IBIS_AVAILABLE = False
    SNOWFLAKE_BACKEND_AVAILABLE = False
    ibis = None


class SnowflakeIbisBackend:
    """
    Backend for connecting to Snowflake using Ibis.

    This class manages Snowflake connections through Ibis and provides data loading
    functionality that works with the Narwhals abstraction layer.
    """

    def __init__(self, connection_params: Optional[Dict[str, Any]] = None):

        self.connection = None
        self._ibis = ibis

        self.connection_params = self._load_from_config()

        if connection_params:
            self.connection_params.update(connection_params)

        if not IBIS_AVAILABLE:
            print("Ibis not available - only mocked tests will work")
        elif not SNOWFLAKE_BACKEND_AVAILABLE:
            print("Ibis Snowflake backend not available - install with: pip install ibis-framework[snowflake]")
        else:
            print("Ibis framework with Snowflake backend available")

    def _load_from_config(self) -> Dict[str, Any]:

        try:
            import sys
            import os

            sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
            from common.snowflake_connection_config import (
                get_snowflake_connection_params,
            )

            config_params = get_snowflake_connection_params()
            print("Loaded connection configuration from snowflake_connection_config.py")

            return config_params

        except ImportError:
            print("snowflake_connection_config.py not found, using default template")

            return {
                "user": "YOUR_USER@COMPANY.COM",
                "account": "your_account.region.privatelink",
                "database": "YOUR_DATABASE",
                "role": "PUBLIC",
                "schema": "PUBLIC",
                "warehouse": "DEFAULT_WH",
                "authenticator": "externalBrowser",
            }

    def _validate_connection_params(self) -> bool:
        """
        Validate that required connection parameters are present.

        Returns:
            bool: True if all required parameters are present
        """
        required_params = ["account", "user", "database", "schema"]
        missing_params = [p for p in required_params if p not in self.connection_params]

        if missing_params:
            print(f"Missing required connection parameters: {missing_params}")
            print("Please update snowflake_connection_config.py with your Snowflake details")
            return False

        return True

    def _get_connection(self):
        """
        Get or create Ibis connection to Snowflake.
        Uses direct parameter passing instead of connection string.

        Returns:
            Ibis Snowflake connection
        """
        if self.connection is None:
            if not self._validate_connection_params():
                raise ValueError("Connection parameters not properly configured")

            conn_params = {
                "user": self.connection_params["user"],
                "account": self.connection_params["account"],
                "database": self.connection_params["database"],
                "schema": self.connection_params["schema"],
            }

            if "warehouse" in self.connection_params:
                conn_params["warehouse"] = self.connection_params["warehouse"]
            if "role" in self.connection_params:
                conn_params["role"] = self.connection_params["role"]

            if "authenticator" in self.connection_params:
                conn_params["authenticator"] = self.connection_params["authenticator"]
            elif "password" in self.connection_params:
                conn_params["password"] = self.connection_params["password"]
            else:
                print("No authentication method specified")

            self.connection = self._ibis.snowflake.connect(**conn_params)
            print("Ibis connection to Snowflake established")

        return self.connection

    def load_data_from_snowflake(self, query: str, connection_params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Load data from Snowflake using Ibis.

        Args:
            query: SQL query to execute
            connection_params: Optional connection parameters to override config

        Returns:
            Ibis Table expression with query results
        """
        if not self._ibis:
            raise ImportError("Ibis not available. Install with: pip install ibis-framework[snowflake]")

        if connection_params:
            self.connection_params.update(connection_params)
            self.connection = None

        try:
            conn = self._get_connection()

            print("Executing query...")

            result = conn.sql(query)

            print("Query executed successfully")
            print(f"Result type: {type(result)}")

            return result

        except Exception as e:
            print(f"Failed to load data from Snowflake: {e}")
            raise
