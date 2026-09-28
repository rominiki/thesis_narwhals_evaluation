"""
Snowflake pandas API backend integration for Approach 1.
Uses Snowflake's pandas API (Modin) for server-side execution.
"""

from typing import Dict, Any, Optional

from modin.config import AutoSwitchBackend

AutoSwitchBackend.disable()


try:
    from snowflake.snowpark import Session
    import snowflake.snowpark.modin.plugin
    SNOWFLAKE_AVAILABLE = True
except ImportError:
    SNOWFLAKE_AVAILABLE = False
    Session = None

import modin.pandas as pd


class SnowflakeModinBackend:
    def __init__(self, connection_params: Optional[Dict[str, Any]] = None):
        """Initialize Snowflake pandas API backend."""
        self.connection_params = self._load_from_config()

        if connection_params:
            self.connection_params.update(connection_params)

        self.session = None  # Stores Snowpark Session

        if not SNOWFLAKE_AVAILABLE:
            print("Snowflake packages not installed - only mocked tests will work")
        else:
            print("Snowflake pandas API (Modin) backend initialized")

    def _load_from_config(self) -> Dict[str, Any]:
        """Load connection parameters from config file."""
        try:
            from common.snowflake_connection_config import (
                get_snowflake_connection_params,
            )

            config_params = get_snowflake_connection_params()
            print("Loaded connection configuration")
            return config_params
        except ImportError:
            print("Config file not found")
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
        """Validate required connection parameters."""
        required_params = ["account", "user", "database", "schema"]
        missing_params = [p for p in required_params if p not in self.connection_params]

        if missing_params:
            print(f"Missing required parameters: {missing_params}")
            return False
        return True

    def _get_or_create_session(self):
        """Get existing Snowpark Session or create new one."""
        if not SNOWFLAKE_AVAILABLE:
            if Session is None:
                raise ImportError(
                    "Snowflake packages not installed. "
                    "Install with: pip install snowflake-snowpark-python"
                )

        if self.session is None:
            if not self._validate_connection_params():
                raise ValueError("Connection parameters not configured")

            print("Creating Snowpark Session...")
            self.session = Session.builder.configs(self.connection_params).create()
            print("Snowpark Session created (required for Modin pandas API)")

        return self.session

    def load_data_from_snowflake(self, query: str) -> Any:

        try:
            self._get_or_create_session()

            print("Reading from Snowflake using pandas API...")

            df = pd.read_snowflake(query)

            print("Created lazy Modin DataFrame")
            print("(Data still in Snowflake, operations will be lazy)")

            return df

        except Exception as e:
            print(f"Failed to load data: {e}")
            raise
