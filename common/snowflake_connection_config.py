"""
Snowflake connection configuration.

Credentials have been removed for security reasons.
Replace the placeholder values with your Snowflake connection details.
"""

# Snowflake connection parameters
snowflake_connection_params = dict(
    user="<username>",
    account="<account>.<region>.privatelink",
    database="<database>",
    role="<role>",
    schema="<schema>",
    warehouse="<warehouse>",
    authenticator="externalBrowser",
)


def get_snowflake_connection_params():
    """
    Get Snowflake connection parameters.

    Returns:
        Dict with connection parameters ready to use
    """
    return snowflake_connection_params.copy()
