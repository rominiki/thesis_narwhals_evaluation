"""
Test Snowflake pandas API (Modin) with Narwhals.
"""

import sys

sys.path.append("..")

from modin_based_snowflake_integration.snowflake_modin_backend import SnowflakeModinBackend
from common.narwhals_mvag import compute_moving_aggregation_narwhals


def main():
    backend = SnowflakeModinBackend()

    query = """
    SELECT LISTING_ID,
        DATE,
        VALUE as INPUT
    FROM <returns_table>
    WHERE listing_id = <sample_listing_id> and date < '2010-05-01'
    ORDER BY date desc LIMIT 1000
    """

    print("\n1. Loading data from Snowflake...")
    modin_df = backend.load_data_from_snowflake(query)

    print("\n2. Processing with Narwhals...")
    result = compute_moving_aggregation_narwhals(
        data=modin_df, window_before=3, window_end=2, aggregation_method="mean"
    )

    print("\n3. Materializing results...")
    print("This should be the point where the query runs in Snowflake.")

    local_result = result.to_pandas()

    print("\n Results materialized!")
    print(f"Shape: {local_result.shape}")  # (1000,4)
    print("\nSample results:")
    print(local_result.head())

    return result


if __name__ == "__main__":
    main()
