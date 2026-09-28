"""
Simple script to connect to Snowflake using Ibis, query <returns_table> table, and process with Narwhals.
"""

import sys
import os

sys.path.append("..")

from ibis_based_snowflake_integration.snowflake_ibis_backend import SnowflakeIbisBackend

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.narwhals_mvag import compute_moving_aggregation_narwhals


def main():

    backend = SnowflakeIbisBackend()


    query = """
    SELECT
        LISTING_ID,
        DATE,
        VALUE as INPUT
    FROM <returns_table>
    WHERE listing_id = <sample_listing_id> and date < '2010-05-01'
    LIMIT 1000
    """

    print("Loading data from Snowflake using Ibis...")
    ibis_table = backend.load_data_from_snowflake(query)

    print("\nProcessing with Narwhals...")
    result = compute_moving_aggregation_narwhals(
        data=ibis_table, window_before=3, window_end=2, aggregation_method="sum"
    )

    print("Executing Query In Snowflake (materialization)...")

    try:
        if hasattr(result, "execute"):  # means if result is an Ibis table
            executed_result = result.execute()
            executed_result = executed_result.sort_values(["LISTING_ID", "DATE"])
            print("\n Query executed successfully!")
            print(f" Result shape: {executed_result.shape}")
            print("\nSample results:")
            print(executed_result.head(10))
        else:
            print(result.head())

    except Exception as e:
        print(f"\n Could not execute query: {e}")
        raise

    return result


if __name__ == "__main__":
    main()
