"""
Simple script to connect to Snowflake using Ibis, query dividend and price tables,
and compute ratio with Narwhals.
"""

import sys
import os

sys.path.append("..")

from ibis_based_snowflake_integration.snowflake_ibis_backend import SnowflakeIbisBackend

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from common.narwhals_ratio import calculate_ratio_narwhals


def main():

    backend = SnowflakeIbisBackend()

    query = """
    SELECT n.LISTING_ID,
                     d.DATE,
           n.VALUE as NUMERATOR,
           d.VALUE as DENOMINATOR
    FROM <dividend_table> n
    JOIN <price_table> d
            ON n.LISTING_ID = d.LISTING_ID
         AND d.DATE BETWEEN n.FROM_DATE AND n.TO_DATE
            WHERE n.listing_id = <sample_listing_id> AND d.date < '2026-05-01'
    LIMIT 1000
    """

    print("\nLoading data from Snowflake using Ibis...")
    ibis_table = backend.load_data_from_snowflake(query)

    print("\nProcessing with Narwhals...")
    result = calculate_ratio_narwhals(
        data=ibis_table,
        numerator_col="NUMERATOR",
        denominator_col="DENOMINATOR",
        result_col="VALUE",
        method="ratio",
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
