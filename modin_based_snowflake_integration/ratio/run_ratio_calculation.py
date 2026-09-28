"""
Test Snowflake pandas API (Modin) with Narwhals for ratio calculation.
"""

import sys

sys.path.append("..")

from modin_based_snowflake_integration.snowflake_modin_backend import SnowflakeModinBackend
from common.narwhals_ratio import calculate_ratio_narwhals


def main():
    backend = SnowflakeModinBackend()

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

    print("\n1. Loading data from Snowflake...")
    modin_df = backend.load_data_from_snowflake(query)

    print("\n2. Processing with Narwhals...")
    result = calculate_ratio_narwhals(
        data=modin_df,
        numerator_col="NUMERATOR",
        denominator_col="DENOMINATOR",
        result_col="VALUE",
        method="ratio",
    )

    print("\n3. Materializing results...")
    print("This should be the point where the query runs in Snowflake.")

    local_result = result.to_pandas()

    print("\n Results materialized!")
    print(f"Shape: {local_result.shape}")
    print("\nSample results:")
    print(local_result.head())

    return result


if __name__ == "__main__":
    main()
