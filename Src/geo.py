import pandas as pd

def fill_geo_from_lookup(
    data,
    geo_cols=None,
    key_cols=("beat", "ward", "community_area"),
    required_col="district"
):
    """
    Backfill missing geographic attributes in a crime dataset using a beat‑level lookup.

    This function identifies rows with complete geographic information (including
    district) and builds a lookup table keyed on beat, ward, and
    community_area. It then merges this lookup back into the full dataset and
    fills missing values in geographic columns using the corresponding lookup
    values. Only columns with missing values are updated. Temporary merge columns
    are removed before returning the final DataFrame.

    Parameters
    ----------
    data : pd.DataFrame
        Input crime dataframe.
    geo_cols : list[str], optional
        Geographic columns to include in the lookup table.
    key_cols : tuple[str], optional
        Columns used to merge the df with the lookup table.
    required_col : str, optional
        Column that must be non-null for a row to be included in the lookup.

    Returns
    -------
    pd.DataFrame
        Updated the dataframe by filling in missing geo fields

    Notes
    -----
    - Rows missing a district are excluded from the lookup table to ensure
      reliable backfilling.
    - The merge uses beat, ward, and community_area to maintain a
      consistent geographic hierarchy.
    - Only columns ending in _filled (created during the merge) are used for
      backfilling, and they are removed afterward.
    """

    # Default geographic columns
    if geo_cols is None:
        geo_cols = [
            "beat", "ward", "community_area", "zip_code",
            "zip_code_area", "primary_neighborhood",
            "neighborhood_area", "district"
        ]

    # 1. Build a lookup table
    lookup = (
        data.dropna(subset=[required_col])
          .drop_duplicates(subset=["beat"])
          [geo_cols]
    )

    # 2. Merge
    data = data.merge(
        lookup,
        on=list(key_cols),
        how="left",
        suffixes=("", "_filled")
    )

    # 3. Vectorized backfill
    filled_cols = data.filter(regex="_filled$").columns
    # iterate
    for col in filled_cols:
        base_col = col.removesuffix("_filled")
        data[base_col] = data[base_col].fillna(data[col])

    # 4. Cleanup
    data.drop(columns=filled_cols, inplace=True)

    return data