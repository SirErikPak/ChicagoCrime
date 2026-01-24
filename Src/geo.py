import pandas as pd
import pyarrow as pa

# Define Arrow string dtype for consistent string handling
arrow_string = pd.ArrowDtype(pa.string())


def fill_from_composite_key(data: pd.DataFrame, key_col: list, target_str: str) ->  pd.DataFrame:
    """
    Fills missing values in a target column based on a mapping derived from a composite key.

    This function identifies unique associations between a set of key columns and a 
    target column. It builds a lookup table from rows where all values are present,
    validates that the mapping is not ambiguous (one key mapping to multiple values), 
    and then applies that mapping to the entire dataset using a vectorized approach.

    Args:
        data (pd.DataFrame): The input DataFrame containing the keys and target column.
        target_col (list): The name of the column to be filled (passed as a single-item list).
        key_cols (str): A list of column names used to create the composite lookup key.

    Returns:
        pd.DataFrame: The DataFrame with the target column updated based on the lookup.

    Raises:
        ValueError: If a composite key maps to more than one unique value in the target column.
    """
    # Initialize
    before = data[target_str].isna().sum()
    # Identify rows where both the keys and the target are present
    valid_mask = data[key_col + [target_str]].notna().all(axis=1)

    # Build lookup from complete rows 
    lookup = data.loc[valid_mask, key_col + [target_str]].drop_duplicates(subset=key_col)
    
    # Convert to string once, then add them together with the separator (lookup)
    lookup['composite_key'] = ["_".join(row) for row in lookup[key_col].astype(arrow_string).fillna('missing').values]
    
    # Convert to string once, then add them together with the separator (Orig)
    data['composite_key'] = ["_".join(row) for row in data[key_col].astype(arrow_string).fillna('missing').values]

    # Detect ambiguous composite-key mappings (keep=False only for rows that are completely unique)
    dupes = lookup.duplicated(subset=key_col, keep=False)

    # detect any composite key that maps to more than one target value
    if dupes.any():
        ambiguous = lookup.loc[dupes].sort_values(key_col)
        raise ValueError(
            f"Ambiguous mapping detected for keys {key_col}:\n{ambiguous}"
        )

    # Each composite key becomes the index &target_str value remains the column
    update_map = lookup.set_index('composite_key')[target_str]
    
    # Vectorized operation for performance
    update_mask = data[target_str].isna()
    data.loc[update_mask, target_str] = data.loc[update_mask, 'composite_key'].map(update_map)
    
    # Initialize
    after = data[target_str].isna().sum()
    difference = before - after
    print(f"--- Fill Summary: Column ({target_str}) {difference:,} missing values filled ---")

    return data.drop(columns=['composite_key'])



def fill_geo_from_lookup(data: pd.DataFrame, key_col: str, fill_cols: list) -> pd.DataFrame:
    """
    Fills missing values in specific columns by creating a lookup table 
    based on a unique key column and reports the number of rows affected.
    """
    # Record initial null counts for reporting (.isnull() produces a boolean DataFrame)
    initial_nulls = data[fill_cols].isnull().sum()

    # Build the lookup table
    lookup = (
        data.loc[data[key_col].notna(), [key_col] + fill_cols]
        .dropna()
        .drop_duplicates(subset=[key_col])
        .astype({key_col: arrow_string})
    )

    # Merge the lookup back onto the original data
    df_merged = data.merge(
        lookup, 
        on=key_col, 
        how="left", 
        suffixes=("", "_from_lookup")
    )

    # Fill NaNs and track changes
    for col in fill_cols:
        lookup_col_name = f"{col}_from_lookup"
        df_merged[col] = df_merged[col].fillna(df_merged[lookup_col_name])
    
    # Calculate how many rows were affected
    final_nulls = df_merged[fill_cols].isnull().sum()
    affected_counts = initial_nulls - final_nulls

    # Print summary
    print(f"--- Fill Summary (Key: {key_col}) ---")
    for col in fill_cols:
        print(f"Column '{col}': {(affected_counts[col]):,} rows filled.")

    # Cleanup
    cols_to_drop = [f"{col}_from_lookup" for col in fill_cols]
    df_merged = df_merged.drop(columns=cols_to_drop)

    return df_merged






# def location_fill(data: pd.DataFrame, geo_cols: list = None, key_index: str = None) -> pd.DataFrame:
#     """
#     Fills missing geographic attributes by mapping them from rows with the same location.
    
#     Args:
#         data: A pandas DataFrame containing a key_index & geo_cols column.
#         geo_cols: A list of columns to impute. If None, exit function.
#         key_index: A string column used for lookup
              
#     Returns:
#         A DataFrame with imputed missing values in the specified geographic columns.
#     """
#     # exit if none are provided for geo_cols & key_index
#     if geo_cols is None:
#         print("Lookup Table Column(s) List missing!")
#         return data
#     if key_index is None:
#         print("Key Index Column string missing!")
#         return data
        
#     # Build a lookup table 
#     lookup = (
#         data[[key_index] + geo_cols]         # Selected column(s) only
#         .dropna(subset=[key_index])          # Drop rows where the key is missing
#         .dropna(how='all', subset=geo_cols)  # Drop rows where all geo columns are null
#         .drop_duplicates(subset=[key_index]) # Keep only the first occurrence of each key
#         .set_index(key_index)                # Make the key the index
#     )
    
#     # Reindex the lookup to match the original 'location' series
#     # We use the original index to ensure alignment during the update
#     filler_data = lookup.reindex(data[key_index]).set_index(data.index)
    
#     # 3. Vectorized Fill
#     # combine_first fills nulls in 'data' with values from 'filler_data.' 
#     data[geo_cols] = data[geo_cols].combine_first(filler_data[geo_cols])

#     return data


# def fill_geo_from_lookup(
#     data: pd.DataFrame,
#     geo_cols: list = None,
#     key_cols: tuple = None,
#     required_col: str = None
# ):
#     """
#     Backfill missing geographic attributes in a crime dataset using a beat‑level lookup.

#     This function identifies rows with complete geographic information (including
#     district) and builds a lookup table keyed on beat, ward, and
#     community_area. It then merges this lookup back into the full dataset and
#     fills missing values in geographic columns using the corresponding lookup
#     values. Only columns with missing values are updated. Temporary merge columns
#     are removed before returning the final DataFrame.

#     Parameters
#     ----------
#     data : pd.DataFrame
#         Input crime dataframe.
#     geo_cols : list[str], optional
#         Geographic columns to include in the lookup table.
#     key_cols : tuple[str], optional
#         Columns used to merge the df with the lookup table.
#     required_col : str, optional
#         Column that must be non-null for a row to be included in the lookup.

#     Returns
#     -------
#     pd.DataFrame
#         Updated the dataframe by filling in missing geo fields

#     Notes
#     -----
#     - Rows missing a district are excluded from the lookup table to ensure
#       reliable backfilling.
#     - The merge uses beat, ward, and community_area to maintain a
#       consistent geographic hierarchy.
#     - Only columns ending in _filled (created during the merge) are used for
#       backfilling, and they are removed afterward.
#     """

#     # input checks
#     if geo_cols is None:
#         print("Geo Columns List missing!")
#         return data  
#     if key_cols is None:
#         print("Key Columns List missing!")
#         return data
#     if required_col is None:
#         print("Required Column string missing!")
#         return data 


#     # 1. Build a lookup table
#     lookup = (
#         data.dropna(subset=[required_col])
#           .drop_duplicates(subset=["beat"])
#           [geo_cols]
#     )

#     # 2. Merge
#     data = data.merge(
#         lookup,
#         on=list(key_cols),
#         how="left",
#         suffixes=("", "_filled")
#     )

#     # 3. Vectorized backfill
#     filled_cols = data.filter(regex="_filled$").columns
#     # iterate
#     for col in filled_cols:
#         base_col = col.removesuffix("_filled")
#         data[base_col] = data[base_col].fillna(data[col])

#     # 4. Cleanup
#     data.drop(columns=filled_cols, inplace=True)

#     return data