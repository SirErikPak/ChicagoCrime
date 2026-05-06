import pandas as pd          # data manipulation
import pyarrow as pa         # arrow array construction + typed scalars
import pyarrow.compute as pc # C++ compute kernels


# Typed Arrow string dtype used by composite-key helpers below.
arrow_string = pd.ArrowDtype(pa.string())


# --------------------------------------------------------------------------------------------------
# Impute missing values using composite key lookup
# --------------------------------------------------------------------------------------------------
def impute_data(
    data_df:     pd.DataFrame,
    keys:        list,
    update_cols: list,
    mask:        pd.Series,
    verbose:     bool = True,
) -> pd.DataFrame:
    """
    Impute missing values in ``update_cols`` using a composite-key lookup.

    The function fills missing values only when both conditions are true:
    1. The row is selected by ``mask``.
    2. A matching row with the same composite key has complete ``update_cols`` data.

    Duplicate key names are removed while preserving the first occurrence.
    Every key must exist as a DataFrame column, and every update column must
    exist in the input. Existing non-null values are never overwritten.

    Args:
        data_df (pd.DataFrame): Input DataFrame. The function works on a copy.
        keys (list): Columns used to build the composite key. Duplicate names
            are removed while preserving order.
        update_cols (list): Columns whose missing values should be imputed.
        mask (pd.Series): Boolean Series marking rows eligible for imputation.
        verbose (bool): Print imputation summary. Defaults to True.

    Returns:
        pd.DataFrame: Copy of the DataFrame with missing values filled where a
            match was found. Original non-null values are preserved.
    """
    # ── 0. Surgical Copy ──────────────────────────────────────────────────────
    # Shallow copy of full DataFrame; deep copy only the columns we mutate.
    # Avoids allocating memory for the entire DataFrame on 8.5M+ row inputs.
    data = data_df.copy(deep=False)
    for col in update_cols:
        data[col] = data[col].copy()

    # ── 1. Composite Key via PyArrow C++ Kernel ───────────────────────────────
    # pa.string() (32-bit offsets) is safe for short key strings at 8.5M rows
    # and avoids the typed-scalar mismatch that pa.large_string() requires.
    pa_arrays = []
    for k in keys:
        series = data[k]
        if isinstance(series.dtype, pd.ArrowDtype):
            arr = series.array._pa_array.combine_chunks()
        else:
            arr = pa.array(series, from_pandas=True)                # NaN-safe
        pa_arrays.append(pc.cast(arr, pa.string()))

    # null_handling="emit_null" - if ANY key column is null the composite key
    # is null, preventing false matches against unrelated rows in the lookup.
    # Matches pandas null-propagation semantics of the original version.
    composite_pa          = pc.binary_join_element_wise(
        *pa_arrays, "_", null_handling="emit_null"
    )
    data['composite_key'] = pd.arrays.ArrowExtensionArray(composite_pa)

    # ── 2. Build Lookup Table ─────────────────────────────────────────────────
    # Retain only rows where ALL update_cols are populated.
    # Column-select before drop_duplicates reduces memory carried through dedup.
    # keep='first' is significantly faster than mode on 8.5M rows.
    look_up = (
        data.loc[~data[update_cols].isna().any(axis=1),
                 ['composite_key'] + update_cols]
            .drop_duplicates(subset=['composite_key'], keep='first')
            .set_index('composite_key')
    )

    # ── 3. Pre-filter Target Rows ─────────────────────────────────────────────
    # Intersect mask with rows that have at least one null in update_cols.
    # Eliminates reindex overhead on rows that need no imputation - critical
    # when missing data is sparse relative to the full mask size.
    target_mask = mask & data[update_cols].isna().any(axis=1)

    # ── 4. Vectorized Imputation ──────────────────────────────────────────────
    # Both counters initialised before the conditional - prevents NameError
    # in the verbose block when target_mask is entirely False.
    filled_count = 0
    updated_rows = set()   # unique row index - drives "Rows not updated" metric

    if target_mask.any():
        # One reindex across ALL columns - replaces N separate .map() calls.
        target_keys         = data.loc[target_mask, 'composite_key']
        imputed_block       = look_up.reindex(target_keys)
        imputed_block.index = target_keys.index   # restore original index

        for col in update_cols:
            # Only fill cells that are null in original AND matched in lookup.
            null_here = data.loc[target_mask, col].isna()
            fill_vals = imputed_block.loc[null_here, col].dropna()  # called once

            if not fill_vals.empty:
                data.loc[fill_vals.index, col] = fill_vals
                filled_count += len(fill_vals)
                updated_rows.update(fill_vals.index.tolist())   # O(1) per row

    # ── 5. Cleanup ────────────────────────────────────────────────────────────
    data.drop(columns=['composite_key'], inplace=True)

    # ── 6. Report ─────────────────────────────────────────────────────────────
    # Make it conditional on number of columns
    if verbose:
        before_counts = int(mask.sum())
        rows_updated  = len(updated_rows)
        print(f"{'-' * 5} Imputation complete {'-' * 5}")
        print(f"Total rows in mask: {before_counts:,}")
        print(f"Rows updated:       {rows_updated:,}")
        print(f"Rows not updated:   {before_counts - rows_updated:,}")
        if len(update_cols) > 1:
            print(f"Values imputed:     {filled_count:,}")  # only meaningful for multi-col
 
    return data



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
    valid_bool_mask = data[key_col + [target_str]].notna().all(axis=1)

    # Build lookup from complete rows 
    lookup = data.loc[valid_bool_mask, key_col + [target_str]].drop_duplicates(subset=key_col)
    
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
    update_bool_mask = data[target_str].isna()
    data.loc[update_bool_mask, target_str] = data.loc[update_bool_mask, 'composite_key'].map(update_map)
    
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