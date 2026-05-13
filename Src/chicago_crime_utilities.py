import pandas as pd
import textwrap
import pyarrow as pa         # arrow array construction + typed scalars
import pyarrow.compute as pc # C++ compute kernels

# Typed Arrow string dtype used by composite-key helpers below.
arrow_string = pd.ArrowDtype(pa.string())

# --------------------------------------------------------------------------------------------------
# 1. Identify and display columns with null values, optimized for large datasets (e.g., PyArrow-backed).
# --------------------------------------------------------------------------------------------------
def any_nans(data: pd.DataFrame) -> None:
    """
    Identifies and displays columns containing null values with their counts and percentages.
    
    Optimized for large datasets (e.g., PyArrow-backed) by using vectorized operations.
    
    Args:
        data: The pandas DataFrame to inspect.
    """
    total_rows = len(data)
    
    # 1. Get counts of nulls for all columns
    null_counts = data.isnull().sum()
    
    # 2. Filter for only columns that have at least one null
    null_counts = null_counts[null_counts > 0]
    
    if not null_counts.empty:
        print(f"--- Missing Values Found (Total Rows: {total_rows:,}) ---")
        
        # 3. Calculate percentage and build summary table
        percent = (null_counts / total_rows) * 100
        
        summary = pd.DataFrame({
            'Count': null_counts,
            'Percentage': percent.map("{:.4f}%".format)
        }).sort_values(by='Count', ascending=False)
        
        print(summary)
    else:
        print(f"Clean Dataset: No NaNs found across {total_rows:,} rows.")


# --------------------------------------------------------------------------------------------------
# 2. Print a wrapped, sorted list of unique values and the total count.
# --------------------------------------------------------------------------------------------------
def wrap_unique(df: pd.DataFrame, col: str, width: int = 80):
    """
    Prints a wrapped, sorted list of unique values and the total count.
    
    Args:
        df: The input DataFrame.
        col: The column name to inspect.
        width: The maximum character width for wrapping the output.
    """
    # Get unique values and drop nulls for a cleaner list
    s = df[col]
    unique_series = s.dropna().unique()
    
    # Sort using pandas' native sorting (handles Arrow/Extension types better)
    vals = pd.Series(unique_series).sort_values().tolist()
    
    # Check if we dropped nulls to inform the user
    null_count = s.isna().sum()
    unique_count = len(vals)
    
    # Format the list string
    vals_str = ", ".join(map(str, vals))
    wrapped_list = textwrap.fill(f"[{vals_str}]", width=width)
    
    # Build output
    stats = f"::::: Unique Count: {unique_count}"
    if null_count > 0:
        stats += f" (+ {null_count:,} nulls)"
        
    print(f"{wrapped_list}\n{stats}")


# --------------------------------------------------------------------------------------------------
# 3. Convert a wide pivot table (crime types per year) into a tidy long-format DataFrame suitable 
# for plotting or statistical analysis.
# --------------------------------------------------------------------------------------------------
def melt_pivot_for_plotting(pivot_data: pd.DataFrame, id_vars: str, var_name: str):
    """
    Convert a wide pivot table (crime types per year) into a tidy long-format
    DataFrame suitable for plotting or statistical analysis.
    """
    data = (
        pivot_data
        .reset_index()               # normal column so it can be used as an identifier during melting
        .melt(                       # converts wide/long format
            id_vars=id_vars,         # Columns to keep fixed - do NOT unpivot these. multiple identifiers: List
            var_name=var_name,       # Name of the new column that will contain the former column headers
            value_name='count'       # Name of the new column that will contain the cell values.”
        )
        .fillna(0)
        .convert_dtypes(dtype_backend='numpy_nullable')  # Convert each column to the NumPy‑nullable dtype
    )
    
    return data


# --------------------------------------------------------------------------------------------------
# 4. Create a crosstab of crime counts by year and crime type, optimized for large datasets (e.g., PyArrow-backed).
# --------------------------------------------------------------------------------------------------
def make_crime_year_crosstab(data: pd.DataFrame, colA: str, rowB: str):
    """
    Create a crosstab of crime counts.
    
    Parameters
    ----------
    data : pandas.DataFrame
    
    Returns
    -------
    pandas.DataFrame
        Pivot table
    """
    # Aggregate counts
    crime = (
        data.groupby([colA, rowB], observed=False)
          .size()
          .reset_index(name='count')  # GroupBy results return a Series with a MultiIndex 
    )                                 # converts the index levels back into normal columns
    
    # Cast Arrow categoricals to str for pivot compatibility
    crime[colA] = crime[colA].astype(str)
    crime[rowB] = crime[rowB].astype(str)

    # Pivot to wide format
    # index becomes the rows of the pivot table
    # columns becomes the columns  
    # values fill the cells of the matrix
    # index X columns crosstab
    pivot = (
        crime
            .pivot(index=rowB, columns=colA, values='count')
            .sort_index()
            .fillna(0)    # any crime type NaN columns, the pivot table will have NaN in that cell
    )                     # Replace missing count values with 0 (counts-missing means zero incidents)

    return pivot

# --------------------------------------------------------------------------------------------------
# 5. Deduplicate by case_number, keeping the latest 'updated_on', and generate a report of how many 
# records were deleted per case_number.
# --------------------------------------------------------------------------------------------------
def deduplicate_and_report(data_df, verbose=True, n_top=10):
    """
    Deduplicates by case_number, keeping the latest 'updated_on'.
    Optimized using Arrow backend logic.
    """
    # ------------------------------------------------------------
    # 5-1:  Copy the DataFrame to avoid modifying the original
    # ------------------------------------------------------------
    data = data_df.copy()

    # ------------------------------------------------------------
    # 5-2: Ensure updated_on is a datetime (critical for sorting)
    # ------------------------------------------------------------
    data['updated_on'] = pd.to_datetime(data['updated_on'])
    
    # ------------------------------------------------------------
    # 5-3: Count original records per case_number for the report
    # We use size() here as it is faster than value_counts() for this purpose
    # ------------------------------------------------------------
    original_counts = data.groupby('case_number', observed=True).size()

    # ------------------------------------------------------------
    # 5-4: Sort and Drop (Speed Optimization)
    # Sorting by updated_on DESC allows us to 'keep=first' effectively
    # Using engine='pyarrow' if available in your environment for the sort
    # ------------------------------------------------------------
    df_sorted = data.sort_values(
        by=['case_number', 'updated_on'], 
        ascending=[True, False]
    )
    
    # Keep the first (which is now the latest date)
    df_cleaned = df_sorted.drop_duplicates(subset=['case_number'], keep='first')

    # ------------------------------------------------------------
    # 5-5: Generate the Deletion Report
    # We can do this by comparing the original counts to the final counts after deduplication.
    # ------------------------------------------------------------
    final_counts = df_cleaned.groupby('case_number', observed=True).size()
    
    # Align counts and subtract to find deleted records
    report = (original_counts - final_counts).rename("records_deleted").reset_index()
    
    # Only include cases where deletions actually occurred
    report = report[report['records_deleted'] > 0].sort_values(by='records_deleted', ascending=False)

    if verbose:
    # Verbose reporting of the deduplication process
        print(f"{'='*60}")
        print(f"DEDUPLICATION SUMMARY (Arrow Backend)")
        print(f"{'='*60}")
        print(f"Original Rows : {len(data_df):,}")
        print(f"Cleaned Rows  : {len(df_cleaned):,}")
        print(f"Total Deleted : {(len(data_df) - len(df_cleaned)):,}")
        print(f"\nTOP {n_top} DELETIONS BY CASE NUMBER:")
        print(report.head(n_top).to_string(index=False))

    return {'df_cleaned':df_cleaned, 'report':report}


# --------------------------------------------------------------------------------------------------
# 6. Impute missing values using composite key lookup
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
    # ----------------------------------------------------------------
    # 6-A:  Surgical Copy
    # Shallow copy of full DataFrame; deep copy only the columns we mutate.
    # Avoids allocating memory for the entire DataFrame on 8.5M+ row inputs.
    # ----------------------------------------------------------------
    data = data_df.copy(deep=False)
    for col in update_cols:
        data[col] = data[col].copy()

    # ----------------------------------------------------------------
    # 1-B:  Composite Key via PyArrow C++ Kernel
    # pa.string() (32-bit offsets) is safe for short key strings at 8.5M rows
    # and avoids the typed-scalar mismatch that pa.large_string() requires.
    # ----------------------------------------------------------------
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

    # ----------------------------------------------------------------
    # 6-C: Build Lookup Table
    # Retain only rows where ALL update_cols are populated.
    # Column-select before drop_duplicates reduces memory carried through dedup.
    # keep='first' is significantly faster than mode on 8.5M rows.
    # ----------------------------------------------------------------
    look_up = (
        data.loc[~data[update_cols].isna().any(axis=1),
                 ['composite_key'] + update_cols]
            .drop_duplicates(subset=['composite_key'], keep='first')
            .set_index('composite_key')
    )

    # ----------------------------------------------------------------
    # 6-D: Pre-filter Target Rows 
    # Intersect mask with rows that have at least one null in update_cols.
    # Eliminates reindex overhead on rows that need no imputation - critical
    # when missing data is sparse relative to the full mask size.
    # ----------------------------------------------------------------
    target_mask = mask & data[update_cols].isna().any(axis=1)

    # ----------------------------------------------------------------
    # 6-E: Vectorized Imputation with Mapping
    # Both counters initialised before the conditional - prevents NameError
    # in the verbose block when target_mask is entirely False.
    # ----------------------------------------------------------------
    filled_count = 0
    updated_rows = set()   # unique row index - drives "Rows not updated" metric

    # Only attempt to map if there are target rows to update - avoids creating a 
    # large intermediate Series of NaNs when target_mask is empty.
    if target_mask.any():
        # One reindex across ALL columns - replaces N separate .map() calls.
        target_keys         = data.loc[target_mask, 'composite_key']
        imputed_block       = look_up.reindex(target_keys)
        imputed_block.index = target_keys.index   # restore original index

        # Loop through update_cols once, filling from the imputed block where nulls exist.
        for col in update_cols:
            # Only fill cells that are null in original AND matched in lookup.
            null_here = data.loc[target_mask, col].isna()
            fill_vals = imputed_block.loc[null_here, col].dropna()  # called once

            # fill_vals is a Series with the same index as data, but only non-null 
            # where we have a match to fill.
            if not fill_vals.empty:
                data.loc[fill_vals.index, col] = fill_vals
                filled_count += len(fill_vals)
                updated_rows.update(fill_vals.index.tolist())   # O(1) per row

    # ----------------------------------------------------------------
    # 6-F: Cleanup
    # ----------------------------------------------------------------
    data.drop(columns=['composite_key'], inplace=True)

    # ----------------------------------------------------------------
    # 6-G: Reporting - only meaningful when verbose and multiple 
    # update_cols (filled_count is total cells filled, not rows updated)
    # ---------------------------------------------------------------- 
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


# --------------------------------------------------------------------------------------------------
# 7. Fill missing values using composite key lookup
# --------------------------------------------------------------------------------------------------
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


# --------------------------------------------------------------------------------------------------
# 8. Report NaN counts and unique value distributions for specified columns.
# --------------------------------------------------------------------------------------------------
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