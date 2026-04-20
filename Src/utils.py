import pandas as pd
import textwrap

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



def melt_pivot_for_plotting(pivot_data: pd.DataFrame, id_vars: str, var_name: str):
    """
    Convert a wide pivot table (crime types per year) into a tidy long-format
    DataFrame suitable for plotting or statistical analysis.
    """
    data = (
        pivot_data
        .reset_index()               # normal column so it can be used as an identifier during melting
        .melt(                       #  converts wide/long format
            id_vars=id_vars,         # Columns to keep fixed — do NOT unpivot these. multiple identifiers: List
            var_name=var_name,       # Name of the new column that will contain the former column headers
            value_name='count'       # Name of the new column that will contain the cell values.”
        )
        .fillna(0)
        .convert_dtypes(dtype_backend='numpy_nullable')  # Convert each column to the NumPy‑nullable dtype
    )
    
    return data



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
    # index × columns crosstab
    pivot = (
        crime
            .pivot(index=rowB, columns=colA, values='count')
            .sort_index()
            .fillna(0)    # any crime type NaN columns, the pivot table will have NaN in that cell
    )                     # Replace missing count values with 0 (counts—missing means zero incidents)

    return pivot