import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import scipy.cluster.hierarchy as hierarchy
from scipy.spatial.distance import pdist, squareform
from matplotlib.colors import LinearSegmentedColormap
from scatterd import scatterd
# Python file imports
import hierarchy_clustering
from typing import Tuple

# ---------------------------------------------------------------------
# 1. Fancy PCA Plot
# ---------------------------------------------------------------------
def fancy_pca_plot(
    data: pd.DataFrame,
    pc_scores: np.ndarray,
    pca_model_object: object,
    labels: np.ndarray = None,
    file: str = None,
    point_size: int = 90,
    font_size: int = 14,
    density: bool = True
):
    """
    Enhanced PCA scatter plot with optional density overlay and sorted loadings.

    Parameters
    ----------
    data : pd.DataFrame
        Original dataset used for PCA (columns = variables).
    pc_scores : np.ndarray
        PCA-transformed coordinates (n_samples x n_components).
    pca_model_object : object
        Fitted PCA model (must expose `.components_` and `.explained_variance_ratio_`).
    labels : array-like, optional
        Cluster IDs or group labels for coloring points.
    file : str, optional
        If provided, the plot will be saved to `../Image/{file}.png`.
    point_size : int
        Marker size.
    font_size : int
        Font size for labels.
    density : bool
        Whether to overlay a KDE density estimate behind the scatter.
    """

    # Compute and sort PCA loadings (features × PCs) for interpretation.
    # The DataFrame is constructed from the PCA components and indexed by
    # original feature names so downstream code can inspect which features
    # drive PC1.
    loadings = pd.DataFrame(
        pca_model_object.components_.T,
        index=data.columns,
        columns=[f"PC{i+1}" for i in range(pca_model_object.n_components_)]
    ).sort_values("PC1", ascending=False)

    # Density settings (only if enabled). We pass these through to `scatterd`
    # which will render a filled contour KDE behind the scatter when provided.
    args_density = (
        {'fill': True, 'thresh': 0, 'levels': 100, 'cmap': "vlag"}
        if density else None
    )

    # Build scatter plot with optional density overlay
    fig, ax = scatterd(
        pc_scores[:, 0],
        pc_scores[:, 1],
        labels=labels,
        grid=None,
        fontcolor='k',
        fontsize=font_size,
        s=point_size,
        verbose=0,
        args_density=args_density
    )

    # Axis labels with explained variance
    evr = pca_model_object.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({evr[0]*100:.2f}% Explained Variance)", fontsize=font_size)
    ax.set_ylabel(f"PC2 ({evr[1]*100:.2f}% Explained Variance)", fontsize=font_size)

    # display explained variance (print to console for quick inspection)
    print("----- PCA Variance Explained -----")
    print(f"Total PC1 & PC2 Variance: {pca_model_object.explained_variance_ratio_[:2].sum()*100:.2f}%")
    print("Explained Variance Ratios:", pca_model_object.explained_variance_ratio_)

    # Legend for clusters if labels are provided
    if labels is not None:
        ax.legend(
            title="Cluster",
            loc="upper left",
            fontsize=font_size - 2,
            title_fontsize=font_size,
            frameon=False
        )
    # Create a small DataFrame of the first two PC coordinates indexed by
    # the original sample index; useful for downstream plotting or labeling.
    df_pca = pd.DataFrame(data=pc_scores[:, :2], columns=['PC1', 'PC2'], index=data.index)

    # Save file if requested
    if file:
        plt.savefig(f"../Image/{file}.png", dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.show()

    return loadings, df_pca

# ---------------------------------------------------------------------
# 2. Correlation Heatmap
# ---------------------------------------------------------------------
def correlation_heatmap(data: pd.DataFrame, figsize: tuple=(10,8)) -> None:
    """
    Plot a heatmap of pairwise correlation distances between rows of a DataFrame.

    This function computes the correlation distance (1 - Pearson r) between all
    pairs of rows in the input DataFrame, converts the condensed distance vector
    into a full square distance matrix, and visualizes it as a heatmap. The
    diagonal is bool_masked because self-distances are always zero and not meaningful
    for interpretation. A custom green->yellow->red colormap is used to highlight
    low, medium, and high distances on a fixed scale from 0 to 2.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame where each row represents an observation (e.g., a crime type)
        and each column represents a feature or time point (e.g., yearly Z-scores).
        Correlation distances are computed between rows.
    
    figsize : tuple, optional
        Size of the resulting heatmap figure in inches. Default is (10, 8).

    Notes
    -----
    - Correlation distance is defined as:  d = 1 - corr(x, y)
      Values range from:
        0 -> perfectly correlated (same shape)
        1 -> uncorrelated
        2 -> perfectly anti-correlated (opposite shape)
    - The diagonal is bool_masked because each row has distance 0 to itself.
    - The heatmap uses a fixed color scale (0 to 2) for comparability across runs.

    Returns
    -------
    None
        Displays the heatmap using matplotlib.
    """
    
    # Compute condensed distance matrix (correlation distance)
    dist_condensed = pdist(data, metric='correlation')
    
    # Full distance matrix
    dist_matrix = squareform(dist_condensed)
    
    # Custom colormap: green -> yellow -> red
    colors = [(0, 1, 0), (1, 1, 0), (1, 0, 0)]  # RGB: green -> yellow -> red
    cmap = LinearSegmentedColormap.from_list('pos_neutral_neg', colors, N=256)

    # diagonal boolean bool_mask - meaningful distances off the diagonal
    bool_mask = np.eye(dist_matrix.shape[0], dtype=bool)
    
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        dist_matrix,
        mask=bool_mask,
        xticklabels=data.index,
        yticklabels=data.index,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        cbar_kws={'label': 'Correlation Distance (1 - r)'},
        vmin=0, vmax=2  # fixed scale for all matrices
    )
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.title('Pairwise Correlation Distance', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# 3. Hierarchical Clustering Dendrogram 
# ---------------------------------------------------------------------
def plot_clustering(
    data: pd.DataFrame, 
    method: str, 
    metric: str, 
    figsize: tuple=(8,4), 
    rotation: int=45
) -> None:
    """Perform hierarchical clustering and plot a dendrogram.

    Parameters
    ----------
    data : pd.DataFrame
        Rows are observations and columns are features.
    method : str
        Linkage method ('single', 'complete', 'average', 'ward', etc.).
    metric : str
        Distance metric ('euclidean', 'correlation', etc.).
    figsize : tuple, optional
        Figure size in inches.
    rotation : int, optional
        Rotation angle for leaf labels.

    Returns
    -------
    None
        Displays a dendrogram plot. The underlying linkage matrix is computed
        via `hierarchy_clustering.linkage_matrix` but is not returned.
    """
    # call helper linkage matrix function
    Z = hierarchy_clustering.linkage_matrix(data, method, metric)
    
    # Plot dendrogram
    plt.figure(figsize=figsize)
    hierarchy.dendrogram(Z, labels=data.index.tolist(), leaf_rotation=rotation)
    
    # Fix label alignment
    for lbl in plt.gca().get_xticklabels(): 
        lbl.set_ha('right')
    
    plt.title(f'Hierarchical Clustering (Linkage: {method.capitalize()})')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# 4. Melt Pivot Table for Plotting  
# ---------------------------------------------------------------------
def bar_plot(data: pd.DataFrame, 
             count: str, 
             index: str, 
             crime_code: str, 
             column: str,
             col_wrap: int=3) -> None:
    """
    Creates a faceted horizontal bar plot showing crime distributions across different categories.

    This function uses Seaborn's catplot to generate a grid of bar charts. Each subplot 
    represents a unique value from the 'column' parameter (e.g., a specific District), 
    displaying the frequency of incidents for various crime codes.

    Args:
        data (pd.DataFrame): The source dataframe containing crime statistics.
        count (str): The column name representing the numerical frequency of incidents
            (plotted on the x-axis).
        index (str): Controls whether the data is treated as indexed when creating
            the title. Pass `'I'` to indicate indexed data; otherwise pass any
            other value.
        crime_code (str): The column name representing the categorical crime
            classification (plotted on the y-axis and used for color encoding).
        column (str): The column name used to create the faceted grid (e.g., 'District').

    Returns:
        None: The function displays the plot using plt.show().

    Note:
        The function assumes 'matplotlib.ticker' is imported as 'mtick' and 
        'seaborn' as 'sns'. It automatically applies 'viridis' styling, 
        adds comma-formatted value labels to the bars, and wraps the grid 
        at the specified number of columns.
    """
    # Use a context manager for temporary styling
    # This prevents your function from permanently changing global plot settings
    with sns.axes_style("whitegrid"):
        g = sns.catplot(
            data=data, 
            kind="bar",
            x=count, 
            y=crime_code, 
            hue=crime_code,
            col=column, 
            col_wrap=col_wrap,
            height=4, 
            aspect=1.5,
            palette="viridis",
            sharex=False,
            legend=False
        )

    # Index or Non-Indexed
    if index == 'I':
        head = 'Indexed'
    else:
        head = 'Non-Indexed'

    # Overall tile
    g.fig.suptitle(f"FBI {head} Crime Distribution by District (2001-2025)", 
                   fontsize=22, fontweight="bold")
    # Each plot title
    g.set_titles("{col_name} District", size=16, fontweight="bold", pad=20) 
    
    # Iterate through each subplot (ax) to add labels and formatting
    for ax in g.axes.flatten(): 
        ax.set_xlabel("Number of Incidents", fontsize=12)
        ax.set_ylabel("FBI Crime Code", fontsize=12)
        ax.tick_params(labelbottom=True)
        
        # Add actual counts at the end of bars with comma formatting
        for container in ax.containers:
            ax.bar_label(container, fmt='{:,.0f}', padding=4, fontsize=9)
        
        # Format the X-axis ticks with commas as well
        ax.xaxis.set_major_formatter(mtick.StrMethodFormatter('{x:,.0f}'))

    # Final layout adjustment
    plt.subplots_adjust(hspace=0.6, top=0.9) # Manual control often beats tight_layout in FacetGrids
    # Save the figure to a file
    plt.savefig(f"../Image/district_bar.png", dpi=600)
    plt.show()


# ---------------------------------------------------------------------
# 5. Line Plot with Peak Indicators
# ---------------------------------------------------------------------
def line_plot(data: pd.DataFrame, column_name: str, category_name: str, numeric_name: str, 
                  image_path: str = None, col_wrap=4, rotation: int=45, ha='right') -> None:
    """
    Creates a faceted line plot for time-series crime data with peak indicators.

    This function generates a grid of subplots (one for each unique value in column_name),
    draws a line plot showing trends over category_name, and marks the maximum 
    value in each facet with a vertical dashed line.

    Args:
        data (pd.DataFrame): The tidy (long-form) DataFrame containing the data.
        column_name (str): The column used to create the grid (e.g., 'fbi_code_desc').
        category_name (str): The x-axis variable (e.g., 'month' or 'quarter').
        numeric_name (str): The y-axis variable (e.g., 'count').

    """     
    # Create the FacetGrid
    # 'sharex=True' to keep category alignment, but 'sharey=False' 
    # to see trends in low-volume categories.
    g = sns.FacetGrid(
        data,
        col=column_name,
        col_wrap=col_wrap,
        hue=column_name,
        sharey=False,
        height=4,
        aspect=1.2
    )
    # initializ
    if image_path is not None:
        image_name = image_path + "yeary_crimne_type_line_plot.png"
            
    # Map the lineplot
    g.map(sns.lineplot, category_name, numeric_name, marker='o')
    
    # Optimize: Use GroupBy to avoid repeated manual filtering
    # This aligns the vertical lines perfectly with the faceted plots.
    grouped = data.groupby(column_name, sort=False)
    
    for ax, (name, facet_data) in zip(g.axes.flatten(), grouped):
        if not facet_data.empty:
            # Find the peak
            max_idx = facet_data[numeric_name].idxmax()
            max_cat = facet_data.loc[max_idx, category_name]
            peak_val = facet_data.loc[max_idx, numeric_name]
            
            # Vertical line and Peak label
            ax.axvline(x=max_cat, color='black', linestyle='--', alpha=0.5, zorder=0)
            ax.text(
                x=max_cat, 
                y=peak_val, 
                s=f' {peak_val:,.0f}', 
                va='bottom', 
                fontsize=9, 
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
            )

        # Force x-labels on EVERY subplot
        ax.tick_params(labelbottom=True, labelsize=9)
        # Apply labels and rotation
        ax.set_xlabel(category_name.capitalize())
        ax.set_ylabel(numeric_name.capitalize())
        plt.setp(ax.get_xticklabels(), rotation=rotation, ha=ha)

    # Add Title
    g.set_titles("{col_name}")
    
    # Tight layout handles spacing, but we add hspace for the extra x-labels
    plt.subplots_adjust(hspace=0.7)
    g.tight_layout()
    # Save the figure to a file
    if image_path:
        plt.savefig(f"{image_name}", dpi=600)
        print(f"Full plot saved to: {image_name}")
        print()
    plt.show()


# ---------------------------------------------------------------------
# 6. Stacked Bar Plot with Percentage Labels
# ---------------------------------------------------------------------
def stacked_bar_plot(data_df, title='', image_path = None,
                     figsize=(12, 8), cmap='Set2', sort_index=True):
    """
    Generates a horizontal stacked bar chart showing percentage distribution per row,
    sorted alphabetically by the index.

    Args:
        data (pd.DataFrame): DataFrame where index is the category and columns are sub-groups.
        title (str): The title of the plot.
        cmap (str): The Matplotlib colormap name.
        sort_index (bool): If True, sorts the y-axis alphabetically (A-Z from top to bottom).

    Returns:
        None: Displays a Matplotlib plot.
    """
    if  image_path is not None:
        # initialize 
        image_name = image_path + "stack_bar_crime_plot.png"
    
    # Sort Data by Index
    # ascending=False ensures 'A' is at the top and 'Z' is at the bottom in barh
    if sort_index:
        data = data_df.sort_index(ascending=False)

    # Calculate row-wise percentages
    # Convert counts to percentages per row so the stacked bar sums to 100%
    data_pct = data.div(data.sum(axis=1), axis=0) * 100
    
    # Setup the figure and primary plot
    ax = data_pct.plot(
        kind='barh', 
        stacked=True, 
        figsize=figsize, 
        colormap=cmap,
        edgecolor='white', 
        linewidth=0.5
    )
    
    # Pre-calculate the maximum column for each row for highlighting
    max_cols = data_pct.idxmax(axis=1)
    
    # Annotate bar segments
    for i, container in enumerate(ax.containers):
        col_name = data_pct.columns[i]
        
        for j, bar in enumerate(container):
            width = bar.get_width()
            if width <= 2.0: # Skip labels for tiny segments to prevent clutter
                continue
            
            is_max = (col_name == max_cols.iloc[j])
            text_color = 'red' if is_max else 'white'
            
            ax.text(
                bar.get_x() + width/2, 
                bar.get_y() + bar.get_height()/2, 
                f'{width:.1f}%', 
                ha='center', 
                va='center', 
                color=text_color, 
                weight='bold' if is_max else 'normal',
                fontsize=9
            )
    
    # Aesthetics
    plt.title(f'Relative Distribution of Crimes Across {title}', fontsize=14, pad=15)
    plt.xlabel('Percentage of Total Count', fontweight='bold')
    plt.ylabel('FBI Crime Type', fontweight='bold')
    plt.legend(title=title, bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    plt.xlim(0, 100)
    plt.tight_layout()
    if image_path:
        plt.savefig(image_name, dpi=300, bbox_inches='tight')
        print(f"Full plot saved to: {image_name}")
        print()
    plt.show()


# ---------------------------------------------------------------------
# 7. Inconsistency Plot
# ---------------------------------------------------------------------
def plot_cuts(heights: np.array, incons: np.array) -> None:
    """Plot inconsistency profile across linkage merge heights.

    Parameters
    ----------
    heights : np.ndarray
        Array of merge heights (from hierarchical clustering).
    incons : np.ndarray
        Array of inconsistency coefficients corresponding to each merge.

    Returns
    -------
    None
        Displays a line plot of inconsistency vs merge height.
    """
    # Sort by height (optional but makes the plot cleaner)
    order = np.argsort(heights)
    h_sorted = heights[order]
    i_sorted = incons[order]
    
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(h_sorted, i_sorted, marker='o', linestyle='-', color='steelblue')
    
    plt.axhline(1.0, color='red', linestyle='--', label='Inconsistency = 1.0')
    plt.xlabel("Merge Height")
    plt.ylabel("Inconsistency Coefficient")
    plt.title("Inconsistency Profile Across Merge Heights")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


# ---------------------------------------------------------------------
# 8. Crime Count Plots
# ---------------------------------------------------------------------
def plot_crime_counts(
    esp_results: dict,
    column: str = "Gambling",
    figsize: Tuple[int, int] = (14, 8),
    bins: int = 40,
    show: bool = True,
) -> dict:
    """Plot a pivot column's raw time series and its non-zero distribution.

    This helper extracts a single column (time series) from
    ``esp_results['pivot_data']``, computes basic zero/non-zero summary
    statistics, plots the raw counts and a histogram of non-zero values,
    and returns the computed objects for downstream inspection or tests.

    Parameters
    ----------
    esp_results : dict
        Mapping expected to contain key ``'pivot_data'`` whose value is a
        pandas-compatible mapping (DataFrame or dict-like) of time series.
    column : str, optional
        The pivot column to analyze and plot. Default is ``'Gambling'``.
    figsize : tuple[int, int], optional
        Figure size passed to ``plt.subplots``. Defaults to ``(14, 8)``.
    bins : int, optional
        Number of bins used for the non-zero histogram. Defaults to 40.
    show : bool, optional
        If True (default), call ``plt.show()`` before returning.

    Returns
    -------
    dict
        A dictionary with the following keys:
        - ``series`` : pd.Series   -- the extracted series (index reset)
        - ``stats``  : dict        -- summary statistics (n_total, n_zeros, n_nonzero,
                                     mean_nonzero, median_nonzero, max_nonzero)
        - ``fig``    : matplotlib.figure.Figure
        - ``axes``   : numpy.ndarray of Axes

    Notes
    -----
    - The function treats values equal to ``0`` as structural zeros and
      excludes them from the histogram (the histogram only shows non-zero
      counts).
    - Missing or absent columns raise a ``KeyError`` to fail fast in tests.
    """
    # Validate input structure and extract the pivot mapping
    pivot = esp_results.get("pivot_data")
    if pivot is None:
        raise KeyError("esp_results must contain key 'pivot_data'.")

    # Fail fast if requested column is missing (helps unit tests and callers)
    if column not in pivot:
        raise KeyError(f"Column '{column}' not found in esp_results['pivot_data'].")

    # Build a flat pandas Series (keep original index for year-month labels)
    series_with_index = pd.Series(pivot[column])
    date_index = series_with_index.index
    series = series_with_index.reset_index(drop=True)

    # Summary counts: total, zeros, and non-zero bool_mask
    n_total = len(series)
    n_zeros = int((series == 0).sum())
    nonzero = series[series > 0]

    # Basic numeric summaries for non-zero values; None when no non-zero entries
    stats = {
        "n_total": int(n_total),
        "n_zeros": int(n_zeros),
        "n_nonzero": int(len(nonzero)),
        "mean_nonzero": float(nonzero.mean()) if len(nonzero) > 0 else None,
        "median_nonzero": float(nonzero.median()) if len(nonzero) > 0 else None,
        "max_nonzero": int(nonzero.max()) if len(nonzero) > 0 else None,
    }

    # Create two stacked subplots: top shows the raw time series, bottom shows
    # the histogram of the non-zero values (distribution of observed counts).
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    # Top subplot: raw counts over time with a horizontal zero reference
    axes[0].plot(series.values, color="steelblue", linewidth=0.8)
    axes[0].axhline(0, color="red", linewidth=0.5, linestyle="--")
    axes[0].set_title(f"{column} - Raw Counts Across {n_total} Months")
    axes[0].set_xlabel("Year-Month")
    axes[0].set_ylabel("Count")
    axes[0].set_xlim(0, max(0, n_total - 1))
    
    # Set x-axis ticks with year-month labels (yyyy-mm format)
    tick_positions = np.linspace(0, n_total - 1, min(12, n_total), dtype=int)
    tick_labels = []
    for i in tick_positions:
        date_val = date_index[i]
        if hasattr(date_val, 'strftime'):
            tick_labels.append(date_val.strftime('%Y-%m'))
        else:
            tick_labels.append(str(date_val)[:7])
    axes[0].set_xticks(tick_positions)
    axes[0].set_xticklabels(tick_labels, rotation=45, ha='right')

    # Annotate number of zero months in the plot area (visible summary)
    axes[0].text(
        0.01,
        0.105,
        f"Zero months: {n_zeros} / {n_total}",
        transform=axes[0].transAxes,
        fontsize=10,
        verticalalignment="top",
        color="red",
    )

    # Bottom subplot: histogram of the observed (positive) counts only
    axes[1].hist(nonzero.values, bins=bins, color="steelblue", edgecolor="white")
    axes[1].set_title(f"{column} - Distribution of Non-Zero Counts")
    axes[1].set_xlabel("Count")
    axes[1].set_ylabel("Frequency")

    # Summary annotation on the histogram: show non-zero sample size and mean
    axes[1].text(
        0.98, 0.95,                              # X near right, Y near top
        (f"Non-zero months: {len(nonzero)}\n"
        f"Mean: {stats['mean_nonzero']:,.1f}"   # Added comma for large numbers
        if stats["mean_nonzero"] is not None else "No non-zero months"),
        transform=axes[1].transAxes,
        fontsize=10,
        fontfamily='monospace',                  # Keeps numbers and colons aligned
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7) # Added background for readability
    )
    # axes[1].text(
    #     0.99,
    #     0.105,
    #     f"Non-zero months: {len(nonzero)}\nMean: {stats['mean_nonzero']:.1f}" if stats["mean_nonzero"] is not None else "No non-zero months",
    #     transform=axes[1].transAxes,
    #     fontsize=10,
    #     verticalalignment="top",
    #     horizontalalignment="right",
    # )

    plt.tight_layout()
    if show:
        plt.show()

    # Return the series, computed stats, and figure/axes for tests or further use
    return {
        "series": series,
        "stats": stats,
        "fig": fig,
        "axes": axes,
    }


# ---------------------------------------------------------------------
# 9. Crime Coordinate Scatter Plot with Rasterization
# ---------------------------------------------------------------------
def plot_crime_coordinates(
    data_df: pd.DataFrame, 
    image_path: str = None
):
    """
    Plots ALL points in a large dataset using rasterization to prevent 
    memory crashes and long render times.
    """
    if image_path is not None:
        # initialize 
        image_name = image_path + "scatter_crime_plot.png"
    # 1. Prepare Data
    min_date = data_df['date'].min().strftime('%Y-%m')
    max_date = data_df['date'].max().strftime('%Y-%m')
    # Selecting only necessary columns to save RAM during the plot call
    cols = ['x_coordinate', 'y_coordinate', 'fbi_index_code']
    df_plot = data_df[cols].dropna()
    df_plot = df_plot.loc[df_plot['x_coordinate'] > 0]

    # 2. Initialize Figure
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # 3. Create Scatter Plot with Rasterization
    # rasterized=True is the key for 8.5M points
    sns.scatterplot(
        data=df_plot,
        x='x_coordinate',
        y='y_coordinate',
        hue='fbi_index_code',
        alpha=0.01,         # Low alpha is critical to see density with 8.5M points
        s=1.0,              # Smaller point size for better clarity at high density
        edgecolor=None,     # Removing edges saves massive rendering time
        ax=ax,
        rasterized=True     # Renders points as a bitmap layer within the vector plot
    )

    # 4. Styling
    ax.set_facecolor('white')
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    
    # 5. Legend Configuration
    legend = ax.legend(
        title="Crime Indexed",
        loc="upper right",
        markerscale=10,      # Scale up markers because s=0.5 is invisible in legend
        frameon=True,
        bbox_to_anchor=(1.15, 1)
    )
    # Force legend markers to be visible
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
        if hasattr(handle, "set_sizes"):
            handle.set_sizes([100])

    plt.title(f"Crime Dataset ({min_date} to {max_date}): (N={len(df_plot):,})", fontsize=16)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")

    # 6. Save with High DPI
    # High DPI ensures the rasterized points don't look blurry
    if image_path:
        plt.savefig(image_name, dpi=300, bbox_inches='tight')
        print(f"Full plot saved to: {image_name}")
        print()
        
    plt.show()