import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import scipy.cluster.hierarchy as hierarchy
from scipy.spatial.distance import pdist, squareform
from matplotlib.colors import LinearSegmentedColormap
from scatterd import scatterd
# Python file imports
import hierarchy_clustering
from typing import Tuple


# ---------------------------------------------------------------------
# 0. Fancy PCA Plot Bundle
# ---------------------------------------------------------------------
def fancy_pca_plot_bundle(results, era_config, image_path=None, point_size=90, font_size=14):
    """
    Generate a seamless, presentation‑quality PCA visualization with:
        • KDE density background (vlag colormap)
        • Era‑colored scatter overlays
        • Programmatic background color matching (no white corners)
        • Clean axis labeling with variance ratios
        • Automatic era assignment from timestamps
        • Optional PNG export

    This version focuses on visual polish:
        - Background color is sampled directly from the 'vlag' colormap
          to eliminate corner artifacts.
        - Margins and padding are tuned for balanced framing.
        - Labels and titles use consistent spacing and color contrast.

    Parameters
    ----------
    results : dict
        PCA output bundle containing:
            'coordinates_normalized' : ndarray (n_samples X 2)
                Normalized PC1–PC2 coordinates.
            'observation_index' : array-like
                Timestamps for each observation.
            'variance_ratio' : array-like
                Explained variance for PC1 and PC2.
    era_config : dict
        Mapping of era name → (start_date, end_date, color).
        Example:
            {
                'Pre-COVID': ('2014-01-01', '2020-03-01', 'blue'),
                'COVID':     ('2020-03-01', '2022-06-01', 'red'),
                'Post-COVID':('2022-06-01', '2024-01-01', 'green')
            }
    image_path : str or None, default None
        If provided, saves the figure to:
            f"{image_path}PCA_Seamless_Match.png"
    point_size : int, default 90
        Marker size for scatter points.
    font_size : int, default 14
        Base font size for titles, labels, and legend.

    Returns
    -------
    dict
        {
            'figure': matplotlib.figure.Figure,
            'axis'  : matplotlib.axes.Axes
        }
    """

    # ------------------------------------------------------------
    # 1. EXTRACT PCA OUTPUTS
    # ------------------------------------------------------------
    coords = results['coordinates_normalized']      # PC1–PC2 coordinates
    timestamps = results['observation_index']       # Raw timestamps
    variance_ratio = results['variance_ratio']      # Explained variance for PC1/PC2
    pc1, pc2 = coords[:, 0], coords[:, 1]           # Split for convenience

    # Metadata for title
    n_months = len(timestamps)
    k_categories = coords.shape[1]

    # ------------------------------------------------------------
    # 2. ERA LABEL ASSIGNMENT
    # ------------------------------------------------------------
    # Convert timestamps to YYYY-MM-DD strings for comparison
    ts_str_array = np.array([str(ts)[:10] for ts in timestamps])

    # Default label for timestamps outside defined eras
    era_labels = np.full(len(ts_str_array), "Other", dtype=object)

    # Assign each timestamp to its era based on start/end boundaries
    for name, (start, end, _) in era_config.items():
        mask = (ts_str_array >= start) & (ts_str_array < end)
        era_labels[mask] = name

    # ------------------------------------------------------------
    # 3. PROGRAMMATIC BACKGROUND COLOR MATCH
    # ------------------------------------------------------------
    # Sample the 'vlag' colormap at 0.0 to match KDE background
    match_color = plt.get_cmap("vlag")(0.0)

    # ------------------------------------------------------------
    # 4. BASE DENSITY PLOT (KDE)
    # ------------------------------------------------------------
    args_density = {
        'fill': True,
        'thresh': 0,
        'levels': 100,
        'cmap': "vlag"
    }

    # Render KDE density (points added later)
    fig, ax = scatterd(
        pc1, pc2,
        labels=era_labels,
        s=0,                       # No points yet — only density
        density=True,
        args_density=args_density,
        verbose=0
    )

    # ------------------------------------------------------------
    # 5. FIX CORNERS + ADD MARGINS
    # ------------------------------------------------------------
    fig.set_facecolor(match_color)   # Match figure background
    ax.set_facecolor(match_color)    # Match axes background

    ax.margins(0.15)                 # Add breathing room around clusters

    # ------------------------------------------------------------
    # 6. ERA-COLORED SCATTER OVERLAY
    # ------------------------------------------------------------
    for era_name, (start, end, color) in era_config.items():
        mask = (era_labels == era_name)
        if np.any(mask):
            ax.scatter(
                pc1[mask], pc2[mask],
                c=color,
                s=point_size,
                edgecolor='white',
                linewidth=0.8,
                alpha=0.9,
                zorder=10
            )

    # ------------------------------------------------------------
    # 7. TITLES & AXIS LABELS
    # ------------------------------------------------------------
    ax.set_title(
        f"Structural Realignment of Chicago Crime\n"
        f"CLR-Transformed Latent Space | $N={n_months}, K={k_categories}$",
        fontsize=font_size + 2,
        fontweight='bold',
        pad=35,
        color='white'
    )

    ax.set_xlabel(
        f"PC1 ({variance_ratio[0] * 100:.1f}%) - Regime Shift",
        fontsize=font_size,
        color='white',
        labelpad=15
    )
    ax.set_ylabel(
        f"PC2 ({variance_ratio[1] * 100:.1f}%) - Volatility",
        fontsize=font_size,
        color='white',
        labelpad=15
    )

    # White ticks for dark background
    ax.tick_params(colors='white', labelsize=font_size - 2)

    # ------------------------------------------------------------
    # 8. LEGEND (ERA COLORS + DATE RANGES)
    # ------------------------------------------------------------
    handles = [
        mpatches.Patch(color=c, label=f"{n}\n({s} to {e})")
        for n, (s, e, c) in era_config.items()
    ]

    ax.legend(
        handles=handles,
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        fontsize=font_size - 5,
        frameon=True,
        shadow=True
    )

    # ------------------------------------------------------------
    # 9. SUBTLE GRID + SPINE CLEANUP
    # ------------------------------------------------------------
    ax.grid(True, color='white', alpha=0.1, linestyle='--')

    # Remove white spines by matching background color
    for spine in ax.spines.values():
        spine.set_edgecolor(match_color)

    # ------------------------------------------------------------
    # 10. OPTIONAL SAVE
    # ------------------------------------------------------------
    if image_path:
        plt.savefig(
            f"{image_path}PCA_Seamless_Match.png",
            bbox_inches='tight',
            facecolor=fig.get_facecolor()
        )

    return {"figure": fig, "axis": ax}


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

    # Compute and sort PCA loadings (features X PCs) for interpretation.
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
# 8. Crime Coordinate Scatter Plot with Rasterization
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