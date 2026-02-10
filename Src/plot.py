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
    Enhanced PCA scatter plot with density overlay and sorted loadings.

    Parameters
    ----------
    data : DataFrame
        Original dataset used for PCA (columns = variables).
    pc_scores : ndarray
        PCA-transformed coordinates (n_samples x n_components).
    pca_model_object : PCA
        Fitted PCA model.
    labels : array-like, optional
        Cluster IDs or group labels for coloring points.
    clusters : array-like, optional
        If provided, saves the figure to ../Image/{file}.png.
    point_size : int
        Marker size.
    font_size : int
        Font size for labels.
    density : bool
        Whether to overlay KDE density.
    """

    # Compute and sort loadings for interpretation
    loadings = pd.DataFrame(
        pca_model_object.components_.T,
        index=data.columns,
        columns=[f"PC{i+1}" for i in range(pca_model_object.n_components_)]
    ).sort_values("PC1", ascending=False)

    # Density settings (only if enabled)
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

    # dispaly explained variance
    print("----- PCA Variance Explained -----")
    print(f"Total PC1 & PC2 Variance: {pca_model_object.explained_variance_ratio_[:2].sum():.2f}")
    print(f"Explained Variance Ratios: \n"
          f"{pca_model_object.explained_variance_ratio_}\n") 

    # Legend for clusters if labels are provided
    if labels is not None:
        ax.legend(
            title="Cluster",
            loc="upper left",
            fontsize=font_size - 2,
            title_fontsize=font_size,
            frameon=False
        )
    # Create a dataframe for the map
    df_pca = pd.DataFrame(data = pc_scores[:, :2], columns = ['PC1', 'PC2'], index=data.index)

    # Save file if requested
    if file:
        plt.savefig(f"../Image/{file}.png", dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.show()

    return loadings, df_pca


def correlation_heatmap(data: pd.DataFrame, figsize: tuple=(10,8)) -> None:
    """
    Plot a heatmap of pairwise correlation distances between rows of a DataFrame.

    This function computes the correlation distance (1 - Pearson r) between all
    pairs of rows in the input DataFrame, converts the condensed distance vector
    into a full square distance matrix, and visualizes it as a heatmap. The
    diagonal is masked because self-distances are always zero and not meaningful
    for interpretation. A custom green→yellow→red colormap is used to highlight
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
        0 → perfectly correlated (same shape)
        1 → uncorrelated
        2 → perfectly anti-correlated (opposite shape)
    - The diagonal is masked because each row has distance 0 to itself.
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

    # diagonal boolean mask - meaningful distances off the diagonal
    mask = np.eye(dist_matrix.shape[0], dtype=bool)
    
    plt.figure(figsize=figsize)
    ax = sns.heatmap(
        dist_matrix,
        mask=mask,
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


def plot_clustering(
    data: pd.DataFrame, 
    method: str, 
    metric: str, 
    figsize: tuple=(8,4), 
    rotation: int=45
) -> None:
    """
    Perform hierarchical clustering and plot a dendrogram.

    Parameters:
    - data: pd.DataFrame, rows as observations, columns as features
    - method: str, linkage method ('single', 'complete', 'average', 'ward', etc.)
    - metric: str, distance metric ('euclidean', 'correlation', etc.)
    - figsize: tuple, figure size
    - rotation: int, rotation angle for leaf labels
    - return_linkage: bool, if True, return the linkage matrix

    Returns:
    - Z: linkage matrix (only if return_linkage=True)
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
        index (str): The column name used for indexing (not explicitly used in the 
            Seaborn call, but often required for pre-aggregated dataframes.
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


def line_plot(data: pd.DataFrame, column_name: str, category_name: str, numeric_name: str, col_wrap=4, rotation: int=45, ha='right') -> None:
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

    Returns:
        None: Displays the plot using plt.show().
    """
    # Create the FacetGrid
    # 'sharex=True' to keep category alignment but 'sharey=False' 
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
    plt.savefig(f"../Image/{category_name}_line.png", dpi=600) 
    plt.show()



def stacked_bar_plot(data, title='', figsize=(12, 8), cmap='Set2', sort_index=True):
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
    # Sort Data by Index
    # ascending=False ensures 'A' is at the top and 'Z' is at the bottom in barh
    if sort_index:
        data = data.sort_index(ascending=False)

    # Calculate row-wise percentages
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
    plt.title('Relative Distribution of Crimes Across ' + title + ' %', fontsize=14, pad=15)
    plt.xlabel('Percentage of Total Count', fontweight='bold')
    plt.ylabel('FBI Crime Type', fontweight='bold')
    plt.legend(title=title, bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.show()


def plot_cuts(heights: np.array, incons: np.array) -> None:
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