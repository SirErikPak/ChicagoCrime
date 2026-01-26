import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


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