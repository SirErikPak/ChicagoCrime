import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd



# ── 1. Crime History Plot ────────────────────────────────────────────────
def plot_crime_history(crime_name, baseline_df, era_data_filled):
    """
    Plots the full timeline of a specific crime category across all eras
    with baseline and era markers.
    """
    # Combine data from all eras for the specific crime
    combined_data = (
        pd.concat([
            era_data_filled['pre_covid'],
            era_data_filled['covid'],
            era_data_filled['post_covid']
        ])
        .query("fbi_code_desc == @crime_name")  # use @ to reference variable in query
        .sort_values('date')
    )

    if combined_data.empty:
        print(f"No data found for crime: {crime_name}")
        return

    # Extract baseline mean from the summary table
    try:
        b_mean = baseline_df.loc[
            baseline_df['fbi_code_desc'] == crime_name, 'mean'
        ].values[0]
    except IndexError:
        b_mean = 0
        print(f"Warning: No baseline mean found for {crime_name}")

    # Derive the year range dynamically from the data instead of hardcoding
    year_start = combined_data['date'].min().year   
    year_end   = combined_data['date'].max().year   

    # Initialize the plot
    fig, ax = plt.subplots(figsize=(14, 4))

    # Plot the primary time series line
    ax.plot(combined_data['date'], combined_data['crime_count'],
            linewidth=1, color='steelblue', label='Monthly Count')

    # Vertical markers for era transitions
    ax.axvline(pd.Timestamp('2020-03-01'), color='orange', linestyle='--', label='COVID start',      alpha=0.7)
    ax.axvline(pd.Timestamp('2023-01-01'), color='green',  linestyle='--', label='Post-COVID start', alpha=0.7)

    # Horizontal line for the historical pre-COVID average
    ax.axhline(b_mean, color='red', linestyle=':',
               label=f'Baseline Mean ({b_mean:.1f})')

    # Labels and visual formatting
    ax.set_title(f'{crime_name} - Monthly Counts ({year_start}–{year_end})', fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Incidents')
    ax.legend(loc='upper left', frameon=True)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.show()



# ── 2. Seasonal Amplitude Plot ────────────────────────────────────────────────
def plot_seasonal_amplitude(data_df, available_crimes, n_months=60, ncols=3, palette_name="viridis"):
    """
    Plots a grid of time series subplots to visualize seasonal amplitude and 
    variance regimes across different crime categories.
    
    Parameters:
    -----------
    data_df : pandas.DataFrame
        The source dataframe containing 'fbi_code_desc' and 'crime_count' columns.
    available_crimes : list
        List of strings identifying the crimes to be extracted and plotted.
    n_months : int, default 60
        The maximum number of data points (months) to display per plot.
    ncols : int, default 3
        Number of columns in the resulting Matplotlib figure grid.
    palette_name : str, default "viridis"
        The Seaborn color palette used to differentiate the subplots.
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The generated figure object.
    axes : numpy.ndarray
        Array of the specific axes objects used in the plot.
    """
    # 1. Setup Theme and Data Extraction
    sns.set_theme(style="whitegrid", context="talk")

    # Extract time series into a dictionary, filtering for minimum data requirements (>12 months)
    data_dict = {
        crime: data_df.loc[data_df['fbi_code_desc'] == crime, 'crime_count'].values
        for crime in available_crimes
        if len(data_df.loc[data_df['fbi_code_desc'] == crime, 'crime_count'].values) > 12
    }

    n = len(data_dict)
    if n == 0:
        print("No data available to plot (all series < 12 months or list empty).")
        return None, None
        
    # Calculate grid dimensions based on the number of crimes
    nrows = int(np.ceil(n / ncols))
    palette = sns.color_palette(palette_name, n_colors=n)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 5 * nrows))
    
    # Flatten axes for easy iteration; handle edge case where subplots() returns a single Ax object
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    # 2. Iterate and Plot
    for i, (label, series) in enumerate(data_dict.items()):
        ax = axes[i]
        
        # Determine the window of data to show (capped by n_months)
        actual_slice = min(len(series), n_months)
        data_to_plot = series[:actual_slice]
        x = np.arange(len(data_to_plot))
        
        # Primary trend line
        sns.lineplot(x=x, y=data_to_plot, ax=ax, color=palette[i], linewidth=2.5, zorder=3)
        
        # Shaded area to emphasize the "amplitude" or volume of the series
        ax.fill_between(x, data_to_plot, alpha=0.15, color=palette[i])
        
        # Horizontal reference line representing the mean of the shown period
        avg = np.mean(data_to_plot)
        ax.axhline(avg, color='black', linestyle='--', linewidth=1, alpha=0.4, label='Mean')
        
        # Titles and labels (truncated title to prevent overlap)
        ax.set_title(f'Regime: {label[:25]}', fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel("Months", fontsize=12)
        ax.set_ylabel("Monthly Count", fontsize=12)
        
        # Aesthetic cleanup: remove outer box/spines and soften the grid
        sns.despine(ax=ax, left=True)
        ax.grid(True, axis='y', alpha=0.3)

    # 3. Handle empty subplots (if n is not a perfect multiple of ncols)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    # 4. Final Polish and Title
    plt.suptitle(f'Diagnostic: Seasonal Amplitude Test\n(Visualizing variance regimes for n={n} candidates)', 
                 fontsize=20, y=1.02, fontweight='bold')
    
    # Remove horizontal grid lines only
    for ax in axes:
        ax.yaxis.grid(False)
    plt.tight_layout()
    plt.show()


# ── 3. Global Mean-Variance Plot ────────────────────────────────────────
def plot_global_mean_variance(df, n_labels=5):
    """
    Visualizes the mean-variance relationship for all crime categories to detect 
    overdispersion and heteroscedasticity.

    Parameters:
    -----------
    df : pandas.DataFrame
        Tidy dataframe containing 'fbi_code_desc' and 'crime_count' columns.
    n_labels : int, default 5
        Number of highly overdispersed crimes to highlight with labels.

    Returns:
    --------
    None (Displays a Matplotlib figure)
    """
    # 1. Statistical Aggregation
    # Group by crime type to find the total mean and variance across the time series
    global_stats = (
        df.groupby('fbi_code_desc')['crime_count']
        .agg(['mean', 'var'])
        .reset_index()
    )

    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(12, 8))

    # 2. Dispersion Analysis
    # Dispersion ratio (V/M) > 1 indicates overdispersion (common in crime data)
    global_stats['dispersion'] = global_stats['var'] / global_stats['mean']
    
    # Use log10 dispersion for the color mapping to handle large ranges gracefully
    scatter = ax.scatter(
        global_stats['mean'], global_stats['var'],
        c=np.log10(global_stats['dispersion']),
        cmap='RdYlGn_r', s=100, alpha=0.8, edgecolors='grey', linewidths=0.5,
        zorder=3
    )
    
    # Add colorbar with LaTeX-rendered label to avoid font warnings
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label(r'$\log_{10}(\mathrm{Dispersion\ Ratio:}\ Var/Mean)$', fontsize=11)

    # 3. Baseline Guidelines
    # Generating log-spaced range for smooth lines on a log-scale plot
    x_range = np.logspace(
        np.log10(global_stats['mean'].min() * 0.8),
        np.log10(global_stats['mean'].max() * 1.2),
        200
    )
    
    # Poisson Baseline: Variance = Mean
    ax.plot(x_range, x_range, color='steelblue', ls='--', lw=1.5, alpha=0.7, label='Poisson (V = M)')
    # Negative Binomial (NB) Guidelines: visualizing 10x and 100x overdispersion
    ax.plot(x_range, 10 * x_range, color='tomato', ls=':', lw=1.5, alpha=0.7, label='NB guideline (V = 10M)')
    ax.plot(x_range, 100 * x_range, color='firebrick', ls='-.', lw=1.5, alpha=0.7, label='NB guideline (V = 100M)')

    # 4. Axis Scaling & Titles
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title("Global Mean–Variance Landscape (All Crimes)", fontsize=16, fontweight='bold', pad=50)
    ax.set_xlabel("Mean Monthly Count (log scale)", fontsize=12)
    ax.set_ylabel("Variance (log scale)", fontsize=12)
    ax.legend(fontsize=10)

    # 5. Dynamic Staggered Labeling
    # Identify the top N crimes where variance is highest relative to the mean
    top_var = global_stats.nlargest(n_labels, 'dispersion')

    # Sort candidates by variance to prevent arrow-crossing in the vertical stack
    top_var = top_var.sort_values('var', ascending=True).reset_index(drop=True)

    # offset text 60 pixels to the left of the data points
    x_offset = -60  
    # Create evenly spaced vertical offsets so labels don't overlap each other
    y_offsets = np.linspace(-40, 40, n_labels)

    for i, row in top_var.iterrows():
        oy = y_offsets[i]
        ax.annotate(
            row['fbi_code_desc'],
            xy=(row['mean'], row['var']),      # Target coordinates (data space)
            xytext=(x_offset, oy),             # Label coordinates (pixel offset space)
            textcoords='offset points',
            fontsize=9,
            ha='right',                        # Anchor text at the end of the arrow
            arrowprops=dict(
                arrowstyle='->',
                color='gray',
                lw=0.8,
                shrinkA=4,                     # Distance from label
                shrinkB=4                      # Distance from data point
            )
        )

    # 6. Final Polish
    sns.despine()
    plt.tight_layout()
    plt.show()


# ── 4. Mean-Variance Analysis ────────────────────────────────────────
def plot_mean_variance_analysis(data, crime_type, window=12, log_scale=False):
    """
    Performs a rolling mean-variance diagnostic on a specific crime category 
    to identify heteroscedasticity and overdispersion.

    Parameters:
    -----------
    data : pandas.DataFrame
        Long-form dataframe containing 'date', 'fbi_code_desc', and 'crime_count'.
    crime_type : str
        The specific fbi_code_desc to filter and analyze.
    window : int, default 12
        The size of the rolling window (in months) for calculating statistics.
    log_scale : bool, default False
        If True, applies a log-log transformation to the axes. Helpful for
        high-volume crimes or exponential mean-variance relationships.
    """
    # 1. Filter and Calculate Rolling Statistics
    # We set the index to date to ensure temporal ordering, though the
    # rolling operation primarily relies on the row sequence.
    series = data[data['fbi_code_desc'] == crime_type].set_index('date')['crime_count']
    
    stats = pd.DataFrame({
        'Rolling Mean': series.rolling(window).mean(),
        'Rolling Var': series.rolling(window).var()
    }).dropna() # Remove the initial 'window-1' rows where stats are NaN

    # 2. Setup Plot Theme
    sns.set_theme(style="whitegrid", context="notebook")
    fig, ax = plt.subplots(figsize=(10, 6))

    # 3. Create Scatter with Regression Trend
    # regplot visualizes the relationship; if the line is steep, the series
    # likely requires a multiplicative decomposition or log-transform.
    sns.regplot(
        data=stats, x='Rolling Mean', y='Rolling Var',
        scatter_kws={'alpha': 0.5, 's': 60, 'color': '#2a9d8f'},
        line_kws={'color': '#e76f51', 'label': 'Trend (Mean-Var)'},
        ax=ax
    )

    # 4. Identity Line (Poisson Baseline: Var = Mean)
    # This acts as a 'Goldilocks' line. Dots significantly above this line 
    # indicate 'Overdispersion', common in aggregated crime data.
    if not stats.empty:
        combined_max = max(stats['Rolling Mean'].max(), stats['Rolling Var'].max())
        ax.plot([0, combined_max], [0, combined_max], 
                color='black', linestyle=':', alpha=0.6, label='Poisson Baseline (V=M)')

    # 5. Scaling and Titles
    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f"Log-Log Mean vs Variance: {crime_type}", fontweight='bold', pad=15)
    else:
        ax.set_title(f"Mean vs Variance: {crime_type}", fontweight='bold', pad=15)

    # 6. Final Polish
    ax.set_xlabel("Rolling Mean (Window=12)", fontsize=11)
    ax.set_ylabel("Rolling Variance (Window=12)", fontsize=11)
    ax.legend(frameon=True)
    sns.despine()
    plt.tight_layout()
    plt.show()