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
def plot_seasonal_amplitude(
    data_df: pd.DataFrame,
    available_crimes: list[str],
    n_months: int = 60,
    ncols: int = 3,
    palette_name: str = 'viridis',
) -> None:
    """
    Grid of time-series subplots showing seasonal amplitude and variance
    regimes across crime categories.

    Parameters
    ----------
    data_df         : Long-form DataFrame with columns fbi_code_desc, crime_count.
    available_crimes: Crime categories to plot.
    n_months        : Maximum months displayed per panel (default 60).
    ncols           : Grid columns (default 3).
    palette_name    : Seaborn palette for panel colors (default 'viridis').
    """
    # 1. Extract series
    data_dict = {
        crime: data_df.loc[data_df['fbi_code_desc'] == crime, 'crime_count'].values
        for crime in available_crimes
        if (data_df['fbi_code_desc'] == crime).sum() > 12
    }

    if not data_dict:
        raise ValueError("No series with more than 12 months found in available_crimes.")

    # 2. Grid geometry
    n       = len(data_dict)
    nrows   = int(np.ceil(n / ncols))
    palette = sns.color_palette(palette_name, n_colors=n)

    sns.set_theme(style='whitegrid', context='talk')
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 5 * nrows), squeeze=False)
    axes = axes.flatten()

    # 3. Plot panels
    for i, (label, series) in enumerate(data_dict.items()):
        ax           = axes[i]
        data_to_plot = series[:n_months]
        x            = np.arange(len(data_to_plot))

        sns.lineplot(x=x, y=data_to_plot, ax=ax,
                     color=palette[i], linewidth=2.5, zorder=3)
        ax.fill_between(x, data_to_plot, alpha=0.15, color=palette[i])
        ax.axhline(data_to_plot.mean(), color='black', ls='--',
                   lw=1, alpha=0.4, label='Mean')

        ax.set(title=label[:30], xlabel='Months', ylabel='Monthly Count')
        ax.title.set_fontweight('bold')
        ax.legend(frameon=False, fontsize=10)
        ax.yaxis.grid(True, alpha=0.3)
        ax.xaxis.grid(False)
        sns.despine(ax=ax, left=True)

    # 4. Hide unused panels
    for ax in axes[n:]:
        ax.set_visible(False)

    # 5. Figure title
    fig.suptitle(
        f'Seasonal Amplitude Diagnostic  -  {n} crime categories',
        fontsize=20, fontweight='bold', y=1.02,
    )

    plt.tight_layout()


# ── 3. Global Mean-Variance Plot ────────────────────────────────────────
def plot_global_mean_variance(df, figsize: tuple = (12, 6), n_labels: int = 5) -> None:
    """
    Visualizes the mean-variance relationship across all crime categories to diagnose 
    dispersion regimes and statistical distribution suitability.

    This diagnostic plot helps determine if crime series follow a Poisson distribution 
    (where Variance ≈ Mean) or exhibit overdispersion (Variance > Mean). High 
    overdispersion suggests the need for Negative Binomial models or log-transformations 
    prior to time-series decomposition.

    Args:
        df (pd.DataFrame): Long-form dataframe containing at least 'fbi_code_desc' 
            and 'crime_count' columns.
        figsize (tuple, optional): Dimensions of the resulting figure. Defaults to (12, 6).
        n_labels (int, optional): Number of extreme outliers (most and least dispersed) 
            to label on the plot. Defaults to 5.

    Returns:
        None: Displays a Matplotlib figure.
    """
    # 1. Aggregate
    global_stats = (
        df.groupby('fbi_code_desc')['crime_count']
        .agg(['mean', 'var'])
        .reset_index()
        .query('mean > 0')
        .assign(dispersion=lambda d: d['var'] / d['mean'])
    )

    # 2. Figure
    sns.set_theme(style='whitegrid')
    fig, ax = plt.subplots(figsize=figsize)

    sc = ax.scatter(
        global_stats['mean'], global_stats['var'],
        c=np.log10(global_stats['dispersion']),
        cmap='RdYlGn_r', s=100, alpha=0.85,
        edgecolors='grey', linewidths=0.5, zorder=3
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r'$\log_{10}(\mathrm{Var\,/\,Mean})$', fontsize=11)

    # 3. Reference lines
    x_range = np.logspace(
        np.log10(global_stats['mean'].min() * 0.5),
        np.log10(global_stats['mean'].max() * 2.0),
        300
    )
    for y, ls, color, label in [
        (x_range,        '--', 'steelblue', 'Poisson (V = M)'),
        (10  * x_range,  ':',  'tomato',    'NB (V = 10M)'),
        (100 * x_range,  '-.', 'firebrick', 'NB (V = 100M)'),
    ]:
        ax.plot(x_range, y, ls=ls, color=color, lw=1.5, alpha=0.7, label=label)

    # 4. Axes and Title
    ax.set(xscale='log', yscale='log',
           xlabel='Mean Monthly Count (log scale)',
           ylabel='Variance (log scale)')
    ax.set_title('Global Mean–Variance Landscape', fontsize=16,
                 fontweight='bold', pad=30)

    # 5. Labels (collision-safe)
    top = global_stats.nlargest(n_labels, 'dispersion')
    low = global_stats.nsmallest(n_labels, 'dispersion')
    
    # ── lobal Mean-Variance Plot Labels ────────────────────────────────
    def _place_labels(subset, side):
        """
        Convert each point to display (pixel) space, spread labels evenly
        along the y-pixel axis, then convert anchor positions back to data
        space for annotate(). This makes spacing uniform regardless of scale.
        """
        # Transform data coords → display (pixel) coords
        pts = ax.transData.transform(
            [(r['mean'], r['var']) for _, r in subset.iterrows()]
        )   # shape (n, 2) in pixels

        # Sort by pixel-y so linspace offsets run bottom-to-top consistently
        order   = np.argsort(pts[:, 1])
        pts     = pts[order]
        rows    = [subset.iloc[i] for i in order]

        # Spread label anchors over a fixed pixel band centered on the cluster
        y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
        padding  = 40                                    # px above/below outermost point
        y_spread = np.linspace(y_min - padding, y_max + padding, len(rows))
        x_offset = -90 if side == 'left' else 90        # px left/right of point

        for (px, py), y_label, row in zip(pts, y_spread, rows):
            # Label anchor in display space → back to data space
            lx, ly = ax.transData.inverted().transform(
                (px + x_offset, y_label)
            )
            ax.annotate(
                row['fbi_code_desc'],
                xy=(row['mean'], row['var']),
                xytext=(lx, ly),
                xycoords='data', textcoords='data',
                fontsize=9, ha='right' if side == 'left' else 'left',
                va='center',
                arrowprops=dict(arrowstyle='->', color='dimgray',
                                lw=0.8, shrinkB=4),
            )

    _place_labels(top, 'left')
    _place_labels(low, 'right')

    ax.legend(loc='upper left', framealpha=0.9)
    sns.despine()
    plt.tight_layout()


# ── 4. Mean-Variance Analysis Plot ────────────────────────────────────────
def plot_mean_variance_analysis(
    data: pd.DataFrame,
    crime_type: str,
    window: int = 12,
    figsize: tuple = (22, 6)
) -> None:
    """
    Rolling mean-variance diagnostic for a single crime category.

    Renders two side-by-side panels — linear and log-log — so scale-dependent
    patterns (e.g. exponential mean-variance growth) are immediately visible
    without re-running the function.

    Points well above the Poisson baseline (V = M) indicate overdispersion,
    suggesting a negative-binomial or zero-inflated model may be appropriate.

    Parameters
    ----------
    data       : Long-form DataFrame with columns: date, fbi_code_desc, crime_count.
    crime_type : Value of fbi_code_desc to analyse.
    window     : Rolling window size in months (default 12).
    """
    # 1. Rolling stats
    series = (
        data.loc[data['fbi_code_desc'] == crime_type]
        .set_index('date')['crime_count']
    )
    stats = (
        pd.DataFrame({
            'Rolling Mean': series.rolling(window).mean(),
            'Rolling Var':  series.rolling(window).var(),
        })
        .dropna()
    )

    if stats.empty:
        raise ValueError(f"No data after rolling window for crime_type={crime_type!r}.")

    # 2. Figure — two panels sharing the same data but different scales 
    sns.set_theme(style='whitegrid', context='notebook')
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    fig.suptitle(f'Mean vs Variance: {crime_type}', fontsize=14, fontweight='bold', y=1.02)

    combined_max = max(stats['Rolling Mean'].max(), stats['Rolling Var'].max())

    panels = [
        (axes[0], 'linear', 'linear', 'Linear'),
        (axes[1], 'log',    'log',    'Log-Log'),
    ]

    for ax, xscale, yscale, scale_label in panels:
        # Scatter + OLS trend
        sns.regplot(
            data=stats, x='Rolling Mean', y='Rolling Var', ax=ax,
            scatter_kws={'alpha': 0.5, 's': 60, 'color': '#2a9d8f'},
            line_kws={'color': '#e76f51', 'label': f'OLS trend (window={window})'},
        )
        # Poisson baseline
        ax.plot(
            [0, combined_max], [0, combined_max],
            color='black', ls=':', lw=1.2, alpha=0.6, label='Poisson baseline (V = M)',
        )
        ax.set(
            xscale=xscale, yscale=yscale,
            title=f'{scale_label} Scale',
            xlabel=f'Rolling Mean (window={window})',
            ylabel=f'Rolling Variance (window={window})',
        )
        ax.legend(frameon=True)

    sns.despine()
    plt.tight_layout()