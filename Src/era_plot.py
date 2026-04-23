import matplotlib.pyplot as plt
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