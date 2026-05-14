
# Era Configuration settings for defining eras and parameters for 
# crime data analysis and visualization.
# Map era labels to keys in the input dictionaries
era_map = {
    'pre_covid' : 'Pre-COVID',
    'covid'     : 'COVID',
    'post_covid': 'Post-COVID'
}

# Era boundaries for defining the three eras based on time periods, which are 
# used to categorize crime data into distinct phases for analysis and visualization.
era_boundaries = {
    # end time
    'Pre-COVID'  : '2020-02',
    'COVID'      : '2022-12',
    'Post-COVID' : '2023-01' 
}

# Era configuration for plotting, including start and end dates for each era 
# and associated colors for visualization.
era_config = {
    'Pre-COVID' : ('2001-01-01', '2020-03-01', 'blue'),
    'COVID'     : ('2020-03-01', '2023-01-01', 'red'),
    'Post-COVID': ('2023-01-01', '2025-12-01', 'green'),
}

"""
_DATE_KEY: The key for the date column in the dataset, used for grouping and analysis.
_COUNTER_KEY: The key for the count of crimes, used for aggregating crime data.
_GROUP_KEY: The key for the crime type or category, used for grouping crime data.
_EPS_GRID: A list of small values representing the Jeffreys prior grid for smoothing in 
    statistical analysis, which helps to prevent overfitting and provides a more robust 
    estimation of probabilities in the presence of sparse data.
"""
# Configuration for the crime data aggregation and analysis, including keys for date, 
# crime count, and crime type, as well as a predefined grid of epsilon values for smoothing 
# in the CLR method. (clr_utilities.py) 
config_agg = {
    "_DATE_KEY"     : "year_month",
    "_COUNTER_KEY"  : "crime_count",
    "_GROUP_KEY"    : "primary_description",
}

"""
n_per_decade: int
    Number of grid points per decade (logarithmic spacing).
include_fixed: tuple of float
    A set of fixed epsilon values to always include in the grid.
min_multiplier_candidates: tuple of float
    Candidate multipliers for determining the lower bound of the grid based on data.
q_low: float
    Quantile used to determine the lower bound of the grid from the data distribution.
floor: float
    Minimum value for any epsilon in the grid to prevent numerical issues.
min_step: float
    Minimum relative step size between consecutive grid points to ensure a well-spaced grid.
"""
# Configuration for the epsilon grid used in the CLR method, which includes parameters for generating a 
# logarithmically spaced grid of epsilon values, as well as fixed values and constraints to ensure numerical 
# stability and appropriate coverage of the parameter space for the analysis. (clr_eps_grid.py) 
config_grid = {
    "_N_PER_DECADE": 6,
    "_INCLUDE_FIXED": (1e-5, 1e-4, 1e-3, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.5, 1.0),
    "_MIN_MULTIPLIER_CANDIDATES": (0.1, 0.25, 0.5, 1.0),
    "_Q_LOW": 0.05,
    "_FLOOR": 1e-12,
    "_MIN_STEP": 0.20,
    }