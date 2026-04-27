"""
_DATE_KEY: The key for the date column in the dataset, used for grouping and analysis.
_COUNTER_KEY: The key for the count of crimes, used for aggregating crime data.
_GROUP_KEY: The key for the crime type or category, used for grouping crime data.
_EPS_GRID: A list of small values representing the Jeffreys prior grid for smoothing in 
    statistical analysis, which helps to prevent overfitting and provides a more robust 
    estimation of probabilities in the presence of sparse data.
"""
# Global constants
config = {
    "_DATE_KEY"     : "year_month",
    "_COUNTER_KEY"  : "crime_count",
    "_GROUP_KEY"    : "fbi_code_desc",
}