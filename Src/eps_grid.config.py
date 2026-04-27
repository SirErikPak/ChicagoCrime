
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
# EPS Grid Configuration for build_eps_grid.py
config = {
    "n_per_decade": 6,
    "include_fixed": [1e-5, 1e-4, 1e-3, 0.01, 0.03, 0.05, 0.07, 0.1, 0.5, 1.0],
    "min_multiplier_candidates": [0.1, 0.25, 0.5, 1.0],
    "q_low": 0.05,
    "floor": 1e-12,
    "min_step": 0.20,
    }