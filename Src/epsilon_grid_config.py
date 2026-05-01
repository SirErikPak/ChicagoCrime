
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
    "_N_PER_DECADE": 6,
    "_INCLUDE_FIXED": (1e-5, 1e-4, 1e-3, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.1, 0.5, 1.0),
    "_MIN_MULTIPLIER_CANDIDATES": (0.1, 0.25, 0.5, 1.0),
    "_Q_LOW": 0.05,
    "_FLOOR": 1e-12,
    "_MIN_STEP": 0.20,
    }