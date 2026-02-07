import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis

# Set a random seed for reproducibility
SEED=1776


def perform_sklearn_fa(data: pd.DataFrame, n_factors: int = 3, rotation: str='varimax', seed: int=SEED):
    """
    Performs Factor Analysis using sklearn with enhanced metrics.
    
    Parameters:
    - data: pd.DataFrame (standardized/scaled)
    - n_factors: int, number of factors to extract
    - rotation: str, 'varimax' or 'quartimax'
    
    Returns:
    - results: dict containing the fitted model, loadings, scores, and communalities
    """
    # Validation: Ensure no missing values (sklearn FA can't handle NaNs)
    if data.isnull().any().any():
        raise ValueError("Data contains NaNs. Please impute or drop missing values.")

    # Initialize and Fit
    # Using 'randomized' SVD can be faster for large datasets
    fa = FactorAnalysis(n_components=n_factors, rotation=rotation, random_state=seed)
    factor_scores = fa.fit_transform(data)
    
    # Extract Loadings
    # Loadings represent the correlation between variables and factors
    loadings = pd.DataFrame(
        fa.components_.T,
        index=data.columns,
        columns=[f"Factor{i+1}" for i in range(n_factors)]
    )
    
    # Calculate Communality and Uniqueness
    # Communality = sum of squared loadings across factors for a variable
    # It shows the % of variance in a variable explained by all factors
    loadings['Communality'] = np.sum(fa.components_**2, axis=0)
    loadings['Uniqueness'] = 1 - loadings['Communality']
    
    # 5. Sort for interpretation
    loadings = loadings.sort_values("Factor1", ascending=False)
    
    return {
        "model": fa,
        "loadings": loadings,
        "scores": factor_scores,
        "noise_variance": fa.noise_variance_ # Variance not captured by the factor model
    }