import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scatterd import scatterd
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from typing import Mapping, Dict, Any, Tuple
import seaborn as sns
from sklearn.decomposition import PCA
from numpy.linalg import norm
from scipy.linalg import subspace_angles


# ---------------------------------------------------------------------
# 0. Fancy PCA Plot Bundle
# ---------------------------------------------------------------------
def fancy_pca_plot_bundle(results: Mapping[float, pd.DataFrame], 
                          era_config: Dict[str, tuple], eps=None,
                          image_save: str = None, image_path: str = None,
                          point_size: int=90, font_size: int=14):
    """
    Generate a seamless, presentation‑quality PCA visualization with:
        • KDE density background (vlag colormap)
        • Era‑colored scatter overlays
        • Programmatic background color matching (no white corners)
        • Clean axis labeling with variance ratios
        • Automatic era assignment from timestamps
        • Optional PNG export

    This version focuses on visual polish:
        - Background color is sampled directly from the 'vlag' colormap
          to eliminate corner artifacts.
        - Margins and padding are tuned for balanced framing.
        - Labels and titles use consistent spacing and color contrast.

    Parameters
    ----------
    results : dict
        PCA output bundle containing:
            'coordinates_normalized' : ndarray (n_samples × 2)
                Normalized PC1–PC2 coordinates.
            'observation_index' : array-like
                Timestamps for each observation.
            'variance_ratio' : array-like
                Explained variance for PC1 and PC2.
    era_config : dict
        Mapping of era name → (start_date, end_date, color).
        Example:
            {
                'Pre-COVID': ('2014-01-01', '2020-03-01', 'blue'),
                'COVID':     ('2020-03-01', '2022-06-01', 'red'),
                'Post-COVID':('2022-06-01', '2024-01-01', 'green')
            }
    image_path : str or None, default None
        file path to save the figure to:
    image_save : str or None, default None
        image file name to save the figure as (e.g., "PCA_Seamless_Match.png").
          If None, the figure is not saved.
    point_size : int, default 90
        Marker size for scatter points.
    font_size : int, default 14
        Base font size for titles, labels, and legend.

    Returns
    -------
    dict
        {
            'figure': matplotlib.figure.Figure
        }
    """

    # ------------------------------------------------------------
    # 0-A. EXTRACT PCA OUTPUTS
    # ------------------------------------------------------------
    coords         = results[eps]['coords_norm']        # Normalized PC coordinates
    timestamps     = results[eps]['observation_index']  # Raw timestamps
    variance_ratio = results[eps]['variance_ratio']     # Explained variance
    pc1, pc2       = coords[:, 0], coords[:, 1]         # Split for convenience

    # Metadata for title
    n_months = len(timestamps)
    k_categories = coords.shape[1]

    # ------------------------------------------------------------
    # 0-B. ERA LABEL ASSIGNMENT
    # ------------------------------------------------------------
    # Convert timestamps to YYYY-MM-DD strings for comparison
    ts_str_array = np.array([str(ts)[:10] for ts in timestamps])

    # Default label for timestamps outside defined eras
    era_labels = np.full(len(ts_str_array), "Other", dtype=object)

    # Assign each timestamp to its era based on start/end boundaries
    for name, (start, end, _) in era_config.items():
        mask = (ts_str_array >= start) & (ts_str_array < end)
        era_labels[mask] = name

    # ------------------------------------------------------------
    # 0-C. PROGRAMMATIC BACKGROUND COLOR MATCH
    # ------------------------------------------------------------
    # Sample the 'vlag' colormap at 0.0 to match KDE background
    match_color = plt.get_cmap("vlag")(0.0)

    # ------------------------------------------------------------
    # 0-D. BASE DENSITY PLOT (KDE)
    # ------------------------------------------------------------
    args_density = {
        'fill': True,
        'thresh': 0,
        'levels': 100,
        'cmap': "vlag"
    }

    # Render KDE density (points added later)
    fig, ax = scatterd(
        pc1, pc2,
        labels=era_labels,
        s=0,                       # No points yet — only density
        density=True,
        args_density=args_density,
        verbose=0
    )

    # ------------------------------------------------------------
    # 0-E. FIX CORNERS + ADD MARGINS
    # ------------------------------------------------------------
    fig.set_facecolor(match_color)   # Match figure background
    ax.set_facecolor(match_color)    # Match axes background
    ax.margins(0.15)                 # Add breathing room around clusters

    # ------------------------------------------------------------
    # 0-F. ERA-COLORED SCATTER OVERLAY
    # ------------------------------------------------------------
    for era_name, (start, end, color) in era_config.items():
        mask = (era_labels == era_name)
        if np.any(mask):
            ax.scatter(
                pc1[mask], pc2[mask],
                c=color,
                s=point_size,
                edgecolor='white',
                linewidth=0.8,
                alpha=0.9,
                zorder=10
            )

    # ------------------------------------------------------------
    # 0-G. TITLES & AXIS LABELS
    # ------------------------------------------------------------
    ax.set_title(
        rf"Structural Realignment of Chicago Crime ($\epsilon$={eps:.2f})"
        f"\nCLR-Transformed Latent Space | $N={n_months}, K={k_categories}$",
        fontsize=font_size + 2,
        fontweight='bold',
        pad=35,
        color='white'
    )

    ax.set_xlabel(
        f"PC1 ({variance_ratio[0] * 100:.1f}%) - Regime Shift",
        fontsize=font_size,
        color='white',
        labelpad=15
    )
    ax.set_ylabel(
        f"PC2 ({variance_ratio[1] * 100:.1f}%) - Volatility",
        fontsize=font_size,
        color='white',
        labelpad=15
    )

    # White ticks for dark background
    ax.tick_params(colors='white', labelsize=font_size - 2)

    # ------------------------------------------------------------
    # 0-H. LEGEND (ERA COLORS + DATE RANGES)
    # ------------------------------------------------------------
    handles = [
        mpatches.Patch(color=c, label=f"{n}\n({s} to {e})")
        for n, (s, e, c) in era_config.items()
    ]

    ax.legend(
        handles=handles,
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        fontsize=font_size - 5,
        frameon=True,
        shadow=True
    )

    # ------------------------------------------------------------
    # 0-I. SUBTLE GRID + SPINE CLEANUP
    # ------------------------------------------------------------
    ax.grid(True, color='white', alpha=0.1, linestyle='--')

    # Remove white spines by matching background color
    for spine in ax.spines.values():
        spine.set_edgecolor(match_color)

    # ------------------------------------------------------------
    # 0-J. OPTIONAL SAVE
    # ------------------------------------------------------------
    if image_save and image_path:
        plt.savefig(
            f"{image_path}{image_save}",
            bbox_inches='tight',
            facecolor=fig.get_facecolor()
        )

    return {"figure": fig}


# ---------------------------------------------------------------------------
# 1. Three-PC loadings plot
# ---------------------------------------------------------------------------
def plot_three_pc_loadings(
    clr_data: pd.DataFrame,
    chosen_eps: float,
    exclude_features: list = None,
    figsize: tuple[float, float] = (20, 10),
    label_fmt: str = "%.3f",
    verbose: bool = True,
    image_save: str = None,
    image_path: str = None
) -> dict:
    """
    Plot the first three principal component loadings with stable orientation
    and modern visualization aesthetics.

    This function:
    - Optionally excludes user‑specified features before PCA
    - Fits PCA to CLR‑transformed data (all components, but only PC1–PC3 are plotted)
    - Normalizes the sign of each component using a robust rule:
        * If the feature with the largest absolute loading is negative,
          the entire component (loadings + scores) is flipped.
      This ensures consistent orientation across runs, epsilons, and datasets.
    - Orders features by PC1 loading to create a stable shared y‑axis
    - Produces three aligned horizontal bar charts (PC1, PC2, PC3)
    - Returns raw loadings, normalized loadings, PC coordinates, variance ratios,
      singular values, and excluded features.

    Parameters
    ----------
    clr_data : pd.DataFrame
        CLR‑transformed feature table (samples × features).
    chosen_eps : float
        Epsilon value used for annotation and manifest printing.
    exclude_features : list, optional
        List of feature names to drop before PCA.
    figsize : tuple
        Figure size for the 3‑panel layout.
    label_fmt : str
        Format string for bar‑label numeric annotations.
    verbose : bool
        Whether to print a PCA manifest summary.
    image_save : str, optional
        If provided, saves the figure to this filename (path must be handled externally).

    Returns
    -------
    dict
        {
            "figure": matplotlib Figure,
            "loadings_raw": raw PCA loadings (before sign normalization),
            "loadings_norm": sign‑aligned loadings,
            "coords_raw": raw PC scores,
            "coords_norm": normalized PC scores,
            "variance_ratio": explained variance ratios,
            "singular_values": singular values3,
            "excluded": list of excluded features,
            "observation_index": index of observations (samples)
        }
    """
    # ------------------------------------------------------------
    # 1-A. Feature Exclusion & Manifest
    # ------------------------------------------------------------
    working_data = clr_data.copy()

    # Drop only features that actually exist in the DataFrame
    if exclude_features:
        exclude_features = sorted(set(exclude_features))
        excluded_found = [f for f in exclude_features if f in working_data.columns]
        working_data = working_data.drop(columns=excluded_found)
    else:
        excluded_found = []

    # Optional manifest printout for transparency
    if verbose:
        width, val_w , lbl_w = 50, 10, 20
        print("\n" + "═" * width)
        print(f"{'📊 PCA DATA MANIFEST (ε = ' + f'{chosen_eps:.2g}' + ')':^{width}}")
        print("─" * width)
        print(f"{'✅ Observations':<{lbl_w}} {':':>{val_w}} {working_data.shape[0]:>{val_w}}")
        print(f"{'✅ Included Features':<{lbl_w}} {':':>{val_w}} {working_data.shape[1]:>{val_w}}")
        if excluded_found:
            print(f"{'🚫 Excluded':<{lbl_w}} {':':>{val_w}} {len(excluded_found):>{val_w}}")
            for item in excluded_found:
                print(f"   * {item}")
        else:
            print(f"{'🚫 Excluded':<{lbl_w}} {':':>{val_w}} {'None':>{val_w}}")
        print("═" * width + "\n")

    # ------------------------------------------------------------
    # 1-B. PCA Fit & Robust Sign Normalization
    # ------------------------------------------------------------
    pca = PCA()  # Fit all components; we will use only the first three
    coords_raw = pca.fit_transform(working_data.values)   # PC scores
    loadings_raw = pca.components_                        # PC loadings
    ratios = pca.explained_variance_ratio_                # Variance explained
    observation_index = working_data.index.values         # Sample identifiers

    # Note: The PCA is fitted to the CLR‑transformed data, and the raw loadings and scores are extracted.
    loadings_norm = []
    coords_norm   = []

    # Normalize signs so that each PC is oriented consistently:
    # If the feature with the largest absolute loading is negative,
    # flip the entire component (loadings + scores).
    for i in range(len(loadings_raw)):
        comp = loadings_raw[i].copy()
        score = coords_raw[:, i].copy()

        # Identify the anchor feature (largest absolute loading)
        if comp[np.argmax(np.abs(comp))] < 0:
            comp  *= -1
            score *= -1
        # This sign normalization ensures that the direction of each principal 
        # component is consistent across different runs, epsilon values, and datasets.
        #  By anchoring the orientation to the feature with the largest absolute loading, 
        # we create a stable reference point for interpreting the components, which is 
        # crucial for comparative analyses and visualizations.
        loadings_norm.append(comp)
        coords_norm.append(score)

    loadings_norm = np.array(loadings_norm)

    # Order features by PC1 loading for a stable shared y-axis
    order = pd.Series(loadings_norm[0], index=working_data.columns).sort_values().index

    # ------------------------------------------------------------
    # 1-C. Modern Plotting (Three Horizontal Bar Charts)
    # ------------------------------------------------------------
    sns.set_style("white")
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)

    pc_colors = ["#D85A30", "#185FA5", "#51A351"]  # Distinct PC colors
    neg_color = "#95a5a6"                              # Soft gray for negative loadings

    # Iterate over the first three components and create horizontal bar charts with consistent feature ordering,
    for i, ax in enumerate(axes):
        # Reindex to enforce consistent feature ordering
        ser = pd.Series(loadings_norm[i], index=working_data.columns).reindex(order)

        # Color positive vs negative loadings
        colors = [pc_colors[i] if val >= 0 else neg_color for val in ser.values]

        # Draw horizontal bars
        bars = ax.barh(
            ser.index, ser.values,
            color=colors, edgecolor="white", lw=0.5
        )

        # Annotate bars with numeric values
        ax.bar_label(
            bars, fmt=label_fmt, padding=5,
            fontsize=10, fontweight="bold", color="#2c3e50"
        )

        # Reference vertical line at zero
        ax.axvline(0, color="#2d3436", lw=1.2, zorder=3)

        # Panel title with variance explained
        ax.set_title(
            f"PC{i+1}\n{ratios[i]:.1%} Variance",
            fontweight="bold", size=14, pad=15
        )

        # Light grid for readability
        ax.xaxis.grid(True, ls=":", alpha=0.6)
        sns.despine(ax=ax, left=True, bottom=False)

        # Y-axis formatting
        if i == 0:
            ax.tick_params(axis="y", labelsize=11, left=False, pad=60)
            plt.setp(ax.get_yticklabels(), fontweight="bold", color="#2c3e50", ha="right")
        else:
            ax.tick_params(axis="y", length=0)

    # ------------------------------------------------------------
    # 1-D. Layout & Spacing Refinement
    # ------------------------------------------------------------
    axes[0].set_ylabel("Crime Features", fontweight="bold", size=14, labelpad=160)

    plt.subplots_adjust(left=0.45, wspace=0.15, top=0.85, bottom=0.1)

    plt.suptitle(
        f"Principal Component Loadings (PC1–PC3) | ε = {chosen_eps:.2g}\n"
        f"Cumulative Variance: {ratios.sum():.1%}",
        fontweight="bold", size=18, y=0.98
    )

    # Optional save
    if image_save and image_path:
        fig.savefig(image_path + image_save, dpi=300, bbox_inches='tight')

    # ------------------------------------------------------------
    # Return all useful PCA artifacts
    # ------------------------------------------------------------
    return {
        "figure": fig,
        "loadings_raw": loadings_raw,
        "loadings_norm": loadings_norm,
        "coords_raw": coords_raw,
        "coords_norm": np.array(coords_norm).T, 
        "variance_ratio": ratios,
        "singular_values": pca.singular_values_,
        "observation_index": observation_index,
        "excluded": excluded_found
    }

# ---------------------------------------------------------------------------
# 2. PCA stability comparison across epsilon values
# ---------------------------------------------------------------------------
def compare_pca_stability(
    results,
    eps1,
    eps2,
    n_components=3,
    drift_threshold=0.95,
    show_plot=True,
    image_path: str = None,
    image_save: str = None
):
    """
    Compare PCA stability across two epsilon values.

    Computes variance-structure drift, loading-vector alignment,
    subspace rotation, and sample-embedding stability between the PCA
    decompositions at eps1 and eps2. Produces:

      - Singular-value drift (global variance structure)
      - Cosine similarity of loading vectors (directional stability)
      - Principal angles between PC subspaces (rotation-invariant drift)
      - Correlation of sample coordinates (embedding stability)
      - Composite 0–1 stability index
      - Optional 3-panel diagnostic dashboard

    Parameters
    ----------
    results : dict
        Mapping eps → PCA result bundles containing singular values,
        loadings, and sample coordinates.

    eps1, eps2 : float
        Epsilon values to compare.

    n_components : int, default=3
        Number of principal components to evaluate.

    drift_threshold : float, default=0.95
        Minimum acceptable cosine similarity for PC robustness.

    show_plot : bool, default=True
        Whether to render the diagnostic dashboard.

    image_path : str, optional
        Path to save the figure. If provided, saves as PNG at 300 DPI.

    image_save : str, optional
        Image file name to save the figure. If provided, saves as PNG at 300 DPI.

    Returns
    -------
    dict
        {
            "composite_score": float,
            "angles": np.ndarray,
            "cosines": np.ndarray
        }
    """
    # -------------------------------------------------
    # 2-A: Extract PCA artifacts for both eps values with safe access
    # -------------------------------------------------
    r1, r2 = results[eps1], results[eps2]

    # Helper for safe extraction with fallback keys
    def _get(bundle, keys):
        for k in keys:
            val = bundle.get(k)
            if val is not None:
                return val
        raise KeyError(f"Data missing: {keys}")

    # Extract singular values, loadings, and coordinates
    sv1 = _get(r1, ['singular_values'])[:n_components]
    sv2 = _get(r2, ['singular_values'])[:n_components]
    L1  = _get(r1, ['loadings_norm'])[:n_components]
    L2  = _get(r2, ['loadings_norm'])[:n_components]
    C1  = _get(r1, ['coords_norm'])[:, :n_components]
    C2  = _get(r2, ['coords_norm'])[:, :n_components]

    # -------------------------------------------------
    # 2-B: Compute stability metrics for variance structure, 
    # loading alignment, subspace rotation, and embedding consistency
    # -------------------------------------------------
    # Global variance-structure drift (L2 difference of singular values)
    sv_diff_norm = norm(sv1 - sv2)

    # Directional stability: cosine similarity of loading vectors
    cosines = np.einsum('ij,ij->i', L1, L2) / (norm(L1, axis=1) * norm(L2, axis=1))

    # Rotation-invariant drift: principal angles between PC subspaces
    angles_deg = np.degrees(subspace_angles(L1.T, L2.T))

    # Embedding stability: correlation of sample coordinates
    coord_corrs = np.array([
        np.corrcoef(C1[:, i], C2[:, i])[0, 1]
        for i in range(n_components)
    ])

    # Identify the PC with the largest directional drift
    worst_idx = np.argmin(cosines)
    worst_val = cosines[worst_idx]

    # -------------------------------------------------
    # 2-C: Compute a composite stability score combining all metrics into a unified index
    # -------------------------------------------------
    # Penalize singular-value drift
    sv_score = np.exp(-sv_diff_norm)

    # Average cosine similarity of loading vectors
    cos_score = np.mean(cosines)

    # Average correlation of sample coordinates
    coord_score = np.mean(coord_corrs)

    # Penalize large subspace rotations
    angle_score = np.exp(-np.radians(angles_deg).mean())

    # Unified stability index
    composite_score = np.mean([sv_score, cos_score, coord_score, angle_score])

    # -------------------------------------------------
    # 2-D: Print a formatted summary of stability metrics and optionally render a diagnostic dashboard
    # -------------------------------------------------
    width, lbl_w, val_w = 65, 37, 12
    print("\n" + "═" * width)
    print(f"{'🔍 PCA STABILITY CROSS-CHECK':^{width}}")
    print(f"{f'(ε={eps1} vs ε={eps2})':^{width}}")
    print(f"{f'Number of Principal Components: {n_components}':^{width}}")
    print("─" * width)
    print(f"⚖️  {'SV Magnitude Shift (L2)':<{lbl_w}} : {sv_diff_norm:>{val_w}.4f}")
    print(f"📐  {'Max Subspace Angle':<{lbl_w}} : {f'{angles_deg.max():.2f}°':>{val_w}}")
    print(f"💎  {'Composite Health Score':<{lbl_w}} : {composite_score:>{val_w}.4f}")
    print("─" * width)
    for i in range(n_components):
        status = "✅" if cosines[i] >= drift_threshold else "⚠️"
        print(f"PC{i+1} Robustness (Cosine Similarity) {status:<5} : {cosines[i]:>{val_w}.4f}")
    print("═" * width + "\n")

    # -------------------------------------------------
    # 2-E: If show_plot is True, render a 3-panel diagnostic dashboard visualizing 
    # loading stability, subspace rotation, and contribution patterns
    # -------------------------------------------------
    if show_plot:
        sns.set_theme(style="white", context="talk")
        plt.rcParams['font.family'] = 'sans-serif'

        fig = plt.figure(figsize=(24, 8), facecolor="#F8F9FA", constrained_layout=True)
        gs = fig.add_gridspec(1, 3)

        # Color palette for the dashboard elements
        accent, drift, safe, neutral = "#1A5276", "#CB4335", "#28B463", "#D5DBDB"

        # Panel A: Loading stability with cosine similarity and coordinate correlation
        ax1 = fig.add_subplot(gs[0, 0])
        x, bw = np.arange(n_components), 0.35
        pcs = [f"PC{i+1}" for i in range(n_components)]
        # Plot loading cosine bars with conditional coloring based on drift threshold, 
        # and a distinct color for the worst drift
        ax1.bar(
            x - bw/2,
            cosines,
            bw,
            color=[drift if i == worst_idx else accent for i in range(n_components)],
            label='Loading Cosine',
            edgecolor='white',
            lw=2,
            alpha=0.9,
            zorder=3
        )
        # Overlay coordinate correlation bars with a different color and slight transparency
        ax1.bar(
            x + bw/2,
            coord_corrs,
            bw,
            color=neutral,
            label='Coord Corr',
            edgecolor='white',
            lw=2,
            alpha=0.7,
            zorder=3
        )
        # Reference line for drift threshold
        ax1.axhline(
            drift_threshold,
            color=drift,
            ls='--',
            lw=1.5,
            alpha=0.4,
            label='Stability Threshold',
            zorder=2
        )
        # Annotate the worst loading drift with a callout box and arrow
        ax1.annotate(
            f'LARGEST DRIFT: {pcs[worst_idx]}',
            xy=(worst_idx - bw/2, worst_val),
            xytext=(worst_idx - bw/2, worst_val + 0.12),
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='900',
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=drift, lw=2),
            arrowprops=dict(arrowstyle="->", color=drift, lw=1.5),
            zorder=5
        )

        ax1.set_title("COMPONENT ROBUSTNESS", fontweight="900", size=14, pad=25, loc='left')
        ax1.set_ylim(min(0.5, worst_val - 0.15), 1.3)
        ax1.set_xticks(x)
        ax1.set_xticklabels(pcs, fontweight="bold")
        ax1.legend(frameon=False, loc='lower left', bbox_to_anchor=(0, -0.28), ncol=3, fontsize=10)

        # Panel B: Subspace rotation
        ax2 = fig.add_subplot(gs[0, 1])
        angle_labels = [f"θ{i+1}" for i in range(len(angles_deg))]
        sns.barplot(
            x=angle_labels,
            y=angles_deg,
            ax=ax2,
            hue=angle_labels,
            palette="GnBu_d",
            legend=False,
            alpha=0.8
        )
        # Annotate the maximum angle with a label and an arrow
        max_v = angles_deg.max()
        ax2.annotate(
            f'MAX TILT: {max_v:.2f}°',
            xy=(0, max_v),
            xytext=(0, max_v + (max_v * 0.3)),
            ha='center',
            fontweight='bold',
            size=11,
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=safe, lw=2),
            arrowprops=dict(arrowstyle="->", color=safe)
        )

        ax2.set_title("SUBSPACE ROTATION", fontweight="900", size=14, pad=25, loc='left')
        ax2.set_ylabel("Degrees (°)")
        ax2.set_ylim(0, max_v * 1.8)

        # Panel C — Contribution matrix
        ax3 = fig.add_subplot(gs[0, 2])
        U, _, _ = np.linalg.svd(L1 @ L2.T)
        contrib_norm = np.abs(U.T) / np.abs(U.T).sum(axis=0, keepdims=True)
        # Note: The contribution matrix is visualized as a heatmap where each cell represents 
        # the normalized contribution of a PC from the first decomposition to the tilt observed 
        # in the second decomposition. The heatmap uses a blue color palette to indicate the 
        # strength of contributions, with annotations showing the exact values for clarity. 
        # This panel provides insight into which components are most responsible for the 
        # observed drift, complementing the loading stability and subspace rotation analyses
        # in the first two panels. 
        sns.heatmap(
            contrib_norm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            cbar=False,
            xticklabels=angle_labels,
            yticklabels=pcs,
            ax=ax3,
            linewidths=2,
            linecolor='white',
            annot_kws={"size": 12, "fontweight": "bold"}
        )
        ax3.set_title("PC CONTRIBUTION TO TILT", fontweight="900", size=14, pad=25, loc='left')

        sns.despine(left=True, bottom=True)
        for ax in [ax1, ax2]:
            ax.yaxis.grid(True, linestyle='--', alpha=0.3)

        plt.suptitle(
            f"PCA STABILITY DIAGNOSTIC: COMPOSITE HEALTH {composite_score:.2f}",
            fontsize=24,
            fontweight='900',
            color='#1B2631',
            y=1.08
        )
        
        # Save the figure if image_save and image_path are provided
        if image_save and image_path:
            fig.savefig(image_path+image_save, dpi=300, bbox_inches='tight')
        
        plt.show()

    return {"composite_score": composite_score, "angles": angles_deg, "cosines": cosines}


# ---------------------------------------------------------------------------
# 3. PCA sensitivity visualization across multiple epsilon values
# ---------------------------------------------------------------------------
def plot_pca_results(
    results: Dict,
    title: str = "PCA Noise Sensitivity Analysis",
    threshold_var: float = 0.80,
    k_components: int = None,
    image_path: str = None,
    image_save: str = None
):
    """
    Visualize PCA sensitivity across multiple epsilon values.

    Produces a two‑panel diagnostic figure comparing:
      • The singular value spectrum across eps values
      • The cumulative variance explained across eps values

    The function automatically determines the target number of components
    either by:
      (A) user‑specified k_components, or
      (B) the smallest k such that cumulative variance ≥ threshold_var.

    The selected k is highlighted on both panels with vertical and
    horizontal reference lines. Tick labels corresponding to the selected
    component are emphasized for readability.

    Parameters
    ----------
    results : dict
        Mapping eps → PCA result bundles containing singular values and
        variance ratios.

    title : str, default "PCA Noise Sensitivity Analysis"
        Main title for the figure.

    threshold_var : float, default 0.80
        Minimum cumulative variance required to determine k if
        k_components is not provided.

    k_components : int, optional
        Explicit number of components to highlight. Overrides threshold_var.

    save_path : str, optional
        Optional path to save the figure.

    Returns
    -------
    None
        Displays the figure and optionally saves it.
    """
    # -------------------------------------------------
    # 3-A: Setup and extract baseline PCA info
    # -------------------------------------------------
    sns.set_theme(style="white")
    plt.rcParams["font.family"] = "sans-serif"
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5))

    # Note: The function begins by setting up a clean and modern visual 
    # style using Seaborn's white theme and a sans-serif font. It then 
    # extracts the singular values and variance ratios for the first epsilon 
    # value in the results, which serves as the baseline for determining the 
    # target number of components (k). The color palette is defined to ensure 
    # that each epsilon value is visually distinct in the subsequent plots, 
    # enhancing readability and interpretability of the PCA sensitivity 
    # analysis across different noise thresholds.  
    epsilons = sorted(results.keys())
    palette = ["#2d3436", "#d85a30"]

    # Extract singular values and variance ratios for the
    first_eps = epsilons[0]
    s_vals_first = results[first_eps]["singular_values"]
    v_ratios_first = results[first_eps]["variance_ratio"]
    cum_var_first = np.cumsum(v_ratios_first)

    # -------------------------------------------------
    # 3-B: Determine target number of components (k)
    # -------------------------------------------------
    if k_components is not None:
        target_k = int(k_components)
    elif threshold_var is not None:
        target_k = int(np.argmax(cum_var_first >= threshold_var) + 1)
    else:
        target_k = 3
    # Note: The target number of components (k) is determined based on either a 
    # user‑specified value or a variance threshold. If k_components is provided, 
    # it is used directly. Otherwise, the function computes the cumulative variance 
    # explained by the components and selects the smallest k such that the cumulative 
    # variance meets or exceeds the specified threshold (default 80%). This dynamic 
    # selection ensures that the visualization focuses on the most relevant components 
    # for interpreting PCA sensitivity across epsilon values. 
    var_at_k = cum_var_first[target_k - 1]
    k_singular_val = s_vals_first[target_k - 1]

    # -------------------------------------------------
    # 3-C: Global titles and subtitles
    # -------------------------------------------------
    subtitle = f"Variance Captured at k={target_k}: {var_at_k:.2f} ({var_at_k:.1%})"
    fig.suptitle(title, fontsize=18, fontweight="bold", x=0.5, ha="center", y=0.97)
    fig.text(0.5, 0.92, subtitle, fontsize=12, color="#636e72", ha="center")

    # -------------------------------------------------
    # 3-D: Plot singular values and cumulative variance
    # -------------------------------------------------
    # The function iterates over each epsilon value in the results, plotting the singular 
    # value spectrum and cumulative variance explained for each. The singular values 
    # are plotted as a line with markers to show how the variance structure changes across 
    # components, while the cumulative variance plot illustrates how much total variance is 
    # captured as more components are included. Each epsilon value is represented with a distinct 
    # color from the defined palette, and legends are added to differentiate between them. 
    # This dual-panel visualization allows for a comprehensive comparison of PCA sensitivity across 
    # different noise thresholds, highlighting how the choice of epsilon affects the variance structure
    # and the interpretability of the principal components.  
    for i, eps in enumerate(epsilons):
        s_vals = results[eps]["singular_values"]
        ratios = results[eps]["variance_ratio"]

        # Singular value spectrum
        sns.lineplot(
            x=range(1, len(s_vals) + 1),
            y=s_vals,
            marker="o",
            color=palette[i],
            label=f"ε = {eps}",
            ax=axes[0]
        )

        # Cumulative variance explained
        sns.lineplot(
            x=range(1, len(ratios) + 1),
            y=np.cumsum(ratios),
            marker="o",
            color=palette[i],
            label=f"ε = {eps}",
            ax=axes[1]
        )

    # -------------------------------------------------
    # 3-E: Formatting, tick logic, and highlighting k
    # -------------------------------------------------
    two_dec_formatter = FuncFormatter(lambda x, pos: f"{x:.2f}")
    # Note: Both panels include a vertical reference line at the target 
    # number of components (k) and a horizontal reference line at the corresponding 
    # singular value (for the left panel) or cumulative variance (for the right panel). 
    # The tick labels corresponding to k are bolded and color-highlighted to draw attention 
    # to the selected component, which is critical for interpreting the stability analysis. 
    # The formatting ensures that the key insights about PCA sensitivity are immediately visible 
    # and accessible to the viewer.
    for i, ax in enumerate(axes):
        ax.set_xlabel("Component Index" if i == 0 else "Number of Components", fontweight="medium")
        ax.set_ylabel("Singular Value" if i == 0 else "Cumulative Variance", fontweight="medium")

        # Vertical line at k
        ax.axvline(target_k, color="#636e72", ls="--", lw=1.2, alpha=0.6)

        if i == 0:
            # Horizontal line at singular value of component k
            ax.axhline(k_singular_val, color="#b2bec3", ls=":", lw=1)
            yticks = list(ax.get_yticks())
            yticks = [t for t in yticks if abs(t - k_singular_val) > (max(s_vals_first) * 0.05)]
            ax.set_yticks(sorted(yticks + [k_singular_val]))
        else:
            # Horizontal line at cumulative variance of component k
            ax.axhline(var_at_k, color="#b2bec3", ls=":", lw=1.5)
            ax.yaxis.set_major_formatter(two_dec_formatter)
            yticks = list(ax.get_yticks())
            yticks = [t for t in yticks if abs(t - var_at_k) > 0.04]
            ax.set_yticks(sorted(yticks + [var_at_k]))

        # Ensure target_k is included in x‑ticks
        xticks = sorted(list(set(list(ax.get_xticks()) + [target_k])))
        ax.set_xticks([t for t in xticks if t >= 0])

        # Bold and color-highlight the tick corresponding to k
        plt.draw()
        for tick in ax.get_xticklabels():
            if tick.get_text() == str(target_k):
                tick.set_fontweight("bold")
                tick.set_color("#d85a30")

        for tick in ax.get_yticklabels():
            try:
                clean_text = tick.get_text().replace('−', '-').strip()
                val = float(clean_text)
                target = k_singular_val if i == 0 else var_at_k
                if abs(val - target) < 0.01:
                    tick.set_fontweight("bold")
                    tick.set_color("#d85a30")
            except ValueError:
                continue

    # Global styling and legend
    titles = ["Singular Value Spectrum", "Cumulative Variance Explained"]
    for i, ax in enumerate(axes):
        ax.set_title(titles[i], loc="center", fontweight="semibold", pad=20)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)

    sns.despine(trim=True)
    plt.tight_layout(rect=[0, 0.05, 1, 0.92])
    
    # Save the figure if image_save and image_path are provided
    if image_save and image_path:
        fig.savefig(image_path+image_save, dpi=300, bbox_inches='tight')
    
    plt.show()


# ---------------------------------------------------------------------------
# 4. Aitchison variance structure visualization across epsilon values
# ---------------------------------------------------------------------------
def plot_aitchison(
    clr_data: Mapping[float, pd.DataFrame],
    chosen_eps: float | None = None,
    compare_eps: float | None = None,
    annotate: bool = True,
    figsize: tuple[float, float] = (12, 5.5)
) -> dict[str, Any]:
    """
    Plot Aitchison variance diagnostics across epsilon thresholds.

    This visualization shows:
    - Total Aitchison variance (left y-axis)
    - PC1 variance ratio (right y-axis)
    - Optional vertical markers for a chosen ε and a comparison ε
    - Optional annotated callout boxes for both markers

    The plot uses a dual-axis layout:
    - Left axis: total variance (blue)
    - Right axis: PC1 ratio (coral)
    - Vertical lines: chosen ε (green), alternative ε (gray)

    Parameters
    ----------
    clr_data : Mapping[float, pd.DataFrame]
        Dictionary mapping epsilon → CLR-transformed DataFrame.
    chosen_eps : float or None
        Primary epsilon to highlight with a vertical line and annotation.
    compare_eps : float or None
        Secondary epsilon to highlight for comparison.
    annotate : bool
        Whether to draw callout boxes for chosen_eps and compare_eps.
    figsize : tuple[float, float]
        Figure size passed to matplotlib.

    Returns
    -------
    dict
        A dictionary containing:
        - "figure": the matplotlib Figure
        - "ax1": left y-axis (total variance)
        - "ax2": right y-axis (PC1 ratio)
    """
    # Color palette for the plot elements
    c_blue, c_coral, c_green, c_gray = "#185FA5", "#D85A30", "#3B6D11", "#666666"

    # Compute the Aitchison profile DataFrame from the provided CLR data
    profile = compute_aitchison_profile(clr_data)

    # Set up the dual-axis plot with a clean, modern aesthetic
    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=figsize)
    ax2 = ax1.twinx()

    # ------------------------------------------------------------------
    # 4-1. Base series: Total variance (left) and PC1 ratio (right)
    # ------------------------------------------------------------------
    # Note: The total variance is plotted on the primary y-axis with a distinct color a
    # nd marker style to ensure it stands out as the main focus of the plot. The use of 
    # circular markers and a solid line style emphasizes the continuity of the variance 
    # trend across epsilon values. 
    l1, = ax1.plot(
        profile.index, profile["total_variance"],
        color=c_blue, marker="o", ms=5, lw=1.8,
        label="Total Variance"
    )
    # Note: The PC1 ratio is plotted on the secondary y-axis with a distinct 
    # color and marker style to differentiate it from the total variance series. 
    # The dashed line style emphasizes that this is a ratio metric, while the square 
    # markers provide visual contrast to the circular markers used for total variance.
    l2, = ax2.plot(
        profile.index, profile["pc1_ratio"],
        color=c_coral, marker="s", ms=5, lw=1.8, ls="--",
        label="PC1 Ratio"
    )

    # Legend handles will be built in a specific order
    handles = [l1, l2]

    # ------------------------------------------------------------------
    # 4-2. Primary chosen epsilon marker
    # ------------------------------------------------------------------
    # Note: The chosen epsilon is highlighted with a distinct color and a vertical line.
    if chosen_eps is not None:
        v_line = ax1.axvline(
            chosen_eps, color=c_green, ls=":", lw=1.8,
            label=rf"Chosen $\epsilon$ ({chosen_eps:.2g})"
        )
        handles.append(v_line)
        # Note: The annotation for the chosen epsilon is designed to be prominent and informative,
        # with a callout box that includes key metrics. The offset is tuned to avoid overlap
        # with the data points while maintaining a clear visual connection to the marker.
        if annotate:
            # Offset tuned for readability: left-shifted, higher placement
            _add_callout(
                ax1, profile, chosen_eps,
                offset=(-60, 100),
                prefix=r"$\epsilon_{chosen}$",
                ha="right"
            )

    # ------------------------------------------------------------------
    # 4-3. Secondary comparison epsilon marker
    # ------------------------------------------------------------------
    if compare_eps is not None:
        c_line = ax1.axvline(
            compare_eps, color=c_gray, ls="--", lw=1.2, alpha=0.6,
            label=rf"Alt $\epsilon$ ({compare_eps:.2g})"
        )
        handles.append(c_line)
        # Note: The annotation for the alternative epsilon is intentionally less prominent 
        # and placed to avoid overlap with the primary marker, ensuring both are readable 
        # without cluttering the plot.
        if annotate:
            # Offset tuned for readability: right-shifted, slightly lower
            _add_callout(
                ax1, profile, compare_eps,
                offset=(60, 40),
                prefix=r"$\epsilon_{alt}$",
                ha="left"
            )

    # Apply consistent styling to axes, grid, and ticks
    _style_axes(ax1, ax2, c_blue, c_coral)
    _finalize_layout(fig, ax1, handles)

    return {"figure": fig}

# ---------------------------------------------------------------------------
# Helper function 4-A: to add annotated callout boxes for epsilon markers
# ---------------------------------------------------------------------------
def _add_callout(ax, profile, eps, offset, prefix, ha):
    """
    Add a labeled callout box pointing to the nearest epsilon value.

    Parameters
    ----------
    ax : matplotlib Axes
        Axis on which to draw the annotation.
    profile : DataFrame
        Aitchison profile indexed by epsilon.
    eps : float
        Target epsilon (will snap to nearest available index).
    offset : tuple[int, int]
        Pixel offset for the annotation text box.
    prefix : str
        LaTeX prefix for the bold label (e.g., $\\epsilon_{chosen}$).
    ha : str
        Horizontal alignment of the text ("left" or "right").
    """
    # Snap to the nearest available epsilon in the profile index
    idx = profile.index.get_indexer([eps], method="nearest")[0]
    snapped_eps = profile.index[idx]
    row = profile.iloc[idx]

    # Construct the label text with LaTeX formatting and key metrics
    label_text = (
        f"{prefix}: {snapped_eps:.2g}" + "\n"
        f"Total Var: {row.total_variance:.3g}\n"
        f"PC1 Ratio: {row.pc1_ratio:.1%}"
    )

    # Draw the annotation with a styled box and an arrow pointing to the data point
    ax.annotate(
        label_text,
        xy=(snapped_eps, row.total_variance),
        xytext=offset,
        textcoords="offset points",
        fontsize=10,
        ha=ha, va="center",
        bbox=dict(
            boxstyle="round,pad=0.5",
            fc="white", ec="#CCCCCC",
            alpha=1.0, lw=1, zorder=10
        ),
        # Arrow properties for the callout, with a subtle curve to enhance readability
        arrowprops=dict(
            arrowstyle="->",
            connectionstyle="arc3,rad=.1",
            color="#333", lw=1.5
        ),
        zorder=11
    )

# ---------------------------------------------------------------------------
# Helper function 4-B: to apply consistent styling to the Aitchison plot axes
# ---------------------------------------------------------------------------
def _style_axes(ax1, ax2, c1, c2):
    """
    Apply consistent styling to the dual-axis Aitchison plot.
    """
    # Logarithmic x-axis for better spacing of epsilon values
    ax1.set_xscale("log")

    # Axis labels with color coding and bold styling
    lbl_sz, tick_sz = 12, 10
    ax1.set_xlabel(r"$\epsilon$ (pseudocount)", size=lbl_sz, fontweight="medium")
    ax1.set_ylabel("Total Aitchison Variance", color=c1, size=lbl_sz, fontweight="bold")
    ax2.set_ylabel("PC1 Variance Ratio", color=c2, size=lbl_sz, fontweight="bold")

    # Style spines, grid, and ticks for both axes
    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.xaxis.grid(True, linestyle="--", alpha=0.3, which="both")
        ax.tick_params(labelsize=tick_sz)

    # Hide the right spine of ax1 and left spine of ax2 for a cleaner look
    ax1.spines["right"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    # Light grid on the left axis for better readability of variance values
    ax1.yaxis.grid(True, linestyle="--", alpha=0.3)

    # Color the tick labels to match their respective axes
    ax1.tick_params(axis="y", labelcolor=c1)
    ax2.tick_params(axis="y", labelcolor=c2)

# ---------------------------------------------------------------------------
# Helper function 4-C: to finalize the layout, title, and legend for the Aitchison plot
# ---------------------------------------------------------------------------
def _finalize_layout(fig, ax, handles):
    """
    Finalize title, legend, and layout spacing for the Aitchison plot.
    """
    # Main title with LaTeX formatting
    ax.set_title(
        r"CLR Variance Structure vs $\varepsilon$",
        loc="left", pad=25,
        fontweight="bold", size=13
    )

    # Legend order: Total, PC1, Chosen, Alt
    ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.22),
        ncol=4,
        frameon=False,
        fontsize=9,
        columnspacing=1.2,
        handletextpad=0.4
    )
    # Adjust the bottom margin to accommodate the legend without overlap
    plt.subplots_adjust(bottom=0.28)


# ---------------------------------------------------------------------------
# 4-4. Math Kernels (Aitchison Geometry)
# ---------------------------------------------------------------------------
"""
Math Kernels for Aitchison Geometry
-----------------------------------
This module provides helper functions for analyzing compositional data
expressed in CLR (centered log-ratio) coordinates. It includes:

- Total Aitchison variance across all CLR dimensions
- Variance ratio explained by the first principal component (PC1)
- A convenience routine to compute these metrics across multiple
  epsilon‑thresholded CLR datasets

All functions assume inputs are NumPy arrays or pandas DataFrames
containing valid finite CLR-transformed values.
"""
# ------------------------------------------------------------------
# 4-4-A: Total Aitchison variance computation
# ------------------------------------------------------------------
def get_total_aitchison_variance(X: np.ndarray) -> float:
    """
    Compute the total Aitchison variance of a CLR matrix.

    Parameters
    ----------
    X : np.ndarray
        A 2D array of CLR-transformed compositional data.

    Returns
    -------
    float
        The sum of variances across all CLR coordinates.
        Returns NaN if the array is empty or contains non-finite values.
    """
    # Validate input: must be non-empty and finite
    if X.size == 0 or not np.isfinite(X).all():
        return np.nan

    # Variance is computed per column; sum gives total Aitchison variance
    return float(np.var(X, axis=0, ddof=0).sum())

# ------------------------------------------------------------------
# 4-4-B: PC1 variance ratio computation using SVD
# ------------------------------------------------------------------
def get_pc1_variance_ratio(X: np.ndarray) -> float:
    """
    Compute the proportion of total variance explained by the first
    principal component (PC1) using SVD.

    Parameters
    ----------
    X : np.ndarray
        A 2D array of CLR-transformed compositional data.

    Returns
    -------
    float
        The variance ratio of PC1 (largest singular value squared divided
        by total variance). Returns NaN on invalid input or SVD failure.
    """
    # Validate input
    if X.size == 0 or not np.isfinite(X).all():
        return np.nan

    # Center the data before SVD
    Xc = X - X.mean(axis=0, keepdims=True)
    # SVD can fail for degenerate matrices, so we catch exceptions
    try:
        # SVD: singular values s correspond to sqrt of eigenvalues
        _, s, _ = np.linalg.svd(Xc, full_matrices=False)

        # PC1 variance ratio = s1^2 / sum(s_i^2)
        return float(s[0]**2 / np.sum(s**2))

    except Exception:
        # SVD can fail for degenerate matrices
        return np.nan

# ------------------------------------------------------------------
# 4-4-C: Convenience routine to compute Aitchison profile across multiple epsilons
# ------------------------------------------------------------------
def compute_aitchison_profile(clr_data):
    """
    Compute the total Aitchison variance and the PC1 variance ratio 
    for a dictionary of CLR datasets indexed by epsilon thresholds.

    Parameters
    ----------
    clr_data : dict
        A mapping {eps: DataFrame} where each value is a pandas DataFrame
        containing CLR-transformed data for that epsilon threshold.

    Returns
    -------
    pandas.DataFrame
        A DataFrame indexed by epsilon with columns:
        - 'total_variance'
        - 'pc1_ratio'
    """
    records = []

    # Iterate in sorted epsilon order for reproducibility
    for eps in sorted(clr_data):
        # Convert DataFrame to float NumPy array
        m = clr_data[eps].values.astype(float)

        # Compute metrics
        records.append({
            "eps": eps,
            "total_variance": get_total_aitchison_variance(m),
            "pc1_ratio": get_pc1_variance_ratio(m)
        })

    # Return tidy DataFrame indexed by epsilon
    return pd.DataFrame(records).set_index("eps")


# ---------------------------------------------------------------------------
# 5. PC1 loading stability visualization across epsilon values
# ---------------------------------------------------------------------------
def plot_pc1_loading_stability(
    clr_data: Mapping[float, pd.DataFrame],
    chosen_eps: float | None = None,
    top_k: int = 10,
    figsize: tuple[float, float] = (12, 8),
) -> dict:
    """
    Plot PC1 loading stability across epsilon values.

    This visualization shows how the top-K high-variance features behave
    in the first principal component as epsilon varies. It highlights:

    - Feature trajectories (PC1 loadings vs epsilon)
    - Optional vertical marker for a chosen epsilon
    - A legend entry indicating the chosen epsilon
    - Log-scaled epsilon axis for clarity

    Parameters
  ----
    clr_data : Mapping[float, pd.DataFrame]
        Dictionary mapping epsilon → CLR DataFrame.
    chosen_eps : float or None
        Epsilon to highlight with a vertical line.
    top_k : int
        Number of high-variance features to track.
    figsize : tuple
        Figure size.

    Returns
  -
    dict
        {
            "figure": matplotlib Figure,
            "data": DataFrame of aligned PC1 loadings
        }
    """
    # ------------------------------------------------------------------
    # 5-1. Compute PC1 loadings for top-K features across all epsilons
    # ------------------------------------------------------------------
    df_loadings = _compute_pc1_stability(clr_data, chosen_eps, top_k=top_k)
    eps_values  = sorted(clr_data)

    # 5-2. Setup figure and color palette
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=figsize, dpi=100)

    # Distinct colors for each feature trajectory
    colors = sns.color_palette("husl", n_colors=len(df_loadings.columns))
    # ------------------------------------------------------------------
    # 5-3. Plot PC1 loadings for each feature across epsilons
    # ------------------------------------------------------------------
    for i, feature in enumerate(df_loadings.columns):
        ax.plot(
            df_loadings.index,
            df_loadings[feature],
            marker="o",
            markersize=4,
            linewidth=1.5,
            alpha=0.8,
            color=colors[i],
            label=feature,
            markeredgecolor="white",
            markeredgewidth=0.5
        )

    # ------------------------------------------------------------------
    # 5-4. Highlight chosen epsilon with a vertical line and add to legend
    # ------------------------------------------------------------------
    if chosen_eps is not None:
        # Vertical line
        ax.axvline(
            chosen_eps, linestyle="--", color="#2d3436",
            linewidth=1.2, alpha=0.6
        )

        # Shaded region around chosen epsilon for emphasis
        ax.axvspan(
            chosen_eps * 0.9, chosen_eps * 1.1,
            color="gray", alpha=0.05
        )

        # Legend proxy for chosen epsilon
        eps_proxy = Line2D(
            [0], [0],
            color="#2d3436",
            linestyle="--",
            linewidth=1.2,
            label=f"Target ε: {chosen_eps:.2g}"
        )
        # Append to existing legend handles
        handles, labels = ax.get_legend_handles_labels()
        handles.append(eps_proxy)
    else:
        handles, labels = ax.get_legend_handles_labels()

    # ------------------------------------------------------------------
    # 5-5. Formatting and styling
    # ------------------------------------------------------------------
    ax.set_xscale("log")
    ax.set_xlabel("Epsilon (ε) Pseudocount", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_ylabel("PC1 Loading Value", fontsize=14, fontweight="bold", labelpad=10)
    ax.set_title(
        f"Stability of Top {top_k} High-Variance Features",
        fontsize=16, loc="left", pad=20, fontweight="bold"
    )
    # Grid and spines
    sns.despine(trim=True, offset=10)
    ax.grid(True, axis="y", color="#f0f0f0", linestyle="-", zorder=0)
    ax.axhline(0, color="#d1d1d1", linewidth=1.0, zorder=1)

    # Tick styling
    ax.tick_params(axis="both", which="major", labelsize=12, colors="#2d3436")
    ax.tick_params(axis="x", which="minor", bottom=False)

    # ------------------------------------------------------------------
    # 5-6. Legend formatting
    # ------------------------------------------------------------------
    leg = ax.legend(
        handles=handles,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
        fontsize=10,
        title="Features & Settings"
    )
    # Style the legend title if it exists
    if leg:
        plt.setp(leg.get_title(), fontsize=12, fontweight="bold")

    # ------------------------------------------------------------------
    # 5-7. Informative caption about the feature selection method
    # ------------------------------------------------------------------
    fig.text(
        0.05, -0.05,
        f"Features selected by variance at ε={chosen_eps if chosen_eps else eps_values[0]}.",
        fontsize=9, color="#636e72", style="italic"
    )
    # Final layout adjustments
    fig.tight_layout()

    return {"figure": fig, "data": df_loadings}

# ---------------------------------------------------------------------------
# Helper functions 5-A: for PC1 stability computation
# ---------------------------------------------------------------------------
def _filter_by_variance(clr_df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """
    Select the top-K highest-variance CLR features.

    This step ensures that PCA focuses on the features that actually move
    across samples, rather than low-variance or static components.

    Parameters
  ----
    clr_df : pd.DataFrame
        CLR-transformed feature table.
    top_k : int
        Number of highest-variance features to retain.

    Returns
  -
    pd.DataFrame
        Filtered DataFrame containing only the top-K movers.
    """
    # Compute per-feature variance and sort descending
    variances = clr_df.var().sort_values(ascending=False)

    # Select the top-K feature names
    keep_cols = variances.head(top_k).index

    # Return filtered CLR matrix
    return clr_df[keep_cols]

# ---------------------------------------------------------------------------
# Helper functions 5-B: for computing and aligning PC1 loadings across epsilons
# ---------------------------------------------------------------------------
def _compute_pc1_stability(
    clr_data: Mapping[float, pd.DataFrame],
    chosen_eps: float | None = None,
    top_k: int = 10
) -> pd.DataFrame:
    """
    Compute aligned PC1 loadings across epsilon values.

    Steps:
    1. Identify the top-K high-variance features at a reference epsilon.
    2. Compute PC1 loadings for these features at every epsilon.
    3. Align the sign of PC1 across epsilons using a stable anchor feature.

    Parameters
  ----
    clr_data : Mapping[float, pd.DataFrame]
        Dictionary mapping epsilon → CLR DataFrame.
    chosen_eps : float or None
        Reference epsilon for selecting high-variance features.
        If None, the smallest epsilon is used.
    top_k : int
        Number of high-variance features to track.

    Returns
  -
    pd.DataFrame
        Rows = epsilon values, columns = selected features,
        values = aligned PC1 loadings.
    """
    eps_values = sorted(clr_data)

    # Reference epsilon for selecting movers
    ref_eps = chosen_eps if chosen_eps is not None else eps_values[0]

    # ------------------------------------------------------------------
    # 5-B-1. Identify movers at the reference epsilon
    # ------------------------------------------------------------------
    ref_df_filtered = _filter_by_variance(clr_data[ref_eps], top_k=top_k)
    target_features = ref_df_filtered.columns

    # ------------------------------------------------------------------
    # 5-B-2. Compute reference PC1 to establish orientation anchor
    # ------------------------------------------------------------------
    ref_X        = ref_df_filtered.values.astype(float)
    ref_Xc       = ref_X - ref_X.mean(axis=0)
    _, _, Vt_ref = np.linalg.svd(ref_Xc, full_matrices=False)

    # Anchor = feature with the largest absolute loading in PC1
    anchor_idx = np.argmax(np.abs(Vt_ref[0]))
    # Note: The anchor feature is selected based on the largest absolute loading in the reference PC1. 
    # This ensures that the sign alignment is based on the most influential feature, providing a stable 
    # reference point for comparing PC1 loadings across different epsilon values. By aligning the sign 
    # of the PC1 loadings to this anchor, we can meaningfully interpret the trajectories of the selected 
    # features across epsilon thresholds, as they will be oriented in a consistent manner relative to the 
    # most significant contributor to the variance in the reference dataset.
    pc1_rows = []

    # ------------------------------------------------------------------
    # 5-B-3. Compute aligned PC1 loadings for each epsilon
    # ------------------------------------------------------------------
    for eps in eps_values:
        # Use the same feature set across all epsilons
        X = clr_data[eps][target_features].values.astype(float)
        # Center the data before SVD
        Xc = X - X.mean(axis=0)

        # SVD -> PC1 loadings: Vt[0] is the first right singular vector (PC1 loadings)
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)

        # Align the sign of PC1 loadings to the reference anchor feature
        aligned_pc1 = Vt[0] * np.sign(Vt[0][anchor_idx])
        pc1_rows.append(aligned_pc1)

    return pd.DataFrame(pc1_rows, index=eps_values, columns=target_features)


# ---------------------------------------------------------------------------
# 6. Crime count visualization with a modern data-journalism style
# ---------------------------------------------------------------------------
def plot_crime_counts(
    esp_results: dict,
    column: str = None,
    figsize: Tuple[int, int] = (12, 10),
    bins: int = 35,
    show: bool = True,
) -> Dict[str, Any]:
    """
    Plot monthly crime counts using a modern “data‑journalism” visual style.

    This function produces a two‑panel figure:
    1. **Trend Plot (Top)** — A line chart showing the monthly trajectory of a
       selected crime category, with a status‑bar annotation summarizing:
       - number of zero‑count months
       - total time span

    2. **Distribution Plot (Bottom)** — A histogram (with KDE overlay) of all
       non‑zero monthly counts, including a compact statistical summary:
       - mean
       - median
       - maximum observed count

    The design emphasizes:
    - consistent typography across panels
    - aligned “status bar” annotations
    - clean, minimalistic styling suitable for reports or journalism‑style graphics

    Parameters
    ----------
    esp_results : dict
        Dictionary containing processed ESP output. Must include a key
        `"pivot_data"` mapping to a DataFrame indexed by dates.
    column : str
        Name of the crime category (column in `pivot_data`) to visualize.
    figsize : tuple of int
        Size of the overall figure (width, height).
    bins : int
        Maximum number of histogram bins for the distribution plot.
    show : bool
        Whether to display the figure immediately.

    Returns
    -------
    dict
        {
            "series": pd.Series of monthly counts (index reset),
            "fig": matplotlib Figure object
        }
    """
    # ------------------------------------------------------------
    # Robust Data Extraction
    # ------------------------------------------------------------
    pivot = esp_results.get("pivot_data")
    if pivot is None or column not in pivot:
        raise KeyError(f"Data or column '{column}' not found.")

    # Extract the time series and preserve original date index
    series_raw = pd.Series(pivot[column])
    date_index = series_raw.index
    series = series_raw.reset_index(drop=True)

    # Non-zero values for distribution analysis
    nonzero = series[series > 0]

    # ------------------------------------------------------------
    # Theme Configuration (consistent across both subplots)
    # ------------------------------------------------------------
    sns.set_theme(style="white", context="paper")

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize,
        gridspec_kw={'height_ratios': [1.2, 1]}
    )

    PRIMARY    = "#1A237E"  # Deep navy for main line
    ACCENT     = "#00BFA5"  # Emerald for histogram
    TITLE_SIZE = 16
    LABEL_SIZE = 9

    # ------------------------------------------------------------
    # 6-A. TOP PLOT — Monthly Trend Line
    # ------------------------------------------------------------
    sns.lineplot(
        x=series.index, y=series.values,
        ax=ax1, color=PRIMARY, linewidth=2
    )

    # Soft fill under the line for visual depth
    ax1.fill_between(series.index, series.values, color=PRIMARY, alpha=0.04)

    # Title and axis labels
    ax1.set_title(
        column.upper(), loc='left',
        fontsize=TITLE_SIZE, fontweight='800',
        pad=30, color="#111111"
    )
    ax1.set_ylabel("INCIDENTS", fontsize=8, fontweight='700',
                   labelpad=12, color="#9E9E9E")
    ax1.set_xlabel("")
    ax1.set_xlim(0, len(series) - 1)

    # Year ticks (approx. 6 evenly spaced)
    tick_pos = np.linspace(0, len(series) - 1, 6, dtype=int)
    ax1.set_xticks(tick_pos)
    ax1.set_xticklabels(
        [
            date_index[i].strftime('%Y')
            if hasattr(date_index[i], 'strftime')
            else str(date_index[i])[:4]
            for i in tick_pos
        ],
        fontsize=9, color="#757575"
    )

    # Status-bar annotation (aligned right)
    top_info = (
        f"ZERO MONTHS: {len(series) - len(nonzero)}   •   "
        f"TOTAL SPAN: {len(series)}"
    )
    ax1.text(
        1.0, 1.05, top_info,
        transform=ax1.transAxes,
        ha='right', va='bottom',
        fontsize=LABEL_SIZE, fontweight='bold',
        color="#757575", fontfamily='monospace'
    )

    # ------------------------------------------------------------
    # 6-B. BOTTOM PLOT — Distribution of Non-Zero Counts
    # ------------------------------------------------------------
    if not nonzero.empty:
        # Smart binning for small integer ranges
        actual_bins = (
            min(bins, int(nonzero.max() - nonzero.min()) + 1)
            if nonzero.max() < 10 else bins
        )

        sns.histplot(
            nonzero, bins=actual_bins, kde=True,
            ax=ax2, color=ACCENT, edgecolor="white", alpha=0.5
        )

        # Style the KDE line if present
        if ax2.lines:
            ax2.lines[0].set_color(PRIMARY)
            ax2.lines[0].set_linewidth(1.5)

        ax2.set_title(
            "OBSERVED INTENSITY", loc='left',
            fontsize=TITLE_SIZE - 2, fontweight='800',
            pad=30, color="#111111"
        )
        ax2.set_xlabel(
            "COUNT PER MONTH",
            fontsize=8, fontweight='700',
            labelpad=10, color="#9E9E9E"
        )

        # Status-bar annotation (mirrors top plot)
        stats_text = (
            f"AVG {nonzero.mean():.1f}   •   "
            f"MED {nonzero.median():.1f}   •   "
            f"MAX {nonzero.max():.0f}"
        )
        ax2.text(
            1.0, 1.05, stats_text,
            transform=ax2.transAxes,
            ha='right', va='bottom',
            fontsize=LABEL_SIZE, fontfamily='monospace',
            color="#757575", fontweight='bold'
        )
    else:
        # If all values are zero
        ax2.text(
            0.5, 0.5, "NO DATA RECORDED",
            ha='center', va='center',
            color="#999999", fontweight='bold'
        )

    # ------------------------------------------------------------
    # Final Aesthetic Polish
    # ------------------------------------------------------------
    sns.despine(offset=20, trim=True)
    ax1.grid(axis='y', color="#EEEEEE", linestyle='-', linewidth=0.5)
    ax2.grid(axis='y', color="#EEEEEE", linestyle='-', linewidth=0.5)

    plt.tight_layout(pad=5.0)

    if show:
        plt.show()

    return {"series": series, "fig": fig}