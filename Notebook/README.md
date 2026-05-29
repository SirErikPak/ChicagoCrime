{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "a4e5469b-127a-4a13-a841-a877625f26ca",
   "metadata": {},
   "source": [
    "# Chicago Crime Composition Analysis (2001–2026)\n",
    "\n",
    "> Testing whether Chicago's crime composition broke at COVID or whether the changes that look like a COVID shock were already underway.\n",
    "\n",
    "![PC1 trajectory with changepoint methods](figures/regime_segmentation.png)\n",
    "\n",
    "## Abstract\n",
    "\n",
    "Many cities responded to COVID-era crime trends by reorganizing policing budgets and policies on the assumption that the pandemic caused a discrete break in how crime occurred. Using 25 years of Chicago crime data (2001–2026, $\\approx$8.5 million reported incidents across 25 FBI categories), this project tests that assumption: did Chicago's crime composition, the mix of crime types reported each month, actually shift during the pandemic, or were the changes that look like a COVID shock already underway?\n",
    "\n",
    "The data shows the latter. Four independent statistical methods all point to gradual compositional evolution rather than a discrete pandemic-era regime shift. The largest post-pandemic changes trace to the multi-year decline of drug-enforcement categories, not to the pandemic itself.\n",
    "\n",
    "## Key Findings\n",
    "\n",
    "- **No detected regime break at COVID.** Pelt changepoint detection with BIC penalty returns zero breakpoints across penalty multipliers from  $0.5\\times$ to  $4\\times$ BIC, and across minimum-segment-length choices from 12 to 36 months. The pandemic period does not register as a structural shift in the multivariate compositional system.\n",
    "\n",
    "- **The largest post-2020 changes are tail-of-trend, not shock.** Drug Abuse Violations declined steadily from a 2008 peak; the post-COVID share is the continuation of a 15-year trajectory. Liquor Laws broke structurally in 2015, five years before COVID. Only three of 24 categories show a clean single-break point under Zivot-Andrews testing.\n",
    "\n",
    "- **The dependence structure between crime types shifted across eras, but the mechanism is compositional, not behavioral.** Drug Abuse Violations and Liquor Laws are both high-share, declining categories that drive most of the cross-era correlation shifts. As they shrank, the negative correlations they mechanically imposed on other categories' shares weakened. This is compositional pressure release, not the formation of new criminal patterns.\n",
    "\n",
    "## Methods Used\n",
    "\n",
    "- **Centered Log-Ratio (CLR) transformation** with grid-searched pseudocount selection and adjacent-value sensitivity check\n",
    "- **Three-way sensitivity structure** (unscaled vs z-scored; full vs vice-excluded) following Aitchison's compositional framework\n",
    "- **Principal Component Analysis** on the pooled 304-month panel\n",
    "- **Per-era covariance estimation** with Ledoit-Wolf shrinkage (essential for the small COVID-era sample)\n",
    "- **Stationarity testing** (ADF, KPSS, Zivot-Andrews) with FDR correction across 24 dense categories\n",
    "- **Block bootstrap** (6-month blocks, 10,000 iterations) for era-pair mean comparison\n",
    "- **Multivariate changepoint detection**: Dynp (confirmatory, forced n=2) and Pelt (exploratory, BIC penalty sweep with dimensionality-scaled penalties)\n",
    "- **Frobenius-distance comparison** of per-era LW-shrunk correlation matrices, with two within-era baselines (halves and short-window resampling)\n",
    "- **Element-wise correlation differences** to identify the specific category-pairs driving cross-era dependence shifts\n",
    "\n",
    "## What I Learned \n",
    "\n",
    "**Following the data when it contradicts the framing.** My initial scope statement asked, \"Where are the structural breaks during COVID?\" assuming the data would identify break dates near the pandemic boundaries. The changepoint analysis rejected that framing, and Pelt found zero breaks under conventional BIC penalties, and Dynp's forced breaks landed years before COVID. I rewrote the scope to test the assumption rather than confirm it, and rewrote the conclusion to clearly present the negative result. This cost a few revision cycles but is the right move; portfolios that find what they expected to find are less interesting than portfolios that let the data overturn the framing.\n",
    "\n",
    "**Documenting data quality issues honestly rather than working around them.** The companion visualization notebook surfaced a Ward field artifact: $approx$694K incidents tagged to Ward 50 turned out to be an ETL-pipeline missing-data sentinel, not a real ward attribution. Filtering would have removed both the bad rows and $\\approx$91K legitimate West Ridge incidents that couldn't be separated. Rather than present ward-level findings derived from contaminated data, I omitted the Ward section with a clear note explaining what was found and why the omission was the right call. Documenting why a section *isn't* there is better than presenting fragile numbers; in industry roles, knowing when not to ship is a load-bearing skill.\n",
    "\n",
    "**Sample-size discipline on multivariate methods.** Per-era PCA would have been a natural extension, but the COVID era has 34 months against 25 dimensions, which is well below the 5d–10d threshold for stable eigenvector estimation. I used Frobenius distance and element-wise correlation differences as descriptive alternatives that don't require eigenvector stability, and explicitly noted in the writeup why per-era PCA was not attempted. Choosing the right tool for the data scale is a judgment call that's easier to get wrong than right.\n",
    "\n",
    "## Limitations\n",
    "\n",
    "- **Reporting lag** - The Chicago Data Portal backfills recent months. April 2026 data is partly affected; the post-COVID tail should be interpreted with this in mind.\n",
    "- **Enforcement vs. occurrence** - Reported counts measure police activity, not crime occurrence. For enforcement-driven categories (drug abuse, vice, weapons, liquor laws), counts reflect enforcement priorities. Because CLR shares are compositionally coupled, shifts in the enforcement of high-share categories propagate to the share statistics of every other category.\n",
    "- **Compositional vs. absolute** - All findings describe changes in *relative* crime composition, not changes in absolute incident counts.\n",
    "- **Ward analysis omitted** due to data-quality artifacts in the source field (see notebook for diagnostic).\n",
    "\n",
    "## What I'd Do Next\n",
    "\n",
    "- **Multi-city comparison.** Replicating this analysis on Los Angeles or New York would test whether Chicago's gradual-evolution finding generalizes or is city-specific.\n",
    "- **Out-of-sample validation. Training the dependence-structure model on pre-2020 data and projecting onto post-2020 months would quantify how anomalous the pandemic period is relative to a pre-pandemic baseline.\n",
    "- **Generative model.** A Hidden Markov Model on the CLR series would let the data speak to whether discrete states exist (predicted answer based on the current null: no, but worth confirming).\n",
    "\n",
    "## Repository Structure"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "28f0095b-c91e-4fb0-86a4-a18b837b5f11",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "AB_test",
   "language": "python",
   "name": "ab_test"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
