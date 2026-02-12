# Chicago Crime Analysis (2001 - 2025)

The data from the [Chicago Data Portal](https://data.cityofchicago.org/browse?category=Public+Safety&sortBy=most_accessed&page=1&pageSize=20) and Crime dataset were enriched using multiple datasets from the portal. We initially stored them in a PostgreSQL database to generate the enriched dataset by joining multiple datasets using the_geom. However, we discovered inconsistencies in the police beat, district, and sector fields. All missing fields were determined using multiple fields to generate the most accurate information.

We designate the primary Crime dataset as the authoritative source of truth. To ensure consistency and address missing values, we perform internal imputation using data from other sources to fill corresponding NaN entries in location-based fields.

* [Data Wrangling Notebook](Notebook/ChicagoCrimeWrangle.ipynb)
* [Data Visualize Notebook](Notebook/ChicagoCrimeVisualize.ipynb)
* [Data Visualizing Covid Notebook](Notebook/ChicagoCrimeCovidVisualize.ipynb)
* [Structural Analysis of Crime (Pre$\rightarrow$Covid$\rightarrow$Post)](ChicagoCrimeAnalysisEra.ipynb)
* [Cluster Analysis Notebook](Notebook/ChicagoCrimeExploratoryClusterAnalysis.ipynb)

## Chicago Crime Plot (2001 - 2025)

![Percentage Crime Plot](Image/scatter_crime_plot.png)

## Understanding Percentage of Crime (Pre COVID & COVID & Post COVID)

![Crime Plot](Image/bar_analysis.png)

![Panemic Impact Bar](Image/impact_of_covid.png)

![Panemic Impact Box - Indexed](Image/box_indexed_crime_plot.png)

![Panemic Impact Box - Non-Indexed](Image/box_nonindexed_crime_plot.png)

## Summary

The dataset provides a fascinating look at how crime patterns shifted across the "Pre-COVID," "COVID," and "Post-COVID" eras. Looking at the raw counts and the percentage changes, a few clear narratives emerge regarding public safety and social behavior during these periods. The data tells a very consistent story: while total volumes for "traditional" street crimes like Burglary or Drug Abuse dropped significantly, the "quality" or severity of crime shifted toward more violent or high-impact categories like Weapons Violations and Motor Vehicle Theft.

* The most jarring shifts occurred in high-impact violent crimes and weapons violations. Even as many other crimes dropped, these categories saw massive growth from the Pre-Covid to Post-Covid eras.
* Homicide (1st or 2nd Degree): Saw a 54.2% increase from Pre-COVID to Post-COVID. Interestingly, it peaked during the Covid period and has since cooled slightly (-36.5% from Covid to Post-Covid), though it remains well above the baseline.
* Weapons Violations: This is the most extreme shift in the table. There was a 165.4% increase from Pre-COVID to Post-COVID.
* Motor Vehicle Theft: A massive surge in the Post-Covid era, up 91.1% compared to the pre-COVID baseline. Unlike Homicide, this trend is still accelerating, growing 29.5% just between the Covid and Post-Covid periods.
* The "Lockdown" Effect (The Declines) Crimes that generally require "face-to-face" interaction or people being out in public spaces saw massive drops, many of which have not returned to previous levels.
* Drug Abuse Violations: Dropped by 76% from Pre-COVID to Post-COVID.Prostitution & Gambling: Both saw near-total collapses (down 91% and 96.7% respectively), likely due to the closure of physical venues and shifts in police priorities.
* Burglary: Dropped by 50.5% Post-Covid. This is often attributed to more people working from home, making residential targets "occupied" and therefore riskier for burglars.
* The Fraud stands out as an outlier while it spiked by 61% from pre-COVID to post-COVID. However, the data shows an incredibly high "Impact" score during the Covid period (108.9%), suggesting a massive wave of activity (potentially related to relief fund scams or digital shift) that has begun to subside.
* Comparison of Key Offenses (Proportions of Total Crime): The shift in the "nature" of crime is best seen in the Crime Proportion columns. While Larceny-Theft remains the most common crime (roughly 21-23% of all records), other categories shifted the "makeup" of the city's crime profile:

## Understanding Z-Scores and Crime Anomalies

Z-score analysis highlights specialized anomalies—places where a specific crime type is occurring at a rate far beyond what is statistically normal for the rest of the city.

* **Z-score of 3.0**: The value is higher than about 99.87% of all values in a normal distribution (extreme outlier)
* **Z-score of 2.0**: The value is higher than about 97.7% of all values (significantly higher than historical average)
* **Z-score of -1.0**: The value is lower than about 84% of all values (mild decrease)

![Z-Score Heatmap](Image/z-score_heatmape.png)

## Interpreting the Clustering Heatmaps

### 1. The Color Intensity (The Values)

Since we are using Z-scores, look at the color first:

* **Deep Red**: The crime count for that specific year was significantly above its historical average
* **Deep Blue**: The crime count for that specific year was significantly below its historical average
* **White/Light Colors**: The crime count was "normal" or very close to the 25-year mean

### 2. The Dendrogram (The "Tree" Structure)

The branches on the top (or side) tell you how similar items are:

* **Close Neighbors**: If two crime types are connected by a short "U-shaped" branch, they have very similar patterns over the 25-year period
* **Branch Length**: The longer the vertical lines of the branch before they join, the less similar those two groups are

### 3. Reading the "Clusters" (The Groups)

* **The "Legacy" Cluster**: Crimes that were rampant in the early 2000s but have since crashed (like Drug Abuse Violations or Prostitution). These will be Red on the top-left and deep Blue on the bottom-right
* **The "Resilient" Cluster**: Crimes that have stayed consistently high or "Red" for almost the entire 25 years. These are the persistent issues Chicago faces
* **The "Modern Surge" Cluster**: Crimes that were blue for years but have recently turned bright Red (like Motor Vehicle Theft and Weapons Violations)

### 4. Reading the X and Y Axis Together

* **Vertical Red Streaks**: If you see a vertical column of red across many crimes for a single year (like 2020), it suggests an External Event (like the Pandemic or civil unrest) affected almost everything at once
* **Horizontal Red Streaks**: If one crime type is red across many years, it shows a long-term sustained increase for that specific category
* **Anomalies**: Look for a single "pop" of color that breaks a trend

### 5. Why the "Rotation" Helps

By rotating the plot, you can now scan horizontally to see which crimes behave similarly in the same year, and scan vertically to see how a single crime's "fortune" has changed from 2001 to 2025.

![Cluster Heatmap Analysis](Image/cluster_Heatmap_analysis.png)

## Cluster Indexed Crime Analysis [Notebook](Notebook/ChicagoCrimeExploratoryClusterAnalysis.ipynb)

### Summary (Indexed Crimes)

#### Correlation Distance + Complete Linkage on Z‑Scored Crime Time Series

This analysis examines how indexed crime categories in the dataset move over time by applying hierarchical clustering to **Z‑scored time series** using **correlation distance** and **complete linkage**. This configuration isolates **patterns of co‑movement**, allowing us to identify crime types that rise and fall together relative to their own historical baselines.

#### Key Insight

The resulting clusters do **not** reflect crime volume or severity. Instead, they reveal **synchronized temporal behavior**, crime types that tend to spike, dip, or shift together. These shared patterns often point to common underlying drivers such as environmental conditions, social dynamics, or policy changes.

#### Cluster Findings

![Cluster Plot](Image/cluster_index_analysis.png)

##### Cluster 1: Lethal Violence

* Homicide – 1st or 2nd Degree
* Involuntary Manslaughter / Reckless Homicide
  * These offenses show tightly aligned anomaly patterns. When one deviates from its norm, the other typically does as well. This suggests shared situational or behavioral dynamics influencing lethal outcomes.

##### Cluster 2: Interpersonal Violence + Motor Vehicle Theft

* Aggravated Assault
* Criminal Sexual Assault
* Motor Vehicle Theft
  * Despite spanning both violent and property crime, these categories exhibit remarkably similar temporal rhythms. Their co‑movement suggests that broader social or environmental factors may be influencing both interpersonal violence and vehicle theft.

##### Cluster 3: Property Crime + Related Offenses

* Aggravated Battery
* Arson
* Burglary
* Larceny – Theft
* Robbery
  * This larger cluster reflects a group of offenses that share long‑term trends and seasonal patterns. Their synchronized behavior suggests they respond similarly to opportunity structures, economic conditions, or shifts in routine activity patterns.

#### Centroids Analysis

![Centroids Plot](Image/cluster_centroids_analysis.png)

#### Insights & Observations

* Inverse Relationships: Cluster 1 and Cluster 3 often move in opposite directions. For example, in 2020, Cluster 1 reached a high of 1.62, while Cluster 3 reached a low of -0.91.
* This group stayed below average for nearly 15 years, hitting its lowest point around 2013. However, it saw a surge starting in 2016, peaking in 2020. Interestingly, it shows a very sharp downward trend toward 2025.
* Cluster 2 started very strongly in the early 2000s but crashed hard between 2012 and 2015. It had a brief surge in the late 2010s but is currently drifting back into negative territory.
* Cluster 3 maintained high positive values for the first decade (2001–2011). Since 2014, it has been in long-term decline and has remained the lowest-performing group for several years.
* The 2024-2025 Pivot: All three clusters are currently showing a downward trend as of 2025. Cluster 1, which was dominant during the pandemic years, has crashed back down to -0.95.
* Volatility: Cluster 1 shows the most dramatic "swing," moving from deep negatives in the early 2010s to sharp positives in the early 2020s.

##### Conclusion

In calculating the mean Z-score path for each cluster, we revealed the group's underlying pulse. These centroids provide a clear visualization of the shared momentum within a cluster, allowing analysts to distinguish between long-term systemic shifts and short-term anomalies.

From a strategic perspective, these centroids serve as benchmarks for operational planning and resource allocation. Instead of managing dozens of individual crime categories, decision-makers can monitor a few primary centroid paths to identify which "temporal wave" a specific offense is riding. This simplifies the transition from data to action, as departments can align their long-term strategies with the specific lifecycle, whether a steady decline or a recent surge, which is both characterized by the cluster's central path.

Finally, centroid analysis enhances forecasting accuracy by providing a stable baseline for comparative assessment. By measuring the distance between a current crime trend and its assigned centroid, analysts can detect patterns that might precede a major shift in the criminal landscape. Ultimately, these centroids transform abstract statistical correlations into a manageable set of strategic profiles, enabling more precise interventions.
