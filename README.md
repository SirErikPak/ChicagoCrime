# Chicago Crime Analysis (2001 -2025)

The data from the [Chicago Data Portal](https://data.cityofchicago.org/browse?category=Public+Safety&sortBy=most_accessed&page=1&pageSize=20) and Crime dataset were enriched using multiple datasets from the portal. We initially stored them in a PostgreSQL database to generate the enriched dataset by joining multiple datasets using the_geom. However, we discovered inconsistencies in the police beat, district, and sector fields. All missing fields were determined using multiple fields to generate the most accurate information.

We designate the primary Crime dataset as the authoritative source of truth. To ensure consistency and address missing values, we perform internal imputation using data from other sources to fill corresponding NaN entries in location-based fields.

* ![Data Wrangling Notebook](Notebook/ChicagoCrimeWrangle.ipynb)
* ![Data Visulize Notebook](Notebook/ChicagoCrimeVisualize.ipynb)

## Understanding Z-Scores and Crime Anomalies

Z-score analysis highlights specialized anomalies—places where a specific crime type is occurring at a rate far beyond what is statistically normal for the rest of the city.

- **Z-score of 3.0**: The value is higher than about 99.87% of all values in a normal distribution (extreme outlier)
- **Z-score of 2.0**: The value is higher than about 97.7% of all values (significantly higher than historical average)
- **Z-score of -1.0**: The value is lower than about 84% of all values (mild decrease)

![Z-Score Heatmap](Image/z-score_heatmape.png)

## Interpreting the Clustering Heatmaps

### 1. The Color Intensity (The Values)

Since we are using Z-scores, look at the color first:

- **Deep Red**: The crime count for that specific year was significantly above its historical average
- **Deep Blue**: The crime count for that specific year was significantly below its historical average
- **White/Light Colors**: The crime count was "normal" or very close to the 25-year mean

### 2. The Dendrogram (The "Tree" Structure)

The branches on the top (or side) tell you how similar items are:

- **Close Neighbors**: If two crime types are connected by a short "U-shaped" branch, they have very similar patterns over the 25-year period
- **Branch Length**: The longer the vertical lines of the branch before they join, the less similar those two groups are

### 3. Reading the "Clusters" (The Groups)

- **The "Legacy" Cluster**: Crimes that were rampant in the early 2000s but have since crashed (like Drug Abuse Violations or Prostitution). These will be Red on the top-left and deep Blue on the bottom-right
- **The "Resilient" Cluster**: Crimes that have stayed consistently high or "Red" for almost the entire 25 years. These are the persistent issues Chicago faces
- **The "Modern Surge" Cluster**: Crimes that were blue for years but have recently turned bright Red (like Motor Vehicle Theft and Weapons Violations)

### 4. Reading the X and Y Axis Together

- **Vertical Red Streaks**: If you see a vertical column of red across many crimes for a single year (like 2020), it suggests an External Event (like the Pandemic or civil unrest) affected almost everything at once
- **Horizontal Red Streaks**: If one crime type is red across many years, it shows a long-term sustained increase for that specific category
- **Anomalies**: Look for a single "pop" of color that breaks a trend

### 5. Why the "Rotation" Helps

By rotating the plot, you can now scan horizontally to see which crimes behave similarly in the same year, and scan vertically to see how a single crime's "fortune" has changed from 2001 to 2025.

![Cluster Heatmap Analysis](Image/cluster_Heatmap_analysis.png)

## Cluster Indexed Crime Analysis [Notebook](Notebook/ChicagoCrimeExploratoryAnalysis.ipynb)
### Summary (Indexed Crimes)
#### Correlation Distance + Complete Linkage on Z‑Scored Crime Time Series
This analysis examines how indexed crime categories in the dataset move over time by applying hierarchical clustering to **Z‑scored time series** using **correlation distance** and **complete linkage**. This configuration isolates **patterns of co‑movement**, allowing us to identify crime types that rise and fall together relative to their own historical baselines.

#### Key Insight
The resulting clusters do **not** reflect crime volume or severity. Instead, they reveal **synchronized temporal behavior**, crime types that tend to spike, dip, or shift together. These shared patterns often point to common underlying drivers such as environmental conditions, social dynamics, or policy changes.

#### Cluster Findings
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
##### Conclusion
Using Z‑scores with correlation distance and complete linkage reveals meaningful structure in the temporal behavior of crime categories. The clusters represent **coherent groups of crimes that move together over time**, offering valuable insight for forecasting, operational planning, and strategic decision‑making.