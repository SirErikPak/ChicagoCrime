# Chicago Crime Analysis

**Work In Progress.....**

The data from the Chicago Data Portal and Crime Data set were enriched using multiple datasets from the portal. We initially stored them in the PostgreSQL database to generate the enriched dataset by joining multiple datasets using the_geom, but we discovered inconsistencies in the police beat, district, and sector fields. All missing fields were determined using multiple fields to generate the most accurate information, but there may be errors during the data wrangling process.


We designate the primary Crime dataset as the authoritative source of truth. To ensure consistency and address missing values, we perform internal imputation using data from other sources to fill corresponding NaN entries in location-based fields.



<img src="/Image/z-score_heatmape.png" width="3600" height="auto" />