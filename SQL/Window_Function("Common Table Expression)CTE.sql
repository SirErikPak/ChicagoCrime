-- select	year as per_year, fbi_code, count(fbi_code)
-- from	chicago.crimes
-- group by per_year, fbi_code


WITH per_crime_year_counts AS (
    SELECT 
        year as per_year,
        fbi_code,
        COUNT(*) AS per_crime_count
    FROM chicago.crimes
    GROUP BY per_year, fbi_code
),
stats AS (
    SELECT
        per_year,
        fbi_code,
        per_crime_count,
        AVG(per_crime_count) OVER (PARTITION BY per_year) AS mean_year,
        STDDEV(per_crime_count) OVER (PARTITION BY per_year) AS std_year
    FROM per_crime_year_counts
)
SELECT
    per_year,
    fbi_code,
    per_crime_count,
	mean_year,
	std_year,
    (per_crime_count - mean_year) / NULLIF(std_year, 0) AS z_score
FROM stats
ORDER BY per_year, fbi_code;
