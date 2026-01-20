SELECT setseed(0.5);

SELECT IUCR, stddev(year::numeric) AS year_std, COUNT(*)
FROM chicago.chicago_crimes
GROUP BY IUCR
ORDER BY RANDOM()
LIMIT 2;
