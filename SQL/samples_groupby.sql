select ward, count(ward)
from chicago.chicago_crimes
group by rollup(ward)


SELECT 
  COALESCE(ward::text, 'TOTAL') AS ward_display,
  COUNT(ward) AS non_null_count
FROM chicago.chicago_crimes
GROUP BY ROLLUP(ward)
ORDER BY 
  CASE WHEN ward IS NULL THEN 999 END NULLS LAST, 
  ward;


SELECT 
  COALESCE(ward::text, 'TOTAL'),
  COUNT(*) AS non_null_count
FROM chicago.chicago_crimes
GROUP BY ROLLUP(ward)
ORDER BY ward NULLS LAST;


select ward, count(0)
from chicago.chicago_crimes
group by rollup(ward)
-- having ward is null

select ward, count(ward)
from chicago.chicago_crimes
group by ward
having ward is null


select ward, count(ward)
from chicago.chicago_crimes
group by ward