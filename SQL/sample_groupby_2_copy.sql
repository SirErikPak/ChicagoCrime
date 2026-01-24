select setseed(0.5);

select *
from chicago.chicago_crimes
order by random()
limit 20


select district, year, count(*)
from chicago.chicago_crimes
group by district, year

select year, count(1), count(*)
from chicago.chicago_crimes
group by year


select ward, count(ward)
from chicago.chicago_crimes
group by rollup(ward)

select ward, count(1)
from chicago.chicago_crimes
group by rollup(ward)

select count(0)
from chicago.chicago_crimes



SELECT 
  COALESCE(ward::text, 'TOTAL') AS ward_display,
  COUNT(1) AS non_null_count
FROM chicago.chicago_crimes
GROUP BY ROLLUP(ward)
ORDER BY 
  CASE WHEN ward IS NULL THEN 999 END NULLS LAST, 
  ward;