select	extract(year from date)::int as year,
		primary_type,
		count(*) as crime_count
from	chicago.crimes
group by extract(year from date)::int, primary_type
		
		