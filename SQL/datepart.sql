select distinct DATE_PART('year', arrest_date)
from chicago.arrests
order by DATE_PART('year', arrest_date) desc