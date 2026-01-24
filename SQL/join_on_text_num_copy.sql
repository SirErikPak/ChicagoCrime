select	distinct(primary_neighborhood)
from chicago.staging_neighborhoods
order by primary_neighborhood


select count(*)
from chicago.crimes c inner join chicago.staging_neighborhoods n
on c.community_area = REPLACE(n.shape_area, ',', '')::numeric;

