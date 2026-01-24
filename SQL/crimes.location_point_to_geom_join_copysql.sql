SELECT ST_AsText(location::geometry) 
FROM chicago.crimes 
LIMIT 5;


select distinct(sector)
from chicago.police_beats


select * from chicago.crimes 
LIMIT 5;

SELECT c.case_number, n.primary_neighborhood
FROM chicago.crimes c
JOIN chicago.neighborhoods n 
  ON ST_Intersects(
    ST_FlipCoordinates(ST_SetSRID(c.location::geometry, 4326)), 
    n.the_geom
  )

 LIMIT 10;