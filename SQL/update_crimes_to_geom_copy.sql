select count(*)
from chicago.crimes

select	*
from	chicago.crimes
limit 	5

ALTER TABLE chicago.crimes
ADD COLUMN the_geom geometry(Point, 4326);

-- ST_X(points_col) → extracts longitude/x coordinate
-- ST_Y(points_col) → extracts latitude/y coordinate
-- ST_MakePoint(x, y) → creates new point geometry
-- ST_SetSRID(..., 4326) → sets spatial reference to WGS8
-- ST_X(points)    → extracts x-coordinate (longitude)
-- ST_Y(points)    → extracts y-coordinate (latitude) 
-- ST_MakePoint()  → creates geometry Point
-- ST_SetSRID()    → adds SRID 4326 (WGS84)

UPDATE chicago.crimes
SET the_geom = ST_SetSRID(ST_MakePoint(longitude::double precision, latitude::double precision), 4326);