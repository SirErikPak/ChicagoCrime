drop table if exists chicago.crime_enriched;

create table chicago.crime_enriched as
select c.case_number,
       c.date,
	   c.block,
	   c.iucr,
	   i.primary_description,
	   i.secondary_description,
	   i.index_code,
       c.primary_type,
       c.description,
	   c.location_description,
       c.arrest,
       c.domestic,
       c.beat,
       c.district,
       c.ward,
       c.community_area as community_code,
	   c.year,
	   c.updated_on,
	   c.fbi_code,
       z.zip_code,
       z.shape_area as zip_code_area,
       n.primary_neighborhood,
	   n.secondary_neighborhood,
       n.shape_area as neighborhood_area,
       p.district as p_district,
       p.sector as p_sector,
	   p.beat_number as p_beat,
	   ca.area_number as ca_community_code,
	   ca.community_name as ca_community_name,
	   ca.shape_area as ca_community_area,
	   c.latitude,
	   c.longitude,
	   x_coordinate,
	   y_coordinate
	   -- a.arrest_date,
	   -- a.race
from   chicago.crime c 
left join chicago.neighborhood n 
 on ST_Intersects(ST_FlipCoordinates(ST_SetSRID(c.location::geometry, 4326)), n.the_geom)
left join chicago.zip_code z
 on ST_Intersects(ST_FlipCoordinates(ST_SetSRID(c.location::geometry, 4326)), z.the_geom)
left join chicago.staging_police_beat p
 on ST_Intersects(ST_FlipCoordinates(ST_SetSRID(c.location::geometry, 4326)), p.the_geom)
left join chicago.iucr_code as i
 on c.iucr = i.iucr
left join chicago.community_area as ca
 on ST_Intersects(ST_FlipCoordinates(ST_SetSRID(c.location::geometry, 4326)), ca.the_geom);
 
-- left join chicago.arrests a
-- on c.case_number = a.case_number;
-- arrest table can have multiple arrest for the same case 

-- Server‑side COPY (must be run by a superuser with write access there)
COPY chicago.crime_enriched TO '/tmp/chicago_crimes_export.csv' WITH (FORMAT CSV, HEADER, DELIMITER ',');
COPY chicago.arrest TO '/tmp/chicago_arrests_export.csv' WITH (FORMAT CSV, HEADER, DELIMITER ',');

-- -- copy out
-- copy chicago.crimes_enriched TO '/Users/sir/Desktop/Project/ChicagoCrime/Data/chicago_crimes_export.csv' 
-- WITH (FORMAT CSV, HEADER, DELIMITER ',');

-- copy chicago.arrests TO '/Users/sir/Desktop/Project/ChicagoCrime/Data/chicago_arrests_export.csv' 
-- WITH (FORMAT CSV, HEADER, DELIMITER ',');

