drop table if exists chicago.crimes_enriched;

create table chicago.crimes_enriched as
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
       c.community_area,
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
	   p.beat_num as p_beat,
	   c.location
	   -- a.arrest_date,
	   -- a.race
from   chicago.crimes c 
left join chicago.neighborhoods n 
on ST_Intersects(c.the_geom, n.the_geom)
left join chicago.zip_codes z
 on ST_Intersects(c.the_geom, z.the_geom)
left join chicago.staging_police_beats p
 on ST_Intersects(c.the_geom, p.the_geom)
left join chicago.iucr_codes as i
on c.iucr = i.iucr
-- left join chicago.arrests a
-- on c.case_number = a.case_number;
-- arrest table can have multiple arrest for the same case 