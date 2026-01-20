INSERT INTO chicago.neighborhoods (
    the_geom, 
    primary_neighborhood, 
    secondary_neighborhood, 
    shape_area, 
    shape_len
)
SELECT 
    the_geom,
    primary_neighborhood, 
    secondary_neighborhood, 
    REPLACE(shape_area, ',', '')::double precision,
    REPLACE(shape_len, ',', '')::double precision
FROM chicago.staging_neighborhoods;


INSERT INTO chicago.zip_codes (
	the_geom,
	objectid,
	zip_code,
	shape_area,
	shape_len)
SELECT
	the_geom,
	objectid,
	zip_code,
    REPLACE(shape_area, ',', '')::double precision,
    REPLACE(shape_len, ',', '')::double precision
FROM chicago.staging_zip_codes



select beat_num, count(*)
from chicago.staging_police_beats
group by beat_num
having count(*) > 1

select row_number() over(),
		*
from chicago.staging_police_beats
where beat_num = 3100


select the_geom, count(*)
from chicago.staging_police_beats
group by the_geom
having count(*) > 1;


select n.primary_neighborhood, n.the_geom, b.*
from chicago.neighborhoods n right join chicago.staging_police_beats b
on n.the_geom = b.the_geom
where b.beat_num = 3100


select 
    n.primary_neighborhood, 
    b.beat_num, 
    b.the_geom,
    ROW_NUMBER() OVER(PARTITION BY b.beat_num ORDER BY n.primary_neighborhood) as row_num
from chicago.neighborhoods n 
inner join chicago.staging_police_beats b 
    on ST_Intersects(n.the_geom, b.the_geom)
where b.beat_num = 3100;


select 
    beat_num, 
    ST_Area(the_geom) as area_size,
    ST_AsText(ST_Centroid(the_geom)) as center_point,
    ROW_NUMBER() OVER(PARTITION BY beat_num ORDER BY ST_Area(the_geom) DESC) as rank
from chicago.staging_police_beats
where beat_num = 3100;



WITH ranked_beats AS (
    select 
        beat_num, 
        the_geom,
        ROW_NUMBER() OVER(PARTITION BY beat_num ORDER BY ST_Area(the_geom) DESC) as rank
    from chicago.staging_police_beats
    where beat_num = 3100
)
select 
    a.beat_num,
    a.rank as main_rank,
    b.rank as ghost_rank,
    -- Calculate distance in meters, then convert to miles
    ST_Distance(a.the_geom::geography, b.the_geom::geography) / 1609.34 as distance_miles
from ranked_beats a
join ranked_beats b on a.beat_num = b.beat_num
where a.rank = 1 and b.rank > 1;



select 
    n.primary_neighborhood, 
    b.beat_num
from chicago.neighborhoods n
inner join (
    select beat_num, ST_Union(the_geom) as merged_geom
    from chicago.staging_police_beats
    group by beat_num
) b on ST_Intersects(n.the_geom, b.merged_geom)
where b.beat_num = '3100';

	