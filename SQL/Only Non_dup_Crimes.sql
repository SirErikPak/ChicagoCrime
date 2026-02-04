drop table if exists chicago.non_dup_crimes;

-- remove any duplicate case_number by taking the 
-- latest updated_on column
create table chicago.non_dup_crimes as
select	t.case_number,
		t.date,
		t.block,
		t.iucr,
		t.primary_type,
		t.description,
		t.location_description,
		t.arrest,
		t.domestic,
		t.beat,
		t.district,
		pb.sector,
		t.ward,
		t.community_area as community_code,
		ca.community_name,
		ca.shape_area as community_area,
		t.fbi_code,
		t.x_coordinate,
		t.y_coordinate,
		t.year,
		t.latitude,
		t.longitude
from	(
			select	*, 
				row_number() over (
					partition by case_number 
					order by updated_on desc
				) as rn
			from	chicago.crimes
		) t

left join chicago.community_areas as ca
on	t.community_area = ca.area_number
left join chicago.police_beats as pb
on t.beat = pb.beat and t.district = pb.district
where t.rn = 1;

-- copy out
copy chicago.non_dup_crimes TO '/Users/sir/Desktop/Project/ChicagoCrime/Data/chicago_crimes_non_dup_export.csv' 
WITH (FORMAT CSV, HEADER, DELIMITER ',');
