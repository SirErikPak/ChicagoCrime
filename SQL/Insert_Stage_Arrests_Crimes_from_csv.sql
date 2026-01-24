truncate table chicago.arrests;
COPY chicago.arrests (
    cb_no, case_number, arrest_date, race, 
    charge_1_statute, charge_1_description, charge_1_type, charge_1_class,
    charge_2_statute, charge_2_description, charge_2_type, charge_2_class,
    charge_3_statute, charge_3_description, charge_3_type, charge_3_class,
    charge_4_statute, charge_4_description, charge_4_type, charge_4_class,
    charges_statute, charges_description, charges_type, charges_class
)
FROM '/Users/sir/Desktop/Project/ChicagoCrime/Data/Arrests.csv'
WITH (FORMAT CSV, HEADER);
ANALYZE chicago.arrests;


truncate table chicago.crimes;
COPY chicago.crimes (
    id, case_number, date, block, iucr, primary_type, 
    description, location_description, arrest, domestic, 
    beat, district, ward, community_area, fbi_code, 
    x_coordinate, y_coordinate, year, updated_on, 
    latitude, longitude, location
)
FROM '/Users/sir/Desktop/Project/ChicagoCrime/Data/Crime_2001_to_Present.csv'
WITH (FORMAT CSV, HEADER);
ANALYZE chicago.crimes;


truncate table chicago.iucr_codes;
COPY chicago.iucr_codes (
    iucr, 
    primary_description, 
    secondary_description, 
    index_code, 
    active
)
FROM '/Users/sir/Desktop/Project/ChicagoCrime/Data/IUCR.csv'
WITH (FORMAT CSV, HEADER);
ANALYZE chicago.iucr_codes;

-- IUCR code update
update chicago.iucr_codes
set iucr = LPAD(iucr, 4, '0')
where length(iucr) = 3;

select *
from  chicago.iucr_codes
limit 10


truncate table chicago.staging_police_beats;
COPY chicago.staging_police_beats 
FROM '/Users/sir/Desktop/Project/ChicagoCrime/Data/Police_Beat.csv'
WITH (FORMAT CSV, HEADER);

truncate table chicago.staging_community_areas;
COPY chicago.staging_community_areas
FROM '/Users/sir/Desktop/Project/ChicagoCrime/Data/Community_Areas.csv'
WITH (FORMAT CSV, HEADER);

truncate table chicago.staging_neighborhoods 
COPY chicago.staging_neighborhoods 
FROM '/Users/sir/Desktop/Project/ChicagoCrime/Data/Neighborhoods.csv'
WITH (FORMAT CSV, HEADER);

truncate table chicago.staging_zip_codes;
copy chicago.staging_zip_codes
from '/Users/sir/Desktop/Project/ChicagoCrime/Data/ZIP_Codes.csv'
WITH (FORMAT CSV, HEADER);


