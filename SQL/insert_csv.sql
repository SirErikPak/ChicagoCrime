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


COPY chicago.chicago_crimes (
    id, case_number, date, block, iucr, primary_type, 
    description, location_description, arrest, domestic, 
    beat, district, ward, community_area, fbi_code, 
    x_coordinate, y_coordinate, year, updated_on, 
    latitude, longitude, location
)
FROM '/Users/sir/Desktop/Project/ChicagoCrime/Data/Crime_2001_to_Present.csv'
WITH (FORMAT CSV, HEADER);


COPY chicago.iucr_codes (
    iucr, 
    primary_description, 
    secondary_description, 
    index_code, 
    active
)
FROM '/Users/sir/Desktop/Project/ChicagoCrime/Data/IUCR.csv'
WITH (FORMAT CSV, HEADER);


COPY chicago.staging_police_beats 
FROM '/Users/sir/Desktop/Project/ChicagoCrime/Data/Police_Beat.csv'
WITH (FORMAT CSV, HEADER);


COPY chicago.staging_neighborhoods 
FROM '/Users/sir/Desktop/Project/ChicagoCrime/Data/Neighborhoods.csv'
WITH (FORMAT CSV, HEADER);

copy chicago.staging_zip_codes
from '/Users/sir/Desktop/Project/ChicagoCrime/Data/Boundaries_ZIP_Codes.csv'
WITH (FORMAT CSV, HEADER);
