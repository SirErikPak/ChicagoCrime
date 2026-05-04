-- =====================================================
-- Chicago Crime Staging Import
-- =====================================================

-- 1. Create table
CREATE TABLE IF NOT EXISTS chicago.staging_crime (
    id integer NOT NULL,
    case_number character varying(20),
    date timestamp without time zone,
    block character varying(100),
    iucr character varying(10),
    primary_type character varying(100),
    description text,
    location_description text,
    arrest boolean,
    domestic boolean,
    beat integer,
    district numeric,
    ward numeric,
    community_area numeric,
    fbi_code character varying(10),
    x_coordinate numeric,
    y_coordinate numeric,
    year integer,
    updated_on timestamp without time zone,
    latitude numeric(15,12),
    longitude numeric(15,12),
    location point
);

-- 2. Clear existing data
TRUNCATE TABLE chicago.staging_crime;

-- 3. Import csv data
COPY chicago.staging_crime FROM '/tmp/Crime_2001_to_Present.csv'
WITH (FORMAT CSV, HEADER, DELIMITER ',');


-- 4. Verify import
SELECT 
    count(*) as total_rows,
    min(date) as date_range_start,
    max(date) as date_range_end,
    count(distinct case_number) as unique_cases
FROM chicago.staging_crime;

-- =====================================================
-- Chicago Arreste Staging Import
-- =====================================================
-- 1. Create table
CREATE TABLE IF NOT EXISTS chicago.staging_arrest
(
    cb_no integer NOT NULL,
    case_number character varying(20),
    arrest_date timestamp without time zone,
    race character varying(50),
    charge_1_statute character varying(100),
    charge_1_description text,
    charge_1_type character varying(10),
    charge_1_class character varying(10),
    charge_2_statute character varying(100),
    charge_2_description text,
    charge_2_type character varying(10),
    charge_2_class character varying(10),
    charge_3_statute character varying(100),
    charge_3_description text,
    charge_3_type character varying(10),
    charge_3_class character varying(10),
    charge_4_statute character varying(100),
    charge_4_description text,
    charge_4_type character varying(10),
    charge_4_class character varying(10),
    charges_statute text,
    charges_description text,
    charges_type text,
    charges_class text
);

-- 2. Clear existing data
TRUNCATE chicago.staging_arrest;

-- 3. Import csv data
COPY chicago.staging_arrest FROM '/tmp/arrests.csv' 
WITH (FORMAT CSV, HEADER true, DELIMITER ',', NULL '', QUOTE '"');

-- 4. Verify import
SELECT 
    count(*) as total_rows,
    min(arrest_date) as date_range_start,
    max(arrest_date) as date_range_end,
    count(distinct case_number) as unique_cases
FROM chicago.staging_arrest;

-- =====================================================
-- Chicago IUCR: Staging Import
-- =====================================================
-- 1. Create table
CREATE TABLE IF NOT EXISTS chicago.staging_iucr_code (
    iucr char(4) NOT NULL,
    primary_description text,
    secondary_description text,
    index_code char(1),
    active boolean
);

-- 2. Clear existing data
TRUNCATE chicago.staging_iucr_code RESTART IDENTITY;

-- 3. Import csv data
COPY chicago.staging_iucr_code FROM '/tmp/iucr_codes.csv' 
WITH (FORMAT CSV, HEADER true, DELIMITER ',', NULL '', QUOTE '"');

-- 4. Verify import
SELECT 
    count(*) as total_rows,
    count(distinct iucr) as unique_iucr
FROM chicago.staging_iucr_code;

-- =====================================================
-- Chicago ZIP: Staging Import
-- =====================================================
-- 1. Create table
CREATE TABLE IF NOT EXISTS chicago.staging_zip_code
(
    the_geom geometry(MultiPolygon,4326),
    objectid integer NOT NULL,
    zip_code character varying(15),
    shape_area text,
    shape_len text
);

-- 2. Clear existing data
TRUNCATE chicago.staging_zip_code;

-- 3. Import csv data
COPY chicago.staging_zip_code FROM '/tmp/ZIP_Codes.csv' 
WITH (FORMAT CSV, HEADER true, DELIMITER ',', NULL '', QUOTE '"');

-- 4. Verify import
SELECT 
    count(*) as total_rows,
    count(distinct zip_code) as zip_code
FROM chicago.staging_zip_code;


-- =====================================================
-- Chicago Community Area: Staging Import
-- =====================================================
-- 1. Create table
CREATE TABLE IF NOT EXISTS chicago.staging_community_area
(
    the_geom geometry(MultiPolygon,4326) NOT NULL,
    area_number smallint,
    community text,
	area_number1 int,
    shape_area text,
    shape_len text
);

-- 2. Clear existing data
TRUNCATE chicago.staging_community_area;

-- 3. Import csv data
COPY chicago.staging_community_area FROM '/tmp/Community_Areas.csv' 
WITH (FORMAT CSV, HEADER true, DELIMITER ',', NULL '', QUOTE '"');

-- 4. Verify import
SELECT 
    count(*) as total_rows,
    count(distinct community) as community
FROM chicago.staging_community_area;

-- =====================================================
-- Chicago Neighborhood: Staging Import
-- =====================================================
-- 1. Create table
CREATE TABLE IF NOT EXISTS chicago.staging_neighborhood
(
    the_geom geometry(MultiPolygon,4326),
    primary_neighborhood text,
    secondary_neighborhood text,
    shape_area text,
    shape_len text
);

-- 2. Clear existing data
TRUNCATE chicago.staging_neighborhood;

-- 3. Import csv data
COPY chicago.staging_neighborhood FROM '/tmp/Neighborhoods.csv' 
WITH (FORMAT CSV, HEADER true, DELIMITER ',', NULL '', QUOTE '"');

-- 4. Verify import
SELECT 
    count(*) as total_rows,
    count(distinct primary_neighborhood) as primary_neighborhood,
	count(distinct secondary_neighborhood) as secondary_neighborhood
FROM chicago.staging_neighborhood;

-- =====================================================
-- Chicago Neighborhood: Staging Import
-- =====================================================
-- 1. Create table
CREATE TABLE IF NOT EXISTS chicago.staging_neighborhood
(
    the_geom geometry(MultiPolygon,4326),
    primary_neighborhood text,
    secondary_neighborhood text,
    shape_area text,
    shape_len text
);

-- 2. Clear existing data
TRUNCATE chicago.staging_neighborhood;

-- 3. Import csv data
COPY chicago.staging_neighborhood FROM '/tmp/Neighborhoods.csv' 
WITH (FORMAT CSV, HEADER true, DELIMITER ',', NULL '', QUOTE '"');

-- 4. Verify import
SELECT 
    count(*) as total_rows,
    count(distinct primary_neighborhood) as primary_neighborhood,
	count(distinct secondary_neighborhood) as secondary_neighborhood
FROM chicago.staging_neighborhood;

-- =====================================================
-- Chicago Neighborhood: Staging Import
-- =====================================================
-- 1. Create table
CREATE TABLE IF NOT EXISTS chicago.staging_neighborhood
(
    the_geom geometry(MultiPolygon,4326),
    primary_neighborhood text,
    secondary_neighborhood text,
    shape_area text,
    shape_len text
);

-- 2. Clear existing data
TRUNCATE chicago.staging_neighborhood;

-- 3. Import csv data
COPY chicago.staging_neighborhood FROM '/tmp/Neighborhoods.csv' 
WITH (FORMAT CSV, HEADER true, DELIMITER ',', NULL '', QUOTE '"');

-- 4. Verify import
SELECT 
    count(*) as total_rows,
    count(distinct primary_neighborhood) as primary_neighborhood,
	count(distinct secondary_neighborhood) as secondary_neighborhood
FROM chicago.staging_neighborhood;

-- =====================================================
-- Chicago Police Beat: Staging Import
-- =====================================================
-- 1. Create table
CREATE TABLE IF NOT EXISTS chicago.staging_police_beat
(
    the_geom geometry(MultiPolygon,4326) NOT NULL,
    district smallint NOT NULL,
    sector smallint NOT NULL,
    beat smallint NOT NULL,
	beat_number smallint NOT NULL
);

-- 2. Clear existing data
TRUNCATE chicago.staging_police_beat;

-- 3. Import csv data
COPY chicago.staging_police_beat FROM '/tmp/Police_Beat.csv' 
WITH (FORMAT CSV, HEADER true, DELIMITER ',', NULL '', QUOTE '"');

-- 4. Verify import
SELECT 
    count(*) as total_rows,
    count(distinct district) as district,
	count(distinct sector) as sector,
	count(distinct beat) as beat,
	count(distinct beat_number) as beat_number
FROM chicago.staging_police_beat;