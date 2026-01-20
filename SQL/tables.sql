CREATE EXTENSION IF NOT EXISTS postgis;

-- 1. Reference Table (Load this first)
CREATE TABLE chicago.iucr_codes (
    iucr VARCHAR(10) PRIMARY KEY, -- Changed to VARCHAR for leading zeros
    primary_description TEXT,
    secondary_description TEXT,
    index_code CHAR(1),
    active BOOLEAN
);

-- 2. Main Crime Table
CREATE TABLE chicago.crimes (
    id SERIAL PRIMARY KEY,
    case_number VARCHAR(20),
    date TIMESTAMP,
    block VARCHAR(100),
    iucr VARCHAR(10), -- Match the VARCHAR in iucr_codes
    primary_type VARCHAR(100),
    description TEXT,
    location_description TEXT,
    arrest BOOLEAN,
    domestic BOOLEAN,
    beat INTEGER,
    district NUMERIC,
    ward NUMERIC,
    community_area NUMERIC,
    fbi_code VARCHAR(10),
    x_coordinate NUMERIC,
    y_coordinate NUMERIC,
    year INTEGER,
    updated_on TIMESTAMP,
    latitude NUMERIC(15, 12),
    longitude NUMERIC(15, 12),
    location POINT
);

-- 3. Arrests Table
CREATE TABLE chicago.arrests (
    cb_no INTEGER PRIMARY KEY,
    case_number VARCHAR(20),
    arrest_date TIMESTAMP,
    race VARCHAR(50),
    charge_1_statute VARCHAR(100),
    charge_1_description TEXT,
    charge_1_type VARCHAR(10),
    charge_1_class VARCHAR(10),
    charge_2_statute VARCHAR(100),
    charge_2_description TEXT,
    charge_2_type VARCHAR(10),
    charge_2_class VARCHAR(10),
    charge_3_statute VARCHAR(100),
    charge_3_description TEXT,
    charge_3_type VARCHAR(10),
    charge_3_class VARCHAR(10),
    charge_4_statute VARCHAR(100),
    charge_4_description TEXT,
    charge_4_type VARCHAR(10),
    charge_4_class VARCHAR(10),
    charges_statute TEXT,
    charges_description TEXT,
    charges_type TEXT,
    charges_class TEXT
);

-- 4. Boundaries Table
CREATE TABLE chicago.staging_zip_codes (
    the_geom GEOMETRY(MultiPolygon, 4326),
    objectid INTEGER PRIMARY KEY,
    zip_code Varchar(15),
    shape_area TEXT,
    shape_len TEXT
);


CREATE TABLE chicago.zip_codes (
    the_geom GEOMETRY(MultiPolygon, 4326),
    objectid INTEGER PRIMARY KEY,
    zip_code Varchar(15),
    shape_area DOUBLE PRECISION,
    shape_len DOUBLE PRECISION
);

-- 5. Neighborhoods
CREATE TABLE chicago.staging_neighborhoods (
    the_geom GEOMETRY(MultiPolygon, 4326),
    primary_neighborhood TEXT,
    secondary_neighborhood TEXT,
    shape_area TEXT,
    shape_len TEXT
);


CREATE TABLE chicago.neighborhoods (
    the_geom GEOMETRY(MultiPolygon, 4326),
    primary_neighborhood TEXT,
    secondary_neighborhood TEXT,
    shape_area DOUBLE PRECISION,
    shape_len DOUBLE PRECISION
);

-- 6. Police Beats
CREATE TABLE chicago.staging_police_beats (
    the_geom GEOMETRY(MultiPolygon, 4326) NOT NULL,
    district SMALLINT NOT NULL, -- Chicago has districts 1-25
    sector SMALLINT NOT NULL,   -- Typically 1-3 digits
    beat SMALLINT NOT NULL,     -- Local beat identifier
    beat_num INTEGER  -- Unique identifier (e.g., 2432)
);

CREATE TABLE chicago.police_beats (
    the_geom GEOMETRY(MultiPolygon, 4326) NOT NULL,
    district SMALLINT NOT NULL, -- Chicago has districts 1-25
    sector SMALLINT NOT NULL,   -- Typically 1-3 digits
    beat SMALLINT NOT NULL,     -- Local beat identifier
    beat_num INTEGER PRIMARY KEY -- Unique identifier (e.g., 2432)
);