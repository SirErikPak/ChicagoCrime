-- =====================================================
-- Chicago Arrests: Staging -> Production Transfer
-- =====================================================

BEGIN;

-- 1. Clear production table
TRUNCATE TABLE chicago.arrest 
RESTART IDENTITY 
CASCADE;

-- 2. Bulk transfer from staging  
INSERT INTO chicago.arrest
SELECT * FROM chicago.staging_arrest;

COMMIT;

-- 3. Update statistics
ANALYZE chicago.arrest;

-- 4. Verify transfer
SELECT 
    'staging_arrest' as source, count(*)::bigint as rows 
FROM chicago.staging_arrest
UNION ALL
SELECT 'arrest', count(*)::bigint as rows 
FROM chicago.arrest;

-- =====================================================
-- Chicago Community Area: Staging -> Production Transfer
-- =====================================================

BEGIN;

-- 1. Clear production table
TRUNCATE TABLE chicago.community_area;

-- 2. Bulk transfer from staging  
INSERT INTO chicago.community_area
select 	the_geom, 
		area_number, 
		community, 
		replace(shape_area, ',', '')::double precision,
   		replace(shape_len, ',', '')::double precision
from chicago.staging_community_area;

COMMIT;

-- 3. Update statistics
ANALYZE  chicago.community_area;

-- 4. Verify transfer
SELECT 
    'staging_community_area' as source, count(*)::bigint as rows 
FROM chicago.staging_community_area
UNION ALL
SELECT 'community_area', count(*)::bigint as rows 
FROM chicago.community_area;


-- =====================================================
-- Staging -> Production Crime Table
-- =====================================================

-- 1. Atomic truncate + insert (single transaction)
BEGIN;

-- Clear production table
TRUNCATE TABLE chicago.crime 
RESTART IDENTITY 
CASCADE;

-- Bulk transfer from staging
INSERT INTO chicago.crime
SELECT * FROM chicago.staging_crime;

COMMIT;

-- 2. Update statistics for query optimizer
ANALYZE chicago.crime;

-- 3. Verify row counts match
SELECT 
    'staging' as source, count(*) as rows FROM chicago.staging_crime
UNION ALL
    SELECT 'production' as source, count(*) as rows FROM chicago.crime;


-- =====================================================
-- Staging -> Production IUCR Table
-- =====================================================

-- 1. Atomic truncate + insert (single transaction)
BEGIN;

-- Clear production table
TRUNCATE TABLE chicago.iucr_code;

-- Bulk transfer from staging
INSERT INTO chicago.iucr_code
SELECT 
	-- Pad IUCR codes to exactly 4 characters with leading zeros
    LPAD(iucr::text, 4, '0') as iucr_padded,
    primary_description,
    secondary_description,
	index_code,
	active
FROM chicago.staging_iucr_code;

COMMIT;

-- 2. Update statistics for query optimizer
ANALYZE chicago.iucr_code;

-- 3. Verify row counts match
SELECT 
    'staging' as source, count(*) as rows FROM chicago.staging_iucr_code
UNION ALL
    SELECT 'production' as source, count(*) as rows FROM chicago.iucr_code;


-- =====================================================
-- Staging -> Production Neighborhood Table
-- =====================================================

-- 1. Atomic truncate + insert (single transaction)
BEGIN;

-- Clear production table
TRUNCATE TABLE chicago.neighborhood;

-- Bulk transfer from staging
INSERT INTO chicago.neighborhood
SELECT 
    the_geom,
    primary_neighborhood, 
    secondary_neighborhood, 
    replace(shape_area, ',', '')::double precision,
    replace(shape_len, ',', '')::double precision
FROM chicago.staging_neighborhood;

COMMIT;

-- 2. Update statistics for query optimizer
ANALYZE chicago.neighborhood;

-- 3. Verify row counts match
SELECT 
    'staging' as source, count(*) as rows FROM chicago.staging_neighborhood
UNION ALL
    SELECT 'production' as source, count(*) as rows FROM chicago.neighborhood;


-- =====================================================
-- Staging -> Production Police Beat Table
-- =====================================================

-- 1. Atomic truncate + insert (single transaction)
BEGIN;

-- Clear production table
TRUNCATE TABLE chicago.police_beat;

-- Bulk transfer from staging
INSERT INTO chicago.police_beat
SELECT the_geom, district, sector, beat, beat_number
FROM chicago.staging_police_beat

COMMIT;

-- 2. Update statistics for query optimizer
ANALYZE  chicago.police_beat;

-- 3. Verify row counts match
SELECT 
    'staging' as source, count(*) as rows FROM chicago.staging_police_beat
UNION ALL
    SELECT 'production' as source, count(*) as rows FROM chicago.police_beat;


-- -- finds exact duplicate combos
-- SELECT district, sector, beat, beat_number, count(*)
-- FROM chicago.staging_police_beat
-- GROUP BY district, sector, beat, beat_number
-- HAVING COUNT(*) > 1;

-- -- DELETE duplicates from staging_police_beat (keep first row per district+beat)
-- with ranked_rows as (
--     select ctid, *,
--            row_number() over (partition by district, sector, beat, beat_number ORDER BY ctid) as rn
--     FROM chicago.staging_police_beat
-- )

-- -- select *
-- -- from ranked_rows
-- -- where rn> 1

-- delete from chicago.staging_police_beat 
-- using ranked_rows
-- where chicago.staging_police_beat.ctid = ranked_rows.ctid 
-- and ranked_rows.rn > 1;


-- =====================================================
-- Staging -> Production Zip COde Table
-- =====================================================

-- 1. Atomic truncate + insert (single transaction)
BEGIN;

-- Clear production table
TRUNCATE TABLE chicago.zip_code;

-- Bulk transfer from staging
INSERT INTO chicago.zip_code
SELECT 
    the_geom,
    objectid, 
    zip_code, 
    replace(shape_area, ',', '')::double precision,
    replace(shape_len, ',', '')::double precision
FROM chicago.staging_zip_code;

COMMIT;

-- 2. Update statistics for query optimizer
ANALYZE chicago.zip_code;

-- 3. Verify row counts match
SELECT 
    'staging' as source, count(*) as rows FROM chicago.staging_zip_code
UNION ALL
    SELECT 'production' as source, count(*) as rows FROM chicago.zip_code;

