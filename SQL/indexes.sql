-- CREATE INDEX cidx_crimes_id ON chicago.chicago_crimes (id);
-- CLUSTER chicago.crimes USING cidx_chicago_crimes_id;

-- Indexing for spatial performance
ALTER TABLE chicago.crimes
ADD COLUMN the_geom geometry(Point, 4326);

UPDATE chicago.crimes
SET the_geom = ST_SetSRID(ST_MakePoint(longitude::double precision, latitude::double precision), 4326);
CREATE INDEX idx_chicago_crimes_geom ON chicago.chicago_crimes USING GIST (the_geom);
CREATE INDEX idx_staging_police_beats_geom ON chicago.staging_police_beats USING GIST (the_geom);

CREATE INDEX idx_police_beats_geom ON chicago.police_beats USING GIST (the_geom);
CREATE INDEX idx_zip_codes_geom ON chicago.zip_codes USING GIST (the_geom);
CREATE INDEX idx_neighborhoods_geom ON chicago.neighborhoods USING GIST (the_geom);
CREATE INDEX idx_police_beats_geom ON chicago.police_beats USING GIST (the_geom);
