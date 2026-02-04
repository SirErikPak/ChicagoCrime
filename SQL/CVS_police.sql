COPY (
    SELECT district, sector, beat
    FROM chicago.police_beats
    GROUP BY district, sector, beat
    ORDER BY beat
) 
TO '/Users/sir/Desktop/Project/ChicagoCrime/Data/police_beats_export.csv' 
WITH (FORMAT CSV, HEADER);
