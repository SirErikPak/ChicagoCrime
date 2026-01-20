copy chicago.crimes_enriched TO '/Users/sir/Desktop/Project/ChicagoCrime/Data/chicago_crimes_export.csv' 
WITH (FORMAT CSV, HEADER, DELIMITER ',');

copy chicago.arrests TO '/Users/sir/Desktop/Project/ChicagoCrime/Data/chicago_arrests_export.csv' 
WITH (FORMAT CSV, HEADER, DELIMITER ',');