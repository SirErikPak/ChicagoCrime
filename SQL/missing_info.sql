select 	(select count(*) from chicago.crimes) as full_data,
		(select count(*)from chicago.crimes_enriched) as enriched_data,
		((select count(*) from chicago.crimes) - (select count(*)from chicago.crimes_enriched)) as missing_from_crimes,
		((SELECT COUNT(*) FROM chicago.crimes) - (SELECT COUNT(*) FROM chicago.crimes_enriched)) * 100.0 
       / (SELECT COUNT(*) FROM chicago.crimes) AS percentage_missing;