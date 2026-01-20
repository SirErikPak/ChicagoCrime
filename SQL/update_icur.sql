select *
from chicago.crimes as c inner join chicago.arrests as a
on c.case_number = a.case_number and c.year > 2024 and a.race != 'BLACK' and c.location_description = 'APARTMENT'
where c.primary_type = 'BATTERY'

select max(length(iucr))
from chicago.iucr_codes 

ALTER TABLE chicago.iucr_codes 
ALTER COLUMN iucr TYPE CHAR(4);

select *
from chicago.iucr_codes


create table chicago.iucr_hold as
select *
from chicago.iucr_codes


select 
	case
		when length(iucr) = 3 then LPAD(iucr, 4, '0')
		else iucr
	end as formated_iucr
from chicago.iucr_codes


update chicago.iucr_codes
set iucr = LPAD(iucr, 4, '0')
where length(iucr) = 3;





select *
from chicago.iucr_hold








