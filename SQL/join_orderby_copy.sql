select	*
from	chicago.chicago_crimes
limit 10

select 	*
from	chicago.iucr_codes


select	primary_type, description, count(description)
from	chicago.chicago_crimes
group by primary_type, description
order by 1

-- select count(*)
select c.iucr as crime_data, i.iucr as codes_data, i.active, i.primary_description, i.secondary_description, c.primary_type, c.description
from  chicago.chicago_crimes as c right join chicago.iucr_codes as i
on i.iucr = c.iucr
where c.iucr is null


select	*
from	chicago.neighborhoods



select *
from chicago.chicago_crimes 
where iucr = '4420'

select *
from chicago.iucr_codes 
where iucr = '0462'


select	distinct primary_type, description
from chicago.chicago_crimes 
where primary_type like '%HOMICIDE%'

select	distinct primary_type, description
from chicago.crimes 
where primary_type like '%STA%'


select *
from chicago.iucr_codes
where active;


select *
from chicago.iucr_codes
where not active;


