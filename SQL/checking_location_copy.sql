select count(*)
from chicago.crimes
where location is null
--8375747

select count(*)
from chicago.crimes
where latitude is  null


select location
from chicago.crimes
TABLESAMPLE BERNOULLI (50)


select location
from chicago.crimes
where location is null