select	case_number
from	chicago.crimes
group by case_number
having count(case_number) > 1



select	*
from	chicago.crimes
where	case_number in (
	select	distinct(case_number)
	from	chicago.crimes
	group by case_number
	having count(*)>2
)

select	*
from	(
			select	*, 
				row_number() over (
					partition by case_number 
					order by date desc
				) as rn
			from	chicago.crimes
		) t
where rn>2



select 	*
from	chicago.crimes
where	case_number in('G023235', 'G137655')