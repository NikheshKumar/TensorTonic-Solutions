-- Write your SQL query here

select emp.name, emp.salary, de.dept_name

from departments de
inner join employees emp on de.id = emp.dept_id
order by emp.name ASC
