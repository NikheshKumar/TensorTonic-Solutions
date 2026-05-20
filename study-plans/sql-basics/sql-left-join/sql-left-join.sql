-- Write your SQL query here
select c.name, c.city, COALESCE(SUM(od.amount),0) as total_spent
from customers c 
left join orders od on c.id = od.customer_id

group by c.name, c.city 
order by total_spent DESC, c.name ASC