-- Write your SQL query here
select category,
    Count(*) as total_sales,
    SUM(amount) as total_revenue,
    ROUND(AVG(discount)::numeric, 2) as avg_discount

from SALES
group by category
order by total_revenue DESC, category ASC;


