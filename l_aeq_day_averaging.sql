CREATE TABLE nord_2023_04_17_to_23 AS
SELECT AVG(l_aeq_hour) AS l_aeq_day, id, tmsnumber, geometry, in_building
FROM (
	SELECT l_aeq_hour, id, tmsnumber, geometry, in_building
    FROM nord_2023_04_17_hours
    WHERE l_aeq_hour IS NOT NULL
    
    UNION ALL
	
    SELECT l_aeq_hour, id, tmsnumber, geometry, in_building
    FROM nord_2023_04_18_hours
    WHERE l_aeq_hour IS NOT NULL
    
    UNION ALL
    
    SELECT l_aeq_hour, id, tmsnumber, geometry, in_building
    FROM nord_2023_04_19_hours
    WHERE l_aeq_hour IS NOT NULL
    
    UNION ALL
    
	SELECT l_aeq_hour, id, tmsnumber, geometry, in_building
    FROM nord_2023_04_20_hours
    WHERE l_aeq_hour IS NOT NULL
    
    UNION ALL
	
	SELECT l_aeq_hour, id, tmsnumber, geometry, in_building
    FROM nord_2023_04_21_hours
    WHERE l_aeq_hour IS NOT NULL
    
    UNION ALL
	
	SELECT l_aeq_hour, id, tmsnumber, geometry, in_building
    FROM nord_2023_04_22_hours
    WHERE l_aeq_hour IS NOT NULL
    
    UNION ALL
	
	SELECT l_aeq_hour, id, tmsnumber, geometry, in_building
    FROM nord_2023_04_23_hours
    WHERE l_aeq_hour IS NOT NULL    
) AS subquery
WHERE in_building = FALSE
GROUP BY id, tmsnumber, geometry, in_building;
