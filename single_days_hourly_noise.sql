-- Replace date values with correct dates

CREATE TABLE nord_2023_04_23_hours
AS
SELECT
    points.id,
    points.tmsnumber,
    CASE WHEN in_building = FALSE THEN l_all.delta_l_total ELSE 0 END AS delta_l_total,
	CASE WHEN in_building = FALSE THEN l_all.l_aeq_hour ELSE 29 END as l_aeq_hour,
    points.geometry,
	points.in_building,
    l_all.date
FROM
    points
	JOIN
		(SELECT 
			nord_deltas.point_id,
			roads.tmsnumber,
		 	nord_deltas.road_id,
			10 * LOG10(SUM(POWER(10, ((nord_deltas.delta_l_r + nord_deltas.delta_l_ms + nord_deltas.delta_l_av + nord_deltas.delta_l_alpha + (((2 * gradient * 1000) / 100) + ((3 * gradient * 1000) / 100) * LOG(1 + ((count_heavy / (count_heavy + count_light)) * 100)))) / 10)))) AS delta_l_total,
			10 * LOG10(SUM(POWER(10, ((nord_deltas.delta_l_r + nord_deltas.delta_l_ms + nord_deltas.delta_l_av + nord_deltas.delta_l_alpha + nord_l1.l_aeq10m_mixed + (((2 * gradient * 1000) / 100) + ((3 * gradient * 1000) / 100) * LOG(1 + ((count_heavy / (count_heavy + count_light)) * 100)))) / 10)))) AS l_aeq_hour,
			nord_l1.date
		FROM 
			nord_deltas
		JOIN 
			roads ON roads.id = nord_deltas.road_id and roads.sub_id = nord_deltas.road_sub
			JOIN nord_l1 ON nord_l1.tmsnumber = roads.tmsnumber
		 		JOIN traffic ON traffic.tmsnumber = nord_l1.tmsnumber and traffic.date = nord_l1.date
		WHERE
			nord_l1.date >= '2023-04-23 07:00:00' AND nord_l1.date < '2023-04-23 22:00:00'
		GROUP BY 
			nord_deltas.point_id, roads.tmsnumber, nord_l1.date, nord_deltas.road_id
		) AS l_all ON points.id = l_all.point_id

GROUP BY
    points.id, points.tmsnumber, points.in_building, points.geometry, l_all.date, l_all.delta_l_total, l_all.l_aeq_hour
	
ORDER BY id;
