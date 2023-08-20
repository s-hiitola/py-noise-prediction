CREATE TABLE nord_L1 AS
SELECT DISTINCT tmsnumber, date,
	CASE WHEN count_light > 0 THEN
		CASE WHEN speed_light >= 40 THEN (73.5 + 25 * log(speed_light/50)) + 10 * log(count_light/3600)
			ELSE 71.1 + 10 * log(count_light/3600) END
		ELSE 0 END AS L_Aeq10m_light,
	CASE WHEN count_heavy > 0 THEN
		CASE WHEN speed_heavy >= 50 THEN (80.5 + 30 * log(speed_heavy/50)) + 10 * log(count_heavy/3600)
			ELSE 80.5 + 10 * log(count_heavy/3600) END
		ELSE 0 END AS L_Aeq10m_heavy,
	CASE
		WHEN count_light > 0 AND count_heavy > 0 THEN
			10 * log(POWER(10, CASE WHEN speed_light >= 40 THEN
					(73.5 + 25 * log(speed_light/50)) + 10 * log(count_light/3600)
				ELSE 71.1 + 10 * log(count_light/3600)
				END/ 10) + 
				POWER(10, CASE WHEN speed_heavy >= 50 THEN
					(80.5 + 30 * log(speed_heavy/50)) + 10 * log(count_heavy/3600)
				ELSE 80.5 + 10 * log(count_heavy/3600)
				END / 10))
		WHEN count_light > 0 THEN
			CASE WHEN speed_light >= 40 THEN
				(73.5 + 25 * log(speed_light/50)) + 10 * log(count_light/3600)
				ELSE 71.1 + 10 * log(count_light/3600)
				END
		ELSE
			CASE WHEN speed_heavy >= 50 THEN
				(80.5 + 30 * log(speed_heavy/50)) + 10 * log(count_heavy/3600)
				ELSE 80.5 + 10 * log(count_heavy/3600)
				END
	END AS L_Aeq10m_mixed
	
FROM (
	SELECT tmsnumber, date,
		count_light,
		count_heavy,
		speed_light,
		speed_heavy,
		ROW_NUMBER() OVER (PARTITION BY tmsnumber, date) AS row_num
	FROM traffic
) AS subquery
WHERE row_num = 1;
