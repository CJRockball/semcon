-- extract_wafers.sql
-- Full wide frame: sensor readings joined to labels, chronological.
-- Params: start, end (datetime, inclusive); cutoff (datetime or NULL,
--         NULL -> split = 'unassigned')
-- Returns: wafer_id, s001..s590, target, timestamp, split
-- Convention: cv is inclusive of the cutoff, holdout exclusive.
-- Callers must assert no timestamp equals cutoff exactly.
SELECT r.*, l.target, l.timestamp,
       CASE
           WHEN :cutoff IS NULL THEN 'unassigned'
           WHEN l.timestamp <= :cutoff THEN 'cv'
           ELSE 'holdout'
       END AS split
FROM sensor_readings r
JOIN wafer_labels l USING (wafer_id)
WHERE l.timestamp BETWEEN :start AND :end
ORDER BY l.timestamp;