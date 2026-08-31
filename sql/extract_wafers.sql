-- Extract wafer sensor readings joined to labels.
-- Params: :start (datetime, inclusive), :end (datetime, inclusive)
-- Returns: wafer_id, s001..s590, target, timestamp — chronological order.
SELECT r.*, l.target, l.timestamp
FROM sensor_readings r
JOIN wafer_labels l USING (wafer_id)
WHERE l.timestamp BETWEEN :start AND :end
ORDER BY l.timestamp;