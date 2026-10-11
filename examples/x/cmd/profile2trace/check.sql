-- Compare cuda_kernel slices with CUPTI_ACTIVITY_KIND_KERNEL in the source DB.
SELECT category, count(*) AS slices, sum(dur) AS total_ns
FROM slice
GROUP BY category
ORDER BY category;

-- Completed input intervals must not become unfinished slices.
SELECT count(*) AS unfinished_slices FROM slice WHERE dur < 0;

-- Any nonzero parser error requires investigation.
SELECT name, value FROM stats
WHERE severity = 'error' AND value != 0
ORDER BY name;
