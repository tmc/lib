-- cudaprof named-query schema v1, qualified with Trace Processor v58.2.
-- Roles come from versioned annotations, never display names.
CREATE PERFETTO VIEW cudaprof_track_v1 AS
SELECT s.track_id,
 EXTRACT_ARG(s.arg_set_id, 'debug.cudaprof_role') AS role,
 EXTRACT_ARG(s.arg_set_id, 'debug.cudaprof_clock') AS clock,
 EXTRACT_ARG(s.arg_set_id, 'debug.cudaprof_arm') AS arm,
 EXTRACT_ARG(s.arg_set_id, 'debug.cudaprof_capture_sha256') AS capture_sha256,
 EXTRACT_ARG(s.arg_set_id, 'debug.cudaprof_journal') AS journal,
 EXTRACT_ARG(s.arg_set_id, 'debug.cudaprof_device') AS device,
 EXTRACT_ARG(s.arg_set_id, 'debug.cudaprof_scope_id') AS scope_id,
 EXTRACT_ARG(s.arg_set_id, 'debug.cudaprof_aggregation') AS aggregation,
 EXTRACT_ARG(s.arg_set_id, 'debug.cudaprof_canonical') AS canonical
FROM slice s
WHERE EXTRACT_ARG(s.arg_set_id, 'debug.cudaprof_schema') = 'cudaprof.track.v1'
AND EXTRACT_ARG(s.arg_set_id, 'debug.cudaprof_schema_version') = 1;

-- Duplicate metadata must not multiply source records. Reject that track.
CREATE PERFETTO VIEW cudaprof_unique_track_v1 AS
SELECT * FROM cudaprof_track_v1 WHERE track_id IN (
 SELECT track_id FROM cudaprof_track_v1 GROUP BY track_id HAVING COUNT(*) = 1
);

CREATE PERFETTO VIEW cudaprof_inventory_v1 AS
SELECT role, clock, arm, capture_sha256, journal, device,
 COUNT(*) AS tracks
FROM cudaprof_unique_track_v1
GROUP BY role, clock, arm, capture_sha256, journal, device;

-- Only sole raw device representations are eligible. Summary and hierarchy
-- tracks never participate. Raw source identity is retained independently of
-- Trace Processor's local slice id.
CREATE PERFETTO VIEW cudaprof_raw_device_v1 AS
SELECT s.id, s.ts, s.dur, t.clock, t.arm, t.capture_sha256,
 EXTRACT_ARG(s.arg_set_id,'debug.raw_journal') AS journal,
 EXTRACT_ARG(s.arg_set_id,'debug.device_id') AS device,
 EXTRACT_ARG(s.arg_set_id,'debug.raw_record_sequence') AS raw_record_sequence,
 EXTRACT_ARG(s.arg_set_id,'debug.cudaprof_activity_kind') AS activity_kind,
 COALESCE(EXTRACT_ARG(s.arg_set_id,'debug.raw_symbol'),s.name) AS raw_symbol,
 EXTRACT_ARG(s.arg_set_id,'debug.grid_x') AS grid_x,
 EXTRACT_ARG(s.arg_set_id,'debug.grid_y') AS grid_y,
 EXTRACT_ARG(s.arg_set_id,'debug.grid_z') AS grid_z,
 EXTRACT_ARG(s.arg_set_id,'debug.block_x') AS block_x,
 EXTRACT_ARG(s.arg_set_id,'debug.block_y') AS block_y,
 EXTRACT_ARG(s.arg_set_id,'debug.block_z') AS block_z
FROM slice s JOIN cudaprof_unique_track_v1 t ON t.track_id=s.track_id
WHERE t.role='raw_device' AND t.aggregation='none'
 AND s.dur>=0
 AND EXTRACT_ARG(s.arg_set_id,'debug.raw_record_sequence') IS NOT NULL
 AND EXTRACT_ARG(s.arg_set_id,'debug.raw_journal') IS NOT NULL
 AND EXTRACT_ARG(s.arg_set_id,'debug.cudaprof_activity_kind') IN ('kernel','memcpy','memset');

-- A duplicated native record is not another independent launch.
CREATE PERFETTO VIEW cudaprof_unique_device_v1 AS
SELECT * FROM cudaprof_raw_device_v1 WHERE (capture_sha256,arm,journal,raw_record_sequence) IN (
 SELECT capture_sha256,arm,journal,raw_record_sequence
 FROM cudaprof_raw_device_v1
 GROUP BY capture_sha256,arm,journal,raw_record_sequence HAVING COUNT(*)=1
);

-- Running MAX(end), not LAG(end), handles nested/overlapping intervals.
CREATE PERFETTO VIEW cudaprof_ordered_device_v1 AS
SELECT *, MAX(ts+dur) OVER (
 PARTITION BY capture_sha256,arm,journal,device,clock
 ORDER BY ts,ts+dur,id ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
) AS previous_end
FROM cudaprof_unique_device_v1;
CREATE PERFETTO VIEW cudaprof_device_islands_v1 AS
SELECT *, SUM(CASE WHEN previous_end IS NULL OR ts>previous_end THEN 1 ELSE 0 END)
 OVER (PARTITION BY capture_sha256,arm,journal,device,clock ORDER BY ts,ts+dur,id) AS island
FROM cudaprof_ordered_device_v1;
CREATE PERFETTO VIEW cudaprof_recorded_busy_v1 AS
SELECT capture_sha256,arm,journal,device,clock,island,
 MIN(ts) AS ts,MAX(ts+dur)-MIN(ts) AS dur
FROM cudaprof_device_islands_v1
GROUP BY capture_sha256,arm,journal,device,clock,island;
CREATE PERFETTO VIEW cudaprof_recorded_gaps_v1 AS
SELECT capture_sha256,arm,journal,device,clock,ts AS end_ns,previous_end AS start_ns,ts-previous_end AS gap_ns
FROM cudaprof_ordered_device_v1 WHERE previous_end IS NOT NULL AND ts>previous_end;
CREATE PERFETTO VIEW cudaprof_device_totals_v1 AS
SELECT capture_sha256,arm,journal,device,clock,COUNT(*) AS busy_intervals,SUM(dur) AS recorded_union_ns
FROM cudaprof_recorded_busy_v1 GROUP BY capture_sha256,arm,journal,device,clock;
CREATE PERFETTO VIEW cudaprof_kernel_shapes_v1 AS
SELECT capture_sha256,arm,journal,device,raw_symbol,grid_x,grid_y,grid_z,block_x,block_y,block_z,
 COUNT(*) AS launches,MIN(dur) AS min_ns,MAX(dur) AS max_ns,SUM(dur) AS sum_ns
FROM cudaprof_unique_device_v1 WHERE activity_kind='kernel'
GROUP BY capture_sha256,arm,journal,device,raw_symbol,grid_x,grid_y,grid_z,block_x,block_y,block_z;

CREATE PERFETTO VIEW cudaprof_qualifications_v1 AS
SELECT 'role_sensitive_analysis' AS gate,
 CASE WHEN COUNT(*)>0 THEN 'available' ELSE 'unavailable' END AS state,
 'explicit track schema v1; no collector/workload/correctness qualification inferred' AS reason
FROM cudaprof_unique_track_v1;
CREATE PERFETTO VIEW cudaprof_delivery_cadence_v1 AS
SELECT 'unavailable' AS state,'request identity and start endpoint must be retained; no timestamp-based request guess' AS reason;

-- These are retained native match receipts, not joins inferred from trace time.
-- Track roles and instant schemas must both be explicit. Domain stays scoped.
CREATE PERFETTO VIEW cudaprof_submission_raw_v1 AS
SELECT t.capture_sha256,t.arm,
 EXTRACT_ARG(s.arg_set_id,'debug.raw_journal') AS journal,
 EXTRACT_ARG(s.arg_set_id,'debug.raw_activity_sequence') AS raw_activity_sequence,
 EXTRACT_ARG(s.arg_set_id,'debug.raw_activity_index') AS raw_activity_index,
 EXTRACT_ARG(s.arg_set_id,'debug.domain') AS domain,
 EXTRACT_ARG(s.arg_set_id,'debug.api_status') AS api_status,
 EXTRACT_ARG(s.arg_set_id,'debug.range_status') AS range_status,
 EXTRACT_ARG(s.arg_set_id,'debug.metadata_status') AS metadata_status,
 EXTRACT_ARG(s.arg_set_id,'debug.provenance') AS provenance,
 EXTRACT_ARG(s.arg_set_id,'debug.native_match_hex') AS native_match_hex
FROM slice s JOIN cudaprof_unique_track_v1 t ON t.track_id=s.track_id
WHERE t.role='submission_receipt'
 AND EXTRACT_ARG(s.arg_set_id,'debug.cudaprof_schema')='cudaprof.correlation.v1'
 AND EXTRACT_ARG(s.arg_set_id,'debug.cudaprof_schema_version')=1
 AND t.capture_sha256=EXTRACT_ARG(s.arg_set_id,'debug.cudaprof_capture_sha256')
 AND t.arm=EXTRACT_ARG(s.arg_set_id,'debug.cudaprof_arm');

CREATE PERFETTO VIEW cudaprof_submission_unique_v1 AS
SELECT * FROM cudaprof_submission_raw_v1 WHERE
 (capture_sha256,arm,journal,raw_activity_sequence,domain) IN (
 SELECT capture_sha256,arm,journal,raw_activity_sequence,domain
 FROM cudaprof_submission_raw_v1
 WHERE journal IS NOT NULL AND journal!='' AND raw_activity_sequence>0
 AND LENGTH(capture_sha256)=64 AND capture_sha256 NOT GLOB '*[^0-9a-fA-F]*'
 AND domain IN ('driver','runtime') AND native_match_hex IS NOT NULL AND LENGTH(native_match_hex)>0 AND provenance IS NOT NULL
 AND api_status IN ('matched','missing','ambiguous','unsupported')
 AND range_status IN ('matched','missing','ambiguous','unsupported')
 AND metadata_status IN ('matched','missing','ambiguous','unsupported')
 GROUP BY capture_sha256,arm,journal,raw_activity_sequence,domain HAVING COUNT(*)=1
);

CREATE PERFETTO VIEW cudaprof_submission_coverage_raw_v1 AS
SELECT t.capture_sha256,t.arm,
 EXTRACT_ARG(s.arg_set_id,'debug.domain') AS domain,
 EXTRACT_ARG(s.arg_set_id,'debug.activities') AS activities
FROM slice s JOIN cudaprof_unique_track_v1 t ON t.track_id=s.track_id
WHERE t.role='submission_receipt'
 AND EXTRACT_ARG(s.arg_set_id,'debug.cudaprof_schema')='cudaprof.correlation_coverage.v1'
 AND EXTRACT_ARG(s.arg_set_id,'debug.cudaprof_schema_version')=1
 AND t.capture_sha256=EXTRACT_ARG(s.arg_set_id,'debug.cudaprof_capture_sha256')
 AND t.arm=EXTRACT_ARG(s.arg_set_id,'debug.cudaprof_arm');

CREATE PERFETTO VIEW cudaprof_submission_coverage_v1 AS
SELECT c.capture_sha256,c.arm,c.domain,c.activities,
 CASE WHEN c.activities>=0 AND c.domain IN ('driver','runtime')
 AND LENGTH(c.capture_sha256)=64 AND c.capture_sha256 NOT GLOB '*[^0-9a-fA-F]*'
 AND c.activities=(SELECT COUNT(*) FROM cudaprof_submission_unique_v1 u
 WHERE u.capture_sha256=c.capture_sha256 AND u.arm=c.arm AND u.domain=c.domain)
 AND c.activities=(SELECT COUNT(*) FROM cudaprof_submission_raw_v1 u
 WHERE u.capture_sha256=c.capture_sha256 AND u.arm=c.arm AND u.domain=c.domain)
 THEN 'available' ELSE 'unavailable' END AS state
FROM cudaprof_submission_coverage_raw_v1 c WHERE
 (c.capture_sha256,c.arm,c.domain) IN (
 SELECT capture_sha256,arm,domain FROM cudaprof_submission_coverage_raw_v1
 GROUP BY capture_sha256,arm,domain HAVING COUNT(*)=1
);

CREATE PERFETTO VIEW cudaprof_correlations_v1 AS
SELECT c.state,'coverage' AS row_kind,c.capture_sha256,c.arm,NULL AS journal,
 NULL AS raw_activity_sequence,NULL AS raw_activity_index,c.domain,
 NULL AS api_status,NULL AS range_status,NULL AS metadata_status,NULL AS provenance,
 NULL AS native_match_hex,
 'retained native submission identity; supplied provenance does not establish collector completeness or causality' AS reason
FROM cudaprof_submission_coverage_v1 c
UNION ALL
SELECT 'available','activity',u.capture_sha256,u.arm,u.journal,u.raw_activity_sequence,
 u.raw_activity_index,u.domain,u.api_status,u.range_status,u.metadata_status,
 u.provenance,u.native_match_hex,NULL
FROM cudaprof_submission_unique_v1 u JOIN cudaprof_submission_coverage_v1 c
 ON c.capture_sha256=u.capture_sha256 AND c.arm=u.arm AND c.domain=u.domain
WHERE c.state='available';

CREATE PERFETTO VIEW cudaprof_unmatched_v1 AS
SELECT * FROM cudaprof_correlations_v1 WHERE row_kind='coverage'
 OR api_status!='matched' OR range_status!='matched' OR metadata_status!='matched';
