package metalprof_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/tmc/lib/apple/x/metal/metalcapture"
	"github.com/tmc/lib/apple/x/metal/metalprof"
)

func TestParseReplayProfilerOutput(t *testing.T) {
	validJSON := []byte(`{
		"dispatches": [
			{"index": 0, "pipeline_id": 8, "function_name": "vector_add", "duration_us": 35, "simd_share": 0.0},
			{"index": 1, "pipeline_id": 10, "function_name": "compute", "duration_us": 12, "simd_share": 0.85}
		],
		"pipelines": [
			{"pipeline_id": 8, "function_name": "vector_add", "recorded_statistics":["Instruction count","Spilled bytes"], "instruction_count": 9, "spilled_bytes": 0},
			{"pipeline_id": 10, "function_name": "compute", "recorded_statistics":["Instruction count","Spilled bytes"], "instruction_count": 20, "spilled_bytes": 0}
		],
		"total_dispatch_time_us": 47
	}`)

	prof, obs, err := metalprof.ParseReplayProfilerOutput(validJSON)
	if err != nil {
		t.Fatalf("ParseReplayProfilerOutput failed: %v", err)
	}

	if got := prof.SampleType[1].Type; got != "inferred_dispatch_time" {
		t.Fatalf("replay metric = %q, want inferred_dispatch_time", got)
	}
	if _, _, err := metalprof.ParseReplayProfilerOutput([]byte(`{"dispatches":[{"duration_us":9223372036854775},{"duration_us":1}]}`)); err == nil {
		t.Fatal("accepted a replay total that overflows nanoseconds")
	}

	if len(prof.Sample) != 2 {
		t.Errorf("got %d samples, want 2", len(prof.Sample))
	}
	if prof.Sample[0].Value[1] != 35000 {
		t.Errorf("sample 0 duration = %d, want 35000", prof.Sample[0].Value[1])
	}
	if prof.Sample[1].Value[1] != 12000 {
		t.Errorf("sample 1 duration = %d, want 12000", prof.Sample[1].Value[1])
	}

	// Verify measured zero is preserved and inferred duration observation ID
	var foundZero, foundDuration bool
	for _, o := range obs {
		if o.ID == "replay_dispatch_0_simd_share" {
			foundZero = true
			if o.Value.FloatVal != 0.0 {
				t.Errorf("expected simd_share 0.0, got %f", o.Value.FloatVal)
			}
		}
		if o.ID == "replay_dispatch_0_inferred_duration" {
			foundDuration = true
			if o.Value.IntVal != 35000 {
				t.Errorf("expected inferred duration 35000 ns, got %d", o.Value.IntVal)
			}
		}
	}
	if !foundZero {
		t.Errorf("did not preserve measured zero simd_share for dispatch 0")
	}
	if !foundDuration {
		t.Errorf("did not find replay_dispatch_0_inferred_duration")
	}

	// Total mismatch control
	mismatchJSON := []byte(`{
		"dispatches": [
			{"index": 0, "function_name": "vector_add", "duration_us": 35}
		],
		"total_dispatch_time_us": 100
	}`)
	if _, _, err := metalprof.ParseReplayProfilerOutput(mismatchJSON); err == nil {
		t.Errorf("expected error on total_dispatch_time_us mismatch")
	}

	// Negative duration control
	negativeJSON := []byte(`{
		"dispatches": [
			{"index": 0, "function_name": "vector_add", "duration_us": -10}
		]
	}`)
	if _, _, err := metalprof.ParseReplayProfilerOutput(negativeJSON); err == nil {
		t.Errorf("expected error on negative duration")
	}

	// Empty payload control
	if _, _, err := metalprof.ParseReplayProfilerOutput([]byte(`{}`)); err == nil {
		t.Errorf("expected error on empty {} export")
	}
	if _, _, err := metalprof.ParseReplayProfilerOutput([]byte(``)); err == nil {
		t.Errorf("expected error on empty byte export")
	}
}

func TestPipelineIDSpecializationDisambiguation(t *testing.T) {
	// Two distinct pipeline specializations with the same function name "compute"
	jsonPayload := []byte(`{
		"metal_device_name": "Apple M4 Max",
		"dispatches": [
			{"index": 0, "pipeline_id": 10, "function_name": "compute", "duration_us": 20},
			{"index": 1, "pipeline_id": 12, "function_name": "compute", "duration_us": 30}
		],
		"pipelines": [
			{"pipeline_id": 10, "function_name": "compute", "recorded_statistics":["Instruction count","Spilled bytes"], "instruction_count": 25, "spilled_bytes": 0},
			{"pipeline_id": 12, "function_name": "compute", "recorded_statistics":["Instruction count","Spilled bytes"], "instruction_count": 40, "spilled_bytes": 16}
		],
		"total_dispatch_time_us": 50
	}`)

	prof, obs, err := metalprof.ParseReplayProfilerOutput(jsonPayload)
	if err != nil {
		t.Fatalf("ParseReplayProfilerOutput: %v", err)
	}

	// Verify dispatches have distinct pipeline IDs
	if len(prof.Sample) != 2 {
		t.Fatalf("got %d samples, want 2", len(prof.Sample))
	}
	if prof.Sample[0].Label["pipeline_id"][0] != "10" {
		t.Errorf("sample 0 pipeline_id = %s, want 10", prof.Sample[0].Label["pipeline_id"][0])
	}
	if prof.Sample[1].Label["pipeline_id"][0] != "12" {
		t.Errorf("sample 1 pipeline_id = %s, want 12", prof.Sample[1].Label["pipeline_id"][0])
	}

	// Verify pipeline observation IDs do not collide
	obsMap := make(map[string]metalcapture.Observation)
	for _, o := range obs {
		obsMap[o.ID] = o
	}

	obs10, ok10 := obsMap["compiled_pipeline_10_compute_instructions"]
	obs12, ok12 := obsMap["compiled_pipeline_12_compute_instructions"]
	if !ok10 || !ok12 {
		t.Fatalf("missing pipeline-specific observation IDs; map keys: %v", obsMap)
	}
	if obs10.Value.IntVal != 25 {
		t.Errorf("pipeline 10 instructions = %d, want 25", obs10.Value.IntVal)
	}
	if obs12.Value.IntVal != 40 {
		t.Errorf("pipeline 12 instructions = %d, want 40", obs12.Value.IntVal)
	}
	if obs10.Device != "Apple M4 Max" || obs12.Device != "Apple M4 Max" {
		t.Errorf("expected device name preserved in observations")
	}
}

func TestEnrichTraceReplayTimeoutScratchRetention(t *testing.T) {
	// Create a real dummy trace directory so ReplayTraceContext begins execution
	dir := t.TempDir()
	dummyTrace := filepath.Join(dir, "dummy.gputrace")
	if err := os.MkdirAll(dummyTrace, 0755); err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Millisecond)
	defer cancel()

	time.Sleep(2 * time.Millisecond) // Ensure context is already done
	_, err := metalprof.EnrichTraceReplay(ctx, dummyTrace, 1*time.Millisecond)
	if err == nil {
		t.Fatal("expected error on expired context")
	}

	// Verify errors.Is for both the unresolved ownership sentinel and context deadline
	if !errors.Is(err, metalprof.ErrReplayUnresolvedOwnership) {
		t.Errorf("error %v does not wrap ErrReplayUnresolvedOwnership", err)
	}
	if !errors.Is(err, context.DeadlineExceeded) && !errors.Is(err, context.Canceled) {
		t.Errorf("error %v does not wrap context deadline or cancellation", err)
	}
}
