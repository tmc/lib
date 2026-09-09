package metalcapture_test

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/apple/x/metal/metalcapture"
)

func minimalProfile() *profile.Profile {
	return &profile.Profile{
		SampleType: []*profile.ValueType{
			{Type: "gpu_time", Unit: "nanoseconds"},
		},
		PeriodType: &profile.ValueType{
			Type: "gpu_time", Unit: "nanoseconds",
		},
	}
}

func TestRoundTrip(t *testing.T) {
	c := &metalcapture.Capture{
		Profile: minimalProfile(),
		Files: map[string][]byte{
			"kernel.metal": []byte("kernel void add() {}"),
			"kernel.air":   []byte("air-bytecode-stub"),
		},
		Code: metalcapture.CodeFiles{
			Source: "kernel.metal",
			AIR:    "kernel.air",
		},
		Metadata: json.RawMessage(`{"device":"Apple M4 Max","passes":12}`),
	}

	var buf bytes.Buffer
	if err := metalcapture.Write(&buf, c); err != nil {
		t.Fatalf("Write: %v", err)
	}

	readBack, err := metalcapture.Read(&buf)
	if err != nil {
		t.Fatalf("Read: %v", err)
	}

	if readBack.Code.Source != "kernel.metal" {
		t.Errorf("Code.Source = %q, want kernel.metal", readBack.Code.Source)
	}
	if readBack.Code.AIR != "kernel.air" {
		t.Errorf("Code.AIR = %q, want kernel.air", readBack.Code.AIR)
	}
	if string(readBack.Files["kernel.metal"]) != "kernel void add() {}" {
		t.Errorf("Files[kernel.metal] mismatch")
	}
	if string(readBack.Metadata) != `{"device":"Apple M4 Max","passes":12}` {
		t.Errorf("Metadata mismatch: %s", string(readBack.Metadata))
	}
}

func TestWriteDirReadDir(t *testing.T) {
	dir := t.TempDir()
	c := &metalcapture.Capture{
		Profile: minimalProfile(),
		Files: map[string][]byte{
			"kernel.metal": []byte("kernel void add() {}"),
		},
		Code: metalcapture.CodeFiles{
			Source: "kernel.metal",
		},
	}

	if err := metalcapture.WriteDir(dir, c); err != nil {
		t.Fatalf("WriteDir: %v", err)
	}

	readBack, err := metalcapture.ReadDir(dir)
	if err != nil {
		t.Fatalf("ReadDir: %v", err)
	}

	if readBack.Code.Source != "kernel.metal" {
		t.Errorf("Source = %q, want kernel.metal", readBack.Code.Source)
	}
}

func TestCorruptedDigest(t *testing.T) {
	c := &metalcapture.Capture{
		Profile: minimalProfile(),
		Files: map[string][]byte{
			"kernel.metal": []byte("kernel void add() {}"),
		},
		Code: metalcapture.CodeFiles{
			Source: "kernel.metal",
		},
	}

	var buf bytes.Buffer
	if err := metalcapture.Write(&buf, c); err != nil {
		t.Fatalf("Write: %v", err)
	}

	p, err := profile.ParseData(buf.Bytes())
	if err != nil {
		t.Fatal(err)
	}

	// Corrupt file in comment
	for i, c := range p.Comments {
		if strings.HasPrefix(c, "#metalprof:capture:") {
			nl := strings.Index(c, "\n")
			prefix := c[:nl+1]
			var raw map[string]any
			if err := json.Unmarshal([]byte(c[nl+1:]), &raw); err != nil {
				t.Fatal(err)
			}
			files := raw["files"].(map[string]any)
			files["kernel.metal"] = "corrupted content"
			tampered, _ := json.Marshal(raw)
			p.Comments[i] = prefix + string(tampered)
		}
	}

	var tamperedBuf bytes.Buffer
	if err := p.Write(&tamperedBuf); err != nil {
		t.Fatal(err)
	}

	if _, err := metalcapture.Read(&tamperedBuf); err == nil {
		t.Fatal("Read accepted corrupted file with mismatched checksum")
	}
}

func TestValidationErrors(t *testing.T) {
	// Nil capture
	if err := metalcapture.Write(new(bytes.Buffer), nil); err == nil {
		t.Error("expected error for nil capture")
	}

	// Reserved filename
	badCap := &metalcapture.Capture{
		Profile: minimalProfile(),
		Files: map[string][]byte{
			"gpu.pprof": []byte("bad"),
		},
		Code: metalcapture.CodeFiles{
			Source: "gpu.pprof",
		},
	}
	if err := metalcapture.Write(new(bytes.Buffer), badCap); err == nil {
		t.Error("expected error for reserved filename")
	}

	// Missing code file in files
	missingCap := &metalcapture.Capture{
		Profile: minimalProfile(),
		Files: map[string][]byte{
			"other.txt": []byte("hello"),
		},
		Code: metalcapture.CodeFiles{
			Source: "kernel.metal",
		},
	}
	if err := metalcapture.Write(new(bytes.Buffer), missingCap); err == nil {
		t.Error("expected error for missing source file")
	}
}

func TestObservationsRoundTrip(t *testing.T) {
	dispatchZero := 0
	dispatchOne := 1
	c := &metalcapture.Capture{
		Profile: minimalProfile(),
		Files: map[string][]byte{
			"kernel.metal": []byte("kernel void add() {}"),
		},
		Code: metalcapture.CodeFiles{
			Source: "kernel.metal",
		},
		Observations: []metalcapture.Observation{
			{
				ID:          "gpu_time",
				Name:        "GPU Duration",
				Unit:        "nanoseconds",
				Kind:        metalcapture.MetricKindDuration,
				Aggregation: metalcapture.AggregationSum,
				Scope:       metalcapture.ScopeStageBoundary,
				Origin:      metalcapture.OriginLive,
				Status:      metalcapture.StatusCollected,
				Pipeline:    "add",
				DispatchID:  &dispatchZero,
				Value:       metalcapture.NewIntObservationValue(12345),
			},
			{
				ID:          "replay_simd_share",
				Name:        "SIMD Execution Share",
				Unit:        "ratio",
				Kind:        metalcapture.MetricKindRatio,
				Aggregation: metalcapture.AggregationMean,
				Scope:       metalcapture.ScopeDispatch,
				Origin:      metalcapture.OriginReplay,
				Status:      metalcapture.StatusCollected,
				Pipeline:    "add",
				DispatchID:  &dispatchOne,
				Value:       metalcapture.NewFloatObservationValue(0.875),
				NonAdditive: true,
			},
		},
	}

	var buf bytes.Buffer
	if err := metalcapture.Write(&buf, c); err != nil {
		t.Fatalf("Write: %v", err)
	}

	readBack, err := metalcapture.Read(&buf)
	if err != nil {
		t.Fatalf("Read: %v", err)
	}

	if len(readBack.Observations) != 2 {
		t.Fatalf("got %d observations, want 2", len(readBack.Observations))
	}

	obs0 := readBack.Observations[0]
	if obs0.ID != "gpu_time" || obs0.Origin != metalcapture.OriginLive || obs0.Value.IntVal != 12345 {
		t.Errorf("unexpected observation 0: %+v", obs0)
	}

	obs1 := readBack.Observations[1]
	if obs1.ID != "replay_simd_share" || obs1.Origin != metalcapture.OriginReplay || !obs1.NonAdditive || obs1.Value.FloatVal != 0.875 {
		t.Errorf("unexpected observation 1: %+v", obs1)
	}
}

func TestLargeIntegerObservation(t *testing.T) {
	// Value greater than 2^53 (9,007,199,254,740,992)
	const largeInt = int64(1<<55 + 1234567)
	c := &metalcapture.Capture{
		Profile: minimalProfile(),
		Files: map[string][]byte{
			"kernel.metal": []byte("kernel void add() {}"),
		},
		Code: metalcapture.CodeFiles{
			Source: "kernel.metal",
		},
		Observations: []metalcapture.Observation{
			{
				ID:          "hardware_counter_ticks",
				Name:        "Raw Timestamp Ticks",
				Unit:        "ticks",
				Kind:        metalcapture.MetricKindCount,
				Aggregation: metalcapture.AggregationLast,
				Scope:       metalcapture.ScopeStageBoundary,
				Origin:      metalcapture.OriginLive,
				Value:       metalcapture.NewIntObservationValue(largeInt),
			},
		},
	}

	var buf bytes.Buffer
	if err := metalcapture.Write(&buf, c); err != nil {
		t.Fatalf("Write: %v", err)
	}

	readBack, err := metalcapture.Read(&buf)
	if err != nil {
		t.Fatalf("Read: %v", err)
	}

	if len(readBack.Observations) != 1 {
		t.Fatalf("got %d observations, want 1", len(readBack.Observations))
	}

	obs := readBack.Observations[0]
	if !obs.Value.IsInt || obs.Value.IntVal != largeInt {
		t.Errorf("large integer precision lost: got %d (IsInt: %v), want %d", obs.Value.IntVal, obs.Value.IsInt, largeInt)
	}
}

func TestDuplicateNamePipelineIdentity(t *testing.T) {
	// Test dispatches with identical function names but distinct pipeline IDs and dispatch IDs
	d0 := 0
	d1 := 1
	c := &metalcapture.Capture{
		Profile: minimalProfile(),
		Files: map[string][]byte{
			"kernel.metal": []byte("kernel void add() {}"),
		},
		Code: metalcapture.CodeFiles{
			Source: "kernel.metal",
		},
		Observations: []metalcapture.Observation{
			{
				ID:         "replay_dispatch_duration",
				Name:       "Replay Dispatch Duration",
				Unit:       "nanoseconds",
				Kind:       metalcapture.MetricKindDuration,
				Scope:      metalcapture.ScopeDispatch,
				Origin:     metalcapture.OriginReplay,
				Pipeline:   "add:specialization_A",
				DispatchID: &d0,
				Value:      metalcapture.NewIntObservationValue(15000),
			},
			{
				ID:         "replay_dispatch_duration",
				Name:       "Replay Dispatch Duration",
				Unit:       "nanoseconds",
				Kind:       metalcapture.MetricKindDuration,
				Scope:      metalcapture.ScopeDispatch,
				Origin:     metalcapture.OriginReplay,
				Pipeline:   "add:specialization_B",
				DispatchID: &d1,
				Value:      metalcapture.NewIntObservationValue(16000),
			},
		},
	}

	var buf bytes.Buffer
	if err := metalcapture.Write(&buf, c); err != nil {
		t.Fatalf("Write: %v", err)
	}

	readBack, err := metalcapture.Read(&buf)
	if err != nil {
		t.Fatalf("Read: %v", err)
	}

	if len(readBack.Observations) != 2 {
		t.Fatalf("got %d observations, want 2", len(readBack.Observations))
	}

	if readBack.Observations[0].Pipeline != "add:specialization_A" || *readBack.Observations[0].DispatchID != 0 {
		t.Errorf("pipeline identity 0 mismatch: %+v", readBack.Observations[0])
	}
	if readBack.Observations[1].Pipeline != "add:specialization_B" || *readBack.Observations[1].DispatchID != 1 {
		t.Errorf("pipeline identity 1 mismatch: %+v", readBack.Observations[1])
	}
}

func TestV1BackwardCompatibility(t *testing.T) {
	// Construct a synthetic v1 capture comment
	p := minimalProfile()
	v1Data := `{"manifest":{"version":1,"code":{"source":"kernel.metal"},"sha256":{"kernel.metal":"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"}},"files":{"kernel.metal":""}}`
	p.Comments = []string{
		"Metal source, intermediate representation, and runtime artifacts are embedded; use metalprof for inspection.",
		"#metalprof:capture:v1\n" + v1Data,
	}

	var buf bytes.Buffer
	if err := p.Write(&buf); err != nil {
		t.Fatal(err)
	}

	readBack, err := metalcapture.Read(&buf)
	if err != nil {
		t.Fatalf("Read v1: %v", err)
	}

	if readBack.Code.Source != "kernel.metal" {
		t.Errorf("v1 Code.Source = %q, want kernel.metal", readBack.Code.Source)
	}
}
