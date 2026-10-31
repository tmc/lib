package nvtelemetry_test

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"math"
	"testing"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

// Changed bytes below are derived refusal controls, never additional authentic
// captures. Only the first observed annotation's selected numeric field changes.
func pytorch210IntervalMutation(t *testing.T, field, value string, missing, duplicate bool) []byte {
	t.Helper()
	var root map[string]json.RawMessage
	if err := json.Unmarshal(pytorch210Fixture(t, "eager"), &root); err != nil {
		t.Fatal(err)
	}
	var events []json.RawMessage
	if err := json.Unmarshal(root["traceEvents"], &events); err != nil {
		t.Fatal(err)
	}
	var first map[string]json.RawMessage
	if err := json.Unmarshal(events[0], &first); err != nil {
		t.Fatal(err)
	}
	if string(first["cat"]) != `"user_annotation"` {
		t.Fatal("actual first annotation changed")
	}
	if !duplicate {
		if missing {
			delete(first, field)
		} else {
			first[field] = json.RawMessage(value)
		}
		var err error
		events[0], err = json.Marshal(first)
		if err != nil {
			t.Fatal(err)
		}
	} else {
		// Preserve the original member and add a second identical/conflicting
		// key. json.RawMessage retains both through the enclosing marshal.
		prefix := []byte(`{"` + field + `":` + value + `,`)
		events[0] = append(prefix, events[0][bytes.IndexByte(events[0], '{')+1:]...)
	}
	var err error
	root["traceEvents"], err = json.Marshal(events)
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(root)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func readPyTorch210Interval(t *testing.T, data []byte) (nvtelemetry.WorkloadExport, []nvtelemetry.WorkloadAnnotation, error) {
	t.Helper()
	defer func() {
		if p := recover(); p != nil {
			t.Fatalf("public reader panicked instead of returning an error: %v", p)
		}
	}()
	return nvtelemetry.ReadPyTorchAnnotations(bytes.NewReader(data), pytorch210Source(), nvtelemetry.WorkloadLimits{})
}

func TestPyTorch210CapturedIntervalRefusals(t *testing.T) {
	for _, tt := range []struct {
		name, field, value string
		missing, duplicate bool
	}{
		{name: "missing_ts", field: "ts", missing: true},
		{name: "missing_dur", field: "dur", missing: true},
		{name: "null_ts", field: "ts", value: "null"},
		{name: "null_dur", field: "dur", value: "null"},
		{name: "string_ts", field: "ts", value: `"1"`},
		{name: "string_dur", field: "dur", value: `"1"`},
		{name: "huge_exponent_ts", field: "ts", value: "1e100000"},
		{name: "huge_exponent_dur", field: "dur", value: "1e100000"},
		{name: "tiny_exponent_ts", field: "ts", value: "1e-1001"},
		{name: "fractional_ns_ts", field: "ts", value: "0.0001"},
		{name: "fractional_ns_dur", field: "dur", value: "0.0001"},
		{name: "int64_overflow_ts", field: "ts", value: "9223372036854775.808"},
		{name: "int64_overflow_dur", field: "dur", value: "9223372036854775.808"},
		{name: "end_overflow", field: "ts", value: "9223372036854775.807"},
		{name: "negative_duration", field: "dur", value: "-0.001"},
		{name: "duplicate_ts_conflict", field: "ts", value: "0", duplicate: true},
		{name: "duplicate_dur_conflict", field: "dur", value: "0", duplicate: true},
		{name: "duplicate_ts_identical", field: "ts", value: "2230438240646.515", duplicate: true},
		{name: "duplicate_dur_identical", field: "dur", value: "118.958", duplicate: true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			data := pytorch210IntervalMutation(t, tt.field, tt.value, tt.missing, tt.duplicate)
			export, annotations, err := readPyTorch210Interval(t, data)
			if err == nil {
				t.Fatalf("invalid captured interval admitted: %+v", annotations)
			}
			if annotations != nil {
				t.Fatal("partial annotation population returned on error")
			}
			sum := sha256.Sum256(data)
			if !bytes.Equal(export.Raw, data) || export.RawSHA256 != hex.EncodeToString(sum[:]) {
				t.Fatal("rejected numeric input lost raw byte identity")
			}
		})
	}
}

func TestPyTorch210CapturedIntervalPositives(t *testing.T) {
	for _, tt := range []struct {
		name, start, duration   string
		wantStart, wantDuration int64
	}{
		{"actual", "2230438240646.515", "118.958", 2230438240646515, 118958},
		{"zero", "0", "0", 0, 0},
		{"ordinary_exact_ns", "1.250", "0.125", 1250, 125},
		{"last_int64_ns", "9223372036854775.807", "0", math.MaxInt64, 0},
	} {
		t.Run(tt.name, func(t *testing.T) {
			data := pytorch210Fixture(t, "eager")
			if tt.name != "actual" {
				data = pytorch210IntervalMutation(t, "ts", tt.start, false, false)
				// Set duration without reconstructing any producer field.
				var root map[string]json.RawMessage
				if err := json.Unmarshal(data, &root); err != nil {
					t.Fatal(err)
				}
				var events []json.RawMessage
				if err := json.Unmarshal(root["traceEvents"], &events); err != nil {
					t.Fatal(err)
				}
				var first map[string]json.RawMessage
				if err := json.Unmarshal(events[0], &first); err != nil {
					t.Fatal(err)
				}
				first["dur"] = json.RawMessage(tt.duration)
				var err error
				events[0], err = json.Marshal(first)
				if err != nil {
					t.Fatal(err)
				}
				root["traceEvents"], err = json.Marshal(events)
				if err != nil {
					t.Fatal(err)
				}
				data, err = json.Marshal(root)
				if err != nil {
					t.Fatal(err)
				}
			}
			export, annotations, err := readPyTorch210Interval(t, data)
			if err != nil || len(annotations) != 2 {
				t.Fatalf("valid captured interval: annotations=%d error=%v", len(annotations), err)
			}
			first := annotations[0]
			if first.RawStart != tt.start || first.RawDuration != tt.duration || first.StartNanos != tt.wantStart || first.DurationNanos != tt.wantDuration {
				t.Fatalf("numeric spelling/conversion changed: %+v", first)
			}
			if !bytes.Equal(export.Raw, data) || export.Source.ClockDomain != "" || export.Source.ClockEpochID != "" {
				t.Fatal("raw bytes or unknown clock declaration changed")
			}
		})
	}
}
