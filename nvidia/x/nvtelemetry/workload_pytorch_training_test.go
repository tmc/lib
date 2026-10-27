package nvtelemetry_test

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

// This fixture is the unchanged actual PyTorch 2.10 CPU profile of the finite
// task-owned adaptation of karpathy/pytorch-made at 81886ccc8270cdf3. Its model
// source is unchanged; the CPU dimensions, binary inputs and finite Adam loop
// are declared adaptations, not a deployed application or native GPU capture.
func trainingProfile(t *testing.T) []byte {
	t.Helper()
	b, err := os.ReadFile("testdata/pytorch-2.10.0-cpu-made-training.chrome.json")
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", sha256.Sum256(b)) != "0896543a44c9fd597c59ac6bf0e0de472bd7ad38a12a4143267ea28a2a4cbe8e" {
		t.Fatal("actual training profile changed")
	}
	return b
}
func TestPyTorch210CPUTraining(t *testing.T) {
	b := trainingProfile(t)
	e, a, err := nvtelemetry.ReadPyTorchAnnotations(bytes.NewReader(b), pytorch210Source(), nvtelemetry.WorkloadLimits{})
	if err != nil {
		t.Fatal(err)
	}
	var raw struct {
		Events []struct {
			Cat, Ph, Name  string
			PID, TID, Args json.RawMessage
		} `json:"traceEvents"`
	}
	if err := json.Unmarshal(b, &raw); err != nil {
		t.Fatal(err)
	}
	i, starts, finishes := 0, 0, 0
	for _, v := range raw.Events {
		if v.Cat == "fwdbwd" {
			switch v.Ph {
			case "s":
				starts++
			case "f":
				finishes++
			default:
				t.Fatal(v)
			}
		}
		if v.Cat != "user_annotation" {
			continue
		}
		if i >= len(a) || a[i].Name != v.Name || !bytes.Equal(a[i].PID, v.PID) || !bytes.Equal(a[i].TID, v.TID) || !bytes.Equal(a[i].Args, v.Args) {
			t.Fatalf("annotation %d changed", i)
		}
		i++
	}
	if i != 15 || len(a) != i || starts != 27 || finishes != 27 || !bytes.Equal(e.Raw, b) || len(e.Metrics) != 0 {
		t.Fatalf("population annotations=%d/%d flows=%d/%d", i, len(a), starts, finishes)
	}
	if e.Source.ClockDomain != "" || e.Source.ClockEpochID != "" {
		t.Fatal("invented clock")
	}
}

// Mutations are adversarial controls, not additional authentic captures.
func TestPyTorch210CPUFlowRefusals(t *testing.T) {
	for _, tt := range []struct{ name, key, value string }{
		{"phase", "ph", `"t"`}, {"finish binding", "bp", `"e"`}, {"name", "name", `"other"`}, {"missing id", "id", ""},
		{"null id", "id", "null"}, {"string id", "id", `"9"`}, {"negative id", "id", "-1"}, {"fractional id", "id", "9.1"}, {"overflow id", "id", "18446744073709551616"}, {"unknown category", "cat", `"unknown_flow"`},
	} {
		t.Run(tt.name, func(t *testing.T) {
			var root map[string]json.RawMessage
			if err := json.Unmarshal(trainingProfile(t), &root); err != nil {
				t.Fatal(err)
			}
			var events []map[string]json.RawMessage
			if err := json.Unmarshal(root["traceEvents"], &events); err != nil {
				t.Fatal(err)
			}
			changed := false
			for _, e := range events {
				if string(e["cat"]) == `"fwdbwd"` && string(e["ph"]) == `"s"` {
					if tt.value == "" {
						delete(e, tt.key)
					} else {
						e[tt.key] = json.RawMessage(tt.value)
					}
					changed = true
					break
				}
			}
			if !changed {
				t.Fatal("missing positive flow")
			}
			root["traceEvents"], _ = json.Marshal(events)
			b, err := json.Marshal(root)
			if err != nil {
				t.Fatal(err)
			}
			if _, _, err := nvtelemetry.ReadPyTorchAnnotations(bytes.NewReader(b), pytorch210Source(), nvtelemetry.WorkloadLimits{}); err == nil {
				t.Fatal("unsupported CPU flow admitted")
			}
		})
	}
	t.Run("finish missing end binding", func(t *testing.T) {
		b := bytes.Replace(trainingProfile(t), []byte(`"bp": "e"`), []byte(`"bp": "s"`), 1)
		if bytes.Equal(b, trainingProfile(t)) {
			t.Fatal("mutation missing")
		}
		if _, _, err := nvtelemetry.ReadPyTorchAnnotations(bytes.NewReader(b), pytorch210Source(), nvtelemetry.WorkloadLimits{}); err == nil {
			t.Fatal("unsupported finish binding admitted")
		}
	})
	t.Run("device", func(t *testing.T) {
		var root map[string]json.RawMessage
		json.Unmarshal(trainingProfile(t), &root)
		root["deviceProperties"] = json.RawMessage(`[{"id":0}]`)
		b, err := json.Marshal(root)
		if err != nil {
			t.Fatal(err)
		}
		if _, _, err := nvtelemetry.ReadPyTorchAnnotations(bytes.NewReader(b), pytorch210Source(), nvtelemetry.WorkloadLimits{}); err == nil {
			t.Fatal("device capture admitted")
		}
	})
	t.Run("legacy bp stays uninterpreted", func(t *testing.T) {
		var root map[string]json.RawMessage
		if err := json.Unmarshal(trainingProfile(t), &root); err != nil {
			t.Fatal(err)
		}
		var events []map[string]json.RawMessage
		if err := json.Unmarshal(root["traceEvents"], &events); err != nil {
			t.Fatal(err)
		}
		for _, e := range events {
			if string(e["cat"]) == `"fwdbwd"` {
				e["bp"] = json.RawMessage(`7`)
				break
			}
		}
		root["traceEvents"], _ = json.Marshal(events)
		b, err := json.Marshal(root)
		if err != nil {
			t.Fatal(err)
		}
		source := pytorch210Source()
		source.Version = "2.7.0"
		e, annotations, err := nvtelemetry.ReadPyTorchAnnotations(bytes.NewReader(b), source, nvtelemetry.WorkloadLimits{})
		if err != nil || len(annotations) != 15 || !bytes.Equal(e.Raw, b) {
			t.Fatal("legacy raw metadata changed", err)
		}
	})

}
