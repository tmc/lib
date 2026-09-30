package isolation

import (
	"encoding/json"
	"strings"
	"testing"
)

func fixture() (*Receipt, Binding) {
	b := Binding{Session: "session", Process: Process{Host: "host", PID: 12, Start: "boot:123"}, GPUs: []string{"GPU-a"}, Window: Window{Domain: "CLOCK_MONOTONIC", Start: 100, End: 200}, CompletionKnown: true}
	r := &Receipt{Version: 1, Observer: "observer", ObserverVersion: "1", Method: "nvml-process-residency", Binding: b, Window: b.Window, OwnershipMethod: "live-process-generation-ancestry", OwnershipResolved: true, RequestedInterval: 50, ObservedInterval: 50, Samples: 3, BoundarySamples: true, Devices: []DeviceCoverage{{GPU: "GPU-a", Samples: 3, First: 100, Last: 200, MaxGap: 50}}}
	return r, b
}
func TestEvaluate(t *testing.T) {
	for _, tt := range []struct {
		name, want string
		change     func(*Receipt, *Binding)
	}{
		{"clear", "clear", func(*Receipt, *Binding) {}},
		{"outside", "contaminated", func(r *Receipt, b *Binding) {
			r.Outside = []Observation{{Process: Process{Host: "host", PID: 99, Start: "boot:456"}, GPU: "GPU-a", First: 150, Last: 150, ResidentBytes: 1}}
		}},
		{"error and outside", "contaminated", func(r *Receipt, b *Binding) {
			r.Errors = []string{"query failed"}
			r.Outside = []Observation{{Process: Process{Host: "host", PID: 99, Start: "boot:456"}, GPU: "GPU-a", First: 150, Last: 150}}
		}},
		{"before admission", "clear", func(r *Receipt, b *Binding) {
			r.Outside = []Observation{{Process: Process{Host: "host", PID: 99, Start: "boot:456"}, GPU: "GPU-a", First: 90, Last: 90, QueryWindow: &Window{Domain: "CLOCK_MONOTONIC", Start: 80, End: 90}}}
		}},
		{"query straddles admission", "unchecked", func(r *Receipt, b *Binding) {
			r.Outside = []Observation{{Process: Process{Host: "host", PID: 99, Start: "boot:456"}, GPU: "GPU-a", First: 110, Last: 110, QueryWindow: &Window{Domain: "CLOCK_MONOTONIC", Start: 90, End: 110}}}
		}},
		{"outside campaign only", "clear", func(r *Receipt, b *Binding) {
			r.Outside = []Observation{{Process: Process{Host: "host", PID: 99, Start: "boot:456"}, GPU: "GPU-a", First: 300, Last: 300}}
		}},
		{"PID reused", "unchecked", func(r *Receipt, b *Binding) { b.Process.Start = "boot:999" }},
		{"clock mismatch", "unchecked", func(r *Receipt, b *Binding) { b.Window.Domain = "wall" }},
		{"unknown completion", "unchecked", func(r *Receipt, b *Binding) { b.CompletionKnown = false }},
		{"ancestry unresolved", "unchecked", func(r *Receipt, b *Binding) { r.OwnershipResolved = false }},
		{"preflight", "unchecked", func(r *Receipt, b *Binding) { r.Samples = 1 }},
		{"observer errors", "unchecked", func(r *Receipt, b *Binding) { r.Errors = []string{"permission denied"} }},
		{"missing device", "unchecked", func(r *Receipt, b *Binding) { b.GPUs = append(b.GPUs, "GPU-b") }},
		{"partial device coverage", "unchecked", func(r *Receipt, b *Binding) { r.Devices = nil }},
		{"boundary gap", "unchecked", func(r *Receipt, b *Binding) { r.Window.Start = 101 }},
		{"uncertain clock", "unchecked", func(r *Receipt, b *Binding) { r.Window.Uncertainty = 1 }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			r, b := fixture()
			tt.change(r, &b)
			got := Evaluate(r, "hash", b)
			if got.State != tt.want {
				t.Fatalf("state=%s reasons=%v", got.State, got.Reasons)
			}
		})
	}
	if got := Evaluate(nil, "", Binding{}); got.State != "unchecked" {
		t.Fatal(got)
	}
}
func TestRead(t *testing.T) {
	r, _ := fixture()
	raw, _ := json.Marshal(r)
	got, hash, err := Read(strings.NewReader(string(raw)))
	if err != nil || hash == "" || got.Version != 1 {
		t.Fatal(got, hash, err)
	}
	for _, raw := range []string{string(raw) + " {}", `{"version":1,"version":1}`, `{"version":2}`, `{"version":1,"unexpected":true}`, strings.Repeat("x", MaxBytes+1)} {
		if _, _, err := Read(strings.NewReader(raw)); err == nil {
			t.Fatal("accepted invalid receipt")
		}
	}
}
func TestRequire(t *testing.T) {
	for _, state := range []string{"clear", "unchecked", "contaminated"} {
		for _, required := range []bool{false, true} {
			err := Require(Result{State: state}, required)
			want := state == "contaminated" || required && state != "clear"
			if (err != nil) != want {
				t.Fatal(state, required, err)
			}
		}
	}
}
