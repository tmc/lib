package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/isolation"
)

func TestCaptureIsolationRoundTrip(t *testing.T) {
	b := isolation.Binding{Session: "capture", Process: isolation.Process{Host: "host", PID: 1, Start: "boot:123"}, GPUs: []string{"GPU-a"}, Window: isolation.Window{Domain: "CLOCK_MONOTONIC", Start: 100, End: 200}}
	r := isolation.Receipt{Version: 1, Method: "nvml-process-residency", Binding: b, Window: b.Window, Outside: []isolation.Observation{{Process: isolation.Process{Host: "host", PID: 99, Start: "boot:456"}, GPU: "GPU-a", First: 150, Last: 150}}}
	raw, _ := json.Marshal(r)
	result := captureResult{Session: "capture", IsolationBinding: b, isolationRaw: raw}
	receipt, hash, _ := isolation.Read(bytes.NewReader(raw))
	out := isolation.Evaluate(receipt, hash, b)
	result.Isolation = &out
	metadata, err := json.Marshal(struct {
		Status captureStatus `json:"status"`
	}{compactCaptureStatus(result)})
	if err != nil {
		t.Fatal(err)
	}
	source := &cudacapture.Capture{CodeUnavailable: map[string]string{"source": "fixture", "ptx": "fixture", "cubin": "fixture", "sass": "fixture"}, Profile: &profile.Profile{SampleType: []*profile.ValueType{{Type: "kernel_time", Unit: "ns"}}}, Files: map[string][]byte{"capture-status.json": metadata, isolation.Attachment: raw}}
	var packed bytes.Buffer
	if err := cudacapture.Write(&packed, source); err != nil {
		t.Fatal(err)
	}
	cap, err := cudacapture.Read(&packed)
	if err != nil {
		t.Fatal(err)
	}
	got := captureIsolation(cap)
	if got.State != "contaminated" || got.SHA256 != hash {
		t.Fatalf("%+v", got)
	}
	if !bytes.Equal(cap.Files[isolation.Attachment], raw) {
		t.Fatal("receipt bytes changed")
	}
	cap.Files[isolation.Attachment] = append(raw, []byte(" {}")...)
	if got := captureIsolation(cap); got.State != "unchecked" {
		t.Fatal(got)
	}
}
func TestCaptureIsolationInvalidRetained(t *testing.T) {
	path := filepath.Join(t.TempDir(), "receipt.json")
	raw := []byte("invalid receipt")
	if err := os.WriteFile(path, raw, 0600); err != nil {
		t.Fatal(err)
	}
	var r captureResult
	attachCaptureIsolation(&r, path)
	if r.Isolation.State != "unchecked" || !bytes.Equal(raw, r.isolationRaw) {
		t.Fatal(r.Isolation)
	}
}
func TestCaptureProcessGeneration(t *testing.T) {
	p := captureProcessIdentity(os.Getpid())
	if p.PID != os.Getpid() || p.Start == "" {
		t.Skip("process generation unavailable")
	}
	owned, known := isolationDescendant(os.Getpid(), p)
	if !known || !owned {
		t.Fatal(owned, known)
	}
	p.Start = "other"
	owned, known = isolationDescendant(os.Getpid(), p)
	if !known || owned {
		t.Fatal(owned, known)
	}
}

func TestCaptureIsolationObserverPreserved(t *testing.T) {
	r := captureResult{Isolation: &isolation.Result{Version: 1, State: "contaminated", Reasons: []string{"outside"}}, isolationRaw: []byte("receipt")}
	attachCaptureIsolation(&r, "")
	if r.Isolation.State != "contaminated" || string(r.isolationRaw) != "receipt" {
		t.Fatal(r.Isolation)
	}
}
func TestIsolationSampler(t *testing.T) {
	parent := captureProcessIdentity(os.Getppid())
	if parent.Start == "" {
		t.Skip("process generations unavailable")
	}
	owner := captureProcessIdentity(os.Getpid())
	w := isolationWatch{owner: owner, receipt: isolation.Receipt{Version: 1, OwnershipResolved: true}}
	w.query = func(query string) ([][]string, error) {
		if query == "--query-gpu=uuid" {
			return [][]string{{"GPU-fixture"}}, nil
		}
		return [][]string{{"GPU-fixture", strconv.Itoa(parent.PID), "5"}}, nil
	}
	w.sample()
	if len(w.receipt.Outside) != 1 || w.receipt.Outside[0].ResidentBytes != 5<<20 || w.receipt.Outside[0].QueryWindow == nil {
		t.Fatalf("%+v", w.receipt)
	}
	w.query = func(string) ([][]string, error) { return nil, fmt.Errorf("fixture unavailable") }
	w.sample()
	if len(w.receipt.Outside) != 1 || len(w.receipt.Errors) == 0 {
		t.Fatal("query failure erased contradiction")
	}
}
