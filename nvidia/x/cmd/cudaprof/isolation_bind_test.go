package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/isolation"
)

func isolationBindFixture(t *testing.T) (*cudacapture.Capture, isolation.Binding) {
	t.Helper()
	b := isolation.Binding{Session: "session", Process: isolation.Process{Host: "host", PID: 12, Start: "boot:123"}, GPUs: []string{"GPU-a"}, Window: isolation.Window{Domain: "CLOCK_MONOTONIC", Start: 100, End: 200}, CompletionKnown: true}
	status, _ := json.Marshal(struct {
		Status struct {
			Binding isolation.Binding `json:"isolation_binding"`
		} `json:"status"`
	}{Status: struct {
		Binding isolation.Binding `json:"isolation_binding"`
	}{b}})
	cap := &cudacapture.Capture{Profile: &profile.Profile{SampleType: []*profile.ValueType{{Type: "kernel_time", Unit: "ns"}}}, CodeUnavailable: map[string]string{"source": "fixture", "ptx": "fixture", "cubin": "fixture", "sass": "fixture"}, Files: map[string][]byte{"capture-status.json": status}}
	return cap, b
}
func isolationBindReceipt(t *testing.T, b isolation.Binding, outside bool) []byte {
	t.Helper()
	r := isolation.Receipt{Version: 1, Observer: "fixture", ObserverVersion: "1", Method: "nvml-process-residency", Binding: b, Window: b.Window, OwnershipMethod: "live-process-generation-ancestry", OwnershipResolved: true, RequestedInterval: 50, ObservedInterval: 50, Samples: 3, BoundarySamples: true, Devices: []isolation.DeviceCoverage{{GPU: "GPU-a", Samples: 3, First: 100, Last: 200, MaxGap: 50}}}
	if outside {
		r.Outside = []isolation.Observation{{Process: isolation.Process{Host: "host", PID: 99, Start: "boot:456"}, GPU: "GPU-a", First: 150, Last: 150, ResidentBytes: 1}}
	}
	raw, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	return raw
}
func isolationBindDigest(raw []byte) string { h := sha256.Sum256(raw); return hex.EncodeToString(h[:]) }
func TestIsolationBindRetainsContamination(t *testing.T) {
	cap, b := isolationBindFixture(t)
	bad := isolationBindReceipt(t, b, true)
	good := isolationBindReceipt(t, b, false)
	cap.Files[isolation.Attachment] = bad
	if err := bindIsolationCapture(cap, isolationBindDigest([]byte("original")), good, isolationBindDigest(good)); err != nil {
		t.Fatal(err)
	}
	if got := captureIsolation(cap); got.State != "contaminated" {
		t.Fatal(got)
	}
	if !bytes.Equal(cap.Files[isolation.Attachment], bad) {
		t.Fatal("old receipt changed")
	}
	if err := bindIsolationCapture(cap, isolationBindDigest([]byte("second")), good, isolationBindDigest(good)); err != nil {
		t.Fatal(err)
	}
	if got := captureIsolation(cap); got.State != "contaminated" {
		t.Fatal(got)
	}
	var packed bytes.Buffer
	if err := cudacapture.Write(&packed, cap); err != nil {
		t.Fatal(err)
	}
	loaded, err := cudacapture.Read(&packed)
	if err != nil {
		t.Fatal(err)
	}
	if got := captureIsolation(loaded); got.State != "contaminated" {
		t.Fatal(got)
	}
	manifest, _, err := isolation.ReadManifest(bytes.NewReader(loaded.Files[isolation.ManifestAttachment]))
	if err != nil {
		t.Fatal(err)
	}
	loaded.Files[manifest.Sources[1].Attachment] = []byte("tampered")
	if got := captureIsolation(loaded); got.State != "unchecked" {
		t.Fatal(got)
	}
}
func TestIsolationBindCommand(t *testing.T) {
	cap, b := isolationBindFixture(t)
	dir := t.TempDir()
	input := filepath.Join(dir, "original.pprof")
	output := filepath.Join(dir, "bound.pprof")
	receipt := filepath.Join(dir, "receipt.json")
	var packed bytes.Buffer
	if err := cudacapture.Write(&packed, cap); err != nil {
		t.Fatal(err)
	}
	original := append([]byte(nil), packed.Bytes()...)
	if err := os.WriteFile(input, original, 0600); err != nil {
		t.Fatal(err)
	}
	raw := isolationBindReceipt(t, b, false)
	if err := os.WriteFile(receipt, raw, 0600); err != nil {
		t.Fatal(err)
	}
	var result bytes.Buffer
	if err := runIsolationCommand([]string{"bind", "-receipt", receipt, "-o", output, input}, &result); err != nil {
		t.Fatal(err)
	}
	unchanged, err := os.ReadFile(input)
	if err != nil || !bytes.Equal(unchanged, original) {
		t.Fatal("original capture changed")
	}
	loaded, _, err := readAlignmentCapture(output, cudacapture.Limits{})
	if err != nil {
		t.Fatal(err)
	}
	if got := captureIsolation(loaded); got.State != "clear" {
		t.Fatal(got)
	}
	if err := runIsolationCommand([]string{"bind", "-receipt", receipt, "-o", output, input}, &result); err == nil {
		t.Fatal("overwrote existing output")
	}
	alias := filepath.Join(dir, "alias.pprof")
	if err := os.Link(input, alias); err != nil {
		t.Fatal(err)
	}
	if err := isolationOutputPath(alias, input); err == nil {
		t.Fatal("accepted hardlink collision")
	}
	if err := os.WriteFile(receipt, append(raw, []byte(" {}")...), 0600); err != nil {
		t.Fatal(err)
	}
	badOutput := filepath.Join(dir, "bad.pprof")
	if err := runIsolationCommand([]string{"bind", "-receipt", receipt, "-o", badOutput, input}, &result); err == nil {
		t.Fatal("accepted trailing receipt")
	}
	if _, err := os.Stat(badOutput); !os.IsNotExist(err) {
		t.Fatal("published malformed receipt")
	}
}
func TestIsolationBindLegacy(t *testing.T) {
	cap, b := isolationBindFixture(t)
	delete(cap.Files, "capture-status.json")
	raw := isolationBindReceipt(t, b, false)
	if err := bindIsolationCapture(cap, isolationBindDigest([]byte("legacy")), raw, isolationBindDigest(raw)); err != nil {
		t.Fatal(err)
	}
	if got := captureIsolation(cap); got.State != "unchecked" {
		t.Fatal(got)
	}
}

func TestIsolationBindHelp(t *testing.T) {
	for _, args := range [][]string{nil, {"help"}, {"-h"}, {"--help"}} {
		var out bytes.Buffer
		if err := runIsolationCommand(args, &out); err != nil || !bytes.Contains(out.Bytes(), []byte("isolation bind")) {
			t.Fatal(err, out.String())
		}
	}
}
func TestIsolationBindRejectsContradictoryStatus(t *testing.T) {
	for _, raw := range []string{`null`, `{"status":null}`, `{"status":{"isolation_binding":{},"isolation_binding":{}}}`, `{"status":{}} {}`} {
		cap, b := isolationBindFixture(t)
		cap.Files["capture-status.json"] = []byte(raw)
		receipt := isolationBindReceipt(t, b, false)
		if err := bindIsolationCapture(cap, isolationBindDigest([]byte("input")), receipt, isolationBindDigest(receipt)); err == nil {
			t.Fatal("accepted malformed capture status")
		}
	}
}

func TestIsolationBindIdentityMismatch(t *testing.T) {
	cap, b := isolationBindFixture(t)
	b.Process.Start = "reused-generation"
	raw := isolationBindReceipt(t, b, false)
	if err := bindIsolationCapture(cap, isolationBindDigest([]byte("input")), raw, isolationBindDigest(raw)); err == nil {
		t.Fatal("accepted wrong process generation")
	}
}
