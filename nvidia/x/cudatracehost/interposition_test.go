package cudatracehost

import (
	"bytes"
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaflightrecorder"
	"github.com/tmc/lib/nvidia/x/cudaprobe"
)

func TestInterpositionProducerRoundTrip(t *testing.T) {
	for _, status := range []string{"", "unknown", "declared_active", "declared_inactive"} {
		t.Run(status, func(t *testing.T) {
			declaration := cudaprobe.Interposition{Status: status}
			if status == "declared_active" || status == "declared_inactive" {
				declaration.Evidence = "operator run manifest"
			}
			cfg := Config{PID: 7, Library: "/selected/libcuda.so", Version: "fixture", Symbols: []string{"cuLaunchKernel", "cuCtxSynchronize"}, Interposition: []InterpositionDeclaration{{Symbol: "cuLaunchKernel", Interposition: declaration}}}
			declarations, err := validateInterposition(cfg)
			if err != nil {
				t.Fatal(err)
			}
			cfg.Interposition[0].Interposition.Status = "declared_inactive"
			id := probeIdentity(cfg, 99, "cuLaunchKernel", "build", declarations["cuLaunchKernel"])
			if declaration.Status == "" {
				declaration.Status = "unknown"
			}
			if id.Process.PID != 7 || id.Process.Generation != 99 || id.Library != cfg.Library || id.Symbol != "cuLaunchKernel" || id.BuildID != "build" || id.Interposition != declaration {
				t.Fatalf("producer identity: %+v", id)
			}
			other := probeIdentity(cfg, 99, "cuCtxSynchronize", "build", declarations["cuCtxSynchronize"])
			if other.Interposition != (cudaprobe.Interposition{Status: "unknown"}) {
				t.Fatal("declaration leaked to unselected symbol")
			}
			c, err := NewCorrelator([]Target{{Name: "launch", Identity: id, Arguments: 0}}, 2)
			if err != nil {
				t.Fatal(err)
			}
			if _, err := c.Observe(Observation{Name: "launch_in", TID: 1, Timestamp: 10}); err != nil {
				t.Fatal(err)
			}
			capture, err := c.Observe(Observation{Name: "launch_out", TID: 1, Timestamp: 20, ReturnKnown: true})
			if err != nil || capture == nil {
				t.Fatalf("capture: %v", err)
			}
			event, err := cudaprobe.Decode(*capture)
			if err != nil {
				t.Fatal(err)
			}
			record, err := event.Record(time.Now())
			if err != nil {
				t.Fatal(err)
			}
			want := status
			if want == "" {
				want = "unknown"
			}
			if record.Labels["probe.interposition"] != want {
				t.Fatalf("status %q", record.Labels["probe.interposition"])
			}
			if _, ok := record.Labels["probe.interposed"]; ok {
				t.Fatal("ambiguous boolean emitted")
			}
			if want != "unknown" && (record.Labels["probe.interposition_source"] != "caller_reported" || record.Labels["probe.interposition_evidence"] != declaration.Evidence) {
				t.Fatal("declaration provenance lost")
			}
			recorder, err := cudaflightrecorder.New(cudaflightrecorder.Config{Capacity: 1})
			if err != nil {
				t.Fatal(err)
			}
			if err := recorder.Start(); err != nil {
				t.Fatal(err)
			}
			defer recorder.Stop()
			recorder.Push(record)
			var b bytes.Buffer
			if _, err := recorder.Snapshot(context.Background(), &b, "interposition"); err != nil {
				t.Fatal(err)
			}
			bundle, err := cudaflightrecorder.ReadBundle(&b)
			if err != nil {
				t.Fatal(err)
			}
			var records []cudaflightrecorder.Record
			if err := json.Unmarshal(bundle.Files["gpu_records.json"], &records); err != nil {
				t.Fatal(err)
			}
			if len(records) != 1 || records[0].Labels["probe.interposition"] != want {
				t.Fatal("bundle lost status")
			}
		})
	}
}

func TestInterpositionAdmissionBeforeAttachment(t *testing.T) {
	valid := InterpositionDeclaration{Symbol: "cuLaunchKernel", Interposition: cudaprobe.Interposition{Status: "declared_active", Evidence: "caller"}}
	for _, tc := range []struct {
		name         string
		declarations []InterpositionDeclaration
	}{
		{"duplicate", []InterpositionDeclaration{valid, valid}},
		{"conflict", []InterpositionDeclaration{valid, {Symbol: valid.Symbol, Interposition: cudaprobe.Interposition{Status: "declared_inactive", Evidence: "caller"}}}},
		{"not selected", []InterpositionDeclaration{{Symbol: "other", Interposition: valid.Interposition}}},
		{"empty evidence", []InterpositionDeclaration{{Symbol: valid.Symbol, Interposition: cudaprobe.Interposition{Status: "declared_active"}}}},
		{"oversize evidence", []InterpositionDeclaration{{Symbol: valid.Symbol, Interposition: cudaprobe.Interposition{Status: "declared_active", Evidence: strings.Repeat("x", 1025)}}}},
		{"invalid", []InterpositionDeclaration{{Symbol: valid.Symbol, Interposition: cudaprobe.Interposition{Status: "verified"}}}},
		{"too many", make([]InterpositionDeclaration, 129)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cfg := Config{Symbols: []string{"cuLaunchKernel"}, Interposition: tc.declarations}
			if _, err := validateInterposition(cfg); err == nil {
				t.Fatal("accepted declaration")
			}
			// This invalid target would otherwise fail the OS/path/PID gate first.
			if _, err := Open(cfg); err == nil || (!strings.Contains(err.Error(), "interposition") && !strings.Contains(err.Error(), "evidence")) {
				t.Fatalf("validation did not precede target access: %v", err)
			}
		})
	}
}

func TestInterpositionManifestSnapshot(t *testing.T) {
	c := &Collector{manifest: Manifest{Interposition: []cudaprobe.Identity{{Symbol: "cuLaunchKernel", Interposition: cudaprobe.Interposition{Status: "declared_active", Evidence: "caller"}}}}}
	m := c.Manifest()
	m.Interposition[0].Interposition.Status = "declared_inactive"
	if c.Manifest().Interposition[0].Interposition.Status != "declared_active" {
		t.Fatal("caller mutated manifest")
	}
}
