package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudaprof/counters"
	"github.com/tmc/lib/nvidia/x/cudaprof/iteration"
	"github.com/tmc/lib/nvidia/x/cudaprof/shape"
	"github.com/tmc/lib/nvidia/x/cuptirange"
)

func counterReceiptScratch(t *testing.T) string {
	t.Helper()
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	base := filepath.Join(home, "tmp")
	if err = os.MkdirAll(base, 0700); err != nil {
		t.Fatal(err)
	}
	dir, err := os.MkdirTemp(base, "counter-receipt-test-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	return dir
}
func writeCounterJSON(t *testing.T, path string, value any) {
	t.Helper()
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	if err = os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
}
func testCounterIdentity() shape.Identity {
	return shape.Identity{Phase: "decode", Token: 0, Operation: "gemv", DType: "float32", CacheLength: 32, Grid: [3]int{1, 1, 1}, Block: [3]int{32, 1, 1}}
}
func testCounterShapeSource() counterShapeSource {
	id := testCounterIdentity()
	launch := shape.Launch{Activity: 0, Journal: "j", RawSymbol: "gemv<float>", Start: 1, Duration: 2, Identity: &id, APIStatus: "matched", MetadataStatus: "matched"}
	arm := counterShapeArm{SHA256: strings.Repeat("a", 64), Completeness: "passed", Stationarity: iteration.Gate{Failures: []string{"stationarity failed"}}}
	return counterShapeSource{Version: 1, Domain: "driver", Arms: []counterShapeArm{arm, arm}, Shape: shape.Report{Version: 1, Pass: true, Groups: []shape.Group{{Identity: id, Base: []shape.Launch{launch}, Target: []shape.Launch{launch}}}}}
}
func TestCounterReceiptSelect(t *testing.T) {
	dir := counterReceiptScratch(t)
	idPath := filepath.Join(dir, "id.json")
	shapePath := filepath.Join(dir, "shape.json")
	writeCounterJSON(t, idPath, testCounterIdentity())
	for _, tt := range []struct {
		name    string
		edit    func(*counterShapeSource)
		wantErr bool
	}{
		{"qualified historical selection", func(*counterShapeSource) {}, false},
		{"unqualified", func(s *counterShapeSource) { s.Shape.Pass = false }, true},
		{"ambiguous shape", func(s *counterShapeSource) { s.Shape.Groups = append(s.Shape.Groups, s.Shape.Groups[0]) }, true},
		{"ambiguous launch", func(s *counterShapeSource) {
			s.Shape.Groups[0].Base = append(s.Shape.Groups[0].Base, s.Shape.Groups[0].Base[0])
		}, true},
		{"incomplete capture", func(s *counterShapeSource) { s.Arms[0].Completeness = "partial" }, true},
		{"missing capture hash", func(s *counterShapeSource) { s.Arms[0].SHA256 = "" }, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			source := testCounterShapeSource()
			tt.edit(&source)
			writeCounterJSON(t, shapePath, source)
			var out bytes.Buffer
			err := runCounterReceipt([]string{"select", "-shape", shapePath, "-identity", idPath}, &out)
			if (err != nil) != tt.wantErr {
				t.Fatalf("error=%v", err)
			}
			var report counterSelectionReport
			if err = json.Unmarshal(out.Bytes(), &report); err != nil {
				t.Fatal(err)
			}
			if (report.Selection == nil) != tt.wantErr || len(report.ShapeFileSHA256) != 64 || len(report.Source.Arms[0].Stationarity.Failures) != 1 {
				t.Fatalf("report=%s", out.String())
			}
		})
	}
}
func testNativeCounterReceipt() counters.Receipt {
	value := 10.0
	return counters.Receipt{Identity: testCounterIdentity(), CaptureSHA256: strings.Repeat("a", 64), Journal: "j", RawSymbol: "gemv<float>", Device: "uuid", Chip: "chip", DriverVersion: "driver", ToolVersion: "tool", Replay: "user; state restored", ReplayComplete: true, Properties: cuptirange.MetricProperties{Name: "metric", Description: "native definition", Dimension: "byte", HardwareUnit: "sm", Type: cupti.MetricTypeCounter, Scope: cupti.MetricCollectionScopeContext, CUPTIVersion: 130400}, Value: &value, Overhead: "unmeasured"}
}
func TestCounterReceiptCompare(t *testing.T) {
	dir := counterReceiptScratch(t)
	basePath := filepath.Join(dir, "base.json")
	targetPath := filepath.Join(dir, "target.json")
	writeCounterJSON(t, basePath, testNativeCounterReceipt())
	for _, tt := range []struct {
		name    string
		edit    func(*counters.Receipt)
		wantErr bool
	}{
		{"compatible", func(r *counters.Receipt) { v := 20.0; r.Value = &v }, false},
		{"mismatched native definition", func(r *counters.Receipt) { r.Properties.Dimension = "cycle" }, true},
		{"native failure", func(r *counters.Receipt) { r.Failure = "permission denied" }, true},
		{"incomplete replay", func(r *counters.Receipt) { r.ReplayComplete = false }, true},
	} {
		t.Run(tt.name, func(t *testing.T) {
			target := testNativeCounterReceipt()
			tt.edit(&target)
			writeCounterJSON(t, targetPath, target)
			var out bytes.Buffer
			err := runCounterReceipt([]string{"compare", basePath, targetPath}, &out)
			if (err != nil) != tt.wantErr {
				t.Fatalf("error=%v", err)
			}
			var report struct {
				Comparison counters.Comparison `json:"comparison"`
			}
			if err = json.Unmarshal(out.Bytes(), &report); err != nil {
				t.Fatal(err)
			}
			if report.Comparison.Compatible == tt.wantErr || (report.Comparison.Delta == nil) != tt.wantErr {
				t.Fatalf("comparison=%s", out.String())
			}
		})
	}
}
func TestCounterIdentityRequiresExplicitZeroFields(t *testing.T) {
	var id shape.Identity
	for _, data := range []string{`{"phase":"decode","operation":"gemv","dtype":"f32","grid":[1,1,1],"block":[32,1,1]}`, `{"phase":"decode","token":null,"operation":"gemv","dtype":"f32","cache_length":0,"grid":[1,1,1],"block":[32,1,1]}`} {
		if err := decodeCounterIdentity([]byte(data), &id); err == nil {
			t.Fatal("missing explicit token/cache length accepted")
		}
	}
}

func TestCounterJSONRejectsAmbiguousInput(t *testing.T) {
	for _, data := range []string{`{"phase":"decode","phase":"prefill"}`, `{"phase":"decode","Phase":"prefill"}`, `{"phase":"decode"} {}`, `{"typo":"decode"}`} {
		var identity shape.Identity
		if err := decodeCounterJSON([]byte(data), &identity); err == nil {
			t.Fatalf("accepted %s", data)
		}
	}
}
