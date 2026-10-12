package cudadiageval_test

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudadiageval"
	"strings"
	"testing"
)

func ExampleCompareTool() {
	p := cudadiageval.DefaultProtocol(cudadiageval.PhaseQualification)
	tool := cudadiageval.ToolSpec{Name: "fixture", Version: "1", Configuration: "cpu-control", Lineage: "fixture", Support: "unknown"}
	p.Tools = []cudadiageval.ToolSpec{tool}
	p.PropertyLimits = map[string]float64{"attribution": 0}
	_ = p.Freeze()
	receipt := cudadiageval.ToolReceipt{Tool: tool, ProtocolDigest: p.FrozenDigest}
	data, _ := json.Marshal(receipt)
	raw := string(data)
	receipt.RawDigest = fmt.Sprintf("%x", sha256.Sum256(data))
	gates, _ := cudadiageval.CompareTool(p, receipt, strings.NewReader(raw))
	fmt.Println(gates[0].Property, gates[0].Status)
	// Output: attribution unknown
}

func TestComparisonPreservesUnknownAndRejectsDrift(t *testing.T) {
	p := cudadiageval.DefaultProtocol(cudadiageval.PhaseQualification)
	tool := cudadiageval.ToolSpec{Name: "control", Version: "1", Configuration: "fixed", Lineage: "control", Support: "measured"}
	p.Tools = []cudadiageval.ToolSpec{tool}
	p.PropertyLimits = map[string]float64{"attribution": 0}
	if err := p.Freeze(); err != nil {
		t.Fatal(err)
	}
	raw := ""
	r := cudadiageval.ToolReceipt{Tool: tool, ProtocolDigest: p.FrozenDigest, Measurements: []cudadiageval.PropertyMeasurement{{Property: "attribution", Status: "measured", Value: 0}}}
	data, _ := json.Marshal(r)
	raw = string(data)
	r.RawDigest = fmt.Sprintf("%x", sha256.Sum256(data))
	gates, err := cudadiageval.CompareTool(p, r, strings.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	if gates[0].Status != "passed" || gates[1].Status != "unknown" {
		t.Fatalf("gates %v", gates)
	}
	r.Synthetic = true
	r.RawDigest = ""
	data, _ = json.Marshal(r)
	raw = string(data)
	r.RawDigest = fmt.Sprintf("%x", sha256.Sum256(data))
	gates, err = cudadiageval.CompareTool(p, r, strings.NewReader(raw))
	if err != nil || gates[0].Status != "unknown" {
		t.Fatalf("synthetic admission %v %v", gates, err)
	}
	r.Tool.Version = "2"
	if _, err := cudadiageval.CompareTool(p, r, strings.NewReader(raw)); err == nil {
		t.Fatal("admitted tool drift")
	}
	r.Tool = tool
	if _, err := cudadiageval.CompareTool(p, r, strings.NewReader("changed")); err == nil {
		t.Fatal("admitted artifact drift")
	}
}
