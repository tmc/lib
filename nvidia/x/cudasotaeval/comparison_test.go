package cudasotaeval_test

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudasotaeval"
	"strings"
	"testing"
)

func ExampleCompareTool() {
	p := cudasotaeval.DefaultProtocol(cudasotaeval.PhaseQualification)
	tool := cudasotaeval.ToolSpec{Name: "fixture", Version: "1", Configuration: "cpu-control", Lineage: "fixture", Support: "unknown"}
	p.Tools = []cudasotaeval.ToolSpec{tool}
	p.PropertyLimits = map[string]float64{"attribution": 0}
	_ = p.Freeze()
	receipt := cudasotaeval.ToolReceipt{Tool: tool, ProtocolDigest: p.FrozenDigest}
	data, _ := json.Marshal(receipt)
	raw := string(data)
	receipt.RawDigest = fmt.Sprintf("%x", sha256.Sum256(data))
	gates, _ := cudasotaeval.CompareTool(p, receipt, strings.NewReader(raw))
	fmt.Println(gates[0].Property, gates[0].Status)
	// Output: attribution unknown
}

func TestComparisonPreservesUnknownAndRejectsDrift(t *testing.T) {
	p := cudasotaeval.DefaultProtocol(cudasotaeval.PhaseQualification)
	tool := cudasotaeval.ToolSpec{Name: "control", Version: "1", Configuration: "fixed", Lineage: "control", Support: "measured"}
	p.Tools = []cudasotaeval.ToolSpec{tool}
	p.PropertyLimits = map[string]float64{"attribution": 0}
	if err := p.Freeze(); err != nil {
		t.Fatal(err)
	}
	raw := ""
	r := cudasotaeval.ToolReceipt{Tool: tool, ProtocolDigest: p.FrozenDigest, Measurements: []cudasotaeval.PropertyMeasurement{{Property: "attribution", Status: "measured", Value: 0}}}
	data, _ := json.Marshal(r)
	raw = string(data)
	r.RawDigest = fmt.Sprintf("%x", sha256.Sum256(data))
	gates, err := cudasotaeval.CompareTool(p, r, strings.NewReader(raw))
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
	gates, err = cudasotaeval.CompareTool(p, r, strings.NewReader(raw))
	if err != nil || gates[0].Status != "unknown" {
		t.Fatalf("synthetic admission %v %v", gates, err)
	}
	r.Tool.Version = "2"
	if _, err := cudasotaeval.CompareTool(p, r, strings.NewReader(raw)); err == nil {
		t.Fatal("admitted tool drift")
	}
	r.Tool = tool
	if _, err := cudasotaeval.CompareTool(p, r, strings.NewReader("changed")); err == nil {
		t.Fatal("admitted artifact drift")
	}
}
