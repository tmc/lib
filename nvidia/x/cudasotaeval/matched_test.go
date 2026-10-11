package cudasotaeval_test

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudasotaeval"
	"math"
	"strings"
	"testing"
)

func matchProtocol(t testing.TB) cudasotaeval.Protocol {
	t.Helper()
	p := cudasotaeval.DefaultProtocol(cudasotaeval.PhaseQualification)
	p.Tools = []cudasotaeval.ToolSpec{{Name: "a", Version: "1.2", Configuration: "fixed", Lineage: "a", Support: "measured"}, {Name: "b", Version: "2.3", Configuration: "fixed", Lineage: "b", Support: "measured"}}
	p.PropertyLimits = map[string]float64{"cost": 100}
	p.Comparison = &cudasotaeval.ComparisonSpec{Task: "same-task", Candidate: "a", Comparator: "b", Properties: map[string]cudasotaeval.MeasurementScope{"cost": {Unit: "ms", Denominator: 10}}}
	if err := p.Freeze(); err != nil {
		t.Fatal(err)
	}
	return p
}
func matchReceipt(p cudasotaeval.Protocol, arm int) cudasotaeval.ToolReceipt {
	u := 1.0
	return cudasotaeval.ToolReceipt{Task: "same-task", Tool: p.Tools[arm], ProtocolDigest: p.FrozenDigest, Measurements: []cudasotaeval.PropertyMeasurement{{Property: "cost", Status: "measured", Unit: "ms", Denominator: 10, Value: float64(10 + arm*10), Uncertainty: &u}}}
}
func matchRaw(t testing.TB, r *cudasotaeval.ToolReceipt) string {
	t.Helper()
	r.RawDigest = ""
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatal(err)
	}
	r.RawDigest = fmt.Sprintf("%x", sha256.Sum256(b))
	return string(b)
}
func TestMatchedBounds(t *testing.T) {
	p := matchProtocol(t)
	c, b := matchReceipt(p, 0), matchReceipt(p, 1)
	cr, br := matchRaw(t, &c), matchRaw(t, &b)
	out, err := cudasotaeval.CompareTools(p, c, b, strings.NewReader(cr), strings.NewReader(br))
	if err != nil || len(out) != 1 || out[0].Effect != 10 || out[0].Uncertainty != 2 || out[0].Status != "candidate_lower" {
		t.Fatalf("%+v %v", out, err)
	}
	b.Measurements[0].Value = 11
	br = matchRaw(t, &b)
	out, err = cudasotaeval.CompareTools(p, c, b, strings.NewReader(cr), strings.NewReader(br))
	if err != nil || out[0].Status != "overlap" {
		t.Fatalf("%+v %v", out, err)
	}
	b.Measurements = nil
	br = matchRaw(t, &b)
	out, err = cudasotaeval.CompareTools(p, c, b, strings.NewReader(cr), strings.NewReader(br))
	if err != nil || out[0].Status != "withheld" {
		t.Fatalf("%+v %v", out, err)
	}
}
func TestMatchedRefusals(t *testing.T) {
	for _, tt := range []struct {
		name   string
		mutate func(*cudasotaeval.ToolReceipt)
	}{
		{"synthetic", func(r *cudasotaeval.ToolReceipt) { r.Synthetic = true }},
		{"task", func(r *cudasotaeval.ToolReceipt) { r.Task = "other" }},
		{"protocol", func(r *cudasotaeval.ToolReceipt) { r.ProtocolDigest = strings.Repeat("0", 64) }},
		{"version", func(r *cudasotaeval.ToolReceipt) { r.Tool.Version = "unverified" }},
		{"unit", func(r *cudasotaeval.ToolReceipt) { r.Measurements[0].Unit = "seconds" }},
		{"denominator", func(r *cudasotaeval.ToolReceipt) { r.Measurements[0].Denominator = 99 }},
		{"uncertainty", func(r *cudasotaeval.ToolReceipt) { r.Measurements[0].Uncertainty = nil }},
		{"negative bound", func(r *cudasotaeval.ToolReceipt) { v := -1.0; r.Measurements[0].Uncertainty = &v }},
		{"overflow", func(r *cudasotaeval.ToolReceipt) { v := math.MaxFloat64; r.Measurements[0].Uncertainty = &v }},
	} {
		t.Run(tt.name, func(t *testing.T) {
			p := matchProtocol(t)
			c, b := matchReceipt(p, 0), matchReceipt(p, 1)
			tt.mutate(&b)
			if tt.name == "overflow" {
				v := math.MaxFloat64
				c.Measurements[0].Uncertainty = &v
			}
			cr, br := matchRaw(t, &c), matchRaw(t, &b)
			if _, err := cudasotaeval.CompareTools(p, c, b, strings.NewReader(cr), strings.NewReader(br)); err == nil {
				t.Fatal("admitted contradiction")
			}
		})
	}
}
func TestMatchedRegistration(t *testing.T) {
	p := matchProtocol(t)
	p.Comparison.Properties["cost"] = cudasotaeval.MeasurementScope{Unit: "ms", Denominator: 11}
	if p.Validate() == nil {
		t.Fatal("admitted changed frozen denominator")
	}
	for _, version := range []string{"unknown", "unverified", "HEAD", "latest", "synthetic"} {
		p := matchProtocol(t)
		p.Frozen = false
		p.FrozenDigest = ""
		p.Tools[1].Version = version
		if p.Freeze() == nil {
			t.Fatalf("admitted %s", version)
		}
	}
	p = matchProtocol(t)
	p.Frozen = false
	p.FrozenDigest = ""
	p.Tools[1].Lineage = p.Tools[0].Lineage
	if p.Freeze() == nil {
		t.Fatal("admitted shared lineage")
	}
}
func ExampleCompareTools() {
	p := cudasotaeval.DefaultProtocol(cudasotaeval.PhaseQualification)
	// A property-only protocol cannot compare tools without frozen arm scope.
	_, err := cudasotaeval.CompareTools(p, cudasotaeval.ToolReceipt{}, cudasotaeval.ToolReceipt{}, nil, nil)
	fmt.Println(err != nil)
	// Output: true
}

func TestComparisonDuplicateRawFields(t *testing.T) {
	for _, field := range []string{`"task":"same-task",`, `"Task":"same-task",`} {
		p := matchProtocol(t)
		c, b := matchReceipt(p, 0), matchReceipt(p, 1)
		cr, br := matchRaw(t, &c), matchRaw(t, &b)
		br = "{" + field + br[1:]
		b.RawDigest = fmt.Sprintf("%x", sha256.Sum256([]byte(br)))
		if _, err := cudasotaeval.CompareTools(p, c, b, strings.NewReader(cr), strings.NewReader(br)); err == nil {
			t.Fatalf("admitted %s", field)
		}
	}
}
