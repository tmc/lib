package main

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudasotaeval"
)

func compareCommandFixture(t *testing.T) ([]string, string, cudasotaeval.Protocol) {
	t.Helper()
	root := t.TempDir()
	p := cudasotaeval.DefaultProtocol(cudasotaeval.PhaseQualification)
	p.Tools = []cudasotaeval.ToolSpec{{Name: "candidate", Version: "1.0.0", Configuration: "fixed", Lineage: "candidate", Support: "measured"}, {Name: "comparator", Version: "2.0.0", Configuration: "fixed", Lineage: "comparator", Support: "measured"}}
	p.PropertyLimits = map[string]float64{"cost": 100}
	p.Comparison = &cudasotaeval.ComparisonSpec{Task: "same-workload", Candidate: "candidate", Comparator: "comparator", Properties: map[string]cudasotaeval.MeasurementScope{"cost": {Unit: "ms", Denominator: 10}}}
	if err := p.Freeze(); err != nil {
		t.Fatal(err)
	}
	write := func(name string, v any) {
		b, err := json.Marshal(v)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(root, name), b, 0600); err != nil {
			t.Fatal(err)
		}
	}
	write("protocol.json", p)
	uncertainty := 1.0
	for i, name := range []string{"candidate", "comparator"} {
		receipt := cudasotaeval.ToolReceipt{Tool: p.Tools[i], Task: p.Comparison.Task, ProtocolDigest: p.FrozenDigest, Measurements: []cudasotaeval.PropertyMeasurement{{Property: "cost", Status: "measured", Value: float64(10 + 10*i), Unit: "ms", Denominator: 10, Uncertainty: &uncertainty}}}
		write(name+".json", receipt)
	}
	return []string{"compare", "-protocol", filepath.Join(root, "protocol.json"), "-candidate", filepath.Join(root, "candidate.json"), "-comparator", filepath.Join(root, "comparator.json")}, root, p
}
func TestCompareCommandMatchedAndWithheld(t *testing.T) {
	args, root, _ := compareCommandFixture(t)
	var out bytes.Buffer
	if err := run(args, &out, &bytes.Buffer{}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), `"status":"candidate_lower"`) {
		t.Fatalf("matched output: %s", &out)
	}
	b, err := os.ReadFile(filepath.Join(root, "comparator.json"))
	if err != nil {
		t.Fatal(err)
	}
	var receipt cudasotaeval.ToolReceipt
	if err := json.Unmarshal(b, &receipt); err != nil {
		t.Fatal(err)
	}
	receipt.Measurements = nil
	b, _ = json.Marshal(receipt)
	if err := os.WriteFile(filepath.Join(root, "comparator.json"), b, 0600); err != nil {
		t.Fatal(err)
	}
	out.Reset()
	if err := run(args, &out, &bytes.Buffer{}); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(out.String(), `"status":"withheld"`) {
		t.Fatalf("missing output promoted: %s", &out)
	}
}
func TestCompareCommandRefusals(t *testing.T) {
	for _, name := range []string{"protocol mutation", "case alias", "duplicate key", "trailing", "unit", "null", "external digest"} {
		t.Run(name, func(t *testing.T) {
			args, root, p := compareCommandFixture(t)
			path := filepath.Join(root, "protocol.json")
			b, _ := os.ReadFile(path)
			switch name {
			case "protocol mutation":
				p.Comparison.Task = "other"
				b, _ = json.Marshal(p)
			case "case alias":
				b = bytes.Replace(b, []byte(`"name":`), []byte(`"Name":`), 1)
			case "duplicate key":
				b = append([]byte(`{"name":"overridden",`), b[1:]...)
			case "trailing":
				b = append(b, []byte(` {}`)...)
			case "null":
				b = []byte("null")
			case "unit", "external digest":
				path = filepath.Join(root, "comparator.json")
				b, _ = os.ReadFile(path)
				var r cudasotaeval.ToolReceipt
				if err := json.Unmarshal(b, &r); err != nil {
					t.Fatal(err)
				}
				if name == "unit" {
					r.Measurements[0].Unit = "seconds"
				} else {
					r.RawDigest = strings.Repeat("1", 64)
				}
				b, _ = json.Marshal(r)
			}
			if err := os.WriteFile(path, b, 0600); err != nil {
				t.Fatal(err)
			}
			if err := run(args, &bytes.Buffer{}, &bytes.Buffer{}); err == nil {
				t.Fatal("contradiction accepted")
			}
		})
	}
}
func TestCompareCommandShortWrite(t *testing.T) {
	args, _, _ := compareCommandFixture(t)
	if err := run(args, compareShortWriter{}, &bytes.Buffer{}); err != io.ErrShortWrite {
		t.Fatalf("short write: %v", err)
	}
}

type compareShortWriter struct{}

func (compareShortWriter) Write(b []byte) (int, error) { return len(b) - 1, nil }
func TestComparisonInputBounds(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "input")
	if err := os.WriteFile(path, []byte("123456789"), 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := readComparisonFile(path, 8); err == nil {
		t.Fatal("oversized accepted")
	}
	if _, err := readComparisonFile(root, 8); err == nil {
		t.Fatal("directory accepted")
	}
}
