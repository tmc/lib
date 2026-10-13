package cudainspect

import (
	"encoding/hex"
	"os"
	"strings"
	"testing"
)

func TestReportDeterministicAndUnknownsExplicit(t *testing.T) {
	i, err := NewInspector(".")
	if err != nil {
		t.Fatal(err)
	}
	a, err := i.Inspect([]byte("fixture"), "kernel")
	if err != nil {
		t.Fatal(err)
	}
	b, err := i.Inspect([]byte("fixture"), "kernel")
	if err != nil {
		t.Fatal(err)
	}
	if a.Digest != b.Digest || a.ByteLength != 7 {
		t.Fatalf("nondeterministic report: %+v %+v", a, b)
	}
	if a.RegCountKnown || a.PeakGPRKnown || len(a.Unknowns) == 0 {
		t.Fatalf("unavailable facts were collapsed: %+v", a)
	}
}

func TestInspectELFReportsStaticFacts(t *testing.T) {
	raw, err := os.ReadFile("../cudasass/testdata/tkinfo-ptxas-13.0-sm121.cubin.hex")
	if err != nil {
		t.Fatal(err)
	}
	cubin, err := hex.DecodeString(strings.Join(strings.Fields(string(raw)), ""))
	if err != nil {
		t.Fatal(err)
	}

	i, err := NewInspector(".")
	if err != nil {
		t.Fatal(err)
	}
	report, err := i.Inspect(cubin, "kernel")
	if err != nil {
		t.Fatal(err)
	}
	if report.ELFClass == "unknown" || len(report.Sections) == 0 {
		t.Fatalf("static ELF facts missing: %+v", report)
	}
	if report.Toolchain == "" {
		t.Fatalf("toolchain stamp missing: %+v", report)
	}
	if report.RegCountKnown || report.PeakGPRKnown {
		t.Fatalf("fixture must not invent kernel-specific register facts: %+v", report)
	}
}
