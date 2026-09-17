package cudarecipe

import "testing"

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
