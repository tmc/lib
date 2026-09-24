package trainstat

import (
	"flag"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

// Report records vs for tb under the result name, where "" means tb's own
// row. It sets one attribute per value, keyed trainstat:<unit> or
// trainstat:<name>:<unit>, which the trainstat command reads from
// go test -json or -v output. If tb is a *testing.B and name is "", Report
// also calls ReportMetric so that the benchmark line carries the values.
//
// Report never fails the test on a value: quality is for the trainstat
// gate to judge. An invalid name or unit is a test error.
func Report(tb testing.TB, name string, vs ...Value) {
	tb.Helper()
	prefix := "trainstat:"
	if name != "" {
		if !validUnit(name) {
			tb.Errorf("trainstat: invalid result name %q", name)
			return
		}
		prefix += name + ":"
	}
	b, _ := tb.(*testing.B)
	for _, v := range vs {
		if !validUnit(v.Unit) {
			tb.Errorf("trainstat: invalid unit %q", v.Unit)
			continue
		}
		tb.Attr(prefix+v.Unit, strconv.FormatFloat(v.V, 'g', -1, 64))
		if b != nil && name == "" {
			b.ReportMetric(v.V, v.Unit)
		}
	}
}

// Artifact returns a Writer on tb.ArtifactDir()/trainstat.txt, closed when
// the test and its subtests complete. When the test binary runs without
// -test.artifacts, the directory would be removed, so Artifact returns a
// Writer that discards its output.
func Artifact(tb testing.TB) *Writer {
	tb.Helper()
	if f := flag.Lookup("test.artifacts"); f == nil || f.Value.String() != "true" {
		return NewWriter(io.Discard)
	}
	f, err := os.Create(filepath.Join(tb.ArtifactDir(), "trainstat.txt"))
	if err != nil {
		tb.Fatalf("trainstat: %v", err)
	}
	tb.Cleanup(func() {
		if err := f.Close(); err != nil {
			tb.Errorf("trainstat: %v", err)
		}
	})
	return NewWriter(f)
}
