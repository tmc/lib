package main

import (
	"bytes"
	"errors"
	"io"
	"strings"
	"testing"
)

// healthy is a synthetic row shaped after the retained post-reboot GB10
// baseline: the CUDA probe completes, and
// the link and driver-model fields still read as they did when the device
// was refusing to initialize. It must qualify, or the gate refuses a
// working machine.
func healthy() map[string]string {
	return map[string]string{
		"name":                                  "NVIDIA GB10",
		"driver_version":                        "580.173.02",
		"pci.bus_id":                            "000f:01:00.0",
		"pcie.link.gen.current":                 "1",
		"pcie.link.gen.max":                     "1",
		"pcie.link.width.current":               "1",
		"pcie.link.width.max":                   "16",
		"ecc.mode.current":                      "[N/A]",
		"ecc.errors.uncorrected.volatile.total": "[N/A]",
	}
}

// eccField builds a healthy row carrying one ECC error-count value. The
// field is the only one where an unreadable value and an unsupported
// feature must be told apart: an integrated part reports no ECC on a good
// day, but ERR! there is a device failing to answer.
func eccField(value string) map[string]string {
	row := healthy()
	row["ecc.errors.uncorrected.volatile.total"] = value
	return row
}

func goodProbe() cudaProbe {
	return cudaProbe{InitOK: true, DeviceCount: 1, Name: "NVIDIA GB10", Major: 12, Minor: 1, TotalMem: 119 << 30}
}

func TestJudge(t *testing.T) {
	degraded := healthy()
	degraded["name"] = "Unknown"
	errored := healthy()
	errored["driver_version"] = "ERR!"
	ecc := eccField("3")

	tests := []struct {
		name    string
		gpus    []map[string]string
		probe   cudaProbe
		exit    int
		smiErr  string
		expect  expectation
		refusal string // substring of the first refusal; empty means it qualifies
	}{
		{name: "healthy", gpus: []map[string]string{healthy()}, probe: goodProbe()},
		{
			name:    "expected width refuses the narrow link",
			gpus:    []map[string]string{healthy()},
			probe:   goodProbe(),
			expect:  expectation{LinkWidth: 16},
			refusal: "link width 1 below expected 16",
		},
		{
			name:    "expected gen refuses the old link",
			gpus:    []map[string]string{healthy()},
			probe:   goodProbe(),
			expect:  expectation{LinkGen: 5},
			refusal: "link gen 1 below expected 5",
		},
		{
			name:    "required field Unknown",
			gpus:    []map[string]string{degraded},
			probe:   goodProbe(),
			refusal: `name is "Unknown"`,
		},
		{
			name:    "required field ERR!",
			gpus:    []map[string]string{errored},
			probe:   goodProbe(),
			refusal: `driver_version is "ERR!"`,
		},
		{
			name:    "ECC field is ERR!",
			gpus:    []map[string]string{eccField("ERR!")},
			probe:   goodProbe(),
			refusal: `is "ERR!", neither a count nor an unsupported marker`,
		},
		{
			name:    "ECC field is Unknown",
			gpus:    []map[string]string{eccField("Unknown")},
			probe:   goodProbe(),
			refusal: "neither a count nor an unsupported marker",
		},
		{
			name:    "ECC field is prose",
			gpus:    []map[string]string{eccField("no idea")},
			probe:   goodProbe(),
			refusal: "neither a count nor an unsupported marker",
		},
		{
			name:    "ECC field is empty",
			gpus:    []map[string]string{eccField("")},
			probe:   goodProbe(),
			refusal: "neither a count nor an unsupported marker",
		},
		{name: "ECC not supported", gpus: []map[string]string{eccField("[Not Supported]")}, probe: goodProbe()},
		{name: "ECC N/A", gpus: []map[string]string{eccField("N/A")}, probe: goodProbe()},
		{name: "ECC disabled", gpus: []map[string]string{eccField("Disabled")}, probe: goodProbe()},
		{name: "ECC zero errors", gpus: []map[string]string{eccField("0")}, probe: goodProbe()},
		{
			name:    "uncorrected ECC errors",
			gpus:    []map[string]string{ecc},
			probe:   goodProbe(),
			refusal: "3 uncorrected ECC errors",
		},
		{
			name:    "listing does not rescue a failed init",
			gpus:    []map[string]string{healthy()},
			probe:   cudaProbe{InitError: "CUDA_ERROR_NO_DEVICE"},
			refusal: "driver did not initialize",
		},
		{
			name:    "no devices",
			gpus:    []map[string]string{healthy()},
			probe:   cudaProbe{InitOK: true},
			refusal: "reports no devices",
		},
		{
			name:    "device stopped answering",
			gpus:    []map[string]string{healthy()},
			probe:   cudaProbe{InitOK: true, DeviceCount: 1, ProbeError: "name: CUDA_ERROR_INVALID_DEVICE"},
			refusal: "did not answer",
		},
		{
			name:    "zero memory",
			gpus:    []map[string]string{healthy()},
			probe:   cudaProbe{InitOK: true, DeviceCount: 1, Name: "NVIDIA GB10"},
			refusal: "zero memory",
		},
		{
			name:    "nvidia-smi failed",
			probe:   goodProbe(),
			exit:    9,
			smiErr:  "NVML init failure",
			refusal: "nvidia-smi failed: exit 9",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := judge(tt.gpus, tt.probe, tt.exit, tt.smiErr, tt.expect)
			if tt.refusal == "" {
				if len(got) != 0 {
					t.Fatalf("qualified device refused: %v", got)
				}
				return
			}
			if len(got) == 0 {
				t.Fatalf("no refusal, want one containing %q", tt.refusal)
			}
			if !contains(got, tt.refusal) {
				t.Fatalf("refusals %v, want one containing %q", got, tt.refusal)
			}
		})
	}
}

func contains(refusals []string, want string) bool {
	for _, r := range refusals {
		if strings.Contains(r, want) {
			return true
		}
	}
	return false
}

func TestEccErrors(t *testing.T) {
	for _, value := range []string{"[N/A]", "N/A", "n/a", "[Not Supported]", "Disabled"} {
		if state, _ := eccErrors(eccField(value)); state != eccUnsupported {
			t.Errorf("%q classified as %v, want unsupported", value, state)
		}
	}
	for _, value := range []string{"ERR!", "Unknown", "no idea", ""} {
		if state, _ := eccErrors(eccField(value)); state != eccMalformed {
			t.Errorf("%q classified as %v, want malformed", value, state)
		}
	}
	if state, n := eccErrors(eccField("7")); state != eccCount || n != 7 {
		t.Errorf("count = %v %d, want eccCount 7", state, n)
	}
}

func TestParseSMI(t *testing.T) {
	row := "NVIDIA GB10, 580.173.02, 000f:01:00.0, 1, 1, 1, 16, [N/A], [N/A]"
	gpus, err := parseSMI(row + "\n")
	if err != nil {
		t.Fatal(err)
	}
	if len(gpus) != 1 {
		t.Fatalf("got %d gpus, want 1", len(gpus))
	}
	if gpus[0]["pci.bus_id"] != "000f:01:00.0" {
		t.Fatalf("bus id %q", gpus[0]["pci.bus_id"])
	}
	if gpus[0]["pcie.link.width.max"] != "16" {
		t.Fatalf("link width max %q", gpus[0]["pcie.link.width.max"])
	}

	if _, err := parseSMI(row + "\n" + row + "\n"); err != nil {
		t.Fatalf("two identical rows: %v", err)
	}
	if _, err := parseSMI("NVIDIA GB10, 580.173.02\n"); err == nil {
		t.Fatal("short row accepted")
	}
	if _, err := parseSMI("\n \n"); err == nil {
		t.Fatal("empty output accepted")
	}
}

func TestIsUnusable(t *testing.T) {
	for _, s := range []string{"Unknown", "unknown", "ERR!", "N/A", "[N/A]", "[Not Supported]", "", "  "} {
		if !isUnusable(s) {
			t.Errorf("%q accepted as usable", s)
		}
	}
	for _, s := range []string{"NVIDIA GB10", "580.173.02", "0"} {
		if isUnusable(s) {
			t.Errorf("%q rejected as unusable", s)
		}
	}
}

func TestInteger(t *testing.T) {
	row := map[string]string{"a": "12", "b": "[N/A]"}
	if n, ok := integer(row, "a"); !ok || n != 12 {
		t.Fatalf("a = %d %v", n, ok)
	}
	if _, ok := integer(row, "b"); ok {
		t.Fatal("[N/A] parsed as an integer")
	}
	if _, ok := integer(row, "missing"); ok {
		t.Fatal("missing field parsed as an integer")
	}
}

// failingWriter stands in for a full or closed output sink.
type failingWriter struct{ err error }

func (w failingWriter) Write(p []byte) (int, error) { return 0, w.err }

type shortWriter struct{}

func (shortWriter) Write(p []byte) (int, error) { return len(p) - 1, nil }

func TestWriteAll(t *testing.T) {
	if err := writeAll(failingWriter{err: errors.New("no space left on device")}, []byte("x")); err == nil {
		t.Fatal("a failed write reported success")
	}
	if err := writeAll(shortWriter{}, []byte("xy")); err != io.ErrShortWrite {
		t.Fatalf("short write reported %v, want ErrShortWrite", err)
	}
	var buf bytes.Buffer
	if err := writeAll(&buf, []byte("xy")); err != nil || buf.String() != "xy" {
		t.Fatalf("good write: %v %q", err, buf.String())
	}
}
