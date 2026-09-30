package tracecontract

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/perfetto"
)

func TestSourceIdentity(t *testing.T) {
	index := uint64(0)
	good := Source{CaptureSHA256: strings.Repeat("0", 64), Journal: "j", Kind: "activity", Index: &index}
	for _, tc := range []struct {
		name   string
		change func(*Source)
	}{{"missing hash", func(s *Source) { s.CaptureSHA256 = "" }}, {"missing kind", func(s *Source) { s.Kind = "" }}, {"sequence and index", func(s *Source) { s.Sequence = 1 }}, {"missing ordinal", func(s *Source) { s.Index = nil }}, {"loaded slice id", func(s *Source) { s.Kind = "slice" }}} {
		t.Run(tc.name, func(t *testing.T) {
			s := good
			tc.change(&s)
			if _, err := s.String(); err == nil {
				t.Fatalf("accepted %+v", s)
			}
		})
	}
	if _, err := good.String(); err != nil {
		t.Fatal(err)
	}
}

func TestCanonicalTracks(t *testing.T) {
	track := Track{UUID: 1, Name: "total", Role: "gpu_union", Clock: "j:monotonic", Scope: "token:0", Provenance: "observed", Aggregation: "gpu_union", Canonical: true}
	duplicate := track
	duplicate.UUID = 2
	duplicate.Parent = 1
	if err := ValidateTracks([]Track{track, duplicate}); err == nil {
		t.Fatal("parent/child duplicate canonical total accepted")
	}
	duplicate.Canonical = false
	if err := ValidateTracks([]Track{track, duplicate}); err != nil {
		t.Fatal(err)
	}
	track.Parent = 2
	if err := ValidateTracks([]Track{track, duplicate}); err == nil {
		t.Fatal("parent cycle accepted")
	}
}

func TestReceiptIndependentGates(t *testing.T) {
	r := receiptFixture()
	r.Gates["collector"] = Gate{"passed", "complete native receipts", "collector"}
	data, err := r.JSON()
	if err != nil {
		t.Fatal(err)
	}
	var decoded Receipt
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}
	if decoded.Gates["correctness"].Disposition != "unchecked" {
		t.Fatalf("collector result promoted correctness: %+v", decoded)
	}
	delete(r.Gates, "workload")
	if _, err := r.JSON(); err == nil {
		t.Fatal("missing independent gate accepted")
	}
}

func TestBindingsExactBytes(t *testing.T) {
	bindings, members := bindingFixture()
	original, err := bindings.JSON()
	if err != nil {
		t.Fatal(err)
	}
	bindings.Members[0], bindings.Members[1] = bindings.Members[1], bindings.Members[0]
	reordered, _ := bindings.JSON()
	if !bytes.Equal(original, reordered) {
		t.Fatal("binding encoding depends on member order")
	}
	members[".alignment.json"] = []byte("changed")
	if err := bindings.Verify(members); err == nil {
		t.Fatal("changed alignment accepted")
	}
	bindings, members = bindingFixture()
	members["hidden.txt"] = []byte("unexpected")
	if err := bindings.Verify(members); err == nil {
		t.Fatal("unbound member accepted")
	}
	for _, name := range []string{"../trace", "/trace", `a\trace`, ".cudaprof-bindings.json"} {
		if _, err := Digest(name, "trace", strings.Repeat("0", 64), nil); err == nil {
			t.Fatalf("accepted path %q", name)
		}
	}
}

// This test verifies SQL accessibility using the official importer, not a
// locally decoded approximation of TrackEvent annotations.
func TestSemanticMetadataImport(t *testing.T) {
	processor := os.Getenv("CUDAPROF_TEST_TRACE_PROCESSOR")
	if processor == "" {
		t.Skip("official trace processor not configured")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	dir, err := os.MkdirTemp(filepath.Join(home, "tmp"), "cudaprof-semantic-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)
	path := filepath.Join(dir, "semantic.pftrace")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	writer := perfetto.NewWriter(f)
	device := uint32(0)
	track := Track{UUID: 1, Name: "display name can change", Role: "gpu_union", Arm: "base", CaptureSHA256: strings.Repeat("0", 64), Journal: "j", Device: &device, Scope: "token:0", Level: 0, Clock: "base:j:monotonic", Provenance: "observed", Aggregation: "gpu_union", Canonical: true}
	if err := track.Write(writer, 0); err != nil {
		t.Fatal(err)
	}
	if err := receiptFixture().Write(writer, 1, 0); err != nil {
		t.Fatal(err)
	}
	if err := writer.WriteSliceBegin(1, 10, "GPU union", "summary"); err != nil {
		t.Fatal(err)
	}
	if err := writer.WriteSliceEnd(1, 20); err != nil {
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	query := `SELECT EXTRACT_ARG(arg_set_id,'debug.cudaprof_role') AS role,EXTRACT_ARG(arg_set_id,'debug.cudaprof_device') AS device,EXTRACT_ARG(arg_set_id,'debug.cudaprof_canonical') AS canonical FROM slice WHERE name='cudaprof.track.v1'; SELECT EXTRACT_ARG(arg_set_id,'debug.cudaprof_gate_correctness') AS correctness FROM slice WHERE name='cudaprof.receipt.v1'; SELECT sum(value) AS errors FROM stats WHERE severity='error';`
	output, err := exec.Command(processor, "query", path, query).CombinedOutput()
	if err != nil {
		t.Fatalf("import: %v\n%s", err, output)
	}
	if !bytes.Contains(output, []byte(`"gpu_union",0,1`)) || !bytes.Contains(output, []byte(`"unchecked"`)) || !bytes.Contains(output, []byte("\"errors\"\n0")) {
		t.Fatalf("metadata absent from SQL or import errors:\n%s", output)
	}
}
