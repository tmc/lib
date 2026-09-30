package perfetto_test

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/x/perfetto"
)

func TestTrackDescriptorPreservesUUIDIdentity(t *testing.T) {
	var output bytes.Buffer
	w := perfetto.NewWriter(&output)
	if err := w.WriteTrackDescriptor(1, "group", 0); err != nil {
		t.Fatal(err)
	}
	for _, id := range []uint64{2, 3} {
		if err := w.WriteTrackDescriptor(id, "same name", 1); err != nil {
			t.Fatal(err)
		}
	}
	// TrackDescriptor.sibling_merge_behavior is field 15, varint2 (NONE).
	// Same-name siblings must not fall back to BY_TRACK_NAME, which can
	// collapse caller-defined lanes and lose deeply overlapping events.
	if got := bytes.Count(output.Bytes(), []byte{15 << 3, 2}); got != 3 {
		t.Fatalf("independent descriptors=%d, want3", got)
	}
}

// Set CUDAPROF_TEST_TRACE_PROCESSOR to an official trace_processor_shell to
// verify that same-name siblings stay separate in its imported data model.
func TestTrackDescriptorSameNameImport(t *testing.T) {
	processor := os.Getenv("CUDAPROF_TEST_TRACE_PROCESSOR")
	if processor == "" {
		t.Skip("official trace processor not configured")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	root := filepath.Join(home, "tmp")
	if err := os.MkdirAll(root, 0700); err != nil {
		t.Fatal(err)
	}
	dir, err := os.MkdirTemp(root, "perfetto-siblings-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)
	path := filepath.Join(dir, "siblings.pftrace")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	w := perfetto.NewWriter(f)
	if err := w.WriteTrackDescriptor(1, "parent", 0); err != nil {
		t.Fatal(err)
	}
	for id := uint64(2); id < 602; id++ {
		if err := w.WriteTrackDescriptor(id, "same name", 1); err != nil {
			t.Fatal(err)
		}
		if err := w.WriteInstant(id, 100, "evidence", "observed"); err != nil {
			t.Fatal(err)
		}
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	query := `SELECT count(*) AS events,max(depth) AS depth,count(distinct track_id) AS tracks FROM slice WHERE name='evidence'; SELECT sum(value) AS errors FROM stats WHERE severity='error';`
	output, err := exec.Command(processor, "query", path, query).CombinedOutput()
	if err != nil {
		t.Fatalf("import: %v\n%s", err, output)
	}
	if !bytes.Contains(output, []byte("600,0,600")) || !bytes.Contains(output, []byte("\"errors\"\n0")) {
		t.Fatalf("import lost or nested sibling evidence:\n%s", output)
	}
}
