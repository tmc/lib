package traceanalysis

import (
	"bytes"
	"encoding/csv"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/perfetto"
)

// SQL qualification is opt-in with an existing local executable. No download
// or GPU work occurs. The supported fixture runner is Perfetto v58.2.
func TestSQLParity(t *testing.T) {
	executable := os.Getenv("CUDAPROF_TRACE_PROCESSOR")
	if executable == "" {
		t.Skip("set CUDAPROF_TRACE_PROCESSOR to a local v58.2 executable")
	}
	version, err := exec.Command(executable, "--version").CombinedOutput()
	if err != nil || !strings.HasPrefix(string(version), "Perfetto v58.2") {
		t.Fatalf("unsupported processor: %s %v", version, err)
	}
	records := timeline.Records{Activities: []timeline.Activity{{Journal: "p", Sequence: 1, Subtype: "kernel", Name: "kernel", Start: 10, End: 20}, {Journal: "p", Sequence: 2, Subtype: "kernel", Name: "kernel", Start: 15, End: 25}, {Journal: "p", Sequence: 3, Subtype: "memcpy", Start: 30, End: 40}}}
	session, err := New(records, Options{CollectorComplete: true})
	if err != nil {
		t.Fatal(err)
	}
	var trace bytes.Buffer
	writer := perfetto.NewWriter(&trace)
	device := uint32(0)
	for _, id := range []uint64{1, 2} {
		track := tracecontract.Track{UUID: id, Name: "misleading summary name", Role: "raw_device", Clock: "cupti-relative", Journal: "p", Device: &device, CaptureSHA256: strings.Repeat("a", 64), Provenance: "observed", Aggregation: "none"}
		if err := track.Write(writer, 0); err != nil {
			t.Fatal(err)
		}
	}
	// An aggregate representation repeats raw work on another track; SQL must
	// exclude it irrespective of this track's convincing display label.
	summary := tracecontract.Track{UUID: 3, Name: "device 0 stream 0", Role: "summary_gpu_union", Clock: "cupti-relative", Journal: "p", Device: &device, CaptureSHA256: strings.Repeat("a", 64), Scope: "capture", Provenance: "observed", Aggregation: "gpu_union", Canonical: true}
	if err := summary.Write(writer, 0); err != nil {
		t.Fatal(err)
	}
	annotation := func(index int) []perfetto.DebugAnnotation {
		a := records.Activities[index]
		return []perfetto.DebugAnnotation{perfetto.NewStringAnnotation("raw_journal", a.Journal), perfetto.NewUintAnnotation("raw_record_sequence", a.Sequence), perfetto.NewStringAnnotation("cudaprof_activity_kind", a.Subtype), perfetto.NewUintAnnotation("device_id", 0), perfetto.NewStringAnnotation("raw_symbol", a.Name)}
	}
	begin := func(track, at uint64, index int) {
		if err := writer.WriteSliceBeginWithAnnotations(track, at, "kernel", "cuda", annotation(index), nil); err != nil {
			t.Fatal(err)
		}
	}
	end := func(track, at uint64) {
		if err := writer.WriteSliceEnd(track, at); err != nil {
			t.Fatal(err)
		}
	}
	begin(1, 10, 0)
	begin(3, 10, 0)
	begin(2, 15, 1)
	end(1, 20)
	end(2, 25)
	end(3, 25)
	begin(1, 30, 2)
	end(1, 40)
	dir := t.TempDir()
	path := filepath.Join(dir, "fixture.pftrace")
	if err := os.WriteFile(path, trace.Bytes(), 0600); err != nil {
		t.Fatal(err)
	}
	module, err := filepath.Abs("sql/cudaprof")
	if err != nil {
		t.Fatal(err)
	}
	query := func(sql string) [][]string {
		t.Helper()
		cmd := exec.Command(executable, "--add-sql-package", module, "query", path, "INCLUDE PERFETTO MODULE cudaprof.views; "+sql)
		var stderr bytes.Buffer
		cmd.Stderr = &stderr
		output, err := cmd.Output()
		if err != nil {
			t.Fatalf("SQL: %v: %s", err, stderr.String())
		}
		rows, err := csv.NewReader(bytes.NewReader(output)).ReadAll()
		if err != nil {
			t.Fatal(err)
		}
		return rows
	}
	rows := query("SELECT recorded_union_ns FROM cudaprof_device_totals_v1")
	want := fmt.Sprint(session.Summary().Windows[0].UnionNS)
	if len(rows) != 2 || rows[1][0] != want {
		t.Fatalf("union rows=%v want %s; tracks=%v; args=%v", rows, want, query("SELECT * FROM cudaprof_track_v1"), query("SELECT key,display_value FROM args LIMIT 25"))
	}
	rows = query("SELECT launches,sum_ns FROM cudaprof_kernel_shapes_v1")
	if len(rows) != 2 || rows[1][0] != "2" || rows[1][1] != "20" {
		t.Fatalf("kernel rows=%v", rows)
	}
	rows = query("SELECT gap_ns FROM cudaprof_recorded_gaps_v1")
	if len(rows) != 2 || rows[1][0] != "5" {
		t.Fatalf("gap rows=%v", rows)
	}
	rows = query("SELECT state FROM cudaprof_delivery_cadence_v1")
	if len(rows) != 2 || rows[1][0] != "unavailable" {
		t.Fatal(rows)
	}
	spec, err := filepath.Abs("sql/summary.textproto")
	if err != nil {
		t.Fatal(err)
	}
	cmd := exec.Command(executable, "--add-sql-package", module, "summarize", "--metrics-v2", "all", path, spec)
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("structured summary: %v %s", err, output)
	}
}
