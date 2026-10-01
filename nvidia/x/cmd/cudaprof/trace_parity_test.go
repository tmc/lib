package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof/overview"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
	"github.com/tmc/lib/nvidia/x/cudaprof/traceanalysis"
	"github.com/tmc/lib/nvidia/x/perfetto"
)

// Exercise the real journal decoder and exporter, not a hand-written trace.
// This is CPU-only and opt-in with an existing supported local executable.
func TestTraceExportSQLParity(t *testing.T) {
	processor := os.Getenv("CUDAPROF_TRACE_PROCESSOR")
	if processor == "" {
		t.Skip("set CUDAPROF_TRACE_PROCESSOR to local v58.2")
	}
	b := newJournalBuilder()
	b.traceActivity("kernel", "controlled_kernel", 0, 7, 100, 110, 1)
	b.traceActivity("kernel", "controlled_kernel", 0, 7, 105, 115, 2)
	b.traceActivity("memcpy", "", 0, 7, 120, 130, 3)
	journal := b.finalize()
	records := decodeFixture(t, journal)
	native, err := timeline.Decode(map[string][]byte{"capture.journal": journal}, 100, 1<<20)
	if err != nil {
		t.Fatal(err)
	}
	session, err := traceanalysis.New(native, traceanalysis.Options{CollectorComplete: true})
	if err != nil {
		t.Fatal(err)
	}
	var output bytes.Buffer
	identity := traceExportIdentity{sha256: strings.Repeat("a", 64)}
	if _, err := writeTraceOrigin(&output, records, 100, 0, "", identity); err != nil {
		t.Fatal(err)
	}
	// Summary tracks must not multiply raw activity counts or durations.
	if err := writeOverviewTrace(&output, overview.Analyze(native, "driver", nil), 100, 1<<48); err != nil {
		t.Fatal(err)
	}
	withoutReceipt := append([]byte(nil), output.Bytes()...)
	identity.path = "fixture.pprof"
	if err := writeExportReceipt(perfetto.NewWriter(&output), exportReceipt("trace export", identity, records, captureResult{}), 6<<48, 0); err != nil {
		t.Fatal(err)
	}
	reconciled, err := reconcileTraceImport(output.Bytes(), processor, defaultTraceProcessorLimits())
	if err != nil || reconciled.Disposition != "passed" {
		t.Fatalf("reconciliation: %+v %v", reconciled, err)
	}
	var bad bytes.Buffer
	bad.Write(withoutReceipt)
	wrong := exportReceipt("trace export", identity, records, captureResult{})
	wrong.RawDeviceRecords["kernel"]++
	if err := writeExportReceipt(perfetto.NewWriter(&bad), wrong, 6<<48, 0); err != nil {
		t.Fatal(err)
	}
	refused, err := reconcileTraceImport(bad.Bytes(), processor, defaultTraceProcessorLimits())
	if err != nil || refused.Disposition != "failed" {
		t.Fatalf("mismatch: %+v %v", refused, err)
	}
	query := func(name string) [][]string {
		t.Helper()
		sql, err := traceanalysis.SQL(name)
		if err != nil {
			t.Fatal(err)
		}
		receipt, err := queryTraceProcessor(output.Bytes(), processor, sql)
		if err != nil {
			t.Fatal(err)
		}
		return receipt.Rows
	}
	rows := query("kernel-shapes")
	encoded, _ := json.Marshal(rows)
	if len(rows) != 2 {
		t.Fatalf("kernel shape rows: %s", encoded)
	}
	columns := map[string]int{}
	for i, name := range rows[0] {
		columns[name] = i
	}
	if rows[1][columns["launches"]] != "2" || rows[1][columns["sum_ns"]] != "20" {
		t.Fatalf("shape: %s", encoded)
	}
	rows = query("recorded-gaps")
	columns = map[string]int{}
	for i, name := range rows[0] {
		columns[name] = i
	}
	if len(rows) != 3 {
		t.Fatalf("gaps rows: %v", rows)
	}
	want := session.Summary().Windows[0].UnionNS
	found := false
	for _, row := range rows[1:] {
		if row[columns["measurement"]] == "recorded_union" && row[columns["duration_ns"]] == "25" {
			found = true
		}
	}
	if !found || want != 25 {
		t.Fatalf("native union=%d SQL=%v", want, rows)
	}
}

func TestTraceViewResourceOverrides(t *testing.T) {
	path := filepath.Join(t.TempDir(), "input.pftrace")
	if err := os.WriteFile(path, []byte{10, 2, 8, 1}, 0600); err != nil {
		t.Fatal(err)
	}
	for _, run := range []func([]string, *bytes.Buffer) error{
		func(a []string, b *bytes.Buffer) error { return runTraceSummary(a, b) },
		func(a []string, b *bytes.Buffer) error {
			return runTraceQuery(append(a[:len(a)-1], "inventory", a[len(a)-1]), b)
		},
	} {
		for _, bound := range []string{"0", "1"} {
			if err := run([]string{"-input-bytes", bound, path}, new(bytes.Buffer)); err == nil {
				t.Fatalf("accepted input bound %s", bound)
			}
		}
	}
}

func TestTraceSummaryRejectsInputAlias(t *testing.T) {
	input := filepath.Join(t.TempDir(), "input.pprof")
	writeTestCaptureFile(t, input, testCapture(nil, map[string]any{"run_completeness": "partial"}))
	original, err := os.ReadFile(input)
	if err != nil {
		t.Fatal(err)
	}
	aliases := []string{input, filepath.Join(filepath.Dir(input), "hardlink"), filepath.Join(filepath.Dir(input), "symlink")}
	if err := os.Link(input, aliases[1]); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(input, aliases[2]); err != nil {
		t.Fatal(err)
	}
	for _, alias := range aliases {
		if err := runTraceSummary([]string{"-json", "-html", alias, input}, new(bytes.Buffer)); err == nil {
			t.Fatalf("accepted alias %s", alias)
		}
		current, err := os.ReadFile(input)
		if err != nil || !bytes.Equal(current, original) {
			t.Fatalf("input changed: %v", err)
		}
	}
}

func TestTraceCheckRequiresIsolation(t *testing.T) {
	input := filepath.Join(t.TempDir(), "input.pprof")
	writeTestCaptureFile(t, input, testCapture(nil, map[string]any{"run_completeness": "partial"}))
	var output bytes.Buffer
	err := runTraceCheck([]string{"-json", "-require", "isolation", input}, &output)
	if err == nil {
		t.Fatal("missing observer qualified")
	}
	var receipt traceCheckReport
	if err := json.Unmarshal(output.Bytes(), &receipt); err != nil {
		t.Fatal(err)
	}
	if receipt.Checks["isolation"].Disposition != "unavailable" {
		t.Fatalf("isolation: %+v", receipt.Checks["isolation"])
	}
}

func TestTraceQueryOutputBudget(t *testing.T) {
	if err := runTraceQuery([]string{"-list", "-json", "-output-bytes", "1"}, new(bytes.Buffer)); err == nil {
		t.Fatal("output budget ignored")
	}
}
