package main

import (
	"io"
	"os"
	"path/filepath"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaprof/overview"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

func exportResourceFixture(count int) timeline.Records {
	records := timeline.Records{Activities: make([]timeline.Activity, count)}
	for i := range count {
		records.Activities[i] = timeline.Activity{Journal: "fixture.journal", Sequence: uint64(i + 1), Subtype: "kernel", Name: "controlled_raw_kernel", Start: int64(3*i + 1), End: int64(3*i + 2)}
	}
	return records
}
func BenchmarkLegacyOverviewExport(b *testing.B) {
	records := exportResourceFixture(10000)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		report := overview.Analyze(records, "driver", nil)
		if err := writeOverviewTrace(io.Discard, report, 1, 1<<48); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkRealRawExport profiles the exact CLI path on an existing capture.
// It collects no new workload and is disabled unless a fixture is supplied.
func BenchmarkRealRawExport(b *testing.B) {
	path := os.Getenv("CUDAPROF_EXPORT_RESOURCE_CAPTURE")
	if path == "" {
		b.Skip("existing resource capture not configured")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		b.Fatal(err)
	}
	dir, err := os.MkdirTemp(filepath.Join(home, "tmp"), "cudaprof-export-resource-")
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { os.RemoveAll(dir) })
	output := filepath.Join(dir, "raw.pftrace")
	b.ReportAllocs()
	for range b.N {
		if err := runExportCommand([]string{"-detail=raw", "-o", output, path}); err != nil {
			b.Fatal(err)
		}
	}
}
