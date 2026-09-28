//go:build native

package cudaschedule_test

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudaschedule"
)

// TestCampaign_FindingReproducesFromSeed runs a caller-provided native harness.
// The executable receives the generated module.ptx path as its last argument;
// it must compile and execute that exact image and print a recognized mismatch
// marker when the seeded finding reproduces.
func TestCampaign_FindingReproducesFromSeed(t *testing.T) {
	if _, err := os.Stat("/dev/nvidiactl"); err != nil {
		t.Skip("no NVIDIA device")
	}
	if os.Getenv("CUDA_SCHEDULE_NATIVE") != "1" {
		t.Skip("set CUDA_SCHEDULE_NATIVE=1 to run the native schedule campaign")
	}
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("native campaign requires GPU_LOCK_HELD; run under gpu-lock")
	}
	executable := os.Getenv("CUDA_SCHEDULE_EXECUTABLE")
	if executable == "" {
		t.Fatal("set CUDA_SCHEDULE_EXECUTABLE to a harness that compiles and runs module.ptx")
	}
	lease, err := cudalock.Acquire(context.Background(), cudalock.WithLabel(t.Name()))
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Release()

	const ptx = `.version 8.7
.target sm_90
.address_size 64
.visible .entry seeded(){
    atom.global.add.u32 %r, [%rd], 1;
    ret;
}
`
	report, err := cudaschedule.RunCampaign(context.Background(), cudaschedule.CampaignOptions{
		PTXSource: ptx, Executable: executable, SeedStart: 17, SeedEnd: 18,
		Intensity: 2, ConfirmRuns: 3, Timeout: 30 * time.Second,
		BuildVariant: func(_ context.Context, _ uint64, _ string) (string, []string, error) {
			return executable, nil, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Seeds) != 1 || report.Seeds[0].Seed != 17 || report.Seeds[0].Confirmation == nil || !report.Seeds[0].Confirmation.Confirmed {
		t.Fatalf("seeded finding did not reproduce: %+v", report.Seeds)
	}
}
