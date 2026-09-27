package cudaschedule_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudaschedule"
)

const CAMPAIGN_PTX = `.version 8.7
.target sm_90
.address_size 64

.visible .entry simple_kernel(
    .param .u64 data
)
{
    .reg .pred %p0;
    .reg .b32 %r0;
    .reg .b64 %rd1;
L_loop:
    atom.global.add.u32 %r0, [%rd1], 1;
    @%p0 bra L_loop;
    ret;
}
`

func TestCampaign_RejectsZeroConfirmRuns(t *testing.T) {
	ctx := context.Background()
	opts := cudaschedule.CampaignOptions{
		PTXSource:   CAMPAIGN_PTX,
		SeedStart:   0,
		SeedEnd:     10,
		ConfirmRuns: 0, // Must be rejected
	}

	_, err := cudaschedule.RunCampaign(ctx, opts)
	if err == nil {
		t.Fatalf("RunCampaign succeeded with ConfirmRuns=0, want error")
	}
	if !strings.Contains(err.Error(), "confirm_runs must be at least 1") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestCampaign_RejectsInvalidSeedRange(t *testing.T) {
	ctx := context.Background()
	opts := cudaschedule.CampaignOptions{
		PTXSource:   CAMPAIGN_PTX,
		SeedStart:   10,
		SeedEnd:     10, // Invalid: start must be less than end
		ConfirmRuns: 1,
	}

	_, err := cudaschedule.RunCampaign(ctx, opts)
	if err == nil {
		t.Fatalf("RunCampaign succeeded with SeedStart >= SeedEnd, want error")
	}
	if !strings.Contains(err.Error(), "invalid seed range") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestCampaign_PendingWithoutDevice(t *testing.T) {
	ctx := context.Background()
	opts := cudaschedule.CampaignOptions{
		PTXSource:   CAMPAIGN_PTX,
		SeedStart:   0,
		SeedEnd:     10,
		ConfirmRuns: 1,
		DeviceProbe: func() bool { return false }, // Explicitly no device
	}

	report, err := cudaschedule.RunCampaign(ctx, opts)
	if !errors.Is(err, cudaerrors.ErrHardwarePending) {
		t.Fatalf("RunCampaign without device returned %v, want cudaerrors.ErrHardwarePending", err)
	}
	if report != nil && len(report.Seeds) > 0 {
		t.Fatalf("expected no verdicts reported when device is pending, got %d seeds", len(report.Seeds))
	}
}

func TestCampaign_PluggableRunnerSweep(t *testing.T) {
	ctx := context.Background()

	// Seed 2 causes Mismatch; others Pass
	runner := func(ctx context.Context, seed uint64, ptx string) (cudaschedule.RunRecord, error) {
		if strings.Contains(ptx, "nanosleep.u32") && strings.Contains(ptx, "seed=2") {
			return cudaschedule.RunRecord{
				Exited:   true,
				ExitCode: 0,
				Stdout:   "MISMATCH detected in race interleaving\n",
			}, nil
		}
		return cudaschedule.RunRecord{
			Exited:   true,
			ExitCode: 0,
			Stdout:   "PASS: all checks passed\n",
		}, nil
	}

	opts := cudaschedule.CampaignOptions{
		PTXSource:   CAMPAIGN_PTX,
		SeedStart:   0,
		SeedEnd:     5,
		Intensity:   1.0,
		ConfirmRuns: 3,
		RunVariant:  runner,
	}

	report, err := cudaschedule.RunCampaign(ctx, opts)
	if err != nil {
		t.Fatalf("RunCampaign failed: %v", err)
	}

	if len(report.Seeds) != 5 {
		t.Fatalf("report contains %d seeds, want 5", len(report.Seeds))
	}

	findingCount := report.FindingCount()
	if findingCount != 1 {
		t.Fatalf("report.FindingCount() = %d, want 1", findingCount)
	}

	// Verify seed 2 result
	seed2 := report.Seeds[2]
	if seed2.Verdict != cudaschedule.VerdictMismatch {
		t.Errorf("seed 2 verdict = %v, want Mismatch", seed2.Verdict)
	}
	if seed2.Confirmation == nil {
		t.Fatalf("seed 2 missing confirmation summary")
	}
	if !seed2.Confirmation.Confirmed {
		t.Errorf("seed 2 expected to be confirmed across 3 confirmation runs")
	}
	if seed2.Confirmation.Findings != 3 {
		t.Errorf("seed 2 findings = %d, want 3", seed2.Confirmation.Findings)
	}
}
