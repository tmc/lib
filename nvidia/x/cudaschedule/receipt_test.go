package cudaschedule_test

import (
	"context"
	"os/exec"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaschedule"
)

// runSyntheticCampaign gives the existing host-only fake workloads explicit
// synthetic load receipts. It models a receipt, not CUDA execution. Receipt
// refusal tests call RunCampaign directly without this adapter.
func runSyntheticCampaign(ctx context.Context, opts cudaschedule.CampaignOptions) (*cudaschedule.CampaignReport, error) {
	if opts.RunVariant != nil {
		runner := opts.RunVariant
		opts.RunVariant = func(ctx context.Context, seed uint64, pristine bool, ptx string) (cudaschedule.RunRecord, error) {
			rec, err := runner(ctx, seed, pristine, ptx)
			if rec.PTXDigest == "" {
				rec.PTXDigest = testPTXDigest(ptx)
			}
			return rec, err
		}
	} else {
		build := opts.BuildVariant
		opts.BuildVariant = func(ctx context.Context, seed uint64, pristine bool, path string) (string, []string, error) {
			executable, args := opts.Executable, opts.Args
			if build != nil {
				var err error
				executable, args, err = build(ctx, seed, pristine, path)
				if err != nil {
					return "", nil, err
				}
			}
			if executable == "" {
				return "", nil, nil
			}
			if _, err := exec.LookPath(executable); err != nil {
				return executable, args, nil
			}
			// This synthetic shell workload reads the actual supplied file in
			// its child process. It does not receive a digest from the controller.
			const script = `for path do :; done
if command -v sha256sum >/dev/null 2>&1; then
 digest=$(sha256sum "$path")
else
 digest=$(shasum -a 256 "$path")
fi
digest=${digest%% *}
printf 'CUDASCHEDULE_LOADED_PTX_SHA256:%s\n' "$digest"
exec "$@"`
			wrapped := []string{"-c", script, "synthetic-receipt", executable}
			wrapped = append(wrapped, args...)
			return "/bin/sh", wrapped, nil
		}
	}
	return cudaschedule.RunCampaign(ctx, opts)
}

// classifySyntheticRecord preserves the legacy outcome discriminator tests
// using explicitly synthetic evidence for their synthetic recorded workloads.
func classifySyntheticRecord(rec cudaschedule.RunRecord) cudaschedule.Verdict {
	rec = syntheticEvidence(rec)
	return cudaschedule.Classify(rec)
}

func syntheticEvidence(rec cudaschedule.RunRecord) cudaschedule.RunRecord {
	if rec.PTXDigest == "" && rec.ExpectedPTXDigest == "" {
		rec.PTXDigest = strings.Repeat("1", 64)
		rec.ExpectedPTXDigest = rec.PTXDigest
		rec.PTXDigestSource = "runner_receipt"
	}
	if rec.Pristine != nil {
		nested := syntheticEvidence(*rec.Pristine)
		rec.Pristine = &nested
	}
	if rec.Baseline != nil {
		nested := syntheticEvidence(*rec.Baseline)
		rec.Baseline = &nested
	}
	return rec
}
