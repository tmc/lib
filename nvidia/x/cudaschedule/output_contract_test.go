package cudaschedule_test

import (
	"context"
	"encoding/json"
	"math"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudaoutputcoverage"
	"github.com/tmc/lib/nvidia/x/cudaschedule"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

func recordedOutput(kind string) *cudaoutputcoverage.RecordedOutput {
	t := cudaoutputcoverage.TwoFill{
		Contract: cudaoutputcoverage.Contract{Buffer: "out", Elements: 1006, Domain: []cudaoutputcoverage.Range{{Start: 0, End: 1003}}, Kind: cudaoutputcoverage.AtLeastOnce, Deterministic: true},
		FillA:    math.Float32bits(-101), FillB: math.Float32bits(-202),
		Geometry:     cudaoutputcoverage.Geometry{Grid: [3]uint32{4, 1, 1}, Block: [3]uint32{256, 1, 1}},
		SourceDigest: strings.Repeat("a", 64), InputDigest: strings.Repeat("b", 64), Target: "sm_121", MathMode: "precise",
	}
	t.Reference = make([]uint32, 1006)
	t.ReadbackA = make([]uint32, 1006)
	t.ReadbackB = make([]uint32, 1006)
	for i := 0; i < 1003; i++ {
		t.Reference[i] = math.Float32bits(float32(i + 1))
		t.ReadbackA[i], t.ReadbackB[i] = t.Reference[i], t.Reference[i]
	}
	for i := 1003; i < 1006; i++ {
		t.ReadbackA[i], t.ReadbackB[i] = t.FillA, t.FillB
	}
	ordinary := append([]uint32(nil), t.Reference...)
	if kind == "residue" {
		t.Geometry.Grid[0] = 3
		for i := 768; i < 1003; i++ {
			t.ReadbackA[i], t.ReadbackB[i] = t.FillA, t.FillB
		}
	}
	if kind == "wrong_value" {
		ordinary[17] = math.Float32bits(999)
	}
	p := cudatest.CUDAOutputPolicy{Target: t.Target, DType: "fp32", MathMode: t.MathMode, SourceDigest: t.SourceDigest, InputDigest: t.InputDigest, Elements: 1003, NaN: "reject", Infinity: "reject", Zero: "bits"}
	return &cudaoutputcoverage.RecordedOutput{TwoFill: t, Ordinary: ordinary, Policy: p}
}

func TestCampaignOutputRecordedReadbacks(t *testing.T) {
	for _, kind := range []string{"residue", "wrong_value", "healthy"} {
		t.Run(kind, func(t *testing.T) {
			opts := cudaschedule.CampaignOptions{PTXSource: CAMPAIGN_PTX, SeedStart: 1, SeedEnd: 4, Intensity: 2, ConfirmRuns: 2, ArtifactDir: t.TempDir(), DeviceProbe: func() bool { return true }}
			opts.RunVariant = func(_ context.Context, _ uint64, pristine bool, ptx string) (cudaschedule.RunRecord, error) {
				evidence := recordedOutput(kind)
				if pristine {
					evidence = recordedOutput("healthy")
				}
				h := testPTXDigest(ptx)
				evidence.PTXDigest = h
				return cudaschedule.RunRecord{Exited: true, PTXDigest: h, PTXDigestSource: "runner_receipt", Output: evidence}, nil
			}
			report, err := cudaschedule.RunCampaign(context.Background(), opts)
			if err != nil {
				t.Fatal(err)
			}
			want := cudaschedule.VerdictMismatch
			if kind == "healthy" {
				want = cudaschedule.VerdictPass
			}
			seen := 0
			for _, seed := range report.Seeds {
				if seed.Verdict == cudaschedule.VerdictEmptyDraw {
					continue
				}
				seen++
				if seed.Verdict != want || seed.Run == nil || seed.Run.Output == nil {
					t.Fatalf("lost output evidence or wrong verdict: %+v", seed)
				}
				if want == cudaschedule.VerdictMismatch {
					if seed.Confirmation == nil || len(seed.Confirmation.Runs) != 2 {
						t.Fatal("missing independently recorded confirmations")
					}
					for _, run := range seed.Confirmation.Runs {
						if run.Output == nil || cudaschedule.Classify(run) != want {
							t.Fatal("lost confirmation readbacks")
						}
					}
				}
				body, err := json.Marshal(seed.Run)
				if err != nil {
					t.Fatal(err)
				}
				var saved cudaschedule.RunRecord
				if err := json.Unmarshal(body, &saved); err != nil || cudaschedule.Classify(saved) != want {
					t.Fatalf("recorded evidence did not survive serialization: %v", err)
				}
			}
			if seen == 0 {
				t.Fatal("campaign executed no variants")
			}
		})
	}
}

func TestOutputRecordRefusesMalformedEvidence(t *testing.T) {
	for _, kind := range []string{"loaded_digest", "source_binding", "extent", "fill", "domain"} {
		t.Run(kind, func(t *testing.T) {
			out := recordedOutput("healthy")
			h := strings.Repeat("c", 64)
			out.PTXDigest = h
			switch kind {
			case "loaded_digest":
				out.PTXDigest = strings.Repeat("d", 64)
			case "source_binding":
				out.Policy.SourceDigest = strings.Repeat("d", 64)
			case "extent":
				out.TwoFill.ReadbackA = nil
			case "fill":
				out.TwoFill.FillB = out.TwoFill.FillA
			case "domain":
				out.TwoFill.Contract.Domain[0].End = 1007
			}
			run := cudaschedule.RunRecord{Exited: true, PTXDigest: h, ExpectedPTXDigest: h, PTXDigestSource: "runner_receipt", Output: out}
			if cudaschedule.Classify(run) != cudaschedule.VerdictHarnessError {
				t.Fatal("malformed evidence scored")
			}
		})
	}
}

func TestCampaignOutputSnapshotsRunnerStorage(t *testing.T) {
	out := recordedOutput("healthy")
	opts := cudaschedule.CampaignOptions{PTXSource: CAMPAIGN_PTX, SeedStart: 1, SeedEnd: 4, Intensity: 2, ConfirmRuns: 1, ArtifactDir: t.TempDir(), DeviceProbe: func() bool { return true }}
	opts.RunVariant = func(_ context.Context, _ uint64, _ bool, ptx string) (cudaschedule.RunRecord, error) {
		out.PTXDigest = testPTXDigest(ptx)
		return cudaschedule.RunRecord{Exited: true, PTXDigest: out.PTXDigest, PTXDigestSource: "runner_receipt", Output: out}, nil
	}
	report, err := cudaschedule.RunCampaign(context.Background(), opts)
	if err != nil {
		t.Fatal(err)
	}
	out.TwoFill.Contract.Domain[0].End = 1
	out.TwoFill.Reference[0] = 0
	out.TwoFill.ReadbackA[0] = 0
	out.TwoFill.ReadbackB[0] = 0
	out.Ordinary[0] = 0
	out.Policy.SourceDigest = "changed"
	if report.Baseline.Output.TwoFill.Contract.Domain[0].End != 1003 || cudaschedule.Classify(report.Baseline) != cudaschedule.VerdictPass {
		t.Fatal("baseline evidence aliases runner storage")
	}
	for _, seed := range report.Seeds {
		if seed.Run != nil && (seed.Run.Output.TwoFill.Contract.Domain[0].End != 1003 || cudaschedule.Classify(*seed.Run) != cudaschedule.VerdictPass) {
			t.Fatal("variant evidence aliases runner storage")
		}
	}
}
