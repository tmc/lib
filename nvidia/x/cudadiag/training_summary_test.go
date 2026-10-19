package cudadiag_test

import (
	"bytes"
	"context"
	"errors"
	"io"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudadiag"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func trainingSummaryFixture(change func(*nvtelemetry.TrainingPhaseContext, *nvtelemetry.TrainingContext, *nvtelemetry.OperatorContext)) (cudadiag.TrainingSummaryInput, error) {
	raw, operators, source := operatorSummaryInputs()
	policy := nvtelemetry.TrainingQualityPolicy{ID: "q1", Model: "model-sha", Data: "data-sha", Configuration: "config-sha", Seed: "seed:7", Precision: "fp32", LossScaling: "none", Optimizer: "sgd", LearningRateSchedule: "fixed", Batching: "two-microbatches", Oracle: "independent-output", Tolerance: "registered-tolerance", Metric: "output-error", ConvergenceWindow: "unqualified-short-run", Numerical: "accepted", Evidence: "application-oracle-log"}
	detail := nvtelemetry.TrainingPhaseContext{Version: 2, RunID: "run", Policies: []nvtelemetry.TrainingQualityPolicy{policy}, Phases: []nvtelemetry.TrainingPhase{}}
	updates := nvtelemetry.TrainingContext{Version: 1, RunID: "run", Updates: []nvtelemetry.TrainingUpdate{}}
	checks := nvtelemetry.TrainingCheckpointContext{Version: 1, RunID: "run", Checkpoints: []nvtelemetry.TrainingCheckpoint{}}
	generation := uint64(0)
	for rank := uint32(0); rank < 2; rank++ {
		for _, attempt := range []string{"initial", "resume"} {
			updates.Updates = append(updates.Updates, nvtelemetry.TrainingUpdate{Attempt: attempt, Step: 1, Rank: rank, WorldSize: 2, Outcome: "updated", Quality: "accepted", QualityPolicy: "q1", Evidence: "independent-update-log"})
			for micro := uint64(0); micro < 2; micro++ {
				id := attempt + string(rune('0'+rank)) + string(rune('a'+micro))
				detail.Phases = append(detail.Phases, nvtelemetry.TrainingPhase{ID: id, Attempt: attempt, Restart: attempt, Worker: string(rune('0' + rank)), OptimizerStep: 1, LoggingStep: 100, Epoch: 2, Batch: 4, Microbatch: micro, AccumulationStep: micro, AccumulationCount: 2, Rank: rank, WorldSize: 2, CommunicatorEpoch: "comm:1", Phase: "backward", QualityPolicy: "q1", Evidence: "independent-phase-log", Operators: []nvtelemetry.TrainingOperatorReference{{GroupID: operators.Groups[0].ID, Binding: operators.Groups[0].Binding}}})
			}
		}
		for _, state := range []string{"published", "restored"} {
			checkpoint := nvtelemetry.TrainingCheckpoint{CheckpointID: "checkpoint", State: state, Digest: strings.Repeat("c", 64), Storage: "rank-storage", SourceAttempt: "initial", SavedStep: 0, Rank: rank, WorldSize: 2, Evidence: "checkpoint-log"}
			if state == "restored" {
				checkpoint.RestoredAttempt = "resume"
				checkpoint.ResumeGeneration = &generation
			}
			checks.Checkpoints = append(checks.Checkpoints, checkpoint)
		}
	}
	if change != nil {
		change(&detail, &updates, &operators)
	}
	var db, ub, cb, ob bytes.Buffer
	l := nvtelemetry.WorkloadLimits{}
	for _, err := range []error{nvtelemetry.WriteTrainingPhaseContext(&db, detail, l), nvtelemetry.WriteTrainingContext(&ub, updates, l), nvtelemetry.WriteTrainingCheckpointContext(&cb, checks, l), nvtelemetry.WriteOperatorContext(&ob, operators, l)} {
		if err != nil {
			return cudadiag.TrainingSummaryInput{}, err
		}
	}
	src := nvtelemetry.WorkloadSource{Provider: "application", Version: "training-context-v1", RunID: "run"}
	ds := src
	ds.Version = "training-context-v2"
	cs := src
	cs.Version = "checkpoint-context-v1"
	return cudadiag.TrainingSummaryInput{Activities: bytes.NewReader(raw), Operators: &ob, OperatorSource: source, Details: &db, DetailSource: ds, Outcomes: &ub, OutcomeSource: src, Checkpoints: &cb, CheckpointSource: cs, ExpectedPolicy: policy}, nil
}

func TestTrainingSummaryPublicComposition(t *testing.T) {
	in, err := trainingSummaryFixture(nil)
	if err != nil {
		t.Fatal(err)
	}
	out, err := cudadiag.AnalyzeTrainingJSONL(context.Background(), in, nvtelemetry.WorkloadLimits{})
	if err != nil {
		t.Fatal(err)
	}
	if out.Quality != "accepted" || out.Recovery.Updates.Accepted != 2 || len(out.Details.Phases) != 8 || out.Operators.KernelPopulation.Exact != 1 || out.Operators.MemberEdgePopulation.Associated != 2 || len(out.Edges) != 8 {
		t.Fatal(out)
	}
	for _, e := range out.Edges {
		if e.Outcome != cudaworkload.OutcomeAssociated {
			t.Fatal(e)
		}
	}
	if len(out.Recovery.Recoveries) != 1 || !out.Recovery.Recoveries[0].RecoveryAvailable || out.Recovery.Recoveries[0].ObservedAcceptedLoss == nil || *out.Recovery.Recoveries[0].ObservedAcceptedLoss != 1 || out.Recovery.Recoveries[0].UpdatePopulationComplete {
		t.Fatal(out.Recovery)
	}
	if out.ContextSHA256 == "" || out.Operators.ContextSHA256 == "" || out.Recovery.TrainingRawSHA256 == "" || out.Recovery.CheckpointRawSHA256 == "" {
		t.Fatal("lost original digest")
	}
}

func TestTrainingSummaryWithholds(t *testing.T) {
	for _, tt := range []struct {
		name    string
		change  func(*nvtelemetry.TrainingPhaseContext, *nvtelemetry.TrainingContext, *nvtelemetry.OperatorContext)
		quality string
	}{
		{"missing rank", func(_ *nvtelemetry.TrainingPhaseContext, u *nvtelemetry.TrainingContext, _ *nvtelemetry.OperatorContext) {
			u.Updates = u.Updates[:3]
		}, "unknown"},
		{"policy precision", func(d *nvtelemetry.TrainingPhaseContext, _ *nvtelemetry.TrainingContext, _ *nvtelemetry.OperatorContext) {
			d.Policies[0].Precision = "fp16"
		}, "unknown"},
		{"wrong numerical output", func(d *nvtelemetry.TrainingPhaseContext, _ *nvtelemetry.TrainingContext, _ *nvtelemetry.OperatorContext) {
			d.Policies[0].Numerical = "rejected"
		}, "rejected"},
		{"rejected outcome", func(_ *nvtelemetry.TrainingPhaseContext, u *nvtelemetry.TrainingContext, _ *nvtelemetry.OperatorContext) {
			u.Updates[0].Quality = "rejected"
		}, "rejected"},
		{"compiler mismatch", func(d *nvtelemetry.TrainingPhaseContext, _ *nvtelemetry.TrainingContext, _ *nvtelemetry.OperatorContext) {
			d.Phases[0].Operators[0].Binding.CompilerBuild = "changed"
		}, "unknown"},
		{"replay mismatch", func(d *nvtelemetry.TrainingPhaseContext, _ *nvtelemetry.TrainingContext, _ *nvtelemetry.OperatorContext) {
			d.Phases[0].Operators[0].Binding.Replay = "different-pass"
		}, "unknown"},
		{"native missing", func(d *nvtelemetry.TrainingPhaseContext, _ *nvtelemetry.TrainingContext, _ *nvtelemetry.OperatorContext) {
			d.Phases[0].Operators[0].Binding.Context.Context = 0
		}, "unknown"},
		{"phase missing", func(d *nvtelemetry.TrainingPhaseContext, _ *nvtelemetry.TrainingContext, _ *nvtelemetry.OperatorContext) {
			d.Phases = d.Phases[:1]
		}, "unknown"},
		{"no operator refs", func(d *nvtelemetry.TrainingPhaseContext, _ *nvtelemetry.TrainingContext, _ *nvtelemetry.OperatorContext) {
			d.Phases[0].Operators = nil
		}, "unknown"},
		{"comm mismatch", func(d *nvtelemetry.TrainingPhaseContext, _ *nvtelemetry.TrainingContext, _ *nvtelemetry.OperatorContext) {
			for i := range d.Phases {
				if d.Phases[i].Rank == 1 {
					d.Phases[i].CommunicatorEpoch = "comm:changed"
				}
			}
		}, "unknown"},
		{"code missing", func(_ *nvtelemetry.TrainingPhaseContext, _ *nvtelemetry.TrainingContext, o *nvtelemetry.OperatorContext) {
			delete(o.Artifacts, o.Groups[0].Binding.Code.ArtifactDigest)
		}, "unknown"},
		{"comm unknown", func(d *nvtelemetry.TrainingPhaseContext, _ *nvtelemetry.TrainingContext, _ *nvtelemetry.OperatorContext) {
			for i := range d.Phases {
				d.Phases[i].CommunicatorEpoch = ""
			}
		}, "unknown"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			in, err := trainingSummaryFixture(tt.change)
			if err != nil {
				t.Fatal(err)
			}
			out, err := cudadiag.AnalyzeTrainingJSONL(context.Background(), in, nvtelemetry.WorkloadLimits{})
			if err != nil || out.Quality != tt.quality || len(out.Reasons) == 0 || out.Operators.KernelPopulation.Total != 1 || len(out.Operators.Activities.Joins) != 1 {
				t.Fatal(out, err)
			}
		})
	}
}

func TestTrainingSummaryReaderFailures(t *testing.T) {
	for _, mode := range []string{"cancel", "nil", "run", "raw", "policy"} {
		t.Run(mode, func(t *testing.T) {
			in, err := trainingSummaryFixture(nil)
			if err != nil {
				t.Fatal(err)
			}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			switch mode {
			case "cancel":
				cancel()
			case "nil":
				in.Details = nil
			case "run":
				in.DetailSource.RunID = "other"
			case "raw":
				in.Activities = bytes.NewBufferString("corrupt\n")
			case "policy":
				in.ExpectedPolicy.ID = "other"
			}
			out, err := cudadiag.AnalyzeTrainingJSONL(ctx, in, nvtelemetry.WorkloadLimits{})
			if mode == "policy" {
				if err != nil || out.Quality != "unknown" {
					t.Fatal(out, err)
				}
				return
			}
			if err == nil {
				t.Fatal("bad input admitted")
			}
			if mode == "cancel" && !errors.Is(err, context.Canceled) {
				t.Fatal(err)
			}
		})
	}
	// Replaying actual retained v1 bytes remains a distinct reader, not decoded
	// field substitution. A truncated context cannot become an empty success.
	in, err := trainingSummaryFixture(nil)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := io.ReadAll(in.Outcomes)
	if err != nil {
		t.Fatal(err)
	}
	in.Outcomes = bytes.NewReader(raw[:len(raw)-1])
	if _, err := cudadiag.AnalyzeTrainingJSONL(context.Background(), in, nvtelemetry.WorkloadLimits{}); err == nil {
		t.Fatal("truncation accepted")
	}
}
