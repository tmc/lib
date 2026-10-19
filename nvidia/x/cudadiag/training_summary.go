package cudadiag

import (
	"context"
	"errors"
	"fmt"
	"io"
	"reflect"

	"github.com/tmc/lib/nvidia/x/cudaworkload"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

// TrainingSummaryInput supplies distinct original interchanges. Outcomes and
// Checkpoints are the unchanged version 1 application formats. Details is the
// version 2 phase extension. Operator metadata and activities use the existing
// operator analysis contract. ExpectedPolicy retains the independently preregistered policy;
// its absence withholds useful-work qualification. Readers remain caller-owned.
type TrainingSummaryInput struct {
	Activities       io.Reader
	Operators        io.Reader
	OperatorSource   nvtelemetry.WorkloadSource
	Details          io.Reader
	DetailSource     nvtelemetry.WorkloadSource
	Outcomes         io.Reader
	OutcomeSource    nvtelemetry.WorkloadSource
	Checkpoints      io.Reader
	CheckpointSource nvtelemetry.WorkloadSource
	ExpectedPolicy   nvtelemetry.TrainingQualityPolicy
}

// TrainingPhaseEdge relates one phase/reference to existing operator evidence.
// Even an exact kernel join does not upgrade supplied phase membership to exact.
type TrainingPhaseEdge struct {
	PhaseID string
	GroupID string
	Outcome cudaworkload.Outcome
	Reason  string
}

// TrainingSummary retains observed activity, supplied phases, optimizer counts
// and checkpoint recovery separately. Quality is accepted, rejected or unknown
// for the supplied observed update population only, never a convergence or
// comparative speedup claim. Reasons retain missing or contradictory evidence.
// Policy details and all raw digests remain available for audit.
type TrainingSummary struct {
	Version       int
	ContextSHA256 string
	Details       nvtelemetry.TrainingPhaseContext
	Operators     OperatorSummary
	Recovery      nvtelemetry.TrainingRecoverySummary
	Edges         []TrainingPhaseEdge
	Quality       string
	Reasons       []string
	Limits        []string
}

// AnalyzeTrainingJSONL reads original context bytes, calls AnalyzeOperatorJSONL
// and reuses SummarizeTrainingRecovery's all-rank update/checkpoint rules. It
// never allocates operator duration/counters to phases or adds overlapping time.
// Cancellation is checked between records; a blocking Reader is caller-owned.
func AnalyzeTrainingJSONL(ctx context.Context, in TrainingSummaryInput, l nvtelemetry.WorkloadLimits) (TrainingSummary, error) {
	var out TrainingSummary
	if ctx == nil || in.Activities == nil || in.Operators == nil || in.Details == nil || in.Outcomes == nil || in.Checkpoints == nil {
		return out, errors.New("training summary: missing context")
	}
	if err := ctx.Err(); err != nil {
		return out, err
	}
	de, details, err := nvtelemetry.ReadTrainingPhaseContext(telemetryReader{ctx, in.Details}, in.DetailSource, l)
	if err != nil {
		return out, fmt.Errorf("training summary details: %w", err)
	}
	te, updates, err := nvtelemetry.ReadTrainingContext(telemetryReader{ctx, in.Outcomes}, in.OutcomeSource, l)
	if err != nil {
		return out, fmt.Errorf("training summary outcomes: %w", err)
	}
	ce, _, err := nvtelemetry.ReadTrainingCheckpointContext(telemetryReader{ctx, in.Checkpoints}, in.CheckpointSource, l)
	if err != nil {
		return out, fmt.Errorf("training summary checkpoints: %w", err)
	}
	recovery, err := nvtelemetry.SummarizeTrainingRecovery(te, ce, l)
	if err != nil {
		return out, fmt.Errorf("training summary recovery: %w", err)
	}
	ops, err := AnalyzeOperatorJSONL(ctx, in.Activities, in.Operators, in.OperatorSource, l)
	if err != nil {
		return out, err
	}
	if details.RunID != recovery.RunID || details.RunID != ops.Context.RunID {
		return out, errors.New("training summary run mismatch")
	}
	out = TrainingSummary{Version: 1, ContextSHA256: de.RawSHA256, Details: details, Operators: ops, Recovery: recovery, Quality: "unknown", Limits: []string{"application-supplied phases, outcomes and policies; authenticity and oracle correctness unverified", "phase membership is at most associated; native activity population remains independent", "no additive phase duration, GPU shares, improvement or convergence inference", "omitted updates, microbatches and phases cannot be detected from supplied population"}}
	reason := func(s string) {
		for _, r := range out.Reasons {
			if r == s {
				return
			}
		}
		out.Reasons = append(out.Reasons, s)
	}
	policies := map[string]nvtelemetry.TrainingQualityPolicy{}
	for _, p := range details.Policies {
		policies[p.ID] = p
	}
	policy, ok := policies[in.ExpectedPolicy.ID]
	if in.ExpectedPolicy.ID == "" || !ok {
		reason("expected quality policy unavailable")
	} else {
		if !reflect.DeepEqual(policy, in.ExpectedPolicy) {
			reason("quality policy identity/details mismatch")
		}
		// ConvergenceWindow is retained but never used to qualify local correctness.
		for _, s := range []string{policy.Model, policy.Data, policy.Configuration, policy.Seed, policy.Precision, policy.LossScaling, policy.Optimizer, policy.LearningRateSchedule, policy.Batching, policy.Oracle, policy.Tolerance, policy.Metric, policy.Evidence} {
			if s == "" {
				reason("quality policy details unavailable")
				break
			}
		}
		if policy.Numerical != "accepted" {
			reason("numerical oracle did not accept")
		}
	}
	type key struct {
		attempt string
		step    uint64
		rank    uint32
	}
	observed := map[key]nvtelemetry.TrainingUpdate{}
	for _, u := range updates.Updates {
		observed[key{u.Attempt, u.Step, u.Rank}] = u
		if u.QualityPolicy != in.ExpectedPolicy.ID {
			reason("optimizer quality policy mismatch")
		}
	}
	covered := map[key]bool{}
	type stepKey struct {
		attempt string
		step    uint64
	}
	communicators := map[stepKey]string{}
	groups := map[string]nvtelemetry.OperatorGroup{}
	for _, g := range ops.Context.Groups {
		groups[g.ID] = g
	}
	max := l.MaxMetrics
	if max == 0 {
		max = 10000
	}
	for _, p := range details.Phases {
		if err := ctx.Err(); err != nil {
			return TrainingSummary{}, err
		}
		sk := stepKey{p.Attempt, p.OptimizerStep}
		if prior, ok := communicators[sk]; ok && prior != p.CommunicatorEpoch {
			reason("communicator epoch mismatch across ranks")
		}
		communicators[sk] = p.CommunicatorEpoch
		k := key{p.Attempt, p.OptimizerStep, p.Rank}
		u, ok := observed[k]
		if !ok || u.WorldSize != p.WorldSize {
			reason("phase optimizer/rank identity unavailable or contradictory")
		} else {
			covered[k] = true
		}
		if p.QualityPolicy != in.ExpectedPolicy.ID {
			reason("phase quality policy mismatch")
		}
		if p.CommunicatorEpoch == "" {
			reason("communicator epoch unavailable")
		}
		if len(p.Operators) == 0 {
			reason("phase operator reference unavailable")
		}
		for _, ref := range p.Operators {
			if len(out.Edges) >= max {
				return TrainingSummary{}, errors.New("training summary edge limit exceeded")
			}
			edge := TrainingPhaseEdge{PhaseID: p.ID, GroupID: ref.GroupID, Outcome: cudaworkload.OutcomeMissing, Reason: "operator group unavailable"}
			g, found := groups[ref.GroupID]
			if found {
				b := g.Binding
				retainedCode := false
				if b.Code != nil {
					for _, hash := range ops.RawArtifactSHA256 {
						if hash == b.Code.ArtifactDigest {
							retainedCode = true
						}
					}
				}
				switch {
				case !reflect.DeepEqual(ref.Binding, b):
					edge.Outcome = cudaworkload.OutcomeAmbiguous
					edge.Reason = "supplied execution/compiler binding differs"
				case b.Process.PID == 0 || b.Process.Generation == 0 || b.OpID == 0 || b.Context.Context == 0 || b.Code == nil || !retainedCode || b.ModuleID == "" || b.ModuleGeneration == 0 || b.Pass == "" || b.Replay == "" || b.CompilerBuild == "":
					edge.Reason = "native/code/replay binding unavailable"
				default:
					for _, e := range ops.Edges {
						if e.GroupID != ref.GroupID {
							continue
						}
						if e.Outcome == cudaworkload.OutcomeAssociated && edge.Outcome == cudaworkload.OutcomeMissing {
							edge.Outcome = cudaworkload.OutcomeAssociated
							edge.Reason = "supplied phase and compiler membership; not native phase correlation"
						}
						if e.Outcome != cudaworkload.OutcomeAssociated {
							edge.Outcome = e.Outcome
							edge.Reason = e.Reason
							break
						}
					}
				}
			}
			if edge.Outcome != cudaworkload.OutcomeAssociated {
				reason("phase operator attribution withheld")
			}
			out.Edges = append(out.Edges, edge)
		}
	}
	for k := range observed {
		if !covered[k] {
			reason("optimizer update phase context unavailable")
		}
	}
	if recovery.Updates.Incomplete != 0 {
		reason("optimizer rank population incomplete")
	}
	if recovery.Updates.Unknown != 0 || recovery.Updates.Skipped != 0 || recovery.Updates.Failed != 0 || recovery.Updates.Accepted == 0 {
		reason("observed optimizer population not wholly accepted")
	}
	if recovery.Updates.Rejected != 0 {
		reason("optimizer quality rejected")
	}
	if recovery.Updates.Rejected != 0 || policy.Numerical == "rejected" {
		out.Quality = "rejected"
	} else if len(out.Reasons) == 0 {
		out.Quality = "accepted"
	}
	return out, nil
}
