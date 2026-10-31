package cudadiag

import (
	"context"
	"fmt"
	"io"
	"maps"

	"github.com/tmc/lib/nvidia/x/cudaprof/tracecontract"
	"github.com/tmc/lib/nvidia/x/cudaworkload"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

// ServingInput names a version 2 serving context and the evidence it is
// checked against. Movement, when present, is the movement input of the same
// run; its activity analysis also resolves batch operations. Without it,
// transfer and batch operation edges are withheld. Before and After, when
// both are present, are aggregate serving exports whose counter window is
// computed by CounterWorkloadWindow over the declared interval and clock.
type ServingInput struct {
	Serving  io.Reader
	Source   nvtelemetry.WorkloadSource
	Limits   nvtelemetry.WorkloadLimits
	Movement *MovementInput

	Before, After                       *nvtelemetry.WorkloadExport
	WindowStartNanos, WindowEndNanos    int64
	WindowClockDomain, WindowClockEpoch string
}

// ServingAttemptSummary keeps one supplied attempt's outcome with the batch
// and transfer edges admitted for it. Shared batches are listed, not divided.
type ServingAttemptSummary struct {
	Attempt      nvtelemetry.ServingAttemptRef `json:"attempt"`
	Role         string                        `json:"role"`
	Outcome      string                        `json:"outcome"`
	OutputTokens *uint64                       `json:"output_tokens,omitempty"`
	Batches      []string                      `json:"batches"`
	Transfers    []string                      `json:"transfers"`
}

// ServingEdge is one evaluated supplied relation. Kind is route,
// batch_member, batch_operation, cache_use or transfer. Target is the
// frontend, batch ID, batch ID and operation, cache namespace and
// generation, or transfer ID. A native_id basis may be observed, an
// annotation basis at most associated, and an unknown basis is withheld.
// Transfer and batch operation edges are at most associated: movement
// request identities are supplied annotations. Evidence is the supplied
// reference. Finding.Artifacts of an admitted edge holds the raw digests it
// relies on: the native_id evidence digest, the owning cache declaration's
// evidence when that is a SHA-256 digest, the movement record's raw digest
// and the activity join's evidence reference. Annotation evidence is not a
// digest and is never listed.
type ServingEdge struct {
	Kind     string                         `json:"kind"`
	Attempt  *nvtelemetry.ServingAttemptRef `json:"attempt,omitempty"`
	Target   string                         `json:"target"`
	Basis    string                         `json:"basis"`
	Evidence string                         `json:"evidence,omitempty"`
	Finding  tracecontract.Finding          `json:"finding"`
}

// ServingSummary is a versioned serving incident summary. It retains the
// movement summary and aggregate window it consumed and the digests of their
// sources. Aggregate deltas are never attributed to requests or batches.
type ServingSummary struct {
	Version       int                               `json:"version"`
	ContextSHA256 string                            `json:"context_sha256"`
	Source        nvtelemetry.WorkloadSource        `json:"source"`
	Context       nvtelemetry.ServingContextV2      `json:"context"`
	Outcomes      nvtelemetry.ServingContextSummary `json:"outcomes"`
	Attempts      []ServingAttemptSummary           `json:"attempts"`
	Edges         []ServingEdge                     `json:"edges"`
	Movement      *MovementSummary                  `json:"movement,omitempty"`
	Aggregate     []nvtelemetry.WorkloadDelta       `json:"aggregate"`
	AggregateRaw  []string                          `json:"aggregate_raw_sha256"`
	// AggregateSource retains the before-export's supplied context after the
	// endpoint identities have passed window validation. It is separate from
	// Source, which identifies the application serving context. Nil means no
	// aggregate window; a nil Rank remains unknown, not rank zero. Neither
	// the source nor its rank is authenticated by this report.
	AggregateSource *nvtelemetry.WorkloadSource `json:"aggregate_source,omitempty"`
	Limits          []string                    `json:"limits"`
}

// SummarizeServing reads a version 2 serving context, summarizes its
// version 1 attempt outcomes, runs SummarizeMovement and, when supplied, the
// aggregate counter window, and evaluates each supplied edge independently.
// An edge with a missing or unresolved attempt, a worker epoch or cache
// generation that contradicts the attempt, an unknown basis, or a transfer
// whose movement record names another attempt is withheld alone. Batch
// operations require an exact activity join and remain shared work.
func SummarizeServing(ctx context.Context, in ServingInput) (ServingSummary, error) {
	var out ServingSummary
	if ctx == nil || in.Serving == nil {
		return out, fmt.Errorf("serving summary: missing context or reader")
	}
	if err := ctx.Err(); err != nil {
		return out, err
	}
	if in.Movement != nil && in.Movement.Source.RunID != in.Source.RunID {
		return out, fmt.Errorf("serving summary: movement run differs from serving run")
	}
	if (in.Before == nil) != (in.After == nil) || in.Before != nil && (in.Before.Source.RunID != in.Source.RunID || in.After.Source.RunID != in.Source.RunID) {
		return out, fmt.Errorf("serving summary: aggregate window needs both exports of the serving run")
	}
	export, c, err := nvtelemetry.ReadServingContextV2(telemetryReader{ctx, in.Serving}, in.Source, in.Limits)
	if err != nil {
		return out, fmt.Errorf("serving summary context: %w", err)
	}
	outcomes, err := nvtelemetry.SummarizeServingContext(c.V1(), in.Limits)
	if err != nil {
		return out, fmt.Errorf("serving summary outcomes: %w", err)
	}
	var movement *MovementSummary
	if in.Movement != nil {
		m, err := SummarizeMovement(ctx, *in.Movement)
		if err != nil {
			return out, fmt.Errorf("serving summary: %w", err)
		}
		movement = &m
	}
	out = ServingSummary{Version: 1, ContextSHA256: export.RawSHA256, Source: export.Source, Context: c, Outcomes: outcomes, Movement: movement, Edges: []ServingEdge{}, Aggregate: []nvtelemetry.WorkloadDelta{}, AggregateRaw: []string{}}
	if in.Before != nil {
		deltas, err := nvtelemetry.CounterWorkloadWindow(*in.Before, *in.After, in.WindowStartNanos, in.WindowEndNanos, in.WindowClockDomain, in.WindowClockEpoch)
		if err != nil {
			return ServingSummary{}, fmt.Errorf("serving summary aggregate: %w", err)
		}
		out.Aggregate = deltas
		out.AggregateRaw = []string{in.Before.RawSHA256, in.After.RawSHA256}
		source := in.Before.Source
		if source.Rank != nil {
			rank := *source.Rank
			source.Rank = &rank
		}
		source.Units = maps.Clone(source.Units)
		out.AggregateSource = &source
	}
	out.Limits = []string{
		"application-supplied serving context; producer authenticity unverified and not a provider request-log import",
		"aggregate window deltas are not attributed to requests, attempts or batches",
		"shared batch operations allocate no per-request device time, counters or energy",
		"accepted attempts are not distinct successful requests",
		"batch operations match operation IDs within the supplied process scope and the capture's run; the worker-to-process binding is supplied, not verified",
	}

	e := servingEvaluator{c: c, movement: movement, attempts: make(map[nvtelemetry.ServingAttemptRef]int)}
	for i, a := range c.Attempts {
		ref := nvtelemetry.ServingAttemptRef{RequestID: a.RequestID, Attempt: a.Attempt, Worker: a.Worker, WorkerEpoch: a.WorkerEpoch}
		e.attempts[ref] = i
		out.Attempts = append(out.Attempts, ServingAttemptSummary{Attempt: ref, Role: a.Role, Outcome: a.Outcome, OutputTokens: a.OutputTokens, Batches: []string{}, Transfers: []string{}})
	}
	for i, r := range c.Routes {
		code, reason := e.resolve(r.Attempt)
		out.Edges = append(out.Edges, e.edge(fmt.Sprintf("serving-route-%d", i), "route", &c.Routes[i].Attempt, r.Frontend, r.Basis, r.Evidence, "supplied routing decision "+r.Decision, code, reason))
	}
	for i, b := range c.Batches {
		for j, m := range b.Members {
			code, reason := e.resolve(m)
			if code == "" {
				code, reason = batchMember(b, c.Attempts[e.attempts[m]])
			}
			edge := e.edge(fmt.Sprintf("serving-batch-%d-member-%d", i, j), "batch_member", &c.Batches[i].Members[j], b.ID, b.Basis, b.Evidence, "attempt is a member of a shared batch; no per-request share", code, reason)
			out.Edges = append(out.Edges, edge)
			if edge.Finding.Admission.Admitted {
				a := &out.Attempts[e.attempts[m]]
				a.Batches = append(a.Batches, b.ID)
			}
		}
		for j, op := range b.OpIDs {
			ref, code, reason := e.operation(b, op)
			edge := e.edge(fmt.Sprintf("serving-batch-%d-op-%d", i, j), "batch_operation", nil, fmt.Sprintf("%s/%d", b.ID, op), b.Basis, b.Evidence, "batch operation exactly joined to captured activity; shared work", code, reason)
			capAssociated(&edge)
			if ref != "" {
				edge.Finding.Artifacts = append(edge.Finding.Artifacts, ref)
			}
			out.Edges = append(out.Edges, edge)
		}
	}
	for i, u := range c.CacheUses {
		code, reason := e.resolve(u.Attempt)
		owner := ""
		if code == "" {
			owner, code, reason = e.cache(u)
		}
		edge := e.edge(fmt.Sprintf("serving-cache-%d", i), "cache_use", &c.CacheUses[i].Attempt, u.Namespace+"/"+u.Generation, u.Basis, u.Evidence, "supplied cache "+u.Event+"; cache blocks are not general device memory", code, reason)
		if edge.Finding.Admission.Admitted && validHealthCapacityHash(owner) {
			edge.Finding.Artifacts = append(edge.Finding.Artifacts, owner)
		}
		out.Edges = append(out.Edges, edge)
	}
	for i, t := range c.Transfers {
		code, reason := e.resolve(t.Attempt)
		raw := ""
		if code == "" {
			raw, code, reason = e.transfer(t)
		}
		edge := e.edge(fmt.Sprintf("serving-transfer-%d", i), "transfer", &c.Transfers[i].Attempt, t.TransferID, t.Basis, t.Evidence, "attempt references a supplied movement transfer; request identity is an annotation", code, reason)
		capAssociated(&edge)
		if raw != "" {
			edge.Finding.Artifacts = append(edge.Finding.Artifacts, raw)
		}
		out.Edges = append(out.Edges, edge)
		if edge.Finding.Admission.Admitted {
			a := &out.Attempts[e.attempts[t.Attempt]]
			a.Transfers = append(a.Transfers, t.TransferID)
		}
	}
	return out, nil
}

// capAssociated lowers an observed edge to associated.
func capAssociated(e *ServingEdge) {
	if e.Finding.Class == tracecontract.FindingObserved {
		e.Finding.Class = tracecontract.FindingAssociated
	}
}

type servingEvaluator struct {
	c        nvtelemetry.ServingContextV2
	movement *MovementSummary
	attempts map[nvtelemetry.ServingAttemptRef]int
}

func (e servingEvaluator) resolve(ref nvtelemetry.ServingAttemptRef) (string, string) {
	if ref.RequestID == "" || ref.Attempt == "" || ref.Worker == "" || ref.WorkerEpoch == "" {
		return tracecontract.ReasonEvidenceUnavailable, "attempt identity missing"
	}
	if _, ok := e.attempts[ref]; !ok {
		return tracecontract.ReasonCorrelationUnmatched, "attempt identity matches no supplied attempt"
	}
	return "", ""
}

func (e servingEvaluator) edge(id, kind string, ref *nvtelemetry.ServingAttemptRef, target, basis, evidence, msg, code, reason string) ServingEdge {
	class := tracecontract.FindingAssociated
	switch basis {
	case "native_id":
		class = tracecontract.FindingObserved
	case "unknown":
		if code == "" {
			code, reason = tracecontract.ReasonEvidenceUnavailable, "join basis unknown"
		}
	}
	f := tracecontract.Finding{ID: id, Rule: "serving_" + kind, Class: class, Artifacts: []string{}, Requirements: []tracecontract.Requirement{}, Message: msg}
	if code != "" {
		f.Class = tracecontract.FindingUnavailable
		f.Admission = tracecontract.Admission{Status: "withheld", ReasonCode: code, Reason: reason}
	} else {
		f.Admission = tracecontract.Admission{Admitted: true, Status: "admitted", ReasonCode: tracecontract.ReasonAdmitted, Reason: "supplied identities resolved"}
		if basis == "native_id" {
			// ReadServingContextV2 validated native_id evidence as a SHA-256 digest.
			f.Artifacts = append(f.Artifacts, evidence)
		}
	}
	var copied *nvtelemetry.ServingAttemptRef
	if ref != nil {
		r := *ref
		copied = &r
	}
	return ServingEdge{Kind: kind, Attempt: copied, Target: target, Basis: basis, Evidence: evidence, Finding: f}
}

func batchMember(b nvtelemetry.ServingBatch, a nvtelemetry.ServingAttempt) (string, string) {
	if b.Worker != a.Worker || b.WorkerEpoch != a.WorkerEpoch {
		return tracecontract.ReasonIdentityMismatch, "batch worker or restart epoch differs from attempt"
	}
	if a.Role != "unknown" && b.Role != "unknown" && b.Role != "combined" && a.Role != b.Role {
		return tracecontract.ReasonIdentityMismatch, "batch role differs from attempt role"
	}
	return "", ""
}

// operation requires a supplied batch process scope, a movement capture of
// that process in the serving run, and every activity join for op to be
// exact, with at least one. An ambiguous join names the operation only
// through its host submission. Kernel time is not divided among members.
func (e servingEvaluator) operation(b nvtelemetry.ServingBatch, op cudaworkload.OpID) (ref, code, reason string) {
	if e.movement == nil {
		return "", tracecontract.ReasonEvidenceUnavailable, "no movement capture"
	}
	if b.Process == nil {
		return "", tracecontract.ReasonEvidenceUnavailable, "batch process scope not supplied"
	}
	for _, j := range e.movement.Activities.Joins {
		sub := j.Join.HostSubmission
		if j.Join.OpID != op && (sub == nil || sub.OpID != op) {
			continue
		}
		if j.Run != e.c.RunID || sub == nil || sub.Process != *b.Process {
			return "", tracecontract.ReasonIdentityMismatch, "batch operation captured in another run or process"
		}
		if j.Join.Outcome != cudaworkload.OutcomeExact {
			return "", tracecontract.ReasonCorrelationUnmatched, "batch operation activity join is not exact"
		}
		ref = j.EvidenceRef
	}
	if ref == "" {
		return "", tracecontract.ReasonCorrelationUnmatched, "no exact activity for batch operation"
	}
	return ref, "", ""
}

// cache checks the use against its declared cache generation and returns
// the owning declaration's evidence.
func (e servingEvaluator) cache(u nvtelemetry.ServingCacheUse) (owner, code, reason string) {
	for _, c := range e.c.Caches {
		if c.Namespace != u.Namespace || c.Generation != u.Generation {
			continue
		}
		if c.Worker != u.Attempt.Worker || c.WorkerEpoch != u.Attempt.WorkerEpoch {
			return "", tracecontract.ReasonIdentityMismatch, "cache generation belongs to another worker epoch"
		}
		return c.Evidence, "", ""
	}
	return "", tracecontract.ReasonCorrelationUnmatched, "cache namespace generation not declared"
}

// transfer checks the reference against the movement record and returns
// the record's raw digest. A record naming only the request, or no request,
// cannot contradict the attempt; a different value does.
func (e servingEvaluator) transfer(t nvtelemetry.ServingTransferRef) (raw, code, reason string) {
	if e.movement == nil {
		return "", tracecontract.ReasonEvidenceUnavailable, "no movement capture"
	}
	if e.movement.Context.SourceClass == "modeled" {
		return "", tracecontract.ReasonBackendUnsupported, "modeled movement source"
	}
	for _, m := range e.movement.Context.Transfers {
		if m.ID != t.TransferID {
			continue
		}
		if m.RequestID != "" && m.RequestID != t.Attempt.RequestID || m.Attempt != "" && m.Attempt != t.Attempt.Attempt {
			return "", tracecontract.ReasonIdentityMismatch, "movement transfer names another attempt"
		}
		return m.RawSHA256, "", ""
	}
	return "", tracecontract.ReasonCorrelationUnmatched, "transfer not in movement context"
}
