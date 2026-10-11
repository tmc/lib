package cudaworkload

import (
	"context"
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
)

// NativeOperation calls fn with the operation's external ID on the pinned
// thread. The returned annotation does not invent a native correlation ID.
// ctx cancellation is checked before entry; cancellation does not interrupt
// native code or imply device completion. fn must arrange its own cancellation.
func (s *Scope) NativeOperation(ctx context.Context, cs ContextStreamID, symbol, evidence string, fn func() error) (HostSubmission, error) {
	var sub HostSubmission
	if ctx == nil || fn == nil {
		return sub, errors.New("cudaworkload: invalid native operation")
	}
	if symbol == "" || evidence == "" {
		return sub, errors.New("cudaworkload: native operation lacks evidence identity")
	}
	if s == nil || s.annotator == nil || nilCorrelationHook(s.annotator.corrHook) || s.annotator.corrKind <= cupti.ExternalCorrelationKindInvalid || s.annotator.corrKind >= cupti.ExternalCorrelationKindSize {
		return sub, errors.New("cudaworkload: external correlation is not enabled")
	}
	err := s.NativeCall(func() error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if err := fn(); err != nil {
			return err
		}
		sub = s.NewSubmission(0, cs, GraphReplayGeneration{}, symbol, evidence)
		return nil
	})
	if err != nil {
		return HostSubmission{}, err
	}
	return sub, nil
}

// ActivityBatch is a frozen CUPTI collection. ClockEpoch identifies the actual
// producer timestamp epoch; it is never derived from host wall time. Records
// and their ordering remain unchanged in the returned evidence.
type ActivityBatch struct {
	Session, Run, ClockEpoch, EvidenceRef string
	Process                               ProcessGenID
	ExternalKind                          cupti.ExternalCorrelationKind
	Records                               []cuptiactivity.Record
	// MaxRecords bounds records and annotations together; zero means 4096.
	MaxRecords int
}

// ActivityJoin retains a raw record and its operation classification. Exact
// means an explicit external-ID link, not clock alignment or causal diagnosis.
type ActivityJoin struct {
	Record                                       cuptiactivity.Record
	Session, Run, Clock, ClockEpoch, EvidenceRef string
	Join                                         JoinResult
}

// JoinActivities joins a completed batch against operation annotations.
// It scans the entire batch before emitting results, so later contradictory
// links cannot retroactively invalidate an emitted exact result. Native IDs
// alone and timestamp proximity never identify operations. Graph replay needs
// a separate explicit replay producer and is deliberately not inferred here.
func JoinActivities(batch ActivityBatch, annotations []HostSubmission) ([]ActivityJoin, error) {
	limit := batch.MaxRecords
	if limit == 0 {
		limit = 4096
	}
	if limit < 1 || limit > 65536 || len(batch.Records) > limit || len(annotations) > limit-len(batch.Records) {
		return nil, errors.New("cudaworkload: activity batch exceeds bound")
	}
	if batch.Session == "" || batch.Run == "" || batch.ClockEpoch == "" || batch.EvidenceRef == "" || batch.Process.PID == 0 || batch.Process.Generation == 0 {
		return nil, errors.New("cudaworkload: incomplete collection provenance")
	}
	scoped := make([]cuptiactivity.ScopedRecord, len(batch.Records))
	for i, r := range batch.Records {
		if r.ProcessID != 0 && r.ProcessID != batch.Process.PID {
			return nil, fmt.Errorf("cudaworkload: record %d contradicts collection process", i)
		}
		if r.Complete && (r.Start == 0 || r.End == 0 || r.End < r.Start) {
			return nil, fmt.Errorf("cudaworkload: record %d has contradictory complete timestamps", i)
		}
		scoped[i] = cuptiactivity.ScopedRecord{Session: batch.Session, Run: batch.Run, Process: batch.Process.String(), Record: r}
	}
	matches, err := cuptiactivity.MatchExternal(scoped, batch.ExternalKind)
	if err != nil {
		return nil, err
	}
	byOp := make(map[OpID][]HostSubmission)
	for i, sub := range annotations {
		if sub.Process != batch.Process || sub.Workload.Process != batch.Process || sub.Workload.RunID != batch.Run || sub.OpID == 0 || sub.Provenance != ProvenanceCUPTI || sub.EvidenceRef == "" || sub.Symbol == "" {
			return nil, fmt.Errorf("cudaworkload: annotation %d lacks matching producer identity", i)
		}
		sub.Workload = sub.Workload.Clone()
		byOp[sub.OpID] = append(byOp[sub.OpID], sub)
	}
	out := make([]ActivityJoin, 0, len(matches))
	for _, m := range matches {
		r := batch.Records[m.Index]
		act := DeviceActivity{Process: batch.Process, NativeCorrelationID: r.CorrelationID, RuntimeCorrelationID: r.RuntimeCorrelationID, Kind: r.Kind.String(), Name: r.Name, Context: ContextStreamID{uint64(r.ContextID), uint64(r.StreamID)}, StartNs: r.Start, EndNs: r.End, DeviceID: r.DeviceID, EvidenceRef: batch.EvidenceRef}
		res := JoinResult{Outcome: OutcomeMissing, Reason: ReasonNoHostSubmission, DeviceActivity: &act, EvidenceRef: batch.EvidenceRef, RawIDs: rawFromActivity(act)}
		if m.Status == "ambiguous" {
			res.Outcome = OutcomeAmbiguous
			res.Reason = ReasonConflictingExternalID
		}
		if m.Status == "matched" {
			act.ExternalCorrelationID = m.ExternalID
			res.RawIDs = rawFromActivity(act)
			subs := byOp[OpID(m.ExternalID)]
			if len(subs) > 1 {
				res.Outcome = OutcomeAmbiguous
				res.Reason = ReasonConflictingOperations
			}
			if len(subs) == 1 {
				sub := subs[0]
				if r.GraphID != 0 || r.GraphNodeID != 0 || !sub.GraphReplay.IsZero() || r.ThreadID != 0 && sub.TID != 0 && uint64(r.ThreadID) != sub.TID {
					res.Outcome = OutcomeAmbiguous
					res.Reason = ReasonConflictingOperations
				} else {
					res = (&Joiner{}).buildExactOrAmbiguousMatch(sub, act)
				}
			}
		}
		out = append(out, ActivityJoin{Record: r, Session: batch.Session, Run: batch.Run, Clock: "cupti", ClockEpoch: batch.ClockEpoch, EvidenceRef: batch.EvidenceRef, Join: res})
	}
	return out, nil
}
