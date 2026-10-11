package cudaworkload

import (
	"sync"
)

// Outcome represents the classification of a correlation join result.
type Outcome string

const (
	// OutcomeExact denotes an unambiguous, 1-to-1 correlation link independently
	// established via explicit correlation IDs or verified provenance.
	OutcomeExact Outcome = "exact"
	// OutcomeAssociated denotes a contextual link (such as context, stream, or thread
	// association) where explicit correlation was not recorded.
	OutcomeAssociated Outcome = "associated"
	// OutcomeAmbiguous denotes multiple candidate matches or conflicting correlation links.
	OutcomeAmbiguous Outcome = "ambiguous"
	// OutcomeMissing denotes an activity or submission with no observed counterpart.
	OutcomeMissing Outcome = "missing"
)

// Reason provides an explanation for why a specific Outcome was reached.
type Reason string

const (
	// ReasonExplicitExternalMatch indicates an exact match on CUPTI external correlation ID.
	ReasonExplicitExternalMatch Reason = "explicit_external_correlation_match"
	// ReasonExplicitNativeMatch indicates an exact match on native correlation ID.
	ReasonExplicitNativeMatch Reason = "explicit_native_correlation_match"
	// ReasonGraphReplayMatch indicates an exact match on graph node and replay generation.
	ReasonGraphReplayMatch Reason = "explicit_graph_replay_match"
	// ReasonStreamContextMatch indicates circumstantial association on context/stream.
	ReasonStreamContextMatch Reason = "associated_stream_context_active_scope"
	// ReasonConflictingExternalID indicates conflicting external correlation IDs were observed.
	ReasonConflictingExternalID Reason = "conflicting_external_correlation_ids"
	// ReasonConflictingOperations indicates multiple host operations claimed the same correlation ID.
	ReasonConflictingOperations Reason = "multiple_operations_for_correlation_id"
	// ReasonNoHostSubmission indicates a device activity had no matching host submission.
	ReasonNoHostSubmission Reason = "no_matching_host_submission"
	// ReasonNoDeviceActivity indicates a host submission had no matching device activity.
	ReasonNoDeviceActivity Reason = "no_matching_device_activity"
	// ReasonLateAttach indicates the activity timestamp was prior to the collector attach window.
	ReasonLateAttach Reason = "late_attach_unobserved_host_submission"
	// ReasonExpiredFIFO indicates the record expired from the bounded pending FIFO queue.
	ReasonExpiredFIFO Reason = "pending_buffer_fifo_expired"
	// ReasonProcessGenMismatch indicates PIDs matched but process generations differed.
	ReasonProcessGenMismatch Reason = "process_generation_mismatch"
	// ReasonUnannotatedActivity indicates unannotated GPU execution.
	ReasonUnannotatedActivity Reason = "unannotated_device_activity"
)

// HostSubmission represents an API call or submission made on the host.
type HostSubmission struct {
	Process             ProcessGenID          `json:"process"`
	OpID                OpID                  `json:"op_id"`
	ParentOpID          OpID                  `json:"parent_op_id,omitempty"`
	Workload            WorkloadIdentity      `json:"workload"`
	NativeCorrelationID uint32                `json:"native_correlation_id"`
	Provenance          CorrelationProvenance `json:"provenance"`
	Context             ContextStreamID       `json:"context"`
	GraphReplay         GraphReplayGeneration `json:"graph_replay,omitempty"`
	TID                 uint64                `json:"tid,omitempty"`
	TimestampNs         uint64                `json:"timestamp_ns"`
	// TimestampClock and TimestampEpoch qualify TimestampNs. Omitted fields in
	// legacy records mean unknown; they must not be inferred from magnitude.
	TimestampClock string `json:"timestamp_clock,omitempty"`
	TimestampEpoch string `json:"timestamp_epoch,omitempty"`
	Symbol         string `json:"symbol"`
	EvidenceRef    string `json:"evidence_ref,omitempty"`
}

// DeviceActivity represents an executed GPU activity observed via CUPTI or driver.
type DeviceActivity struct {
	Process               ProcessGenID          `json:"process"`
	NativeCorrelationID   uint32                `json:"native_correlation_id"`
	RuntimeCorrelationID  uint32                `json:"runtime_correlation_id,omitempty"`
	ExternalCorrelationID uint64                `json:"external_correlation_id,omitempty"`
	Kind                  string                `json:"kind"` // "kernel", "memcpy", "memset", "sync"
	Name                  string                `json:"name"`
	Context               ContextStreamID       `json:"context"`
	GraphReplay           GraphReplayGeneration `json:"graph_replay,omitempty"`
	StartNs               uint64                `json:"start_ns"`
	EndNs                 uint64                `json:"end_ns"`
	DeviceID              uint32                `json:"device_id"`
	EvidenceRef           string                `json:"evidence_ref,omitempty"`
}

// JoinResult holds the classification, reason, and preserved evidence of a join.
type JoinResult struct {
	Outcome        Outcome               `json:"outcome"`
	Reason         Reason                `json:"reason"`
	OpID           OpID                  `json:"op_id,omitempty"`
	ParentOpID     OpID                  `json:"parent_op_id,omitempty"`
	Workload       WorkloadIdentity      `json:"workload"`
	HostSubmission *HostSubmission       `json:"host_submission,omitempty"`
	DeviceActivity *DeviceActivity       `json:"device_activity,omitempty"`
	Provenance     CorrelationProvenance `json:"provenance,omitempty"`
	RawIDs         RawIDs                `json:"raw_ids"`
	EvidenceRef    string                `json:"evidence_ref,omitempty"`
}

// JoinConfig configures the behavior and limits of a [Joiner].
type JoinConfig struct {
	// MaxPendingRecords sets the maximum number of unmatched submissions or
	// activities retained in the pending FIFO buffer. Defaults to 4096 if <= 0.
	MaxPendingRecords int
	// LateAttachCutoffNs designates a timestamp before which unjoined activities
	// are immediately classified as OutcomeMissing with ReasonLateAttach.
	LateAttachCutoffNs uint64
}

// JoinMetrics reports aggregate join metrics and fractions.
type JoinMetrics struct {
	TotalSubmissions uint64 `json:"total_submissions"`
	TotalActivities  uint64 `json:"total_activities"`
	ExactJoins       uint64 `json:"exact_joins"`
	AssociatedJoins  uint64 `json:"associated_joins"`
	AmbiguousJoins   uint64 `json:"ambiguous_joins"`
	MissingJoins     uint64 `json:"missing_joins"`
	FalseExactJoins  uint64 `json:"false_exact_joins"`
}

// Precision reports the fraction of exact joins that were true positives.
// In verified test harnesses tracking FalseExactJoins, any false exact join
// degrades precision.
func (m JoinMetrics) Precision() float64 {
	totalExact := m.ExactJoins + m.FalseExactJoins
	if totalExact == 0 {
		return 1.0
	}
	return float64(m.ExactJoins) / float64(totalExact)
}

// Recall reports the fraction of total device activities successfully joined
// with exact certainty.
func (m JoinMetrics) Recall() float64 {
	if m.TotalActivities == 0 {
		return 0.0
	}
	return float64(m.ExactJoins) / float64(m.TotalActivities)
}

// MissingFraction reports the fraction of total device activities that are missing.
func (m JoinMetrics) MissingFraction() float64 {
	if m.TotalActivities == 0 {
		return 0.0
	}
	return float64(m.MissingJoins) / float64(m.TotalActivities)
}

// AmbiguousFraction reports the fraction of total device activities that are ambiguous.
func (m JoinMetrics) AmbiguousFraction() float64 {
	if m.TotalActivities == 0 {
		return 0.0
	}
	return float64(m.AmbiguousJoins) / float64(m.TotalActivities)
}

type corrKey struct {
	Process     ProcessGenID
	Correlation uint32
	GraphReplay GraphReplayGeneration
}

type extKey struct {
	Process    ProcessGenID
	ExternalID uint64
}

// Joiner coordinates evidence-preserving correlation between host submissions
// and device activities.
type Joiner struct {
	mu  sync.Mutex
	cfg JoinConfig

	// Submissions pending activity match
	pendingSubs []HostSubmission
	subByCorr   map[corrKey][]int
	subByExt    map[extKey][]int

	// Activities pending submission match
	pendingActs []DeviceActivity
	actByCorr   map[corrKey][]int

	metrics JoinMetrics
}

// NewJoiner instantiates a Joiner with the provided configuration.
func NewJoiner(cfg JoinConfig) *Joiner {
	if cfg.MaxPendingRecords <= 0 {
		cfg.MaxPendingRecords = 4096
	}
	return &Joiner{
		cfg:       cfg,
		subByCorr: make(map[corrKey][]int),
		subByExt:  make(map[extKey][]int),
		actByCorr: make(map[corrKey][]int),
	}
}

// Metrics returns a snapshot of current join metrics.
func (j *Joiner) Metrics() JoinMetrics {
	j.mu.Lock()
	defer j.mu.Unlock()
	return j.metrics
}

// RecordFalseExact increments the count of false exact joins in validation harnesses.
func (j *Joiner) RecordFalseExact() {
	j.mu.Lock()
	defer j.mu.Unlock()
	j.metrics.FalseExactJoins++
}

// SubmitHost enqueues a host submission, joining it against pending activities
// or queuing it in the bounded FIFO buffer.
func (j *Joiner) SubmitHost(sub HostSubmission) []JoinResult {
	j.mu.Lock()
	defer j.mu.Unlock()

	j.metrics.TotalSubmissions++
	var results []JoinResult

	// Try matching against pending device activities
	if sub.NativeCorrelationID != 0 {
		k := corrKey{Process: sub.Process, Correlation: sub.NativeCorrelationID, GraphReplay: sub.GraphReplay}
		if actIndices, ok := j.actByCorr[k]; ok && len(actIndices) > 0 {
			if len(actIndices) > 1 {
				res := JoinResult{Outcome: OutcomeAmbiguous, Reason: ReasonConflictingOperations, HostSubmission: &sub}
				j.metrics.AmbiguousJoins++
				return []JoinResult{res}
			}
			// Found matching activity
			actIdx := actIndices[0]
			j.actByCorr[k] = actIndices[1:]
			if len(j.actByCorr[k]) == 0 {
				delete(j.actByCorr, k)
			}
			act := j.pendingActs[actIdx]
			// Remove the consumed activity; keeping its timestamps would let Flush
			// report it a second time as missing.
			j.pendingActs = append(j.pendingActs[:actIdx], j.pendingActs[actIdx+1:]...)
			j.rebuildActIndices()

			res := j.buildExactOrAmbiguousMatch(sub, act)
			j.updateMetrics(res.Outcome)
			return []JoinResult{res}
		}
	}

	// No pending activity matched; enqueue into bounded host buffer
	if len(j.pendingSubs) >= j.cfg.MaxPendingRecords {
		// Evict oldest host submission via FIFO
		evicted := j.pendingSubs[0]
		j.pendingSubs = j.pendingSubs[1:]
		j.rebuildSubIndices()

		evictRes := JoinResult{
			Outcome:        OutcomeMissing,
			Reason:         ReasonExpiredFIFO,
			OpID:           evicted.OpID,
			ParentOpID:     evicted.ParentOpID,
			Workload:       evicted.Workload,
			HostSubmission: &evicted,
			Provenance:     evicted.Provenance,
			RawIDs: RawIDs{
				ProcessGenID:        evicted.Process,
				NativeCorrelationID: evicted.NativeCorrelationID,
				ContextStream:       evicted.Context,
				GraphReplay:         evicted.GraphReplay,
				TID:                 evicted.TID,
			},
			EvidenceRef: evicted.EvidenceRef,
		}
		j.metrics.MissingJoins++
		results = append(results, evictRes)
	}

	idx := len(j.pendingSubs)
	j.pendingSubs = append(j.pendingSubs, sub)
	if sub.NativeCorrelationID != 0 {
		k := corrKey{Process: sub.Process, Correlation: sub.NativeCorrelationID, GraphReplay: sub.GraphReplay}
		j.subByCorr[k] = append(j.subByCorr[k], idx)
	}
	if sub.OpID != 0 {
		k := extKey{Process: sub.Process, ExternalID: uint64(sub.OpID)}
		j.subByExt[k] = append(j.subByExt[k], idx)
	}

	return results
}

// SubmitDevice enqueues a device activity, attempting to join it against pending
// host submissions or queuing it in the bounded FIFO buffer.
func (j *Joiner) SubmitDevice(act DeviceActivity) []JoinResult {
	j.mu.Lock()
	defer j.mu.Unlock()

	j.metrics.TotalActivities++
	var results []JoinResult

	// Check late attach: if activity start timestamp is before late attach cutoff
	// and no host submission is immediately found
	var matchedSub *HostSubmission
	var matchedIdx = -1

	// Check native correlation IDs (including memcpy runtime correlation)
	corrs := []uint32{act.NativeCorrelationID}
	if act.RuntimeCorrelationID != 0 {
		corrs = append(corrs, act.RuntimeCorrelationID)
	}

	var candidateIdx = -1
	for _, c := range corrs {
		if c == 0 {
			continue
		}
		k := corrKey{Process: act.Process, Correlation: c, GraphReplay: act.GraphReplay}
		if subIndices, ok := j.subByCorr[k]; ok && len(subIndices) > 0 {
			if len(subIndices) > 1 {
				// Ambiguous: multiple host submissions for the same correlation key
				res := JoinResult{
					Outcome:        OutcomeAmbiguous,
					Reason:         ReasonConflictingOperations,
					DeviceActivity: &act,
					RawIDs:         rawFromActivity(act),
					EvidenceRef:    act.EvidenceRef,
				}
				j.metrics.AmbiguousJoins++
				return []JoinResult{res}
			}
			if candidateIdx != -1 && candidateIdx != subIndices[0] {
				j.metrics.AmbiguousJoins++
				return []JoinResult{{Outcome: OutcomeAmbiguous, Reason: ReasonConflictingOperations, DeviceActivity: &act, RawIDs: rawFromActivity(act)}}
			}
			candidateIdx = subIndices[0]
			matchedIdx = candidateIdx
			s := j.pendingSubs[matchedIdx]
			matchedSub = &s
		}
	}

	// If native correlation didn't match, check external correlation if present
	if matchedSub == nil && act.ExternalCorrelationID != 0 {
		k := extKey{Process: act.Process, ExternalID: act.ExternalCorrelationID}
		if subIndices, ok := j.subByExt[k]; ok && len(subIndices) > 0 {
			if len(subIndices) > 1 {
				res := JoinResult{
					Outcome:        OutcomeAmbiguous,
					Reason:         ReasonConflictingOperations,
					DeviceActivity: &act,
					RawIDs:         rawFromActivity(act),
					EvidenceRef:    act.EvidenceRef,
				}
				j.metrics.AmbiguousJoins++
				return []JoinResult{res}
			}
			matchedIdx = subIndices[0]
			s := j.pendingSubs[matchedIdx]
			matchedSub = &s
		}
	}

	if matchedSub != nil {
		// Remove matched host submission from pending
		j.removePendingSub(matchedIdx)
		res := j.buildExactOrAmbiguousMatch(*matchedSub, act)
		j.updateMetrics(res.Outcome)
		return []JoinResult{res}
	}

	// No host match found. Check for late attach.
	if j.cfg.LateAttachCutoffNs != 0 && act.StartNs < j.cfg.LateAttachCutoffNs {
		res := JoinResult{
			Outcome:        OutcomeMissing,
			Reason:         ReasonLateAttach,
			DeviceActivity: &act,
			RawIDs:         rawFromActivity(act),
			EvidenceRef:    act.EvidenceRef,
		}
		j.metrics.MissingJoins++
		return []JoinResult{res}
	}

	// Check if this activity belongs to an active stream/context in any pending submission
	// without explicit correlation ID.
	// We only report this as Associated, never Exact.
	var streamMatches []HostSubmission
	for _, sub := range j.pendingSubs {
		if sub.Process == act.Process && sub.Context == act.Context && sub.NativeCorrelationID == 0 {
			streamMatches = append(streamMatches, sub)
		}
	}
	if len(streamMatches) == 1 {
		matched := streamMatches[0]
		res := JoinResult{
			Outcome:        OutcomeAssociated,
			Reason:         ReasonStreamContextMatch,
			OpID:           matched.OpID,
			ParentOpID:     matched.ParentOpID,
			Workload:       matched.Workload,
			HostSubmission: &matched,
			DeviceActivity: &act,
			Provenance:     matched.Provenance,
			RawIDs:         rawFromActivityAndSub(act, matched),
			EvidenceRef:    act.EvidenceRef,
		}
		j.metrics.AssociatedJoins++
		return []JoinResult{res}
	} else if len(streamMatches) > 1 {
		res := JoinResult{
			Outcome:        OutcomeAmbiguous,
			Reason:         ReasonStreamContextMatch,
			DeviceActivity: &act,
			RawIDs:         rawFromActivity(act),
			EvidenceRef:    act.EvidenceRef,
		}
		j.metrics.AmbiguousJoins++
		return []JoinResult{res}
	}

	// Enqueue into pending device activities
	if len(j.pendingActs) >= j.cfg.MaxPendingRecords {
		// Evict oldest activity via FIFO
		evicted := j.pendingActs[0]
		j.pendingActs = j.pendingActs[1:]
		j.rebuildActIndices()

		evictRes := JoinResult{
			Outcome:        OutcomeMissing,
			Reason:         ReasonExpiredFIFO,
			DeviceActivity: &evicted,
			RawIDs:         rawFromActivity(evicted),
			EvidenceRef:    evicted.EvidenceRef,
		}
		j.metrics.MissingJoins++
		results = append(results, evictRes)
	}

	actIdx := len(j.pendingActs)
	j.pendingActs = append(j.pendingActs, act)
	if act.NativeCorrelationID != 0 {
		k := corrKey{Process: act.Process, Correlation: act.NativeCorrelationID, GraphReplay: act.GraphReplay}
		j.actByCorr[k] = append(j.actByCorr[k], actIdx)
	}

	return results
}

// Flush emits all remaining unmatched pending submissions and activities as Missing.
func (j *Joiner) Flush() []JoinResult {
	j.mu.Lock()
	defer j.mu.Unlock()

	var results []JoinResult
	for _, sub := range j.pendingSubs {
		results = append(results, JoinResult{
			Outcome:        OutcomeMissing,
			Reason:         ReasonNoDeviceActivity,
			OpID:           sub.OpID,
			ParentOpID:     sub.ParentOpID,
			Workload:       sub.Workload,
			HostSubmission: &sub,
			Provenance:     sub.Provenance,
			RawIDs: RawIDs{
				ProcessGenID:        sub.Process,
				NativeCorrelationID: sub.NativeCorrelationID,
				ContextStream:       sub.Context,
				GraphReplay:         sub.GraphReplay,
				TID:                 sub.TID,
			},
			EvidenceRef: sub.EvidenceRef,
		})
		j.metrics.MissingJoins++
	}
	j.pendingSubs = nil
	j.subByCorr = make(map[corrKey][]int)
	j.subByExt = make(map[extKey][]int)

	for _, act := range j.pendingActs {
		if act.NativeCorrelationID == 0 && act.StartNs == 0 {
			continue
		}
		results = append(results, JoinResult{
			Outcome:        OutcomeMissing,
			Reason:         ReasonNoHostSubmission,
			DeviceActivity: &act,
			RawIDs:         rawFromActivity(act),
			EvidenceRef:    act.EvidenceRef,
		})
		j.metrics.MissingJoins++
	}
	j.pendingActs = nil
	j.actByCorr = make(map[corrKey][]int)

	return results
}

func (j *Joiner) buildExactOrAmbiguousMatch(sub HostSubmission, act DeviceActivity) JoinResult {
	raw := rawFromActivityAndSub(act, sub)

	// If process generation does not match, never produce exact match
	if sub.Process != act.Process {
		return JoinResult{
			Outcome:        OutcomeMissing,
			Reason:         ReasonProcessGenMismatch,
			OpID:           sub.OpID,
			ParentOpID:     sub.ParentOpID,
			Workload:       sub.Workload,
			HostSubmission: &sub,
			DeviceActivity: &act,
			Provenance:     sub.Provenance,
			RawIDs:         raw,
			EvidenceRef:    act.EvidenceRef,
		}
	}

	// Context, graph replay, and a complete process generation must agree even
	// when native IDs match. Native IDs are local and may be reused.
	if sub.Process.PID == 0 || sub.Process.Generation == 0 || sub.OpID == 0 || sub.Context != act.Context || sub.GraphReplay != act.GraphReplay {
		return JoinResult{Outcome: OutcomeAmbiguous, Reason: ReasonConflictingOperations,
			HostSubmission: &sub, DeviceActivity: &act, RawIDs: raw, EvidenceRef: act.EvidenceRef}
	}

	// Verify external correlation ID if present on activity
	if act.ExternalCorrelationID != 0 && uint64(sub.OpID) != act.ExternalCorrelationID {
		return JoinResult{
			Outcome:        OutcomeAmbiguous,
			Reason:         ReasonConflictingExternalID,
			OpID:           sub.OpID,
			ParentOpID:     sub.ParentOpID,
			Workload:       sub.Workload,
			HostSubmission: &sub,
			DeviceActivity: &act,
			Provenance:     sub.Provenance,
			RawIDs:         raw,
			EvidenceRef:    act.EvidenceRef,
		}
	}

	reason := ReasonExplicitNativeMatch
	if !sub.GraphReplay.IsZero() || !act.GraphReplay.IsZero() {
		if sub.GraphReplay != act.GraphReplay {
			return JoinResult{
				Outcome:        OutcomeMissing,
				Reason:         ReasonNoHostSubmission,
				DeviceActivity: &act,
				RawIDs:         raw,
				EvidenceRef:    act.EvidenceRef,
			}
		}
		reason = ReasonGraphReplayMatch
	} else if act.ExternalCorrelationID != 0 {
		reason = ReasonExplicitExternalMatch
	}

	return JoinResult{
		Outcome:        OutcomeExact,
		Reason:         reason,
		OpID:           sub.OpID,
		ParentOpID:     sub.ParentOpID,
		Workload:       sub.Workload,
		HostSubmission: &sub,
		DeviceActivity: &act,
		Provenance:     sub.Provenance,
		RawIDs:         raw,
		EvidenceRef:    act.EvidenceRef,
	}
}

func (j *Joiner) updateMetrics(o Outcome) {
	switch o {
	case OutcomeExact:
		j.metrics.ExactJoins++
	case OutcomeAssociated:
		j.metrics.AssociatedJoins++
	case OutcomeAmbiguous:
		j.metrics.AmbiguousJoins++
	case OutcomeMissing:
		j.metrics.MissingJoins++
	}
}

func (j *Joiner) removePendingSub(idx int) {
	j.pendingSubs = append(j.pendingSubs[:idx], j.pendingSubs[idx+1:]...)
	j.rebuildSubIndices()
}

func (j *Joiner) rebuildSubIndices() {
	j.subByCorr = make(map[corrKey][]int)
	j.subByExt = make(map[extKey][]int)
	for i, s := range j.pendingSubs {
		if s.NativeCorrelationID != 0 {
			k := corrKey{Process: s.Process, Correlation: s.NativeCorrelationID, GraphReplay: s.GraphReplay}
			j.subByCorr[k] = append(j.subByCorr[k], i)
		}
		if s.OpID != 0 {
			k := extKey{Process: s.Process, ExternalID: uint64(s.OpID)}
			j.subByExt[k] = append(j.subByExt[k], i)
		}
	}
}

func (j *Joiner) rebuildActIndices() {
	j.actByCorr = make(map[corrKey][]int)
	for i, a := range j.pendingActs {
		if a.NativeCorrelationID != 0 {
			k := corrKey{Process: a.Process, Correlation: a.NativeCorrelationID, GraphReplay: a.GraphReplay}
			j.actByCorr[k] = append(j.actByCorr[k], i)
		}
	}
}

func rawFromActivity(act DeviceActivity) RawIDs {
	return RawIDs{
		ProcessGenID:          act.Process,
		NativeCorrelationID:   act.NativeCorrelationID,
		RuntimeCorrelationID:  act.RuntimeCorrelationID,
		ExternalCorrelationID: act.ExternalCorrelationID,
		ContextStream:         act.Context,
		GraphReplay:           act.GraphReplay,
	}
}

func rawFromActivityAndSub(act DeviceActivity, sub HostSubmission) RawIDs {
	r := rawFromActivity(act)
	r.TID = sub.TID
	return r
}
