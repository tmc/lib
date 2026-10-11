package ncclinspector

import (
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"
)

// Sentinel errors for collective stall identity validation.
var (
	ErrTruncatedHash       = errors.New("ncclinspector: truncated communicator hash")
	ErrInvalidHash         = errors.New("ncclinspector: invalid communicator hash")
	ErrRankMismatch        = errors.New("ncclinspector: rank ordinal out of bounds")
	ErrDuplicateSequence   = errors.New("ncclinspector: duplicate collective sequence on rank")
	ErrRankRestarted       = errors.New("ncclinspector: rank restart or sequence rewind detected")
	ErrIncompatibleClocks  = errors.New("ncclinspector: incompatible clock domains")
	ErrMissingRanks        = errors.New("ncclinspector: missing collective rank observation")
	ErrObservationCapacity = errors.New("ncclinspector: observation capacity exhausted")
)

// Stage represents one distinct phase in a collective operation's lifecycle.
type Stage string

const (
	StageHostEnqueue    Stage = "host_enqueue"
	StageDeviceActivity Stage = "device_activity"
	StageCompletion     Stage = "completion"
)

// ClockDomain identifies an independent timing source.
type ClockDomain string

const (
	ClockDomainHostMonotonic ClockDomain = "host_monotonic"
	ClockDomainHostRealtime  ClockDomain = "host_realtime"
	ClockDomainGPUDevice     ClockDomain = "gpu_device"
)

// ClockMapping defines a clock domain with measured offset and uncertainty.
type ClockMapping struct {
	Domain        ClockDomain
	OffsetNS      int64
	UncertaintyNS int64 // measured uncertainty (+/- nanoseconds, must be non-negative)
	// ReferenceDomain and ReferenceEpoch identify the destination clock.
	// EvidenceRef must name the measurement used for an inter-host mapping.
	ReferenceDomain ClockDomain
	ReferenceEpoch  string
	EvidenceRef     string
}

// BoundedTimestamp represents a timestamp in nanoseconds with measured uncertainty.
type BoundedTimestamp struct {
	Nanoseconds   int64
	UncertaintyNS int64
	Domain        ClockDomain
}

// Aligned maps a bounded timestamp to the reference timeline using a clock mapping.
func (bt BoundedTimestamp) Aligned(m ClockMapping) (BoundedTimestamp, error) {
	if bt.Domain != "" && m.Domain != "" && bt.Domain != m.Domain {
		return BoundedTimestamp{}, fmt.Errorf("%w: timestamp domain %q != mapping domain %q", ErrIncompatibleClocks, bt.Domain, m.Domain)
	}
	if bt.Domain == "" || m.Domain == "" || bt.UncertaintyNS < 0 || m.UncertaintyNS < 0 || bt.UncertaintyNS > math.MaxInt64-m.UncertaintyNS {
		return BoundedTimestamp{}, fmt.Errorf("%w: invalid clock uncertainty or domain", ErrIncompatibleClocks)
	}
	if m.OffsetNS > 0 && bt.Nanoseconds > math.MaxInt64-m.OffsetNS || m.OffsetNS < 0 && bt.Nanoseconds < math.MinInt64-m.OffsetNS {
		return BoundedTimestamp{}, fmt.Errorf("%w: timestamp overflow", ErrIncompatibleClocks)
	}
	unc := bt.UncertaintyNS + m.UncertaintyNS
	domain := m.Domain
	if m.ReferenceDomain != "" {
		domain = m.ReferenceDomain
	}
	return BoundedTimestamp{
		Nanoseconds:   bt.Nanoseconds + m.OffsetNS,
		UncertaintyNS: unc,
		Domain:        domain,
	}, nil
}

// CollectiveObservation captures validated evidence for one rank's participation
// in a collective operation.
type CollectiveObservation struct {
	CommunicatorHash string // Must be full 18-char hex string: 0x followed by 16 hex digits
	Rank             int
	TotalRanks       int
	Sequence         uint64
	Operation        string
	Bytes            uint64
	HostPID          int
	Hostname         string
	ProducerRevision string
	// ClockEpoch identifies an explicitly mapped common clock epoch. A clock
	// domain name alone does not establish synchronization across hosts.
	ClockEpoch       string
	ClockEvidenceRef string

	// Explicit separate stages
	HostEnqueueStart BoundedTimestamp
	HostEnqueueEnd   BoundedTimestamp
	DeviceStart      BoundedTimestamp
	DeviceEnd        BoundedTimestamp
	Completion       BoundedTimestamp

	HasHostEnqueue    bool
	HasDeviceActivity bool
	HasCompletion     bool

	// Transport observations (missing transport observations remain explicitly false)
	TransportObserved bool
}

// DelayDiagnosis classifies the attributed cause of a collective delay.
type DelayDiagnosis string

const (
	// DiagnosisNormal indicates both ranks completed within expected bounds without notable skew.
	DiagnosisNormal DelayDiagnosis = "normal"

	// DiagnosisHostEnqueueSkew indicates one rank enqueued the collective significantly later on the host CPU.
	DiagnosisHostEnqueueSkew DelayDiagnosis = "host_enqueue_skew"

	// DiagnosisDeviceStartDelay indicates peer enqueued on time but was delayed starting on device due to prior GPU work.
	DiagnosisDeviceStartDelay DelayDiagnosis = "device_start_delay"

	// DiagnosisExecutionImbalance indicates both ranks started on device concurrently but one rank took longer in kernel execution.
	DiagnosisExecutionImbalance DelayDiagnosis = "execution_imbalance"

	// DiagnosisOrderIndeterminate indicates measured timestamp difference between ranks is within clock uncertainty bounds.
	DiagnosisOrderIndeterminate DelayDiagnosis = "order_indeterminate"

	// DiagnosisPartialRank indicates one or more expected ranks are missing observations.
	DiagnosisPartialRank DelayDiagnosis = "partial_rank"

	// DiagnosisIncompatibleClocks indicates compared timestamps use different clock domains without valid mapping.
	DiagnosisIncompatibleClocks DelayDiagnosis = "incompatible_clocks"

	// DiagnosisIdentityRejected indicates input failed strict validation (e.g. truncated hash, duplicate sequence, rank restart, swapped rank).
	DiagnosisIdentityRejected DelayDiagnosis = "identity_rejected"
)

// RankDelayDetail provides per-rank breakdown across execution stages.
type RankDelayDetail struct {
	Rank             int
	HostEnqueueNS    int64
	DeviceActivityNS int64
	CompletionNS     int64
	TotalObservedNS  int64
}

// TwoRankDelayReport records the attribution findings for a two-rank collective.
type TwoRankDelayReport struct {
	CommunicatorHash string
	Sequence         uint64
	Operation        string

	Diagnosis DelayDiagnosis
	Reason    string
	Admitted  bool

	// DelayedRank is the attributed culprit rank index, or -1 if indeterminate or not applicable.
	DelayedRank int

	// Measured delay interval and clock uncertainty in nanoseconds
	DelayNS       int64
	UncertaintyNS int64

	// Stage breakdown per rank
	Ranks map[int]RankDelayDetail

	// Partial-rank reporting
	ReportedRanks []int
	MissingRanks  []int

	// Explicit transport observation status
	TransportObserved bool
}

// ValidateCommunicatorHash checks that hash is a full 64-bit hex hash (18 chars: "0x" followed by 16 hex digits).
// An earlier 17-byte C buffer truncation bug emitted 16-character strings ("0x" + 14 hex chars), which is rejected.
func ValidateCommunicatorHash(hash string) error {
	if len(hash) < 18 {
		if strings.HasPrefix(hash, "0x") && len(hash) == 16 {
			return fmt.Errorf("%w: hash %q has 14 hex digits (want 16)", ErrTruncatedHash, hash)
		}
		return fmt.Errorf("%w: hash %q length %d < 18", ErrTruncatedHash, hash, len(hash))
	}
	if len(hash) > 18 {
		return fmt.Errorf("%w: hash %q length %d > 18", ErrInvalidHash, hash, len(hash))
	}
	if !strings.HasPrefix(hash, "0x") && !strings.HasPrefix(hash, "0X") {
		return fmt.Errorf("%w: hash %q missing 0x prefix", ErrInvalidHash, hash)
	}
	rawHex := hash[2:]
	if _, err := hex.DecodeString(rawHex); err != nil {
		return fmt.Errorf("%w: hash %q contains non-hex characters", ErrInvalidHash, hash)
	}
	return nil
}

// ObservationFromRecord extracts and validates a CollectiveObservation from an Inspector Record.
func ObservationFromRecord(r Record, clock ClockMapping) (CollectiveObservation, error) {
	if r.Collective == nil {
		return CollectiveObservation{}, errors.New("ncclinspector: record does not contain collective observation")
	}
	if err := ValidateCommunicatorHash(r.Header.ID); err != nil {
		return CollectiveObservation{}, err
	}
	if r.Header.Rank < 0 || r.Header.Rank >= r.Header.Ranks {
		return CollectiveObservation{}, fmt.Errorf("%w: rank %d not in [0, %d)", ErrRankMismatch, r.Header.Rank, r.Header.Ranks)
	}
	if r.Metadata.PID <= 0 {
		return CollectiveObservation{}, errors.New("ncclinspector: invalid process ID")
	}

	if clock.Domain == "" || clock.UncertaintyNS < 0 {
		return CollectiveObservation{}, fmt.Errorf("%w: missing domain or negative uncertainty", ErrIncompatibleClocks)
	}
	if ts := r.Collective.EventTraceTS; ts != nil {
		limit := uint64(math.MaxInt64 / 1000)
		if ts.CollectiveStartTS > limit || ts.CollectiveStopTS > limit {
			return CollectiveObservation{}, errors.New("ncclinspector: timestamp overflow")
		}
		for _, ke := range ts.KernelEvents {
			if ke.KernelStartTS > limit || ke.KernelStopTS > limit || ke.KernelRecordTS > limit || ke.KernelStopTS < ke.KernelStartTS {
				return CollectiveObservation{}, errors.New("ncclinspector: invalid kernel interval")
			}
		}
	}

	obs := CollectiveObservation{
		CommunicatorHash: r.Header.ID,
		Rank:             r.Header.Rank,
		TotalRanks:       r.Header.Ranks,
		Sequence:         r.Collective.Sequence,
		Operation:        r.Collective.Name,
		Bytes:            r.Collective.Bytes,
		HostPID:          r.Metadata.PID,
		Hostname:         r.Metadata.Hostname,
		ProducerRevision: r.Metadata.GitRevision,
	}

	if r.Collective.EventTraceTS != nil {
		ts := r.Collective.EventTraceTS
		if ts.CollectiveStartTS > 0 && ts.CollectiveStopTS >= ts.CollectiveStartTS {
			obs.HostEnqueueStart = BoundedTimestamp{
				Nanoseconds:   int64(ts.CollectiveStartTS) * 1000,
				UncertaintyNS: clock.UncertaintyNS,
				Domain:        clock.Domain,
			}
			obs.HostEnqueueEnd = BoundedTimestamp{
				Nanoseconds:   int64(ts.CollectiveStopTS) * 1000,
				UncertaintyNS: clock.UncertaintyNS,
				Domain:        clock.Domain,
			}
			obs.HasHostEnqueue = true
		}

		if len(ts.KernelEvents) > 0 {
			var minStart, maxStop uint64
			for i, ke := range ts.KernelEvents {
				if i == 0 || ke.KernelStartTS < minStart {
					minStart = ke.KernelStartTS
				}
				if i == 0 || ke.KernelStopTS > maxStop {
					maxStop = ke.KernelStopTS
				}
			}
			if minStart > 0 && maxStop >= minStart {
				obs.DeviceStart = BoundedTimestamp{
					Nanoseconds:   int64(minStart) * 1000,
					UncertaintyNS: clock.UncertaintyNS,
					Domain:        clock.Domain,
				}
				obs.DeviceEnd = BoundedTimestamp{
					Nanoseconds:   int64(maxStop) * 1000,
					UncertaintyNS: clock.UncertaintyNS,
					Domain:        clock.Domain,
				}
				obs.HasDeviceActivity = true
			}
			// kernel_record_ts is record publication, not independently observed
			// collective completion. Callers must supply completion evidence.
		}
	}
	// Aggregate duration and record publication time do not establish a
	// device start/end interval. Preserve missing stage evidence.
	if clock.ReferenceEpoch != "" || clock.ReferenceDomain != "" || clock.EvidenceRef != "" || clock.OffsetNS != 0 {
		if clock.ReferenceEpoch == "" || clock.ReferenceDomain == "" || clock.EvidenceRef == "" {
			return CollectiveObservation{}, fmt.Errorf("%w: incomplete mapping evidence", ErrIncompatibleClocks)
		}
		align := func(ts *BoundedTimestamp) error {
			// The record has no independent timestamp uncertainty; the mapping
			// uncertainty is added exactly once by Aligned.
			ts.UncertaintyNS = 0
			mapped, err := ts.Aligned(clock)
			if err != nil {
				return err
			}
			*ts = mapped
			return nil
		}
		if obs.HasHostEnqueue {
			if err := align(&obs.HostEnqueueStart); err != nil {
				return CollectiveObservation{}, err
			}
			if err := align(&obs.HostEnqueueEnd); err != nil {
				return CollectiveObservation{}, err
			}
		}
		if obs.HasDeviceActivity {
			if err := align(&obs.DeviceStart); err != nil {
				return CollectiveObservation{}, err
			}
			if err := align(&obs.DeviceEnd); err != nil {
				return CollectiveObservation{}, err
			}
		}
		if obs.HasCompletion {
			if err := align(&obs.Completion); err != nil {
				return CollectiveObservation{}, err
			}
		}
		obs.ClockEpoch = clock.ReferenceEpoch
		obs.ClockEvidenceRef = clock.EvidenceRef
	}

	return obs, nil
}

// TwoRankAttributor evaluates observations across two ranks of a collective.
type TwoRankAttributor struct {
	mu           sync.Mutex
	activePID    map[string]map[int]int    // [commHash][rank] -> pid
	lastSeq      map[string]map[int]uint64 // [commHash][rank] -> last observed sequence
	observations map[string]map[uint64]map[int]CollectiveObservation
	count        int
	limit        int
	rankSources  map[string]map[int]rankSource
}

type rankSource struct{ hostname, revision string }

// NewTwoRankAttributor creates an initialized TwoRankAttributor.
func NewTwoRankAttributor() *TwoRankAttributor {
	return &TwoRankAttributor{
		activePID:    make(map[string]map[int]int),
		lastSeq:      make(map[string]map[int]uint64),
		observations: make(map[string]map[uint64]map[int]CollectiveObservation),
		limit:        4096,
		rankSources:  make(map[string]map[int]rankSource),
	}
}

// AddObservation validates and stores an observation for a collective operation.
// It detects and rejects truncated hashes, out-of-range ranks, duplicate sequence numbers,
// and rank restarts (PID changes or sequence rewinds).
func (a *TwoRankAttributor) AddObservation(obs CollectiveObservation) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	if err := ValidateCommunicatorHash(obs.CommunicatorHash); err != nil {
		return err
	}
	if obs.TotalRanks != 2 {
		return fmt.Errorf("ncclinspector: TwoRankAttributor requires totalRanks=2, got %d", obs.TotalRanks)
	}
	if obs.Rank < 0 || obs.Rank >= obs.TotalRanks {
		return fmt.Errorf("%w: rank %d out of bounds for total ranks %d", ErrRankMismatch, obs.Rank, obs.TotalRanks)
	}
	if obs.HostPID <= 0 {
		return errors.New("ncclinspector: invalid process ID")
	}
	if a.count >= a.limit {
		return ErrObservationCapacity
	}
	if sources := a.rankSources[obs.CommunicatorHash]; sources != nil {
		if prior, exists := sources[obs.Rank]; exists && prior != (rankSource{obs.Hostname, obs.ProducerRevision}) {
			return fmt.Errorf("%w: rank host or producer changed", ErrRankRestarted)
		}
	}

	check := func(ts BoundedTimestamp) error {
		if ts.Domain == "" || ts.Nanoseconds < 0 || ts.Nanoseconds > math.MaxInt64/4 || ts.UncertaintyNS < 0 || ts.UncertaintyNS > math.MaxInt64/4 {
			return fmt.Errorf("%w: invalid timestamp or uncertainty", ErrIncompatibleClocks)
		}
		return nil
	}
	interval := func(start, end BoundedTimestamp) error {
		if err := check(start); err != nil {
			return err
		}
		if err := check(end); err != nil {
			return err
		}
		if start.Domain != end.Domain || end.Nanoseconds < start.Nanoseconds {
			return fmt.Errorf("%w: reversed interval or incompatible endpoints", ErrIncompatibleClocks)
		}
		return nil
	}
	if obs.HasHostEnqueue {
		if err := interval(obs.HostEnqueueStart, obs.HostEnqueueEnd); err != nil {
			return err
		}
	}
	if obs.HasDeviceActivity {
		if err := interval(obs.DeviceStart, obs.DeviceEnd); err != nil {
			return err
		}
	}
	if obs.HasCompletion {
		if err := check(obs.Completion); err != nil {
			return err
		}
		if obs.HasDeviceActivity && (obs.Completion.Domain != obs.DeviceEnd.Domain || obs.Completion.Nanoseconds < obs.DeviceEnd.Nanoseconds) {
			return fmt.Errorf("%w: completion precedes device end", ErrIncompatibleClocks)
		}
	}
	// Verify active process identity and sequence monotonicity for communicator & rank
	pids, ok := a.activePID[obs.CommunicatorHash]
	if !ok {
		if len(a.activePID) >= a.limit {
			return ErrObservationCapacity
		}
		pids = make(map[int]int)
		a.activePID[obs.CommunicatorHash] = pids
	}
	if priorPID, exists := pids[obs.Rank]; exists {
		if priorPID != obs.HostPID {
			return fmt.Errorf("%w: PID changed from %d to %d on comm %s rank %d", ErrRankRestarted, priorPID, obs.HostPID, obs.CommunicatorHash, obs.Rank)
		}
	} else {
		pids[obs.Rank] = obs.HostPID
	}

	seqs, ok := a.lastSeq[obs.CommunicatorHash]
	if !ok {
		seqs = make(map[int]uint64)
		a.lastSeq[obs.CommunicatorHash] = seqs
	}
	if last, exists := seqs[obs.Rank]; exists {
		if obs.Sequence == last {
			return ErrDuplicateSequence
		}
		if obs.Sequence < last {
			return fmt.Errorf("%w: sequence rewind from %d to %d on comm %s rank %d", ErrRankRestarted, last, obs.Sequence, obs.CommunicatorHash, obs.Rank)
		}
	}

	// Store observation and check for duplicate sequence
	commMap, ok := a.observations[obs.CommunicatorHash]
	if !ok {
		commMap = make(map[uint64]map[int]CollectiveObservation)
		a.observations[obs.CommunicatorHash] = commMap
	}
	seqMap, ok := commMap[obs.Sequence]
	if !ok {
		seqMap = make(map[int]CollectiveObservation)
		commMap[obs.Sequence] = seqMap
	}
	if _, exists := seqMap[obs.Rank]; exists {
		return fmt.Errorf("%w: rank %d already observed for comm %s seq %d", ErrDuplicateSequence, obs.Rank, obs.CommunicatorHash, obs.Sequence)
	}

	seqMap[obs.Rank] = obs
	if a.rankSources[obs.CommunicatorHash] == nil {
		a.rankSources[obs.CommunicatorHash] = make(map[int]rankSource)
	}
	a.rankSources[obs.CommunicatorHash][obs.Rank] = rankSource{obs.Hostname, obs.ProducerRevision}
	a.count++
	seqs[obs.Rank] = obs.Sequence
	return nil
}

// Attribute analyzes the two-rank collective observation for the given communicator and sequence.
// It reports partial-rank status if only one rank reported, rejects incompatible clocks,
// withholds rank-order when clock uncertainty exceeds the compared interval, and attributes
// delays to host enqueue skew, device start delay, or execution duration imbalance.
func (a *TwoRankAttributor) Attribute(commHash string, seq uint64) (TwoRankDelayReport, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	if err := ValidateCommunicatorHash(commHash); err != nil {
		return TwoRankDelayReport{
			CommunicatorHash: commHash,
			Sequence:         seq,
			Diagnosis:        DiagnosisIdentityRejected,
			Reason:           err.Error(),
			DelayedRank:      -1,
		}, err
	}

	commMap, ok := a.observations[commHash]
	if !ok {
		return TwoRankDelayReport{}, fmt.Errorf("ncclinspector: no observations for communicator %s", commHash)
	}
	seqMap, ok := commMap[seq]
	if !ok {
		return TwoRankDelayReport{}, fmt.Errorf("ncclinspector: no observations for communicator %s sequence %d", commHash, seq)
	}

	var reportedRanks []int
	for r := 0; r < 2; r++ {
		if _, present := seqMap[r]; present {
			reportedRanks = append(reportedRanks, r)
		}
	}

	// Partial rank reporting: only 1 of 2 ranks observed
	if len(reportedRanks) < 2 {
		missing := 0
		if len(reportedRanks) == 1 && reportedRanks[0] == 0 {
			missing = 1
		}
		var op string
		if len(reportedRanks) > 0 {
			op = seqMap[reportedRanks[0]].Operation
		}
		return TwoRankDelayReport{
			CommunicatorHash:  commHash,
			Sequence:          seq,
			Operation:         op,
			Diagnosis:         DiagnosisPartialRank,
			Reason:            fmt.Sprintf("partial-rank observation: rank %d present, rank %d missing", reportedRanks[0], missing),
			Admitted:          false,
			DelayedRank:       -1,
			ReportedRanks:     reportedRanks,
			MissingRanks:      []int{missing},
			TransportObserved: false,
		}, nil
	}

	r0 := seqMap[0]
	r1 := seqMap[1]

	details := make(map[int]RankDelayDetail)
	for _, r := range []CollectiveObservation{r0, r1} {
		var hostNS, devNS, compNS int64
		if r.HasHostEnqueue {
			hostNS = r.HostEnqueueEnd.Nanoseconds - r.HostEnqueueStart.Nanoseconds
		}
		if r.HasDeviceActivity {
			devNS = r.DeviceEnd.Nanoseconds - r.DeviceStart.Nanoseconds
		}
		if r.HasCompletion && r.HasDeviceActivity {
			compNS = r.Completion.Nanoseconds - r.DeviceEnd.Nanoseconds
		}
		total := hostNS + devNS + compNS
		details[r.Rank] = RankDelayDetail{
			Rank:             r.Rank,
			HostEnqueueNS:    hostNS,
			DeviceActivityNS: devNS,
			CompletionNS:     compNS,
			TotalObservedNS:  total,
		}
	}

	report := TwoRankDelayReport{
		CommunicatorHash:  commHash,
		Sequence:          seq,
		Operation:         r0.Operation,
		DelayedRank:       -1,
		Ranks:             details,
		ReportedRanks:     []int{0, 1},
		MissingRanks:      nil,
		TransportObserved: r0.TransportObserved && r1.TransportObserved,
	}

	if r0.Operation == "" || r0.Operation != r1.Operation || r0.Bytes != r1.Bytes {
		report.Diagnosis = DiagnosisIdentityRejected
		report.Reason = "collective operation or byte count differs between ranks"
		return report, nil
	}
	if !(r0.HasHostEnqueue && r1.HasHostEnqueue) && !(r0.HasDeviceActivity && r1.HasDeviceActivity) {
		report.Diagnosis = DiagnosisPartialRank
		report.Reason = "no shared observed stage"
		return report, nil
	}
	if r0.Hostname != r1.Hostname && (r0.Hostname == "" || r1.Hostname == "" || r0.ClockEpoch == "" || r0.ClockEpoch != r1.ClockEpoch || r0.ClockEvidenceRef == "" || r1.ClockEvidenceRef == "") {
		report.Diagnosis = DiagnosisIncompatibleClocks
		report.Reason = "cross-host clock mapping evidence missing"
		return report, nil
	}
	// Incompatible clock domains check
	if (r0.HasHostEnqueue && r1.HasHostEnqueue && r0.HostEnqueueStart.Domain != r1.HostEnqueueStart.Domain) ||
		(r0.HasDeviceActivity && r1.HasDeviceActivity && r0.DeviceStart.Domain != r1.DeviceStart.Domain) {
		report.Diagnosis = DiagnosisIncompatibleClocks
		report.Reason = "incompatible clock domains between rank 0 and rank 1"
		report.Admitted = false
		return report, nil
	}

	// Stage 1: Host Enqueue Comparison
	var hostSkewIndeterminate bool
	if r0.HasHostEnqueue && r1.HasHostEnqueue {
		uncertainty := r0.HostEnqueueStart.UncertaintyNS + r1.HostEnqueueStart.UncertaintyNS
		delta := r1.HostEnqueueStart.Nanoseconds - r0.HostEnqueueStart.Nanoseconds
		absDelta := abs(delta)

		if absDelta > uncertainty {
			// Definite host enqueue skew established
			report.DelayNS = absDelta
			report.UncertaintyNS = uncertainty
			report.Admitted = true
			if delta > 0 {
				report.DelayedRank = 1
				report.Diagnosis = DiagnosisHostEnqueueSkew
				report.Reason = fmt.Sprintf("rank 1 enqueued %d ns after rank 0 (uncertainty +/- %d ns)", absDelta, uncertainty)
			} else {
				report.DelayedRank = 0
				report.Diagnosis = DiagnosisHostEnqueueSkew
				report.Reason = fmt.Sprintf("rank 0 enqueued %d ns after rank 1 (uncertainty +/- %d ns)", absDelta, uncertainty)
			}
			return report, nil
		}
		if absDelta > 0 && absDelta <= uncertainty {
			hostSkewIndeterminate = true
			report.DelayNS = absDelta
			report.UncertaintyNS = uncertainty
		}
		// If absDelta <= uncertainty, host enqueue is concurrent within uncertainty bounds.
		// Proceed to check device start and execution stages.
	}

	// Stage 2: Device Activity Start Comparison
	var devSkewIndeterminate bool
	if r0.HasDeviceActivity && r1.HasDeviceActivity {
		uncertainty := r0.DeviceStart.UncertaintyNS + r1.DeviceStart.UncertaintyNS
		delta := r1.DeviceStart.Nanoseconds - r0.DeviceStart.Nanoseconds
		absDelta := abs(delta)

		if absDelta > uncertainty {
			report.DelayNS = absDelta
			report.UncertaintyNS = uncertainty
			report.Admitted = true
			if delta > 0 {
				report.DelayedRank = 1
				report.Diagnosis = DiagnosisDeviceStartDelay
				report.Reason = fmt.Sprintf("rank 1 device kernel started %d ns after rank 0 (uncertainty +/- %d ns)", absDelta, uncertainty)
			} else {
				report.DelayedRank = 0
				report.Diagnosis = DiagnosisDeviceStartDelay
				report.Reason = fmt.Sprintf("rank 0 device kernel started %d ns after rank 1 (uncertainty +/- %d ns)", absDelta, uncertainty)
			}
			return report, nil
		}

		if absDelta > 0 && absDelta <= uncertainty {
			devSkewIndeterminate = true
			report.DelayNS = absDelta
			report.UncertaintyNS = uncertainty
		}

		// Stage 3: Execution Duration Imbalance
		dur0 := r0.DeviceEnd.Nanoseconds - r0.DeviceStart.Nanoseconds
		dur1 := r1.DeviceEnd.Nanoseconds - r1.DeviceStart.Nanoseconds
		durDelta := dur1 - dur0
		absDurDelta := abs(durDelta)
		durUncertainty := r0.DeviceStart.UncertaintyNS + r0.DeviceEnd.UncertaintyNS + r1.DeviceStart.UncertaintyNS + r1.DeviceEnd.UncertaintyNS

		if absDurDelta > durUncertainty {
			report.DelayNS = absDurDelta
			report.UncertaintyNS = durUncertainty
			report.Admitted = true
			report.Diagnosis = DiagnosisExecutionImbalance
			if durDelta > 0 {
				report.DelayedRank = 1
				report.Reason = fmt.Sprintf("rank 1 device execution duration exceeded rank 0 by %d ns", absDurDelta)
			} else {
				report.DelayedRank = 0
				report.Reason = fmt.Sprintf("rank 0 device execution duration exceeded rank 1 by %d ns", absDurDelta)
			}
			return report, nil
		}
	}

	// If any stage had a non-zero delta within clock uncertainty, withhold rank-order
	if hostSkewIndeterminate || devSkewIndeterminate {
		report.Diagnosis = DiagnosisOrderIndeterminate
		report.Reason = "measured timestamp difference is within clock uncertainty bounds; rank-order and peer causality withheld"
		report.Admitted = false
		report.DelayedRank = -1
		return report, nil
	}

	if !r0.HasCompletion || !r1.HasCompletion {
		report.Diagnosis = DiagnosisPartialRank
		report.Reason = "completion evidence missing"
		return report, nil
	}
	// Both ranks balanced and normal
	report.Diagnosis = DiagnosisNormal
	report.Reason = "two-rank collective completed normally within bounded uncertainty"
	report.Admitted = true
	report.DelayedRank = -1
	return report, nil
}

// Forget releases the retained observations for a reported collective.
// Rank identity and the last accepted sequence remain retained to detect
// restarts and rewinds. A new capture run should use a new attributor.
func (a *TwoRankAttributor) Forget(commHash string, seq uint64) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if sequences := a.observations[commHash]; sequences != nil {
		a.count -= len(sequences[seq])
		delete(sequences, seq)
	}
}

func abs(n int64) int64 {
	if n < 0 {
		return -n
	}
	return n
}
