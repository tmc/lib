package ncclinspector

import (
	"errors"
	"fmt"
	"slices"
	"strings"
)

// Refusals returned by [BindProfiler].
var (
	// ErrProfilerBinding reports a supplied identity that does not match the
	// trace or the Inspector record.
	ErrProfilerBinding = errors.New("profiler: binding refused")

	// ErrProfilerUnavailable reports a required fact that the trace cannot
	// supply.
	ErrProfilerUnavailable = errors.New("profiler: required fact unavailable")
)

// ProfilerNeed names a fact class a caller requires before a binding is
// returned. Each unavailable need is refused instead of reported as zero.
type ProfilerNeed string

const (
	// NeedPhases requires initial_sync, compute and final_sync, each admitted
	// by the profile with a nonempty GPU timer interval, on every bound channel.
	NeedPhases ProfilerNeed = "phases"

	// NeedLoss requires family loss or population counts. The example
	// profiler exports none.
	NeedLoss ProfilerNeed = "loss"

	// NeedProgressAge requires a progress counter refresh age. The example
	// profiler exports none.
	NeedProgressAge ProfilerNeed = "progress_age"

	// NeedCrossRankTimeline requires phases placed on a common timeline.
	// Raw GPU timers carry no calibration to any reference clock here.
	NeedCrossRankTimeline ProfilerNeed = "cross_rank_timeline"
)

// ProfilerBinding is the independently supplied identity that associates one
// example-profiler COLL scope with one Inspector record. Every field is
// asserted by the caller; BindProfiler checks them for agreement with the
// trace and the record but cannot verify their authenticity.
type ProfilerBinding struct {
	// Communicator is the full hash, 0x followed by 16 hex digits. Case is
	// ignored.
	Communicator string
	Rank         int
	// Incarnation names the communicator incarnation. It is never inferred
	// from the hash, process ID or file name, and nothing in the trace or
	// the record can verify it; it is carried as asserted.
	Incarnation string
	// Host is the hostname asserted for the process; it must equal the
	// Inspector record's hostname. The trace carries no hostname.
	Host string
	PID  int
	// Operation and Sequence select the enclosing COLL scope. Sequence is
	// per operation kind.
	Operation string
	Sequence  uint64
	// Channels lists the KernelCh channels to bind. Each must appear exactly
	// once in the selected COLL scope and, when the record lists kernel
	// events, among the record's observed channel IDs.
	Channels []int
	// SizeUnverified accepts a datatype the producer prints without a known
	// element size. Otherwise Count must convert to the record's byte size.
	SizeUnverified bool
	// Evidence names the source of the supplied identity.
	Evidence string
	Require  []ProfilerNeed
}

// ProfilerAsserted holds the facts BindProfiler took from the caller.
type ProfilerAsserted struct {
	Binding ProfilerBinding
	// Build is the plugin ABI and build named in the capture context.
	Build string
	Clock ClockMapping
}

// ProfilerUnavailable names a fact family the binding cannot supply.
type ProfilerUnavailable struct{ Family, Reason string }

// ProfilerBound is one Inspector collective observation bound to its
// example-profiler phase facts. Observation is the unmodified result of
// [ObservationFromRecord]: phase timestamps never replace its stages.
// Trace fields are observed in the artifact; Asserted fields are not.
type ProfilerBound struct {
	Profile        ProfilerProfile
	ArtifactSHA256 string
	// Collective is the selected COLL scope as exported, with raw lines.
	Collective  ProfilerCollective
	Observation CollectiveObservation
	Asserted    ProfilerAsserted
	// Bytes is the producer Count converted to bytes, when known.
	Bytes       uint64
	BytesKnown  bool
	Channels    []ProfilerChannelResult
	Unavailable []ProfilerUnavailable
}

// ProfilerChannelResult is the phase result for one bound channel.
// StartGpuClk and StopGpuClk are raw nanoseconds on one GPU's timer.
type ProfilerChannelResult struct {
	Channel                 int
	StartGpuClk, StopGpuClk uint64
	// Withheld says why the channel and its phases are not admitted.
	Withheld       string
	KernelNS       int64
	KernelReported bool
	Phases         []ProfilerPhaseResult
	// Absent lists phase names the trace does not contain. The producer
	// skips absent endpoints and zero-duration boundaries, so absence is
	// not loss and not an observation of no progress.
	Absent []string
}

// ProfilerPhaseResult is one exported phase between two producer boundaries.
type ProfilerPhaseResult struct {
	Name     string
	ID       int
	From, To string
	// ReportedNS is the within-device duration from the raw GPU timers.
	ReportedNS int64
	Reported   bool
	// HostCallbackMicros is the printed host callback time, not a GPU start.
	HostCallbackMicros string
	// Reason says why the phase has no admitted duration.
	Reason string
	Raw    []byte
}

var phaseBounds = [...][2]string{
	{"begin", "after_open"},
	{"after_open", "before_close"},
	{"before_close", "end"},
}

// BindProfiler associates one COLL scope of t with the Inspector record r
// using only the supplied binding: full communicator hash, rank, incarnation,
// process, operation, sequence and enclosing channel scopes. Event IDs, the
// constant tid and timestamps never select anything. The record must agree
// with the binding and the trace; its observation comes from
// [ObservationFromRecord] with the supplied clock mapping and is not
// augmented. The result reports reported GPU durations per phase and lists
// every fact family it cannot supply. Required needs that cannot be met are
// refused with [ErrProfilerUnavailable]. The trace must come from
// [ReadProfilerTrace]; that provenance marker survives copying the struct,
// and caller edits to its exported fields after reading are not detected.
func BindProfiler(t *ProfilerTrace, b ProfilerBinding, r Record, clock ClockMapping) (*ProfilerBound, error) {
	if t == nil || !t.read {
		return nil, fmt.Errorf("%w: trace was not produced by ReadProfilerTrace", ErrProfilerBinding)
	}
	if err := ValidateCommunicatorHash(b.Communicator); err != nil {
		return nil, fmt.Errorf("%w: %w", ErrProfilerBinding, err)
	}
	b.Communicator = strings.ToLower(b.Communicator)
	if b.Incarnation == "" || b.Host == "" || b.Evidence == "" || b.Operation == "" || b.PID < 1 || b.Rank < 0 || len(b.Channels) == 0 {
		return nil, fmt.Errorf("%w: incarnation, host, evidence, operation, process, rank and channels are required", ErrProfilerBinding)
	}
	for i, ch := range b.Channels {
		if slices.Contains(b.Channels[:i], ch) {
			return nil, fmt.Errorf("%w: channel %d listed twice", ErrProfilerBinding, ch)
		}
	}
	for _, n := range b.Require {
		switch n {
		case NeedPhases, NeedLoss, NeedProgressAge, NeedCrossRankTimeline:
		default:
			return nil, fmt.Errorf("%w: unknown need %q", ErrProfilerBinding, string(n))
		}
	}
	if t.PID != b.PID {
		return nil, fmt.Errorf("%w: trace process %d differs from binding %d", ErrProfilerBinding, t.PID, b.PID)
	}
	if r.Collective == nil || strings.ToLower(r.Header.ID) != b.Communicator || r.Metadata.Hostname != b.Host || r.Header.Rank != b.Rank || r.Metadata.PID != b.PID ||
		r.Collective.Name != b.Operation || r.Collective.Sequence != b.Sequence {
		return nil, fmt.Errorf("%w: record differs from binding", ErrProfilerBinding)
	}

	// Observed kernel channels are the union of the record's timestamp and
	// sequence-number kernel events.
	var observed []int
	if ts := r.Collective.EventTraceTS; ts != nil {
		for _, k := range ts.KernelEvents {
			observed = append(observed, k.ChannelID)
		}
	}
	if sn := r.Collective.EventTraceSN; sn != nil {
		for _, k := range sn.KernelEvents {
			observed = append(observed, k.ChannelID)
		}
	}
	if len(observed) > 0 {
		for _, ch := range b.Channels {
			if !slices.Contains(observed, ch) {
				return nil, fmt.Errorf("%w: channel %d is not among the record's observed kernel channels", ErrProfilerBinding, ch)
			}
		}
	}

	var sel *ProfilerCollective
	for i := range t.Collectives {
		c := &t.Collectives[i]
		if c.Communicator() == b.Communicator && c.Rank == b.Rank && c.Name == b.Operation && c.Sequence == b.Sequence {
			// ReadProfilerTrace refuses duplicate identities.
			sel = c
			break
		}
	}
	if sel == nil {
		return nil, fmt.Errorf("%w: no collective scope matches", ErrProfilerBinding)
	}

	bytes, known := sel.Bytes()
	switch {
	case known && bytes != r.Collective.Bytes:
		return nil, fmt.Errorf("%w: count converts to %d bytes, record has %d", ErrProfilerBinding, bytes, r.Collective.Bytes)
	case !known && !b.SizeUnverified:
		return nil, fmt.Errorf("%w: datatype %q has no known size", ErrProfilerUnavailable, sel.Datatype)
	}

	obs, err := ObservationFromRecord(r, clock)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrProfilerBinding, err)
	}

	out := &ProfilerBound{
		Profile:        t.Capture.Profile,
		ArtifactSHA256: t.Capture.SHA256,
		Collective:     *sel,
		Observation:    obs,
		Asserted: ProfilerAsserted{
			Binding: b,
			Build:   t.Capture.Build,
			Clock:   clock,
		},
		Bytes:      bytes,
		BytesKnown: known,
	}
	out.Asserted.Binding.Channels = slices.Clone(b.Channels)
	out.Asserted.Binding.Require = slices.Clone(b.Require)

	complete := true
	for _, want := range b.Channels {
		var found *ProfilerChannel
		for i := range sel.KernelChannels {
			if sel.KernelChannels[i].Channel != want {
				continue
			}
			if found != nil {
				return nil, fmt.Errorf("%w: channel %d appears twice", ErrProfilerBinding, want)
			}
			found = &sel.KernelChannels[i]
		}
		if found == nil {
			return nil, fmt.Errorf("%w: channel %d is not in the collective scope", ErrProfilerBinding, want)
		}
		res, err := channelResult(*found)
		if err != nil {
			return nil, fmt.Errorf("%w: %w", ErrProfilerBinding, err)
		}
		for _, p := range res.Phases {
			if !p.Reported {
				complete = false
			}
		}
		if len(res.Absent) != 0 || res.Withheld != "" {
			complete = false
		}
		out.Channels = append(out.Channels, res)
	}

	out.Unavailable = []ProfilerUnavailable{
		{"loss", "the example profiler exports no drop, eviction or population counts; a closed array proves framing only"},
		{"progress_age", "no progress counter or refresh age is exported to this trace"},
		{"cross_rank_timeline", "raw GPU timers have no supplied calibration to a reference clock; host callback times are not GPU starts"},
		{"completion", "the COLL end is a serialized printed scope, not device completion"},
		{"api_event_rank", t.Source.apiRankNote()},
	}
	for _, n := range b.Require {
		switch {
		case n == NeedPhases && !complete:
			return nil, fmt.Errorf("%w: not every bound channel has all three admitted phases", ErrProfilerUnavailable)
		case n == NeedLoss, n == NeedProgressAge, n == NeedCrossRankTimeline:
			return nil, fmt.Errorf("%w: %s", ErrProfilerUnavailable, n)
		}
	}
	return out, nil
}

func (s ProfilerSource) apiRankNote() string {
	if s.APIRanks {
		return "the example writer prints no rank on API events"
	}
	return "the example writer prints no rank on API events, and this producer leaves the descriptor rank zero"
}

func channelResult(c ProfilerChannel) (ProfilerChannelResult, error) {
	res := ProfilerChannelResult{Channel: c.Channel, StartGpuClk: c.StartGpuClk, StopGpuClk: c.StopGpuClk, Withheld: c.Withheld}
	if c.Withheld == "" {
		res.KernelNS, res.KernelReported = interval(c.StartGpuClk, c.StopGpuClk)
	}
	seen := make([]bool, len(phaseNames))
	for _, p := range c.Phases {
		if p.ID < 0 || p.ID >= len(phaseNames) {
			return res, fmt.Errorf("phase ID %d out of range", p.ID)
		}
		seen[p.ID] = true
		r := ProfilerPhaseResult{
			Name:               p.Name,
			ID:                 p.ID,
			From:               phaseBounds[p.ID][0],
			To:                 phaseBounds[p.ID][1],
			HostCallbackMicros: p.Micros,
			Raw:                p.Raw,
		}
		switch {
		case p.Withheld != "":
			r.Reason = p.Withheld
		default:
			var ok bool
			r.ReportedNS, ok = p.ReportedNS()
			r.Reported = ok
			if !ok {
				r.Reason = "empty or reversed GPU timer interval"
			}
		}
		res.Phases = append(res.Phases, r)
	}
	for i, name := range phaseNames {
		if !seen[i] {
			res.Absent = append(res.Absent, name)
		}
	}
	return res, nil
}

func interval(start, stop uint64) (int64, bool) {
	return ProfilerPhase{StartGpuClk: start, StopGpuClk: stop}.ReportedNS()
}
