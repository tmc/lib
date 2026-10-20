package ncclinspector

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"regexp"
	"strconv"
	"strings"
)

// ProfilerProfile names one exact, source-selected producer of the NCCL
// example profiler's v7 Chrome trace JSON array. The trace carries no
// in-band version or schema tag, so the caller selects the profile.
type ProfilerProfile string

const (
	// ProfilerNCCL2323 is plugins/profiler/example at NCCL v2.32.3-1.
	ProfilerNCCL2323 ProfilerProfile = "nccl-v2.32.3-1/example-profiler-v7"

	// ProfilerNCCL2312 is plugins/profiler/example at NCCL v2.31.2-1. Its
	// trace writer is byte-identical to 2.32.3, but its channel bound,
	// phase gating and API event ranks differ.
	ProfilerNCCL2312 ProfilerProfile = "nccl-v2.31.2-1/example-profiler-v7"
)

// ProfilerSource records the pinned producer facts a profile selects.
type ProfilerSource struct {
	Commit           string // NVIDIA/nccl commit
	PrintEventSHA256 string // plugins/profiler/example/print_event.cc
	PluginSHA256     string // plugins/profiler/example/plugin.cc

	// MaxChannels bounds KernelCh channel IDs (example event.h MAX_CHANNELS).
	MaxChannels int

	// RegularPhasesGated reports that regular collective and point-to-point
	// work never fires kernel-phase callbacks: src/plugin/profiler.cc sets
	// workPhases to nullptr for them. Otherwise a non-symmetric phase record
	// is retained but not admitted as a source-supported fact.
	RegularPhasesGated bool

	// APIRanks reports that group, collective, point-to-point API and kernel
	// launch event descriptors carry the communicator rank. Earlier producers
	// leave it zero. The example writer prints no API event rank in either
	// profile; collective and point-to-point ranks are set in both.
	APIRanks bool
}

var profilerSources = map[ProfilerProfile]ProfilerSource{
	ProfilerNCCL2323: {
		Commit:             "12df1a11afad322be5a204a2db890161cbf8131d",
		PrintEventSHA256:   "0c7e4424d630a2448622043bf8205e739dd4a8425d6af1caec5b9726b6640a4c",
		PluginSHA256:       "30b1315ee3447ad508ed6b71d17d671d252f6e2028dcbef2a2f55685c94b4335",
		MaxChannels:        64,
		RegularPhasesGated: true,
		APIRanks:           true,
	},
	ProfilerNCCL2312: {
		Commit:           "7b83616df3ae082a1f32bb74c27458bfe8153a13",
		PrintEventSHA256: "0c7e4424d630a2448622043bf8205e739dd4a8425d6af1caec5b9726b6640a4c",
		PluginSHA256:     "7fe7f2cf1fdc030fbc29edaa262a3a5cd1c390a6c99e8600d94e77c3c10d467e",
		MaxChannels:      32,
	},
}

// Source returns the pinned producer facts for p.
// It refuses profiles that have not been source-selected.
func (p ProfilerProfile) Source() (ProfilerSource, error) {
	s, ok := profilerSources[p]
	if !ok {
		return ProfilerSource{}, fmt.Errorf("profiler: unsupported profile %q", string(p))
	}
	return s, nil
}

// ProfilerCapture is caller-supplied context for one trace artifact.
// No version, host, communicator incarnation or build is inferred from the
// file name, process ID or event presence.
type ProfilerCapture struct {
	Profile ProfilerProfile
	// Build names the loaded plugin ABI and build. It is asserted, not verified.
	Build string
	// SHA256 is the expected lowercase hex digest of the raw artifact bytes.
	SHA256 string
}

// ProfilerTrace is one decoded example-profiler trace in input order.
// A closed array proves framing only: the producer keeps finite event pools
// and exports no drop, eviction or population counts.
type ProfilerTrace struct {
	Capture       ProfilerCapture
	Source        ProfilerSource
	PID           int // process ID printed by every event; zero for an empty trace
	Events        int // events, excluding the terminal empty object
	Collectives   []ProfilerCollective
	PointToPoints []ProfilerPointToPoint
	// Other retains exported events that are not interpreted.
	Other []ProfilerEvent

	// read marks a trace produced by ReadProfilerTrace; BindProfiler
	// requires it. It survives copying the struct, and edits to the
	// exported fields after reading are not detected.
	read bool
}

// ProfilerEvent is an exported event retained without interpretation.
type ProfilerEvent struct {
	Index                 int // input event index
	Category, Name, Phase string
	Raw                   []byte
}

// ProfilerCollective is one serialized COLL scope. Sequence is the
// producer's per-operation-kind counter, not a communicator-wide order.
// Count is in elements, not bytes. BeginMicros and EndMicros are the exact
// printed host callback timestamps, in microseconds since plugin start.
type ProfilerCollective struct {
	Index                         int // input event index of the COLL begin
	Name                          string
	Sequence                      uint64
	CommHash                      uint64
	Rank                          int
	Count                         uint64
	Datatype, Algorithm, Protocol string
	Channels                      int // reported nChannels
	KernelVariant                 string
	Symmetric                     bool // IsSymColl
	BeginMicros, EndMicros        string
	KernelChannels                []ProfilerChannel
	Begin, End                    []byte // exact raw lines
}

// Communicator returns the full 64-bit communicator hash as 0x and 16 hex digits.
func (c ProfilerCollective) Communicator() string { return fmt.Sprintf("0x%016x", c.CommHash) }

// Bytes converts Count to bytes using the pinned producer's datatype names
// and sizes. It reports false when the printed datatype has no known size;
// the producer prints ncclUint8 as "Unknown".
func (c ProfilerCollective) Bytes() (uint64, bool) {
	size, ok := datatypeSizes[c.Datatype]
	if !ok || c.Count > math.MaxUint64/size {
		return 0, false
	}
	return c.Count * size, true
}

// From ncclDatatypeToString (src/collectives.cc) and ncclTypeSize
// (src/include/collectives.h) at the pinned commit.
var datatypeSizes = map[string]uint64{
	"ncclInt8": 1, "ncclFloat8e4m3": 1, "ncclFloat8e5m2": 1,
	"ncclFloat16": 2, "ncclBfloat16": 2,
	"ncclInt32": 4, "ncclUint32": 4, "ncclFloat32": 4,
	"ncclInt64": 8, "ncclUint64": 8, "ncclFloat64": 8,
}

// ProfilerPointToPoint is one serialized P2P scope. The example exports no
// point-to-point sequence; it is not interchangeable with Inspector p2p_sn.
type ProfilerPointToPoint struct {
	Index                  int
	Name                   string
	CommHash               uint64
	Rank, Peer             int
	Count                  uint64
	Datatype               string
	Channels               int
	BeginMicros, EndMicros string
	KernelChannels         []ProfilerChannel
	Begin, End             []byte
}

// ProfilerChannel is one KernelCh scope inside its enclosing COLL or P2P
// scope. StartGpuClk and StopGpuClk are raw GPU timer values in
// nanoseconds; zero means the producer recorded none.
type ProfilerChannel struct {
	Channel                 int
	StartGpuClk, StopGpuClk uint64
	BeginMicros, EndMicros  string
	Phases                  []ProfilerPhase
	Begin, End              []byte

	// Withheld is set when the channel cannot be attributed to its enclosing
	// scope or its GPU boundaries disagree. The producer reuses finite pool
	// slots and clears only the proxy counts, so a channel the new operation
	// never ran can print the previous occupant's data. Its host begin then
	// precedes the enclosing scope's begin. The producer also derives every
	// phase and the KernelCh endpoints from one per-work timestamp array, so
	// phases that do not share their boundaries exactly came from more than
	// one array. The channel and its phases are retained but not admitted.
	Withheld string
}

// ProfilerPhase is one kernel phase complete event: initial_sync (0),
// compute (1) or final_sync (2), between adjacent producer boundaries.
// Micros is the host callback time, not the GPU phase start. DurationMicros
// is the printed rendering of the GPU timer difference. A nonempty Withheld
// gives the reason the profile does not admit the phase as a source fact.
type ProfilerPhase struct {
	Name                    string
	ID, Channel             int
	StartGpuClk, StopGpuClk uint64
	Micros, DurationMicros  string
	Withheld                string
	Raw                     []byte
}

// ReportedNS returns the within-device phase duration from the raw GPU timer
// endpoints. It reports false when an endpoint is absent or the interval is
// empty or reversed. The duration is on one GPU's timer; it is not a
// position on any host or cross-rank timeline.
func (p ProfilerPhase) ReportedNS() (int64, bool) {
	if p.StartGpuClk == 0 || p.StopGpuClk <= p.StartGpuClk || p.StopGpuClk-p.StartGpuClk > math.MaxInt64 {
		return 0, false
	}
	return int64(p.StopGpuClk - p.StartGpuClk), true
}

// From profilerPhaseNames in src/plugin/profiler.cc.
var phaseNames = []string{"initial_sync", "compute", "final_sync"}

// ReadProfilerTrace reads one finite example-profiler trace: "[\n", one
// event object per line followed by ",\n", and the terminal "{}]\n" that
// the producer writes at finalization. It refuses unsupported profiles,
// digest mismatches, truncation, duplicate keys, null or unexpected fields,
// unknown categories and inconsistent scopes. Phases are associated only
// with their enclosing serialized COLL or P2P and KernelCh scopes; event
// IDs, the constant tid and timestamps are never used for association.
// A channel whose phases disagree on a shared GPU boundary is withheld, not
// refused, so that other channels' phases remain available.
// The caller must arrange cancellation for a reader that can block.
func ReadProfilerTrace(r io.Reader, c ProfilerCapture, limits Limits) (*ProfilerTrace, error) {
	l, err := limits.checked()
	if err != nil {
		return nil, err
	}
	src, err := c.Profile.Source()
	if err != nil {
		return nil, err
	}
	if c.Build == "" || len(c.Build) > 4096 {
		return nil, errors.New("profiler: plugin build identity required")
	}
	if !sha256Hex.MatchString(c.SHA256) {
		return nil, errors.New("profiler: artifact SHA-256 required")
	}
	if r == nil {
		return nil, errors.New("profiler: nil reader")
	}
	b, err := io.ReadAll(io.LimitReader(r, int64(l.Bytes)+1))
	if err != nil {
		return nil, fmt.Errorf("profiler read: %w", err)
	}
	if len(b) > l.Bytes {
		return nil, errors.New("profiler: byte limit exceeded")
	}
	sum := sha256.Sum256(b)
	if hex.EncodeToString(sum[:]) != c.SHA256 {
		return nil, errors.New("profiler: artifact digest mismatch")
	}
	p := &profilerParser{t: &ProfilerTrace{Capture: c, Source: src}, collectives: make(map[collectiveKey]bool)}
	if err := p.parse(b, l); err != nil {
		return nil, err
	}
	p.t.read = true
	return p.t, nil
}

var (
	sha256Hex    = regexp.MustCompile(`^[0-9a-f]{64}$`)
	printedFloat = regexp.MustCompile(`^-?[0-9]+\.[0-9]{6}$`) // C %f
	printedUint  = regexp.MustCompile(`^(0|[1-9][0-9]*)$`)    // C %lu
	printedInt   = regexp.MustCompile(`^-?(0|[1-9][0-9]*)$`)  // C %d
)

type collectiveKey struct {
	hash     uint64
	rank     int
	name     string
	sequence uint64
}

type profilerParser struct {
	t           *ProfilerTrace
	collectives map[collectiveKey]bool

	// Open scope state. At most one COLL or P2P scope and one KernelCh
	// scope within it are open; the producer never nests them further.
	coll        *ProfilerCollective
	p2p         *ProfilerPointToPoint
	scopeID     string
	channels    *[]ProfilerChannel
	lastChannel int
	channel     *ProfilerChannel
	channelID   string
	lastPhase   int
	channelBody bool // a child event followed the channel's phases
}

func (p *profilerParser) parse(b []byte, l Limits) error {
	if !bytes.HasPrefix(b, []byte("[\n")) {
		return errors.New("profiler: missing trace array opening")
	}
	b = b[2:]
	for {
		i := bytes.IndexByte(b, '\n')
		if i < 0 {
			return errors.New("profiler: truncated trace: missing terminal object")
		}
		line := b[:i+1]
		b = b[i+1:]
		if len(line)-1 > l.Line {
			return errors.New("profiler: line limit exceeded")
		}
		if string(line) == "{}]\n" {
			if len(b) != 0 {
				return errors.New("profiler: data after terminal object")
			}
			break
		}
		if p.t.Events >= l.Records {
			return errors.New("profiler: event limit exceeded")
		}
		if !bytes.HasSuffix(line, []byte("},\n")) {
			return fmt.Errorf("profiler event %d: missing event separator", p.t.Events)
		}
		if err := p.event(line[:len(line)-2], append([]byte(nil), line...)); err != nil {
			return fmt.Errorf("profiler event %d: %w", p.t.Events, err)
		}
		p.t.Events++
	}
	if p.coll != nil || p.p2p != nil {
		return errors.New("profiler: unterminated collective or point-to-point scope")
	}
	return nil
}

// Categories the pinned print_event.cc writes that are not interpreted.
// Only proxy and network events are printed inside a COLL or P2P scope.
var otherCategories = map[string]bool{
	"GROUP_API": false, "COLL_API": false, "P2P_API": false, "KERNEL_LAUNCH": false, "GROUP": false,
	"CE_COLL": false, "CE_SYNC": false, "CE_BATCH": false,
	"PROXY": true, "NET": true, "NET_IB": true, "NET_SOCK": true,
}

type event struct {
	m                 map[string]json.RawMessage
	name, cat, ph, id string
	ts                string
}

func (p *profilerParser) event(obj, raw []byte) error {
	if err := uniqueJSON(obj); err != nil {
		return err
	}
	var e event
	if err := json.Unmarshal(obj, &e.m); err != nil || e.m == nil {
		return errors.New("malformed event object")
	}
	var err error
	if e.name, err = text(e.m, "name"); err != nil {
		return err
	}
	if e.cat, err = text(e.m, "cat"); err != nil {
		return err
	}
	if e.ph, err = text(e.m, "ph"); err != nil {
		return err
	}
	if e.id, err = token(e.m, "id", printedInt); err != nil {
		return err
	}
	if e.ts, err = token(e.m, "ts", printedFloat); err != nil {
		return err
	}
	pid, err := integer(e.m, "pid")
	if err != nil {
		return err
	}
	tid, err := integer(e.m, "tid")
	if err != nil {
		return err
	}
	if pid < 1 || p.t.PID != 0 && pid != p.t.PID {
		return errors.New("inconsistent process ID")
	}
	p.t.PID = pid
	if tid != 1 {
		return errors.New("unexpected thread ID")
	}
	index := p.t.Events
	switch {
	case e.cat == "COLL" && e.ph == "b":
		return p.collBegin(e, index, raw)
	case e.cat == "COLL" && e.ph == "e":
		return p.scopeEnd(e, raw, p.coll != nil)
	case e.cat == "P2P" && e.ph == "b":
		return p.p2pBegin(e, index, raw)
	case e.cat == "P2P" && e.ph == "e":
		return p.scopeEnd(e, raw, p.p2p != nil)
	case e.cat == "GPU" && e.ph == "b" && e.name == "KernelCh":
		return p.channelBegin(e, raw)
	case e.cat == "GPU" && e.ph == "e" && e.name == "KernelCh":
		return p.channelEnd(e, raw)
	case e.cat == "GPU" && e.ph == "X":
		return p.phase(e, raw)
	}
	inScope, ok := otherCategories[e.cat]
	if !ok {
		return fmt.Errorf("unsupported event category %q", e.cat)
	}
	if e.ph != "b" && e.ph != "e" {
		return fmt.Errorf("unsupported event phase %q", e.ph)
	}
	want := []string{"name", "cat", "ph", "id", "pid", "tid", "ts"}
	if _, ok := e.m["args"]; ok {
		if _, err := args(e.m); err != nil {
			return err
		}
		want = append(want, "args")
	}
	if err := exactKeys(e.m, want...); err != nil {
		return err
	}
	if (p.coll != nil || p.p2p != nil) && !inScope {
		return fmt.Errorf("%s event inside an open collective or point-to-point scope", e.cat)
	}
	if p.channel != nil {
		p.channelBody = true
	}
	p.t.Other = append(p.t.Other, ProfilerEvent{Index: index, Category: e.cat, Name: e.name, Phase: e.ph, Raw: raw})
	return nil
}

func (p *profilerParser) collBegin(e event, index int, raw []byte) error {
	a, err := beginArgs(e, "SeqNum", "CommHash", "Rank", "Count", "Datatype", "Algorithm", "Protocol", "nChannels", "KernelVariant", "IsSymColl")
	if err != nil {
		return err
	}
	if p.coll != nil || p.p2p != nil {
		return errors.New("nested collective scope")
	}
	c := ProfilerCollective{Index: index, Name: e.name, BeginMicros: e.ts, Begin: raw}
	var sym int
	for _, f := range []error{
		unsigned(a, "SeqNum", &c.Sequence),
		unsigned(a, "CommHash", &c.CommHash),
		signed(a, "Rank", &c.Rank),
		unsigned(a, "Count", &c.Count),
		textField(a, "Datatype", &c.Datatype),
		textField(a, "Algorithm", &c.Algorithm),
		textField(a, "Protocol", &c.Protocol),
		signed(a, "nChannels", &c.Channels),
		textField(a, "KernelVariant", &c.KernelVariant),
		signed(a, "IsSymColl", &sym),
	} {
		if f != nil {
			return f
		}
	}
	if c.Name == "" || c.Rank < 0 || c.Channels < 0 || sym != 0 && sym != 1 {
		return errors.New("invalid collective identity")
	}
	c.Symmetric = sym == 1
	k := collectiveKey{c.CommHash, c.Rank, c.Name, c.Sequence}
	if p.collectives[k] {
		return errors.New("duplicate collective identity")
	}
	p.collectives[k] = true
	p.coll = &c
	p.open(e.id, &c.KernelChannels)
	return nil
}

func (p *profilerParser) p2pBegin(e event, index int, raw []byte) error {
	a, err := beginArgs(e, "CommHash", "Rank", "Peer", "Count", "Datatype", "nChannels")
	if err != nil {
		return err
	}
	if p.coll != nil || p.p2p != nil {
		return errors.New("nested point-to-point scope")
	}
	s := ProfilerPointToPoint{Index: index, Name: e.name, BeginMicros: e.ts, Begin: raw}
	for _, f := range []error{
		unsigned(a, "CommHash", &s.CommHash),
		signed(a, "Rank", &s.Rank),
		signed(a, "Peer", &s.Peer),
		unsigned(a, "Count", &s.Count),
		textField(a, "Datatype", &s.Datatype),
		signed(a, "nChannels", &s.Channels),
	} {
		if f != nil {
			return f
		}
	}
	if s.Name == "" || s.Rank < 0 || s.Peer < 0 || s.Channels < 0 {
		return errors.New("invalid point-to-point identity")
	}
	p.p2p = &s
	p.open(e.id, &s.KernelChannels)
	return nil
}

func (p *profilerParser) open(id string, channels *[]ProfilerChannel) {
	p.scopeID = id
	p.channels = channels
	p.lastChannel = -1
}

func (p *profilerParser) scopeEnd(e event, raw []byte, open bool) error {
	if err := exactKeys(e.m, "name", "cat", "ph", "id", "pid", "tid", "ts"); err != nil {
		return err
	}
	if !open {
		return fmt.Errorf("%s end without matching begin", e.cat)
	}
	if p.channel != nil {
		return errors.New("scope ends inside an open KernelCh scope")
	}
	if e.id != p.scopeID {
		return errors.New("scope end ID differs from begin")
	}
	if c := p.coll; c != nil {
		if e.name != c.Name {
			return errors.New("collective end name differs from begin")
		}
		c.EndMicros, c.End = e.ts, raw
		p.t.Collectives = append(p.t.Collectives, *c)
		p.coll = nil
	} else {
		s := p.p2p
		if e.name != s.Name {
			return errors.New("point-to-point end name differs from begin")
		}
		s.EndMicros, s.End = e.ts, raw
		p.t.PointToPoints = append(p.t.PointToPoints, *s)
		p.p2p = nil
	}
	p.channels = nil
	return nil
}

func (p *profilerParser) channelBegin(e event, raw []byte) error {
	a, err := beginArgs(e, "Channel", "StartGpuClk", "StopGpuClk")
	if err != nil {
		return err
	}
	if p.channels == nil {
		return errors.New("KernelCh outside a collective or point-to-point scope")
	}
	if p.channel != nil {
		return errors.New("nested KernelCh scope")
	}
	ch := ProfilerChannel{BeginMicros: e.ts, Begin: raw}
	for _, f := range []error{
		signed(a, "Channel", &ch.Channel),
		unsigned(a, "StartGpuClk", &ch.StartGpuClk),
		unsigned(a, "StopGpuClk", &ch.StopGpuClk),
	} {
		if f != nil {
			return f
		}
	}
	if ch.Channel < 0 || ch.Channel >= p.t.Source.MaxChannels {
		return fmt.Errorf("channel %d outside the profile's channel bound", ch.Channel)
	}
	scopeBegin := ""
	if p.coll != nil {
		scopeBegin = p.coll.BeginMicros
	} else {
		scopeBegin = p.p2p.BeginMicros
	}
	if microsBefore(ch.BeginMicros, scopeBegin) {
		ch.Withheld = "channel begins before its enclosing scope: stale pool slot of an earlier operation"
	}
	// The producer prints channels in increasing index order, once each.
	if ch.Channel <= p.lastChannel {
		return errors.New("repeated or out-of-order KernelCh channel")
	}
	p.lastChannel = ch.Channel
	p.channel = &ch
	p.channelID = e.id
	p.lastPhase = -1
	p.channelBody = false
	return nil
}

func (p *profilerParser) channelEnd(e event, raw []byte) error {
	if err := exactKeys(e.m, "name", "cat", "ph", "id", "pid", "tid", "ts"); err != nil {
		return err
	}
	if p.channel == nil {
		return errors.New("KernelCh end without matching begin")
	}
	if e.id != p.channelID {
		return errors.New("KernelCh end ID differs from begin")
	}
	ch := p.channel
	ch.EndMicros, ch.End = e.ts, raw
	if ch.Withheld == "" {
		if why := boundaryConflict(ch); why != "" {
			ch.Withheld = why
			for i := range ch.Phases {
				if ch.Phases[i].Withheld == "" {
					ch.Phases[i].Withheld = why
				}
			}
		}
	}
	*p.channels = append(*p.channels, *ch)
	p.channel = nil
	return nil
}

func (p *profilerParser) phase(e event, raw []byte) error {
	a, err := phaseArgs(e)
	if err != nil {
		return err
	}
	ph := ProfilerPhase{Name: e.name, Micros: e.ts, Raw: raw}
	if ph.DurationMicros, err = token(e.m, "dur", printedFloat); err != nil {
		return err
	}
	for _, f := range []error{
		signed(a, "Channel", &ph.Channel),
		signed(a, "PhaseId", &ph.ID),
		unsigned(a, "StartGpuClk", &ph.StartGpuClk),
		unsigned(a, "StopGpuClk", &ph.StopGpuClk),
	} {
		if f != nil {
			return f
		}
	}
	ch := p.channel
	if ch == nil {
		return errors.New("kernel phase outside a KernelCh scope")
	}
	if p.channelBody {
		return errors.New("kernel phase after channel child events")
	}
	if ph.Channel != ch.Channel || e.id != p.channelID {
		return errors.New("kernel phase channel differs from its enclosing KernelCh")
	}
	if ph.ID < 0 || ph.ID >= len(phaseNames) || ph.Name != phaseNames[ph.ID] {
		return errors.New("kernel phase name and ID disagree")
	}
	// The producer prints each phase slot once, in index order.
	if ph.ID <= p.lastPhase {
		return errors.New("repeated or out-of-order kernel phase")
	}
	p.lastPhase = ph.ID
	if ph.DurationMicros != printedDuration(ph.StartGpuClk, ph.StopGpuClk) {
		return errors.New("kernel phase duration differs from its GPU timer endpoints")
	}
	if ch.Withheld != "" {
		ph.Withheld = ch.Withheld
	} else if p.p2p != nil || !p.coll.Symmetric {
		if p.t.Source.RegularPhasesGated {
			return errors.New("kernel phase for non-symmetric work the profile never instruments")
		}
		ph.Withheld = "regular and point-to-point phase callbacks are not producer-gated in this profile"
	}
	ch.Phases = append(ch.Phases, ph)
	return nil
}

// boundaryConflict reports why the channel's printed GPU boundaries cannot
// come from one producer timestamp array, or "" if they can.
// profilerFireKernelPhaseEvents emits phase i as ts[i] -> ts[i+1], folding a
// missing intermediate boundary onto its outer one and skipping a zero or
// equal pair. So consecutive printed phases share a boundary (with compute
// skipped, ts[1] == ts[2]), the first printed phase starts at ts[BEGIN] and
// the last stops at ts[END]. The device writes ts[BEGIN] and ts[END] with the
// KernelCh start and stop values, which are compared once recorded. The
// writer does not enforce order; the device takes its stamps in program
// order, so a reversed pair means mixed arrays.
func boundaryConflict(ch *ProfilerChannel) string {
	for i, ph := range ch.Phases {
		switch {
		case i > 0 && ch.Phases[i-1].StopGpuClk != ph.StartGpuClk:
			return fmt.Sprintf("%s and %s disagree on their shared GPU boundary", ch.Phases[i-1].Name, ph.Name)
		case ph.StopGpuClk != 0 && ph.StopGpuClk <= ph.StartGpuClk:
			return fmt.Sprintf("%s GPU boundaries are not in increasing order", ph.Name)
		}
	}
	n := len(ch.Phases)
	if n == 0 {
		return ""
	}
	if first := ch.Phases[0]; ch.StartGpuClk != 0 && first.StartGpuClk != ch.StartGpuClk {
		return fmt.Sprintf("%s start differs from the KernelCh GPU start", first.Name)
	}
	if last := ch.Phases[n-1]; ch.StopGpuClk != 0 && last.StopGpuClk != ch.StopGpuClk {
		return fmt.Sprintf("%s stop differs from the KernelCh GPU stop", last.Name)
	}
	return ""
}

// microsBefore reports whether the printed %f microsecond timestamp a is
// earlier than b, comparing the exact decimal digits. A value that does not
// fit int64 once the decimal point is removed cannot be ordered, so it is
// reported as earlier: the caller withholds rather than attributes.
func microsBefore(a, b string) bool {
	x, err1 := strconv.ParseInt(strings.Replace(a, ".", "", 1), 10, 64)
	y, err2 := strconv.ParseInt(strings.Replace(b, ".", "", 1), 10, 64)
	return err1 != nil || err2 != nil || x < y
}

// printedDuration reproduces print_event.cc's dur rendering.
func printedDuration(start, stop uint64) string {
	if stop <= start {
		return "0.000000"
	}
	return strconv.FormatFloat(float64(stop-start)/1000.0, 'f', 6, 64)
}

func beginArgs(e event, keys ...string) (map[string]json.RawMessage, error) {
	if err := exactKeys(e.m, "name", "cat", "ph", "id", "pid", "tid", "ts", "args"); err != nil {
		return nil, err
	}
	a, err := args(e.m)
	if err != nil {
		return nil, err
	}
	return a, exactKeys(a, keys...)
}

func phaseArgs(e event) (map[string]json.RawMessage, error) {
	if err := exactKeys(e.m, "name", "cat", "ph", "id", "pid", "tid", "ts", "dur", "args"); err != nil {
		return nil, err
	}
	a, err := args(e.m)
	if err != nil {
		return nil, err
	}
	return a, exactKeys(a, "Channel", "PhaseId", "StartGpuClk", "StopGpuClk")
}

func args(m map[string]json.RawMessage) (map[string]json.RawMessage, error) {
	v := bytes.TrimSpace(m["args"])
	if len(v) == 0 || v[0] != '{' {
		return nil, errors.New("args must be an object")
	}
	var a map[string]json.RawMessage
	if err := json.Unmarshal(v, &a); err != nil {
		return nil, errors.New("malformed args")
	}
	return a, nil
}

func exactKeys(m map[string]json.RawMessage, keys ...string) error {
	for _, k := range keys {
		if _, ok := m[k]; !ok {
			return fmt.Errorf("missing %s", k)
		}
	}
	if len(m) != len(keys) {
		return errors.New("unexpected event field")
	}
	return nil
}

func text(m map[string]json.RawMessage, k string) (string, error) {
	var s string
	if err := textField(m, k, &s); err != nil {
		return "", err
	}
	return s, nil
}

func textField(m map[string]json.RawMessage, k string, s *string) error {
	v := bytes.TrimSpace(m[k])
	if len(v) == 0 || v[0] != '"' {
		return fmt.Errorf("%s must be a string", k)
	}
	if err := json.Unmarshal(v, s); err != nil {
		return fmt.Errorf("malformed %s", k)
	}
	return nil
}

// token returns the exact printed spelling of a numeric field.
func token(m map[string]json.RawMessage, k string, form *regexp.Regexp) (string, error) {
	v := string(bytes.TrimSpace(m[k]))
	if !form.MatchString(v) {
		return "", fmt.Errorf("%s is not a producer-printed number", k)
	}
	return v, nil
}

func integer(m map[string]json.RawMessage, k string) (int, error) {
	var n int
	return n, signed(m, k, &n)
}

func signed(m map[string]json.RawMessage, k string, n *int) error {
	v, err := token(m, k, printedInt)
	if err != nil {
		return err
	}
	i, err := strconv.ParseInt(v, 10, 32)
	if err != nil {
		return fmt.Errorf("%s out of range", k)
	}
	*n = int(i)
	return nil
}

func unsigned(m map[string]json.RawMessage, k string, n *uint64) error {
	v, err := token(m, k, printedUint)
	if err != nil {
		return err
	}
	if *n, err = strconv.ParseUint(v, 10, 64); err != nil {
		return fmt.Errorf("%s out of range", k)
	}
	return nil
}
