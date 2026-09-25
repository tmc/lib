package advise

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"regexp"
	"sort"
	"strings"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

// hostAPIClass is the kind of host API call a host-cost finding names.
type hostAPIClass int

const (
	hostAPIOther hostAPIClass = iota
	hostAPISyncCopy
	hostAPIAlloc
	hostAPIInstantiate
	hostAPIGraphLaunch
)

// negligibleHostSharePct is the share of the host API window below which a
// host-cost finding is a rounding error. Like the device demotion, a finding
// is demoted only when it is also below negligibleDeviceNanos of host time.
const negligibleHostSharePct = 1.0

// callSiteUnread is the call site of every host-cost finding. The journal
// records each API call's thread and interval but no host stack, so where in
// the program the call was made is not in the capture.
const callSiteUnread = "UNREAD: the journal records API calls without a host stack"

// hostAPICall is one driver or runtime API call from the capture journal.
type hostAPICall struct {
	domain      string
	name        string
	class       hostAPIClass
	start, end  int64
	thread      uint32
	correlation uint32

	// driverCorrelations are the correlations of the driver calls a runtime
	// call made on its behalf, matched by containment. A copy's activity
	// record names the driver call's correlation, and one runtime copy may
	// issue several.
	driverCorrelations []uint32
}

func (c hostAPICall) nanos() int64 { return c.end - c.start }

// hostCopy is the device side of one memory copy, from its activity record.
type hostCopy struct {
	bytes  uint64
	kind   uint8
	stream uint32
	nanos  int64
}

// hostAPIJournal is what the host-cost audit reads from a journal.
type hostAPIJournal struct {
	calls []hostAPICall

	// copies and runtimeCopies index copy activities by the driver and the
	// runtime correlation they name.
	copies        map[uint32]hostCopy
	runtimeCopies map[uint32]hostCopy

	// windowStart and windowEnd bound every API call the journal holds, of
	// any class: the host API window a finding's share is taken of.
	windowStart, windowEnd int64
}

// hostAPIPayload is the subset of a journal payload the host-cost audit reads.
type hostAPIPayload struct {
	Kind               string `json:"kind"`
	Subtype            string `json:"subtype"`
	Domain             string `json:"domain"`
	CallbackID         uint32 `json:"callback_id"`
	ThreadID           uint32 `json:"thread_id"`
	Valid              bool   `json:"valid"`
	Start              uint64 `json:"start"`
	End                uint64 `json:"end"`
	Bytes              uint64 `json:"bytes"`
	CopyKind           uint8  `json:"copy_kind"`
	Stream             uint32 `json:"stream"`
	Correlation        uint32 `json:"correlation"`
	RuntimeCorrelation uint32 `json:"runtime_correlation"`
}

// readHostAPIJournal decodes the API records and copy activities of every
// journal in cap.
func readHostAPIJournal(cap *cudacapture.Capture) (hostAPIJournal, error) {
	journals, err := timeline.Journals(cap)
	if err != nil {
		return hostAPIJournal{}, err
	}
	j := hostAPIJournal{
		copies:        make(map[uint32]hostCopy),
		runtimeCopies: make(map[uint32]hostCopy),
		windowStart:   math.MaxInt64,
	}
	names := make([]string, 0, len(journals))
	for name := range journals {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		if err := j.decode(name, journals[name]); err != nil {
			return hostAPIJournal{}, err
		}
	}
	return j, nil
}

func (j *hostAPIJournal) decode(name string, data []byte) error {
	reader, err := timeline.OpenJournal(data, cudacapture.MaxCaptureInputBytes)
	if err != nil {
		return fmt.Errorf("read journal %s: %w", name, err)
	}
	defer reader.Close()
	d := wire.NewDecoder(reader, wire.Limits{MaxRecords: uint32(timeline.MaxRecords)})
	for {
		r, err := d.Next()
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return fmt.Errorf("decode journal %s: %w", name, err)
		}
		if r.Kind != wire.KindAPI && r.Kind != wire.KindActivity {
			continue
		}
		var p hostAPIPayload
		if json.Unmarshal(r.Payload, &p) != nil {
			return fmt.Errorf("journal %s record %d is not a typed capture record", name, r.Sequence)
		}
		if p.Start == 0 || p.End < p.Start || p.End > math.MaxInt64 {
			continue
		}
		start, end := int64(p.Start), int64(p.End)
		switch {
		case r.Kind == wire.KindAPI && p.Kind == "api":
			j.windowStart = min(j.windowStart, start)
			j.windowEnd = max(j.windowEnd, end)
			apiName, class := hostAPIName(p.Domain, p.CallbackID)
			if class == hostAPIOther {
				continue
			}
			j.calls = append(j.calls, hostAPICall{
				domain: p.Domain, name: apiName, class: class,
				start: start, end: end, thread: p.ThreadID, correlation: p.Correlation,
			})
		case r.Kind == wire.KindActivity && p.Kind == "activity" && p.Subtype == "memcpy" && p.Valid:
			c := hostCopy{bytes: p.Bytes, kind: p.CopyKind, stream: p.Stream, nanos: end - start}
			if p.Correlation != 0 {
				j.copies[p.Correlation] = c
			}
			if p.RuntimeCorrelation != 0 {
				j.runtimeCopies[p.RuntimeCorrelation] = c
			}
		}
	}
}

var (
	runtimeVersionRE = regexp.MustCompile(`V[0-9]+$`)
	driverVersionRE  = regexp.MustCompile(`V([0-9]+)(Ptds|Ptsz)?$`)
)

// hostAPIName names a driver or runtime callback and classifies it.
//
// The class is read from the name: a synchronous copy is a Memcpy that is not
// Async, an allocation is an alloc that is neither Async nor from a pool, and
// graph instantiates and launches are named as such. Other graph calls, which
// only describe copies and allocations, are neither.
func hostAPIName(domain string, cbid uint32) (string, hostAPIClass) {
	var name string
	switch domain {
	case "driver":
		// The stringer spells cuMemcpyHtoD_v2 as CuMemcpyHtoDV2.
		name = strings.TrimPrefix(cupti.DriverTraceCBID(cbid).String(), "DriverCBID")
		name = driverVersionRE.ReplaceAllStringFunc(name, func(m string) string {
			sub := driverVersionRE.FindStringSubmatch(m)
			v := "_v" + sub[1]
			if sub[2] != "" {
				v += "_" + strings.ToLower(sub[2])
			}
			return v
		})
	case "runtime":
		name = strings.TrimPrefix(cupti.RuntimeTraceCBID(cbid).String(), "RuntimeCBID")
		// Runtime callbacks carry the release that introduced them
		// (cudaMemcpy_v3020); the call a reader wrote has no suffix.
		name = runtimeVersionRE.ReplaceAllString(name, "")
	default:
		return "", hostAPIOther
	}
	if strings.Contains(name, "(") || name == "" {
		return name, hostAPIOther
	}
	name = strings.ToLower(name[:1]) + name[1:]
	async := strings.Contains(name, "Async")
	switch {
	case strings.Contains(name, "GraphInstantiate"):
		return name, hostAPIInstantiate
	case strings.Contains(name, "GraphLaunch"):
		return name, hostAPIGraphLaunch
	case strings.Contains(name, "Graph"):
		// cuGraphAddMemcpyNode and its kin build or edit a graph node; they
		// neither copy nor allocate when called.
		return name, hostAPIOther
	case strings.Contains(name, "Memcpy") && !async:
		return name, hostAPISyncCopy
	case (strings.Contains(name, "MemAlloc") || strings.Contains(name, "Malloc") || strings.Contains(name, "HostAlloc")) &&
		!async && !strings.Contains(name, "Pool"):
		return name, hostAPIAlloc
	}
	return name, hostAPIOther
}

// dedupRuntimeDriver drops each driver call that a runtime call of the same
// class on the same thread encloses: a cudaMemcpy and the cuMemcpyHtoD it
// makes are one copy, and counting both would double its host time. The
// runtime call is kept, since it is the call the program made, and it takes
// the driver call's correlation for joining its copy activity.
func dedupRuntimeDriver(calls []hostAPICall) []hostAPICall {
	type key struct {
		thread uint32
		class  hostAPIClass
	}
	runtime := make(map[key][]int)
	for i, c := range calls {
		if c.domain == "runtime" {
			k := key{c.thread, c.class}
			runtime[k] = append(runtime[k], i)
		}
	}
	for _, idx := range runtime {
		sort.Slice(idx, func(a, b int) bool { return calls[idx[a]].start < calls[idx[b]].start })
	}
	out := make([]hostAPICall, 0, len(calls))
	enclosed := make(map[int]bool)
	for i, c := range calls {
		if c.domain != "driver" {
			continue
		}
		idx := runtime[key{c.thread, c.class}]
		// Calls on one thread do not overlap, so the only runtime call that
		// can enclose c is the last one starting at or before it.
		n := sort.Search(len(idx), func(a int) bool { return calls[idx[a]].start > c.start })
		if n == 0 {
			continue
		}
		r := idx[n-1]
		if calls[r].end >= c.end {
			enclosed[i] = true
			calls[r].driverCorrelations = append(calls[r].driverCorrelations, c.correlation)
		}
	}
	for i, c := range calls {
		if !enclosed[i] {
			out = append(out, c)
		}
	}
	return out
}

// copyOf joins a call to the copy activity it caused. A runtime call whose
// driver calls issued several copies reports their summed bytes and device
// time, with the first copy's kind and stream.
func (j hostAPIJournal) copyOf(c hostAPICall) (hostCopy, bool) {
	if c.domain == "runtime" {
		if cp, ok := j.runtimeCopies[c.correlation]; ok {
			return cp, true
		}
		var sum hostCopy
		joined := false
		for _, corr := range c.driverCorrelations {
			cp, ok := j.copies[corr]
			if !ok {
				continue
			}
			if !joined {
				sum.kind, sum.stream = cp.kind, cp.stream
			}
			joined = true
			sum.bytes += cp.bytes
			sum.nanos += cp.nanos
		}
		if joined {
			return sum, true
		}
	}
	// A copy's correlation names the API record that issued it. On a capture
	// with runtime records and no driver records that is the runtime call:
	// the ds4 C prefill capture's two cudaMemcpy calls join this way and no
	// other.
	cp, ok := j.copies[c.correlation]
	return cp, ok
}

// hostCostGroup is a set of calls one finding reports.
type hostCostGroup struct {
	name        string
	class       hostAPIClass
	calls       int
	hostNanos   int64
	deviceNanos int64

	// joined reports that every copy in the group joined an activity, so
	// its sizes, kind, stream and deviceNanos are read. A copy group whose
	// calls joined none is UNREAD on all four.
	joined             bool
	minBytes, maxBytes uint64
	kind               uint8
	stream             uint32
}

// auditHostAPICost reports synchronous copies, allocations and graph
// instantiates that the program repeats. Each holds the host for its whole
// duration, and when it repeats per iteration that time recurs with every
// step: the capture records it in the API records while the kernel profile
// and every device-time finding are blind to it.
func auditHostAPICost(cap *cudacapture.Capture, report *Report) {
	j, err := readHostAPIJournal(cap)
	if err != nil || len(j.calls) == 0 {
		// No journal, or one without API records: the host cost is not in the
		// capture. The launch and duty-cycle audits already say which.
		return
	}
	calls := dedupRuntimeDriver(j.calls)
	window := j.windowEnd - j.windowStart // positive: j.calls is not empty

	// Copies group by call, direction and stream, not size: a loop whose
	// upload grows with the sequence is still one per-iteration copy.
	type copyKey struct {
		name   string
		joined bool
		kind   uint8
		stream uint32
	}
	copies := make(map[copyKey]*hostCostGroup)
	allocs := make(map[string]*hostCostGroup)
	instantiate := &hostCostGroup{name: "graph instantiate", class: hostAPIInstantiate}
	launches := 0
	for _, c := range calls {
		switch c.class {
		case hostAPISyncCopy:
			cp, ok := j.copyOf(c)
			// A device-to-device copy does not synchronize the host even
			// through the synchronous API, so it holds no host time worth
			// reporting here.
			if ok && cupti.ActivityMemcpyKind(cp.kind) == cupti.ActivityMemcpyKindDtod || !ok && strings.Contains(c.name, "DtoD") {
				continue
			}
			k := copyKey{name: c.name, joined: ok}
			if ok {
				k.kind, k.stream = cp.kind, cp.stream
			}
			g := copies[k]
			if g == nil {
				g = &hostCostGroup{name: c.name, class: c.class, joined: ok, minBytes: cp.bytes, maxBytes: cp.bytes, kind: k.kind, stream: k.stream}
				copies[k] = g
			}
			g.minBytes, g.maxBytes = min(g.minBytes, cp.bytes), max(g.maxBytes, cp.bytes)
			g.calls++
			g.hostNanos += c.nanos()
			g.deviceNanos += cp.nanos
		case hostAPIAlloc:
			g := allocs[c.name]
			if g == nil {
				g = &hostCostGroup{name: c.name, class: c.class}
				allocs[c.name] = g
			}
			g.calls++
			g.hostNanos += c.nanos()
		case hostAPIInstantiate:
			instantiate.calls++
			instantiate.hostNanos += c.nanos()
		case hostAPIGraphLaunch:
			launches++
		}
	}

	var groups []*hostCostGroup
	for _, g := range copies {
		if g.calls >= 2 {
			groups = append(groups, g)
		}
	}
	for _, g := range allocs {
		if g.calls >= 2 {
			groups = append(groups, g)
		}
	}
	// Instantiating a graph is expected once per graph. It is per-iteration
	// when it keeps pace with the replays it is meant to amortize over. API
	// records carry no graph handle, so the counts are over all graphs: one
	// graph replayed often can hide another instantiated per replay.
	if instantiate.calls >= 2 && 2*instantiate.calls >= launches {
		groups = append(groups, instantiate)
	}
	sort.Slice(groups, func(a, b int) bool {
		if groups[a].hostNanos != groups[b].hostNanos {
			return groups[a].hostNanos > groups[b].hostNanos
		}
		return groups[a].name < groups[b].name
	})
	for _, g := range groups {
		report.Items = append(report.Items, hostCostItem(g, launches, window))
	}
}

func hostCostItem(g *hostCostGroup, launches int, window int64) Item {
	host := float64(g.hostNanos) / 1e6
	share := math.NaN()
	if window > 0 {
		share = float64(g.hostNanos) / float64(window) * 100
	}
	magnitude := fmt.Sprintf("host %.2f ms over %d calls", host, g.calls)
	if !math.IsNaN(share) {
		magnitude += fmt.Sprintf(", %.2f%% of the host API window", share)
	}
	item := Item{
		Severity:      SeverityCost,
		Category:      CategoryDispatch,
		EvidenceLevel: EvidenceObserved,
		Location:      Location{CallSite: callSiteUnread},
		HostTimeNanos: g.hostNanos,
	}
	switch g.class {
	case hostAPISyncCopy:
		what := "bytes, direction and stream UNREAD: no copy activity joined these calls"
		device := "device copy time UNREAD"
		if g.joined {
			size := fmt.Sprintf("%d B", g.minBytes)
			if g.minBytes != g.maxBytes {
				size = fmt.Sprintf("%d-%d B", g.minBytes, g.maxBytes)
			}
			what = fmt.Sprintf("%s %s on stream %d", size, copyDirection(g.kind), g.stream)
			device = fmt.Sprintf("device copy time %.2f ms", float64(g.deviceNanos)/1e6)
		}
		magnitude += "; " + device
		// The device runs the copy, and waits idle while the host waits.
		item.terms = []string{"memcpy", "idle"}
		item.Finding = fmt.Sprintf("Synchronous copy %s repeats %d times (%s); the host waits out each one", g.name, g.calls, what)
		item.ActionableRecommendation = "Move the copy out of the loop if its data does not change per iteration; otherwise issue it as an async copy from pinned host memory on a non-default stream, so the host does not wait for the transfer."
		item.NextMeasurement = "recapture and compare this finding's host time; locate the call with a host profiler, since the capture holds no call site"
	case hostAPIAlloc:
		magnitude += "; no device work"
		item.Finding = fmt.Sprintf("Allocation %s repeats %d times; each call blocks the host", g.name, g.calls)
		item.ActionableRecommendation = "Allocate once outside the loop and reuse the buffer, or use a stream-ordered pool allocation (cuMemAllocAsync / cudaMallocAsync) so the allocation does not block the host."
		item.NextMeasurement = "recapture and compare this finding's host time; locate the call with a host profiler, since the capture holds no call site"
	case hostAPIInstantiate:
		magnitude += "; no device work"
		item.Finding = fmt.Sprintf("Graph instantiated %d times for %d graph launches; instantiation is not amortized over replays", g.calls, launches)
		item.ActionableRecommendation = "Instantiate each graph once and replay the executable graph; when only parameters change, update the executable graph (cuGraphExecUpdate or the node SetParams calls) instead of instantiating a new one."
		item.NextMeasurement = "count graph instantiates against graph launches after the change; the capture holds no call site"
	}
	item.magnitude = magnitude
	// A finding is negligible when it is a rounding error in host time AND in
	// the host API window, the same two-sided rule the device demotion uses.
	if g.hostNanos < negligibleDeviceNanos && !math.IsNaN(share) && share < negligibleHostSharePct {
		item.Severity = SeverityContext
	}
	return item
}

// copyDirection names a CUPTI memcpy kind the way a reader writes it.
func copyDirection(kind uint8) string {
	switch cupti.ActivityMemcpyKind(kind) {
	case cupti.ActivityMemcpyKindHtod:
		return "host-to-device"
	case cupti.ActivityMemcpyKindDtoh:
		return "device-to-host"
	case cupti.ActivityMemcpyKindDtod:
		return "device-to-device"
	case cupti.ActivityMemcpyKindHtoh:
		return "host-to-host"
	case cupti.ActivityMemcpyKindPtop:
		return "peer-to-peer"
	}
	return strings.TrimPrefix(cupti.ActivityMemcpyKind(kind).String(), "ActivityMemcpyKind")
}
