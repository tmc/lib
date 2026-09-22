package cudacapture

import (
	"encoding/json"
	"fmt"
	"sort"
	"strconv"

	"github.com/google/pprof/profile"
)

// ActivityRecord represents a parsed GPU activity record (kernel execution or memory copy)
// from a CUPTI journal or execution capture.
type ActivityRecord struct {
	Process             uint64    `json:"process,omitempty"`
	Subtype             string    `json:"subtype"`
	Name                string    `json:"name,omitempty"`
	Valid               bool      `json:"valid"`
	Start               uint64    `json:"start"`
	End                 uint64    `json:"end"`
	Bytes               uint64    `json:"bytes,omitempty"`
	Queued              uint64    `json:"queued,omitempty"`
	Submitted           uint64    `json:"submitted,omitempty"`
	CopyKind            uint8     `json:"copy_kind,omitempty"`
	Device              uint32    `json:"device"`
	Context             uint32    `json:"context,omitempty"`
	Stream              uint32    `json:"stream,omitempty"`
	Correlation         uint32    `json:"correlation,omitempty"`
	RuntimeCorrelation  uint32    `json:"runtime_correlation,omitempty"`
	GridDim             [3]uint32 `json:"grid,omitempty"`
	BlockDim            [3]uint32 `json:"block,omitempty"`
	RegistersPerThread  uint32    `json:"registers_per_thread,omitempty"`
	DynamicSharedMemory uint32    `json:"dynamic_shared_memory,omitempty"`
	GraphID             uint32    `json:"graph_id,omitempty"`
	GraphNodeID         uint64    `json:"graph_node_id,omitempty"`

	// ModuleID is the driver's number for the module the kernel was loaded
	// from, or zero when the producer could not tell. A kernel activity
	// record names only the kernel, so two builds of one graph produce
	// records that are otherwise identical; this is what charges a sample to
	// the build that ran.
	ModuleID uint32 `json:"module_id,omitempty"`

	// Value is the byte a memset wrote, set only when Subtype is "memset".
	// It carries omitempty like its neighbours, so a memset of zero -- the
	// common case -- is indistinguishable from an absent value; the subtype
	// is what says the field means anything.
	Value uint32 `json:"value,omitempty"`

	// External correlation fields, set only when Subtype is
	// "external_correlation". Such a record is not device work: it is the
	// producer reporting that the framework pushed ExternalID onto the
	// ExternalKind stack while the native correlation ID in Correlation was
	// live. It carries no interval, so it becomes no sample.
	ExternalKind uint32 `json:"external_kind,omitempty"`
	ExternalID   uint64 `json:"external_id,omitempty"`

	// OverheadKind and ObjectKind are set only when Subtype is "overhead".
	// ObjectKind is load-bearing rather than decorative: a CUPTI overhead
	// names an object, which is usually a host thread, and Device is
	// meaningful only when ObjectKind is a device, a context or a stream.
	// Reading Device without it turns every process-scoped overhead into one
	// belonging to device 0.
	OverheadKind uint32 `json:"overhead_kind,omitempty"`
	ObjectKind   uint32 `json:"object_kind,omitempty"`

	// SyncType and CudaEventID are set only when Subtype is "sync". A
	// synchronization record carries a context and a stream and no device at
	// all, so Device on one of these is zero because nothing filled it, not
	// because the sync was on device 0.
	SyncType    uint32 `json:"sync_type,omitempty"`
	CudaEventID uint32 `json:"cuda_event_id,omitempty"`

	// NVTX marker fields, set only when Subtype is "marker". A range is two
	// records sharing MarkerID on one thread: CUPTI reports the open and the
	// close separately, so no single record holds a duration. Pairing needs
	// the process and thread as well as the id, because ids are unique only
	// within the thread that emitted them.
	ProcessID    uint32 `json:"process_id,omitempty"`
	ThreadID     uint32 `json:"thread_id,omitempty"`
	MarkerDomain string `json:"marker_domain,omitempty"`
	MarkerID     uint32 `json:"marker_id,omitempty"`
	MarkerFlags  uint32 `json:"marker_flags,omitempty"`
}

// NVTX marker flags, as CUPTI reports them in CUpti_ActivityMarker2.flags.
const (
	markerFlagInstantaneous = 1 << 0
	markerFlagStart         = 1 << 1
	markerFlagEnd           = 1 << 2
)

// nvtxRange is one paired NVTX range: a name and the window it was open.
type nvtxRange struct {
	name  string
	start uint64
	end   uint64
}

// nvtxRanges pairs the marker records in activities into ranges.
//
// Journal order is drain order, which several producer threads interleave, so
// pairing sorts by time first. Opens are stacked per (process, thread, id,
// domain): the same id reopened before its close is a legitimate nested range,
// and matching the close to the most recent open is what keeps the nesting
// right. An unpaired close, or an open the target never closed, contributes no
// range -- a window with only one end is not a window.
func nvtxRanges(activities []ActivityRecord) []nvtxRange {
	var markers []ActivityRecord
	for _, a := range activities {
		if a.Subtype == "marker" && a.Valid {
			markers = append(markers, a)
		}
	}
	if len(markers) == 0 {
		return nil
	}
	sort.SliceStable(markers, func(i, j int) bool { return markers[i].Start < markers[j].Start })

	type key struct {
		pid, tid, id uint32
		domain       string
	}
	open := map[key][]ActivityRecord{}
	var out []nvtxRange
	for _, m := range markers {
		k := key{pid: m.ProcessID, tid: m.ThreadID, id: m.MarkerID, domain: m.MarkerDomain}
		switch {
		case m.MarkerFlags&markerFlagStart != 0:
			open[k] = append(open[k], m)
		case m.MarkerFlags&markerFlagEnd != 0:
			stack := open[k]
			if len(stack) == 0 {
				continue
			}
			start := stack[len(stack)-1]
			open[k] = stack[:len(stack)-1]
			out = append(out, nvtxRange{name: nvtxName(start), start: start.Start, end: m.End})
		}
		// An instantaneous marker is a point, not a window, so it encloses
		// nothing and is not a range.
	}
	return out
}

// nvtxName qualifies a range name with its NVTX domain, because two domains
// may use the same range name for unrelated work.
func nvtxName(m ActivityRecord) string {
	if m.MarkerDomain == "" {
		return m.Name
	}
	return m.MarkerDomain + "/" + m.Name
}

// nvtxLabels returns the names of every range open at t, outermost first.
//
// An activity is attributed to the ranges that were open when it STARTED, so
// every activity belongs to a well-defined set and the per-range times sum to
// the total device time. A kernel still running when a range closes counts
// wholly to the range it began in; attributing it by overlap instead would
// count it under two ranges and make the per-range times sum to more than the
// device ever spent.
//
// Every enclosing range is named, not just the innermost, so that focusing an
// outer range in pprof also finds the work inside its nested children.
func nvtxLabels(ranges []nvtxRange, t uint64) []string {
	var open []nvtxRange
	for _, r := range ranges {
		if r.start <= t && t < r.end {
			open = append(open, r)
		}
	}
	if len(open) == 0 {
		return nil
	}
	// Outermost first: an enclosing range opens no later than the range it
	// encloses, and where two open together the longer one is the outer.
	sort.SliceStable(open, func(i, j int) bool {
		if open[i].start != open[j].start {
			return open[i].start < open[j].start
		}
		return open[i].end > open[j].end
	})
	names := make([]string, 0, len(open))
	for _, r := range open {
		names = append(names, r.name)
	}
	return names
}

// UnmarshalJSON unmarshals an ActivityRecord from JSON. It supports array
// dimensions (grid, block, grid_dim, block_dim) and coordinate fields
// (grid_x, grid_y, grid_z, block_x, block_y, block_z), as well as label shorthand
// aliases (registers, smem).
func (a *ActivityRecord) UnmarshalJSON(data []byte) error {
	type Alias ActivityRecord
	var raw struct {
		Alias
		GridDimRaw  *[3]uint32 `json:"grid_dim"`
		BlockDimRaw *[3]uint32 `json:"block_dim"`
		GridX       uint32     `json:"grid_x"`
		GridY       uint32     `json:"grid_y"`
		GridZ       uint32     `json:"grid_z"`
		BlockX      uint32     `json:"block_x"`
		BlockY      uint32     `json:"block_y"`
		BlockZ      uint32     `json:"block_z"`
		Registers   uint32     `json:"registers"`
		Smem        uint32     `json:"smem"`
	}
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	*a = ActivityRecord(raw.Alias)
	if raw.GridDimRaw != nil {
		a.GridDim = *raw.GridDimRaw
	}
	if a.GridDim == [3]uint32{0, 0, 0} && (raw.GridX != 0 || raw.GridY != 0 || raw.GridZ != 0) {
		a.GridDim = [3]uint32{raw.GridX, raw.GridY, raw.GridZ}
	}
	if raw.BlockDimRaw != nil {
		a.BlockDim = *raw.BlockDimRaw
	}
	if a.BlockDim == [3]uint32{0, 0, 0} && (raw.BlockX != 0 || raw.BlockY != 0 || raw.BlockZ != 0) {
		a.BlockDim = [3]uint32{raw.BlockX, raw.BlockY, raw.BlockZ}
	}
	if a.RegistersPerThread == 0 && raw.Registers != 0 {
		a.RegistersPerThread = raw.Registers
	}
	if a.DynamicSharedMemory == 0 && raw.Smem != 0 {
		a.DynamicSharedMemory = raw.Smem
	}
	return nil
}

// MarshalJSON marshals an ActivityRecord, omitting launch geometry and register metrics
// for non-kernel activities or when zero.
func (a ActivityRecord) MarshalJSON() ([]byte, error) {
	type Alias ActivityRecord
	if a.Subtype != "kernel" && a.GridDim == [3]uint32{0, 0, 0} && a.BlockDim == [3]uint32{0, 0, 0} && a.RegistersPerThread == 0 && a.DynamicSharedMemory == 0 {
		var raw struct {
			Alias
			GridDim             any `json:"grid,omitempty"`
			BlockDim            any `json:"block,omitempty"`
			RegistersPerThread  any `json:"registers_per_thread,omitempty"`
			DynamicSharedMemory any `json:"dynamic_shared_memory,omitempty"`
		}
		raw.Alias = Alias(a)
		return json.Marshal(raw)
	}
	return json.Marshal(Alias(a))
}

// AttachKernelLabels adds kernel launch geometry and register allocation labels
// ('grid', 'block', 'registers', 'smem') to the provided label map, and
// 'module_id' when the producer knew which module the kernel came from.
// If labels is nil or rec is not a kernel, AttachKernelLabels does nothing.
func AttachKernelLabels(labels map[string][]string, rec ActivityRecord) {
	if labels == nil || rec.Subtype != "kernel" {
		return
	}
	labels["grid"] = []string{fmt.Sprintf("%d,%d,%d", rec.GridDim[0], rec.GridDim[1], rec.GridDim[2])}
	labels["block"] = []string{fmt.Sprintf("%d,%d,%d", rec.BlockDim[0], rec.BlockDim[1], rec.BlockDim[2])}
	labels["registers"] = []string{strconv.FormatUint(uint64(rec.RegistersPerThread), 10)}
	labels["smem"] = []string{strconv.FormatUint(uint64(rec.DynamicSharedMemory), 10)}
	// Graph attribution is present only for kernels that ran as nodes of a
	// CUDA graph. A zero graphId is CUPTI's "not a graph node", so the labels
	// are omitted entirely rather than set to zero: their presence is the
	// signal, and a zero-valued label would make every launch look graph-like.
	if rec.GraphID != 0 {
		labels["graph_id"] = []string{strconv.FormatUint(uint64(rec.GraphID), 10)}
	}
	if rec.GraphNodeID != 0 {
		labels["graph_node_id"] = []string{strconv.FormatUint(rec.GraphNodeID, 10)}
	}
	// Absent rather than zero, for the same reason as the graph labels: a
	// producer that could not follow the kernel back to a module says
	// nothing, and a consumer splitting its tables by module counts those
	// records in none of them and says how many there were.
	if rec.ModuleID != 0 {
		labels["module_id"] = []string{strconv.FormatUint(uint64(rec.ModuleID), 10)}
	}
}

// BuildActivityProfile converts a slice of ActivityRecord entries into a pprof Profile.
// Kernel samples are annotated with launch geometry and register labels
// ('grid', 'block', 'registers', 'smem'), and with 'graph_id' and
// 'graph_node_id' when the kernel ran as a node of a CUDA graph.
func BuildActivityProfile(session string, targetPID uint64, activities []ActivityRecord) *profile.Profile {
	p, _ := BuildActivityProfileIndex(session, targetPID, activities)
	return p
}

// BuildActivityProfileIndex builds the profile BuildActivityProfile builds and
// additionally reports, for each sample, the index in activities the sample was
// projected from. Callers that attribute samples back to native records need
// that association from the projection itself: the profile holds fewer samples
// than there are records, and which records were skipped is this function's
// decision, not one a caller can safely re-derive.
func BuildActivityProfileIndex(session string, targetPID uint64, activities []ActivityRecord) (*profile.Profile, []int) {
	var indices []int
	p := &profile.Profile{
		SampleType: []*profile.ValueType{
			{Type: "kernel_time", Unit: "nanoseconds"},
			{Type: "copy_time", Unit: "nanoseconds"},
			{Type: "copy_bytes", Unit: "bytes"},
			{Type: "activity", Unit: "count"},
		},
		// Without this pprof opens the last sample type, "activity", which is
		// a launch count and not a duration. Every consumer would otherwise
		// have to remember -sample_index=0 to see device time.
		DefaultSampleType: "kernel_time",
	}
	// NVTX ranges are paired once for the whole capture: the marker records
	// that define them are interleaved with the device work they enclose, so
	// the windows have to be known before any activity can be placed in them.
	ranges := nvtxRanges(activities)
	functions := make(map[string]*profile.Function)
	var nextFunctionID uint64 = 1
	var nextLocationID uint64 = 1
	for recordIndex, activity := range activities {
		if !activity.Valid {
			continue
		}
		if activity.End < activity.Start || activity.End-activity.Start > uint64(^uint64(0)>>1) {
			continue
		}
		// NVTX markers share the activity record kind but are host
		// annotations, not device work. They contribute no device or copy
		// time, so including them would leave the count dimension counting a
		// different population than the two duration dimensions beside it.
		if activity.Subtype == "marker" || activity.Subtype == "capture_range" {
			continue
		}
		// An external-correlation record is a link between a framework scope
		// and a native correlation ID, with no interval of its own. It is
		// carried alongside device work so the join can see both in one
		// stream, and it is skipped here for the same reason a marker is.
		//
		// Neither is the CUPTI overhead tier, the graph replay envelope, nor
		// a synchronization window. The profile's four dimensions are kernel
		// time, memory time, bytes and a count; none of the three has a value
		// for the first three, so each would arrive as a sample counting one
		// of something that took no time and moved no bytes. The count
		// dimension would then be counting a different population than the
		// duration dimensions beside it, which is the same defect the marker
		// exclusion above exists for.
		//
		// A graph trace is excluded for a second reason: its span contains
		// the kernel records of the nodes it replayed, so admitting it would
		// charge that device time twice.
		if activity.Subtype == "external_correlation" || activity.Subtype == "overhead" || activity.Subtype == "graph_trace" || activity.Subtype == "sync" {
			continue
		}
		value := []int64{0, 0, 0, 1}
		if activity.Subtype == "kernel" {
			value[0] = int64(activity.End - activity.Start)
		} else if activity.Subtype == "memcpy" || activity.Subtype == "memset" {
			value[1] = int64(activity.End - activity.Start)
			value[2] = int64(activity.Bytes)
		}
		labels := map[string][]string{
			"subtype":             {activity.Subtype},
			"name":                {activity.Name},
			"session":             {session},
			"pid":                 {strconv.FormatUint(targetPID, 10)},
			"device":              {strconv.FormatUint(uint64(activity.Device), 10)},
			"context":             {strconv.FormatUint(uint64(activity.Context), 10)},
			"stream":              {strconv.FormatUint(uint64(activity.Stream), 10)},
			"correlation":         {strconv.FormatUint(uint64(activity.Correlation), 10)},
			"runtime_correlation": {strconv.FormatUint(uint64(activity.RuntimeCorrelation), 10)},
			"queued":              {strconv.FormatUint(activity.Queued, 10)},
			"submitted":           {strconv.FormatUint(activity.Submitted, 10)},
			"copy_kind":           {strconv.FormatUint(uint64(activity.CopyKind), 10)},
		}
		if activity.Process != 0 {
			labels["pid"] = []string{strconv.FormatUint(activity.Process, 10)}
		}
		// The NVTX ranges this activity ran inside, outermost first. The label
		// is absent rather than empty when it ran inside none, so that pprof's
		// -tagfocus separates "outside every range" from "inside a range whose
		// name happens to be empty".
		if names := nvtxLabels(ranges, activity.Start); len(names) > 0 {
			labels["nvtx"] = names
		}
		if activity.Subtype == "kernel" {
			AttachKernelLabels(labels, activity)
		}
		name := activity.Name
		if activity.Subtype == "memcpy" && name == "" {
			name = "cudaMemcpy"
		}
		if activity.Subtype == "memset" && name == "" {
			name = "cudaMemset"
		}
		fn := functions[name]
		if fn == nil {
			fn = &profile.Function{ID: nextFunctionID, Name: name, SystemName: name}
			nextFunctionID++
			functions[name] = fn
			p.Function = append(p.Function, fn)
		}
		loc := &profile.Location{ID: nextLocationID, Address: 0, Line: []profile.Line{{Function: fn}}}
		nextLocationID++
		p.Location = append(p.Location, loc)
		p.Sample = append(p.Sample, &profile.Sample{Value: value, Label: labels, Location: []*profile.Location{loc}})
		indices = append(indices, recordIndex)
	}
	return p, indices
}
