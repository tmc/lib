package gaps

import "github.com/tmc/lib/nvidia/x/cudaprof/timeline"

// A Report is the gap analysis of one capture.
type Report struct {
	// Capture names the capture analyzed. Analyze leaves it empty for the
	// caller to fill in.
	Capture    string             `json:"capture"`
	Journals   []timeline.Journal `json:"journals"`
	Activities int                `json:"activities"`
	APIRecords int                `json:"api_records"`
	Correlated float64            `json:"correlation_match_pct"`
	WarmupMS   float64            `json:"warmup_ms"`
	// WarmupUntil, when set, replaces WarmupMS: the steady-state window
	// begins at the first kernel whose name matches it.
	WarmupUntil string   `json:"warmup_until,omitempty"`
	Devices     []Device `json:"devices"`

	// Synchronization spans whose context no device-carrying record named, so
	// no device timeline could claim them. They are stated here rather than
	// folded into device 0: an explanation placed on the wrong device is
	// worse than one that says it has no device.
	UnresolvedSyncs     int   `json:"unresolved_sync_records"`
	UnresolvedSyncNanos int64 `json:"unresolved_sync_ns"`

	// Kernels is the per-kernel census: how many times each kernel ran and
	// how much device time it held, both taken from the same records.
	Kernels []Kernel `json:"kernels,omitempty"`

	// UnnamedKernelRecords counts kernel activities the journal carried
	// without a name. They are stated rather than dropped: a census whose
	// rows silently omit a population reports a total that does not add up
	// to the device-active time beside it.
	UnnamedKernelRecords int `json:"unnamed_kernel_records,omitempty"`

	// OpaqueGraphReplays reports that the capture held graph replay envelopes
	// and no device record naming a graph id, so the replays are counted as
	// device activity from their envelopes alone. The idle inside them is not
	// observable and is reported as busy.
	OpaqueGraphReplays bool `json:"opaque_graph_replays,omitempty"`

	// GraphTraceMode is the graph trace mode the agent reported: node records
	// per graph node, or one envelope per replay. Empty is UNREAD: the agent
	// predates the field or the capture carries no capability record. Analyze
	// leaves it empty for the caller, which owns the agent's capability
	// vocabulary, to fill in.
	GraphTraceMode string `json:"graph_trace_mode,omitempty"`
}

// Device holds the whole-run and steady-state timelines of one device.
type Device struct {
	Device      uint32 `json:"device"`
	Overall     *Stats `json:"overall"`
	SteadyState *Stats `json:"steady_state,omitempty"`

	// SteadyActivities counts the activities past the warmup. SteadyState is
	// omitted both when the warmup never ended -- zero here -- and when it
	// excluded nothing, and those two must not read alike: the first means
	// the steady-state window does not exist.
	SteadyActivities int `json:"steady_state_activities"`
}

// Stats describes one merged device timeline.
type Stats struct {
	Label           string       `json:"label"`
	Activities      int          `json:"activities"`
	MergedIntervals int          `json:"merged_intervals"`
	SpanNanos       int64        `json:"span_ns"`
	ActiveNanos     int64        `json:"active_ns"`
	IdleNanos       int64        `json:"idle_ns"`
	DutyCyclePct    float64      `json:"duty_cycle_pct"`
	GapCount        int          `json:"gap_count"`
	Distribution    Distribution `json:"distribution_ns"`
	Concurrency     []Class      `json:"host_concurrency"`
	Attribution     Attribution  `json:"idle_attribution_ns"`
	Overlay         Overlay      `json:"idle_overlay_ns"`
}

// Overlay names the idle time that something in the journal explains
// directly: CUPTI was flushing, the host was waiting on a synchronize, or the
// device was between the nodes of a graph replay.
//
// It overlays [Attribution] rather than extending it. The three numbers there
// partition idle and sum to IdleNanos; these do not. One gap can be both host
// behind and spent inside a cudaStreamSynchronize, one overhead span can cover
// several gaps, and a graph replay envelope can contain both. Adding an
// overlay figure to an attribution figure therefore double counts, and adding
// two overlay figures to each other can exceed idle.
type Overlay struct {
	// CuptiOverheadNanos is idle time covered by a CUPTI overhead span: time
	// the tool cost the run, which the tool would otherwise report as the
	// program leaving the device idle. Overhead records carry a host thread
	// rather than a device, so the same spans are applied to every device.
	CuptiOverheadNanos int64 `json:"cupti_overhead"`
	CuptiOverheadGaps  int   `json:"cupti_overhead_gap_count"`
	// SyncNanos is idle covered by a synchronization span on this device.
	SyncNanos int64 `json:"synchronization"`
	SyncGaps  int   `json:"synchronization_gap_count"`
	// GraphReplayNanos is idle inside a graph replay envelope: the device was
	// between nodes of a launched graph, which is graph scheduling cost and
	// not the host failing to keep up.
	GraphReplayNanos int64 `json:"graph_replay"`
	GraphReplayGaps  int   `json:"graph_replay_gap_count"`
}

// Distribution reports the shape of the idle gaps, not only their total. A
// duty cycle built from a few long synchronization stalls and the same duty
// cycle built from thousands of uniform dispatch gaps are different
// engineering problems, and only the distribution separates them.
type Distribution struct {
	Min    int64   `json:"min"`
	P25    int64   `json:"p25"`
	Median int64   `json:"median"`
	P75    int64   `json:"p75"`
	P90    int64   `json:"p90"`
	P99    int64   `json:"p99"`
	Max    int64   `json:"max"`
	Mean   float64 `json:"mean"`
}

// Class is one host concurrency class with its count and its own idle time
// and idle-gap distribution, so a class that is rare but long is not hidden by
// a class that is frequent and short.
type Class struct {
	Name      string  `json:"name"`
	Meaning   string  `json:"meaning"`
	Count     int     `json:"count"`
	IdleNanos int64   `json:"idle_ns"`
	Median    int64   `json:"median_ns"`
	Mean      float64 `json:"mean_ns"`
}

// Attribution splits idle time by where it was spent, as distinct from
// which class the gap fell into.
type Attribution struct {
	HostBehindNanos   int64 `json:"host_behind"`
	DriverSubmitNanos int64 `json:"driver_submission"`
	HardwareNanos     int64 `json:"hardware_queue"`
	// UnattributedNanos is idle before an activity whose correlation id
	// matched no API record. It is its own arm because there is no mechanism
	// to name: the launch that would say where the time went was not found.
	// It was previously added to HardwareNanos, which reported "we could not
	// find the launch" as "hardware queue latency" -- a specific mechanism
	// claim, and the one a reader acts on. Any future record kind arriving
	// without a correlation id lands here rather than inventing a cause.
	UnattributedNanos int64 `json:"unattributed"`
}

// The host concurrency classes of [Class].
const (
	ClassHostAhead  = "Host Ahead"
	ClassHostDuring = "Host During"
	ClassHostBehind = "Host Behind"
	ClassUnmatched  = "Unmatched"
)

var classMeaning = map[string]string{
	ClassHostAhead:  "launch returned before the previous op ended: idle is hardware queue latency",
	ClassHostDuring: "previous op ended inside the launch call: the GPU waited on the driver",
	ClassHostBehind: "launch started after the previous op ended: the GPU was starved by the host",
	ClassUnmatched:  "no API record correlated to the next op: idle is unattributed",
}

// Kernel is one kernel's launch count, device time, and place in the
// timeline in one capture.
type Kernel struct {
	Name        string `json:"name"`
	Launches    int    `json:"launches"`
	DeviceNanos int64  `json:"device_ns"`

	// MedianOrderFrac is where this kernel's median launch sat in the ORDER
	// of the capture's kernel launches: 0 at the first launch of the run, 1
	// at the last.
	//
	// A launch count says whether the work ran, never when. A change that
	// relocates work -- the same launches, in a different phase of the run --
	// leaves every count identical, and a census reading counts alone reports
	// the kernel as untouched.
	//
	// Order, not elapsed time. Position measured as a fraction of the device
	// span moves whenever any duration changes: a kernel that got slower
	// stretches the span it is divided by, and its own median start slides
	// backwards through a run it never moved in. The launch order cannot be
	// perturbed by a duration at all, so what remains is what the census is
	// asking about -- whether the same work sits in the same part of the run.
	MedianOrderFrac float64 `json:"median_launch_order_frac"`
}
