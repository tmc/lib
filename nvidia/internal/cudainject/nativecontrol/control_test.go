// Package nativecontrol holds the controls for the CUPTI producer in
// ../native. It has no non-test source.
//
// The producer is compiled against the stub CUPTI in native/testcupti and
// driven by native/cupti_agent_control.c, so every scenario exercises the
// shipping code path without a GPU or a CUDA toolkit. The harness reports
// only what it offered; the assertions here decode the journal the producer
// actually wrote.
package nativecontrol_test

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
	"github.com/tmc/lib/nvidia/x/cudaprof/timeline"
)

// Record kinds, mirroring the KIND_ constants in cupti_agent.c.
const (
	kindBootstrap  = 1
	kindActivity   = 2
	kindFinalized  = 3
	kindCapability = 4
	kindFailure    = 5
	kindExternal   = 6
	kindAPI        = 7
	kindHost       = 8
	kindModule     = 9
)

const sessionHex = "00112233445566778899aabbccddeeff"

// finalizedRecord is the producer's terminal accounting record.
type finalizedRecord struct {
	State                string `json:"state"`
	Trigger              string `json:"trigger"`
	Activity             uint64 `json:"activity"`
	Dropped              uint64 `json:"dropped"`
	NativeDropped        uint64 `json:"native_dropped"`
	QueueDropped         uint64 `json:"queue_dropped"`
	QueueOversizeDropped uint64 `json:"queue_oversize_dropped"`
	QueueContentionDrop  uint64 `json:"queue_contention_dropped"`
	QueueCapacityDropped uint64 `json:"queue_capacity_dropped"`
	Offered              uint64 `json:"offered"`
	Retained             uint64 `json:"retained"`
	BackpressureWaits    uint64 `json:"backpressure_waits"`
	BackpressureNanos    uint64 `json:"backpressure_nanos"`
	BackpressureMaxNanos uint64 `json:"backpressure_max_nanos"`
	DrainBatches         uint64 `json:"drain_batches"`
	DrainMaxBatch        uint64 `json:"drain_max_batch"`
	DrainWriteNanos      uint64 `json:"drain_write_nanos"`
	DrainWriteMaxNanos   uint64 `json:"drain_write_max_nanos"`
	Callbacks            uint64 `json:"callbacks"`
	CallbackNanos        uint64 `json:"callback_nanos"`
	CallbackMaxNanos     uint64 `json:"callback_max_nanos"`

	// ByKind splits offered/retained/dropped by the wire kind the record was
	// admitted as, keyed by the producer's own names for those kinds. Without
	// it no loss statement about one kind is separable from the others: the
	// ring admits every kind through one path.
	ByKind map[string]kindCounts `json:"by_kind"`

	// HostSamples counts intervals the host sampler completed;
	// HostSampleRecords counts the journal records they became. They differ
	// whenever a sample held more threads than one payload fits, which is the
	// case the batching exists for.
	HostSamples       uint64 `json:"host_samples"`
	HostSampleRecords uint64 `json:"host_sample_records"`

	// ModuleSeenMax is the producer's duplicate-table bound, and
	// ModulesUnchecked counts the module loads that arrived after it filled
	// and were written without a duplicate check.
	ModuleSeenMax    uint64 `json:"module_seen_max"`
	ModulesUnchecked uint64 `json:"modules_unchecked"`
}

// moduleKernel is a kernel record's launch correlation and the module the
// producer joined it to.
type moduleKernel struct {
	Correlation uint32 `json:"correlation"`
	ModuleID    uint32 `json:"module_id"`
}

// moduleRecord is one module image record.
type moduleRecord struct {
	ID        uint32 `json:"module_id"`
	Stored    bool   `json:"module_stored"`
	Unchecked bool   `json:"module_unchecked"`
}

// kindCounts is one kind's share of the producer's accounting.
type kindCounts struct {
	Offered  uint64 `json:"offered"`
	Retained uint64 `json:"retained"`
	Dropped  uint64 `json:"dropped"`
}

// harnessResult is the single JSON line cupti_agent_control.c prints.
type harnessResult struct {
	Scenario      string `json:"scenario"`
	Threads       int    `json:"threads"`
	Buffers       int    `json:"buffers"`
	PerBuffer     int    `json:"per_buffer"`
	Offered       uint64 `json:"offered"`
	PoolExhausted uint64 `json:"pool_exhausted"`
	AppTerminated int    `json:"app_terminated"`
	ChildExited   int    `json:"child_exited"`
	ChildStatus   int    `json:"child_status"`

	// fork-inert only: how the child ended, and what it observed about the
	// agent it inherited. A negative probe field means the child could not
	// answer that question on this host.
	ChildSignaled      int `json:"child_signaled"`
	ChildTermsig       int `json:"child_termsig"`
	ProbeRead          int `json:"probe_read"`
	ParentPushResult   int `json:"parent_push_result"`
	ParentPopResult    int `json:"parent_pop_result"`
	ParentJournalFDs   int `json:"parent_journal_fds"`
	ParentPipeFDs      int `json:"parent_pipe_fds"`
	ChildBufferGranted int `json:"child_buffer_granted"`
	ChildPushResult    int `json:"child_push_result"`
	ChildPopResult     int `json:"child_pop_result"`
	ChildJournalFDs    int `json:"child_journal_fds"`
	ChildPipeFDs       int `json:"child_pipe_fds"`

	// finalize-export and flush-interval only. After is what the harness
	// offered once the capture was finalized, which must be nothing: the
	// entry points are shut. FinalizeRC and FinalizeRCAgain are the exported
	// finalize's two returns, and Flushes is how many times the producer's
	// interval flush actually reached CUPTI.
	After           int `json:"after"`
	FinalizeRC      int `json:"finalize_rc"`
	FinalizeRCAgain int `json:"finalize_rc_again"`
	Flushes         int `json:"flushes"`

	// injection only: the two returns of InitializeInjection.
	InitRC      int `json:"init_rc"`
	InitRCAgain int `json:"init_rc_again"`

	// module-chains only: how many API and resource callbacks the stub
	// delivered to the producer.
	Fired int `json:"fired"`
}

// cuptiSuccess and cuptiErrorNotSupported are the stub CUPTI's values for
// CUPTI_SUCCESS and CUPTI_ERROR_NOT_SUPPORTED, which is what the producer's
// external-correlation wrappers return once they are shut. The stub numbers
// its own enum; these are not the real CUPTI values.
const (
	cuptiSuccess           = 0
	cuptiErrorNotSupported = 1
)

// capture is one decoded producer journal.
type capture struct {
	counts    map[uint16]uint64
	finalized *finalizedRecord
	// finalizedAt is the finalized record's header timestamp.
	finalizedAt uint64
	capability  capabilityRecord
	// capabilityJSON is the record as written, for assertions about absence.
	capabilityJSON []byte
	failures       []string
	markers        []markerRecord
	memsets        []memsetRecord
	overheads      []overheadRecord
	graphs         []graphTraceRecord
	// graphNodes are the kernel records that name a graph node.
	graphNodes     []graphNodeKernel
	graphLifecycle []timeline.GraphEvent
	syncs          []syncRecord
	hosts          []hostSampleRecord
	modules        []moduleRecord
	// moduleKernels are the module-chains scenario's kernel records.
	moduleKernels []moduleKernel
	// order is every record's kind in journal order. The producer's ordering
	// invariant -- nothing before bootstrap and capability, nothing after
	// finalized -- is a property of the sequence, not of the counts, so it
	// can only be asserted from this.
	order     []uint16
	ranges    []rangeEdgeRecord
	truncated bool
	// deviceActive counts the producer's device_active notes. The note is
	// written straight to the journal rather than through the ring, because
	// the drain that empties the ring is what an exiting target never
	// reaches, so it is out of band by construction and every reconciliation
	// below has to take it out of the journal's side of the equation.
	deviceActive int
}

// rangeEdgeRecord is one capture-range edge as the producer wrote it. These are
// the producer's own bookkeeping rather than device work: they are what lets a
// consumer tell an empty journal from one whose range never opened, so their
// presence and their fields have to be asserted from the journal itself.
type rangeEdgeRecord struct {
	Subtype        string `json:"subtype"`
	Edge           string `json:"edge"`
	API            string `json:"api"`
	Index          uint   `json:"index"`
	ToggleFailures int    `json:"toggle_failures"`
	FlushNanos     uint64 `json:"flush_nanos"`
	Timestamp      uint64 `json:"-"`
}

// markerRecord is one NVTX marker as the producer transcribed it. A range
// occupies two of these: CUPTI reports the open and the close separately, so
// the duration is a property of a pair and of nothing in this struct.
type markerRecord struct {
	Subtype    string `json:"subtype"`
	Name       string `json:"name"`
	Domain     string `json:"marker_domain"`
	ID         uint32 `json:"marker_id"`
	Flags      uint32 `json:"marker_flags"`
	ObjectKind uint32 `json:"object_kind"`
	ProcessID  uint32 `json:"process_id"`
	ThreadID   uint32 `json:"thread_id"`
	Stream     uint32 `json:"stream"`
	Start      uint64 `json:"start"`
	End        uint64 `json:"end"`
}

// memsetRecord is one CUPTI_ACTIVITY_KIND_MEMSET activity as the producer
// transcribed it. A memset is device work with a duration: before the kind was
// collected, the device time it occupied was absent from the journal entirely,
// and every consumer that derives idle time from this stream reported that
// time as idle.
type memsetRecord struct {
	Subtype     string `json:"subtype"`
	Valid       bool   `json:"valid"`
	Start       uint64 `json:"start"`
	End         uint64 `json:"end"`
	Bytes       uint64 `json:"bytes"`
	Value       uint32 `json:"value"`
	Device      uint32 `json:"device"`
	Context     uint32 `json:"context"`
	Stream      uint32 `json:"stream"`
	Correlation uint32 `json:"correlation"`
}

// hostSampleRecord is one part of one host run-state sample. A sample is one
// or more of these: the payload is capped, a process can have more threads
// than fit, and the parts carry the totals so a consumer can tell a complete
// sample from a prefix of one. Truncating instead would report a subset of the
// threads as the whole process.
type hostSampleRecord struct {
	Kind         string            `json:"kind"`
	Sample       uint64            `json:"sample"`
	Part         uint              `json:"part"`
	Wall         uint64            `json:"wall"`
	Last         bool              `json:"last"`
	InPart       int               `json:"in_part"`
	ThreadsTotal int               `json:"threads_total"`
	Unreadable   uint64            `json:"unreadable"`
	Overflowed   uint64            `json:"overflowed"`
	Threads      []hostThreadState `json:"threads"`
}

// hostThreadState is one thread's run state as /proc reported it. The CPU
// counters are in clock ticks, the unit the kernel writes; the tick rate
// travels in the capability record so a consumer never has to assume 100.
type hostThreadState struct {
	TID   uint64 `json:"tid"`
	State string `json:"state"`
	UTime uint64 `json:"utime_ticks"`
	STime uint64 `json:"stime_ticks"`
}

// Marker flags, mirroring CUpti_ActivityFlag in native/testcupti.
const (
	markerFlagInstantaneous = 1 << 0
	markerFlagStart         = 1 << 1
	markerFlagEnd           = 1 << 2
)

// capabilityRecord is what the producer reports it enabled.
type capabilityRecord struct {
	Activity         bool   `json:"activity"`
	APIRecords       bool   `json:"api_records"`
	LayoutCheck      string `json:"layout_check"`
	AgentRevision    string `json:"agent_revision"`
	DriverAPI        bool   `json:"driver_api"`
	RuntimeAPI       bool   `json:"runtime_api"`
	SigtermFinalize  bool   `json:"sigterm_finalize"`
	AtforkGuard      bool   `json:"atfork_guard"`
	NVTXMarkers      bool   `json:"nvtx_markers"`
	Memsets          bool   `json:"memsets"`
	MemsetsRequested bool   `json:"memsets_requested"`
	Overheads        bool   `json:"overheads"`
	GraphTraces      bool   `json:"graph_traces"`
	GraphTraceMode   string `json:"graph_trace_mode"`
	Synchronization  bool   `json:"synchronization"`
	ClocksUnified    bool   `json:"clocks_unified"`
	CUPTIClock       string `json:"cupti_clock"`
	HostSample       string `json:"host_sample"`
	HostSampleHz     int    `json:"host_sample_hz"`
	HostSampleRoot   string `json:"host_sample_root"`
	RootOverride     bool   `json:"host_sample_root_override"`
	ClockTicks       int64  `json:"clock_ticks_per_second"`
	CaptureRange     string `json:"capture_range"`
	Finalize         string `json:"finalize"`
	FlushIntervalMS  int    `json:"flush_interval_ms"`
	DrainBatch       int    `json:"drain_batch"`
	QueueCapacity    int    `json:"queue_capacity"`
	AdmitWaitNanos   int64  `json:"admit_wait_nanos"`
	KernelLayout     string `json:"kernel_layout"`
	CrashTrace       string `json:"crash_trace"`
	CrashTraceDepth  int    `json:"crash_trace_depth"`
}

var (
	buildMu    sync.Mutex
	buildCache = map[string]buildResult{}
	// binRevisions maps a built harness to the agent revision its build was
	// told, so a decode can check the journal against the agent that wrote it
	// rather than assume they are the same. See producerMismatch.
	binRevisions = map[string]string{}
)

type buildResult struct {
	path string
	err  error
}

// agentRevisionPrefix is how the build recipe names the agent's source
// revision. controlBinary reads its own defines rather than being told twice:
// a separately passed expectation could disagree with the define and the
// disagreement would be invisible.
const agentRevisionPrefix = "-DCUDAPROF_AGENT_REVISION="

// agentRevisionOf is the revision a build with these defines stamps into its
// capability record, or "" when the build was not told one.
func agentRevisionOf(defines []string) string {
	rev := ""
	for _, d := range defines {
		if after, ok := strings.CutPrefix(d, agentRevisionPrefix); ok {
			rev = strings.Trim(after, `"`)
		}
	}
	return rev
}

// binRevision is the agent revision a harness built by controlBinary stamps.
func binRevision(bin string) string {
	buildMu.Lock()
	defer buildMu.Unlock()
	return binRevisions[bin]
}

// producerMismatch reports why a journal was not written by the agent this
// test built, or "" when it was.
//
// Every assertion about a capability record reads a claim about a producer.
// Which producer was, until agent_revision existed, an assumption: the test
// builds an agent, runs it, and decodes whatever journal is in the transport
// directory. Nothing in the journal said the two were the same, so a journal
// written by any other agent -- a stale build, a binary already on PATH, a
// directory reused between runs -- would decode cleanly and every assertion
// below it would describe a producer nobody was testing. That is the case
// agent_revision was added for, one level down from where it was added.
//
// A journal with no capability record is not a mismatch. A producer that fails
// before it can write one has nothing to be checked against, and the scenarios
// that cover that failure assert on the failure records instead.
func producerMismatch(c capture, want string) string {
	if c.capabilityJSON == nil {
		return ""
	}
	got := c.capability.AgentRevision
	if got == want {
		return ""
	}
	name := func(rev string) string {
		if rev == "" {
			return "no agent revision (an unstamped build)"
		}
		return "agent revision " + strconv.Quote(rev)
	}
	return fmt.Sprintf("journal reports %s, but this test built the agent with %s: every capability assertion against this journal would describe a producer that is not the one under test", name(got), name(want))
}

// controlBinary compiles the producer, the stub CUPTI, and the harness once
// per set of defines. A missing C toolchain skips: these controls qualify the
// producer, not the host. A toolchain that is present and rejects the source
// fails, because that is the producer's defect and skipping it would let a
// broken build read as a pass.
func controlBinary(t *testing.T, defines ...string) string {
	t.Helper()
	cc := os.Getenv("CC")
	if cc == "" {
		cc = "cc"
	}
	if _, err := exec.LookPath(cc); err != nil {
		t.Skipf("no C toolchain: %v", err)
	}
	key := cc + "\x00" + strings.Join(defines, "\x00")
	buildMu.Lock()
	defer buildMu.Unlock()
	if got, ok := buildCache[key]; ok {
		if got.err != nil {
			t.Fatalf("build producer control harness %v: %v", defines, got.err)
		}
		return got.path
	}
	dir, err := os.MkdirTemp("", "cudaprof-control")
	if err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(dir, "cupti-agent-control")
	const src = "../native"
	// -rdynamic exports the stub's symbols to dlsym(RTLD_DEFAULT). The
	// producer resolves two symbols that way: cuptiActivityGetStructSize,
	// which arrived in CUDA 13.2, and cuptiActivityRegisterTimestampCallback,
	// which an older libcupti does not export. In the field both live in a
	// shared object where RTLD_DEFAULT finds them unaided; here they are
	// linked into the executable, which on ELF is not in the dynamic symbol
	// table unless the link asks. Without this every probe returns NULL, the
	// degraded path is the only one any control runs, and the layout checks
	// are skipped entirely -- all while the suite passes. -Werror is that
	// bargain for the compiler; this is it for the loader.
	//
	// CUDAPROF_REQUIRE_STRUCT_SIZE_FN makes the producer refuse rather than
	// degrade when a build that links the symbol fails to resolve it, so the
	// flag cannot be dropped silently. Production still degrades: there an
	// unresolved symbol means an old runtime, which is a real case.
	args := []string{"-std=c11", "-Wall", "-Wextra", "-Werror", "-pedantic", "-rdynamic", "-I", filepath.Join(src, "testcupti")}
	// One build deliberately omits the symbol, to exercise the branch taken on
	// a CUPTI older than 13.2. Requiring resolution there would refuse the very
	// configuration that arm exists to cover.
	if !slices.Contains(defines, noStructSizeSymbol) {
		args = append(args, "-DCUDAPROF_REQUIRE_STRUCT_SIZE_FN=1")
	}
	// Shrinking the capability buffer is the only way to reach the
	// truncation branch, and the shrunk size is a compile-time constant, so
	// gcc proves the truncation statically and -Werror=format-truncation
	// refuses to build the very arm that exists to observe it at run time.
	// The build cannot produce the binary, the test reports a build failure,
	// and the branch stays uncovered on the one configuration written to
	// cover it.
	//
	// Demoted rather than silenced, and only for this arm: -Wno-error= leaves
	// the warning on stderr, so a truncation that appears in a build nobody
	// asked to truncate still says so.
	if slices.ContainsFunc(defines, func(d string) bool {
		return strings.HasPrefix(d, capabilityBufferDefine)
	}) {
		args = append(args, "-Wno-error=format-truncation")
	}
	args = append(args, defines...)
	args = append(args,
		filepath.Join(src, "cupti_agent.c"),
		filepath.Join(src, "testcupti", "stubcupti.c"),
		filepath.Join(src, "cupti_agent_control.c"),
		"-o", out, "-lpthread")
	b, err := exec.Command(cc, args...).CombinedOutput()
	if err != nil {
		err = errors.New(string(b))
		buildCache[key] = buildResult{err: err}
		t.Fatalf("build producer control harness %v: %v", defines, err)
	}
	buildCache[key] = buildResult{path: out}
	binRevisions[out] = agentRevisionOf(defines)
	return out
}

// run executes one scenario in its own transport directory and returns the
// harness totals, how the process ended, and every journal the run produced.
func run(t *testing.T, args ...string) (harnessResult, syscall.WaitStatus, []capture) {
	t.Helper()
	return runEnv(t, nil, args...)
}

// runEnv is run with extra producer configuration in the environment.
func runEnv(t *testing.T, env []string, args ...string) (harnessResult, syscall.WaitStatus, []capture) {
	t.Helper()
	return runBuilt(t, controlBinary(t), env, args...)
}

// runBuilt is runEnv against a specific harness build.
func runBuilt(t *testing.T, bin string, env []string, args ...string) (harnessResult, syscall.WaitStatus, []capture) {
	t.Helper()
	transport := t.TempDir()
	cmd := exec.Command(bin, args...)
	cmd.Env = append(os.Environ(), "CUDAPROF_TRANSPORT="+transport, "CUDAPROF_SESSION="+sessionHex)
	cmd.Env = append(cmd.Env, env...)
	out, err := cmd.Output()
	if err != nil {
		var exit *exec.ExitError
		// A scenario that ends on a signal is expected to; the caller checks.
		if !errors.As(err, &exit) {
			t.Fatalf("run %v: %v", args, err)
		}
	}
	status, _ := cmd.ProcessState.Sys().(syscall.WaitStatus)
	var result harnessResult
	if line := strings.TrimSpace(string(out)); line != "" {
		if err := json.Unmarshal([]byte(line), &result); err != nil {
			t.Fatalf("harness output %q: %v", line, err)
		}
	}
	return result, status, decodeAll(t, transport, binRevision(bin))
}

// decodeAll decodes every journal in transport, checking each against
// wantRevision -- the agent revision the build under test stamps, which
// binRevision reports for a harness controlBinary built.
func decodeAll(t *testing.T, transport, wantRevision string) []capture {
	t.Helper()
	dirs, err := filepath.Glob(filepath.Join(transport, "process-*"))
	if err != nil {
		t.Fatal(err)
	}
	var out []capture
	for _, dir := range dirs {
		out = append(out, decode(t, filepath.Join(dir, "capture.journal"), wantRevision))
	}
	return out
}

func decode(t *testing.T, path, wantRevision string) capture {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open journal: %v", err)
	}
	defer f.Close()
	var session [16]byte
	raw, err := hex.DecodeString(sessionHex)
	if err != nil {
		t.Fatal(err)
	}
	copy(session[:], raw)

	// The producer emits far more than the package default; these controls
	// bound the journal by scenario size, not by the collector's policy.
	d := wire.NewDecoder(f, wire.Limits{MaxPayload: 4096, MaxRecords: 1 << 22})
	d.ExpectSession(session)
	c := capture{counts: map[uint16]uint64{}}
	for {
		r, err := d.Next()
		if errors.Is(err, io.EOF) {
			break
		}
		if errors.Is(err, io.ErrUnexpectedEOF) {
			// A forced kill can tear the terminal frame. Retaining the valid
			// prefix is the decoder's contract; record that it happened.
			c.truncated = true
			break
		}
		if err != nil {
			t.Fatalf("decode %s after %d records: %v", path, d.Stats().Records, err)
		}
		c.counts[r.Kind]++
		c.order = append(c.order, r.Kind)
		switch r.Kind {
		case kindFinalized:
			var fin finalizedRecord
			if err := json.Unmarshal(r.Payload, &fin); err != nil {
				t.Fatalf("finalized payload %q: %v", r.Payload, err)
			}
			c.finalized = &fin
			c.finalizedAt = r.Timestamp
		case kindCapability:
			if err := json.Unmarshal(r.Payload, &c.capability); err != nil {
				t.Fatalf("capability payload %q: %v", r.Payload, err)
			}
			// Kept raw as well: a field being absent is not the same as its
			// decoding to a zero value, and only the bytes can tell them apart.
			c.capabilityJSON = append([]byte(nil), r.Payload...)
		case kindFailure:
			c.failures = append(c.failures, string(r.Payload))
		case kindModule:
			var m moduleRecord
			if err := json.Unmarshal(r.Payload, &m); err != nil {
				t.Fatalf("module payload %q: %v", r.Payload, err)
			}
			c.modules = append(c.modules, m)
		case kindAPI:
			var e struct {
				Kind        string
				Domain      string
				Event       string `json:"lifecycle_event"`
				Graph       uint64 `json:"graph_handle"`
				Executable  uint64 `json:"executable_handle"`
				Correlation uint32
				Start, End  int64
				Success     *bool
			}
			if err := json.Unmarshal(r.Payload, &e); err != nil {
				t.Fatal(err)
			}
			if e.Kind == "graph_lifecycle" {
				c.graphLifecycle = append(c.graphLifecycle, timeline.GraphEvent{Journal: path, Domain: e.Domain, Kind: e.Event, Graph: e.Graph, Executable: e.Executable, Correlation: e.Correlation, Start: e.Start, End: e.End, Success: e.Success})
			}
		case kindHost:
			var h hostSampleRecord
			if err := json.Unmarshal(r.Payload, &h); err != nil {
				t.Fatalf("host sample payload %q: %v", r.Payload, err)
			}
			c.hosts = append(c.hosts, h)
		case kindActivity:
			// Only markers are decoded here. The high-volume scenarios offer
			// millions of activity records and a JSON unmarshal per record
			// would make the decode, not the producer, the thing under test.
			if bytes.Contains(r.Payload, []byte(`"subtype":"device_active"`)) {
				c.deviceActive++
			}
			if bytes.Contains(r.Payload, []byte(`"subtype":"capture_range"`)) {
				var e rangeEdgeRecord
				if err := json.Unmarshal(r.Payload, &e); err != nil {
					t.Fatalf("capture_range payload %q: %v", r.Payload, err)
				}
				e.Timestamp = r.Timestamp
				c.ranges = append(c.ranges, e)
			}
			if bytes.Contains(r.Payload, []byte(`"subtype":"marker"`)) {
				var m markerRecord
				if err := json.Unmarshal(r.Payload, &m); err != nil {
					t.Fatalf("marker payload %q: %v", r.Payload, err)
				}
				c.markers = append(c.markers, m)
			}
			if bytes.Contains(r.Payload, []byte(`"subtype":"memset"`)) {
				var m memsetRecord
				if err := json.Unmarshal(r.Payload, &m); err != nil {
					t.Fatalf("memset payload %q: %v", r.Payload, err)
				}
				c.memsets = append(c.memsets, m)
			}
			if bytes.Contains(r.Payload, []byte(`"subtype":"overhead"`)) {
				var o overheadRecord
				if err := json.Unmarshal(r.Payload, &o); err != nil {
					t.Fatalf("overhead payload %q: %v", r.Payload, err)
				}
				c.overheads = append(c.overheads, o)
			}
			// Kernel records are decoded only when they name a graph, for
			// the same reason markers are decoded alone.
			if bytes.Contains(r.Payload, []byte(`"subtype":"kernel"`)) && !bytes.Contains(r.Payload, []byte(`"graph_id":0,`)) {
				var k graphNodeKernel
				if err := json.Unmarshal(r.Payload, &k); err != nil {
					t.Fatalf("kernel payload %q: %v", r.Payload, err)
				}
				c.graphNodes = append(c.graphNodes, k)
			}
			if bytes.Contains(r.Payload, []byte(`"name":"cudaprof_module_kernel"`)) {
				var k moduleKernel
				if err := json.Unmarshal(r.Payload, &k); err != nil {
					t.Fatalf("kernel payload %q: %v", r.Payload, err)
				}
				c.moduleKernels = append(c.moduleKernels, k)
			}
			if bytes.Contains(r.Payload, []byte(`"subtype":"graph_trace"`)) {
				var g graphTraceRecord
				if err := json.Unmarshal(r.Payload, &g); err != nil {
					t.Fatalf("graph_trace payload %q: %v", r.Payload, err)
				}
				c.graphs = append(c.graphs, g)
			}
			if bytes.Contains(r.Payload, []byte(`"subtype":"sync"`)) {
				var y syncRecord
				if err := json.Unmarshal(r.Payload, &y); err != nil {
					t.Fatalf("sync payload %q: %v", r.Payload, err)
				}
				c.syncs = append(c.syncs, y)
			}
		}
	}
	if why := producerMismatch(c, wantRevision); why != "" {
		t.Fatalf("decode %s: %s", path, why)
	}
	return c
}

// only returns the single journal a one-process scenario must produce.
func only(t *testing.T, captures []capture) capture {
	t.Helper()
	if len(captures) != 1 {
		t.Fatalf("got %d journals, want 1", len(captures))
	}
	return captures[0]
}

// checkReconciles asserts the producer's own accounting closes: every record
// offered to the ring was either drained to the journal or dropped into
// exactly one named category, and the journal holds what the drain claims.
func checkReconciles(t *testing.T, c capture, offered uint64) {
	t.Helper()
	checkReconcilesOutOfBand(t, c, offered, nil)
}

// checkReconcilesOutOfBand is checkReconciles where outOfBand records, counted
// by wire kind, reached the journal without passing through the ring and so
// cannot appear in the drain's retained count. The constructor writes such a
// failure record directly when it fails before the drain thread exists, and a
// capture-range edge is written the same way.
func checkReconcilesOutOfBand(t *testing.T, c capture, offered uint64, outOfBand map[uint16]uint64) {
	t.Helper()
	fin := c.finalized
	if fin == nil {
		t.Fatal("no finalized record")
	}
	if fin.Offered != offered {
		t.Errorf("producer counted %d offered, harness offered %d", fin.Offered, offered)
	}
	drops := fin.QueueOversizeDropped + fin.QueueContentionDrop + fin.QueueCapacityDropped
	if got := fin.Retained + drops; got != fin.Offered {
		t.Errorf("retained %d + dropped %d = %d, want offered %d",
			fin.Retained, drops, got, fin.Offered)
	}
	if fin.QueueDropped != drops {
		t.Errorf("queue_dropped %d != sum of categories %d", fin.QueueDropped, drops)
	}
	// Every drained record reached the journal, alongside the bootstrap,
	// capability and finalized records written outside the ring.
	var queued uint64
	for kind, n := range c.counts {
		switch kind {
		case kindActivity, kindAPI, kindExternal, kindHost:
			queued += n
		case kindFailure:
			queued += n
		}
	}
	outOfBandTotal := uint64(c.deviceActive)
	for _, n := range outOfBand {
		outOfBandTotal += n
	}
	if queued != fin.Retained+outOfBandTotal {
		t.Errorf("journal holds %d records written outside the fixed three, drain claims %d plus %d out of band", queued, fin.Retained, outOfBandTotal)
	}
	checkPerKindReconciles(t, c, drops, outOfBand)
	if c.counts[kindBootstrap] != 1 || c.counts[kindCapability] != 1 || c.counts[kindFinalized] != 1 {
		t.Errorf("bootstrap=%d capability=%d finalized=%d, want 1 each",
			c.counts[kindBootstrap], c.counts[kindCapability], c.counts[kindFinalized])
	}
}

// checkPerKindReconciles asserts the per-kind split closes against the same
// totals: every kind's offered records were retained or dropped, and summing
// the kinds returns the aggregate. drops is the aggregate queue-drop count,
// which is what the per-kind dropped counters must sum to -- native_dropped is
// CUPTI's own loss and never reaches admission, so it is not in this identity.
func checkPerKindReconciles(t *testing.T, c capture, drops uint64, outOfBand map[uint16]uint64) {
	t.Helper()
	fin := c.finalized
	if fin.ByKind == nil {
		t.Fatal("finalized record carries no by_kind accounting")
	}
	for _, kind := range []string{"activity", "api", "external", "failure", "host"} {
		if _, ok := fin.ByKind[kind]; !ok {
			t.Errorf("by_kind is missing %q", kind)
		}
	}
	var offered, retained, dropped uint64
	for kind, k := range fin.ByKind {
		if k.Retained+k.Dropped != k.Offered {
			t.Errorf("by_kind[%s]: retained %d + dropped %d != offered %d", kind, k.Retained, k.Dropped, k.Offered)
		}
		offered += k.Offered
		retained += k.Retained
		dropped += k.Dropped
	}
	if offered != fin.Offered {
		t.Errorf("by_kind offered sums to %d, aggregate offered %d", offered, fin.Offered)
	}
	if retained != fin.Retained {
		t.Errorf("by_kind retained sums to %d, aggregate retained %d", retained, fin.Retained)
	}
	if dropped != drops {
		t.Errorf("by_kind dropped sums to %d, aggregate queue drops %d", dropped, drops)
	}
	// activity is also counted independently on the drain side, from the
	// record the drain is about to write rather than from admission. The two
	// must agree or one of them is counting the wrong population -- which is
	// exactly the mistake the frozen SGLang loss figure made.
	if got := fin.ByKind["activity"].Retained; got != fin.Activity {
		t.Errorf("by_kind activity retained %d, drain-side activity count %d", got, fin.Activity)
	}
	// The counters must carry the record's own kind, not just sum to the
	// aggregate: each kind's retained count has to equal the number of
	// records of that kind actually in the journal. A kind-agnostic counter
	// passes the sums above and fails here.
	journal := map[string]uint64{
		"activity": c.counts[kindActivity] - outOfBand[kindActivity] - uint64(c.deviceActive),
		"api":      c.counts[kindAPI] - outOfBand[kindAPI],
		"external": c.counts[kindExternal] - outOfBand[kindExternal],
		"failure":  c.counts[kindFailure] - outOfBand[kindFailure],
		"host":     c.counts[kindHost] - outOfBand[kindHost],
	}
	for kind, want := range journal {
		if got := fin.ByKind[kind].Retained; got != want {
			t.Errorf("by_kind[%s] retained %d, journal holds %d records of that kind", kind, got, want)
		}
	}
}

// TestSustained holds the offered rate below the drain's capacity. Nothing may
// be lost and no record may need backpressure.
func TestSustained(t *testing.T) {
	result, status, captures := run(t, "sustained", "1", "16", "512", "200")
	c := only(t, captures)
	checkReconciles(t, c, result.Offered)
	fin := c.finalized
	if fin.QueueCapacityDropped != 0 || fin.QueueContentionDrop != 0 || fin.QueueOversizeDropped != 0 {
		t.Errorf("sustained offer lost records: %+v", fin)
	}
	if fin.State != "collected" {
		t.Errorf("state = %q, want collected", fin.State)
	}
	if fin.Trigger != "exit" {
		t.Errorf("trigger = %q, want exit", fin.Trigger)
	}
	if fin.DrainBatches == 0 || fin.DrainMaxBatch < 2 {
		t.Errorf("drain did not batch: batches=%d max=%d", fin.DrainBatches, fin.DrainMaxBatch)
	}
	if !status.Exited() || status.ExitStatus() != 0 {
		t.Errorf("status = %v, want a clean exit", status)
	}
}

// TestBurst offers whole activity buffers back to back from several threads,
// which is the shape CUPTI delivers under a real serving load. The gate is
// loadAverage reports the host's one-minute load average, or "unknown" if it
// cannot be read. The admission deadline below is wall-clock, so the same code
// retains every record on an idle host and drops records on a busy one. Without
// the load, a pass taken under contention and a pass taken on a quiet machine
// are the same line in a log, and only one of them is evidence.
//
// The standard library exposes no load average and x/sys is outside this
// repository's dependencies, so read what each kernel already publishes as
// text. Never report an empty string: it reads as a quiet host, which is the
// one misreading this exists to prevent.
func loadAverage() string {
	const unknown = "unknown"
	switch runtime.GOOS {
	case "darwin":
		// sysctl -n vm.loadavg prints "{ 1.23 4.56 7.89 }".
		out, err := exec.Command("sysctl", "-n", "vm.loadavg").Output()
		if err != nil {
			return unknown
		}
		if f := strings.Fields(string(out)); len(f) > 1 {
			return f[1]
		}
	case "linux":
		// /proc/loadavg starts "1.23 4.56 7.89".
		out, err := os.ReadFile("/proc/loadavg")
		if err != nil {
			return unknown
		}
		if f := strings.Fields(string(out)); len(f) > 0 {
			return f[0]
		}
	}
	return unknown
}

// checkRetention asserts what the admission path actually promises about a run
// that deliberately overfills the ring.
//
// Not zero loss. A record is charged to queue_capacity_dropped when admission
// reaches its absolute deadline (ADMIT_WAIT_NANOS, cupti_agent.c) without the
// drain publishing a freed slot, and the drain publishes only once its batch
// write returns -- so that deadline is a bound on journal write latency, and
// the agent's own comment puts the longest observed write at 99 ms against a
// 100 ms deadline under concurrent host I/O. Asserting Retained == Offered
// makes every run one Bernoulli draw with a millisecond of slack, which no
// number of green runs can close: a passing burst says the host was quiet, not
// that the code is right.
//
// The mechanism is strictly stronger. Equality fails identically whether the
// drain missed the deadline -- the designed path -- or stayed well inside it,
// which would mean a freed slot existed and was not published, a real defect.
// These four clauses separate those, and the last supplies the half no green
// run can: a drain that never approached the deadline must have lost nothing.
func checkRetention(t *testing.T, c capture, what string) {
	t.Helper()
	fin := c.finalized
	deadline := c.capability.AdmitWaitNanos
	if deadline <= 0 {
		t.Fatalf("capability reports admit_wait_nanos=%d; without the deadline a designed drop cannot be told from a defect", deadline)
	}
	lost := fin.Offered - fin.Retained
	state := fmt.Sprintf("lost=%d of %d capacity=%d contention=%d oversize=%d longest_wait=%dms longest_write=%dms deadline=%dms load1=%s",
		lost, fin.Offered,
		fin.QueueCapacityDropped, fin.QueueContentionDrop, fin.QueueOversizeDropped,
		fin.BackpressureMaxNanos/1e6, fin.DrainWriteMaxNanos/1e6, deadline/1e6, loadAverage())
	switch {
	case lost == 0:
		if fin.DrainWriteMaxNanos >= uint64(deadline) {
			// Not a failure, but worth printing: the run was close enough to
			// the boundary that the next one may cross it.
			t.Logf("%s retained everything with the drain past the deadline: %s", what, state)
		}
	case fin.QueueCapacityDropped != lost:
		// Capacity is the only drop backpressure may lose to. Contention and
		// oversize are defects wherever they appear.
		t.Errorf("%s lost records the admission deadline does not account for: %s", what, state)
	case fin.DrainWriteMaxNanos < uint64(deadline) && fin.BackpressureMaxNanos < uint64(deadline):
		t.Errorf("%s lost records to capacity with neither the drain nor admission reaching the deadline: %s", what, state)
	default:
		t.Logf("%s lost records to the documented capacity path: %s", what, state)
	}
}

// TestBurst offers far more records than the ring holds and checks that the
// batched drain plus bounded backpressure retains what it promised to. See
// checkRetention for why that is not the same as retaining everything.
func TestBurst(t *testing.T) {
	result, status, captures := run(t, "burst", "4", "8", "20000", "0")
	c := only(t, captures)
	checkReconciles(t, c, result.Offered)
	fin := c.finalized
	if result.Offered < 100000 {
		t.Fatalf("burst offered only %d records; the control is not applying pressure", result.Offered)
	}
	checkRetention(t, c, "burst")
	if fin.DrainMaxBatch < 2 {
		t.Errorf("drain wrote one record per batch under burst: %+v", fin)
	}
	if len(c.failures) != 0 {
		t.Errorf("producer reported failures: %q", c.failures)
	}
	if !status.Exited() || status.ExitStatus() != 0 {
		t.Errorf("status = %v, want a clean exit", status)
	}
	t.Logf("burst offered=%d retained=%d batches=%d max_batch=%d backpressure_waits=%d backpressure_ms=%d longest_wait_ms=%d longest_write_ms=%d callback_max_us=%d load1=%s",
		fin.Offered, fin.Retained, fin.DrainBatches, fin.DrainMaxBatch,
		fin.BackpressureWaits, fin.BackpressureNanos/1e6,
		fin.BackpressureMaxNanos/1e6, fin.DrainWriteMaxNanos/1e6,
		fin.CallbackMaxNanos/1e3, loadAverage())
}

// TestTerminationRace signals SIGTERM while callbacks are still offering. The
// producer must finalize from a normal context, attribute the trigger, and
// still reconcile.
func TestTerminationRace(t *testing.T) {
	result, status, captures := run(t, "race", "4", "8", "20000", "0", "20000")
	c := only(t, captures)
	fin := c.finalized
	if fin == nil {
		t.Fatal("SIGTERM produced no finalized record")
	}
	if fin.Trigger != "signal" {
		t.Errorf("trigger = %q, want signal", fin.Trigger)
	}
	if c.counts[kindFinalized] != 1 {
		t.Errorf("finalized records = %d, want exactly 1", c.counts[kindFinalized])
	}
	if c.truncated {
		t.Error("journal was truncated despite an orderly signal finalization")
	}
	// Records offered after finalization began are legitimately refused, so
	// the producer's offered count is the authority here, not the harness.
	if fin.Offered > result.Offered {
		t.Errorf("producer counted %d offered, harness offered at most %d", fin.Offered, result.Offered)
	}
	drops := fin.QueueOversizeDropped + fin.QueueContentionDrop + fin.QueueCapacityDropped
	if got := fin.Retained + drops; got != fin.Offered {
		t.Errorf("retained %d + dropped %d = %d, want offered %d", fin.Retained, drops, got, fin.Offered)
	}
	// Finalizing must not change how the target terminates: the producer
	// restores the default disposition and re-raises once the journal is
	// closed, so the process still dies of SIGTERM.
	if status.Signaled() != true || status.Signal() != syscall.SIGTERM {
		t.Errorf("status = %v, want termination by SIGTERM", status)
	}
}

// TestForcedKillIsNotFinalized is the negative control. SIGKILL cannot be
// handled, so the journal must end without a finalized record. This is the
// case a signal handler can never rescue, and the one Dynamo's worker
// teardown takes.
func TestForcedKillIsNotFinalized(t *testing.T) {
	_, status, captures := run(t, "forced-kill", "4", "8", "20000", "0", "20000")
	c := only(t, captures)
	if c.finalized != nil {
		t.Fatalf("SIGKILL produced a finalized record: %+v", c.finalized)
	}
	if c.counts[kindActivity] == 0 {
		t.Error("no activity reached the journal before the kill; the control proves nothing")
	}
	if c.counts[kindBootstrap] != 1 {
		t.Errorf("bootstrap records = %d, want 1", c.counts[kindBootstrap])
	}
	if !status.Signaled() || status.Signal() != syscall.SIGKILL {
		t.Errorf("status = %v, want termination by SIGKILL", status)
	}
}

// TestHandlerClobberStillFinalizes covers the target that installs its own
// SIGTERM handler after the producer's constructor has run, replacing it.
// CPython does this, so it is the shape every Python serving target takes.
// The producer's signal path is gone, but the guarantee that survives is that
// a target winding down normally still finalizes from atexit.
func TestHandlerClobberStillFinalizes(t *testing.T) {
	result, status, captures := run(t, "clobber", "4", "4", "20000", "0", "20000")
	if result.AppTerminated != 1 {
		t.Fatalf("target never saw SIGTERM: %+v", result)
	}
	if !status.Exited() || status.ExitStatus() != 7 {
		t.Errorf("status = %v, want exit 7: the producer must not change how the target terminates", status)
	}
	c := only(t, captures)
	if c.finalized == nil {
		t.Fatal("no finalized record: clobbering the handler must not cost the atexit path")
	}
	if c.finalized.Trigger != "exit" {
		t.Errorf("trigger = %q, want exit: the producer's handler was replaced, so it cannot be the trigger", c.finalized.Trigger)
	}
	checkReconciles(t, c, result.Offered)
}

// TestRuntimeExitLosesTheJournalWithoutAnExplicitFinalize reproduces, without
// a GPU, the failure a Go capture target hits on real hardware.
//
// All three of the producer's finalization routes run through libc. The ELF
// destructor and atexit are driven by exit(3); the SIGTERM terminator needs a
// signal nobody sends, and a runtime that installs its own SIGTERM disposition
// after this agent's constructor has run has taken that one away too. A
// runtime leaving through the exit_group syscall reaches none of them, and
// because CUPTI's buffers are flushed at finalize they die with the process:
// the target exits 0 and reports its own work done while the capture holds
// nothing.
//
// Measured first on GB10, where a Go target collected zero records and a C++
// target doing the same device work under the same agent and the same cudaprof
// collected cleanly. This is that dissociation in the control harness, and it
// is the oracle for the fix rather than a test written after one: the first
// arm asserts the loss, so deleting cudaprof_finalize brings it back.
func TestRuntimeExitLosesTheJournalWithoutAnExplicitFinalize(t *testing.T) {
	_, status, captures := run(t, "runtime-exit", "4", "4", "256", "0")
	if !status.Exited() || status.ExitStatus() != 0 {
		t.Fatalf("status = %v, want a clean exit: the target believes it succeeded", status)
	}
	c := only(t, captures)
	if c.finalized != nil {
		t.Fatalf("journal was finalized: the reproduction no longer reproduces, so the arm below proves nothing")
	}

	_, status, captures = run(t, "runtime-exit-finalize", "4", "4", "256", "0")
	if !status.Exited() || status.ExitStatus() != 0 {
		t.Fatalf("status = %v, want a clean exit: finalizing must not change how the target terminates", status)
	}
	c = only(t, captures)
	if c.finalized == nil {
		t.Fatal("no finalized record after an explicit cudaprof_finalize: the one route a runtime can take is not working")
	}
	// The trigger names the route, and this one is the host's own call rather
	// than any of the three libc paths the runtime skipped.
	if c.finalized.Trigger != "host" {
		t.Errorf("trigger = %q, want host", c.finalized.Trigger)
	}
	if c.counts[kindActivity] == 0 {
		t.Error("finalized with no activity records: finalization that flushes nothing is the same loss with a record on top")
	}
}

// TestFinalizeOnExitGroup covers CUDAPROF_FINALIZE=exit-group, the route that
// needs nothing from a target leaving through exit_group: the producer holds
// the syscall, finalizes, and lets it continue. The scenario is the one the
// test above shows losing the journal, so a finalized record here is the
// producer's doing.
// requireExitGroup skips a test of exit_group finalization, which needs a
// seccomp user-notification listener and so exists only on linux.
// TestFinalizeOnExitGroupUnavailable covers the other platforms.
func requireExitGroup(t *testing.T) {
	t.Helper()
	if runtime.GOOS != "linux" {
		t.Skipf("exit_group finalization is linux-only; GOOS=%s", runtime.GOOS)
	}
}

// TestFinalizeOnExitGroupUnavailable covers a platform without seccomp: the
// agent must say it could not honor the request rather than claim the mode.
func TestFinalizeOnExitGroupUnavailable(t *testing.T) {
	if runtime.GOOS == "linux" {
		t.Skip("exit_group finalization is available on linux")
	}
	env := []string{"CUDAPROF_FINALIZE=exit-group"}
	_, _, captures := runEnv(t, env, "runtime-exit", "4", "4", "256", "0")
	c := only(t, captures)
	if c.capability.Finalize != "requested_unavailable" {
		t.Errorf("finalize = %q, want requested_unavailable", c.capability.Finalize)
	}
	found := false
	for _, f := range c.failures {
		if strings.Contains(f, "exit-group finalize") {
			found = true
		}
	}
	if !found {
		t.Errorf("failures = %q, want one naming the exit-group finalize", c.failures)
	}
}

func TestFinalizeOnExitGroup(t *testing.T) {
	requireExitGroup(t)
	env := []string{"CUDAPROF_FINALIZE=exit-group"}
	_, status, captures := runEnv(t, env, "runtime-exit", "4", "4", "256", "0")
	if !status.Exited() || status.ExitStatus() != 0 {
		t.Fatalf("status = %v, want a clean exit: holding exit_group must not change how the target terminates", status)
	}
	c := only(t, captures)
	if c.capability.Finalize != "exit-group" {
		t.Fatalf("finalize = %q, want exit-group; failures: %v", c.capability.Finalize, c.failures)
	}
	if c.finalized == nil {
		t.Fatal("no finalized record after exit_group")
	}
	if c.finalized.Trigger != "exit_group" {
		t.Errorf("trigger = %q, want exit_group", c.finalized.Trigger)
	}
	if c.counts[kindActivity] == 0 {
		t.Error("finalized with no activity records")
	}
}

// TestFinalizeOnExitGroupWithChild covers a forked child, which inherits the
// seccomp filter. Its exit_group reaches the parent's listener and must be
// continued with the child's own status, without finalizing the parent's
// journal or hanging; the parent's later exit_group must still finalize and
// keep the parent's status.
func TestFinalizeOnExitGroupWithChild(t *testing.T) {
	requireExitGroup(t)
	env := []string{"CUDAPROF_FINALIZE=exit-group"}
	result, status, captures := runEnv(t, env, "runtime-exit-fork", "1", "8", "512", "0")
	if result.ChildExited != 1 {
		t.Fatal("forked child did not exit: its exit_group was held and never continued")
	}
	if ws := syscall.WaitStatus(result.ChildStatus); !ws.Exited() || ws.ExitStatus() != 5 {
		t.Errorf("child status = %v, want exit 5", ws)
	}
	if !status.Exited() || status.ExitStatus() != 3 {
		t.Fatalf("status = %v, want exit 3", status)
	}
	c := only(t, captures)
	if c.finalized == nil {
		t.Fatal("no finalized record after the parent's exit_group")
	}
	if c.finalized.Trigger != "exit_group" {
		t.Errorf("trigger = %q, want exit_group: the child's exit finalized the parent", c.finalized.Trigger)
	}
	checkReconciles(t, c, result.Offered)
}

// TestFinalizeOnExitGroupWithAgentChild covers a descendant that execs and
// loads an agent of its own. The kernel allows one listener per filter chain,
// so the descendant cannot hold its own exit_group; the parent's watcher must
// finalize it through the rendezvous before letting it exit.
func TestFinalizeOnExitGroupWithAgentChild(t *testing.T) {
	requireExitGroup(t)
	env := []string{"CUDAPROF_FINALIZE=exit-group"}
	result, status, captures := runEnv(t, env, "runtime-exit-exec", "1", "4", "256", "0")
	if result.ChildExited != 1 {
		t.Fatal("exec'd child did not exit")
	}
	if ws := syscall.WaitStatus(result.ChildStatus); !ws.Exited() || ws.ExitStatus() != 0 {
		t.Errorf("child status = %v, want exit 0", ws)
	}
	if !status.Exited() || status.ExitStatus() != 3 {
		t.Fatalf("status = %v, want exit 3", status)
	}
	if len(captures) != 2 {
		t.Fatalf("got %d journals, want the parent's and the child's", len(captures))
	}
	for i, c := range captures {
		if c.capability.Finalize != "exit-group" {
			t.Errorf("journal %d: finalize = %q, want exit-group; failures: %v", i, c.capability.Finalize, c.failures)
		}
		if c.finalized == nil {
			t.Errorf("journal %d: no finalized record", i)
			continue
		}
		if c.finalized.Trigger != "exit_group" {
			t.Errorf("journal %d: trigger = %q, want exit_group", i, c.finalized.Trigger)
		}
		if c.counts[kindActivity] == 0 {
			t.Errorf("journal %d: no activity records", i)
		}
	}
}

// TestActivityOnlyConfiguration covers the second supported configuration.
// API attribution dominates journal volume, so a run may disable it; that has
// to be an explicit, recorded choice rather than a silent degradation, and it
// must not weaken the retention guarantee.
func TestActivityOnlyConfiguration(t *testing.T) {
	env := []string{"CUDAPROF_API_ACTIVITY=0"}
	result, status, captures := runEnv(t, env, "burst", "4", "8", "20000", "0")
	c := only(t, captures)
	if !status.Exited() || status.ExitStatus() != 0 {
		t.Errorf("status = %v, want a clean exit", status)
	}
	if c.capability.APIRecords || c.capability.DriverAPI || c.capability.RuntimeAPI {
		t.Errorf("capability still claims API attribution: %+v", c.capability)
	}
	if !c.capability.Activity {
		t.Error("capability dropped activity attribution")
	}
	if c.counts[kindAPI] != 0 {
		t.Errorf("activity-only run retained %d API records", c.counts[kindAPI])
	}
	if c.counts[kindActivity] == 0 {
		t.Error("activity-only run retained no activity")
	}
	checkReconciles(t, c, result.Offered)
	checkRetention(t, c, "activity-only run")
}

// noStructSizeSymbol builds the stub without cuptiActivityGetStructSize, so
// the producer's dlsym finds nothing and takes the branch a CUPTI older than
// CUDA 13.2 takes.
const noStructSizeSymbol = "-DCUDAPROF_TEST_NO_STRUCT_SIZE_SYMBOL=1"

// capabilityBufferDefine prefixes the define that resizes the capability
// buffer. See controlBinary for why a build carrying it relaxes one -Werror.
const capabilityBufferDefine = "-DCUDAPROF_CAPABILITY_BUFFER="

// TestCapabilityDistinguishesUnverifiedFromVerified is the other arm of the
// layout_check control, and the reason the field is worth carrying.
//
// Every other scenario links a stub that exports cuptiActivityGetStructSize,
// so the probe always resolves and always reports the compiled sizes: the
// "verified" branch is the only one they reach, and a capability record that
// simply asserted the compiled layout names would look identical whether the
// runtime had been checked or not. Without this arm the two branches have
// never been discriminated against each other and the field claims something
// no test has shown.
//
// This arm also pins the deliberate choice not to fail closed. A runtime whose
// layout cannot be verified still collects: the decode path bounds-checks each
// record against the buffer before reading it, and refusing the kind would
// lose memsets on every pre-13.2 runtime to protect against one that reports a
// short record. So memsets stay on, and the record says the layout was not
// checked.
func TestCapabilityDistinguishesUnverifiedFromVerified(t *testing.T) {
	verified, _, verifiedCaptures := run(t, "sustained", "1", "4", "256", "100")
	_ = verified
	withProbe := only(t, verifiedCaptures)

	bin := controlBinary(t, noStructSizeSymbol)
	dir := t.TempDir()
	cmd := exec.Command(bin, "sustained", "1", "4", "256", "100")
	cmd.Env = append(os.Environ(), "CUDAPROF_TRANSPORT="+dir, "CUDAPROF_SESSION="+sessionHex, "CUDAPROF_MEMSETS=1")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("run without the struct-size symbol: %v: %s", err, out)
	}
	withoutProbe := only(t, decodeAll(t, dir, binRevision(bin)))

	if withProbe.capability.LayoutCheck != "cuptiActivityGetStructSize" {
		t.Errorf("with the symbol: layout_check = %q, want %q", withProbe.capability.LayoutCheck, "cuptiActivityGetStructSize")
	}
	if withoutProbe.capability.LayoutCheck != "unavailable" {
		t.Errorf("without the symbol: layout_check = %q, want %q", withoutProbe.capability.LayoutCheck, "unavailable")
	}
	if withProbe.capability.LayoutCheck == withoutProbe.capability.LayoutCheck {
		t.Fatal("both builds report the same layout_check: the field does not discriminate, which is the defect it was added to fix")
	}
	if !withoutProbe.capability.Memsets {
		t.Error("memsets = false without the probe: an unverifiable layout degrades to unverified collection, it does not disable the kind")
	}
	if len(withoutProbe.failures) != 0 {
		t.Errorf("failures = %q, want none: a CUPTI without the symbol is a supported runtime, not a failure", withoutProbe.failures)
	}
}

// agentRevisionDefine builds the agent with a known source revision, the way
// the build recipe in native/README.md does.
const agentRevisionDefine = `-DCUDAPROF_AGENT_REVISION="c0ffee1"`

// TestCapabilityCarriesTheAgentRevision covers the identity half of the
// capability record.
//
// The record states what the producer can do. Until agent_revision it did not
// state which producer, and a reader holding only the artifact could not tell
// a current agent reporting that a runtime lacks a capability from an agent
// built before that capability existed: both say the same thing in the same
// words. The case that prompted it was a verification about to be run against
// a preinstalled binary older than the change it was verifying.
//
// Both arms are needed. Asserting only that a stamped build carries the
// revision would pass on a producer that stamped every build, including one
// whose identity the build never supplied -- and a wrong revision is worse
// than none, because a reader will act on it. So the unstamped arm asserts the
// field is ABSENT rather than "unknown" or empty-but-present.
func TestCapabilityCarriesTheAgentRevision(t *testing.T) {
	_, _, unstamped := run(t, "sustained", "1", "4", "256", "100")
	plain := only(t, unstamped)
	if plain.capability.AgentRevision != "" {
		t.Errorf("unstamped build: agent_revision = %q, want absent: a build that was not told its revision must not name one", plain.capability.AgentRevision)
	}
	if bytes.Contains(plain.capabilityJSON, []byte("agent_revision")) {
		t.Errorf("unstamped build emitted an agent_revision key: %s\n\tan empty value present in the record reads as an answer; absence does not", plain.capabilityJSON)
	}

	bin := controlBinary(t, agentRevisionDefine)
	dir := t.TempDir()
	cmd := exec.Command(bin, "sustained", "1", "4", "256", "100")
	cmd.Env = append(os.Environ(), "CUDAPROF_TRANSPORT="+dir, "CUDAPROF_SESSION="+sessionHex)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("run with a stamped revision: %v: %s", err, out)
	}
	stamped := only(t, decodeAll(t, dir, binRevision(bin)))
	if stamped.capability.AgentRevision != "c0ffee1" {
		t.Errorf("stamped build: agent_revision = %q, want %q", stamped.capability.AgentRevision, "c0ffee1")
	}
	if stamped.capability.AgentRevision == plain.capability.AgentRevision {
		t.Fatal("both builds report the same agent_revision: the field does not discriminate, which is the defect it was added to fix")
	}
}

// TestDecodePinsEveryJournalToTheAgentUnderTest covers the check every other
// capability assertion in this file now rests on.
//
// The assertions read a capability record and conclude something about the
// producer. Which producer was an assumption -- the test builds an agent, runs
// it, and decodes whatever journal appeared in the transport directory -- and
// an assumption that held only because nothing in this suite had yet run two
// different agents. TestCapabilityCarriesTheAgentRevision runs two, and a
// journal from the wrong one decodes exactly as cleanly as a journal from the
// right one. agent_revision is what makes the assumption checkable, so the
// decode checks it.
//
// The mismatch arm is the one worth having. A check that only accepted
// matching revisions would be satisfied by a producerMismatch that returned ""
// for everything, which is the fail-open shape: it would report every journal
// as the agent under test, including the ones that are not.
func TestDecodePinsEveryJournalToTheAgentUnderTest(t *testing.T) {
	_, _, plainCaptures := run(t, "sustained", "1", "4", "256", "100")
	plain := only(t, plainCaptures)

	bin := controlBinary(t, agentRevisionDefine)
	if got := binRevision(bin); got != "c0ffee1" {
		t.Fatalf("binRevision = %q, want %q: the build cache lost the identity it built with", got, "c0ffee1")
	}
	dir := t.TempDir()
	cmd := exec.Command(bin, "sustained", "1", "4", "256", "100")
	cmd.Env = append(os.Environ(), "CUDAPROF_TRANSPORT="+dir, "CUDAPROF_SESSION="+sessionHex)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("run with a stamped revision: %v: %s", err, out)
	}
	stamped := only(t, decodeAll(t, dir, "c0ffee1"))

	// Each journal against the agent that wrote it.
	if why := producerMismatch(plain, ""); why != "" {
		t.Errorf("unstamped journal rejected by its own build: %s", why)
	}
	if why := producerMismatch(stamped, "c0ffee1"); why != "" {
		t.Errorf("stamped journal rejected by its own build: %s", why)
	}
	// Each journal against the other agent. This is the situation the check
	// exists for: both journals are well formed and neither says anything is
	// wrong, and only the revision separates them.
	for _, tc := range []struct {
		name, want string
		c          capture
	}{
		{"a stamped journal read as the default build's", "", stamped},
		{"the default build's journal read as a stamped one", "c0ffee1", plain},
	} {
		if why := producerMismatch(tc.c, tc.want); why == "" {
			t.Errorf("%s: accepted, want a mismatch", tc.name)
		} else if !strings.Contains(why, "c0ffee1") {
			t.Errorf("%s: mismatch does not name the revision it saw: %s", tc.name, why)
		}
	}

	// A producer that failed before writing a capability record has no
	// identity to check, and the scenarios covering that failure assert on the
	// failure records instead. Refusing those journals here would turn this
	// check into a second, worse assertion about the same thing.
	if why := producerMismatch(capture{}, "c0ffee1"); why != "" {
		t.Errorf("journal with no capability record rejected: %s", why)
	}
}

// TestCapabilityReportsProducerBounds pins the producer configuration that the
// receipts cite, so a change to either has to be a deliberate edit here.
func TestCapabilityReportsProducerBounds(t *testing.T) {
	_, _, captures := run(t, "sustained", "1", "4", "256", "100")
	c := only(t, captures)
	if c.capability.DrainBatch != 256 {
		t.Errorf("drain_batch = %d, want 256", c.capability.DrainBatch)
	}
	if c.capability.QueueCapacity != 2048 {
		t.Errorf("queue_capacity = %d, want 2048: the fix is the batched drain, not a larger ring", c.capability.QueueCapacity)
	}
	// checkRetention reads this to tell a designed capacity drop from a
	// defect. A build that stopped reporting it would decode as zero, which
	// that check refuses outright rather than silently treating every loss as
	// legitimate.
	if c.capability.AdmitWaitNanos != 100*1000*1000 {
		t.Errorf("admit_wait_nanos = %d, want 100ms", c.capability.AdmitWaitNanos)
	}
	if !c.capability.SigtermFinalize {
		t.Error("sigterm_finalize = false, want true when the target has a default disposition")
	}
	if !c.capability.AtforkGuard {
		t.Error("atfork_guard = false: a forked child would keep a live-looking agent")
	}
	if !c.capability.APIRecords {
		t.Error("api_records = false, want true by default")
	}
	// The producer degrades when cuptiActivityGetStructSize is absent: every
	// runtime layout check is skipped and the kernel, memcpy and memset
	// layouts are assumed from the headers it was compiled against. That is
	// the right behaviour on a CUPTI older than 13.2 and the wrong one to
	// report as a verified layout, so the two are separate fields. Here the
	// stub exports the symbol, so a run that reports anything else is one
	// where the scenarios below covered none of the checking code.
	if c.capability.LayoutCheck != "cuptiActivityGetStructSize" {
		t.Errorf("layout_check = %q, want %q: the layouts were assumed, not verified", c.capability.LayoutCheck, "cuptiActivityGetStructSize")
	}
	// The optional collection policies are off unless asked for. A reader
	// takes these as the statement of what this process will do, so a default
	// that drifted would be a silent change of behaviour for every run.
	if c.capability.Finalize != "exit" {
		t.Errorf("finalize = %q, want %q by default", c.capability.Finalize, "exit")
	}
	if c.capability.FlushIntervalMS != 0 {
		t.Errorf("flush_interval_ms = %d, want 0 by default", c.capability.FlushIntervalMS)
	}
	if c.capability.Memsets || c.capability.MemsetsRequested {
		t.Errorf("memsets=%v memsets_requested=%v, want false/false by default", c.capability.Memsets, c.capability.MemsetsRequested)
	}
}

// TestForkedChildGoesInert covers fork(2) without exec. The child inherits the
// producer's initialized state and journal descriptor but not its drain
// thread, so finalizing there would join a pthread_t that names no thread and
// hang, and any record it wrote would interleave with the parent's into a
// single journal. The child must exit promptly and contribute nothing.
func TestForkedChildGoesInert(t *testing.T) {
	result, status, captures := run(t, "fork", "1", "8", "512", "0")
	if result.ChildExited != 1 {
		t.Fatalf("forked child did not exit on its own; it hung finalizing an agent it does not own")
	}
	if result.ChildStatus != 0 {
		t.Errorf("child wait status = %d, want 0", result.ChildStatus)
	}
	// One journal, not two: the child must not have opened or written one.
	c := only(t, captures)
	checkReconciles(t, c, result.Offered)
	if c.finalized.State != "collected" {
		t.Errorf("state = %q, want collected", c.finalized.State)
	}
	if !status.Exited() || status.ExitStatus() != 0 {
		t.Errorf("status = %v, want a clean exit", status)
	}
}

// TestForkedChildIsInert is the narrow fork control 6DE59D5A asked for. The
// ordinary-exit case above shows only that an inherited finalizer does not
// hang the child. This one asks whether every other inherited entry point is
// actually shut: the buffer-request callback, both external-correlation
// wrappers, the parent's journal descriptor, both ends of the parent's
// termination pipe, and the inherited SIGTERM handler.
//
// The signal question is the one with a parent-visible consequence. While the
// child holds the write end of the parent's self-pipe and the producer's
// handler, a SIGTERM in the child writes a byte the parent's terminator will
// consume with its own term_signalled still clear -- and that terminator then
// returns, so the parent loses its termination service for the rest of its
// life. The parent therefore signals itself after reaping the child and must
// still finalize.
func TestForkedChildIsInert(t *testing.T) {
	result, status, captures := run(t, "fork-inert", "1", "8", "512", "0")
	if result.ProbeRead != 1 {
		t.Fatalf("child sent no probe report; it did not reach the probes: %+v", result)
	}

	// The child must die by the disposition the target had before the producer
	// loaded, not by the producer's handler and not by the _exit(9) fallback.
	if result.ChildSignaled != 1 || result.ChildTermsig != int(syscall.SIGTERM) {
		t.Errorf("child ended signaled=%d termsig=%d, want termination by SIGTERM (%d): the child kept the producer's handler",
			result.ChildSignaled, result.ChildTermsig, int(syscall.SIGTERM))
	}

	// Inherited entry points. The parent's own result is the baseline: these
	// wrappers refuse a run without external correlation too, so a bare child
	// refusal would prove nothing on its own.
	if result.ParentPushResult != cuptiSuccess || result.ParentPopResult != cuptiSuccess {
		t.Fatalf("external wrappers were already refusing in the parent (push=%d pop=%d); the child probe proves nothing",
			result.ParentPushResult, result.ParentPopResult)
	}
	if result.ChildBufferGranted != 0 {
		t.Errorf("buffer_requested handed the forked child a buffer (%d); the child can then deliver records into the parent's journal", result.ChildBufferGranted)
	}
	if result.ChildPushResult != cuptiErrorNotSupported {
		t.Errorf("cudaprof_push_external in the child = %d, want CUPTI_ERROR_NOT_SUPPORTED (%d)", result.ChildPushResult, cuptiErrorNotSupported)
	}
	if result.ChildPopResult != cuptiErrorNotSupported {
		t.Errorf("cudaprof_pop_external in the child = %d, want CUPTI_ERROR_NOT_SUPPORTED (%d)", result.ChildPopResult, cuptiErrorNotSupported)
	}

	// Descriptors. /proc is how the child observes this, so a host without it
	// leaves the question unanswered rather than passed; the parent's own
	// counts are the evidence that the probe could see anything at all.
	switch {
	case result.ParentJournalFDs < 0 || result.ChildJournalFDs < 0:
		t.Log("descriptor closure unverified on this host: no /proc/self/fd")
	default:
		if result.ParentJournalFDs != 1 {
			t.Errorf("parent held %d journal descriptors, want 1; the probe is not measuring what it claims", result.ParentJournalFDs)
		}
		if result.ChildJournalFDs != 0 {
			t.Errorf("forked child still holds %d descriptors on the parent's journal", result.ChildJournalFDs)
		}
		// The process holds pipes the producer knows nothing about -- a
		// pipeline's own stdio among them -- so the gate is the difference,
		// which must be exactly the two ends of the termination pipe. The
		// child has already discounted the report pipe it still holds.
		if result.ParentPipeFDs < 2 {
			t.Errorf("parent held %d pipe descriptors, want at least the 2 ends of its termination pipe", result.ParentPipeFDs)
		}
		if got, want := result.ParentPipeFDs-result.ChildPipeFDs, 2; got != want {
			t.Errorf("forked child closed %d pipe descriptors (parent %d, child %d), want both ends of the parent's termination pipe",
				got, result.ParentPipeFDs, result.ChildPipeFDs)
		}
	}

	// One journal, and the parent's own termination still works.
	c := only(t, captures)
	if c.finalized == nil {
		t.Fatal("parent did not finalize after its own SIGTERM: the child's signal consumed the parent's termination service")
	}
	if c.finalized.Trigger != "signal" {
		t.Errorf("trigger = %q, want signal", c.finalized.Trigger)
	}
	if c.truncated {
		t.Error("parent journal truncated despite an orderly signal finalization")
	}
	if c.counts[kindFinalized] != 1 {
		t.Errorf("finalized records = %d, want exactly 1", c.counts[kindFinalized])
	}
	checkReconciles(t, c, result.Offered)
	if !status.Signaled() || status.Signal() != syscall.SIGTERM {
		t.Errorf("status = %v, want the parent to still die of SIGTERM", status)
	}
}

// TestAtforkRegistrationFailureIsRecorded covers the case where the guard the
// two fork controls above depend on could not be installed. Nothing in the
// producer can make that safe, so the requirement is only that the run says
// so: a failure record names it and the capability record reports the guard as
// absent, rather than a later reader assuming a protection that was never
// registered. The failure is injected at compile time; a shipping build has no
// way to reach this branch.
func TestAtforkRegistrationFailureIsRecorded(t *testing.T) {
	bin := controlBinary(t, "-DCUDAPROF_TEST_FORCE_ATFORK_FAILURE")
	result, status, captures := runBuilt(t, bin, nil, "sustained", "1", "4", "256", "100")
	c := only(t, captures)
	if c.capability.AtforkGuard {
		t.Error("capability claims atfork_guard with no handler registered")
	}
	var found bool
	for _, f := range c.failures {
		if strings.Contains(f, "pthread_atfork") {
			found = true
		}
	}
	if !found {
		t.Errorf("no failure record names the registration failure; failures = %q", c.failures)
	}
	// The run still produces a usable journal: an unregistered guard is a
	// recorded limitation, not a reason to stop capturing.
	if c.finalized == nil {
		t.Fatal("no finalized record")
	}
	checkReconcilesOutOfBand(t, c, result.Offered, map[uint16]uint64{kindFailure: 1})
	if !status.Exited() || status.ExitStatus() != 0 {
		t.Errorf("status = %v, want a clean exit", status)
	}
}

// TestPerKindAccountingSurvivesLoss is the control for the accounting gap the
// coverage audit exposed: the ring admits every kind through one path, so a
// kind-agnostic drop counter makes any statement about how much *activity* was
// lost unprovable, and the frozen SGLang receipt's 98.31% figure was exactly
// that mistake -- retained activities divided by all-kind drops.
//
// Admission's wait for drain space is compiled to zero so the real capacity
// drop path runs under real pressure. The gate is that each kind's own offered
// count splits into that kind's retained and dropped counts, with the retained
// side independently confirmed against the journal. A counter that charged
// every drop to one kind would still sum correctly and would fail here.
func TestPerKindAccountingSurvivesLoss(t *testing.T) {
	bin := controlBinary(t, "-DADMIT_WAIT_NANOS=0")
	result, _, captures := runBuilt(t, bin, nil, "burst", "4", "8", "20000", "0")
	c := only(t, captures)
	checkReconciles(t, c, result.Offered)
	fin := c.finalized
	if fin.QueueCapacityDropped == 0 {
		t.Fatal("no records were dropped; the control is not applying pressure and proves nothing about attribution")
	}
	activity, api := fin.ByKind["activity"], fin.ByKind["api"]
	if activity.Dropped == 0 || api.Dropped == 0 {
		t.Fatalf("loss did not reach both kinds: activity=%+v api=%+v", activity, api)
	}
	// The harness offers three activity records for every API record. If the
	// counters were kind-agnostic the two kinds could not hold that ratio on
	// both sides of the split.
	if activity.Offered != 3*api.Offered {
		t.Errorf("offered %d activity and %d api, want a 3:1 mix", activity.Offered, api.Offered)
	}
	for _, k := range []struct {
		name string
		got  kindCounts
	}{{"activity", activity}, {"api", api}} {
		if k.got.Retained == 0 {
			t.Errorf("by_kind[%s] retained nothing under loss: %+v", k.name, k.got)
		}
		if k.got.Retained+k.got.Dropped != k.got.Offered {
			t.Errorf("by_kind[%s] does not close: %+v", k.name, k.got)
		}
	}
	if fin.State != "producer_failure" {
		t.Errorf("state = %q, want producer_failure: records were lost", fin.State)
	}
}

// TestForkedChildKeepsApplicationSigterm covers the second boundary 6DE59D5A's
// review named. The child restores the disposition the producer saved at load,
// which is right only while the producer's handler is still the installed one.
// A target that replaces it afterwards -- CPython does -- owns that handler,
// and restoring SIG_DFL over it in the child would change how the child dies.
//
// The harness installs its own handler after load, forks, and has the child
// raise SIGTERM at itself: exit 5 means the application's handler ran, and
// death by SIGTERM means the producer overwrote it. The child is still fully
// disowned -- refused a buffer and both external wrappers -- which is what
// distinguishes preserving the handler from leaving the agent live.
func TestForkedChildKeepsApplicationSigterm(t *testing.T) {
	result, status, captures := run(t, "fork-inert-clobber", "1", "4", "256", "0")
	if result.ProbeRead != 1 {
		t.Fatal("child probe was not reported; nothing about the child is established")
	}
	if result.ChildSignaled != 0 {
		t.Errorf("child died of signal %d; the producer replaced the application's SIGTERM handler", result.ChildTermsig)
	}
	if result.ChildExited != 1 || result.ChildStatus != 5 {
		t.Errorf("child exited %d status %d, want exit 5: the application handler did not run in the child",
			result.ChildExited, result.ChildStatus)
	}
	if result.ParentPushResult != cuptiSuccess || result.ParentPopResult != cuptiSuccess {
		t.Fatalf("parent was already being refused external correlation (push=%d pop=%d); the child's refusal below would prove nothing",
			result.ParentPushResult, result.ParentPopResult)
	}
	if result.ChildBufferGranted != 0 {
		t.Errorf("child was granted an activity buffer (%d)", result.ChildBufferGranted)
	}
	if result.ChildPushResult != cuptiErrorNotSupported || result.ChildPopResult != cuptiErrorNotSupported {
		t.Errorf("child external correlation push=%d pop=%d, want both refused (%d)",
			result.ChildPushResult, result.ChildPopResult, cuptiErrorNotSupported)
	}
	c := only(t, captures)
	checkReconciles(t, c, result.Offered)
	if !status.Exited() || status.ExitStatus() != 0 {
		t.Errorf("status = %v, want a clean exit", status)
	}
}

// TestForkedChildKeepsApplicationSigtermMutation is the negative control for
// the test above: with the guard removed the child restores SIG_DFL over the
// application's handler and dies of the signal instead of handling it. If this
// ever passes, the control above has stopped discriminating.
func TestForkedChildKeepsApplicationSigtermMutation(t *testing.T) {
	bin := controlBinary(t, "-DCUDAPROF_TEST_FORCE_UNCONDITIONAL_SIGTERM_RESTORE")
	result, _, _ := runBuilt(t, bin, nil, "fork-inert-clobber", "1", "4", "256", "0")
	if result.ChildSignaled != 1 || result.ChildTermsig != int(syscall.SIGTERM) {
		t.Fatalf("mutant child was not killed by SIGTERM (signaled=%d termsig=%d status=%d); the guard is not what the control measures",
			result.ChildSignaled, result.ChildTermsig, result.ChildStatus)
	}
}

// TestMajorVersionMismatchIsRejected verifies that the producer rejects cross-major
// execution skew (e.g. CUDA 12 Kernel9 vs CUDA 13 Kernel10, or CUDA 13 vs CUDA 14),
// recording a failure record naming the major version mismatch and refusing to
// collect activity records.
func TestMajorVersionMismatchIsRejected(t *testing.T) {
	tests := []struct {
		name       string
		compileVer string
		runtimeVer string
	}{
		{
			name:       "cuda12_compile_cuda13_runtime",
			compileVer: "27",
			runtimeVer: "130202",
		},
		{
			name:       "cuda13_compile_cuda12_runtime",
			compileVer: "130202",
			runtimeVer: "27",
		},
		{
			name:       "cuda13_compile_cuda14_runtime",
			compileVer: "130202",
			runtimeVer: "140000",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			defines := []string{
				"-DCUPTI_API_VERSION=" + tc.compileVer,
				"-DCUDAPROF_TEST_OVERRIDE_RUNTIME_VERSION=" + tc.runtimeVer,
			}
			bin := controlBinary(t, defines...)
			_, status, captures := runBuilt(t, bin, nil, "sustained", "1", "1", "10", "0")
			c := only(t, captures)

			var found bool
			for _, f := range c.failures {
				if strings.Contains(f, "CUPTI major version mismatch") {
					found = true
				}
			}
			if !found {
				t.Fatalf("no failure record names major version mismatch; failures = %q", c.failures)
			}
			if count := c.counts[kindActivity]; count != 0 {
				t.Errorf("collected %d activity records; expected 0 due to rejection", count)
			}
			if c.capability.Activity {
				t.Error("capability claimed activity support despite major version rejection")
			}
			if !status.Exited() || status.ExitStatus() != 0 {
				t.Errorf("status = %v, want clean exit", status)
			}
		})
	}
}

// TestShortKernelActivityLayoutIsRejected verifies that when the runtime's kernel activity
// struct size is smaller than the compiled layout requirement, the producer fails closed,
// writing a failure record naming the kernel layout incompatibility and refusing activity collection.
func TestShortKernelActivityLayoutIsRejected(t *testing.T) {
	bin := controlBinary(t, "-DCUDAPROF_TEST_STRUCT_SIZE_FN=cudaprof_test_mock_short_kernel_struct_size")
	_, status, captures := runBuilt(t, bin, nil, "sustained", "1", "1", "10", "0")
	c := only(t, captures)

	var found bool
	for _, f := range c.failures {
		if strings.Contains(f, "CUPTI runtime kernel activity layout incompatible") {
			found = true
		}
	}
	if !found {
		t.Fatalf("no failure record names kernel layout incompatibility; failures = %q", c.failures)
	}
	if count := c.counts[kindActivity]; count != 0 {
		t.Errorf("collected %d activity records; expected 0", count)
	}
	if c.capability.Activity {
		t.Error("capability claimed activity support despite kernel layout rejection")
	}
	if !status.Exited() || status.ExitStatus() != 0 {
		t.Errorf("status = %v, want clean exit", status)
	}
}

// TestShortMemcpyActivityLayoutIsRejected verifies that when the runtime's memcpy activity
// struct size is smaller than the compiled layout requirement, the producer fails closed,
// writing a failure record naming the memcpy layout incompatibility and refusing activity collection.
func TestShortMemcpyActivityLayoutIsRejected(t *testing.T) {
	bin := controlBinary(t, "-DCUDAPROF_TEST_STRUCT_SIZE_FN=cudaprof_test_mock_short_memcpy_struct_size")
	_, status, captures := runBuilt(t, bin, nil, "sustained", "1", "1", "10", "0")
	c := only(t, captures)

	var found bool
	for _, f := range c.failures {
		if strings.Contains(f, "CUPTI runtime memcpy activity layout incompatible") {
			found = true
		}
	}
	if !found {
		t.Fatalf("no failure record names memcpy layout incompatibility; failures = %q", c.failures)
	}
	if count := c.counts[kindActivity]; count != 0 {
		t.Errorf("collected %d activity records; expected 0", count)
	}
	if c.capability.Activity {
		t.Error("capability claimed activity support despite memcpy layout rejection")
	}
	if !status.Exited() || status.ExitStatus() != 0 {
		t.Errorf("status = %v, want clean exit", status)
	}
}

// TestNVTXMarkersArriveAsPairs covers the NVTX tier. CUPTI does not report a
// range: it reports a START record and an END record that share an id, each
// with one timestamp, and the duration exists only in the pair. The producer
// transcribes what it received, so the control here is that both halves reach
// the journal with the id intact and neither half claims a duration of its
// own.
//
// One buffer, deliberately. The stub numbers marker ids from one within each
// offer, so a second buffer would reuse ids 1 and 2 and pairing by id alone
// would no longer identify a pair. That ambiguity is real in CUPTI too; this
// control is about the producer's transcription, not about resolving it.
func TestNVTXMarkersArriveAsPairs(t *testing.T) {
	_, _, captures := run(t, "sustained", "1", "1", "8", "0", "0", "4")
	c := only(t, captures)
	if !c.capability.NVTXMarkers {
		t.Fatal("capability reports nvtx_markers = false: the producer never enabled CUPTI_ACTIVITY_KIND_MARKER, so every assertion below would be about an empty set")
	}
	if len(c.markers) != 4 {
		t.Fatalf("got %d markers, want 4 (two ranges)", len(c.markers))
	}
	starts := map[uint32]markerRecord{}
	ends := map[uint32]markerRecord{}
	for _, m := range c.markers {
		if m.Name != "cudaprof_control_range" || m.Domain != "cudaprof" {
			t.Errorf("marker %d: name=%q domain=%q, want the stub's name and domain", m.ID, m.Name, m.Domain)
		}
		// A marker is a point in time. Both timestamps carry the one value
		// CUPTI reported so that a consumer reading a duration off a single
		// record reads zero rather than a fabricated span.
		if m.Start != m.End {
			t.Errorf("marker %d spans %d..%d: one marker has one timestamp", m.ID, m.Start, m.End)
		}
		if m.ProcessID != 1 || m.ThreadID != 2 {
			t.Errorf("marker %d: pid=%d tid=%d, want 1/2 from the thread arm of the object union", m.ID, m.ProcessID, m.ThreadID)
		}
		if m.Stream != 0 {
			t.Errorf("marker %d: stream=%d, want 0: a thread-attributed marker has no stream, and filling one in would put a host range on a device track", m.ID, m.Stream)
		}
		switch m.Flags {
		case markerFlagStart:
			starts[m.ID] = m
		case markerFlagEnd:
			ends[m.ID] = m
		default:
			t.Errorf("marker %d: flags=%d, want START(%d) or END(%d)", m.ID, m.Flags, markerFlagStart, markerFlagEnd)
		}
	}
	for _, id := range []uint32{1, 2} {
		s, ok := starts[id]
		if !ok {
			t.Errorf("range %d has no START record", id)
			continue
		}
		e, ok := ends[id]
		if !ok {
			t.Errorf("range %d has no END record: an unclosed range is the open-range case, not a lost record", id)
			continue
		}
		if e.Start <= s.Start {
			t.Errorf("range %d closes at %d, at or before its open at %d", id, e.Start, s.Start)
		}
	}
}

// rangeResult is the capture-range scenario's line. It reports record counts
// and enabled-kind counts at the same three points, because a run that
// collected nothing with the gate shut and a run that collected nothing
// because every enable failed both report zero records, and only the kind
// count tells them apart.
type rangeResult struct {
	Subscribed   int  `json:"subscribed"`
	Domain       int  `json:"domain"`
	StartEnabled int  `json:"start_enabled"`
	StopEnabled  int  `json:"stop_enabled"`
	FiredStart   int  `json:"fired_start"`
	FiredStop    int  `json:"fired_stop"`
	Before       int  `json:"before"`
	Inside       int  `json:"inside"`
	After        int  `json:"after"`
	BeforeKinds  uint `json:"before_kinds"`
	InsideKinds  uint `json:"inside_kinds"`
	AfterKinds   uint `json:"after_kinds"`

	// Edges is what the producer actually wrote to the journal, which the
	// harness's own counters cannot speak to. Captures is every journal the
	// run produced, for the scenarios whose question is what the journal ends
	// with rather than what CUPTI was told.
	Edges    []rangeEdgeRecord `json:"-"`
	Captures []capture         `json:"-"`
}

// runRange executes the capture-range scenario and decodes its line.
func runRange(t *testing.T, env []string, domain string) rangeResult {
	t.Helper()
	return runRangeScenario(t, "capture-range", env, domain)
}

// runRangeScenario is runRange for one of the scenario's variants: -restart
// runs the start/stop pair twice, -abrupt ends the process with _exit and so
// runs neither the ELF destructor nor the atexit handler.
func runRangeScenario(t *testing.T, scenario string, env []string, domain string) rangeResult {
	t.Helper()
	dir := t.TempDir()
	cmd := exec.Command(controlBinary(t), scenario, "1", "4", "0", "0", domain)
	cmd.Env = append(os.Environ(), "CUDAPROF_TRANSPORT="+dir, "CUDAPROF_SESSION="+sessionHex)
	cmd.Env = append(cmd.Env, env...)
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("capture-range domain %s: %v", domain, err)
	}
	var got rangeResult
	line := bytes.TrimSpace(out)
	if i := bytes.LastIndexByte(line, '\n'); i >= 0 {
		line = line[i+1:]
	}
	if err := json.Unmarshal(line, &got); err != nil {
		t.Fatalf("decode capture-range line %q: %v", line, err)
	}
	// The harness's stdout reports what the stub observed; the journal reports
	// what the producer wrote. Only the second can say whether the edge records
	// exist, and they are the records a consumer reads to tell an unopened
	// range from an empty one.
	for _, c := range decodeAll(t, dir, binRevision(controlBinary(t))) {
		got.Edges = append(got.Edges, c.ranges...)
		got.Captures = append(got.Captures, c)
	}
	return got
}

// TestCaptureRangeGatesCollection is the gate for capture-range gating, and it
// is crossed on purpose: neither arm alone proves anything.
//
// The ungated arm establishes that this fixture CAN deliver records at all
// three points. Without it, a gated run reporting zero inside the range would
// be indistinguishable from a fixture that delivers nothing ever, and the
// gated assertions would pass on a producer that collects nothing at all.
//
// The gated arm then requires the counts to MOVE: shut before the range, open
// inside it, shut again after. A gate that never opened and a gate that never
// closed each fail exactly one of those three, so no single defect satisfies
// all of them.
//
// Both callback domains are covered because a target calls either the runtime
// pair or the driver pair and never both; a gate wired only to the runtime
// domain would silently never open for a target that calls cuProfilerStart.
func TestCaptureRangeGatesCollection(t *testing.T) {
	const gate = "CUDAPROF_CAPTURE_RANGE=cudaProfilerApi"
	for _, d := range []struct {
		name   string
		domain string
	}{
		{"runtime", "2"},
		{"driver", "1"},
	} {
		t.Run(d.name, func(t *testing.T) {
			// Control: no gate requested. Records must flow throughout, or
			// the gated arm below is measuring a fixture that is simply mute.
			open := runRange(t, nil, d.domain)
			if open.Subscribed != 0 {
				t.Errorf("ungated run subscribed to the callback API: %+v", open)
			}
			if open.Before == 0 || open.Inside == 0 || open.After == 0 {
				t.Fatalf("ungated run collected nothing somewhere, so the gated arm cannot discriminate: %+v", open)
			}
			if open.BeforeKinds == 0 {
				t.Fatalf("ungated run enabled no activity kinds: %+v", open)
			}

			got := runRange(t, []string{gate}, d.domain)
			if got.Subscribed != 1 || got.StartEnabled != 1 || got.StopEnabled != 1 {
				t.Fatalf("gated run did not subscribe to domain %s: %+v", d.domain, got)
			}
			if got.FiredStart != 1 || got.FiredStop != 1 {
				t.Fatalf("callbacks were not delivered: %+v", got)
			}
			if got.Before != 0 {
				t.Errorf("collected %d records before the range opened, want 0: %+v", got.Before, got)
			}
			if got.Inside != open.Inside {
				t.Errorf("collected %d records inside the range, want %d as the ungated run delivered: %+v", got.Inside, open.Inside, got)
			}
			if got.After != 0 {
				t.Errorf("collected %d records after the range closed, want 0: %+v", got.After, got)
			}
			// The kind counts separate "the gate is shut" from "nothing was
			// ever enabled". Zero records with a nonzero kind count would mean
			// the gate is not what suppressed them.
			if got.BeforeKinds != 0 || got.AfterKinds != 0 {
				t.Errorf("activity kinds stayed enabled outside the range (before %d, after %d): %+v", got.BeforeKinds, got.AfterKinds, got)
			}
			if got.InsideKinds != open.InsideKinds {
				t.Errorf("enabled %d kinds inside the range, want %d: %+v", got.InsideKinds, open.InsideKinds, got)
			}

			// Everything above reads the harness's counters, which say what
			// CUPTI was told and nothing about what the producer wrote. The
			// edge records are the journal's own account of where the range
			// opened and closed, and a consumer needs them to tell a journal
			// whose range never opened from one that collected nothing. They
			// were asserted nowhere: deleting write_range_edge's write left
			// this whole file green.
			if len(got.Edges) != 2 {
				t.Fatalf("journal holds %d capture-range edges, want a start and a stop: %+v", len(got.Edges), got.Edges)
			}
			start, stop := got.Edges[0], got.Edges[1]
			if start.Edge != "start" || stop.Edge != "stop" {
				t.Errorf("edges are %q then %q, want start then stop", start.Edge, stop.Edge)
			}
			// One range opened, so both edges name range 0. An index taken from
			// the open count rather than the range would put the stop at 1.
			if start.Index != 0 || stop.Index != 0 {
				t.Errorf("edge indices are %d and %d, want both 0 for a single range", start.Index, stop.Index)
			}
			if start.ToggleFailures != 0 || stop.ToggleFailures != 0 {
				t.Errorf("toggling activity kinds failed at an edge: %+v", got.Edges)
			}
			wantAPI := map[string][2]string{
				"2": {"cudaProfilerStart", "cudaProfilerStop"},
				"1": {"cuProfilerStart", "cuProfilerStop"},
			}[d.domain]
			if start.API != wantAPI[0] || stop.API != wantAPI[1] {
				t.Errorf("edges name %q/%q, want %q/%q: the record must say which API opened the range", start.API, stop.API, wantAPI[0], wantAPI[1])
			}

			// The ungated arm must write no edges at all -- otherwise the
			// assertions above would pass on a producer that records edges
			// whether or not it gates anything.
			if len(open.Edges) != 0 {
				t.Errorf("ungated run wrote %d capture-range edges, want none: %+v", len(open.Edges), open.Edges)
			}
		})
	}
}

// TestStopEdgePrecedesFlush checks that a range's stop edge is stamped
// before the producer flushes the range's records. The flush takes time in
// proportion to the records it delivers, so an edge stamped after it charged
// that time to the range: on a GB10 prefill it was 62 ms for C's 3427 kernels
// and 176 ms for Go's 11538, read as device idle at the end of the phase.
func TestStopEdgePrecedesFlush(t *testing.T) {
	const delay = 50 * time.Millisecond
	got := runRange(t, []string{
		"CUDAPROF_CAPTURE_RANGE=cudaProfilerApi",
		fmt.Sprintf("CUDAPROF_TEST_FLUSH_DELAY_MS=%d", delay.Milliseconds()),
	}, "2")
	if len(got.Edges) != 2 || got.Edges[1].Edge != "stop" {
		t.Fatalf("edges = %+v, want a start and a stop", got.Edges)
	}
	stop := got.Edges[1]
	if stop.FlushNanos < uint64(delay) {
		t.Errorf("stop edge flush_nanos = %v, want at least the stub's %v", time.Duration(stop.FlushNanos), delay)
	}
	var fin *capture
	for i := range got.Captures {
		if got.Captures[i].finalized != nil && len(got.Captures[i].ranges) > 0 {
			fin = &got.Captures[i]
		}
	}
	if fin == nil {
		t.Fatal("no finalized journal holds the range edges")
	}
	// The flush lies between the stop edge and finalization. An edge stamped
	// after the flush sits within microseconds of the finalized record.
	if d := time.Duration(fin.finalizedAt - stop.Timestamp); d < delay {
		t.Errorf("finalized %v after the stop edge, want at least the %v flush between them", d, delay)
	}
}

// TestCaptureRangeRefusesUnpairedTriggers covers the arming failure that fails
// CLOSED and says nothing: both cudaProfilerStart cbids refuse while both Stop
// cbids arm. Counting surviving triggers without regard to direction reads that
// as a working gate, shuts collection off, and waits for an edge that no
// surviving callback can ever deliver.
//
// The resulting journal is byte-identical to a correctly gated run whose target
// simply never called cudaProfilerStart -- same "initialized_no_activity", same
// "capture_range":"cudaProfilerApi" -- which is precisely the
// indistinguishable-empty-journal state this gating exists to refuse. The only
// honest answer is to decline the gate and say so in the capability record.
func TestCaptureRangeRefusesUnpairedTriggers(t *testing.T) {
	for _, tt := range []struct {
		name   string
		refuse string
		domain string
	}{
		{"no_start_trigger", "171,308", "2"},
		{"no_stop_trigger", "172,309", "2"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got := runRange(t, []string{
				"CUDAPROF_CAPTURE_RANGE=cudaProfilerApi",
				"CUDAPROF_TEST_REFUSE_CBIDS=" + tt.refuse,
			}, tt.domain)

			// Collection must be ON: an ungated capture is a worse capture,
			// but it is an honest one, and the capability record says the
			// request could not be met. A gate that cannot open is neither.
			if got.BeforeKinds == 0 {
				t.Errorf("activity kinds are off with no usable gate, so the run collects nothing and nothing says why: %+v", got)
			}
			if len(got.Edges) != 0 {
				t.Errorf("wrote %d range edges with an unpaired trigger set: %+v", len(got.Edges), got.Edges)
			}
		})
	}
}

// TestFinalizeOnProfilerStop covers the only finalization trigger a Go target
// can be relied on to reach. A Go program leaves through exit_group: it runs
// no ELF destructor, no atexit handler, and takes no SIGTERM, so all three of
// the producer's other triggers are skipped and the journal ends with whatever
// CUPTI happened to hand over before the process vanished. On a real target
// that was 0 activity records where the same workload in C collected 51.
//
// The scenario therefore ends with _exit, and the arms are crossed the same
// way the gating control is. The default arm must leave the journal
// unfinalized -- it is the bug, reproduced -- or the finalizing arm is not
// evidence about the trigger. Both callback domains are covered because a
// target calls either the runtime pair or the driver pair and never both.
func TestFinalizeOnProfilerStop(t *testing.T) {
	const gate = "CUDAPROF_CAPTURE_RANGE=cudaProfilerApi"
	const mode = "CUDAPROF_FINALIZE=profiler-stop"
	for _, d := range []struct {
		name   string
		domain string
	}{
		{"runtime", "2"},
		{"driver", "1"},
	} {
		t.Run(d.name, func(t *testing.T) {
			// The bug: the same gated capture, the same stop, the same abrupt
			// exit, without the mode. Nothing finalizes it.
			open := runRangeScenario(t, "capture-range-abrupt", []string{gate}, d.domain)
			unfinalized := only(t, open.Captures)
			if unfinalized.finalized != nil {
				t.Fatalf("a stop finalized the capture without %s, so this control proves nothing: %+v", mode, *unfinalized.finalized)
			}
			if unfinalized.capability.Finalize != "exit" {
				t.Errorf("finalize = %q without the mode, want %q", unfinalized.capability.Finalize, "exit")
			}

			got := runRangeScenario(t, "capture-range-abrupt", []string{gate, mode}, d.domain)
			c := only(t, got.Captures)
			if c.finalized == nil {
				t.Fatalf("the stop did not finalize the capture: %+v", got)
			}
			if c.finalized.Trigger != "profiler-stop" {
				t.Errorf("finalized by %q, want %q: the record names the trigger that ran the sequence", c.finalized.Trigger, "profiler-stop")
			}
			if c.finalized.State != "collected" {
				t.Errorf("state = %q, want collected", c.finalized.State)
			}
			if c.capability.Finalize != "profiler-stop" {
				t.Errorf("finalize = %q, want profiler-stop: the record has to say which trigger this process will use", c.capability.Finalize)
			}
			// Terminated is not the same as complete: the producer's own
			// accounting has to close over everything the range collected.
			// The edge records are written outside the ring, like the
			// bootstrap and capability records, so they are in the journal
			// and in no counter.
			checkReconcilesOutOfBand(t, c, uint64(got.Inside), map[uint16]uint64{kindActivity: uint64(len(got.Edges))})
			if c.finalized.Activity == 0 {
				t.Errorf("finalized with no activity, so the stop published an empty capture: %+v", *c.finalized)
			}
			// Finalizing at the stop must not cost the gate: this journal is
			// still a range capture and holds both of its edges.
			if len(got.Edges) != 2 {
				t.Fatalf("journal holds %d capture-range edges, want a start and a stop: %+v", len(got.Edges), got.Edges)
			}
			if got.Before != 0 || got.After != 0 {
				t.Errorf("collected %d records before and %d after the range, want 0 and 0: %+v", got.Before, got.After, got)
			}
		})
	}
}

// TestFinalizeOnProfilerStopWithoutRange covers the target that has no range
// to gate. It never calls Start; it calls Stop only to hand the agent the
// flush its exit path will never run. Gating such a run would collect nothing,
// so the two requests are answered separately from one subscription: the
// capture is ungated from the first instruction and the stop still ends it.
func TestFinalizeOnProfilerStopWithoutRange(t *testing.T) {
	got := runRangeScenario(t, "capture-range-abrupt", []string{"CUDAPROF_FINALIZE=profiler-stop"}, "1")
	c := only(t, got.Captures)
	if got.Subscribed != 1 || got.StopEnabled != 1 {
		t.Fatalf("no subscription for a finalize-only request: %+v", got)
	}
	if got.Before == 0 {
		t.Errorf("collected nothing before the stop, so asking for the trigger alone gated the capture: %+v", got)
	}
	if c.capability.CaptureRange != "none" {
		t.Errorf("capture_range = %q, want none: nothing asked for a range", c.capability.CaptureRange)
	}
	if len(got.Edges) != 0 {
		t.Errorf("wrote %d range edges for an ungated run: %+v", len(got.Edges), got.Edges)
	}
	if c.finalized == nil {
		t.Fatalf("the stop did not finalize the capture: %+v", got)
	}
	if c.finalized.Trigger != "profiler-stop" {
		t.Errorf("finalized by %q, want %q", c.finalized.Trigger, "profiler-stop")
	}
	checkReconciles(t, c, uint64(got.Before+got.Inside))
}

// TestFinalizeOnProfilerStopUnavailable covers the request that cannot be met:
// the stop callback will not arm, so no stop can ever reach the producer. The
// capability record has to say so. A run that reported "profiler-stop" here
// would promise a trigger that does not exist, and the journal it leaves --
// unfinalized, because nothing else runs either -- would look like a target
// that simply never stopped.
func TestFinalizeOnProfilerStopUnavailable(t *testing.T) {
	got := runRangeScenario(t, "capture-range-abrupt", []string{
		"CUDAPROF_FINALIZE=profiler-stop",
		"CUDAPROF_TEST_REFUSE_CBIDS=172,309",
	}, "1")
	c := only(t, got.Captures)
	if c.capability.Finalize != "requested_unavailable" {
		t.Errorf("finalize = %q, want requested_unavailable when no stop trigger armed", c.capability.Finalize)
	}
	// Collection stays on. An unreachable trigger is a reason to say so, not
	// a reason to collect less.
	if got.BeforeKinds == 0 {
		t.Errorf("activity kinds are off after a failed arm: %+v", got)
	}
	if len(c.failures) == 0 {
		t.Error("no failure record says why the trigger could not be armed")
	}
}

// TestProfilerStopRestartKeepsCollecting is why finalizing on a stop is opt-in.
// A stop is not by itself the end of a capture: a target may stop and start
// again, and the default has to survive that. The scenario runs the pair
// twice, so a producer that ended the capture at the first stop writes two
// edges where four are due and finalizes under the wrong trigger.
func TestProfilerStopRestartKeepsCollecting(t *testing.T) {
	got := runRangeScenario(t, "capture-range-restart", []string{"CUDAPROF_CAPTURE_RANGE=cudaProfilerApi"}, "1")
	c := only(t, got.Captures)
	if len(got.Edges) != 4 {
		t.Fatalf("journal holds %d edges, want a start and a stop for each of two ranges: %+v", len(got.Edges), got.Edges)
	}
	for i, want := range []struct {
		edge  string
		index uint
	}{{"start", 0}, {"stop", 0}, {"start", 1}, {"stop", 1}} {
		if got.Edges[i].Edge != want.edge || got.Edges[i].Index != want.index {
			t.Errorf("edge %d is %q index %d, want %q index %d", i, got.Edges[i].Edge, got.Edges[i].Index, want.edge, want.index)
		}
	}
	// The second range collected: the reported counts are that cycle's.
	if got.Inside == 0 {
		t.Errorf("the reopened range collected nothing: %+v", got)
	}
	if c.finalized == nil {
		t.Fatal("no finalized record")
	}
	if c.finalized.Trigger != "exit" {
		t.Errorf("finalized by %q, want exit: a stop must not end the capture unless the mode asked it to", c.finalized.Trigger)
	}
	// Two cycles of the same offer. The four edges are written outside the
	// ring and belong to no counter.
	checkReconcilesOutOfBand(t, c, 2*uint64(got.Inside), map[uint16]uint64{kindActivity: uint64(len(got.Edges))})
}

// TestExportedFinalizeClosesTheJournal covers cudaprof_finalize, the entry
// point for a host that reaches neither the profiler API nor a libc exit path
// and has to say so itself. The scenario calls it and then leaves by _exit, so
// the journal can only be complete if that call completed it. Calling it twice
// covers the idempotence a host cannot otherwise rely on: the second call must
// report the same finalized capture and must not write a second terminal
// record, which checkReconciles asserts by counting them.
func TestExportedFinalizeClosesTheJournal(t *testing.T) {
	result, status, captures := run(t, "finalize-export", "1", "4", "0", "0")
	if status.ExitStatus() != 0 {
		t.Fatalf("harness exited %d", status.ExitStatus())
	}
	if result.FinalizeRC != 0 || result.FinalizeRCAgain != 0 {
		t.Fatalf("cudaprof_finalize returned %d then %d, want 0 both times", result.FinalizeRC, result.FinalizeRCAgain)
	}
	c := only(t, captures)
	if c.finalized == nil {
		t.Fatal("no finalized record: nothing but the export could have written one here")
	}
	if c.finalized.Trigger != "host" {
		t.Errorf("finalized by %q, want %q", c.finalized.Trigger, "host")
	}
	if c.finalized.State != "collected" {
		t.Errorf("state = %q, want collected", c.finalized.State)
	}
	checkReconciles(t, c, result.Offered)
	// Everything the agent owns is shut once the export returns. An offer
	// accepted after the finalized record would be a record no accounting in
	// that journal covers.
	if result.After != 0 {
		t.Errorf("the producer accepted %d records after finalizing", result.After)
	}
}

// TestPeriodicFlushPublishesWithoutATrigger covers the last resort for a target
// that cooperates with nothing: it never fills a CUPTI buffer, never calls the
// profiler API, and leaves by exit_group. The stub holds its records until
// something flushes, so this journal holds them only if the producer's own
// interval flush ran.
//
// What it produces is a partial capture by construction -- records with no
// finalized record after them -- and the control asserts that shape rather
// than hiding it: the consumer recovers such a journal knowingly.
func TestPeriodicFlushPublishesWithoutATrigger(t *testing.T) {
	// Control: default off. The same staged records, the same wait, no
	// interval -- and nothing reaches the journal at all.
	off, _, captures := run(t, "flush-interval", "1", "4", "0", "0", "400")
	quiet := only(t, captures)
	if off.Flushes != 0 {
		t.Errorf("flushed %d times with no interval configured", off.Flushes)
	}
	if off.Offered != 0 {
		t.Errorf("delivered %d records with no interval configured", off.Offered)
	}
	if quiet.capability.FlushIntervalMS != 0 {
		t.Errorf("flush_interval_ms = %d, want 0", quiet.capability.FlushIntervalMS)
	}
	if n := quiet.counts[kindActivity] + quiet.counts[kindAPI]; n != 0 {
		t.Fatalf("journal holds %d records without a flush, so the armed arm cannot discriminate", n)
	}

	on, _, captures := runEnv(t, []string{"CUDAPROF_FLUSH_INTERVAL_MS=50"}, "flush-interval", "1", "4", "0", "0", "400")
	c := only(t, captures)
	if on.Flushes == 0 {
		t.Fatalf("the interval flusher never reached CUPTI: %+v", on)
	}
	if on.Offered == 0 {
		t.Fatalf("a flush happened but delivered nothing: %+v", on)
	}
	if c.capability.FlushIntervalMS != 50 {
		t.Errorf("flush_interval_ms = %d, want 50: the record states the interval this process runs", c.capability.FlushIntervalMS)
	}
	if n := c.counts[kindActivity] + c.counts[kindAPI] - uint64(c.deviceActive); n != on.Offered {
		t.Errorf("journal holds %d of the %d records the flush delivered", n, on.Offered)
	}
	// The flush is also where CUPTI first asks this producer for a buffer, and
	// the note of that request is the only mark of device work a journal
	// ending like this one carries: the records themselves are here because a
	// flush ran, and a target that reaches no flush and no trigger publishes
	// none of them. A consumer reads the note's absence as a process that ran
	// no CUDA, so a run with device work has to write it exactly once, ahead
	// of the records it is about.
	if c.deviceActive != 1 {
		t.Errorf("device_active notes = %d, want 1", c.deviceActive)
	}
	if quiet.deviceActive != 0 {
		t.Errorf("device_active notes = %d without a flush, so the note does not discriminate", quiet.deviceActive)
	}
	if c.finalized != nil {
		t.Errorf("a process that left by _exit wrote a finalized record: %+v", *c.finalized)
	}
}

// TestMemsetsReachTheJournal covers CUPTI_ACTIVITY_KIND_MEMSET end to end: the
// producer's enable, the buffer-completed decode, and the journal record a
// consumer reads.
//
// It exists because the kind was absent from the producer entirely -- not
// enabled, not decoded -- and nothing failed. CUPTI offered no memset records
// because none were asked for, the decode loop had no branch that would have
// missed them, and the journal was complete by its own accounting. The device
// time a cudaMemsetAsync occupied simply was not in the stream, so every
// consumer that derives idle time from it charged that time to idle.
//
// The assertions are on the values the stub wrote, not merely on a count: a
// record that arrives with a zero span or zero bytes would still be a record,
// and it is the span that has to be there for the idle arithmetic to change.
func TestMemsetsReachTheJournal(t *testing.T) {
	// One thread, one buffer, no kernels/memcpys/apis of interest, six memsets.
	// argv: scenario threads buffers per_buffer pause_micros _ markers memsets
	_, status, captures := runEnv(t, []string{"CUDAPROF_MEMSETS=1"}, "sustained", "1", "1", "8", "0", "0", "0", "6")
	c := only(t, captures)
	if !c.capability.Memsets {
		t.Fatal("capability reports memsets = false: the producer never enabled CUPTI_ACTIVITY_KIND_MEMSET, so every assertion below would be about an empty set")
	}
	if len(c.memsets) != 6 {
		t.Fatalf("got %d memset records, want 6", len(c.memsets))
	}
	for i, m := range c.memsets {
		if m.Subtype != "memset" {
			t.Errorf("memset %d: subtype=%q, want \"memset\"", i, m.Subtype)
		}
		if !m.Valid {
			t.Errorf("memset %d: valid=false, want true: the stub wrote a well-ordered span", i)
		}
		// The span is the whole point. A memset carried with start == end
		// would leave the idle arithmetic exactly where it was.
		if m.End <= m.Start {
			t.Errorf("memset %d spans %d..%d, want a positive duration", i, m.Start, m.End)
		}
		if got := m.End - m.Start; got != 40 {
			t.Errorf("memset %d duration = %d ns, want 40 as the stub wrote it", i, got)
		}
		if m.Bytes != 8192 {
			t.Errorf("memset %d: bytes=%d, want 8192", i, m.Bytes)
		}
		if m.Stream != 7 || m.Context != 1 {
			t.Errorf("memset %d: context=%d stream=%d, want 1/7 from the stub record", i, m.Context, m.Stream)
		}
		if m.Correlation == 0 {
			t.Errorf("memset %d has correlation 0, so it can never be joined to the API call that launched it", i)
		}
	}
	if len(c.failures) != 0 {
		t.Errorf("producer reported failures: %q", c.failures)
	}
	if !status.Exited() || status.ExitStatus() != 0 {
		t.Errorf("status = %v, want clean exit", status)
	}
}

// TestMemsetsOptIn covers CUDAPROF_MEMSETS: the kind is enabled only for
// the value 1, the capability record says whether it was asked for, and the
// other kinds are collected either way. Memsets are opt-in because enabling
// them has faulted CUDA graph relaunches inside the driver.
func TestMemsetsOptIn(t *testing.T) {
	tests := []struct {
		name string
		env  []string
		want bool
	}{
		{"unset", nil, false},
		{"zero", []string{"CUDAPROF_MEMSETS=0"}, false},
		{"empty", []string{"CUDAPROF_MEMSETS="}, false},
		{"true", []string{"CUDAPROF_MEMSETS=true"}, false},
		{"one", []string{"CUDAPROF_MEMSETS=1"}, true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// Remove any inherited setting so "unset" means unset.
			t.Setenv("CUDAPROF_MEMSETS", "")
			os.Unsetenv("CUDAPROF_MEMSETS")
			_, status, captures := runEnv(t, tc.env, "sustained", "1", "1", "8", "0", "0", "0", "6")
			c := only(t, captures)
			if c.capability.Memsets != tc.want || c.capability.MemsetsRequested != tc.want {
				t.Errorf("capability memsets=%v memsets_requested=%v, want %v/%v", c.capability.Memsets, c.capability.MemsetsRequested, tc.want, tc.want)
			}
			want := 0
			if tc.want {
				want = 6
			}
			if len(c.memsets) != want {
				t.Errorf("got %d memset records, want %d", len(c.memsets), want)
			}
			if c.counts[kindActivity] == 0 {
				t.Error("run retained no activity")
			}
			if len(c.failures) != 0 {
				t.Errorf("producer reported failures: %q", c.failures)
			}
			if !status.Exited() || status.ExitStatus() != 0 {
				t.Errorf("status = %v, want clean exit", status)
			}
		})
	}
}

// TestUndecodedActivityKindIsReported covers the silence that let the memset
// gap persist. The decode loop recognizes a fixed set of kinds; every other
// anomaly in that loop -- a short record, a malformed buffer -- queues a
// producer failure, but an unrecognized kind was skipped with no record at
// all. A consumer then cannot distinguish a kind that produced nothing from a
// kind whose records this producer threw away.
//
// The control offers a kind the producer has no branch for and never will,
// which is the real shape of this case: a runtime newer than the build
// delivering something the build does not know. Reaching the arm by removing a
// decode branch instead would have tested a binary that is not the production
// one, and would have left a define that silently disables memset collection
// if it ever leaked into a real build -- the same fail-open this arm exists to
// close.
func TestUndecodedActivityKindIsReported(t *testing.T) {
	// argv: scenario threads buffers per_buffer pause_micros _ markers memsets undecoded
	_, status, captures := run(t, "sustained", "1", "1", "8", "0", "0", "0", "0", "4")
	c := only(t, captures)
	var undecoded int
	for _, f := range c.failures {
		if strings.Contains(f, "undecoded CUPTI activity kind") {
			undecoded++
		}
	}
	if undecoded == 0 {
		t.Fatalf("producer skipped a kind it cannot decode and said nothing; failures = %q", c.failures)
	}
	// The kind is named, not merely counted. A failure that says only "an
	// activity was dropped" tells a reader nothing they can act on, and the
	// whole point of the arm is to name what the build did not understand.
	if !strings.Contains(c.failures[0], "9000000") {
		t.Errorf("failure %q does not name the kind that was dropped", c.failures[0])
	}
	// The record is skipped either way -- there is nothing to decode -- so the
	// arm must not fabricate an activity to go with its complaint.
	if len(c.memsets) != 0 {
		t.Errorf("got %d memset records from a run that offered none", len(c.memsets))
	}
	if !status.Exited() || status.ExitStatus() != 0 {
		t.Errorf("status = %v, want clean exit: an unknown kind is reported, not fatal", status)
	}
}

// clockResult is the clock scenario's line. The stub numbers the CUPTI calls
// it receives, so the ordering CUPTI's timestamp-callback contract turns on is
// read back as two sequence numbers rather than inferred from a journal that
// looks identical either way.
type clockResult struct {
	TimestampRegisteredAt uint   `json:"timestamp_registered_at"`
	FirstEnableAt         uint   `json:"first_enable_at"`
	CallbackNanos         uint64 `json:"callback_ns"`
	MonotonicNanos        uint64 `json:"monotonic_ns"`
	RealtimeNanos         uint64 `json:"realtime_ns"`

	// Capability is what the producer told the journal about its clocks. The
	// harness line says what the stub saw; this says what a consumer is told.
	Capability capabilityRecord `json:"-"`

	// Failures is every failure record the run wrote, so an arm can assert
	// both that a refusal is reported and that an unsupported runtime is not
	// reported as one.
	Failures []string `json:"-"`
}

// runClock executes the clock scenario against a specific harness build.
func runClock(t *testing.T, bin string, env []string) clockResult {
	t.Helper()
	dir := t.TempDir()
	cmd := exec.Command(bin, "clock")
	cmd.Env = append(os.Environ(), "CUDAPROF_TRANSPORT="+dir, "CUDAPROF_SESSION="+sessionHex)
	cmd.Env = append(cmd.Env, env...)
	out, err := cmd.Output()
	if err != nil {
		t.Fatalf("clock scenario: %v", err)
	}
	var got clockResult
	line := bytes.TrimSpace(out)
	if i := bytes.LastIndexByte(line, '\n'); i >= 0 {
		line = line[i+1:]
	}
	if err := json.Unmarshal(line, &got); err != nil {
		t.Fatalf("decode clock line %q: %v", line, err)
	}
	c := only(t, decodeAll(t, dir, binRevision(bin)))
	got.Capability = c.capability
	got.Failures = c.failures
	return got
}

// TestTimestampCallbackPrecedesEveryEnable is the control for the producer's
// one clock.
//
// The journal carries two timestamp origins. A kernel or memcpy record's
// header carries CUPTI's timestamp; a capture-range edge, a failure and the
// capability record carry one the producer took itself, from CLOCK_MONOTONIC.
// CUPTI's platform default on Linux is CLOCK_REALTIME, so those two are
// separated by the wall-clock time the host booted: a consumer subtracting one
// from the other gets decades, and nothing in the record says so. Registering
// a timestamp callback is what removes that difference.
//
// Two things have to hold and neither implies the other:
//
//   - The callback must read the producer's own clock. A registration of some
//     other monotonic-looking source would satisfy any "did we register one"
//     check while leaving the offset in place, so the control calls what was
//     registered and reads the result against this process's CLOCK_MONOTONIC
//     and CLOCK_REALTIME. Those two are separated by boot time; no scheduling
//     delay comes near that, so the comparison is not a timing race.
//
//   - The registration must precede the first cuptiActivityEnable. NVIDIA
//     documents that records produced before it keep CUPTI's previous
//     timestamp method, so a late registration yields a journal straddling two
//     bases. That is an ordering property: a journal from a late registration
//     is byte-identical here, and differs in the field only by an offset no
//     field names. So the stub numbers its calls and this asserts the order.
func TestTimestampCallbackPrecedesEveryEnable(t *testing.T) {
	got := runClock(t, controlBinary(t), nil)

	if got.TimestampRegisteredAt == 0 {
		t.Fatalf("producer registered no timestamp callback: %+v", got)
	}
	if got.FirstEnableAt == 0 {
		t.Fatalf("producer enabled no activity kind, so the ordering is vacuous: %+v", got)
	}
	if got.TimestampRegisteredAt >= got.FirstEnableAt {
		t.Errorf("timestamp callback registered at call %d, first cuptiActivityEnable at call %d; registration must come first or earlier records keep CUPTI's previous clock",
			got.TimestampRegisteredAt, got.FirstEnableAt)
	}

	// The registered callback must be on the producer's clock. A window this
	// wide cannot mistake CLOCK_MONOTONIC for CLOCK_REALTIME: the gap between
	// them is the host's uptime offset from the epoch, currently decades.
	const window = 60 * 1e9
	if diff := absDiff(got.CallbackNanos, got.MonotonicNanos); diff > window {
		t.Errorf("registered callback returned %d, %d ns from this process's CLOCK_MONOTONIC (%d); want the same clock",
			got.CallbackNanos, diff, got.MonotonicNanos)
	}
	// The discriminating half: the same reading must NOT match CUPTI's own
	// default. Without this the assertion above would pass on a host where the
	// two clocks happened to agree, and the defect being fixed is precisely
	// that they do not.
	if diff := absDiff(got.CallbackNanos, got.RealtimeNanos); diff <= window {
		t.Errorf("registered callback returned %d, only %d ns from CLOCK_REALTIME (%d); the two clocks are supposed to differ by boot time, so this fixture cannot tell them apart",
			got.CallbackNanos, diff, got.RealtimeNanos)
	}

	if !got.Capability.ClocksUnified {
		t.Errorf("capability record reports clocks_unified=false on a run that unified them: %+v", got.Capability)
	}
	if got.Capability.CUPTIClock != "monotonic" {
		t.Errorf("capability cupti_clock = %q, want %q", got.Capability.CUPTIClock, "monotonic")
	}
}

func absDiff(a, b uint64) uint64 {
	if a > b {
		return a - b
	}
	return b - a
}

// TestTimestampCallbackDegradesVisibly is the control for the other half of
// the contract: a CUPTI that will not unify the clocks must still produce a
// capture, and must not let the consumer read "no error" as "one clock".
//
// Both refusals are covered because they are different failures with the same
// consequence. A runtime predating the API exports no symbol, so there is no
// call and no CUptiResult. A runtime that exports it and rejects the callback
// answers a call this producer made, and is recorded the way every other
// refused CUPTI call is. Either way the journal has to arrive and the
// capability record has to say the timestamps are not comparable.
func TestTimestampCallbackDegradesVisibly(t *testing.T) {
	tests := []struct {
		name      string
		defines   []string
		env       []string
		clock     string
		wantFails int
	}{{
		name:    "symbol absent",
		defines: []string{"-DCUDAPROF_TEST_NO_TIMESTAMP_CALLBACK"},
		clock:   "platform_default_callback_unsupported",
		// No call was made, so there is no CUPTI result to record.
		wantFails: 0,
	}, {
		name:  "cupti refuses",
		env:   []string{"CUDAPROF_TEST_REFUSE_TIMESTAMP=1"},
		clock: "platform_default_callback_refused",
		// A refused call is recorded, and names the operation.
		wantFails: 1,
	}}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bin := controlBinary(t, tt.defines...)
			got := runClock(t, bin, tt.env)

			if got.TimestampRegisteredAt != 0 {
				t.Fatalf("a refusing CUPTI still recorded a registration at call %d: %+v", got.TimestampRegisteredAt, got)
			}
			if got.FirstEnableAt == 0 {
				t.Errorf("degraded run enabled no activity kind; the capture was lost, not degraded: %+v", got)
			}
			if got.Capability.ClocksUnified {
				t.Errorf("capability record claims clocks_unified=true after the registration failed: %+v", got.Capability)
			}
			if got.Capability.CUPTIClock != tt.clock {
				t.Errorf("capability cupti_clock = %q, want %q", got.Capability.CUPTIClock, tt.clock)
			}
			var clockFails int
			for _, f := range got.Failures {
				if strings.Contains(f, "cuptiActivityRegisterTimestampCallback") {
					clockFails++
				}
			}
			if clockFails != tt.wantFails {
				t.Errorf("got %d failure records naming the registration, want %d: %q", clockFails, tt.wantFails, got.Failures)
			}
			// The capture still has to be a capture: everything the producer
			// reports independently of the clocks must be unchanged.
			if !got.Capability.Activity || !got.Capability.APIRecords {
				t.Errorf("degraded run lost collection it was not asked to give up: %+v", got.Capability)
			}
		})
	}
}

// TestTimestampCallbackDegradationIsRecorded checks the failure record that
// accompanies a refusal, separately from the capability record, because a
// consumer reading only one of the two must not be misled by the other.
func TestTimestampCallbackDegradationIsRecorded(t *testing.T) {
	dir := t.TempDir()
	cmd := exec.Command(controlBinary(t), "clock")
	cmd.Env = append(os.Environ(), "CUDAPROF_TRANSPORT="+dir, "CUDAPROF_SESSION="+sessionHex,
		"CUDAPROF_TEST_REFUSE_TIMESTAMP=1")
	if _, err := cmd.Output(); err != nil {
		t.Fatalf("clock scenario: %v", err)
	}
	c := only(t, decodeAll(t, dir, binRevision(controlBinary(t))))
	var found string
	for _, f := range c.failures {
		if strings.Contains(f, "cuptiActivityRegisterTimestampCallback") {
			found = f
		}
	}
	if found == "" {
		t.Fatalf("no failure record names the refused registration: %q", c.failures)
	}
	if !strings.Contains(found, `"state":"cupti_unavailable"`) {
		t.Errorf("failure record %q does not carry the cupti_unavailable state", found)
	}
}

// ---------------------------------------------------------------------------
// Host run-state sampling.
//
// The sampler reads a directory of task stat files. Its default root is
// /proc/self/task, which exists only on Linux, so these controls point it at a
// fixture tree instead. That is not a mock of the sampler: the producer thread
// under test is the shipping one, it enumerates with the shipping code, parses
// with the shipping parser and emits through the shipping ring and drain. What
// the fixture replaces is the kernel, and only the kernel.
//
// NOT MEASURED here, and not measurable on this host: the cost of the sampler
// against a real /proc on a real Linux target. Every timing claim in these
// controls is about a fixture tree on darwin.

// hostFixture writes a task tree the sampler can read: one directory per tid,
// each holding a stat file with the exact bytes given.
func hostFixture(t *testing.T, stats map[string]string) string {
	t.Helper()
	root := t.TempDir()
	for name, text := range stats {
		dir := filepath.Join(root, name)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			t.Fatal(err)
		}
		if text == "" {
			continue // a task directory whose stat file cannot be read
		}
		if err := os.WriteFile(filepath.Join(dir, "stat"), []byte(text), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

// hostEnv is the producer configuration that turns sampling on against root.
func hostEnv(hz int, root string) []string {
	return []string{
		fmt.Sprintf("CUDAPROF_HOST_SAMPLE=%d", hz),
		"CUDAPROF_HOST_SAMPLE_ROOT=" + root,
	}
}

// Real /proc/<pid>/stat lines. The comm fields are the point of two of them:
// comm is the thread name, the kernel does not escape it, and it may contain
// both spaces and parentheses. "(sd-pam)" ships with systemd; any program can
// prctl its name to anything. A parser that scans "%d (%s) %c", or splits on
// whitespace, or takes the FIRST ')' as the end of comm, reads the wrong
// fields for these two lines -- and reads them as plausible numbers rather
// than failing, so nothing downstream can tell.
const (
	statPlain     = "1234 (python3) R 1 1234 1234 0 -1 4194304 12345 0 0 0 512 128 0 0 20 0 8 0 987654 1234567890 4567 18446744073709551615 1 1 0 0 0 0 0\n"
	statParenComm = "4102 (sd) pam (x) D 1 1234 1234 0 -1 1077936128 900 0 0 0 7 9 0 0 20 0 8 0 987655 1 2 3\n"
	statSpaceComm = "4103 (cuda EvtHandlr) S 1 1234 1234 0 -1 1077936128 3 0 0 0 0 1 0 0 20 0 8 0 987656 1 2 3\n"
	// The adversarial case, and the reason the other two are not enough. A
	// parser that stops at the FIRST ')' reads state 'b' and then twelve
	// perfectly well-formed integers out of the middle of the comm, so it
	// returns success with utime 11 and stime 12 -- a plausible thread that
	// never existed. Nothing downstream can tell. Only asserting the values
	// catches it; asserting that the line parsed does not.
	statDecoyComm = "4104 (a) b 1 2 3 4 5 6 7 8 9 10 11 12 (z) T 1 1234 1234 0 -1 1077936128 3 0 0 0 44 55 0 0 20 0 8 0 987657 1 2 3\n"
)

// TestHostSamplesReachTheJournal covers the tier end to end: configuration,
// the sampling thread, the /proc stat parse, the record the drain writes, and
// the producer's own accounting for it.
//
// The assertions are on the parsed values, not on a record count. A sample
// that arrives with the right shape and the wrong fields is worse than no
// sample: it answers "what was the host doing in this device gap" with a
// confident wrong answer, and the gap analysis has no second witness.
func TestHostSamplesReachTheJournal(t *testing.T) {
	root := hostFixture(t, map[string]string{
		"1234": statPlain,
		"4102": statParenComm,
		"4103": statSpaceComm,
		"4104": statDecoyComm,
	})
	_, status, captures := runEnv(t, hostEnv(200, root), "host-sample", "120")
	c := only(t, captures)

	if c.capability.HostSample != "sampling" {
		t.Fatalf("capability host_sample = %q, want \"sampling\": the sampler never ran, so every assertion below would be about an empty set", c.capability.HostSample)
	}
	if c.capability.HostSampleHz != 200 {
		t.Errorf("capability host_sample_hz = %d, want 200", c.capability.HostSampleHz)
	}
	// A fixture run and a real one must never be indistinguishable: the root
	// that was read is named, and the fact that it was overridden is its own
	// field rather than something a reader has to infer from the path.
	if !c.capability.RootOverride || c.capability.HostSampleRoot != root {
		t.Errorf("capability root = %q override = %v, want %q and true", c.capability.HostSampleRoot, c.capability.RootOverride, root)
	}
	// The counters are in ticks. Without the rate a consumer has to assume
	// 100, and a host configured otherwise makes every derived CPU-time
	// figure wrong by a constant nobody can see.
	if c.capability.ClockTicks <= 0 {
		t.Errorf("capability clock_ticks_per_second = %d, want a positive tick rate", c.capability.ClockTicks)
	}
	if len(c.hosts) == 0 {
		t.Fatal("no host sample records in the journal")
	}

	want := map[uint64]hostThreadState{
		1234: {TID: 1234, State: "R", UTime: 512, STime: 128},
		4102: {TID: 4102, State: "D", UTime: 7, STime: 9},
		4103: {TID: 4103, State: "S", UTime: 0, STime: 1},
		4104: {TID: 4104, State: "T", UTime: 44, STime: 55},
	}
	for i, h := range c.hosts {
		if h.Kind != "host_sample" {
			t.Errorf("record %d: kind=%q, want \"host_sample\"", i, h.Kind)
		}
		if !h.Last || h.Part != 0 {
			t.Errorf("record %d: part=%d last=%v, want a single complete part for four threads", i, h.Part, h.Last)
		}
		if h.ThreadsTotal != 4 || h.InPart != 4 {
			t.Errorf("record %d: in_part=%d threads_total=%d, want 4 and 4", i, h.InPart, h.ThreadsTotal)
		}
		if h.Unreadable != 0 || h.Overflowed != 0 {
			t.Errorf("record %d: unreadable=%d overflowed=%d, want 0 and 0 from a fixture every entry of which parses", i, h.Unreadable, h.Overflowed)
		}
		if h.Wall == 0 {
			t.Errorf("record %d has no timestamp, so it cannot be placed against a device gap", i)
		}
		// Not fatal: a wrong count and a wrong value are different defects
		// and a parser can produce either, so the value comparison below has
		// to run even when the count is already wrong.
		if len(h.Threads) != 4 {
			t.Errorf("record %d carries %d threads, want 4", i, len(h.Threads))
		}
		for _, got := range h.Threads {
			w, ok := want[got.TID]
			if !ok {
				t.Errorf("record %d reports tid %d, which is not in the fixture: the parser read the wrong leading field", i, got.TID)
				continue
			}
			if got != w {
				t.Errorf("record %d tid %d = %+v, want %+v", i, got.TID, got, w)
			}
		}
	}
	// Samples advance. A sampler that ran once and then wedged, or that
	// re-emitted one sample forever, produces records that all look right.
	if len(c.hosts) > 1 && c.hosts[len(c.hosts)-1].Sample == c.hosts[0].Sample {
		t.Errorf("every record carries sample index %d: the sampler took one sample and repeated it", c.hosts[0].Sample)
	}
	fin := c.finalized
	if fin == nil {
		t.Fatal("no finalized record")
	}
	if fin.HostSamples == 0 || fin.HostSampleRecords != uint64(len(c.hosts)) {
		t.Errorf("finalized reports %d samples in %d records, journal holds %d", fin.HostSamples, fin.HostSampleRecords, len(c.hosts))
	}
	checkReconciles(t, c, fin.Offered)
	if len(c.failures) != 0 {
		t.Errorf("producer reported failures: %q", c.failures)
	}
	if !status.Exited() || status.ExitStatus() != 0 {
		t.Errorf("status = %v, want clean exit", status)
	}
}

// TestHostSampleOrdering asserts the invariant the consumer enforces: a record
// that arrives before bootstrap and capability is rejected, and the whole
// capture fails with it.
//
// The sampler runs on its own timer, so nothing about its first record is
// naturally ordered against the constructor's two. The producer starts the
// thread only after both records have taken their sequence numbers, and this
// is what says so -- from the journal's own order, not from reading the
// constructor.
func TestHostSampleOrdering(t *testing.T) {
	root := hostFixture(t, map[string]string{"1234": statPlain})
	_, _, captures := runEnv(t, hostEnv(500, root), "host-sample", "80")
	c := only(t, captures)
	if len(c.hosts) == 0 {
		t.Fatal("no host sample records in the journal")
	}
	first, bootstrap, capability, finalized := -1, -1, -1, -1
	for i, kind := range c.order {
		switch kind {
		case kindHost:
			if first < 0 {
				first = i
			}
		case kindBootstrap:
			bootstrap = i
		case kindCapability:
			capability = i
		case kindFinalized:
			finalized = i
		}
	}
	if bootstrap < 0 || capability < 0 || finalized < 0 {
		t.Fatalf("journal order %v is missing one of bootstrap/capability/finalized", c.order)
	}
	if first < bootstrap || first < capability {
		t.Errorf("first host sample is at index %d, bootstrap at %d and capability at %d: a consumer rejects the whole capture for this", first, bootstrap, capability)
	}
	last := -1
	for i, kind := range c.order {
		if kind == kindHost {
			last = i
		}
	}
	if last > finalized {
		t.Errorf("a host sample at index %d follows the finalized record at %d: the sampler outlived finalization", last, finalized)
	}
}

// TestHostSampleSplitsLargeThreadSets covers the payload cap. A record is
// bounded at MAX_PAYLOAD and a serving worker can hold hundreds of threads, so
// one sample becomes several records. What must not happen is a silent
// truncation: a sample reporting 63 of 200 threads is not a smaller sample, it
// is a wrong one, and nothing in a per-record view would show it.
func TestHostSampleSplitsLargeThreadSets(t *testing.T) {
	const threads = 200
	stats := make(map[string]string, threads)
	for i := 0; i < threads; i++ {
		tid := 5000 + i
		stats[fmt.Sprint(tid)] = fmt.Sprintf("%d (worker %d) S 1 1234 1234 0 -1 1077936128 3 0 0 0 %d %d 0 0 20 0 8 0 987656 1 2 3\n", tid, i, i, i*2)
	}
	root := hostFixture(t, stats)
	_, status, captures := runEnv(t, hostEnv(50, root), "host-sample", "150")
	c := only(t, captures)
	if len(c.hosts) == 0 {
		t.Fatal("no host sample records in the journal")
	}

	bySample := map[uint64][]hostSampleRecord{}
	for _, h := range c.hosts {
		bySample[h.Sample] = append(bySample[h.Sample], h)
	}
	split := 0
	for index, parts := range bySample {
		if len(parts) > 1 {
			split++
		}
		seen := map[uint64]bool{}
		var inPart int
		for i, p := range parts {
			if p.Part != uint(i) {
				t.Errorf("sample %d: part %d carries index %d, so a consumer cannot tell which part is missing", index, i, p.Part)
			}
			if p.ThreadsTotal != threads {
				t.Errorf("sample %d part %d: threads_total=%d, want %d on every part", index, p.Part, p.ThreadsTotal, threads)
			}
			if got := p.Last; got != (i == len(parts)-1) {
				t.Errorf("sample %d part %d: last=%v at position %d of %d", index, p.Part, got, i, len(parts))
			}
			if p.InPart != len(p.Threads) {
				t.Errorf("sample %d part %d: in_part=%d but carries %d threads", index, p.Part, p.InPart, len(p.Threads))
			}
			inPart += p.InPart
			for _, th := range p.Threads {
				if seen[th.TID] {
					t.Errorf("sample %d reports tid %d twice across parts", index, th.TID)
				}
				seen[th.TID] = true
			}
		}
		// The identity that makes the split lossless: the parts of one sample
		// account for exactly the threads the sample claims, once each.
		if !parts[len(parts)-1].Last {
			t.Errorf("sample %d has no part marked last, so a consumer can never know it is complete", index)
		}
		if inPart != threads || len(seen) != threads {
			t.Errorf("sample %d: parts carry %d threads (%d distinct), want %d", index, inPart, len(seen), threads)
		}
		if p := parts[0]; p.Overflowed != 0 {
			t.Errorf("sample %d dropped %d threads: %d is under the sampler's own bound", index, p.Overflowed, threads)
		}
	}
	if split == 0 {
		t.Fatalf("%d threads fit in a single record: this control asserts nothing about batching", threads)
	}
	// The producer distinguishes intervals from records precisely so this is
	// checkable rather than inferred.
	fin := c.finalized
	if fin == nil {
		t.Fatal("no finalized record")
	}
	if fin.HostSampleRecords <= fin.HostSamples {
		t.Errorf("finalized reports %d records for %d samples; a split sample must cost more records than intervals", fin.HostSampleRecords, fin.HostSamples)
	}
	checkReconciles(t, c, fin.Offered)
	if !status.Exited() || status.ExitStatus() != 0 {
		t.Errorf("status = %v, want clean exit", status)
	}
}

// TestHostSampleCountsUnreadableTasks covers what a real /proc does constantly:
// a thread exits between the directory scan and the open, and its stat file is
// gone. That is the ordinary case, not a failure, but it must be counted --
// otherwise a sample missing half the process looks exactly like a process
// with half as many threads.
func TestHostSampleCountsUnreadableTasks(t *testing.T) {
	root := hostFixture(t, map[string]string{
		"1234": statPlain,
		"4102": statParenComm,
		"7000": "",                  // a task directory with no stat file
		"7001": "not a stat line\n", // present but unparseable
	})
	_, status, captures := runEnv(t, hostEnv(200, root), "host-sample", "100")
	c := only(t, captures)
	if len(c.hosts) == 0 {
		t.Fatal("no host sample records in the journal")
	}
	for i, h := range c.hosts {
		if h.ThreadsTotal != 2 {
			t.Errorf("record %d: threads_total=%d, want the 2 tasks that parsed", i, h.ThreadsTotal)
		}
		if h.Unreadable != 2 {
			t.Errorf("record %d: unreadable=%d, want 2 -- a task that vanishes mid-scan is counted, not silently dropped", i, h.Unreadable)
		}
		// The unparseable line must not become a thread with invented fields.
		for _, th := range h.Threads {
			if th.TID != 1234 && th.TID != 4102 {
				t.Errorf("record %d reports tid %d, which no readable fixture entry names", i, th.TID)
			}
		}
	}
	if !status.Exited() || status.ExitStatus() != 0 {
		t.Errorf("status = %v, want clean exit: an unreadable task is counted, not fatal", status)
	}
}

// TestHostSampleOffByDefault. The cost of this tier against a real /proc on a
// real Linux target is NOT MEASURED -- it cannot be measured from this host --
// so it is off unless asked for, and the capability record says which of
// "off", "sampling" and a refusal this run was. An absent host record must
// never be ambiguous between "not asked for" and "asked for and broken".
func TestHostSampleOffByDefault(t *testing.T) {
	_, status, captures := run(t, "host-sample", "40")
	c := only(t, captures)
	if c.capability.HostSample != "off" || c.capability.HostSampleHz != 0 {
		t.Errorf("capability host_sample=%q hz=%d, want \"off\" and 0", c.capability.HostSample, c.capability.HostSampleHz)
	}
	if c.capability.RootOverride {
		t.Errorf("capability claims a root override on a run that configured none")
	}
	if n := c.counts[kindHost]; n != 0 {
		t.Errorf("%d host records from a run that never enabled sampling", n)
	}
	if fin := c.finalized; fin == nil || fin.HostSamples != 0 {
		t.Errorf("finalized reports host samples on a run that never enabled sampling: %+v", fin)
	}
	if !status.Exited() || status.ExitStatus() != 0 {
		t.Errorf("status = %v, want clean exit", status)
	}
}

// TestHostSampleRefusesUnreadableRoot. A root that cannot be opened is the
// darwin default and any misconfiguration, and the failure mode to avoid is a
// sampler that retries it every interval for the life of the process. It is
// settled once, at configure time, and reported -- rather than producing a
// journal with no host records that looks the same as sampling being off.
func TestHostSampleRefusesUnreadableRoot(t *testing.T) {
	missing := filepath.Join(t.TempDir(), "no-such-task-tree")
	_, status, captures := runEnv(t, hostEnv(100, missing), "host-sample", "60")
	c := only(t, captures)
	if c.capability.HostSample != "root_unavailable" {
		t.Errorf("capability host_sample = %q, want \"root_unavailable\"", c.capability.HostSample)
	}
	if c.capability.HostSampleHz != 0 {
		t.Errorf("capability host_sample_hz = %d, want 0: the rate was asked for but no sampler runs", c.capability.HostSampleHz)
	}
	if n := c.counts[kindHost]; n != 0 {
		t.Errorf("%d host records from a root that does not exist", n)
	}
	if !status.Exited() || status.ExitStatus() != 0 {
		t.Errorf("status = %v, want clean exit: a missing root disables the tier, it does not fail the capture", status)
	}
}

// overheadRecord is one CUPTI_ACTIVITY_KIND_OVERHEAD activity: CUPTI's own
// cost as a timestamped span. Note what it does not have: a device. The
// producer transcribes the object identity CUPTI gave it, which for a buffer
// flush is a host thread, and ObjectKind is what says whether Device means
// anything at all.
type overheadRecord struct {
	Subtype      string `json:"subtype"`
	Valid        bool   `json:"valid"`
	Start        uint64 `json:"start"`
	End          uint64 `json:"end"`
	OverheadKind uint32 `json:"overhead_kind"`
	ObjectKind   uint32 `json:"object_kind"`
	Device       uint32 `json:"device"`
	Context      uint32 `json:"context"`
	Stream       uint32 `json:"stream"`
	ProcessID    uint32 `json:"process_id"`
	ThreadID     uint32 `json:"thread_id"`
}

// graphTraceRecord is one CUPTI_ACTIVITY_KIND_GRAPH_TRACE activity: the span
// of one graph replay, recorded in graph trace mode only. CUPTI reports it
// instead of the node kernels inside the replay, not beside them.
type graphTraceRecord struct {
	Subtype     string `json:"subtype"`
	Valid       bool   `json:"valid"`
	Start       uint64 `json:"start"`
	End         uint64 `json:"end"`
	Device      uint32 `json:"device"`
	Context     uint32 `json:"context"`
	Stream      uint32 `json:"stream"`
	Correlation uint32 `json:"correlation"`
	GraphID     uint32 `json:"graph_id"`
}

// graphNodeKernel is a kernel record that names the graph node it ran as.
type graphNodeKernel struct {
	ModuleID    uint32 `json:"module_id"`
	Name        string `json:"name"`
	Correlation uint32 `json:"correlation"`
	GraphID     uint32 `json:"graph_id"`
	GraphNodeID uint64 `json:"graph_node_id"`
}

// syncRecord is one CUPTI_ACTIVITY_KIND_SYNCHRONIZATION activity: the window
// a stream, event or context synchronize was open. It names a context and a
// stream and no device, which is the whole reason the consumer has to resolve
// one rather than read one.
type syncRecord struct {
	Subtype     string `json:"subtype"`
	Valid       bool   `json:"valid"`
	Start       uint64 `json:"start"`
	End         uint64 `json:"end"`
	SyncType    uint32 `json:"sync_type"`
	CudaEventID uint32 `json:"cuda_event_id"`
	Context     uint32 `json:"context"`
	Stream      uint32 `json:"stream"`
	Correlation uint32 `json:"correlation"`
}

// TestOverheadsReachTheJournal covers CUPTI_ACTIVITY_KIND_OVERHEAD end to end.
//
// The kind is the tool's own cost, expressed as timestamped spans: buffer
// flushes, instrumentation pauses, lazily loaded modules. Until it was
// collected, every one of those intervals was indistinguishable from the
// device having nothing to do, so `cudaprof gaps` charged the time the tool
// spent to the program it was measuring. That is a distortion in the tool's
// headline number, in the direction that flatters no one: a run profiled more
// heavily reports a worse duty cycle.
//
// The assertion on ObjectKind is the substantive one. An overhead record has
// no deviceId at all; what it has is an object, and for CUPTI's own overhead
// that object is a host thread. A producer that read Device off one of these
// would emit a zero, and a consumer that trusted it would charge every
// overhead in the process to device 0.
func TestOverheadsReachTheJournal(t *testing.T) {
	// argv: scenario threads buffers per_buffer pause_micros _ markers memsets undecoded overheads
	_, status, captures := run(t, "sustained", "1", "1", "8", "0", "0", "0", "0", "0", "5")
	c := only(t, captures)
	if !c.capability.Overheads {
		t.Fatal("capability reports overheads = false: the producer never enabled CUPTI_ACTIVITY_KIND_OVERHEAD, so every assertion below would be about an empty set")
	}
	if len(c.overheads) != 5 {
		t.Fatalf("got %d overhead records, want 5", len(c.overheads))
	}
	for i, o := range c.overheads {
		if o.Subtype != "overhead" {
			t.Errorf("overhead %d: subtype=%q, want \"overhead\"", i, o.Subtype)
		}
		if !o.Valid {
			t.Errorf("overhead %d: valid=false, want true: the stub wrote a well-ordered span", i)
		}
		// The span is the point. An overhead carried with start == end would
		// leave the idle arithmetic downstream exactly where it was.
		if got := o.End - o.Start; got != 70 {
			t.Errorf("overhead %d spans %d..%d (%d ns), want 70 as the stub wrote it", i, o.Start, o.End, got)
		}
		// 1<<16 is CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH. Without the
		// kind a reader cannot say which overhead a gap was, and "the tool
		// cost you something here" is not actionable; "the tool was flushing
		// its activity buffers here" is.
		if o.OverheadKind != 1<<16 {
			t.Errorf("overhead %d: overhead_kind=%d, want %d (CUPTI buffer flush): the kind was dropped, leaving an unattributable span", i, o.OverheadKind, 1<<16)
		}
		// 2 is CUPTI_ACTIVITY_OBJECT_THREAD.
		if o.ObjectKind != 2 {
			t.Errorf("overhead %d: object_kind=%d, want 2 (thread)", i, o.ObjectKind)
		}
		if o.ProcessID != 4242 || o.ThreadID != 99 {
			t.Errorf("overhead %d: process/thread = %d/%d, want 4242/99 from the stub record", i, o.ProcessID, o.ThreadID)
		}
		// A thread-scoped overhead has no device, and the producer must leave
		// the field empty rather than fill it with the zero that a consumer
		// would read as device 0.
		if o.Device != 0 || o.Context != 0 || o.Stream != 0 {
			t.Errorf("overhead %d: device/context/stream = %d/%d/%d, want all zero: the record is scoped to a thread and names no device", i, o.Device, o.Context, o.Stream)
		}
	}
	if len(c.failures) != 0 {
		t.Errorf("producer reported failures: %q", c.failures)
	}
	if !status.Exited() || status.ExitStatus() != 0 {
		t.Errorf("status = %v, want clean exit", status)
	}
}

// TestGraphTracesReachTheJournal covers CUPTI_ACTIVITY_KIND_GRAPH_TRACE end to
// end in graph trace mode, where it is enabled.
//
// The envelope is the span a replay as a whole occupied, and in this mode it
// is the only record of the replay: CUPTI does not deliver the kernels inside
// it. Without the envelope a graph-mode capture would hold no trace of its
// graph work at all.
func TestGraphTracesReachTheJournal(t *testing.T) {
	// argv: ... markers memsets undecoded overheads graph_traces
	_, status, captures := runEnv(t, []string{"CUDAPROF_GRAPH_TRACE=graph"}, "sustained", "1", "1", "8", "0", "0", "0", "0", "0", "0", "4")
	c := only(t, captures)
	if !c.capability.GraphTraces {
		t.Fatal("capability reports graph_traces = false: the producer never enabled CUPTI_ACTIVITY_KIND_GRAPH_TRACE, so every assertion below would be about an empty set")
	}
	if len(c.graphs) != 4 {
		t.Fatalf("got %d graph_trace records, want 4", len(c.graphs))
	}
	for i, g := range c.graphs {
		if g.Subtype != "graph_trace" {
			t.Errorf("graph_trace %d: subtype=%q, want \"graph_trace\"", i, g.Subtype)
		}
		if !g.Valid {
			t.Errorf("graph_trace %d: valid=false, want true", i)
		}
		if got := g.End - g.Start; got != 90 {
			t.Errorf("graph_trace %d spans %d..%d (%d ns), want 90 as the stub wrote it", i, g.Start, g.End, got)
		}
		// The graph id is what joins the envelope to the node records that
		// ran inside it. An envelope with graph id 0 is a span belonging to
		// no graph, which is worth nothing to a reader holding node records.
		if g.GraphID != 5 {
			t.Errorf("graph_trace %d: graph_id=%d, want 5: the envelope cannot be joined to the nodes it replayed", i, g.GraphID)
		}
		if g.Context != 1 || g.Stream != 7 {
			t.Errorf("graph_trace %d: context/stream = %d/%d, want 1/7", i, g.Context, g.Stream)
		}
		if g.Correlation == 0 {
			t.Errorf("graph_trace %d has correlation 0, so it can never be joined to the launch that replayed it", i)
		}
	}
	// Unlike an overhead or a sync, a graph trace does carry a device id, and
	// the stub alternates it: a producer that dropped the field would put
	// every replay on device 0.
	var devices [2]int
	for _, g := range c.graphs {
		if g.Device > 1 {
			t.Fatalf("graph_trace device %d outside the stub's 0/1 alternation", g.Device)
		}
		devices[g.Device]++
	}
	if devices[0] == 0 || devices[1] == 0 {
		t.Errorf("graph_trace devices = %v, want both: the device id the record carries was not transcribed", devices)
	}
	if len(c.failures) != 0 {
		t.Errorf("producer reported failures: %q", c.failures)
	}
	if !status.Exited() || status.ExitStatus() != 0 {
		t.Errorf("status = %v, want clean exit", status)
	}
}

// cuda13Build is the harness built against the CUDA 13 kernel layout, the
// which also carries graphId and graphNodeId.
var cuda13Build = []string{"-DCUPTI_API_VERSION=130202"}

// graphReplays is the done-test's workload: a graph of three kernel nodes
// replayed five times.
const (
	graphNodes   = 3
	graphReplays = 5
)

// TestGraphTraceModeNodeResolvesNodes covers node mode, the default: with
// CUPTI_ACTIVITY_KIND_GRAPH_TRACE left off, every node of every replay reaches
// the journal as a kernel record naming its graph and node, and no envelope
// does. A value other than "graph" is node mode too, and says so.
func TestGraphTraceModeNodeResolvesNodes(t *testing.T) {
	tests := []struct {
		name string
		env  []string
	}{
		{"unset", nil},
		{"node", []string{"CUDAPROF_GRAPH_TRACE=node"}},
		{"unknown", []string{"CUDAPROF_GRAPH_TRACE=envelope"}},
	}
	bin := controlBinary(t, cuda13Build...)
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, status, captures := runBuilt(t, bin, tc.env, "graph-replay", strconv.Itoa(graphNodes), strconv.Itoa(graphReplays))
			c := only(t, captures)
			if got := c.capability.GraphTraceMode; got != "node" {
				t.Errorf("capability graph_trace_mode = %q, want \"node\"", got)
			}
			if c.capability.GraphTraces {
				t.Error("capability graph_traces = true in node mode: the envelope kind was enabled, and it hides the nodes")
			}
			if len(c.graphs) != 0 {
				t.Errorf("got %d graph_trace records in node mode, want 0", len(c.graphs))
			}
			if got, want := len(c.graphNodes), graphNodes*graphReplays; got != want {
				t.Fatalf("got %d kernel records naming a graph node, want %d (%d nodes x %d replays)", got, want, graphNodes, graphReplays)
			}
			perNode := map[uint64]int{}
			perReplay := map[uint32]int{}
			for _, k := range c.graphNodes {
				if k.GraphID != 5 {
					t.Errorf("kernel %q: graph_id = %d, want 5", k.Name, k.GraphID)
				}
				perNode[k.GraphNodeID]++
				perReplay[k.Correlation]++
			}
			for id := uint64(1); id <= graphNodes; id++ {
				if perNode[id] != graphReplays {
					t.Errorf("graph node %d appears %d times, want %d", id, perNode[id], graphReplays)
				}
			}
			if len(perReplay) != graphReplays {
				t.Errorf("node kernels carry %d distinct correlations, want one per replay (%d)", len(perReplay), graphReplays)
			}
			if len(c.failures) != 0 {
				t.Errorf("producer reported failures: %q", c.failures)
			}
			if !status.Exited() || status.ExitStatus() != 0 {
				t.Errorf("status = %v, want clean exit", status)
			}
		})
	}
}

// TestGraphTraceModeGraphReportsEnvelopes covers graph mode: one envelope per
// replay and no node kernels, which is what CUPTI delivers with
// CUPTI_ACTIVITY_KIND_GRAPH_TRACE enabled.
func TestGraphTraceModeGraphReportsEnvelopes(t *testing.T) {
	bin := controlBinary(t, cuda13Build...)
	_, status, captures := runBuilt(t, bin, []string{"CUDAPROF_GRAPH_TRACE=graph"}, "graph-replay", strconv.Itoa(graphNodes), strconv.Itoa(graphReplays))
	c := only(t, captures)
	if got := c.capability.GraphTraceMode; got != "graph" {
		t.Errorf("capability graph_trace_mode = %q, want \"graph\"", got)
	}
	if !c.capability.GraphTraces {
		t.Error("capability graph_traces = false in graph mode")
	}
	if len(c.graphs) != graphReplays {
		t.Errorf("got %d graph_trace records, want %d, one per replay", len(c.graphs), graphReplays)
	}
	if len(c.graphNodes) != 0 {
		t.Errorf("got %d node kernel records in graph mode, want 0", len(c.graphNodes))
	}
	if len(c.failures) != 0 {
		t.Errorf("producer reported failures: %q", c.failures)
	}
	if !status.Exited() || status.ExitStatus() != 0 {
		t.Errorf("status = %v, want clean exit", status)
	}
}

// TestGraphTraceModeRequestedUnavailable covers a graph request the runtime
// cannot meet: its graph trace record is shorter than the producer reads. The
// capability record says the request went unmet, and collection stays in node
// mode rather than losing the graph work altogether.
func TestGraphTraceModeRequestedUnavailable(t *testing.T) {
	bin := controlBinary(t, append([]string{"-DCUDAPROF_TEST_STRUCT_SIZE_FN=cudaprof_test_mock_short_graph_trace_struct_size"}, cuda13Build...)...)
	_, status, captures := runBuilt(t, bin, []string{"CUDAPROF_GRAPH_TRACE=graph"}, "graph-replay", strconv.Itoa(graphNodes), strconv.Itoa(graphReplays))
	c := only(t, captures)
	if got := c.capability.GraphTraceMode; got != "requested_unavailable" {
		t.Errorf("capability graph_trace_mode = %q, want \"requested_unavailable\"", got)
	}
	if c.capability.GraphTraces {
		t.Error("capability graph_traces = true, but the layout check refused the kind")
	}
	if len(c.graphs) != 0 {
		t.Errorf("got %d graph_trace records, want 0", len(c.graphs))
	}
	if got, want := len(c.graphNodes), graphNodes*graphReplays; got != want {
		t.Errorf("got %d node kernel records, want %d: an unmet graph request must leave node mode in effect", got, want)
	}
	if !status.Exited() || status.ExitStatus() != 0 {
		t.Errorf("status = %v, want clean exit", status)
	}
}

// TestSynchronizationsReachTheJournal covers
// CUPTI_ACTIVITY_KIND_SYNCHRONIZATION end to end.
//
// A synchronization span is the host waiting: a stream synchronize, an event
// wait, a context synchronize. It is the single most common reason a device is
// idle, and it was absent from the journal, so the gap report could say how
// much idle there was and never why.
//
// The assertions on Context and Stream, and the absence of any assertion that
// a device was carried, are the substance. CUpti_ActivitySynchronization has
// no deviceId field. The producer therefore cannot supply one and must not
// invent one; resolving it belongs to the consumer, which has records that
// carry both a context and a device, and saying when it could not be resolved
// belongs there too.
func TestSynchronizationsReachTheJournal(t *testing.T) {
	// argv: ... markers memsets undecoded overheads graph_traces syncs
	_, status, captures := run(t, "sustained", "1", "1", "8", "0", "0", "0", "0", "0", "0", "0", "6")
	c := only(t, captures)
	if !c.capability.Synchronization {
		t.Fatal("capability reports synchronization = false: the producer never enabled CUPTI_ACTIVITY_KIND_SYNCHRONIZATION, so every assertion below would be about an empty set")
	}
	if len(c.syncs) != 6 {
		t.Fatalf("got %d sync records, want 6", len(c.syncs))
	}
	for i, y := range c.syncs {
		if y.Subtype != "sync" {
			t.Errorf("sync %d: subtype=%q, want \"sync\"", i, y.Subtype)
		}
		if !y.Valid {
			t.Errorf("sync %d: valid=false, want true", i)
		}
		if got := y.End - y.Start; got != 60 {
			t.Errorf("sync %d spans %d..%d (%d ns), want 60 as the stub wrote it", i, y.Start, y.End, got)
		}
		// 3 is CUPTI_ACTIVITY_SYNCHRONIZATION_TYPE_STREAM_SYNCHRONIZE. The
		// type is what turns "the device was idle" into a sentence naming
		// which call the host was sitting in.
		if y.SyncType != 3 {
			t.Errorf("sync %d: sync_type=%d, want 3 (stream synchronize): the span survived but what it was did not", i, y.SyncType)
		}
		if y.CudaEventID != 3 {
			t.Errorf("sync %d: cuda_event_id=%d, want 3", i, y.CudaEventID)
		}
		// The context is not decoration here: it is the only field through
		// which this record can ever be placed on a device timeline.
		if y.Context != 1 {
			t.Errorf("sync %d: context=%d, want 1: without it the record can never be attributed to a device", i, y.Context)
		}
		if y.Stream != 7 {
			t.Errorf("sync %d: stream=%d, want 7", i, y.Stream)
		}
		if y.Correlation == 0 {
			t.Errorf("sync %d has correlation 0, so it can never be joined to the API call that waited", i)
		}
	}
	if len(c.failures) != 0 {
		t.Errorf("producer reported failures: %q", c.failures)
	}
	if !status.Exited() || status.ExitStatus() != 0 {
		t.Errorf("status = %v, want clean exit", status)
	}
}

// TestCapabilityTruncationNamesItself covers the branch that used to drop the
// capability record without a word.
//
// snprintf into a fixed buffer returns the length it wanted, and the agent
// wrote the record only when that fit. When it did not, nothing was written
// and nothing said so: the reader then refused the next activity record for
// arriving before a capability that was never written, which names the wrong
// record and no cause. The record has grown before -- agent_revision was added
// to it this week -- so the branch is reachable by ordinary work, not only by
// a malformed build.
//
// The buffer is a build-time macro so this can be forced. Waiting for the
// record to outgrow 16384 on its own is not a test.
func TestCapabilityTruncationNamesItself(t *testing.T) {
	run := func(t *testing.T, defines ...string) capture {
		t.Helper()
		bin := controlBinary(t, defines...)
		dir := t.TempDir()
		cmd := exec.Command(bin, "sustained", "1", "4", "256", "100")
		cmd.Env = append(os.Environ(), "CUDAPROF_TRANSPORT="+dir, "CUDAPROF_SESSION="+sessionHex)
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("run: %v: %s", err, out)
		}
		return only(t, decodeAll(t, dir, binRevision(bin)))
	}

	truncated := run(t, capabilityBufferDefine+"64")
	if truncated.counts[kindCapability] != 0 {
		t.Fatalf("capability record fit in 64 bytes: %d written, so this run does not exercise truncation", truncated.counts[kindCapability])
	}
	var said string
	for _, f := range truncated.failures {
		if strings.Contains(f, "capability record needs") {
			said = f
		}
	}
	if said == "" {
		t.Fatalf("capability record was dropped and no failure record said so: failures=%q", truncated.failures)
	}
	// The size is the actionable part: a reader who has to guess how much the
	// record needs cannot tell a buffer that is close from one that is absurd.
	if !strings.Contains(said, "buffer holds 64") {
		t.Errorf("failure does not name the buffer it overflowed: %s", said)
	}

	// The control. The shipped buffer must both hold the record and stay
	// quiet, or the assertion above would pass on a build that always
	// complains.
	normal := run(t)
	if normal.counts[kindCapability] == 0 {
		t.Error("no capability record at the shipped buffer size")
	}
	for _, f := range normal.failures {
		if strings.Contains(f, "capability record needs") {
			t.Errorf("shipped buffer reported truncation: %s", f)
		}
	}
}

// TestModuleTableOverflowIsRecorded loads more distinct modules than the
// producer's duplicate table holds, twice each. Below the bound the repeats
// are recognised and not written again; past it every load is written, and
// each such record, and the finalized total, must say it went unchecked.
func TestModuleTableOverflowIsRecorded(t *testing.T) {
	tests := []struct {
		name          string
		distinct      string
		wantRecords   int
		wantUnchecked uint64
	}{
		{"under", "10", 10, 0},
		{"at", "256", 256, 0},
		{"over", "300", 256 + 2*44, 2 * 44},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, status, captures := runEnv(t, []string{"CUDAPROF_MODULES=1"}, "modules", tt.distinct, "2")
			if status.ExitStatus() != 0 {
				t.Fatalf("harness exited %v", status)
			}
			c := only(t, captures)
			if c.finalized == nil {
				t.Fatal("no finalized record")
			}
			if len(c.modules) != tt.wantRecords {
				t.Errorf("module records = %d, want %d", len(c.modules), tt.wantRecords)
			}
			var unchecked uint64
			for _, m := range c.modules {
				if m.Unchecked {
					unchecked++
				}
			}
			if unchecked != tt.wantUnchecked {
				t.Errorf("records marked unchecked = %d, want %d", unchecked, tt.wantUnchecked)
			}
			if c.finalized.ModulesUnchecked != tt.wantUnchecked {
				t.Errorf("finalized modules_unchecked = %d, want %d", c.finalized.ModulesUnchecked, tt.wantUnchecked)
			}
			if c.finalized.ModuleSeenMax != 256 {
				t.Errorf("finalized module_seen_max = %d, want 256", c.finalized.ModuleSeenMax)
			}
		})
	}
}

// TestLaunchModuleChains drives every chain from a load to a launch through
// the stub: the module API, the library API loaded eagerly and lazily, a
// CUkernel launched directly, and the runtime's host stubs. Each kernel
// record must carry the module its launch came from, and a launch the
// producer cannot place must carry none rather than a stale one.
func TestLaunchModuleChains(t *testing.T) {
	result, status, captures := runEnv(t, []string{"CUDAPROF_MODULES=1"}, "module-chains")
	if status.ExitStatus() != 0 {
		t.Fatalf("harness exited %v", status)
	}
	if result.Fired != 44 || result.Offered != 9 {
		t.Fatalf("stub delivered %d callbacks and %d kernels, want 44 and 9: a callback the producer never enabled was dropped", result.Fired, result.Offered)
	}
	c := only(t, captures)
	if len(c.failures) != 0 {
		t.Errorf("producer recorded failures: %q", c.failures)
	}
	got := map[uint32]uint32{}
	for _, k := range c.moduleKernels {
		got[k.Correlation] = k.ModuleID
	}
	tests := []struct {
		name        string
		correlation uint32
		want        uint32
	}{
		{"eager library kernel", 101, 11},
		{"eager library function", 102, 11},
		{"lazy library function", 103, 12},
		{"lazy library kernel", 104, 12},
		{"kernel loaded by its launch", 105, 13},
		{"runtime stub loaded by its launch", 106, 14},
		{"runtime stub relaunched", 107, 14},
		{"runtime stub after an untracked load", 108, 0},
		{"module function", 109, 16},
	}
	for _, tt := range tests {
		m, ok := got[tt.correlation]
		if !ok {
			t.Errorf("%s: no kernel record with correlation %d", tt.name, tt.correlation)
			continue
		}
		if m != tt.want {
			t.Errorf("%s: module_id = %d, want %d", tt.name, m, tt.want)
		}
	}
}

// TestInitializeInjectionIsTheConstructorsAgent calls the entry point the CUDA
// driver calls in a CUDA_INJECTION64_PATH library. The constructor has already
// set the agent up, so the call must change nothing: the same single journal,
// one bootstrap, and every record accounted for.
func TestInitializeInjectionIsTheConstructorsAgent(t *testing.T) {
	result, status, captures := run(t, "injection", "1", "4")
	if status.ExitStatus() != 0 {
		t.Fatalf("harness exited %d", status.ExitStatus())
	}
	if result.InitRC != 1 || result.InitRCAgain != 1 {
		t.Fatalf("InitializeInjection returned %d then %d, want 1 both times", result.InitRC, result.InitRCAgain)
	}
	c := only(t, captures)
	if c.counts[kindBootstrap] != 1 {
		t.Errorf("bootstrap records = %d, want 1", c.counts[kindBootstrap])
	}
	if c.finalized == nil || c.finalized.State != "collected" {
		t.Fatalf("finalized = %+v, want a collected capture", c.finalized)
	}
	checkReconciles(t, c, result.Offered)
}

// TestCrashTrace checks that a target that dies of a fault leaves its last
// API callbacks on standard error, oldest first and bounded by the requested
// depth, and still dies of the signal it would have died of without the
// agent. Both ways a SIGSEGV arrives are covered: a fault, which recurs when
// the handler returns, and a raised signal, which does not.
func TestCrashTrace(t *testing.T) {
	tests := []struct {
		name   string
		env    []string
		raised string
		want   []string // lines that must appear, in order
		armed  string
	}{
		{
			name:   "fault",
			env:    []string{"CUDAPROF_CRASH_TRACE=3"},
			raised: "0",
			want: []string{
				"cudaprof: signal 11 at 0x8 on thread ",
				"; last 3 of 9 CUDA API callbacks, oldest first:",
				" driver cbid 307 exit correlation 4 cudaprof_control_api",
				" driver cbid 307 enter correlation 5 cudaprof_control_api",
			},
			armed: "armed",
		},
		{
			name:   "raised",
			env:    []string{"CUDAPROF_CRASH_TRACE=100"},
			raised: "1",
			want: []string{
				"cudaprof: signal 11 on thread ",
				"; last 9 of 9 CUDA API callbacks, oldest first:",
				" driver cbid 307 enter correlation 1 cudaprof_control_api",
				" driver cbid 307 enter correlation 5 cudaprof_control_api",
			},
			armed: "armed",
		},
		{
			name:   "refused",
			env:    []string{"CUDAPROF_CRASH_TRACE=3", "CUDAPROF_TEST_REFUSE_DOMAINS=1"},
			raised: "0",
			armed:  "requested_unavailable",
		},
		{
			name:   "off",
			raised: "0",
			armed:  "off",
		},
	}
	bin := controlBinary(t)
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.raised == "1" && runtime.GOOS != "linux" {
				// macOS reports a raised SIGSEGV with a fault's si_code
				// (SEGV_ACCERR), so the handler takes it for a fault that
				// will recur and does not raise it again.
				t.Skip("raise is indistinguishable from a fault on " + runtime.GOOS)
			}
			transport := t.TempDir()
			cmd := exec.Command(bin, "crash-trace", "5", tt.raised)
			cmd.Env = append(os.Environ(), "CUDAPROF_TRANSPORT="+transport, "CUDAPROF_SESSION="+sessionHex)
			cmd.Env = append(cmd.Env, tt.env...)
			var stderr strings.Builder
			cmd.Stderr = &stderr
			_ = cmd.Run()
			status, _ := cmd.ProcessState.Sys().(syscall.WaitStatus)
			if !status.Signaled() || status.Signal() != syscall.SIGSEGV {
				t.Fatalf("harness ended %v, want killed by SIGSEGV", status)
			}
			c := only(t, decodeAll(t, transport, binRevision(bin)))
			if c.capability.CrashTrace != tt.armed {
				t.Errorf("capability crash_trace = %q, want %q", c.capability.CrashTrace, tt.armed)
			}
			out := stderr.String()
			if tt.want == nil {
				if strings.Contains(out, "cudaprof: signal") {
					t.Errorf("crash trace written when not armed:\n%s", out)
				}
				return
			}
			rest := out
			for _, w := range tt.want {
				i := strings.Index(rest, w)
				if i < 0 {
					t.Fatalf("stderr lacks %q after the lines before it:\n%s", w, out)
				}
				rest = rest[i+len(w):]
			}
			if n := strings.Count(out, "cudaprof: signal"); n != 1 {
				t.Errorf("crash trace written %d times, want 1:\n%s", n, out)
			}
		})
	}
}

// TestGraphModuleIdentity follows loaded images through explicit node creation
// and cloning. Driver update ids are literal CUPTI cbids; runtime ids have
// 1000 added so the native harness can select their callback domain.
func TestGraphModuleIdentity(t *testing.T) {
	bin := controlBinary(t, cuda13Build...)
	for _, mutate := range []int{0, 521, 538, 561, 586, 691, 692, 696, 713, 714, 1288, 1326, 1335, 1361, 1446, 1447} {
		t.Run(fmt.Sprint(mutate), func(t *testing.T) {
			arg := strconv.Itoa(mutate)
			_, status, captures := runBuilt(t, bin, []string{"CUDAPROF_MODULES=1"}, "graph-modules", arg)
			c := only(t, captures)
			if !status.Exited() || status.ExitStatus() != 0 {
				t.Fatalf("status=%v", status)
			}
			if len(c.graphNodes) != 6 {
				t.Fatalf("nodes=%d, want 6", len(c.graphNodes))
			}
			for _, k := range c.graphNodes {
				want := uint32(20 + k.GraphNodeID)
				if mutate != 0 {
					want = 0
				}
				if k.ModuleID != want {
					t.Errorf("node %d module=%d, want %d", k.GraphNodeID, k.ModuleID, want)
				}
			}
		})
	}
}

func TestCUDA12GraphNodeIdentity(t *testing.T) {
	bin := controlBinary(t)
	_, status, captures := runBuilt(t, bin, nil, "graph-replay", strconv.Itoa(graphNodes), strconv.Itoa(graphReplays))
	c := only(t, captures)
	if !status.Exited() || status.ExitStatus() != 0 {
		t.Fatalf("status = %v", status)
	}
	if len(c.graphNodes) != graphNodes*graphReplays {
		t.Fatalf("graph nodes = %d, want %d", len(c.graphNodes), graphNodes*graphReplays)
	}
	for _, k := range c.graphNodes {
		if k.GraphID != 5 || k.GraphNodeID == 0 {
			t.Fatalf("lost CUDA 12 graph identity: %+v", k)
		}
	}
}

func TestGraphLifecycleCallbacks(t *testing.T) {
	for _, tc := range []struct {
		name    string
		enabled bool
		flags   []string
	}{
		{"cuda12 off", false, nil}, {"cuda12 on", true, nil}, {"cuda13 on", true, cuda13Build},
	} {
		t.Run(tc.name, func(t *testing.T) {
			bin := controlBinary(t, tc.flags...)
			env := []string{"CUDAPROF_GRAPH_LIFECYCLE=0"}
			if tc.enabled {
				env = []string{"CUDAPROF_GRAPH_LIFECYCLE=1"}
			}
			_, status, captures := runBuilt(t, bin, env, "graph-lifecycle")
			c := only(t, captures)
			if !status.Exited() || status.ExitStatus() != 0 {
				t.Fatalf("status=%v", status)
			}
			if !tc.enabled {
				if len(c.graphLifecycle) != 0 {
					t.Fatal("collected lifecycle while disabled")
				}
				return
			}
			if !bytes.Contains(c.capabilityJSON, []byte(`"graph_lifecycle":"collected"`)) || !bytes.Contains(c.capabilityJSON, []byte(`"graph_lifecycle_coverage":"driver_partial"`)) {
				t.Fatalf("missing capability: %s", c.capabilityJSON)
			}
			if len(c.graphLifecycle) != 8 {
				t.Fatalf("events=%d, want 8", len(c.graphLifecycle))
			}
			links := timeline.GraphLifecycle(c.graphLifecycle)
			for _, index := range []int{2, 5} {
				if links[index].Status != "matched" {
					t.Fatalf("replay %d: %+v", index, links[index])
				}
			}
			if links[2].Executable.Generation != 1 || links[5].Executable.Generation != 2 {
				t.Fatalf("reuse lost: %+v", links)
			}
		})
	}
}

func TestGraphLifecycleUpdates(t *testing.T) {
	bin := controlBinary(t, cuda13Build...)
	_, status, captures := runBuilt(t, bin, []string{"CUDAPROF_GRAPH_LIFECYCLE=1"}, "graph-lifecycle-updates")
	c := only(t, captures)
	if !status.Exited() || status.ExitStatus() != 0 {
		t.Fatalf("status=%v", status)
	}
	if len(c.graphLifecycle) != 9 {
		t.Fatalf("events=%d, want 9", len(c.graphLifecycle))
	}
	links := timeline.GraphLifecycle(c.graphLifecycle)
	if links[4].Status != "matched" || links[4].Executable.Version != 2 || links[4].Source.Version != 2 {
		t.Fatalf("update lost: %+v", links[4])
	}
	if links[6].Status != "matched" || links[6].Executable.Version != 3 || links[6].Source.Handle != 0 {
		t.Fatalf("node update fabricated source: %+v", links[6])
	}
	if links[7].Status != "unsupported" || links[8].Status != "ambiguous" {
		t.Fatalf("child update qualified: %+v", links[7:])
	}
}

func TestGraphRuntimeReplay(t *testing.T) {
	bin := controlBinary(t, cuda13Build...)
	_, status, captures := runBuilt(t, bin, []string{"CUDAPROF_GRAPH_LIFECYCLE=1"}, "graph-runtime-replay")
	c := only(t, captures)
	if !status.Exited() || status.ExitStatus() != 0 {
		t.Fatalf("status=%v", status)
	}
	if len(c.graphLifecycle) != 4 {
		t.Fatalf("events=%d, want 4", len(c.graphLifecycle))
	}
	if !bytes.Contains(c.capabilityJSON, []byte(`"graph_runtime_replay":true`)) {
		t.Fatalf("missing capability: %s", c.capabilityJSON)
	}
	links := timeline.GraphLifecycle(c.graphLifecycle)
	if c.graphLifecycle[3].Domain != "runtime" || c.graphLifecycle[3].Correlation != c.graphLifecycle[2].Correlation || links[3].Status != "matched" || links[2].Status != "matched" || links[3].Executable != links[2].Executable {
		t.Fatalf("runtime handle identity lost or domains collided: events=%+v links=%+v", c.graphLifecycle, links)
	}
}
