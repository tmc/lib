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
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"syscall"
	"testing"

	"github.com/tmc/lib/nvidia/internal/cudainject/wire"
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
	counts     map[uint16]uint64
	finalized  *finalizedRecord
	capability capabilityRecord
	failures   []string
	markers    []markerRecord
	memsets    []memsetRecord
	ranges     []rangeEdgeRecord
	truncated  bool
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

// Marker flags, mirroring CUpti_ActivityFlag in native/testcupti.
const (
	markerFlagInstantaneous = 1 << 0
	markerFlagStart         = 1 << 1
	markerFlagEnd           = 1 << 2
)

// capabilityRecord is what the producer reports it enabled.
type capabilityRecord struct {
	Activity        bool   `json:"activity"`
	APIRecords      bool   `json:"api_records"`
	DriverAPI       bool   `json:"driver_api"`
	RuntimeAPI      bool   `json:"runtime_api"`
	SigtermFinalize bool   `json:"sigterm_finalize"`
	AtforkGuard     bool   `json:"atfork_guard"`
	NVTXMarkers     bool   `json:"nvtx_markers"`
	Memsets         bool   `json:"memsets"`
	CaptureRange    string `json:"capture_range"`
	DrainBatch      int    `json:"drain_batch"`
	QueueCapacity   int    `json:"queue_capacity"`
	KernelLayout    string `json:"kernel_layout"`
}

var (
	buildMu    sync.Mutex
	buildCache = map[string]buildResult{}
)

type buildResult struct {
	path string
	err  error
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
	args := []string{"-std=c11", "-Wall", "-Wextra", "-Werror", "-pedantic", "-I", filepath.Join(src, "testcupti")}
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
	return result, status, decodeAll(t, transport)
}

func decodeAll(t *testing.T, transport string) []capture {
	t.Helper()
	dirs, err := filepath.Glob(filepath.Join(transport, "process-*"))
	if err != nil {
		t.Fatal(err)
	}
	var out []capture
	for _, dir := range dirs {
		out = append(out, decode(t, filepath.Join(dir, "capture.journal")))
	}
	return out
}

func decode(t *testing.T, path string) capture {
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
		switch r.Kind {
		case kindFinalized:
			var fin finalizedRecord
			if err := json.Unmarshal(r.Payload, &fin); err != nil {
				t.Fatalf("finalized payload %q: %v", r.Payload, err)
			}
			c.finalized = &fin
		case kindCapability:
			if err := json.Unmarshal(r.Payload, &c.capability); err != nil {
				t.Fatalf("capability payload %q: %v", r.Payload, err)
			}
		case kindFailure:
			c.failures = append(c.failures, string(r.Payload))
		case kindActivity:
			// Only markers are decoded here. The high-volume scenarios offer
			// millions of activity records and a JSON unmarshal per record
			// would make the decode, not the producer, the thing under test.
			if bytes.Contains(r.Payload, []byte(`"subtype":"capture_range"`)) {
				var e rangeEdgeRecord
				if err := json.Unmarshal(r.Payload, &e); err != nil {
					t.Fatalf("capture_range payload %q: %v", r.Payload, err)
				}
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
		}
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
	checkReconcilesOutOfBand(t, c, offered, 0)
}

// checkReconcilesOutOfBand is checkReconciles where outOfBand records reached
// the journal without passing through the ring. The constructor writes such a
// record directly when it fails before the drain thread exists, so it cannot
// appear in the drain's retained count.
func checkReconcilesOutOfBand(t *testing.T, c capture, offered uint64, outOfBand uint64) {
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
		case kindActivity, kindAPI, kindExternal:
			queued += n
		case kindFailure:
			queued += n
		}
	}
	if queued != fin.Retained+outOfBand {
		t.Errorf("journal holds %d records written outside the fixed three, drain claims %d plus %d out of band", queued, fin.Retained, outOfBand)
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
func checkPerKindReconciles(t *testing.T, c capture, drops, outOfBand uint64) {
	t.Helper()
	fin := c.finalized
	if fin.ByKind == nil {
		t.Fatal("finalized record carries no by_kind accounting")
	}
	for _, kind := range []string{"activity", "api", "external", "failure"} {
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
		"activity": c.counts[kindActivity],
		"api":      c.counts[kindAPI],
		"external": c.counts[kindExternal],
		"failure":  c.counts[kindFailure] - outOfBand,
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

// that the batched drain plus bounded backpressure retains everything.
func TestBurst(t *testing.T) {
	result, status, captures := run(t, "burst", "4", "8", "20000", "0")
	c := only(t, captures)
	checkReconciles(t, c, result.Offered)
	fin := c.finalized
	if result.Offered < 100000 {
		t.Fatalf("burst offered only %d records; the control is not applying pressure", result.Offered)
	}
	if fin.Retained != fin.Offered {
		// Capacity is the only drop backpressure can lose to, and it loses
		// only when the drain does not publish a freed slot within the
		// admission deadline. Report both maxima: the longest wait says how
		// far past that deadline admission went, and the longest batch write
		// says whether the drain was the reason.
		t.Errorf("burst lost %d of %d offered records (capacity=%d contention=%d oversize=%d longest_wait=%dms longest_write=%dms load1=%s)",
			fin.Offered-fin.Retained, fin.Offered,
			fin.QueueCapacityDropped, fin.QueueContentionDrop, fin.QueueOversizeDropped,
			fin.BackpressureMaxNanos/1e6, fin.DrainWriteMaxNanos/1e6, loadAverage())
	}
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
	if c.finalized.Retained != c.finalized.Offered {
		t.Errorf("activity-only run lost %d of %d offered records (capacity=%d contention=%d oversize=%d longest_wait=%dms longest_write=%dms load1=%s)",
			c.finalized.Offered-c.finalized.Retained, c.finalized.Offered,
			c.finalized.QueueCapacityDropped, c.finalized.QueueContentionDrop, c.finalized.QueueOversizeDropped,
			c.finalized.BackpressureMaxNanos/1e6, c.finalized.DrainWriteMaxNanos/1e6, loadAverage())
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
	if !c.capability.SigtermFinalize {
		t.Error("sigterm_finalize = false, want true when the target has a default disposition")
	}
	if !c.capability.AtforkGuard {
		t.Error("atfork_guard = false: a forked child would keep a live-looking agent")
	}
	if !c.capability.APIRecords {
		t.Error("api_records = false, want true by default")
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
	checkReconcilesOutOfBand(t, c, result.Offered, 1)
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
	// harness's own counters cannot speak to.
	Edges []rangeEdgeRecord `json:"-"`
}

// runRange executes the capture-range scenario and decodes its line.
func runRange(t *testing.T, env []string, domain string) rangeResult {
	t.Helper()
	dir := t.TempDir()
	cmd := exec.Command(controlBinary(t), "capture-range", "1", "4", "0", "0", domain)
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
	for _, c := range decodeAll(t, dir) {
		got.Edges = append(got.Edges, c.ranges...)
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
	_, status, captures := run(t, "sustained", "1", "1", "8", "0", "0", "0", "6")
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
