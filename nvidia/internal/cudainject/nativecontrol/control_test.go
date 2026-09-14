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
	"encoding/hex"
	"encoding/json"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
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
	truncated  bool
}

// capabilityRecord is what the producer reports it enabled.
type capabilityRecord struct {
	Activity        bool   `json:"activity"`
	APIRecords      bool   `json:"api_records"`
	DriverAPI       bool   `json:"driver_api"`
	RuntimeAPI      bool   `json:"runtime_api"`
	SigtermFinalize bool   `json:"sigterm_finalize"`
	AtforkGuard     bool   `json:"atfork_guard"`
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
		t.Errorf("burst lost %d of %d offered records (capacity=%d contention=%d oversize=%d longest_wait=%dms longest_write=%dms)",
			fin.Offered-fin.Retained, fin.Offered,
			fin.QueueCapacityDropped, fin.QueueContentionDrop, fin.QueueOversizeDropped,
			fin.BackpressureMaxNanos/1e6, fin.DrainWriteMaxNanos/1e6)
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
	t.Logf("burst offered=%d retained=%d batches=%d max_batch=%d backpressure_waits=%d backpressure_ms=%d longest_wait_ms=%d longest_write_ms=%d callback_max_us=%d",
		fin.Offered, fin.Retained, fin.DrainBatches, fin.DrainMaxBatch,
		fin.BackpressureWaits, fin.BackpressureNanos/1e6,
		fin.BackpressureMaxNanos/1e6, fin.DrainWriteMaxNanos/1e6,
		fin.CallbackMaxNanos/1e3)
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
		t.Errorf("activity-only run lost %d of %d offered records",
			c.finalized.Offered-c.finalized.Retained, c.finalized.Offered)
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
