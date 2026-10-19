package nccltest

import (
	"os"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/ncclinspector"
)

const simulatedStallCommHash = "0x91541b8a24505d53"

// simulatedTwoRankHostEnqueueObservations constructs a synthetic two-rank
// AllReduce observation pair for one shared host at baseNs with rank 1 shifted
// by peerDelay and symmetric per-timestamp uncertaintyNS.
//
// The timestamps are caller-supplied synthetic values (for example wall-clock
// time.Now().UnixNano() in the harness fixture), not independent rank enqueue
// logs or measured host-monotonic hardware timestamps.
func simulatedTwoRankHostEnqueueObservations(hostname string, baseNs int64, peerDelay time.Duration, uncertaintyNS int64) (ncclinspector.CollectiveObservation, ncclinspector.CollectiveObservation) {
	pid := os.Getpid()
	delayNS := peerDelay.Nanoseconds()
	obs0 := ncclinspector.CollectiveObservation{
		CommunicatorHash: simulatedStallCommHash,
		Rank:             0,
		TotalRanks:       2,
		Sequence:         1,
		Operation:        "AllReduce",
		Bytes:            65536,
		HostPID:          pid,
		Hostname:         hostname,
		HostEnqueueStart: ncclinspector.BoundedTimestamp{Nanoseconds: baseNs, UncertaintyNS: uncertaintyNS, Domain: ncclinspector.ClockDomainHostMonotonic},
		HostEnqueueEnd:   ncclinspector.BoundedTimestamp{Nanoseconds: baseNs + 5000, UncertaintyNS: uncertaintyNS, Domain: ncclinspector.ClockDomainHostMonotonic},
		HasHostEnqueue:   true,
	}
	obs1 := ncclinspector.CollectiveObservation{
		CommunicatorHash: simulatedStallCommHash,
		Rank:             1,
		TotalRanks:       2,
		Sequence:         1,
		Operation:        "AllReduce",
		Bytes:            65536,
		HostPID:          pid,
		Hostname:         hostname,
		HostEnqueueStart: ncclinspector.BoundedTimestamp{Nanoseconds: baseNs + delayNS, UncertaintyNS: uncertaintyNS, Domain: ncclinspector.ClockDomainHostMonotonic},
		HostEnqueueEnd:   ncclinspector.BoundedTimestamp{Nanoseconds: baseNs + delayNS + 5000, UncertaintyNS: uncertaintyNS, Domain: ncclinspector.ClockDomainHostMonotonic},
		HasHostEnqueue:   true,
	}
	return obs0, obs1
}

// requireTwoCUDA requires the native opt-in and an inherited GPU lease before
// checking that at least two CUDA devices exist.
func requireTwoCUDA(t testing.TB) []int32 {
	t.Helper()
	requireCUDA(t)
	var devices []int32
	if err := withTwoCUDALease(os.Getenv, func() {
		if err := driver.Init(0); err != nil {
			t.Fatal(err)
		}
		count, err := driver.DeviceGetCount()
		if err != nil {
			t.Fatal(err)
		}
		if count < 2 {
			t.Skipf("two CUDA devices required for multi-GPU collective tests (found %d)", count)
		}
		devices = []int32{0, 1}
	}); err != nil {
		t.Fatalf("two-GPU test requires an inherited GPU lease: %v", err)
	}
	return devices
}

// TestTwoRankDelayedPeerControl combines two-GPU numerical normal/delayed-peer
// controls with a separate simulated same-host attribution fixture. The
// attribution subsection uses synthetic wall-clock timestamps rather than
// independent rank enqueue logs or measured host-monotonic capture.
func TestTwoRankDelayedPeerControl(t *testing.T) {
	devices := requireTwoCUDA(t)

	c := Case{Operation: AllReduce, DataType: Float32, Bytes: 65536}

	// 1. Normal control: no injected delay
	optsNormal := Options{
		Devices:    devices,
		Warmup:     2,
		Iterations: 10,
		Checks:     1,
		DelayRank:  0,
		PeerDelay:  0,
	}
	resNormal, err := Run(c, optsNormal)
	if err != nil {
		t.Fatalf("normal control failed: %v", err)
	}
	if resNormal.Wrong != 0 || resNormal.Comparisons <= 0 {
		t.Fatalf("normal control numerical verification failed: %+v", resNormal)
	}

	// 2. Deliberately delayed-peer control: rank 1 delayed by 10ms
	peerDelay := 10 * time.Millisecond
	optsDelayed := Options{
		Devices:    devices,
		Warmup:     2,
		Iterations: 10,
		Checks:     1,
		DelayRank:  1,
		PeerDelay:  peerDelay,
	}
	resDelayed, err := Run(c, optsDelayed)
	if err != nil {
		t.Fatalf("delayed peer control failed: %v", err)
	}
	// Verify that despite deliberate delay, numerical correctness is preserved by the oracle
	if resDelayed.Wrong != 0 || resDelayed.Comparisons <= 0 {
		t.Fatalf("delayed control numerical verification failed: %+v", resDelayed)
	}

	t.Logf("Normal GPU time: %s, Delayed GPU time: %s, PeerDelay: %s",
		resNormal.GPUTime, resDelayed.GPUTime, peerDelay)

	// Simulated attribution fixture: feed synthetic same-host Inspector
	// observations parameterized by peerDelay (synthetic wall-clock timestamps,
	// not independent measured rank enqueue logs).
	host, err := os.Hostname()
	if err != nil || strings.TrimSpace(host) == "" {
		t.Fatalf("os.Hostname() = %q, %v; want non-empty host identity", host, err)
	}

	attr := ncclinspector.NewTwoRankAttributor()
	obs0, obs1 := simulatedTwoRankHostEnqueueObservations(host, time.Now().UnixNano(), peerDelay, 500)

	if err := attr.AddObservation(obs0); err != nil {
		t.Fatal(err)
	}
	if err := attr.AddObservation(obs1); err != nil {
		t.Fatal(err)
	}

	report, err := attr.Attribute(simulatedStallCommHash, 1)
	if err != nil {
		t.Fatal(err)
	}
	if !report.Admitted || report.Diagnosis != ncclinspector.DiagnosisHostEnqueueSkew || report.DelayedRank != 1 {
		t.Fatalf("unexpected attribution report: %+v", report)
	}
	t.Logf("Simulated attribution fixture diagnosis: %s, delayed rank: %d, delay: %d ns",
		report.Diagnosis, report.DelayedRank, report.DelayNS)
}

// TestCollectorCostComparison compares performance overhead when telemetry/inspection
// recording is disabled versus enabled.
func TestCollectorCostComparison(t *testing.T) {
	devices := requireTwoCUDA(t)
	c := Case{Operation: AllReduce, DataType: Float32, Bytes: 65536}

	// Baseline run (collector disabled)
	t.Setenv("NCCL_INSPECTOR_ENABLE", "0")
	baseline, err := Run(c, Options{Devices: devices, Warmup: 2, Iterations: 10, Checks: 1})
	if err != nil {
		t.Fatalf("baseline run failed: %v", err)
	}

	// Monitored run (collector enabled)
	t.Setenv("NCCL_INSPECTOR_ENABLE", "1")
	monitored, err := Run(c, Options{Devices: devices, Warmup: 2, Iterations: 10, Checks: 1})
	if err != nil {
		t.Fatalf("monitored run failed: %v", err)
	}

	t.Logf("Collector-off: %s, Collector-on: %s", baseline.GPUTime, monitored.GPUTime)
}
