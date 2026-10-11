package nccltest

import (
	"os"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/ncclinspector"
)

// requireTwoCUDA skips the test unless NCCLTEST_CUDA=1 and at least two CUDA devices exist.
func requireTwoCUDA(t testing.TB) []int32 {
	t.Helper()
	requireCUDA(t)
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
	return []int32{0, 1}
}

// TestTwoRankDelayedPeerControl executes normal and deliberately delayed-peer controls
// on two physical GPUs with independent rank timing and numerical output verification.
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

	// Attribution verification: feed simulated Inspector observations matching the run
	attr := ncclinspector.NewTwoRankAttributor()
	commHash := "0x91541b8a24505d53"

	now := time.Now().UnixNano()
	obs0 := ncclinspector.CollectiveObservation{
		CommunicatorHash: commHash,
		Rank:             0,
		TotalRanks:       2,
		Sequence:         1,
		Operation:        "AllReduce",
		Bytes:            65536,
		HostPID:          os.Getpid(),
		HostEnqueueStart: ncclinspector.BoundedTimestamp{Nanoseconds: now, UncertaintyNS: 500, Domain: ncclinspector.ClockDomainHostMonotonic},
		HostEnqueueEnd:   ncclinspector.BoundedTimestamp{Nanoseconds: now + 5000, UncertaintyNS: 500, Domain: ncclinspector.ClockDomainHostMonotonic},
		HasHostEnqueue:   true,
	}
	obs1 := ncclinspector.CollectiveObservation{
		CommunicatorHash: commHash,
		Rank:             1,
		TotalRanks:       2,
		Sequence:         1,
		Operation:        "AllReduce",
		Bytes:            65536,
		HostPID:          os.Getpid(),
		HostEnqueueStart: ncclinspector.BoundedTimestamp{Nanoseconds: now + peerDelay.Nanoseconds(), UncertaintyNS: 500, Domain: ncclinspector.ClockDomainHostMonotonic},
		HostEnqueueEnd:   ncclinspector.BoundedTimestamp{Nanoseconds: now + peerDelay.Nanoseconds() + 5000, UncertaintyNS: 500, Domain: ncclinspector.ClockDomainHostMonotonic},
		HasHostEnqueue:   true,
	}

	if err := attr.AddObservation(obs0); err != nil {
		t.Fatal(err)
	}
	if err := attr.AddObservation(obs1); err != nil {
		t.Fatal(err)
	}

	report, err := attr.Attribute(commHash, 1)
	if err != nil {
		t.Fatal(err)
	}
	if !report.Admitted || report.Diagnosis != ncclinspector.DiagnosisHostEnqueueSkew || report.DelayedRank != 1 {
		t.Fatalf("unexpected attribution report: %+v", report)
	}
	t.Logf("Attributed diagnosis: %s, delayed rank: %d, delay: %d ns",
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
