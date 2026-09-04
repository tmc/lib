package inferbench

import (
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func validCampaign() Campaign {
	h := strings.Repeat("a", 64)
	outputHash, _ := HashJSON([][]int32{{4, 5, 6}})
	w := Workload{PromptTokenIDs: [][]int32{{1, 2, 3}}, ContextTokens: 3, GeneratedTokens: 3, Concurrency: 1, WarmupRuns: 2, Sampler: "greedy", CacheState: "cold"}
	var runs []Run
	now := time.Unix(1, 0)
	for _, arm := range []string{"a", "b"} {
		r := Run{Arm: arm, Identity: Identity{BinarySHA256: h, ModelSHA256: h, TokenizerSHA256: h, ConfigSHA256: h, SourceSHA256: h, Backend: "metal", Precision: "f16", DeviceID: "gpu0"}, Workload: w, Evidence: Evidence{HardwareVerified: true, DeviceResidentKV: true, TimingBoundary: "submit-to-final-device-synchronize", DeviceSynchronized: true, IncludesQueueing: true, RunnerLease: "host.lock", DeviceEvidenceSHA256: h, NullControlSHA256: h, NullControlSamples: 20, NullControlP95: time.Millisecond, NullControlLimit: 2 * time.Millisecond, PersistentRuntime: true, CompilationExcluded: true, WarmupRunsCompleted: 2, RuntimePolicy: PersistentEnginePolicy, AllocationPolicy: ColdIntervalPolicy}}
		for i := 0; i < 20; i++ {
			id := "sample-" + string(rune('A'+i))
			r.Samples = append(r.Samples, Sample{ID: id, TTFT: time.Millisecond, TPOT: time.Millisecond, InterToken: []time.Duration{time.Millisecond, time.Millisecond}, TokensPerSecond: 3000, Dispatches: 1, DispatchMeasured: true, TierReadBytes: map[string]uint64{"device": 0}, TierWriteBytes: map[string]uint64{"device": 0}, TierBytesMeasured: true, PhysicalWaves: []PhysicalWave{{Sequence: uint64(i + 1), Kind: "decode", Requests: []string{id + "-0@0/1"}, Dispatches: 1}}, GeneratedTokenIDs: [][]int32{{4, 5, 6}}, OutputSHA256: outputHash})
		}
		runs = append(runs, r)
	}
	c := Campaign{SchemaVersion: SchemaVersion, RandomSeed: 7, Runs: runs}
	c.Order = Schedule(c.Runs, c.RandomSeed)
	for _, k := range c.Order {
		for ri := range c.Runs {
			if c.Runs[ri].Arm != k.Arm {
				continue
			}
			for si := range c.Runs[ri].Samples {
				if c.Runs[ri].Samples[si].ID == k.ID {
					c.Runs[ri].Samples[si].StartedAt = now
					now = now.Add(time.Millisecond)
					c.Runs[ri].Samples[si].EndedAt = now
					c.Runs[ri].Samples[si].Requests = []RequestTiming{{StartedAt: c.Runs[ri].Samples[si].StartedAt, TokenAt: []time.Time{c.Runs[ri].Samples[si].StartedAt, c.Runs[ri].Samples[si].StartedAt, now}, EndedAt: now}}
				}
			}
		}
	}
	return c
}

func TestValidate(t *testing.T) {
	c := validCampaign()
	if err := c.Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestValidateRejectsUnsafeEvidence(t *testing.T) {
	tests := []struct {
		name   string
		change func(*Campaign)
	}{
		{"unequal work", func(c *Campaign) { c.Runs[1].Workload.GeneratedTokens++ }},
		{"too few samples", func(c *Campaign) { c.Runs[0].Samples = c.Runs[0].Samples[:1]; c.Order = Schedule(c.Runs, c.RandomSeed) }},
		{"cpu fallback", func(c *Campaign) { c.Runs[0].Evidence.CPUFallback = true }},
		{"unsynchronized", func(c *Campaign) { c.Runs[0].Evidence.DeviceSynchronized = false }},
		{"non-finite throughput", func(c *Campaign) { c.Runs[0].Samples[0].TokensPerSecond = math.NaN() }},
		{"throughput outside interval", func(c *Campaign) { c.Runs[0].Samples[0].TokensPerSecond = 100 }},
		{"ttft outside interval", func(c *Campaign) { c.Runs[0].Samples[0].TTFT = 2 * time.Millisecond }},
		{"missing tier counters", func(c *Campaign) { c.Runs[0].Samples[0].TierReadBytes = map[string]uint64{} }},
		{"inferred dispatch counters", func(c *Campaign) { c.Runs[0].Samples[0].DispatchMeasured = false }},
		{"placeholder tier counters", func(c *Campaign) { c.Runs[0].Samples[0].TierBytesMeasured = false }},
		{"missing physical waves", func(c *Campaign) { c.Runs[0].Samples[0].PhysicalWaves = nil }},
		{"double attributed dispatch", func(c *Campaign) { c.Runs[0].Samples[0].PhysicalWaves[0].Dispatches++ }},
		{"duplicate wave member", func(c *Campaign) {
			c.Runs[0].Samples[0].PhysicalWaves[0].Requests = append(c.Runs[0].Samples[0].PhysicalWaves[0].Requests, c.Runs[0].Samples[0].PhysicalWaves[0].Requests[0])
		}},
		{"different model", func(c *Campaign) { c.Runs[1].Identity.ModelSHA256 = strings.Repeat("b", 64) }},
		{"different config", func(c *Campaign) { c.Runs[1].Identity.ConfigSHA256 = strings.Repeat("b", 64) }},
		{"missing null control", func(c *Campaign) { c.Runs[0].Evidence.NullControlSHA256 = "" }},
		{"different null control", func(c *Campaign) { c.Runs[1].Evidence.NullControlSHA256 = strings.Repeat("b", 64) }},
		{"no persistent runtime", func(c *Campaign) { c.Runs[0].Evidence.PersistentRuntime = false }},
		{"compile included", func(c *Campaign) { c.Runs[0].Evidence.CompilationExcluded = false }},
		{"warmup incomplete", func(c *Campaign) { c.Runs[0].Evidence.WarmupRunsCompleted-- }},
		{"noisy host", func(c *Campaign) { c.Runs[0].Evidence.NullControlP95 = 3 * time.Millisecond }},
		{"different output", func(c *Campaign) { c.Runs[1].Samples[0].GeneratedTokenIDs[0][0]++ }},
		{"reduced work", func(c *Campaign) {
			c.Runs[0].Samples[0].GeneratedTokenIDs[0] = c.Runs[0].Samples[0].GeneratedTokenIDs[0][:2]
		}},
		{"overlap", func(c *Campaign) {
			k := c.Order[1]
			for ri := range c.Runs {
				for si := range c.Runs[ri].Samples {
					if c.Runs[ri].Arm == k.Arm && c.Runs[ri].Samples[si].ID == k.ID {
						c.Runs[ri].Samples[si].StartedAt = time.Unix(1, 0)
					}
				}
			}
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := validCampaign()
			tt.change(&c)
			if err := c.Validate(); err == nil {
				t.Fatal("accepted invalid campaign")
			}
		})
	}
}

func TestLease(t *testing.T) {
	p := filepath.Join(t.TempDir(), "runner.lock")
	a, err := AcquireLease(p)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := AcquireLease(p); err == nil {
		t.Fatal("second lease succeeded")
	}
	if err := a.Close(); err != nil {
		t.Fatal(err)
	}
	b, err := AcquireLease(p)
	if err != nil {
		t.Fatal(err)
	}
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestDeviceLeaseRejectsConcurrentCollectors(t *testing.T) {
	dir := t.TempDir()
	a, err := AcquireDeviceLease(dir, "metal", "gpu0")
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	if _, err := AcquireDeviceLease(dir, "metal", "gpu0"); err == nil {
		t.Fatal("second collector acquired the same device")
	}
	b, err := AcquireDeviceLease(dir, "metal", "gpu1")
	if err != nil {
		t.Fatalf("different device: %v", err)
	}
	if err := b.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestLeaseRecoveredAfterProcessExit(t *testing.T) {
	path := filepath.Join(t.TempDir(), "runner.lock")
	cmd := exec.Command(os.Args[0], "-test.run=TestLeaseCrashHelper")
	cmd.Env = append(os.Environ(), "INFERBENCH_CRASH_LEASE="+path)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("helper: %v\n%s", err, out)
	}
	lease, err := AcquireLease(path)
	if err != nil {
		t.Fatalf("reacquire after process exit: %v", err)
	}
	if err := lease.Close(); err != nil {
		t.Fatal(err)
	}
}

func TestLeaseCrashHelper(t *testing.T) {
	path := os.Getenv("INFERBENCH_CRASH_LEASE")
	if path == "" {
		return
	}
	if _, err := AcquireLease(path); err != nil {
		t.Fatal(err)
	}
	os.Exit(0)
}

func TestHashPathChangesWithContentAndName(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "a")
	if err := os.WriteFile(path, []byte("one"), 0600); err != nil {
		t.Fatal(err)
	}
	a, err := HashPath(dir)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(path, filepath.Join(dir, "b")); err != nil {
		t.Fatal(err)
	}
	b, err := HashPath(dir)
	if err != nil {
		t.Fatal(err)
	}
	if a == b {
		t.Fatal("directory hash ignored file name")
	}
}

func TestWriteJSONAtomic(t *testing.T) {
	path := filepath.Join(t.TempDir(), "receipt.json")
	if err := WriteJSONAtomic(path, map[string]int{"n": 1}); err != nil {
		t.Fatal(err)
	}
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(b) != "{\n  \"n\": 1\n}\n" {
		t.Fatalf("receipt = %q", b)
	}
}
