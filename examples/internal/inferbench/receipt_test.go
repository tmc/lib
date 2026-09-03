package inferbench

import (
	"math"
	"os"
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
		r := Run{Arm: arm, Identity: Identity{BinarySHA256: h, ModelSHA256: h, TokenizerSHA256: h, ConfigSHA256: h, SourceSHA256: h, Backend: "metal", Precision: "f16", DeviceID: "gpu0"}, Workload: w, Evidence: Evidence{HardwareVerified: true, DeviceResidentKV: true, TimingBoundary: "submit-to-final-device-synchronize", DeviceSynchronized: true, IncludesQueueing: true, RunnerLease: "host.lock", DeviceEvidenceSHA256: h, NullControlSHA256: h, NullControlSamples: 20, NullControlP95: time.Millisecond, NullControlLimit: 2 * time.Millisecond, PersistentRuntime: true, CompilationExcluded: true, WarmupRunsCompleted: 2}}
		for i := 0; i < 20; i++ {
			r.Samples = append(r.Samples, Sample{ID: "sample-" + string(rune('A'+i)), TTFT: time.Millisecond, TPOT: time.Millisecond, InterToken: []time.Duration{time.Millisecond, time.Millisecond}, TokensPerSecond: 100, Dispatches: 1, DispatchMeasured: true, TierReadBytes: map[string]uint64{"device": 0}, TierWriteBytes: map[string]uint64{"device": 0}, TierBytesMeasured: true, GeneratedTokenIDs: [][]int32{{4, 5, 6}}, OutputSHA256: outputHash})
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
		{"missing tier counters", func(c *Campaign) { c.Runs[0].Samples[0].TierReadBytes = map[string]uint64{} }},
		{"inferred dispatch counters", func(c *Campaign) { c.Runs[0].Samples[0].DispatchMeasured = false }},
		{"placeholder tier counters", func(c *Campaign) { c.Runs[0].Samples[0].TierBytesMeasured = false }},
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
