package inferbench

import (
	"fmt"
	"os"
	"strings"
	"testing"
	"time"
)

func linkedFixture(t *testing.T) (LinkedReceipts, string) {
	t.Helper()
	h := strings.Repeat("a", 64)
	identity := Identity{BinarySHA256: h, ModelSHA256: h, TokenizerSHA256: h, ConfigSHA256: h, SourceSHA256: h, Backend: "cuda", Precision: "f16", DeviceID: "gpu0"}
	other := identity
	other.BinarySHA256 = strings.Repeat("b", 64)
	other.SourceSHA256 = strings.Repeat("c", 64)
	m := ExperimentManifest{
		ID: "experiment-1", Arms: []ExperimentArm{{Name: "packed", Identity: identity}, {Name: "serial", Identity: other}},
		Workload:   Workload{PromptTokenIDs: [][]int32{{1, 2}}, ContextTokens: 2, GeneratedTokens: 2, Concurrency: 1, WarmupRuns: 2, Sampler: "greedy-argmax", CacheState: "cold", OutputMode: ThroughputOutputMode},
		RandomSeed: 7, TimingBoundary: "submit-to-final-device-synchronize", RunnerLease: "cuda:gpu0", PrimaryMetric: "tokens_per_second", PrimaryThreshold: .10, SecondaryLimit: .05,
	}
	for i := 0; i < 20; i++ {
		m.Arms[0].SampleIDs = append(m.Arms[0].SampleIDs, fmt.Sprintf("p-%02d", i))
		m.Arms[1].SampleIDs = append(m.Arms[1].SampleIDs, fmt.Sprintf("s-%02d", i))
	}
	m.Order = Schedule([]Run{{Arm: "packed", Samples: samplesForIDs(m.Arms[0].SampleIDs)}, {Arm: "serial", Samples: samplesForIDs(m.Arms[1].SampleIDs)}}, m.RandomSeed)
	sum, err := HashJSON(m)
	if err != nil {
		t.Fatal(err)
	}
	generated := [][]int32{{3, 4}}
	r := LinkedReceipts{Version: LinkedReceiptVersion, Manifest: m}
	r.Qualification = QualificationReceipt{ManifestSHA256: sum, Backend: "cuda", DeviceID: "gpu0", DeviceSHA256: h, NoCPUFallback: true, DeviceResidentKV: true, LeaseHeld: true, SentinelLimit: time.Millisecond}
	for i := 0; i < 20; i++ {
		r.Qualification.Sentinels = append(r.Qualification.Sentinels, time.Microsecond)
	}
	for i := 9; i < len(m.Order); i += 10 {
		r.Qualification.Periodic = append(r.Qualification.Periodic, Sentinel{AfterIndex: i, Duration: time.Microsecond})
	}
	r.Correctness = []CorrectnessReceipt{
		{ManifestSHA256: sum, Arm: "packed", Generated: generated, FullLogitsSHA256: h, PopulatedKVSHA256: h, Synchronized: true},
		{ManifestSHA256: sum, Arm: "serial", Generated: generated, FullLogitsSHA256: h, PopulatedKVSHA256: h, Synchronized: true},
	}
	base := time.Unix(1, 0).UTC()
	dir := t.TempDir()
	for i, scheduled := range m.Order {
		start := base.Add(time.Duration(3*i) * time.Millisecond)
		end := start.Add(2 * time.Millisecond)
		got := [][]int32{append([]int32(nil), generated[0]...)}
		name := fmt.Sprintf("%06d.json", i)
		if err := os.WriteFile(dir+"/"+name, []byte(scheduled.Arm+"/"+scheduled.ID), 0o600); err != nil {
			t.Fatal(err)
		}
		rawSum, err := HashFile(dir + "/" + name)
		if err != nil {
			t.Fatal(err)
		}
		r.Performance = append(r.Performance, PerformanceReceipt{ManifestSHA256: sum, Index: i, Arm: scheduled.Arm, ID: scheduled.ID, StartedAt: start, EndedAt: end, Requests: []RequestTiming{{StartedAt: start, TokenAt: []time.Time{start.Add(time.Millisecond), end}, EndedAt: end}}, Generated: got, Synchronized: true, Status: "completed", RawFile: name, RawSHA256: rawSum})
	}
	r.Resources = []ResourceReceipt{
		{ManifestSHA256: sum, Arm: "packed", PersistentBytes: 100, PeakBytes: 120, RequestBytes: 20, AfterReleaseBytes: 100, Coverage: "engine-owned allocations"},
		{ManifestSHA256: sum, Arm: "serial", PersistentBytes: 100, PeakBytes: 110, RequestBytes: 10, AfterReleaseBytes: 100, Coverage: "engine-owned allocations"},
	}
	return r, dir
}

func samplesForIDs(ids []string) []Sample {
	samples := make([]Sample, len(ids))
	for i, id := range ids {
		samples[i].ID = id
	}
	return samples
}

func TestLinkedReceiptsSeparatePerformanceFromResourcesAndDiagnostics(t *testing.T) {
	r, dir := linkedFixture(t)
	r.Resources = nil
	r.Diagnostics = nil
	if err := r.ValidatePerformance(); err != nil {
		t.Fatalf("ValidatePerformance: %v", err)
	}
	if err := r.ValidatePublication(dir); err == nil {
		t.Fatal("ValidatePublication accepted missing resource receipts")
	}
	r, dir = linkedFixture(t)
	if err := r.ValidatePublication(dir); err != nil {
		t.Fatalf("ValidatePublication: %v", err)
	}
}

func TestLinkedReceiptsRejectOldGeneratedTokenFalsePass(t *testing.T) {
	// Schema v8 could compare generated-token hashes without linking the timed
	// arms to an untimed full-logit and populated-KV correctness proof.
	r, _ := linkedFixture(t)
	r.Correctness = nil
	if err := r.ValidatePerformance(); err == nil {
		t.Fatal("ValidatePerformance accepted timing without correctness receipts")
	}
}

func TestLinkedReceiptsRejectRawArtifactTampering(t *testing.T) {
	r, dir := linkedFixture(t)
	if err := os.WriteFile(dir+"/"+r.Performance[0].RawFile, []byte("substituted"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := r.ValidatePerformanceRaw(dir); err == nil {
		t.Fatal("ValidatePerformanceRaw accepted a substituted raw artifact")
	}
}

func TestLinkedReceiptsRejectSubstitution(t *testing.T) {
	tests := []struct {
		name string
		edit func(*LinkedReceipts)
	}{
		{"manifest link", func(r *LinkedReceipts) { r.Performance[0].ManifestSHA256 = strings.Repeat("d", 64) }},
		{"output", func(r *LinkedReceipts) { r.Performance[0].Generated[0][0]++ }},
		{"raw file", func(r *LinkedReceipts) { r.Performance[1].RawFile = r.Performance[0].RawFile }},
		{"synchronization", func(r *LinkedReceipts) { r.Performance[0].Synchronized = false }},
		{"sentinel", func(r *LinkedReceipts) { r.Qualification.Sentinels[0] = 2 * time.Millisecond }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r, _ := linkedFixture(t)
			tt.edit(&r)
			if err := r.ValidatePerformance(); err == nil {
				t.Fatal("ValidatePerformance accepted substituted evidence")
			}
		})
	}
}
