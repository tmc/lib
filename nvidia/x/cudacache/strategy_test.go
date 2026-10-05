package cudacache

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudatrial"
)

func strategyFixture() Strategy {
	return Strategy{Family: cudatrial.MemoryBandwidthLimited, TargetArch: "sm_121", Technique: "shared_memory_tiling", SourceGroup: "memory_bandwidth_saturated", Confidence: 0.75, PredictedSpeedup: 1.2}
}

func TestSyntheticTrajectoryRequiresExplicitTrust(t *testing.T) {
	r := cachedTrajectoryFixture()
	var k StrategyKnowledgeBase
	if err := k.RecordTrajectory(r, nil); !errors.Is(err, ErrUnqualifiedOutcome) {
		t.Fatalf("synthetic catalog reward admitted: %v", err)
	}
	c, err := New(Config{RootDir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	if err := c.PutTrajectory(context.Background(), r); !errors.Is(err, ErrUnqualifiedOutcome) {
		t.Fatalf("synthetic persistence admitted: %v", err)
	}
}

func TestKnowledgeBase_JSONRoundTripAndQuery(t *testing.T) {
	k := new(StrategyKnowledgeBase)
	s := strategyFixture()
	s.HistoricalSpeedups = []float64{2}
	s.Observations = 1
	if err := k.Update(s); err != nil {
		t.Fatal(err)
	}
	s.HistoricalSpeedups[0] = 99
	path := filepath.Join(t.TempDir(), "strategies.json")
	if err := k.SaveFile(path); err != nil {
		t.Fatal(err)
	}
	next := new(StrategyKnowledgeBase)
	if err := next.LoadFile(path); err != nil {
		t.Fatal(err)
	}
	q := next.Query(s.Family, s.TargetArch)
	if len(q) != 1 || q[0].HistoricalSpeedups[0] != 2 {
		t.Fatal(q)
	}
	q[0].HistoricalSpeedups[0] = 20
	if next.Query(s.Family, s.TargetArch)[0].HistoricalSpeedups[0] != 2 {
		t.Fatal("query alias")
	}
	if len(next.Query(s.Family, "sm_120")) != 0 {
		t.Fatal("target wildcard")
	}
	var before, after bytes.Buffer
	k.Write(&before)
	for _, bad := range []string{`{"version":1,"version":1}`, `{"Version":1}`, before.String() + `{}`, strings.Replace(before.String(), `"confidence":0.75`, `"confidence":0.25`, 1), strings.Repeat(" ", strategyBytes+1)} {
		if err := k.Load(strings.NewReader(bad)); err == nil {
			t.Fatal("bad catalog admitted")
		}
		after.Reset()
		k.Write(&after)
		if !bytes.Equal(before.Bytes(), after.Bytes()) {
			t.Fatal("failed load mutated catalog")
		}
	}
}

func TestKernelBlasterPinnedCatalog(t *testing.T) {
	b, err := os.ReadFile("testdata/kernelblaster/optimization_database.json")
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("%x", sha256.Sum256(b)) != "4b93cbd87c533225b8be8a933683303c552d152845353dfaf0117d1ac864b4a9" {
		t.Fatal("reference pin changed")
	}
	k := new(StrategyKnowledgeBase)
	if err := k.LoadKernelBlaster(bytes.NewReader(b), "sm_121"); err != nil {
		t.Fatal(err)
	}
	groups := make(map[string]bool)
	for _, family := range []cudatrial.BottleneckFamily{cudatrial.MemoryBandwidthLimited, cudatrial.ComputeThroughputLimited, cudatrial.LatencyOccupancyLimited, cudatrial.MemoryLatencyBound, cudatrial.MemoryBankConflicts, cudatrial.CacheInefficient, cudatrial.HybridBound, cudatrial.APIOverheadDominant} {
		q := k.Query(family, "sm_121")
		if len(q) == 0 {
			t.Fatal("missing category", family)
		}
		for _, s := range q {
			groups[s.SourceGroup] = true
			if len(s.HistoricalSpeedups) != 0 {
				t.Fatal("upstream prediction promoted")
			}
		}
	}
	if len(groups) != 16 {
		t.Fatal(groups)
	}
	bad := bytes.Replace(b, []byte(`"memory_bandwidth_saturated":`), []byte(`"unknown_group":`), 1)
	if err := k.LoadKernelBlaster(bytes.NewReader(bad), "sm_121"); err == nil {
		t.Fatal("unknown group admitted")
	}
}

func cachedTrajectoryFixture() cudatrial.TrajectoryReceipt {
	d := strings.Repeat("a", 64)
	r := cudatrial.TrajectoryReceipt{Version: 1, WorkloadKey: d, Domain: d, TargetArch: "sm_121", ArtifactDigest: d, CompilerKey: d, BaselineArtifactDigest: d, BaselineCompilerKey: d, InputDigest: d, PolicyID: d, AppliedRules: []string{"tile"}, BottleneckState: cudatrial.MemoryBandwidthLimited, MeasuredLatencyNs: 50, BaselineLatencyNs: 100, SpeedupRatio: 2, VerifiedCorrect: true, DeviceUUID: "synthetic-integrity-fixture", LockPID: 123}
	b, _ := json.Marshal(r)
	r.SHA256 = fmt.Sprintf("%x", sha256.Sum256(b))
	return r
}

func TestMutant_CorruptTrajectoryQuarantined(t *testing.T) {
	c, err := New(Config{RootDir: t.TempDir()})
	if err != nil {
		t.Fatal(err)
	}
	r := cachedTrajectoryFixture()
	ctx := context.Background()
	if err := c.PutTrustedTrajectory(ctx, r); err != nil {
		t.Fatal(err)
	}
	got, err := c.GetTrajectory(ctx, r.SHA256)
	if err != nil || got.SpeedupRatio != 2 {
		t.Fatalf("valid: %+v %v", got, err)
	}
	path := filepath.Join(c.receiptsDir, "trajectories", r.SHA256+".json")
	b, _ := os.ReadFile(path)
	b = bytes.Replace(b, []byte(`"tile"`), []byte(`"evil"`), 1)
	if err := os.WriteFile(path, b, 0600); err != nil {
		t.Fatal(err)
	}
	if _, err := c.GetTrajectory(ctx, r.SHA256); !errors.Is(err, ErrCorruptReceipt) {
		t.Fatalf("corrupt returned: %v", err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatal("corrupt source not quarantined")
	}
	q, _ := os.ReadDir(c.quarantineDir)
	if len(q) != 1 {
		t.Fatal("missing quarantine", q)
	}
	if _, err := c.GetTrajectory(ctx, "../escape"); err == nil {
		t.Fatal("unsafe key")
	}
}

func TestKnowledgeBaseTrajectoryConfidenceAndFailures(t *testing.T) {
	k := new(StrategyKnowledgeBase)
	s := strategyFixture()
	s.Confidence = 0.5
	if err := k.Update(s); err != nil {
		t.Fatal(err)
	}
	r := cachedTrajectoryFixture()
	mapping := map[string]string{"tile": s.Technique}
	if err := k.RecordTrustedTrajectory(r, mapping); err != nil {
		t.Fatal(err)
	}
	if err := k.RecordTrustedTrajectory(r, mapping); err != nil {
		t.Fatal(err)
	}
	q := k.Query(s.Family, s.TargetArch)[0]
	if q.Observations != 1 || q.Confidence != 0.75 || len(q.HistoricalSpeedups) != 1 || q.HistoricalSpeedups[0] != 2 {
		t.Fatalf("success/duplicate: %+v", q)
	}
	r.VerifiedCorrect = false
	r.SHA256 = ""
	b, _ := json.Marshal(r)
	r.SHA256 = fmt.Sprintf("%x", sha256.Sum256(b))
	if err := k.RecordTrustedTrajectory(r, mapping); err != nil {
		t.Fatal(err)
	}
	q = k.Query(s.Family, s.TargetArch)[0]
	if q.Observations != 2 || q.Failures != 1 || q.Confidence != 0.5 || len(q.HistoricalSpeedups) != 1 {
		t.Fatalf("failure promoted: %+v", q)
	}
	r.PendingHardware = true
	if err := k.RecordTrustedTrajectory(r, mapping); !errors.Is(err, ErrUnqualifiedOutcome) {
		t.Fatal("pending history admitted", err)
	}
	q = k.Query(s.Family, s.TargetArch)[0]
	if q.Observations != 2 {
		t.Fatal("failed update mutated catalog")
	}
}
