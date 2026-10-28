package autotune_test

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahotswap"
	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
	"github.com/tmc/lib/nvidia/x/egraph/autotune"
)

// knobBuilder emits each candidate's knob preamble as its source.
type knobBuilder struct{ twin autotune.HardwareTwin }

func (b knobBuilder) Build(_ context.Context, c autotune.ScheduleCandidate) (autotune.Variant, error) {
	src, err := autotune.KnobPreamble(b.twin, c)
	if err != nil {
		return autotune.Variant{}, err
	}
	return autotune.Variant{Candidate: c, Source: cudahotswap.Source{Code: src, Entry: "gemm", Filename: c.Name + ".cu"}}, nil
}

// synthetic is a host model of a device: per-candidate median latency,
// stall histogram and range metrics.
type synthetic struct {
	us      map[string]float64
	stalls  map[string]autotune.StallHistogram
	metrics map[string]autotune.RangeMetrics
	sampled []string
}

func (s *synthetic) SampleKernel(_ context.Context, v autotune.Variant, iters int) (autotune.ProfileObservation, autotune.StallHistogram, error) {
	name := v.Candidate.Name
	us, ok := s.us[name]
	if !ok {
		return autotune.ProfileObservation{}, autotune.StallHistogram{}, fmt.Errorf("no model for %s", name)
	}
	s.sampled = append(s.sampled, name)
	m := s.metrics[name]
	if m.RegisterCount == 0 {
		m.RegisterCount = 168
	}
	return autotune.ProfileObservation{MedianUS: us, Metrics: m}, s.stalls[name], nil
}

type recordSwapper struct{ swaps []string }

func (r *recordSwapper) Swap(_ context.Context, v autotune.Variant) error {
	r.swaps = append(r.swaps, v.Candidate.Name)
	return nil
}

func mbarrier(frac float64) autotune.StallHistogram {
	return autotune.StallHistogram{TotalSamples: 1000, WaitMBarrier: uint64(frac * 1000), MathThrottle: uint64((1 - frac) * 300)}
}

// tmaCandidates returns schedules where the analytical model prefers
// shallow pipelines and cluster or swizzle variants of them, while the
// synthetic device is mbarrier-bound until four TMA stages.
func tmaCandidates() ([]autotune.ScheduleCandidate, *synthetic) {
	cands := []autotune.ScheduleCandidate{
		{Name: "s2", Stages: 2, CTAGroup: 1, AnalyticalCostUS: 10},
		{Name: "s2_swz", Stages: 2, CTAGroup: 1, Swizzle128B: true, AnalyticalCostUS: 10.2},
		{Name: "s2_cta2", Stages: 2, CTAGroup: 2, AnalyticalCostUS: 10.4},
		{Name: "s2_sparse", Stages: 2, CTAGroup: 1, UseSparse2to4: true, AnalyticalCostUS: 10.6},
		{Name: "s3", Stages: 3, CTAGroup: 1, AnalyticalCostUS: 12},
		{Name: "s4", Stages: 4, CTAGroup: 1, AnalyticalCostUS: 14},
		{Name: "s6", Stages: 6, CTAGroup: 1, AnalyticalCostUS: 16},
	}
	dev := &synthetic{
		us: map[string]float64{"s2": 30, "s2_swz": 29.5, "s2_cta2": 29, "s2_sparse": 29.8, "s3": 22, "s4": 15, "s6": 16},
		stalls: map[string]autotune.StallHistogram{
			"s2": mbarrier(0.60), "s2_swz": mbarrier(0.60), "s2_cta2": mbarrier(0.58), "s2_sparse": mbarrier(0.60),
			"s3": mbarrier(0.45), "s4": mbarrier(0.10), "s6": mbarrier(0.05),
		},
		metrics: map[string]autotune.RangeMetrics{},
	}
	return cands, dev
}

func newDaemon(t *testing.T, dev autotune.LiveCollector, swap autotune.Swapper, agree autotune.Agreement) *autotune.Daemon {
	t.Helper()
	twin, err := autotune.TwinForTarget("sm_100a")
	if err != nil {
		t.Fatal(err)
	}
	d, err := autotune.NewDaemon(autotune.DaemonConfig{
		Twin:      twin,
		ShapeKey:  "llama3-block-m128",
		MaxTrials: 2,
		Builder:   knobBuilder{twin},
		Swapper:   swap,
		Agreement: agree,
	}, dev)
	if err != nil {
		t.Fatal(err)
	}
	return d
}

func TestDaemonMBarrierSpikeShiftsToFourStages(t *testing.T) {
	cands, dev := tmaCandidates()
	swap := &recordSwapper{}
	d := newDaemon(t, dev, swap, nil)
	var res autotune.StepResult
	for i := range 2 {
		var err error
		res, err = d.Step(t.Context(), cands)
		if err != nil {
			t.Fatalf("step %d: %v", i, err)
		}
		t.Logf("step %d: sampled %v winner %s %.1fus speedup %.2f weights %+v", i, dev.sampled, res.Winner.Name, res.WinnerUS, res.Speedup, res.Weights)
	}
	if res.Winner.Stages != 4 || res.Serving.Name != "s4" {
		t.Fatalf("after 2 steps winner = %s (%d stages), serving %s; want s4", res.Winner.Name, res.Winner.Stages, res.Serving.Name)
	}
	if got, want := strings.Join(swap.swaps, ","), "s3,s4"; got != want {
		t.Errorf("swaps = %s, want %s", got, want)
	}
	if res.BaselineUS != 30 || res.Speedup != 2 {
		t.Errorf("baseline %.1fus speedup %.2f, want 30us and 2", res.BaselineUS, res.Speedup)
	}
}

// Without stall direction the analytical order samples the two-stage
// variants first and does not reach four stages in two steps. This is the
// control for the test above.
func TestDaemonWithoutStallsStaysShallow(t *testing.T) {
	cands, dev := tmaCandidates()
	for k, h := range dev.stalls {
		// Misattribute the waits to a reason no rule acts on.
		dev.stalls[k] = autotune.StallHistogram{TotalSamples: h.TotalSamples}
	}
	d := newDaemon(t, dev, &recordSwapper{}, nil)
	var res autotune.StepResult
	for range 2 {
		var err error
		if res, err = d.Step(t.Context(), cands); err != nil {
			t.Fatal(err)
		}
	}
	if res.Winner.Stages == 4 {
		t.Fatalf("reached 4 stages without stall feedback (sampled %v)", dev.sampled)
	}
}

func TestDaemonSpillDisqualified(t *testing.T) {
	cands, dev := tmaCandidates()
	dev.us["s3"] = 5 // fastest, but spills
	dev.metrics["s3"] = autotune.RangeMetrics{LocalSpillStoreBytes: 64, LocalSpillLoadBytes: 64}
	swap := &recordSwapper{}
	d := newDaemon(t, dev, swap, nil)
	res, err := d.Step(t.Context(), cands)
	if err != nil {
		t.Fatal(err)
	}
	var spillErr error
	for _, tr := range res.Trials {
		if tr.Candidate.Name == "s3" {
			spillErr = tr.Err
		}
	}
	if !errors.Is(spillErr, autotune.ErrSpillDisqualified) {
		t.Fatalf("s3 trial err = %v, want ErrSpillDisqualified", spillErr)
	}
	if res.Winner.Name == "s3" || len(swap.swaps) != 0 {
		t.Fatalf("spilling candidate selected: winner %s swaps %v", res.Winner.Name, swap.swaps)
	}
}

type driftOn string

func (d driftOn) Agree(_ context.Context, _, cand autotune.Variant) error {
	if cand.Candidate.Name == string(d) {
		return errors.New("max abs error 3e-2 > 1e-3")
	}
	return nil
}

func TestDaemonNumericalDriftBlocksSwap(t *testing.T) {
	cands, dev := tmaCandidates()
	swap := &recordSwapper{}
	d := newDaemon(t, dev, swap, driftOn("s3"))
	res, err := d.Step(t.Context(), cands)
	if err != nil {
		t.Fatal(err)
	}
	if res.Swapped || len(swap.swaps) != 0 || res.Serving.Name != "s2" {
		t.Fatalf("drifting s3 swapped in: serving %s swaps %v", res.Serving.Name, swap.swaps)
	}
	res, err = d.Step(t.Context(), cands)
	if err != nil {
		t.Fatal(err)
	}
	if res.Serving.Name != "s4" || strings.Join(swap.swaps, ",") != "s4" {
		t.Fatalf("serving %s swaps %v, want s4 swapped in", res.Serving.Name, swap.swaps)
	}
}

func TestDaemonImproveFloor(t *testing.T) {
	cands := []autotune.ScheduleCandidate{
		{Name: "a", Stages: 2, CTAGroup: 1, AnalyticalCostUS: 1},
		{Name: "b", Stages: 3, CTAGroup: 1, AnalyticalCostUS: 2},
	}
	dev := &synthetic{us: map[string]float64{"a": 100, "b": 99}, stalls: map[string]autotune.StallHistogram{}, metrics: map[string]autotune.RangeMetrics{}}
	swap := &recordSwapper{}
	d := newDaemon(t, dev, swap, nil) // ImproveFloor defaults to 1.5%
	res, err := d.Step(t.Context(), cands)
	if err != nil {
		t.Fatal(err)
	}
	if res.Winner.Name != "b" || res.Swapped || res.Serving.Name != "a" {
		t.Fatalf("1%% faster candidate: winner %s swapped %v serving %s; want winner b, no swap", res.Winner.Name, res.Swapped, res.Serving.Name)
	}
}

func TestDaemonCacheWarmStart(t *testing.T) {
	cands, dev := tmaCandidates()
	d := newDaemon(t, dev, nil, nil)
	for range 2 {
		if _, err := d.Step(t.Context(), cands); err != nil {
			t.Fatal(err)
		}
	}
	e, ok := d.CacheEntry()
	if !ok || e.Winner.Name != "s4" {
		t.Fatalf("CacheEntry = %+v, %v; want winner s4", e, ok)
	}
	var cache autotune.TuneCache
	cache.Put(autotune.CacheEntry{Fingerprint: "zz", ShapeKey: "other"})
	cache.Put(e)
	b1, err := json.Marshal(&cache)
	if err != nil {
		t.Fatal(err)
	}
	var back autotune.TuneCache
	if err := json.Unmarshal(b1, &back); err != nil {
		t.Fatal(err)
	}
	b2, _ := json.Marshal(&back)
	if string(b1) != string(b2) {
		t.Fatalf("cache JSON not deterministic:\n%s\n%s", b1, b2)
	}
	got, ok := back.Get(e.Fingerprint, e.ShapeKey)
	if !ok {
		t.Fatalf("cache lost entry %s/%s", e.Fingerprint, e.ShapeKey)
	}

	// A fresh daemon warmed from the cache finds s4 in its first step.
	_, dev2 := tmaCandidates()
	d2 := newDaemon(t, dev2, nil, nil)
	if err := d2.Warm(got); err != nil {
		t.Fatal(err)
	}
	res, err := d2.Step(t.Context(), cands)
	if err != nil {
		t.Fatal(err)
	}
	if res.Winner.Name != "s4" || len(dev2.sampled) != 2 {
		t.Fatalf("warm start: winner %s after sampling %v; want s4 after 2 samples", res.Winner.Name, dev2.sampled)
	}

	other := got
	other.ShapeKey = "different"
	if err := d2.Warm(other); err == nil {
		t.Errorf("Warm accepted an entry for another workload")
	}
}

func TestTwinFingerprint(t *testing.T) {
	a, _ := autotune.TwinForTarget("sm_100a")
	b, _ := autotune.TwinForTarget("sm_100a")
	if a.Fingerprint() != b.Fingerprint() {
		t.Fatalf("fingerprint not stable")
	}
	b.RackRanks = 1
	if a.Fingerprint() == b.Fingerprint() {
		t.Fatalf("fingerprint ignores RackRanks")
	}
}

func TestRefineRules(t *testing.T) {
	b200, _ := autotune.TwinForTarget("sm_100a")
	gb10, _ := autotune.TwinForTarget("sm_121a")
	c := autotune.ScheduleCandidate{Name: "c", Stages: 3, CTAGroup: 1}
	tests := []struct {
		name string
		twin autotune.HardwareTwin
		h    autotune.StallHistogram
		want autotune.ScheduleWeights
	}{
		{"mbarrier", b200, autotune.StallHistogram{TotalSamples: 100, WaitMBarrier: 41}, autotune.ScheduleWeights{PreferStages: 4}},
		{"mbarrier_below", b200, autotune.StallHistogram{TotalSamples: 100, WaitMBarrier: 40}, autotune.ScheduleWeights{}},
		{"long_scoreboard", b200, autotune.StallHistogram{TotalSamples: 100, LongScoreboard: 50}, autotune.ScheduleWeights{PreferStages: 4}},
		{"mio_tcgen05", b200, autotune.StallHistogram{TotalSamples: 100, MIOThrottle: 30}, autotune.ScheduleWeights{PreferSwizzle128B: true, PreferCTAGroup: 2}},
		{"mio_gb10", gb10, autotune.StallHistogram{TotalSamples: 100, MIOThrottle: 30}, autotune.ScheduleWeights{PreferSwizzle128B: true}},
		{"tmem", b200, autotune.StallHistogram{TotalSamples: 100, WaitTMEM: 31}, autotune.ScheduleWeights{PreferDoubleBufferTMEM: true}},
		{"tmem_gb10", gb10, autotune.StallHistogram{TotalSamples: 100, WaitTMEM: 31}, autotune.ScheduleWeights{}},
		{"math", b200, autotune.StallHistogram{TotalSamples: 100, MathThrottle: 36}, autotune.ScheduleWeights{PreferSparse2to4: true}},
		{"empty", b200, autotune.StallHistogram{}, autotune.ScheduleWeights{}},
	}
	for _, tt := range tests {
		if got := (autotune.ScheduleWeights{}).Refine(tt.twin, c, tt.h); got != tt.want {
			t.Errorf("%s: Refine = %+v, want %+v", tt.name, got, tt.want)
		}
	}
	deep := autotune.ScheduleCandidate{Name: "d", Stages: autotune.MaxStages}
	if got := (autotune.ScheduleWeights{}).Refine(b200, deep, autotune.StallHistogram{WaitMBarrier: 1}); got.PreferStages != autotune.MaxStages {
		t.Errorf("Refine at max stages prefers %d", got.PreferStages)
	}
}

func TestAttributionMap(t *testing.T) {
	ranges := []autotune.PCRange{
		{Start: 0x100, End: 0x200, Label: "tma_load"},
		{Start: 0x00, End: 0x100, Label: "prologue"},
		{Start: 0x200, End: 0x400, Label: "mma"},
	}
	stalls := map[uint64]autotune.StallHistogram{
		0x080: {TotalSamples: 5, LongScoreboard: 5},
		0x100: {TotalSamples: 10, WaitMBarrier: 8},
		0x1f0: {TotalSamples: 4, WaitMBarrier: 4},
		0x200: {TotalSamples: 7, MathThrottle: 7},
		0x400: {TotalSamples: 3, MIOThrottle: 3},
	}
	got, err := autotune.AttributionMap(ranges, stalls)
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]autotune.StallHistogram{
		"prologue":            {TotalSamples: 5, LongScoreboard: 5},
		"tma_load":            {TotalSamples: 14, WaitMBarrier: 12},
		"mma":                 {TotalSamples: 7, MathThrottle: 7},
		autotune.Unattributed: {TotalSamples: 3, MIOThrottle: 3},
	}
	if len(got) != len(want) {
		t.Fatalf("AttributionMap = %+v, want %+v", got, want)
	}
	for k, w := range want {
		if got[k] != w {
			t.Errorf("AttributionMap[%q] = %+v, want %+v", k, got[k], w)
		}
	}
	for _, bad := range [][]autotune.PCRange{
		{{Start: 0, End: 0x10, Label: "a"}, {Start: 0x8, End: 0x20, Label: "b"}},
		{{Start: 0x10, End: 0x10, Label: "empty"}},
		{{Start: 0, End: 0x10}},
	} {
		if _, err := autotune.AttributionMap(bad, stalls); err == nil {
			t.Errorf("AttributionMap(%v) = nil error", bad)
		}
	}
}

func TestHistogramsFromBatch(t *testing.T) {
	inv := cuptipcsampling.ReasonInventory{Reasons: []cuptipcsampling.StallReason{
		{Index: 1, Name: "smsp__pcsamp_warps_issue_stalled_long_scoreboard"},
		{Index: 2, Name: "smsp__pcsamp_warps_issue_stalled_long_scoreboard_not_issued"},
		{Index: 3, Name: "smsp__pcsamp_warps_issue_stalled_barrier"},
		{Index: 4, Name: "smsp__pcsamp_warps_issue_stalled_mio_throttle"},
		{Index: 5, Name: "smsp__pcsamp_warps_issue_stalled_math_pipe_throttle"},
		{Index: 6, Name: "smsp__pcsamp_warps_issue_stalled_selected"},
	}}
	b := cuptipcsampling.Batch{PCs: []cuptipcsampling.PC{
		{Offset: 0x10, Name: "k", Reasons: []cuptipcsampling.StallSamples{{Index: 1, Samples: 4}, {Index: 2, Samples: 4}, {Index: 6, Samples: 2}}},
		{Offset: 0x20, Name: "k", Reasons: []cuptipcsampling.StallSamples{{Index: 3, Samples: 6}, {Index: 4, Samples: 1}, {Index: 5, Samples: 3}}},
		{Offset: 0x10, Name: "other", Reasons: []cuptipcsampling.StallSamples{{Index: 3, Samples: 100}}},
	}}
	got := autotune.HistogramsFromBatch(b, inv, "k")
	want := map[uint64]autotune.StallHistogram{
		0x10: {TotalSamples: 6, LongScoreboard: 4},
		0x20: {TotalSamples: 10, WaitMBarrier: 6, MIOThrottle: 1, MathThrottle: 3},
	}
	for k, w := range want {
		if got[k] != w {
			t.Errorf("HistogramsFromBatch[%#x] = %+v, want %+v", k, got[k], w)
		}
	}
}

func TestKnobPreamble(t *testing.T) {
	b200, _ := autotune.TwinForTarget("sm_100a")
	gb10, _ := autotune.TwinForTarget("sm_121a")
	c := autotune.ScheduleCandidate{Name: "s4", Stages: 4, CTAGroup: 2, Swizzle128B: true, ProducerMaxRegs: 40, ConsumerMaxRegs: 232}
	src, err := autotune.KnobPreamble(b200, c)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"#define AUTOTUNE_TMA_STAGES 4\n",
		"#define AUTOTUNE_CTA_GROUP 2\n",
		"#define AUTOTUNE_SWIZZLE_128B 1\n",
		`setmaxnreg.dec.sync.aligned.u32 40;`,
		`setmaxnreg.inc.sync.aligned.u32 232;`,
	} {
		if !strings.Contains(src, want) {
			t.Errorf("preamble missing %q:\n%s", want, src)
		}
	}
	bad := []struct {
		name string
		twin autotune.HardwareTwin
		c    autotune.ScheduleCandidate
	}{
		{"one_stage", b200, autotune.ScheduleCandidate{Stages: 1, CTAGroup: 1}},
		{"seven_stages", b200, autotune.ScheduleCandidate{Stages: 7, CTAGroup: 1}},
		{"cta2_gb10", gb10, autotune.ScheduleCandidate{Stages: 2, CTAGroup: 2}},
		{"maxnreg_255", b200, autotune.ScheduleCandidate{Stages: 2, CTAGroup: 1, ConsumerMaxRegs: 255}},
		{"maxnreg_16", b200, autotune.ScheduleCandidate{Stages: 2, CTAGroup: 1, ProducerMaxRegs: 16}},
		{"consumer_below_producer", b200, autotune.ScheduleCandidate{Stages: 2, CTAGroup: 1, ProducerMaxRegs: 48, ConsumerMaxRegs: 40}},
		{"maxnreg_gb10", gb10, autotune.ScheduleCandidate{Stages: 2, CTAGroup: 1, ConsumerMaxRegs: 232}},
		{"tmem_gb10", gb10, autotune.ScheduleCandidate{Stages: 2, CTAGroup: 1, DoubleBufferTMEM: true}},
	}
	for _, tt := range bad {
		if _, err := autotune.KnobPreamble(tt.twin, tt.c); err == nil {
			t.Errorf("%s: KnobPreamble accepted invalid schedule", tt.name)
		}
	}
}

type fakeService struct {
	publish bool
	pushed  []string
}

func (f *fakeService) Push(_ context.Context, name string, src cudahotswap.Source, label string) (*cudahotswap.PushResult, error) {
	f.pushed = append(f.pushed, name+"/"+label+"/"+src.Entry)
	return &cudahotswap.PushResult{Published: f.publish}, nil
}

func TestServiceSwapper(t *testing.T) {
	v := autotune.Variant{Candidate: autotune.ScheduleCandidate{Name: "s4"}, Source: cudahotswap.Source{Entry: "gemm"}}
	ok := &fakeService{publish: true}
	if err := (autotune.ServiceSwapper{Service: ok, Name: "decode"}).Swap(t.Context(), v); err != nil {
		t.Fatal(err)
	}
	if got := strings.Join(ok.pushed, ","); got != "decode/s4/gemm" {
		t.Errorf("pushed %q", got)
	}
	refused := &fakeService{}
	if err := (autotune.ServiceSwapper{Service: refused, Name: "decode"}).Swap(t.Context(), v); err == nil {
		t.Errorf("Swap succeeded although the service did not publish")
	}
}

func TestCountSpills(t *testing.T) {
	dis, err := cudasass.ParseString(`
.section .text.gemm,"ax",@progbits
/* 0010 */   STL [R1], R4 ;
/* 0020 */   STL.64 [R1+0x8], R6 ;
/* 0030 */   STS [R0], R4 ;
/* 0040 */   LDL R8, [R1] ;
/* 0050 */   LDS R9, [R0] ;
/* 0060 */   LDS R10, [R0+0x4] ;
/* 0070 */   EXIT ;
`)
	if err != nil {
		t.Fatal(err)
	}
	stl, ldl, err := autotune.CountSpills(dis, "gemm")
	if err != nil {
		t.Fatal(err)
	}
	if stl != 2 || ldl != 1 {
		t.Errorf("CountSpills = %d STL, %d LDL; want 2, 1", stl, ldl)
	}
	if _, _, err := autotune.CountSpills(dis, "missing"); err == nil {
		t.Errorf("CountSpills(missing) = nil error")
	}
}

// spillBuilder reports SASS spills for one candidate.
type spillBuilder struct {
	knobBuilder
	spill string
}

func (b spillBuilder) Build(ctx context.Context, c autotune.ScheduleCandidate) (autotune.Variant, error) {
	v, err := b.knobBuilder.Build(ctx, c)
	if c.Name == b.spill {
		v.SpillStores, v.SpillLoads = 3, 2
	}
	return v, err
}

func TestDaemonSASSSpillNotSampled(t *testing.T) {
	cands, dev := tmaCandidates()
	dev.us["s3"] = 5
	twin, _ := autotune.TwinForTarget("sm_100a")
	d, err := autotune.NewDaemon(autotune.DaemonConfig{
		Twin: twin, MaxTrials: 2, Builder: spillBuilder{knobBuilder{twin}, "s3"},
	}, dev)
	if err != nil {
		t.Fatal(err)
	}
	res, err := d.Step(t.Context(), cands)
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range dev.sampled {
		if s == "s3" {
			t.Fatalf("SASS-spilling s3 was sampled: %v", dev.sampled)
		}
	}
	if res.Winner.Name == "s3" {
		t.Fatalf("SASS-spilling s3 won")
	}
	if !errors.Is(res.Trials[1].Err, autotune.ErrSpillDisqualified) {
		t.Fatalf("trial %s err = %v, want ErrSpillDisqualified", res.Trials[1].Candidate.Name, res.Trials[1].Err)
	}
	// A disqualified candidate is not retried.
	res, err = d.Step(t.Context(), cands)
	if err != nil {
		t.Fatal(err)
	}
	for _, tr := range res.Trials {
		if tr.Candidate.Name == "s3" {
			t.Fatalf("s3 retried in step 2")
		}
	}
}
