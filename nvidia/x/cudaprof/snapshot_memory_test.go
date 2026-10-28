package cudaprof_test

import (
	"errors"
	"fmt"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"testing"

	"github.com/tmc/lib/egraphcore"
	"github.com/tmc/lib/nvidia/x/cudaprof"
)

var errLedgerFull = errors.New("ledger full")

// ledger is a Memory with a fixed limit.
type ledger struct{ limit, live int64 }

func (l *ledger) Reserve(n int64) error {
	if n > l.limit-l.live {
		return errLedgerFull
	}
	l.live += n
	return nil
}

func (l *ledger) Release(n int64) { l.live -= n }

func scopes(t *testing.T, l *ledger) (scratch, result *egraphcore.MemoryScope) {
	t.Helper()
	scratch, err := egraphcore.NewMemoryScope(l)
	if err != nil {
		t.Fatal(err)
	}
	result, err = egraphcore.NewMemoryScope(l)
	if err != nil {
		t.Fatal(err)
	}
	return scratch, result
}

func nestedSnapshot(pcs int) *cudaprof.ProfileSnapshot {
	s := validProfileSnapshot()
	s.Metrics = map[string]int64{"stall_mem": 5, "stall_exec": 7}
	s.MetricUnits = map[string]string{"stall_mem": "samples", "stall_exec": "samples", "stall_pc": "cycles"}
	s.PCStalls = make(map[uint64]map[string]int64)
	for i := 0; i < pcs; i++ {
		// Repeated counts exercise the ascending-PC tie break.
		s.PCStalls[uint64(0x100+16*i)] = map[string]int64{"stall_mem": int64(i % 7), "stall_exec": int64(i % 3), "stall_pc": 2}
	}
	s.PCStalls[1] = map[string]int64{}
	return &s
}

// referenceHotPCs ranks every PC by sorting all hits.
func referenceHotPCs(s *cudaprof.ProfileSnapshot) map[string][]uint64 {
	type hit struct {
		pc    uint64
		count int64
	}
	byMetric := make(map[string][]hit)
	for pc, counters := range s.PCStalls {
		for metric, count := range counters {
			if count > 0 {
				byMetric[metric] = append(byMetric[metric], hit{pc, count})
			}
		}
	}
	want := make(map[string][]uint64)
	for metric, hits := range byMetric {
		sort.Slice(hits, func(i, j int) bool {
			if hits[i].count == hits[j].count {
				return hits[i].pc < hits[j].pc
			}
			return hits[i].count > hits[j].count
		})
		for i := 0; i < len(hits) && i < 10; i++ {
			want[metric] = append(want[metric], hits[i].pc)
		}
	}
	return want
}

func TestGuidanceBoundedMatchesGuidance(t *testing.T) {
	for _, tc := range []struct {
		name string
		snap *cudaprof.ProfileSnapshot
	}{
		{"small", func() *cudaprof.ProfileSnapshot { s := validProfileSnapshot(); return &s }()},
		{"many", nestedSnapshot(500)},
		{"no_pcs", func() *cudaprof.ProfileSnapshot {
			s := validProfileSnapshot()
			s.PCStalls = nil
			return &s
		}()},
		{"sampling", func() *cudaprof.ProfileSnapshot {
			s := validProfileSnapshot()
			s.Sampling = &cudaprof.SamplingProvenance{RunID: "run", ModuleGeneration: 1, KernelName: "k", CUPTIVersion: 1, CollectionMode: 1, TotalSamples: 10}
			return &s
		}()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			target := profileTarget(tc.snap, "work", "", "device")
			want, err := tc.snap.Guidance(target)
			if err != nil {
				t.Fatal(err)
			}
			l := &ledger{limit: 1 << 30}
			scratch, result := scopes(t, l)
			got, err := tc.snap.GuidanceBounded(target, nil, scratch, result)
			if err != nil || !reflect.DeepEqual(got, want) {
				t.Fatalf("bounded guidance differs: err=%v\n got=%+v\nwant=%+v", err, got, want)
			}
			if ref := referenceHotPCs(tc.snap); !reflect.DeepEqual(got.HotPCs, ref) && len(ref)+len(got.HotPCs) != 0 {
				t.Fatalf("hot PCs=%v want %v", got.HotPCs, ref)
			}
			if result.Bytes() <= scratch.Bytes() {
				t.Fatalf("result owner holds %d bytes, scratch %d", result.Bytes(), scratch.Bytes())
			}
			scratch.Close()
			scratch.Close()
			if got.StallCounts["stall_mem"] != 5 {
				t.Fatal("guidance depends on scratch")
			}
			result.Close()
			result.Close()
			if l.live != 0 {
				t.Fatalf("live=%d after closing both scopes", l.live)
			}
		})
	}
}

func TestGuidanceBoundedRefusalBeforeCopy(t *testing.T) {
	snap := nestedSnapshot(2000)
	target := profileTarget(snap, "work", "", "device")
	l := &ledger{limit: 4096}
	scratch, result := scopes(t, l)
	before := l.live
	got, err := snap.GuidanceBounded(target, nil, scratch, result)
	if got != nil || !errors.Is(err, errLedgerFull) {
		t.Fatalf("guidance=%+v err=%v", got, err)
	}
	if l.live != before || result.Err() == nil {
		t.Fatalf("refusal changed the ledger: live=%d before=%d err=%v", l.live, before, result.Err())
	}
	scratch.Close()
	result.Close()
	if l.live != 0 {
		t.Fatalf("live=%d after close", l.live)
	}
}

func TestGuidanceBoundedAdmit(t *testing.T) {
	snap := nestedSnapshot(2000)
	target := profileTarget(snap, "work", "", "device")
	refused := errors.New("refused")
	l := &ledger{limit: 1 << 30}
	scratch, result := scopes(t, l)
	before := l.live
	calls := 0
	got, err := snap.GuidanceBounded(target, func(g *cudaprof.SearchGuidance) error {
		calls++
		if !g.Available || g.StallCounts["stall_exec"] != 7 || g.MetricUnits["stall_pc"] != "cycles" || g.PCStalls != nil || g.HotPCs != nil {
			t.Errorf("unexpected admission view: %+v", g)
		}
		return refused
	}, scratch, result)
	if got != nil || err != refused || calls != 1 || l.live != before {
		t.Fatalf("guidance=%v err=%v calls=%d live=%d before=%d", got, err, calls, l.live, before)
	}
	// Identity failures precede admission.
	bad := target
	bad.Device = "other"
	if _, err := snap.GuidanceBounded(bad, func(*cudaprof.SearchGuidance) error {
		t.Error("admitted a mismatched target")
		return nil
	}, scratch, result); !errors.Is(err, cudaprof.ErrDeviceMismatch) || l.live != before {
		t.Fatalf("err=%v live=%d before=%d", err, l.live, before)
	}
	scratch.Close()
	result.Close()
}

func TestGuidanceBoundedDetaches(t *testing.T) {
	snap := nestedSnapshot(50)
	target := profileTarget(snap, "work", "", "device")
	l := &ledger{limit: 1 << 30}
	scratch, result := scopes(t, l)
	got, err := snap.GuidanceBounded(target, nil, scratch, result)
	if err != nil {
		t.Fatal(err)
	}
	want, err := snap.Guidance(target)
	if err != nil {
		t.Fatal(err)
	}
	snap.Metrics["stall_mem"] = 0
	snap.MetricUnits["stall_mem"] = "cycles"
	for _, counters := range snap.PCStalls {
		counters["stall_mem"] = 99
	}
	delete(snap.PCStalls, 0x100)
	if !reflect.DeepEqual(got, want) {
		t.Fatal("source mutation reached retained guidance")
	}
	scratch.Close()
	result.Close()
}

func TestGuidanceBoundedUnavailable(t *testing.T) {
	snap := nestedSnapshot(10)
	delete(snap.MetricUnits, "stall_exec")
	target := profileTarget(snap, "work", "", "device")
	got, err := snap.Guidance(target)
	if !errors.Is(err, cudaprof.ErrGuidanceUnavailable) || got == nil || got.MetricUnits == nil || !strings.Contains(got.UnavailableReason, "stall_exec") {
		t.Fatalf("unbounded unavailable guidance changed: %+v %v", got, err)
	}
	l := &ledger{limit: 1 << 30}
	scratch, result := scopes(t, l)
	before := result.Bytes()
	got, err = snap.GuidanceBounded(target, nil, scratch, result)
	if !errors.Is(err, cudaprof.ErrGuidanceUnavailable) || got == nil || got.Available || got.UnavailableReason == "" || !reflect.DeepEqual(got.MetricUnits, snap.MetricUnits) {
		t.Fatalf("bounded unavailable guidance: %+v %v", got, err)
	}
	if result.Bytes() <= before {
		t.Fatalf("unavailable guidance and its copies are not charged to result: %d", result.Bytes()-before)
	}
	// A result scope that cannot hold the copies refuses the guidance.
	tight := &ledger{limit: 1 << 30}
	scratch2, result2 := scopes(t, tight)
	tight.limit = tight.live + 8
	if got, err := snap.GuidanceBounded(target, nil, scratch2, result2); got != nil || !errors.Is(err, errLedgerFull) {
		t.Fatalf("uncharged unavailable guidance: %+v %v", got, err)
	}
	scratch2.Close()
	result2.Close()
	scratch.Close()
	result.Close()
	if l.live != 0 {
		t.Fatalf("live=%d", l.live)
	}
}

func TestNewProfileSnapshotCopies(t *testing.T) {
	raw := nestedSnapshot(20)
	got, err := cudaprof.NewProfileSnapshot(*raw)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, raw) {
		t.Fatalf("copy differs:\n got=%+v\nwant=%+v", got, raw)
	}
	raw.Metrics["stall_mem"] = 0
	raw.PCStalls[0x110]["stall_mem"] = 0
	raw.MetricUnits["stall_mem"] = ""
	if got.Metrics["stall_mem"] != 5 || got.PCStalls[0x110]["stall_mem"] != 1 || got.MetricUnits["stall_mem"] != "samples" {
		t.Fatal("copy aliases the source")
	}
	if fmt.Sprint(got.PCStalls[1]) != "map[]" || got.PCStalls[1] == nil {
		t.Fatalf("empty inner map not preserved: %#v", got.PCStalls[1])
	}
}

// allocProbe is a Memory that counts reservations. A scratch probe records
// runtime.MemStats.TotalAlloc at reservation sample; a result probe refuses
// its first reservation after scratch has sampled and records TotalAlloc then.
type allocProbe struct {
	calls, sample int
	at            uint64
	scratch       *allocProbe
}

func (p *allocProbe) Reserve(n int64) error {
	p.calls++
	if p.scratch == nil && p.calls != p.sample || p.scratch != nil && (p.scratch.at == 0 || p.at != 0) {
		return nil
	}
	var s runtime.MemStats
	runtime.ReadMemStats(&s)
	p.at = s.TotalAlloc
	if p.scratch != nil {
		return errLedgerFull
	}
	return nil
}

func (*allocProbe) Release(int64) {}

func TestGuidanceBoundedHotPCsReservedBeforeAllocation(t *testing.T) {
	const metrics = 50000
	snap := validProfileSnapshot()
	counters := make(map[string]int64, metrics)
	for i := range metrics {
		name := fmt.Sprint("m", i)
		counters[name] = 1
		snap.MetricUnits[name] = "samples"
	}
	snap.PCStalls = map[uint64]map[string]int64{16: counters}
	target := profileTarget(&snap, "work", "", "device")
	run := func(scratchProbe, resultProbe *allocProbe) (*cudaprof.SearchGuidance, error) {
		result, err := egraphcore.NewMemoryScope(resultProbe)
		if err != nil {
			t.Fatal(err)
		}
		defer result.Close()
		scratch, err := egraphcore.NewMemoryScope(scratchProbe)
		if err != nil {
			t.Fatal(err)
		}
		defer scratch.Close()
		return snap.GuidanceBounded(target, nil, scratch, result)
	}

	// An unlimited run counts the scratch reservations; the last one is
	// made for the last metric's candidates.
	count := new(allocProbe)
	if _, err := run(count, new(allocProbe)); err != nil {
		t.Fatal(err)
	}
	sp := &allocProbe{sample: count.calls}
	rp := &allocProbe{scratch: sp}
	got, err := run(sp, rp)
	if got != nil || !errors.Is(err, errLedgerFull) || rp.at == 0 {
		t.Fatalf("guidance=%v err=%v refused=%v", got != nil, err, rp.at != 0)
	}
	// Between the last selection reservation and the refused hot-PC
	// reservation only the last candidate slice may be created, not the
	// output map for every metric.
	d := rp.at - sp.at
	if d > 1<<20 {
		t.Fatalf("allocated %d bytes before the hot-PC map was reserved", d)
	}
	t.Logf("allocated %d bytes before the hot-PC map was reserved", d)
}
