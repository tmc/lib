package cudaphase

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math/rand/v2"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
)

// Slots of the clock-mode rows these tests build.
const (
	cStart, cMid, cMid2, cEnd = 0, 1, 2, 3 // cudaphase_clock
	tStart, tEnd              = 8, 9       // cudaphase_now
	smStart, smEnd            = 10, 11     // cudaphase_smid
	clockSlots                = 12
)

var (
	testClock  = Clock{Start: cStart, End: cEnd, TimerStart: tStart, TimerEnd: tEnd, SMEnd: smEnd}
	clockPhase = []Phase{{"load", cStart, cMid}, {"math", cMid, cMid2}, {"store", cMid2, cEnd}}
)

// A planted block runs phases of the given nanoseconds from globaltimer
// start t0, on an SM whose clock reads offset at that moment and runs at
// 2 GHz.
type planted struct {
	t0, offset uint64
	ns         [3]uint64
	sm         uint64
}

func (p planted) row() []uint64 {
	r := make([]uint64, clockSlots)
	r[cStart] = p.offset
	r[cMid] = r[cStart] + 2*p.ns[0]
	r[cMid2] = r[cMid] + 2*p.ns[1]
	r[cEnd] = r[cMid2] + 2*p.ns[2]
	r[tStart] = p.t0
	r[tEnd] = p.t0 + p.ns[0] + p.ns[1] + p.ns[2]
	r[smStart], r[smEnd] = p.sm, p.sm
	return r
}

// TestSummarizeClockPlanted plants phase times on per-SM clocks at 2 GHz
// and checks that SummarizeClock recovers them exactly, with blocks that
// must not move the rate: one too short to measure, one whose
// globaltimer end is off by a timer step, and one that changed SMs, with
// cycles from two unrelated clocks.
func TestSummarizeClockPlanted(t *testing.T) {
	blocks := []planted{
		{t0: 1_000_000, offset: 7_000_000_000, ns: [3]uint64{900, 2500, 300}, sm: 1},
		{t0: 1_000_256, offset: 123_456_789, ns: [3]uint64{1100, 2400, 310}, sm: 2},
		{t0: 1_000_512, offset: 99, ns: [3]uint64{1000, 2600, 290}, sm: 3},
		{t0: 1_001_024, offset: 1 << 40, ns: [3]uint64{950, 2450, 305}, sm: 4},
		{t0: 1_002_048, offset: 555, ns: [3]uint64{1050, 2550, 295}, sm: 5},
		{t0: 1_000_000, offset: 31_000_000, ns: [3]uint64{200, 300, 100}, sm: 6},  // short
		{t0: 1_003_000, offset: 8_000_000, ns: [3]uint64{1000, 2500, 300}, sm: 7}, // outlier timer
		{t0: 1_000_768, offset: 9_000_000, ns: [3]uint64{1000, 2500, 300}, sm: 8}, // migrates
	}
	var buf []uint64
	for i, b := range blocks {
		r := b.row()
		switch i {
		case 5:
			r[tEnd] += 256 // a short span is mostly timer step
		case 6:
			r[tEnd] += 256
		case 7:
			r[smEnd] = 9
			r[cEnd] = 3 // another SM's clock
		}
		buf = append(buf, r...)
	}
	buf = append(buf, make([]uint64, clockSlots)...) // a block that never ran

	s, err := SummarizeClock("planted", buf, clockSlots, clockPhase, smStart, testClock)
	if err != nil {
		t.Fatal(err)
	}
	if s.GHz != 2 || s.RateBlocks != 6 || s.Migrated != 1 || s.Blocks != 8 || s.SMs != 7 {
		t.Errorf("ghz %v from %d blocks, %d migrated, %d blocks on %d SMs; want 2 from 6, 1, 8 on 7",
			s.GHz, s.RateBlocks, s.Migrated, s.Blocks, s.SMs)
	}
	var first, last uint64 = 1 << 63, 0
	for _, b := range blocks[:7] {
		first = min(first, b.t0)
		last = max(last, b.t0+b.ns[0]+b.ns[1]+b.ns[2])
	}
	if want := float64(last-first) / 1e3; s.SpanUS != want {
		t.Errorf("span %v us, want %v", s.SpanUS, want)
	}
	for k, p := range s.Phases {
		var want []float64
		for _, b := range blocks[:7] {
			want = append(want, float64(b.ns[k])/1e3)
		}
		slices.Sort(want)
		got := []float64{p.MinUS, p.MedianUS, p.MaxUS}
		if exp := []float64{want[0], want[(len(want)-1)/2], want[len(want)-1]}; p.Blocks != 7 || !slices.Equal(got, exp) {
			t.Errorf("%s: %d blocks, min/median/max %v; want 7, %v", p.Name, p.Blocks, got, exp)
		}
	}
}

func TestSummarizeClockErrors(t *testing.T) {
	short := planted{t0: 1000, offset: 5, ns: [3]uint64{100, 100, 100}, sm: 1}.row()
	tests := []struct {
		name   string
		buf    []uint64
		smSlot int
		c      Clock
		want   string
	}{
		{"no block long enough", short, smStart, testClock, "no block spans 2000 ns"},
		{"min span", short, smStart, Clock{Start: cStart, End: cEnd, TimerStart: tStart, TimerEnd: tEnd, SMEnd: -1, MinSpanNS: 301}, "no block spans 301 ns"},
		{"clock slot", short, smStart, Clock{Start: cStart, End: 12, TimerStart: tStart, TimerEnd: tEnd, SMEnd: -1}, "clock slot 12 out of range"},
		{"SM end without SM", short, -1, testClock, "SM end slot 11 needs an SM slot"},
	}
	for _, tt := range tests {
		_, err := SummarizeClock("x", tt.buf, clockSlots, clockPhase, tt.smSlot, tt.c)
		if err == nil || !strings.Contains(err.Error(), tt.want) {
			t.Errorf("%s: err %v, want %q", tt.name, err, tt.want)
		}
	}
	if _, err := SummarizeClock("x", short, clockSlots, clockPhase, smStart, Clock{Start: cStart, End: cEnd, TimerStart: tStart, TimerEnd: tEnd, SMEnd: -1, MinSpanNS: 300}); err != nil {
		t.Errorf("min span 300 on a 300 ns block: %v", err)
	}
}

func TestPythonMatchesGoClock(t *testing.T) {
	py, err := exec.LookPath("python3")
	if err != nil {
		t.Skip("no python3")
	}
	r := rand.New(rand.NewPCG(7, 8))
	var script strings.Builder
	script.WriteString(Python)
	var bufs [][]uint64
	for c := 0; c < 20; c++ {
		var buf []uint64
		for b := 1 + r.IntN(30); b > 0; b-- {
			p := planted{t0: 1_000_000 + uint64(r.IntN(4000))*256, offset: r.Uint64N(1 << 50), sm: uint64(r.IntN(9))}
			for k := range p.ns {
				p.ns[k] = uint64(r.IntN(3000))
			}
			row := p.row()
			row[tEnd] += uint64(r.IntN(3)) * 256 // timer steps
			row[cEnd] += uint64(r.IntN(50))      // clock drift
			switch r.IntN(8) {
			case 0:
				row = make([]uint64, clockSlots) // never ran
			case 1:
				row[smEnd]++ // migrated
			case 2:
				row[1+r.IntN(2)] = 0 // a stamp never written
			}
			buf = append(buf, row...)
		}
		// One block long enough to measure the rate.
		buf = append(buf, planted{t0: 999_744, offset: 77, ns: [3]uint64{3000, 1, 1}, sm: 3}.row()...)
		bufs = append(bufs, buf)
		fmt.Fprintf(&script, "print(cudaphase_summary(%q, %s, [(\"load\", 0, 1), (\"math\", 1, 2), (\"store\", 2, 3)], slots=12, sm_slot=10, clock={\"start\": 0, \"end\": 3, \"timer_start\": 8, \"timer_end\": 9, \"sm_end\": 11}))\n",
			fmt.Sprint("case", c), strings.Join(strings.Fields(fmt.Sprint(buf)), ", "))
	}
	out, err := exec.Command(py, "-c", script.String()).CombinedOutput()
	if err != nil {
		t.Fatalf("python3: %v\n%s", err, out)
	}
	got, err := Parse(strings.NewReader(string(out)))
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(bufs) {
		t.Fatalf("python printed %d summaries, want %d\n%s", len(got), len(bufs), out)
	}
	for c, buf := range bufs {
		want, err := SummarizeClock(fmt.Sprint("case", c), buf, clockSlots, clockPhase, smStart, testClock)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got[c], want) {
			t.Errorf("case %d: python\n%+v\ngo\n%+v", c, got[c], want)
		}
	}
}

// TestClockKernel times a busy loop of a known number of cycles. Under
// cudahostemu a cycle is a nanosecond, so the measured clock is 1 GHz.
func TestClockKernel(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "cudaphase.h"), []byte(Header), 0o644); err != nil {
		t.Fatal(err)
	}
	const src = `
#include "cudaphase.h"
__global__ void spin(unsigned long long* trace, long long n) {
    CUDAPHASE_MARK(trace, 0, cudaphase_clock());
    CUDAPHASE_MARK(trace, 8, cudaphase_now());
    CUDAPHASE_MARK(trace, 10, cudaphase_smid());
    unsigned long long c0 = cudaphase_clock();
    while ((long long)(cudaphase_clock() - c0) < n) {
    }
    CUDAPHASE_MARK(trace, 1, cudaphase_clock());
    CUDAPHASE_MARK(trace, 3, cudaphase_clock());
    CUDAPHASE_MARK(trace, 9, cudaphase_now());
    CUDAPHASE_MARK(trace, 11, cudaphase_smid());
}`
	const blocks, cycles = 4, 3_000_000
	out, err := cudahostemu.Run(context.Background(), dir, cudahostemu.Program{
		Source:  src,
		Flags:   []string{"-I", dir},
		Launch:  fmt.Sprintf("eg_emu_launch(dim3(%d), dim3(1), 0, spin, trace, %dLL)", blocks, cycles),
		Buffers: []cudahostemu.Buffer{{Name: "trace", CType: "unsigned long long", Data: make([]byte, blocks*16*8)}},
	})
	if errors.Is(err, cudahostemu.ErrNoCompiler) {
		t.Skip(err)
	}
	if err != nil {
		t.Fatal(err)
	}
	buf := make([]uint64, blocks*16)
	for i := range buf {
		buf[i] = binary.LittleEndian.Uint64(out["trace"][8*i:])
	}
	c := Clock{Start: 0, End: 3, TimerStart: 8, TimerEnd: 9, SMEnd: 11}
	s, err := SummarizeClock("spin", buf, 16, []Phase{{"spin", 0, 1}}, 10, c)
	if err != nil {
		t.Fatal(err)
	}
	if s.RateBlocks != blocks || s.GHz < 0.99 || s.GHz > 1.01 {
		t.Errorf("ghz %v from %d blocks, want 1 from %d", s.GHz, s.RateBlocks, blocks)
	}
	p := s.Phases[0]
	if spun := p.MinUS * 1e3 * s.GHz; p.Blocks != blocks || spun < cycles || spun > 2*cycles {
		t.Errorf("spin: %d blocks, %.0f cycles at least; want %d blocks, %d to %d", p.Blocks, spun, blocks, cycles, 2*cycles)
	}
}
