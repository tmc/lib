package timeline

import (
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
)

// spanOracle classifies every nanosecond from the first start to the last end
// by which classes of kernel cover it. It shares nothing with
// AccountDeviceSpan but the definitions, so ks must span few nanoseconds.
func spanOracle(ks []KernelInterval) DeviceSpan {
	var want DeviceSpan
	if len(ks) == 0 {
		return want
	}
	first, last := ks[0].Start, ks[0].End
	for _, k := range ks {
		first, last = min(first, k.Start), max(last, k.End)
		want.Launches++
		if k.Selected {
			want.SelectedLaunches++
		} else {
			want.OtherLaunches++
		}
	}
	want.SpanNs = last - first
	for ns := first; ns < last; ns++ {
		var selected, other bool
		for _, k := range ks {
			if k.Start <= ns && ns < k.End {
				if k.Selected {
					selected = true
				} else {
					other = true
				}
			}
		}
		switch {
		case selected && other:
			want.OverlapNs++
		case selected:
			want.SelectedNs++
		case other:
			want.OtherNs++
		default:
			want.IdleNs++
		}
	}
	return want
}

// translate returns ks moved so that the last end is math.MaxUint64.
func translate(ks []KernelInterval) []KernelInterval {
	var last uint64
	for _, k := range ks {
		last = max(last, k.End)
	}
	d := math.MaxUint64 - last
	out := slices.Clone(ks)
	for i := range out {
		out[i].Start += d
		out[i].End += d
	}
	return out
}

func TestDeviceSpanPartitionOracle(t *testing.T) {
	s := func(start, end uint64) KernelInterval {
		return KernelInterval{Name: "chain", Start: start, End: end, Selected: true}
	}
	o := func(start, end uint64) KernelInterval { return KernelInterval{Name: "gemm", Start: start, End: end} }
	tests := []struct {
		name    string
		kernels []KernelInterval
		want    DeviceSpan
	}{
		{"empty", nil, DeviceSpan{}},
		{
			"gaps are idle",
			[]KernelInterval{o(100, 200), s(300, 350), o(360, 400)},
			DeviceSpan{Launches: 3, SelectedLaunches: 1, OtherLaunches: 2, SpanNs: 300, SelectedNs: 50, OtherNs: 140, IdleNs: 110},
		},
		{
			"overlap",
			[]KernelInterval{s(100, 300), o(200, 400)},
			DeviceSpan{Launches: 2, SelectedLaunches: 1, OtherLaunches: 1, SpanNs: 300, SelectedNs: 100, OtherNs: 100, OverlapNs: 100},
		},
		{
			"nested",
			[]KernelInterval{o(0, 100), s(20, 30), s(40, 60)},
			DeviceSpan{Launches: 3, SelectedLaunches: 2, OtherLaunches: 1, SpanNs: 100, OtherNs: 70, OverlapNs: 30},
		},
		{
			"same class overlaps itself",
			[]KernelInterval{s(0, 100), s(50, 150), s(50, 60), o(200, 210), o(205, 215)},
			DeviceSpan{Launches: 5, SelectedLaunches: 3, OtherLaunches: 2, SpanNs: 215, SelectedNs: 150, OtherNs: 15, IdleNs: 50},
		},
		{
			"unordered",
			[]KernelInterval{o(250, 400), s(200, 250), o(100, 200)},
			DeviceSpan{Launches: 3, SelectedLaunches: 1, OtherLaunches: 2, SpanNs: 300, SelectedNs: 50, OtherNs: 250},
		},
		{
			"adjacent",
			[]KernelInterval{s(10, 20), o(20, 30), s(30, 40)},
			DeviceSpan{Launches: 3, SelectedLaunches: 2, OtherLaunches: 1, SpanNs: 30, SelectedNs: 20, OtherNs: 10},
		},
		{
			"zero width",
			[]KernelInterval{s(10, 10)},
			DeviceSpan{Launches: 1, SelectedLaunches: 1},
		},
		{
			"zero width in a gap and inside a kernel",
			[]KernelInterval{o(0, 10), s(15, 15), o(20, 30), s(25, 25)},
			DeviceSpan{Launches: 4, SelectedLaunches: 2, OtherLaunches: 2, SpanNs: 30, OtherNs: 20, IdleNs: 10},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := spanOracle(tt.kernels); got != tt.want {
				t.Fatalf("oracle = %+v, want %+v", got, tt.want)
			}
			for _, ks := range [][]KernelInterval{tt.kernels, translate(tt.kernels)} {
				in := slices.Clone(ks)
				got, err := AccountDeviceSpan(ks)
				if err != nil {
					t.Fatal(err)
				}
				if got != tt.want {
					t.Fatalf("AccountDeviceSpan(%v) =\n %+v\nwant\n %+v", ks, got, tt.want)
				}
				if !slices.Equal(ks, in) {
					t.Fatalf("AccountDeviceSpan modified its input: %v, was %v", ks, in)
				}
			}
		})
	}

	r := rand.New(rand.NewPCG(43, 1))
	for iter := range 3000 {
		ks := make([]KernelInterval, 1+r.IntN(8))
		for i := range ks {
			start := uint64(r.IntN(200))
			ks[i] = KernelInterval{Start: start, End: start + uint64(r.IntN(60)), Selected: r.IntN(2) == 0}
		}
		if iter%2 == 1 {
			ks = translate(ks)
		}
		got, err := AccountDeviceSpan(ks)
		if err != nil {
			t.Fatal(err)
		}
		if want := spanOracle(ks); got != want {
			t.Fatalf("timeline %d %v:\nAccountDeviceSpan = %+v\noracle            = %+v", iter, ks, got, want)
		}
	}
}

func TestDeviceSpanActivityRefusals(t *testing.T) {
	chain := func(name string) bool { return strings.HasPrefix(name, "chain_") }
	recs := []cuptiactivity.Record{
		{Kind: cupti.ActivityKindRuntime, Name: "cudaLaunchKernel", Start: 1, End: 5, Complete: true},
		{Kind: cupti.ActivityKindConcurrentKernel, Name: "chain_rmsnorm", Start: 10, End: 20, Complete: true},
		{Kind: cupti.ActivityKindMemcpy, Start: 20, End: 25, Complete: true},
		{Kind: cupti.ActivityKindKernel, Name: "sgemm_128x64", Start: 20, End: 90, Complete: true},
		{Kind: cupti.ActivityKindConcurrentKernel, Name: "chain_softmax", Start: 95, End: 95, Complete: true},
		// Other kinds are ignored whatever their timestamps.
		{Kind: cupti.ActivityKindExternalCorrelation},
		{Kind: cupti.ActivityKindDriver, Name: "cuLaunchKernel", Start: 9, End: 3},
	}
	in := slices.Clone(recs)
	ks, err := KernelIntervalsFromActivity(recs, chain)
	if err != nil {
		t.Fatal(err)
	}
	want := []KernelInterval{
		{Name: "chain_rmsnorm", Start: 10, End: 20, Selected: true},
		{Name: "sgemm_128x64", Start: 20, End: 90},
		{Name: "chain_softmax", Start: 95, End: 95, Selected: true},
	}
	if !slices.Equal(ks, want) {
		t.Fatalf("KernelIntervalsFromActivity = %v, want %v", ks, want)
	}
	if !slices.Equal(recs, in) {
		t.Fatal("KernelIntervalsFromActivity modified its records")
	}
	a, err := AccountDeviceSpan(ks)
	if err != nil {
		t.Fatal(err)
	}
	if a != (DeviceSpan{Launches: 3, SelectedLaunches: 2, OtherLaunches: 1, SpanNs: 85, SelectedNs: 10, OtherNs: 70, IdleNs: 5}) {
		t.Fatalf("AccountDeviceSpan = %+v", a)
	}

	for _, rs := range [][]cuptiactivity.Record{nil, recs[:1]} {
		ks, err := KernelIntervalsFromActivity(rs, chain)
		if err != nil || len(ks) != 0 {
			t.Fatalf("KernelIntervalsFromActivity(%v) = %v, %v; want no kernels", rs, ks, err)
		}
		if a, err := AccountDeviceSpan(ks); err != nil || a != (DeviceSpan{}) {
			t.Fatalf("AccountDeviceSpan(%v) = %+v, %v; want zero", ks, a, err)
		}
	}

	bad := []struct {
		name string
		rec  cuptiactivity.Record
	}{
		{"incomplete concurrent kernel", cuptiactivity.Record{Kind: cupti.ActivityKindConcurrentKernel, Name: "chain_silu_mul", Start: 100, End: 110}},
		{"incomplete kernel", cuptiactivity.Record{Kind: cupti.ActivityKindKernel, Name: "sgemm_64x64"}},
		{"reversed kernel", cuptiactivity.Record{Kind: cupti.ActivityKindConcurrentKernel, Name: "chain_embed", Start: 110, End: 100, Complete: true}},
	}
	for _, tt := range bad {
		t.Run(tt.name, func(t *testing.T) {
			ks, err := KernelIntervalsFromActivity(append(slices.Clone(recs), tt.rec), chain)
			if err == nil {
				t.Fatalf("KernelIntervalsFromActivity = %v, want an error", ks)
			}
		})
	}
	if ks, err := KernelIntervalsFromActivity(recs, nil); err == nil {
		t.Fatalf("KernelIntervalsFromActivity with a nil classifier = %v, want an error", ks)
	}
	if a, err := AccountDeviceSpan([]KernelInterval{{Start: 10, End: 20}, {Name: "chain_embed", Start: 10, End: 5}}); err == nil {
		t.Fatalf("AccountDeviceSpan of a reversed kernel = %+v, want an error", a)
	}
}

func TestDeviceSpanFractions(t *testing.T) {
	a := DeviceSpan{Launches: 4, SelectedLaunches: 1, OtherLaunches: 3, SpanNs: 1000, SelectedNs: 100, OtherNs: 500, OverlapNs: 150, IdleNs: 250}
	f, err := a.Fractions()
	if err != nil {
		t.Fatal(err)
	}
	want := SpanFractions{SelectedLaunches: 0.25, OtherLaunches: 0.75, SelectedSpan: 0.1, OtherSpan: 0.5, OverlapSpan: 0.15, IdleSpan: 0.25}
	if f != want {
		t.Fatalf("Fractions = %+v, want %+v", f, want)
	}

	tests := []struct {
		name string
		a    DeviceSpan
		want string
	}{
		{"no launches", DeviceSpan{}, "no kernel launches"},
		{"negative launches", DeviceSpan{Launches: -1, SpanNs: 10, IdleNs: 10}, "no kernel launches"},
		{"no launches but a span", DeviceSpan{SpanNs: 50_000_000, IdleNs: 50_000_000}, "no kernel launches"},
		{"launches spanning nothing", DeviceSpan{Launches: 1, SelectedLaunches: 1}, "span 0 ns"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f, err := tt.a.Fractions()
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Fractions = %+v, %v; want error containing %q", f, err, tt.want)
			}
		})
	}
}
