package cudaagreement_test

import (
	"math/rand/v2"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cudaagreement"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
)

func TestAccount(t *testing.T) {
	c := func(s, e uint64) cudaagreement.Kernel {
		return cudaagreement.Kernel{Name: "chain", Start: s, End: e, Chain: true}
	}
	o := func(s, e uint64) cudaagreement.Kernel { return cudaagreement.Kernel{Name: "gemm", Start: s, End: e} }
	tests := []struct {
		name    string
		kernels []cudaagreement.Kernel
		want    cudaagreement.SpanAccount
	}{
		{"none", nil, cudaagreement.SpanAccount{}},
		{
			"back to back",
			[]cudaagreement.Kernel{o(100, 200), c(200, 250), o(250, 400)},
			cudaagreement.SpanAccount{Launches: 3, ChainLaunches: 1, OtherLaunches: 2, SpanNs: 300, ChainNs: 50, OtherNs: 250},
		},
		{
			"gaps are idle",
			[]cudaagreement.Kernel{o(100, 200), c(300, 350), o(360, 400)},
			cudaagreement.SpanAccount{Launches: 3, ChainLaunches: 1, OtherLaunches: 2, SpanNs: 300, ChainNs: 50, OtherNs: 140, IdleNs: 110},
		},
		{
			// The reviewer's case: chain [100,300) and other [200,400). The
			// shared 100 ns is overlap, not counted in either class.
			"overlap",
			[]cudaagreement.Kernel{c(100, 300), o(200, 400)},
			cudaagreement.SpanAccount{Launches: 2, ChainLaunches: 1, OtherLaunches: 1, SpanNs: 300, ChainNs: 100, OtherNs: 100, OverlapNs: 100},
		},
		{
			"same class overlaps itself",
			[]cudaagreement.Kernel{c(0, 100), c(50, 150), c(50, 60)},
			cudaagreement.SpanAccount{Launches: 3, ChainLaunches: 3, SpanNs: 150, ChainNs: 150},
		},
		{
			"order does not matter",
			[]cudaagreement.Kernel{o(250, 400), c(200, 250), o(100, 200)},
			cudaagreement.SpanAccount{Launches: 3, ChainLaunches: 1, OtherLaunches: 2, SpanNs: 300, ChainNs: 50, OtherNs: 250},
		},
		{
			"zero-length kernel",
			[]cudaagreement.Kernel{c(10, 10)},
			cudaagreement.SpanAccount{Launches: 1, ChainLaunches: 1},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := cudaagreement.Account(tt.kernels)
			if err != nil {
				t.Fatal(err)
			}
			if got != tt.want {
				t.Fatalf("Account =\n %v\nwant\n %v", got, tt.want)
			}
		})
	}
	if _, err := cudaagreement.Account([]cudaagreement.Kernel{c(10, 5)}); err == nil {
		t.Fatal("Account accepted a kernel that ends before it starts")
	}
}

// TestAccountOracle checks Account against a per-nanosecond classification
// of random timelines, which shares nothing with the sweep but the
// definitions.
func TestAccountOracle(t *testing.T) {
	r := rand.New(rand.NewPCG(36, 1))
	for iter := range 3000 {
		n := 1 + r.IntN(8)
		ks := make([]cudaagreement.Kernel, n)
		for i := range ks {
			s := uint64(r.IntN(200))
			ks[i] = cudaagreement.Kernel{Start: s, End: s + uint64(r.IntN(60)), Chain: r.IntN(2) == 0}
		}
		got, err := cudaagreement.Account(ks)
		if err != nil {
			t.Fatal(err)
		}
		var want cudaagreement.SpanAccount
		first, last := ks[0].Start, ks[0].End
		for _, k := range ks {
			first, last = min(first, k.Start), max(last, k.End)
			want.Launches++
			if k.Chain {
				want.ChainLaunches++
			} else {
				want.OtherLaunches++
			}
		}
		want.SpanNs = last - first
		for ns := first; ns < last; ns++ {
			var chain, other bool
			for _, k := range ks {
				if k.Start <= ns && ns < k.End {
					if k.Chain {
						chain = true
					} else {
						other = true
					}
				}
			}
			switch {
			case chain && other:
				want.OverlapNs++
			case chain:
				want.ChainNs++
			case other:
				want.OtherNs++
			default:
				want.IdleNs++
			}
		}
		if got != want {
			t.Fatalf("timeline %d %v:\nAccount = %v\noracle  = %v", iter, ks, got, want)
		}
	}
}

// TestSpanFractionsSum checks that the parts of an account sum to its
// totals, exactly in integers and to rounding in the fractions.
func TestSpanFractionsSum(t *testing.T) {
	ks := []cudaagreement.Kernel{
		{Name: "gemm", Start: 1000, End: 5000},
		{Name: "rmsnorm", Start: 5100, End: 5400, Chain: true},
		{Name: "gemm", Start: 5350, End: 9000},
		{Name: "softmax", Start: 9500, End: 9800, Chain: true},
	}
	a, err := cudaagreement.Account(ks)
	if err != nil {
		t.Fatal(err)
	}
	if a.ChainNs+a.OtherNs+a.OverlapNs+a.IdleNs != a.SpanNs {
		t.Fatalf("span parts do not sum: %v", a)
	}
	if a.ChainLaunches+a.OtherLaunches != a.Launches {
		t.Fatalf("launch parts do not sum: %v", a)
	}
	f, err := a.Fractions()
	if err != nil {
		t.Fatal(err)
	}
	if s := f.ChainSpan + f.OtherSpan + f.OverlapSpan + f.IdleSpan; s < 1-1e-12 || s > 1+1e-12 {
		t.Fatalf("span fractions sum to %v: %+v", s, f)
	}
	if s := f.ChainLaunches + f.OtherLaunches; s != 1 {
		t.Fatalf("launch fractions sum to %v: %+v", s, f)
	}
	if f.ChainLaunches != 0.5 || f.ChainSpan != 550.0/8800 {
		t.Fatalf("fractions = %+v, want chain launches 0.5 and chain span 550/8800", f)
	}
}

// TestAccountEmptyRegion is the range-open control: a region with no kernels
// has no span, however long it was observed, and no fraction.
func TestAccountEmptyRegion(t *testing.T) {
	a, err := cudaagreement.Account(nil)
	if err != nil {
		t.Fatal(err)
	}
	if a != (cudaagreement.SpanAccount{}) {
		t.Fatalf("Account(nil) = %v, want zero", a)
	}
	// Non-kernel activity inside the region (the API calls that opened and
	// closed it, a memset) contributes nothing.
	recs := []cuptiactivity.Record{
		{Kind: cupti.ActivityKindRuntime, Name: "cudaStreamSynchronize", Start: 0, End: 50_000_000, Complete: true},
		{Kind: cupti.ActivityKindMemset, Start: 10, End: 20, Complete: true},
	}
	ks, err := cudaagreement.KernelsFromActivity(recs, func(string) bool { return true })
	if err != nil {
		t.Fatal(err)
	}
	if a, _ := cudaagreement.Account(ks); a.SpanNs != 0 || a.Launches != 0 {
		t.Fatalf("region of API records only accounts as %v, want zero", a)
	}
}

// TestFractionsRefuseEmptyDenominator checks both denominators before any
// division.
func TestFractionsRefuseEmptyDenominator(t *testing.T) {
	tests := []struct {
		name string
		a    cudaagreement.SpanAccount
		want string
	}{
		{"no launches", cudaagreement.SpanAccount{}, "no kernel launches"},
		{"no launches but a span", cudaagreement.SpanAccount{SpanNs: 50_000_000, IdleNs: 50_000_000}, "no kernel launches"},
		{"launches spanning nothing", cudaagreement.SpanAccount{Launches: 1, ChainLaunches: 1}, "span 0 ns"},
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

func TestKernelsFromActivity(t *testing.T) {
	chain := func(name string) bool { return strings.HasPrefix(name, "chain_") }
	recs := []cuptiactivity.Record{
		{Kind: cupti.ActivityKindConcurrentKernel, Name: "chain_rmsnorm", Start: 10, End: 20, Complete: true},
		{Kind: cupti.ActivityKindRuntime, Name: "cudaLaunchKernel", Start: 1, End: 5, Complete: true},
		{Kind: cupti.ActivityKindKernel, Name: "sgemm_128x64", Start: 20, End: 90, Complete: true},
	}
	ks, err := cudaagreement.KernelsFromActivity(recs, chain)
	if err != nil {
		t.Fatal(err)
	}
	want := []cudaagreement.Kernel{
		{Name: "chain_rmsnorm", Start: 10, End: 20, Chain: true},
		{Name: "sgemm_128x64", Start: 20, End: 90},
	}
	if len(ks) != len(want) || ks[0] != want[0] || ks[1] != want[1] {
		t.Fatalf("KernelsFromActivity = %v, want %v", ks, want)
	}
	bad := append(recs, cuptiactivity.Record{Kind: cupti.ActivityKindConcurrentKernel, Name: "chain_softmax"})
	if _, err := cudaagreement.KernelsFromActivity(bad, chain); err == nil {
		t.Fatal("KernelsFromActivity dropped an incomplete kernel record without an error")
	}
	if _, err := cudaagreement.KernelsFromActivity(recs, nil); err == nil {
		t.Fatal("KernelsFromActivity accepted a nil classifier")
	}
}
