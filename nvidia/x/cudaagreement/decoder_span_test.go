package cudaagreement_test

import (
	"fmt"
	"slices"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	proftimeline "github.com/tmc/lib/nvidia/x/cudaprof/timeline"
	"github.com/tmc/lib/nvidia/x/cuptiactivity"
)

// decoderSpan accounts one region's activity records, with the chain kernels
// selected, through the profiler's device-span route. The records must come
// from one session on one device; the caller checks the collection lost
// nothing.
func decoderSpan(records []cuptiactivity.Record) ([]proftimeline.KernelInterval, proftimeline.DeviceSpan, error) {
	ks, err := proftimeline.KernelIntervalsFromActivity(records, isChainKernel)
	if err != nil {
		return nil, proftimeline.DeviceSpan{}, err
	}
	a, err := proftimeline.AccountDeviceSpan(ks)
	if err != nil {
		return nil, proftimeline.DeviceSpan{}, err
	}
	return ks, a, nil
}

// TestDecoderDeviceSpanRoute runs supplied decoder-shaped records through
// decoderSpan, the route the native span tests use, and checks every part
// and denominator of the result.
func TestDecoderDeviceSpanRoute(t *testing.T) {
	kernel := func(kind cupti.ActivityKind, name string, start, end uint64) cuptiactivity.Record {
		return cuptiactivity.Record{Kind: kind, Name: name, Start: start, End: end, Complete: true}
	}
	recs := []cuptiactivity.Record{
		kernel(cupti.ActivityKindRuntime, "cudaLaunchKernel", 0, 10_000),
		kernel(cupti.ActivityKindConcurrentKernel, "chain_embed", 1000, 1100),
		kernel(cupti.ActivityKindKernel, "ampere_sgemm_128x64_nn", 1100, 1600),
		kernel(cupti.ActivityKindConcurrentKernel, "chain_rmsnorm", 1500, 1700),
		kernel(cupti.ActivityKindMemcpy, "", 1700, 2000),
		kernel(cupti.ActivityKindConcurrentKernel, "chain_softmax", 2000, 2000),
		kernel(cupti.ActivityKindConcurrentKernel, "ampere_sgemm_64x64_tn", 2100, 2400),
	}
	in := slices.Clone(recs)
	ks, a, err := decoderSpan(recs)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(recs, in) {
		t.Fatal("decoderSpan modified its records")
	}
	wantKernels := []proftimeline.KernelInterval{
		{Name: "chain_embed", Start: 1000, End: 1100, Selected: true},
		{Name: "ampere_sgemm_128x64_nn", Start: 1100, End: 1600},
		{Name: "chain_rmsnorm", Start: 1500, End: 1700, Selected: true},
		{Name: "chain_softmax", Start: 2000, End: 2000, Selected: true},
		{Name: "ampere_sgemm_64x64_tn", Start: 2100, End: 2400},
	}
	if !slices.Equal(ks, wantKernels) {
		t.Fatalf("decoderSpan kernels =\n %v\nwant\n %v", ks, wantKernels)
	}
	// Per nanosecond: [1000,1100) chain, [1100,1500) other, [1500,1600)
	// both, [1600,1700) chain, [1700,2100) idle (a memcpy and a zero-length
	// chain kernel), [2100,2400) other.
	want := proftimeline.DeviceSpan{
		Launches: 5, SelectedLaunches: 3, OtherLaunches: 2,
		SpanNs: 1400, SelectedNs: 200, OtherNs: 700, OverlapNs: 100, IdleNs: 400,
	}
	if a != want {
		t.Fatalf("decoderSpan account = %+v, want %+v", a, want)
	}
	f, err := a.Fractions()
	if err != nil {
		t.Fatal(err)
	}
	wantFractions := proftimeline.SpanFractions{
		SelectedLaunches: 3.0 / 5, OtherLaunches: 2.0 / 5,
		SelectedSpan: 200.0 / 1400, OtherSpan: 700.0 / 1400, OverlapSpan: 100.0 / 1400, IdleSpan: 400.0 / 1400,
	}
	if f != wantFractions {
		t.Fatalf("Fractions = %+v, want %+v", f, wantFractions)
	}

	// An empty region accounts as zero and has no fraction.
	ks, a, err = decoderSpan(recs[:1])
	if err != nil || len(ks) != 0 || a != (proftimeline.DeviceSpan{}) {
		t.Fatalf("decoderSpan of API records only = %v, %+v, %v; want no kernels and a zero account", ks, a, err)
	}
	if f, err := a.Fractions(); err == nil {
		t.Fatalf("Fractions of an empty region = %+v, want an error", f)
	}

	// An incomplete chain kernel fails the route rather than being left out.
	bad := append(slices.Clone(recs), cuptiactivity.Record{Kind: cupti.ActivityKindConcurrentKernel, Name: "chain_silu_mul", Start: 2500})
	if ks, a, err := decoderSpan(bad); err == nil {
		t.Fatalf("decoderSpan accepted an incomplete kernel: %v, %+v", ks, a)
	}
}

// spanString formats a as [cudaagreement.SpanAccount.String] does, with the
// selected kernels labeled chain.
func spanString(a proftimeline.DeviceSpan) string {
	return fmt.Sprintf("launches chain=%d other=%d total=%d; span_ns chain=%d other=%d overlap=%d idle=%d total=%d",
		a.SelectedLaunches, a.OtherLaunches, a.Launches, a.SelectedNs, a.OtherNs, a.OverlapNs, a.IdleNs, a.SpanNs)
}
