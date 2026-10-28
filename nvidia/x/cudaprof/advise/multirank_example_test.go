package advise_test

import (
	"fmt"
	"log"
	"strconv"

	"github.com/google/pprof/profile"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudaprof/advise"
)

// kernels builds a capture of timed kernels on one stream, each given as
// name, start and end in nanoseconds.
func kernels(device string, ks ...any) *cudacapture.Capture {
	p := &profile.Profile{SampleType: []*profile.ValueType{{Type: "kernel_time", Unit: "nanoseconds"}}}
	for i := 0; i < len(ks); i += 3 {
		name, start, end := ks[i].(string), ks[i+1].(int), ks[i+2].(int)
		p.Sample = append(p.Sample, &profile.Sample{
			Value:    []int64{int64(end - start)},
			Label:    map[string][]string{"name": {name}, "subtype": {"kernel"}, "stream": {"7"}},
			NumLabel: map[string][]int64{"start": {int64(start)}, "end": {int64(end)}, "correlation": {int64(i/3 + 1)}},
		})
	}
	return &cudacapture.Capture{Metadata: []byte(`{"device":"` + device + `"}`), Profile: p}
}

func ExampleAnalyzeRanks() {
	// Two ranks on one host. Rank 1's GEMM runs 40 us longer, so rank 0
	// waits for it inside every all-reduce.
	var ranks []advise.Rank
	for r, extra := range []int{0, 40_000} {
		var ks []any
		for step := 0; step < 3; step++ {
			t := 1_000_000 * (step + 1)
			ks = append(ks, "gemm", t, t+200_000+extra, "ncclDevKernel_AllReduce_Sum_bf16_RING_LL", t+200_000+extra, t+300_000)
		}
		ranks = append(ranks, advise.Rank{Name: "rank" + strconv.Itoa(r), Capture: kernels("NVIDIA GB200", ks...)})
	}
	report, err := advise.AnalyzeRanks(ranks, advise.AlignClock)
	if err != nil {
		log.Fatal(err)
	}
	tl := report.Ranks
	fmt.Printf("straggler %s, skew %d us\n", tl.Straggler, tl.MeanSkewNanos/1000)
	for _, r := range tl.Ranks {
		fmt.Printf("%s waited %d us\n", r.Name, r.WaitNanos/1000)
	}
	// Output:
	// straggler rank1, skew 40 us
	// rank0 waited 120 us
	// rank1 waited 0 us
}

func ExampleReport_ProjectTo() {
	// A GB10 decode step: 100 kernels of 20 us, 5 us apart.
	var ks []any
	for i := 0; i < 100; i++ {
		t := 1_000_000 + i*25_000
		ks = append(ks, "matvec", t, t+20_000)
	}
	cap := kernels("NVIDIA GB10", ks...)
	report, err := advise.Analyze(cap)
	if err != nil {
		log.Fatal(err)
	}
	if err := report.ProjectTo(cap, "gb300"); err != nil {
		log.Fatal(err)
	}
	p := report.Target
	fmt.Printf("%s: floor %d us, gaps %d us, remedy %s\n", p.Product, p.FloorNanos/1000, p.IdleNanos/1000, p.Remedy)
	// Output:
	// GB300: floor 68 us, gaps 495 us, remedy persistent
}
