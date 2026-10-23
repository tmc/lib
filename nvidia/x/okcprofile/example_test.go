package okcprofile_test

import (
	"fmt"
	"log"

	"github.com/tmc/lib/nvidia/x/okcprofile"
)

func ExampleParseReport() {
	r, err := okcprofile.ParseReport(`kernel                          GPU time  share  calls
void flash::fwd_kernel<128, t...    8.1 µs  52.7%      1
rmsnorm_kernel                      1.2 µs   7.5%      2
... 3 more
total                              15.3 µs
`)
	if err != nil {
		log.Fatal(err)
	}
	for _, row := range r.Rows {
		fmt.Printf("%-28s %6v %d truncated=%v\n", row.Kernel, row.Time, row.Calls, row.Truncated)
	}
	n, exact := r.Launches()
	fmt.Printf("total %v, %d launches visible, exact=%v\n", r.Total, n, exact)
	// Output:
	// void flash::fwd_kernel<128, t  8.1µs 1 truncated=true
	// rmsnorm_kernel                1.2µs 2 truncated=false
	// total 15.3µs, 3 launches visible, exact=false
}

// ExampleElasticities ranks kernels by how much a cut in their time would
// move the score. Each load counts equally in a geometric mean, so B, which
// is 75% of the short decode load, counts nearly as much as A, which is 90%
// of a prefill load fifteen times longer.
func ExampleElasticities() {
	f := &okcprofile.File{
		Benchmarks: []okcprofile.Benchmark{
			{Case: okcprofile.Case{Load: "decode", Weight: 1}, BaselineNS: 30000, MedianNS: 20000},
			{Case: okcprofile.Case{Load: "prefill", Weight: 1}, BaselineNS: 400000, MedianNS: 300000},
		},
		Profiles: []okcprofile.Profile{
			{Case: okcprofile.Case{Load: "decode", Weight: 1}, Report: "B 15.0 µs 75.0% 1\nA 5.0 µs 25.0% 1\ntotal 20.0 µs\n"},
			{Case: okcprofile.Case{Load: "prefill", Weight: 1}, Report: "A 270.0 µs 90.0% 1\nB 30.0 µs 10.0% 1\ntotal 300.0 µs\n"},
		},
	}
	t, err := okcprofile.Elasticities(f)
	if err != nil {
		log.Fatal(err)
	}
	for _, k := range t.Kernels {
		fmt.Printf("%s %.3f\n", k.Kernel, k.E)
	}
	fmt.Printf("score %.3f, coverage %.2f\n", t.Score, t.Coverage)
	// Output:
	// A 0.575
	// B 0.425
	// score 1.414, coverage 1.00
}
