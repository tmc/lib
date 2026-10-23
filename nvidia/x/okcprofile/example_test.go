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
