package ptxasinfo_test

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/ptxasinfo"
)

func Example() {
	out := `ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function '_Z5spillPfPKf' for 'sm_90a'
ptxas info    : Function properties for _Z5spillPfPKf
    992 bytes stack frame, 2232 bytes spill stores, 2660 bytes spill loads
ptxas info    : Used 32 registers, used 0 barriers, 992 bytes cumulative stack size
ptxas info    : Compile time = 128.853 ms
`
	r, err := ptxasinfo.Parse(strings.NewReader(out))
	if err != nil {
		panic(err)
	}
	for _, f := range r.Functions() {
		fmt.Printf("%s %s: %d registers, spills %v (%d B stored)\n", f.Arch, f.Name, f.Registers, f.Spills(), f.SpillStores)
	}
	// Output:
	// sm_90a _Z5spillPfPKf: 32 registers, spills true (2232 B stored)
}
