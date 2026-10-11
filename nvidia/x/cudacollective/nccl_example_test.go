package cudacollective_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudacollective"
)

func ExampleNCCLTransport() {
	membership, _ := cudacollective.NewMembership([]cudacollective.Rank{{ID: 0}})
	// Supply borrowed NCCL handles and CUDA streams before native execution.
	var transport cudacollective.NCCLTransport
	fmt.Println(transport.Available(membership))
	fmt.Println(transport.Close())
	// Output:
	// false
	// <nil>
}
