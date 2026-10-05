package cudarepro

import (
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaschedule"
)

func ExampleIdentify() {
	digest := strings.Repeat("a", 64)
	run := cudaschedule.RunRecord{
		Exited: true, Stdout: "MISMATCH\n",
		PTXDigest: digest, ExpectedPTXDigest: digest, PTXDigestSource: "runner_receipt",
		Mismatch: &cudaschedule.MismatchIdentity{Index: 0, Policy: "exact-uint64"},
	}
	identity, err := Identify(run)
	fmt.Println(identity, err)
	// Output: {0 exact-uint64} <nil>
}

func ExampleIdentity_Matches() {
	identity := Identity{Index: 0, Policy: "exact-uint64"}
	matched, err := identity.Matches(cudaschedule.RunRecord{})
	fmt.Println(matched, err)
	// Output: false compare failure: invalid execution evidence
}
