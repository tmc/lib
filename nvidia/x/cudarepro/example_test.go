package cudarepro

import (
	"context"
	"fmt"
	"strings"

	"github.com/tmc/lib/nvidia/x/cudaschedule"
)

func ExampleMeasure() {
	// A host fixture supplies classified records; native callers use the
	// existing campaign runner and retain its actual execution receipts.
	m, err := Measure(context.Background(), Identity{Policy: "exact-uint64"}, 3,
		func(context.Context) (cudaschedule.RunRecord, error) {
			return mismatchRecord(0), nil
		})
	fmt.Println(m.Reproductions, len(m.Runs), err)
	// Output: 3 3 <nil>
}

func ExampleAcceptRate() {
	i := Identity{Policy: "exact-uint64"}
	m := Measurement{Reproductions: 2, Runs: []cudaschedule.RunRecord{mismatchRecord(0), mismatchRecord(0)}}
	accepted, err := AcceptRate(i, m, m, false)
	fmt.Println(accepted, err)
	// Output: true <nil>
}

func ExampleMeasurement_OneOff() {
	m := Measurement{Reproductions: 1, Runs: []cudaschedule.RunRecord{mismatchRecord(0)}}
	fmt.Println(m.OneOff())
	// Output: true
}

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
