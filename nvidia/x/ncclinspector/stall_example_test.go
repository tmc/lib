package ncclinspector_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/ncclinspector"
)

func ExampleValidateCommunicatorHash() {
	valid := "0x91541b8a24505d53"
	truncated := "0x91541b8a24505d"

	fmt.Println("valid:", ncclinspector.ValidateCommunicatorHash(valid) == nil)
	fmt.Println("truncated:", ncclinspector.ValidateCommunicatorHash(truncated) != nil)

	// Output:
	// valid: true
	// truncated: true
}

func ExampleTwoRankAttributor() {
	attr := ncclinspector.NewTwoRankAttributor()
	commHash := "0x91541b8a24505d53"

	// Rank 0 enqueues on time
	_ = attr.AddObservation(ncclinspector.CollectiveObservation{
		CommunicatorHash: commHash,
		Rank:             0,
		TotalRanks:       2,
		Sequence:         1,
		Operation:        "AllReduce",
		Bytes:            65536,
		HostPID:          1001,
		Hostname:         "fixture-node",
		HostEnqueueStart: ncclinspector.BoundedTimestamp{Nanoseconds: 1_000_000, UncertaintyNS: 50, Domain: ncclinspector.ClockDomainHostMonotonic},
		HostEnqueueEnd:   ncclinspector.BoundedTimestamp{Nanoseconds: 1_010_000, UncertaintyNS: 50, Domain: ncclinspector.ClockDomainHostMonotonic},
		HasHostEnqueue:   true,
	})

	// Rank 1 enqueues late (10ms later)
	_ = attr.AddObservation(ncclinspector.CollectiveObservation{
		CommunicatorHash: commHash,
		Rank:             1,
		TotalRanks:       2,
		Sequence:         1,
		Operation:        "AllReduce",
		Bytes:            65536,
		HostPID:          1002,
		Hostname:         "fixture-node",
		HostEnqueueStart: ncclinspector.BoundedTimestamp{Nanoseconds: 11_000_000, UncertaintyNS: 50, Domain: ncclinspector.ClockDomainHostMonotonic},
		HostEnqueueEnd:   ncclinspector.BoundedTimestamp{Nanoseconds: 11_010_000, UncertaintyNS: 50, Domain: ncclinspector.ClockDomainHostMonotonic},
		HasHostEnqueue:   true,
	})

	report, err := attr.Attribute(commHash, 1)
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	fmt.Printf("diagnosis: %s\n", report.Diagnosis)
	fmt.Printf("delayed rank: %d\n", report.DelayedRank)
	fmt.Printf("admitted: %t\n", report.Admitted)

	// Output:
	// diagnosis: host_enqueue_skew
	// delayed rank: 1
	// admitted: true
}
