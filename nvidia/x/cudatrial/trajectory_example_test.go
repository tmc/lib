package cudatrial_test

import (
	"errors"
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudatrial"
)

func ExampleBottleneckFamily_Valid() {
	fmt.Println(cudatrial.MemoryBandwidthLimited.Valid())
	// Output: true
}

func ExampleObserveTrajectory() {
	r, err := cudatrial.ObserveTrajectory(nil, cudatrial.TrajectoryReceipt{}, cudatrial.Receipt{}, cudatrial.Receipt{})
	fmt.Println(r.PendingHardware, errors.Is(err, cudatrial.ErrUnqualifiedOutcome))
	// Output: true true
}

func ExampleTrajectoryReceipt_Validate() {
	r := cudatrial.TrajectoryReceipt{PendingHardware: true}
	fmt.Println(errors.Is(r.Validate(), cudatrial.ErrUnqualifiedOutcome))
	// Output: true
}

func ExampleTrajectoryReceipt_HardwareObserved() {
	fmt.Println((cudatrial.TrajectoryReceipt{}).HardwareObserved())
	// Output: false
}

func ExampleTrajectoryMemory_ImportTrusted() {
	m, _ := cudatrial.NewTrajectoryMemory(32)
	// Explicit trust does not make an incomplete declaration valid.
	err := m.ImportTrusted(cudatrial.TrajectoryReceipt{PendingHardware: true})
	fmt.Println(errors.Is(err, cudatrial.ErrUnqualifiedOutcome))
	// Output: true
}

func ExampleNewTrajectoryMemory() {
	m, err := cudatrial.NewTrajectoryMemory(32)
	fmt.Println(m != nil, err)
	// Output: true <nil>
}

func ExampleTrajectoryMemory_Record() {
	m, _ := cudatrial.NewTrajectoryMemory(32)
	err := m.Record(cudatrial.TrajectoryReceipt{PendingHardware: true})
	fmt.Println(errors.Is(err, cudatrial.ErrUnqualifiedOutcome))
	// Output: true
}

func ExampleTrajectoryMemory_Rank() {
	m, _ := cudatrial.NewTrajectoryMemory(32)
	fmt.Println(len(m.Rank("compatibility-digest", cudatrial.MemoryBandwidthLimited, "sm_121")))
	// Output: 0
}

func ExampleTrajectoryMemory_Replay() {
	m, _ := cudatrial.NewTrajectoryMemory(32)
	snapshot := m.Replay("compatibility-digest", cudatrial.MemoryBandwidthLimited, "sm_121")
	fmt.Println(len(snapshot.ReceiptIDs), len(snapshot.Rewards))
	// Output: 0 0
}
