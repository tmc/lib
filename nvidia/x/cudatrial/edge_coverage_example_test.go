package cudatrial_test

import (
	"fmt"
	"github.com/tmc/lib/nvidia/x/cudatrial"
)

func ExampleThreadEdgeMap() {
	m, _ := cudatrial.NewThreadEdgeMap(32, 2)
	a, _ := m.Record(0, 5)
	b, _ := m.Record(0, 7)
	c, _ := m.Record(1, 7)
	fmt.Println(a, b, c)
	fmt.Println(m.Snapshot().Quantized()[a])
	// Output:
	// 21 21 23
	// 2
}

func ExamplePersistentCoverage() {
	p, _ := cudatrial.NewPersistentCoverage(32, 1)
	p.Notify(cudatrial.SignalStart)
	p.Record(0, 5)
	p.Notify(cudatrial.SignalEnd)
	p.Notify(cudatrial.SignalStart)
	p.Record(0, 9)
	p.Fault()
	fmt.Println(p.Merged().Bytes()[21], p.Merged().Bytes()[25])
	fmt.Println(p.LastComplete().Bytes()[21])
	// Output:
	// 1 0
	// 1
}
