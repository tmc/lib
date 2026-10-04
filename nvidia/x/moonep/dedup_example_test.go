//go:build !windows

package moonep_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/moonep"
)

func exampleDedupSpec() moonep.DedupSpec {
	return moonep.DedupSpec{Tokens: 1, Width: 17, K: 2, Slots: 2, Groups: [][3]int32{{0, 0, 1}}, DupLoffs: []int32{1}, Counts: [2]int32{1, 1}, RouteSlots: []int32{0, 1}, ExpertIDs: []int32{0, 1}, SourceID: "example-routes", RankMap: []int32{0}, BufferGeneration: 1}
}

func ExampleDedupSources() {
	stages, err := moonep.DedupSources(exampleDedupSpec())
	fmt.Println(len(stages), stages[0].Name, err)
	// Output: 4 dedup_expand <nil>
}
func ExampleCompileDedup() {
	s := exampleDedupSpec()
	s.Width = 0
	_, err := moonep.CompileDedup(s, 12, 1)
	fmt.Println(err != nil)
	// Output: true
}
func ExampleNewDedup() {
	_, err := moonep.NewDedup(nil, nil, nil)
	fmt.Println(err != nil)
	// Output: true
}
func ExampleDedup_Execute() {
	var d *moonep.Dedup
	_, err := d.Execute(nil, nil, cudamem.Slice[float32]{}, cudamem.Slice[float32]{}, 1)
	fmt.Println(err != nil)
	// Output: true
}

func ExampleDedupInvocation_Close() {
	var v moonep.DedupInvocation
	fmt.Println(v.Close())
	// Output: <nil>
}
func ExampleDedupInvocation_Wait() {
	var v moonep.DedupInvocation
	fmt.Println(v.Wait(nil))
	// Output: <nil>
}
func ExampleDedupInvocation_Readback() {
	var v moonep.DedupInvocation
	_, err := v.Readback(nil)
	fmt.Println(err != nil)
	// Output: true
}
func ExampleDedupInvocation_Receipts() {
	var v moonep.DedupInvocation
	fmt.Println(len(v.Receipts()))
	// Output: 0
}
