package cudakv_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudakv"
)

func ExampleBlockAllocator_AllocationGeneration() {
	pool := cudakv.NewBlockAllocator(2, 1)
	_ = pool.AllocSequence(7, nil, 2)
	gen1, ok1 := pool.AllocationGeneration(7)
	pool.FreeSequence(7)
	_, okFree := pool.AllocationGeneration(7)
	_ = pool.AllocSequence(7, nil, 2)
	gen2, ok2 := pool.AllocationGeneration(7)
	fmt.Println(ok1, okFree, ok2, gen2 > gen1)
	// Output: true false true true
}

func ExampleBlockAllocator_ValidateSequence() {
	pool := cudakv.NewBlockAllocator(2, 2)
	_ = pool.AllocSequence(7, nil, 3)
	gen, _ := pool.AllocationGeneration(7)
	table := pool.Table(7)
	errValid := pool.ValidateSequence(7, gen, table, 3)
	errCap := pool.ValidateSequence(7, gen, table, 4)
	pool.FreeSequence(7)
	_ = pool.AllocSequence(7, nil, 3)
	errStale := pool.ValidateSequence(7, gen, pool.Table(7), 3)
	fmt.Println(errValid)
	fmt.Println(errCap)
	fmt.Println(errStale)
	// Output:
	// <nil>
	// cudakv: sequence table or capacity mismatch
	// cudakv: sequence allocation changed
}
