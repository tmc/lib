package nccltest_test

import (
	"context"
	"errors"
	"fmt"
	"github.com/tmc/lib/nvidia/x/nccltest"
	"time"
)

func Example() {
	c := nccltest.Case{Operation: nccltest.AllGather, Bytes: 1024}
	layout, _ := c.Layout(4)
	fmt.Println(layout.SendCount, layout.RecvCount)
	// Output: 64 256
}
func ExampleSizes() {
	sizes, _ := nccltest.Sizes(16, 128, 2)
	fmt.Println(sizes)
	// Output: [16 32 64 128]
}
func ExampleCase_Layout() {
	l, _ := (nccltest.Case{Operation: nccltest.ReduceScatter, Bytes: 1024}).Layout(4)
	fmt.Println(l.SendCount, l.RecvCount, l.RecvOffset)
	// Output: 256 64 64
}
func ExampleCase_Bandwidth() {
	c := nccltest.Case{Operation: nccltest.AllReduce, Bytes: 1000000}
	algorithm, bus, _ := c.Bandwidth(4, time.Millisecond)
	fmt.Println(algorithm, bus)
	// Output: 1 1.5
}
func ExampleCheck() {
	c := nccltest.Case{Operation: nccltest.AllReduce, Bytes: 16}
	count, wrong, _ := nccltest.Check(c, 1, 0, []float32{7, 2, 3, 13})
	fmt.Println(count, wrong)
	// Output: 4 0
}

func ExamplePreMulSum() {
	c := nccltest.Case{Operation: nccltest.AllReduce, Bytes: 4, Reduction: nccltest.PreMulSum}
	// The three rank inputs are 7, 12, 13; upstream's scalars are 1, 2, 1.
	compared, wrong, _ := nccltest.Check(c, 3, 0, []float32{44})
	fmt.Println(compared, wrong)
	// Output: 1 0
}

func ExampleLinearSizes() {
	sizes, err := nccltest.LinearSizes(16, 64, 16)
	fmt.Println(sizes, err)
	// Output: [16 32 48 64] <nil>
}

func ExampleRunCycles() {
	// Invalid counts are rejected before loading CUDA.
	err := nccltest.RunCycles(nccltest.Case{}, nccltest.Options{}, 0,
		func(r nccltest.Result) error { fmt.Println(r.Comparisons); return nil })
	fmt.Println(err)
	// Output: nccltest: cycle count must be positive
}

func ExampleRunSweep() {
	// Empty sweeps are rejected before loading CUDA.
	err := nccltest.RunSweep(nccltest.Case{}, nccltest.Options{Iterations: 1}, nil, 1,
		func(r nccltest.Result) error { fmt.Println(r.Case.Bytes); return nil })
	fmt.Println(err)
	// Output: nccltest: empty size sweep
}

func ExampleRunSweepContext() {
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // A canceled request returns before loading CUDA.
	err := nccltest.RunSweepContext(ctx, nccltest.Case{}, nccltest.Options{}, []int{1024}, 0, func(nccltest.Result) error { return nil })
	fmt.Println(errors.Is(err, context.Canceled))
	// Output: true
}

func ExampleCase_Layout_spread() {
	spread := 0.0
	c := nccltest.Case{Operation: nccltest.AllToAllV, Bytes: 256, Spread: &spread}
	layout, _ := c.Layout(4)
	fmt.Println(layout.SendCount, layout.RecvCount)
	// Output: 64 64
}
