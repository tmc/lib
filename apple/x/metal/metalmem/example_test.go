package metalmem_test

import (
	"fmt"

	"github.com/tmc/lib/apple/x/metal"
	"github.com/tmc/lib/apple/x/metal/metalmem"
)

func ExampleAlloc() {
	dev, err := metal.NewDevice()
	if err != nil {
		fmt.Println(err)
		return
	}
	defer dev.Close()

	values, err := metalmem.AllocFromSlice(dev, []uint32{2, 3, 5})
	if err != nil {
		fmt.Println(err)
		return
	}
	defer values.Free()

	fmt.Println(values.HostSlice())
	// Output: [2 3 5]
}

func ExamplePool() {
	dev, err := metal.NewDevice()
	if err != nil {
		fmt.Println(err)
		return
	}
	defer dev.Close()

	pool := metalmem.NewPool(dev)
	defer pool.Free()
	buf, err := pool.Get(1500)
	if err != nil {
		fmt.Println(err)
		return
	}
	if err := pool.Put(buf, 1500); err != nil {
		fmt.Println(err)
		return
	}
	reused, err := pool.Get(1500)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(buf.GetID() == reused.GetID())
	if err := pool.Put(reused, 1500); err != nil {
		fmt.Println(err)
	}
	// Output: true
}

// scratch holds the activation buffers one decode step writes through.
type scratch struct {
	X      metalmem.Slice[float32]
	Q      metalmem.Slice[float32]
	Logits metalmem.Slice[float32]

	arena *metalmem.Arena
}

// ExampleArena allocates a group of buffers that share a lifetime: one error
// check for the group, and one call to release it.
func ExampleArena() {
	dev, err := metal.NewDevice()
	if err != nil {
		fmt.Println(err)
		return
	}
	defer dev.Close()

	a := metalmem.NewArena(dev)
	s := &scratch{
		X:      metalmem.AllocIn[float32](a, "x", 4096),
		Q:      metalmem.AllocIn[float32](a, "q", 4096),
		Logits: metalmem.AllocIn[float32](a, "logits", 32000),
		arena:  a,
	}
	if err := a.Err(); err != nil {
		a.Free()
		fmt.Println(err)
		return
	}
	defer s.arena.Free()

	fmt.Println(s.Logits.Len())
}

// ExampleUploadIn allocates a buffer already holding host data.
func ExampleUploadIn() {
	dev, err := metal.NewDevice()
	if err != nil {
		fmt.Println(err)
		return
	}
	defer dev.Close()

	a := metalmem.NewArena(dev)
	defer a.Free()

	weights := metalmem.UploadIn(a, "weights", []float32{1, 2, 3, 4})
	if err := a.Err(); err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(weights.HostSlice())
}
