package cudaattn_test

import (
	"context"
	"fmt"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaattn"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"os"
)

func ExampleKDAConfig_Extents() {
	c := cudaattn.KDAConfig{Dimension: 128, Heads: 1, Sequences: 1, Tokens: 1, Scale: 1}
	e, err := c.Extents()
	fmt.Println(e.Vectors, e.State, err)
	// Output: 128 16384 <nil>
}

// This device consumer requires an inherited GPU lock. It deliberately has no
// Output directive, so ordinary host example tests never dispatch hardware.
func ExampleKDA_Execute() {
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		panic(err)
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		panic(err)
	}
	defer dev.Close()
	stream, err := dev.NewStream()
	if err != nil {
		panic(err)
	}
	c := cudaattn.KDAConfig{Dimension: 128, Heads: 1, Sequences: 1, Tokens: 1, Scale: 1}
	k, err := cudaattn.CompileKDA(dev, c)
	if err != nil {
		if k != nil {
			k.Close()
		}
		stream.Close()
		panic(err)
	}
	var allocations []cudamem.Slice[float32]
	defer func() {
		if err := k.Close(); err != nil {
			panic(err)
		}
		if err := stream.Close(); err != nil {
			panic(err)
		}
		for _, s := range allocations {
			if err := s.Free(); err != nil {
				panic(err)
			}
		}
	}()
	alloc := func(data []float32) cudamem.Slice[float32] {
		s, err := cudamem.Alloc[float32](dev, len(data))
		if err != nil {
			panic(err)
		}
		allocations = append(allocations, s)
		if err := s.CopyFromHost(data); err != nil {
			panic(err)
		}
		return s
	}
	unit := make([]float32, 128)
	unit[0] = 1
	value, decay := make([]float32, 128), make([]float32, 128)
	for i := range value {
		value[i] = 0.5
		decay[i] = 0.9
	}
	initialValues := make([]float32, 128*128)
	for i := range initialValues {
		initialValues[i] = 0.01
	}
	ids := []cudaattn.KDAIdentity{{Sequence: "sequence-a", Mode: cudaattn.KDAFP32}}
	initial, err := cudaattn.NewKDAState(c, alloc(initialValues), ids)
	if err != nil {
		panic(err)
	}
	b := cudaattn.KDABuffers{Q: alloc(unit), Key: alloc(unit), Value: alloc(value), Decay: alloc(decay), Beta: alloc([]float32{0.5}), FinalState: alloc(make([]float32, 128*128)), Output: alloc(make([]float32, 128))}
	result, err := k.Execute(context.Background(), stream, initial, ids, []int32{0, 1}, b)
	if result != nil {
		if waitErr := result.Wait(context.Background()); waitErr != nil {
			panic(waitErr)
		}
	}
	if err != nil {
		panic(err)
	}
	state, err := result.State()
	if err != nil {
		panic(err)
	}
	fmt.Println("published prefix:", state.Identities()[0].Prefix)
	// The imported initial snapshot remains unchanged and can start another arm.
}

func ExampleKDASchema() {
	schema, err := cudaattn.KDASchema()
	fmt.Println(len(schema.Params), err)
	// Output: 13 <nil>
}
