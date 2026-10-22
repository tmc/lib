package cudakv_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudakv"
)

func ExampleNewSharedLayerTopology() {
	specs := []cudakv.SharedLayerSpec{
		{LayerID: 0, SourceLayerID: 0, Role: cudakv.LayerKVProducer, IsSWA: true, WindowSize: 128},
		{LayerID: 1, SourceLayerID: 1, Role: cudakv.LayerKVProducer, IsSWA: true, WindowSize: 128},
		{LayerID: 2, SourceLayerID: 1, Role: cudakv.LayerKVSharedConsumer, IsSWA: true, WindowSize: 128},
		{LayerID: 3, SourceLayerID: 1, Role: cudakv.LayerKVSharedConsumer, IsSWA: true, WindowSize: 128},
	}
	topo, err := cudakv.NewSharedLayerTopology(specs)
	if err != nil {
		panic(err)
	}
	slot3, err := topo.PhysicalSlotForLayer(3)
	if err != nil {
		panic(err)
	}
	fmt.Printf("producers=%v slot3=%d savings=%.2f\n", topo.ProducerLayers(), slot3, topo.MemorySavingsRatio())
	// Output:
	// producers=[0 1] slot3=1 savings=0.50
}

func ExampleTokenSlotOffsets() {
	nopeOff, ropeOff, scaleOff, err := cudakv.TokenSlotOffsets(cudakv.FormatMLA_V41_FP8, 4096, 64, 2)
	if err != nil {
		panic(err)
	}
	fmt.Printf("nope=%d rope=%d scale=%d\n", nopeOff, ropeOff, scaleOff)
	// Output:
	// nope=5152 rope=5600 scale=5664
}
