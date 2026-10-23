package cudahostemu_test

import (
	"context"
	"encoding/binary"
	"fmt"
	"log"
	"math"
	"os"

	"github.com/tmc/lib/nvidia/x/cudahostemu"
)

func Example() {
	dir, err := os.MkdirTemp("", "cudahostemu")
	if err != nil {
		log.Fatal(err)
	}
	defer os.RemoveAll(dir)

	const src = `
__global__ void warp_sum(float* x) {
    float v = x[threadIdx.x];
    for (int m = 16; m; m >>= 1) v += __shfl_xor_sync(~0u, v, m);
    if (threadIdx.x == 0) x[0] = v;
}`
	x := make([]byte, 32*4)
	for i := range 32 {
		binary.LittleEndian.PutUint32(x[4*i:], math.Float32bits(float32(i)))
	}
	out, err := cudahostemu.Run(context.Background(), dir, cudahostemu.Program{
		Source:  src,
		Launch:  "eg_emu_launch(dim3(1), dim3(32), 0, warp_sum, x)",
		Buffers: []cudahostemu.Buffer{{Name: "x", CType: "float", Data: x}},
	})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(math.Float32frombits(binary.LittleEndian.Uint32(out["x"])))
	// Output: 496
}
