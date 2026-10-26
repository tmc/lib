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

func ExampleMMAM16N8K16A() {
	row, colPair := cudahostemu.MMAM16N8K16A(5, 2)
	bRowPair, bCol := cudahostemu.MMAM16N8K16B(5, 1)
	cRow, cCol := cudahostemu.MMAM16N8C(5, 3)
	fmt.Printf("A=(%d,%d) B=(%d,%d) C=(%d,%d)\n", row, colPair, bRowPair, bCol, cRow, cCol)
	// Output: A=(1,10) B=(10,1) C=(9,3)
}

func ExampleLDMatrixM8N8X4() {
	mat, row, colBytes := cudahostemu.LDMatrixM8N8X4(19)
	fmt.Printf("matrix=%d row=%d colBytes=%d\n", mat, row, colBytes)
	// Output: matrix=2 row=3 colBytes=12
}

func ExampleWGMMADesc() {
	desc := cudahostemu.WGMMADesc(1024, 16, 1024, cudahostemu.Swizzle128B)
	swz := cudahostemu.SwizzleOffset(128+16, cudahostemu.Swizzle128B)
	fmt.Printf("desc=%#x swz=%d\n", desc, swz)
	// Output: desc=0x4000004000010040 swz=128
}
