package cudaptx_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/cudaptx"
)

func ExampleParseString() {
	ptx := `.version 8.0
.target sm_90
.address_size 64

.visible .entry VectorAdd(
	.param .u64 in1,
	.param .u64 in2,
	.param .u64 out
)
{
	ld.param.u64 %rd1, [in1];
	fma.rn.f32 %f1, %f2, %f3, %f4;
	st.global.f32 [%rd1], %f1;
	ret;
}
`

	mod, err := cudaptx.ParseString(ptx)
	if err != nil {
		panic(err)
	}

	kernel := mod.Kernel("VectorAdd")
	fmt.Printf("Kernel: %s\n", kernel.Name)
	fmt.Printf("FMA Line: %d\n", mod.FirstOpcodeLine("VectorAdd", "fma"))
	// Output:
	// Kernel: VectorAdd
	// FMA Line: 12
}
