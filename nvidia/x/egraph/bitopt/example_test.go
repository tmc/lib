package bitopt_test

import (
	"fmt"

	"github.com/tmc/lib/nvidia/x/egraph/bitopt"
)

// Decode eight E2M1 codes from one word to four bf16x2 registers: each
// half holds a magnitude at bits 6-8 and a sign at bit 15.
func ExampleSolve() {
	p := bitopt.Problem{Words: 1, Outputs: []uint32{0x81C081C0, 0x81C081C0, 0x81C081C0, 0x81C081C0}}
	sol, err := bitopt.Solve(p, 9)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Print(sol.CUDA("dequant"))
	// Output:
	// // dequant: 9 instructions (bitopt).
	// __device__ __forceinline__ void dequant(const uint32_t (&q)[1], uint32_t (&a)[4]) {
	//     const uint32_t t1 = q[0] << 4;
	//     const uint32_t t2 = __funnelshift_l(q[0], q[0], 10);
	//     const uint32_t t3 = q[0] << 6;
	//     const uint32_t t4 = q[0] >> 2;
	//     a[0] = (t3 & 0x80c080c0u) | (t4 & 0x01000100u);
	//     a[1] = (q[0] & 0x81c081c0u);
	//     a[2] = (t1 & 0x81c081c0u);
	//     a[3] = (t2 & 0x81c081c0u);
	// }
}
