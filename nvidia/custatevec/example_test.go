package custatevec_test

import (
	"fmt"
	"math"
	"runtime"
	"unsafe"

	"github.com/tmc/lib/nvidia/custatevec"
	"github.com/tmc/lib/nvidia/driver"
)

// Example shows the index convention that decides whether a caller's
// amplitudes mean what they think they mean. It runs anywhere, because it
// makes no library call: the arithmetic below is what the bit ordering
// documented by this package amounts to.
func Example() {
	const nIndexBits = 3
	fmt.Println(custatevec.BitOrdering)
	for qubit := 0; qubit < nIndexBits; qubit++ {
		// The basis state with only this qubit set.
		index := 1 << qubit
		fmt.Printf("qubit %d set -> state vector index %d\n", qubit, index)
	}
	// Reading a qubit's value back out of an index.
	const index = 5
	for qubit := 0; qubit < nIndexBits; qubit++ {
		fmt.Printf("index %d has qubit %d = %d\n", index, qubit, (index>>qubit)&1)
	}
	// Output:
	// little-endian: index bit 0 is the least significant bit
	// qubit 0 set -> state vector index 1
	// qubit 1 set -> state vector index 2
	// qubit 2 set -> state vector index 4
	// index 5 has qubit 0 = 1
	// index 5 has qubit 1 = 0
	// index 5 has qubit 2 = 1
}

// Example_bellState builds a two-qubit Bell state on a GPU and reads back the
// probability of each value of qubit 0.
//
// It has no "Output" comment, so `go test` compiles it but never runs it: it
// needs an NVIDIA GPU, the CUDA driver and libcustatevec, none of which a test
// machine is assumed to have. It is written to be read as the shape of a
// correct call sequence, and to be run by hand on a machine that has them.
func Example_bellState() {
	const nIndexBits = 2
	const nAmplitudes = 1 << nIndexBits

	// The CUDA context. cuStateVec operates on the current context's device.
	if err := driver.Init(0); err != nil {
		panic(err)
	}
	device, err := driver.NewDeviceGet(0)
	if err != nil {
		panic(err)
	}
	ctx, err := device.PrimaryCtxRetain()
	if err != nil {
		panic(err)
	}
	defer device.PrimaryCtxRelease()
	if err := ctx.SetCurrent(); err != nil {
		panic(err)
	}

	// The state vector is device memory this package neither allocates nor
	// frees. Its size is 2^nIndexBits complex elements.
	sv, err := driver.MemAlloc(nAmplitudes * uint(unsafe.Sizeof(custatevec.Complex128{})))
	if err != nil {
		panic(err)
	}
	defer sv.MemFree()

	h, err := custatevec.Create()
	if err != nil {
		panic(err)
	}
	defer h.Destroy()

	if err := h.InitializeStateVector(uintptr(sv), custatevec.C64F, nIndexBits,
		custatevec.StateVectorTypeZero); err != nil {
		panic(err)
	}

	// Gate matrices are host memory here, read by the library during the call,
	// so they are pinned across it. The bit-position arrays are host memory
	// too and are pinned for the same reason.
	var pin runtime.Pinner
	defer pin.Unpin()
	invSqrt2 := 1 / math.Sqrt2
	hadamard := custatevec.AlignedComplex128(4)
	hadamard[0] = custatevec.MakeComplex128(complex(invSqrt2, 0))
	hadamard[1] = custatevec.MakeComplex128(complex(invSqrt2, 0))
	hadamard[2] = custatevec.MakeComplex128(complex(invSqrt2, 0))
	hadamard[3] = custatevec.MakeComplex128(complex(-invSqrt2, 0))
	pauliX := custatevec.AlignedComplex128(4)
	pauliX[1] = custatevec.MakeComplex128(1)
	pauliX[2] = custatevec.MakeComplex128(1)
	pin.Pin(&hadamard[0])
	pin.Pin(&pauliX[0])

	qubit0 := []int32{0}
	qubit1 := []int32{1}
	pin.Pin(&qubit0[0])
	pin.Pin(&qubit1[0])

	// H on qubit 0.
	if err := h.ApplyMatrix(uintptr(sv), custatevec.C64F, nIndexBits,
		uintptr(unsafe.Pointer(&hadamard[0])), custatevec.C64F, custatevec.MatrixLayoutRow,
		0, &qubit0[0], 1, nil, nil, 0, custatevec.Compute64F, 0, 0); err != nil {
		panic(err)
	}
	// X on qubit 1, controlled by qubit 0.
	if err := h.ApplyMatrix(uintptr(sv), custatevec.C64F, nIndexBits,
		uintptr(unsafe.Pointer(&pauliX[0])), custatevec.C64F, custatevec.MatrixLayoutRow,
		0, &qubit1[0], 1, &qubit0[0], nil, 1, custatevec.Compute64F, 0, 0); err != nil {
		panic(err)
	}

	// The probability of each value of qubit 0. abs2sum is an address because
	// the library accepts host or device memory there; this one is host memory
	// the caller pinned.
	probabilities := make([]float64, 2)
	pin.Pin(&probabilities[0])
	if err := h.Abs2SumArray(uintptr(sv), custatevec.C64F, nIndexBits,
		uintptr(unsafe.Pointer(&probabilities[0])), &qubit0[0], 1, nil, nil, 0); err != nil {
		panic(err)
	}
	fmt.Printf("P(qubit 0 = 0) = %.3f\nP(qubit 0 = 1) = %.3f\n", probabilities[0], probabilities[1])
}
