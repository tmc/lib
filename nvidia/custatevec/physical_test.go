package custatevec_test

import (
	"math"
	"math/cmplx"
	"runtime"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/custatevec"
	"github.com/tmc/lib/nvidia/driver"
)

// This file holds the physical consumer test: a deterministic gate sequence
// run by the real libcustatevec on a real GPU, with the amplitudes, the norm
// and the observation results compared against a host calculation written
// independently below.
//
// It has never been run. The machine it was written on has no NVIDIA GPU, so
// every run of it so far has skipped, and a skip is not evidence. The exact
// prerequisites are named in prerequisites below and repeated in the skip
// message, so that a reader of a test log can tell which one was missing.
//
// Note also what is NOT required: nothing here needs quantum hardware. A state
// vector simulator is ordinary GPU arithmetic. The blocker is a GPU, a driver
// and a library, not a quantum computer.
const prerequisites = `physical validation of this package requires, on one machine:
  1. an NVIDIA GPU with CUDA compute capability 7.0 or newer (cuQuantum 26.06
     supports Volta and later), and its kernel driver;
  2. libcuda.so from that driver, reachable by the dynamic loader;
  3. libcustatevec.so.1 from cuQuantum 26.06.0.17, the release this package's
     headers are pinned to, on Linux x86_64 or sbsa;
  4. Linux: cuQuantum ships no macOS or Windows build of libcustatevec.
No quantum hardware is involved: cuStateVec is a state-vector simulator that
runs on the GPU.`

// requireDevice loads the library and prepares a CUDA context, or skips with
// the precise prerequisite that was missing. It never returns a usable context
// and a skipped test at the same time.
func requireDevice(t *testing.T) {
	t.Helper()
	if err := custatevec.Load(""); err != nil {
		t.Skipf("SKIPPED, NOT VERIFIED: libcustatevec could not be loaded: %v\n\n%s", err, prerequisites)
	}
	if err := driver.Load(""); err != nil {
		t.Skipf("SKIPPED, NOT VERIFIED: the CUDA driver could not be loaded: %v\n\n%s", err, prerequisites)
	}
	if err := driver.Init(0); err != nil {
		t.Skipf("SKIPPED, NOT VERIFIED: cuInit failed: %v\n\n%s", err, prerequisites)
	}
	n, err := driver.DeviceGetCount()
	if err != nil {
		t.Fatalf("cuDeviceGetCount failed after cuInit succeeded: %v", err)
	}
	if n == 0 {
		t.Skipf("SKIPPED, NOT VERIFIED: the CUDA driver reports no devices\n\n%s", prerequisites)
	}
	// Past this point the machine has everything the test needs, so every
	// remaining failure is a real failure and is reported as one.
	device, err := driver.NewDeviceGet(0)
	if err != nil {
		t.Fatalf("cuDeviceGet(0): %v", err)
	}
	ctx, err := device.PrimaryCtxRetain()
	if err != nil {
		t.Fatalf("cuDevicePrimaryCtxRetain: %v", err)
	}
	t.Cleanup(func() { device.PrimaryCtxRelease() })
	if err := ctx.SetCurrent(); err != nil {
		t.Fatalf("cuCtxSetCurrent: %v", err)
	}
}

// reference is a host state vector with the same index convention this package
// documents: amplitude i has qubit b in state (i>>b)&1.
type reference []complex128

// apply multiplies a one-qubit matrix into the reference state, optionally
// under a control qubit. matrix is in row-major order.
func (r reference) apply(matrix [4]complex128, target int, control int) {
	for i := range r {
		if i>>target&1 != 0 {
			continue // visit each pair once, from its target-zero member
		}
		if control >= 0 && i>>control&1 == 0 {
			continue
		}
		j := i | 1<<target
		a, b := r[i], r[j]
		r[i] = matrix[0]*a + matrix[1]*b
		r[j] = matrix[2]*a + matrix[3]*b
	}
}

// TestBellStateOnDevice prepares |00>, applies a Hadamard to qubit 0 and a
// controlled X to qubit 1, and checks the resulting amplitudes, the norm, the
// per-qubit probabilities, a Z-basis parity measurement and an expectation
// value against the host calculation above.
//
// Failures here are loud. A missing GPU, driver or library is a skip naming
// which one was missing; anything else is a t.Fatal.
func TestBellStateOnDevice(t *testing.T) {
	requireDevice(t)

	const nIndexBits = 2
	const nAmplitudes = 1 << nIndexBits
	const elemSize = unsafe.Sizeof(custatevec.Complex128{})

	// The host reference, computed without the library.
	invSqrt2 := 1 / math.Sqrt2
	hadamardMatrix := [4]complex128{
		complex(invSqrt2, 0), complex(invSqrt2, 0),
		complex(invSqrt2, 0), complex(-invSqrt2, 0),
	}
	xMatrix := [4]complex128{0, 1, 1, 0}
	want := reference{1, 0, 0, 0}
	want.apply(hadamardMatrix, 0, -1)
	want.apply(xMatrix, 1, 0)

	sv, err := driver.MemAlloc(nAmplitudes * uint(elemSize))
	if err != nil {
		t.Fatalf("cuMemAlloc(%d): %v", nAmplitudes*uint(elemSize), err)
	}
	defer sv.MemFree()

	h, err := custatevec.Create()
	if err != nil {
		t.Fatalf("custatevecCreate: %v", err)
	}
	defer func() {
		if err := h.Destroy(); err != nil {
			t.Errorf("custatevecDestroy: %v", err)
		}
	}()

	// The workspace query must agree with itself: attaching the size it
	// reports is the documented way to give the handle its own workspace.
	if _, err := h.GetDefaultWorkspaceSize(); err != nil {
		t.Fatalf("custatevecGetDefaultWorkspaceSize: %v", err)
	}

	if err := h.InitializeStateVector(uintptr(sv), custatevec.C64F, nIndexBits,
		custatevec.StateVectorTypeZero); err != nil {
		t.Fatalf("custatevecInitializeStateVector: %v", err)
	}

	var pin runtime.Pinner
	defer pin.Unpin()
	hadamard := custatevec.AlignedComplex128(4)
	pauliX := custatevec.AlignedComplex128(4)
	z := custatevec.AlignedComplex128(4)
	for i := 0; i < 4; i++ {
		hadamard[i] = custatevec.MakeComplex128(hadamardMatrix[i])
		pauliX[i] = custatevec.MakeComplex128(xMatrix[i])
	}
	z[0] = custatevec.MakeComplex128(1)
	z[3] = custatevec.MakeComplex128(-1)
	pin.Pin(&hadamard[0])
	pin.Pin(&pauliX[0])
	pin.Pin(&z[0])

	qubit0 := []int32{0}
	qubit1 := []int32{1}
	both := []int32{0, 1}
	pin.Pin(&qubit0[0])
	pin.Pin(&qubit1[0])
	pin.Pin(&both[0])

	// The workspace queries must be asked before the calls that may need one.
	extra, err := h.ApplyMatrixGetWorkspaceSize(custatevec.C64F, nIndexBits,
		uintptr(unsafe.Pointer(&hadamard[0])), custatevec.C64F, custatevec.MatrixLayoutRow,
		0, 1, 0, custatevec.Compute64F)
	if err != nil {
		t.Fatalf("custatevecApplyMatrixGetWorkspaceSize: %v", err)
	}
	if extra != 0 {
		t.Fatalf("a one-qubit gate on %d index bits asked for %d bytes of extra workspace; "+
			"this test does not allocate one", nIndexBits, extra)
	}

	if err := h.ApplyMatrix(uintptr(sv), custatevec.C64F, nIndexBits,
		uintptr(unsafe.Pointer(&hadamard[0])), custatevec.C64F, custatevec.MatrixLayoutRow,
		0, &qubit0[0], 1, nil, nil, 0, custatevec.Compute64F, 0, 0); err != nil {
		t.Fatalf("custatevecApplyMatrix (H on qubit 0): %v", err)
	}
	if err := h.ApplyMatrix(uintptr(sv), custatevec.C64F, nIndexBits,
		uintptr(unsafe.Pointer(&pauliX[0])), custatevec.C64F, custatevec.MatrixLayoutRow,
		0, &qubit1[0], 1, &qubit0[0], nil, 1, custatevec.Compute64F, 0, 0); err != nil {
		t.Fatalf("custatevecApplyMatrix (X on qubit 1 controlled by qubit 0): %v", err)
	}
	if err := driver.CtxSynchronize(); err != nil {
		t.Fatalf("cuCtxSynchronize: %v", err)
	}

	// Amplitudes, read back from the device and compared elementwise.
	got := make([]custatevec.Complex128, nAmplitudes)
	if err := sv.MemcpyDtoH(unsafe.Pointer(&got[0]), nAmplitudes*uint(elemSize)); err != nil {
		t.Fatalf("cuMemcpyDtoH: %v", err)
	}
	const tol = 1e-12
	for i := range want {
		if d := cmplx.Abs(got[i].Complex() - want[i]); d > tol {
			t.Errorf("amplitude %d = %v, host reference %v (|difference| %g)",
				i, got[i].Complex(), want[i], d)
		}
	}
	// A wrong index convention would put the weight on indices 1 and 2 rather
	// than 0 and 3, so say so explicitly rather than only failing above.
	if cmplx.Abs(got[1].Complex()) > tol || cmplx.Abs(got[2].Complex()) > tol {
		t.Errorf("amplitudes 1 and 2 are not zero (%v, %v): the index-bit ordering "+
			"this package documents (%s) does not match the library's",
			got[1].Complex(), got[2].Complex(), custatevec.BitOrdering)
	}

	// Normalization.
	norm := 0.0
	for i := range got {
		norm += real(got[i].Complex())*real(got[i].Complex()) + imag(got[i].Complex())*imag(got[i].Complex())
	}
	if math.Abs(norm-1) > 1e-12 {
		t.Errorf("norm = %.17g, want 1", norm)
	}

	// Probabilities of each value of qubit 0, into pinned host memory.
	probabilities := make([]float64, 2)
	pin.Pin(&probabilities[0])
	if err := h.Abs2SumArray(uintptr(sv), custatevec.C64F, nIndexBits,
		uintptr(unsafe.Pointer(&probabilities[0])), &qubit0[0], 1, nil, nil, 0); err != nil {
		t.Fatalf("custatevecAbs2SumArray: %v", err)
	}
	for i, p := range probabilities {
		if math.Abs(p-0.5) > 1e-12 {
			t.Errorf("P(qubit 0 = %d) = %.17g, want 0.5", i, p)
		}
	}

	// The parity of both qubits of a Bell state is deterministic: the state
	// has support only on |00> and |11>, both even. The random number must not
	// change the answer, so try both ends of its range.
	for _, randnum := range []float64{0.0, 0.999} {
		parity, err := h.MeasureOnZBasis(uintptr(sv), custatevec.C64F, nIndexBits,
			&both[0], 2, randnum, custatevec.CollapseNone)
		if err != nil {
			t.Fatalf("custatevecMeasureOnZBasis(randnum=%v): %v", randnum, err)
		}
		if parity != 0 {
			t.Errorf("parity of both qubits = %d with randnum %v, want 0 for a Bell state",
				parity, randnum)
		}
	}

	// <Z_0> is 0 on the Bell state. expectationValue is a host pointer.
	var expectation custatevec.Complex128
	pin.Pin(&expectation)
	if _, err := h.ComputeExpectationGetWorkspaceSize(custatevec.C64F, nIndexBits,
		uintptr(unsafe.Pointer(&z[0])), custatevec.C64F, custatevec.MatrixLayoutRow,
		1, custatevec.Compute64F); err != nil {
		t.Fatalf("custatevecComputeExpectationGetWorkspaceSize: %v", err)
	}
	if err := h.ComputeExpectation(uintptr(sv), custatevec.C64F, nIndexBits,
		unsafe.Pointer(&expectation), custatevec.C64F, nil,
		uintptr(unsafe.Pointer(&z[0])), custatevec.C64F, custatevec.MatrixLayoutRow,
		&qubit0[0], 1, custatevec.Compute64F, 0, 0); err != nil {
		t.Fatalf("custatevecComputeExpectation: %v", err)
	}
	if cmplx.Abs(expectation.Complex()) > 1e-12 {
		t.Errorf("<Z_0> on the Bell state = %v, want 0", expectation.Complex())
	}

	// Collapsing on qubit 0 with a random number below P(0) must give |00>,
	// and then <Z_0> must be +1.
	measured, err := h.MeasureOnZBasis(uintptr(sv), custatevec.C64F, nIndexBits,
		&qubit0[0], 1, 0.25, custatevec.CollapseNormalizeAndZero)
	if err != nil {
		t.Fatalf("custatevecMeasureOnZBasis (collapsing): %v", err)
	}
	if measured != 0 {
		t.Fatalf("measured %d with randnum 0.25 and P(0) = 0.5, want 0", measured)
	}
	if err := h.ComputeExpectation(uintptr(sv), custatevec.C64F, nIndexBits,
		unsafe.Pointer(&expectation), custatevec.C64F, nil,
		uintptr(unsafe.Pointer(&z[0])), custatevec.C64F, custatevec.MatrixLayoutRow,
		&qubit0[0], 1, custatevec.Compute64F, 0, 0); err != nil {
		t.Fatalf("custatevecComputeExpectation after collapse: %v", err)
	}
	if d := cmplx.Abs(expectation.Complex() - 1); d > 1e-12 {
		t.Errorf("<Z_0> after collapsing to 0 = %v, want 1", expectation.Complex())
	}

	if err := sv.MemcpyDtoH(unsafe.Pointer(&got[0]), nAmplitudes*uint(elemSize)); err != nil {
		t.Fatalf("cuMemcpyDtoH after collapse: %v", err)
	}
	collapsed := reference{1, 0, 0, 0}
	for i := range collapsed {
		if d := cmplx.Abs(got[i].Complex() - collapsed[i]); d > tol {
			t.Errorf("collapsed amplitude %d = %v, want %v", i, got[i].Complex(), collapsed[i])
		}
	}
}

// TestVersionMatchesPinnedHeader checks that the library this package loads is
// the one its bindings were generated from. A cuQuantum minor release can add
// arguments to an existing call, so a caller running against a different
// version than the pinned header should know.
func TestVersionMatchesPinnedHeader(t *testing.T) {
	if err := custatevec.Load(""); err != nil {
		t.Skipf("SKIPPED, NOT VERIFIED: libcustatevec could not be loaded: %v\n\n%s", err, prerequisites)
	}
	major, err := custatevec.GetProperty(custatevec.MajorVersion)
	if err != nil {
		t.Fatalf("custatevecGetProperty(MAJOR_VERSION): %v", err)
	}
	if major != 1 {
		t.Errorf("libcustatevec major version %d; these bindings are generated from "+
			"CUSTATEVEC_VER_MAJOR 1 and the soname they load is libcustatevec.so.1", major)
	}
	minor, err := custatevec.GetProperty(custatevec.MinorVersion)
	if err != nil {
		t.Fatalf("custatevecGetProperty(MINOR_VERSION): %v", err)
	}
	if minor != 14 {
		t.Logf("libcustatevec is 1.%d; the pinned header is 1.14. The bound calls are "+
			"unchanged across 1.x minors, but regenerate before relying on that.", minor)
	}
	if v := custatevec.GetVersion(); v != uint(major*10000+minor*100) && v/10000 != uint(major) {
		t.Errorf("custatevecGetVersion() = %d, inconsistent with property major.minor %d.%d",
			v, major, minor)
	}
}
