package cudensitymat_test

import (
	"math"
	"math/cmplx"
	"os"
	"runtime"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cudensitymat"
	"github.com/tmc/lib/nvidia/driver"
)

// TestLiouvillianAgainstCPU applies the superoperator -i[H, .] to a known
// 2x2 density matrix on a GPU and compares every element, the trace and the
// hermiticity of the result against the same quantity computed here in
// complex128 arithmetic.
//
// # This test has never been run
//
// The working tree that wrote it is a Mac with no NVIDIA GPU, no CUDA driver
// and no libcudensitymat, and the session was not permitted to reach a machine
// that has them. It is an unexecuted specification of what would count as
// evidence that these bindings work, not a passing test. Nothing below has
// been observed to succeed or to fail.
//
// It is opt-in so that absent hardware cannot look like success. Once
// CUDENSITYMAT_PHYSICAL_TEST=1 is set, every missing prerequisite is a
// failure and nothing skips.
//
// Prerequisites, all of them:
//   - libcudensitymat.so.0 from cuQuantum 26.06 on the loader path, or
//     CUDENSITYMAT_LIB_PATH pointing at it, together with the libraries it
//     links against: libcutensor, libcublas, libcusolver, libcudart.
//   - libcuda.so.1 from an installed NVIDIA driver, and one visible GPU.
//   - a GPU of an architecture cuQuantum 26.06 supports. The library returns
//     ErrArchMismatch rather than computing a wrong answer if it does not.
//
// A single process and a single rank are enough; nothing here is distributed.
//
// # What a failure means
//
// The element comparison is the load-bearing check. The trace and hermiticity
// checks are properties that -i[H, rho] satisfies for Hermitian H, so they
// hold for the right answer and, as TestNonPreservingActionBreaksInvariants
// shows on the same rig, not for every answer. Both are checked against a
// nonzero result, so an output buffer the library never wrote cannot pass.
func TestLiouvillianAgainstCPU(t *testing.T) {
	rig := newRig(t)

	// rho: Hermitian, trace 1, with a nonzero off-diagonal so that the
	// column-major layout and the ket/bra mode pairing both matter.
	rho := mat2{
		{0.7, complex(0.1, 0.2)},
		{complex(0.1, -0.2), 0.3},
	}
	// H = sigma_y. Hermitian and not symmetric, so a transposed layout gives
	// a different answer instead of the same one.
	h := mat2{
		{0, complex(0, -1)},
		{complex(0, 1), 0},
	}

	in := rig.newState(rho)
	out := rig.newState(mat2{})
	rig.zero(out)

	// The library computing the right trace of the input is the first
	// evidence that it reads the attached buffer with the layout assumed
	// here; it does not involve the operator at all.
	if got := rig.trace(in); cmplx.Abs(got-1) > tol {
		t.Fatalf("StateComputeTrace(rho) = %v, want 1; the state layout or the attached storage is wrong", got)
	}

	op := rig.commutator(h)
	rig.prepareAndApply(op, in, out)

	got := rig.read(out)
	want := scale(complex(0, -1), sub(mul(h, rho), mul(rho, h)))
	if diff := maxAbsDiff(got, want); diff > tol {
		t.Fatalf("-i[H,rho] =\n%v\nwant\n%v\nmax elementwise difference %g", got, want, diff)
	}
	if norm := maxAbs(got); norm < tol {
		t.Fatal("output is all zeros; the library did not write the attached output storage")
	}
	if diff := maxAbsDiff(got, dagger(got)); diff > tol {
		t.Errorf("result is not Hermitian: max |M - M^dagger| = %g", diff)
	}
	if tr := trace(got); cmplx.Abs(tr) > tol {
		t.Errorf("trace(-i[H,rho]) = %v, want 0", tr)
	}
	t.Logf("verified -i[H,rho] elementwise against the CPU, |result| = %g", maxAbs(got))

	// ComputeAction is documented as accumulative. Applying the same
	// operator again to the same input must double the output rather than
	// replace it; if it replaced it, a caller who forgot InitializeZero
	// would silently get a different answer than the one documented.
	rig.apply(op, in, out)
	if doubled, want2 := rig.read(out), scale(2, want); maxAbsDiff(doubled, want2) > tol {
		t.Errorf("second ComputeAction gave\n%v\nwant twice the first result\n%v", doubled, want2)
	}

	// The attached component storage is live, not copied at attach time.
	// Overwriting it and recomputing must change the answer.
	rig.write(in, scale(2, rho))
	rig.zero(out)
	rig.apply(op, in, out)
	if got, want2 := rig.read(out), scale(2, want); maxAbsDiff(got, want2) > tol {
		t.Errorf("after overwriting the attached input storage the action gave\n%v\nwant\n%v; the library is not re-reading caller storage", got, want2)
	}
}

// TestNonPreservingActionBreaksInvariants is the negative control for the
// trace and hermiticity checks above. It applies H from the left only, which
// is not a commutator, so the result must be neither Hermitian nor traceless.
// Without it, those two checks would pass on any Hermitian traceless output,
// including one produced for the wrong reason.
//
// Same gate and same prerequisites as TestLiouvillianAgainstCPU, and equally
// unexecuted.
func TestNonPreservingActionBreaksInvariants(t *testing.T) {
	rig := newRig(t)

	rho := mat2{
		{0.7, complex(0.1, 0.2)},
		{complex(0.1, -0.2), 0.3},
	}
	h := mat2{
		{0, complex(0, -1)},
		{complex(0, 1), 0},
	}

	in := rig.newState(rho)
	out := rig.newState(mat2{})
	rig.zero(out)

	// One term, duality zero: H acts on the state's ket mode, which is the
	// action from the left.
	elem := rig.elementary(h)
	term := rig.term(elem, 0, cudensitymat.Complex128{X: 1})
	op := rig.operator(term)
	rig.prepareAndApply(op, in, out)

	got := rig.read(out)
	want := mul(h, rho)
	if diff := maxAbsDiff(got, want); diff > tol {
		t.Fatalf("H rho =\n%v\nwant\n%v\nmax elementwise difference %g", got, want, diff)
	}
	// The point of the control: these two are the assertions the positive
	// test makes, and here they must not hold.
	if diff := maxAbsDiff(got, dagger(got)); diff <= tol {
		t.Errorf("H rho came out Hermitian (max |M - M^dagger| = %g); the hermiticity check in TestLiouvillianAgainstCPU has no discriminating power", diff)
	}
	if tr := trace(got); cmplx.Abs(tr) <= tol {
		t.Errorf("trace(H rho) = %v, want a nonzero value; the trace check in TestLiouvillianAgainstCPU has no discriminating power", tr)
	}

	// Duality must select the side. Acting from the right gives rho H, which
	// differs from H rho for this H and rho.
	right := rig.term(elem, 1, cudensitymat.Complex128{X: 1})
	rightOp := rig.operator(right)
	out2 := rig.newState(mat2{})
	rig.zero(out2)
	rig.prepareAndApply(rightOp, in, out2)
	if got2, want2 := rig.read(out2), mul(rho, h); maxAbsDiff(got2, want2) > tol {
		t.Fatalf("rho H =\n%v\nwant\n%v", got2, want2)
	}
}

// TestExpectationAgainstCPU checks the other bound path, the expectation
// value object, against Tr(H rho).
//
// The interpretation is inferred from the header, which says only that the
// object produces "the unnormalized expectation value" of the operator with
// respect to the state. For the operator built here, a single term applying H
// to the ket mode with unit coefficient, the reading taken is that the result
// is Tr(H rho). If cuDensityMat defines it differently this test fails, which
// is the correct outcome for a reading nobody has checked against hardware.
//
// Same gate and same prerequisites as TestLiouvillianAgainstCPU, and equally
// unexecuted.
func TestExpectationAgainstCPU(t *testing.T) {
	rig := newRig(t)

	rho := mat2{
		{0.7, complex(0.1, 0.2)},
		{complex(0.1, -0.2), 0.3},
	}
	h := mat2{
		{0, complex(0, -1)},
		{complex(0, 1), 0},
	}
	in := rig.newState(rho)

	op := rig.operator(rig.term(rig.elementary(h), 0, cudensitymat.Complex128{X: 1}))
	expectation, err := rig.handle.CreateExpectation(op)
	rig.check(err)
	t.Cleanup(func() { rig.check(expectation.Destroy()) })

	ws, err := rig.handle.CreateWorkspace()
	rig.check(err)
	t.Cleanup(func() { rig.check(ws.Destroy()) })
	rig.check(expectation.Prepare(rig.handle, in.handle, cudensitymat.Compute64F, workspaceLimit, ws, rig.stream))
	rig.attachWorkspace(ws)

	value := rig.alloc(1)
	rig.check(expectation.Compute(rig.handle, 0, 1, 0, nil, in.handle, uintptr(value), ws, rig.stream))
	rig.sync()

	got := rig.readPtr(value, 1)[0]
	want := trace(mul(h, rho))
	if cmplx.Abs(got-want) > tol {
		t.Errorf("expectation = %v, want Tr(H rho) = %v", got, want)
	}
	// Tr(H rho) for Hermitian H and Hermitian rho is real. Checking it
	// separately catches a conjugation error that a near-real want would not.
	if math.Abs(imag(got)) > tol {
		t.Errorf("expectation = %v, want a real value for Hermitian H and rho", got)
	}
}

// tol is the elementwise tolerance. Every quantity here is a sum of at most
// four double-precision products of numbers of order one, so the exact result
// and the computed one differ by a few ulps at most. A loose tolerance would
// let a genuinely wrong layout pass.
const tol = 1e-12

// workspaceLimit is the workspace size limit offered to the Prepare calls. It
// is far more than a 2x2 problem can need; the library reports what it
// actually wants and this test allocates that.
const workspaceLimit = 1 << 26

// elemSize is the size of one complex128 element, matching C64F.
const elemSize = 16

// rig holds the CUDA and cuDensityMat objects shared by the physical tests and
// tears them down in reverse order of creation, which is the order the header
// requires: an expectation before its operator, an operator before its terms,
// everything before the handle, and caller-owned device memory only after the
// object that was reading it is destroyed.
type rig struct {
	t      *testing.T
	handle cudensitymat.Handle
	stream cudensitymat.Stream
	drv    driver.Stream

	// prepared records the workspace prepared for each operator, so that a
	// repeated ComputeAction reuses the preparation rather than silently
	// running against an unprepared descriptor.
	prepared map[cudensitymat.Operator]cudensitymat.WorkspaceDescriptor

	// extents is the state's space-mode extent list. It is borrowed by the
	// library only for the duration of each call, but it is kept alive here
	// because several calls take it.
	extents []int64
	// size is the storage size the library asked for, per state component.
	size uint
}

func newRig(t *testing.T) *rig {
	t.Helper()
	if os.Getenv("CUDENSITYMAT_PHYSICAL_TEST") != "1" {
		t.Skip("set CUDENSITYMAT_PHYSICAL_TEST=1 to run against a real libcudensitymat and NVIDIA GPU; this test has never been run and is not evidence until it has")
	}
	// The CUDA context is per-thread, and every call below must see it.
	runtime.LockOSThread()
	t.Cleanup(runtime.UnlockOSThread)

	r := &rig{t: t, extents: []int64{2}, prepared: map[cudensitymat.Operator]cudensitymat.WorkspaceDescriptor{}}
	r.check(cudensitymat.Load(""))
	t.Logf("cuDensityMat version %d from %s", cudensitymat.GetVersion(), cudensitymat.Path())

	r.check(driver.Init(0))
	dev, err := driver.NewDeviceGet(0)
	r.check(err)
	name, err := dev.GetName()
	r.check(err)
	t.Logf("device 0: %s", name)
	ctx, err := dev.PrimaryCtxRetain()
	r.check(err)
	t.Cleanup(func() { r.check(dev.PrimaryCtxRelease()) })
	r.check(ctx.SetCurrent())

	r.drv, err = driver.NewStream(0)
	r.check(err)
	t.Cleanup(func() { r.check(r.drv.Destroy()) })
	r.stream = cudensitymat.Stream(r.drv)

	r.handle, err = cudensitymat.Create()
	r.check(err)
	t.Cleanup(func() { r.check(r.handle.Destroy()) })

	// Single process, single rank. If this is not what the library reports,
	// the rest of the test is measuring something else.
	ranks, err := r.handle.GetNumRanks()
	r.check(err)
	rank, err := r.handle.GetProcRank()
	r.check(err)
	if ranks != 1 || rank != 0 {
		t.Fatalf("library reports rank %d of %d; this test assumes a single process", rank, ranks)
	}
	return r
}

func (r *rig) check(err error) {
	r.t.Helper()
	if err != nil {
		r.t.Fatal(err)
	}
}

func (r *rig) sync() {
	r.t.Helper()
	r.check(r.drv.Synchronize())
}

// alloc returns n elements of caller-owned device memory, freed after the test.
func (r *rig) alloc(n int) driver.DevicePtr {
	r.t.Helper()
	p, err := driver.MemAlloc(uint(n) * elemSize)
	r.check(err)
	r.t.Cleanup(func() { r.check(p.MemFree()) })
	return p
}

func (r *rig) writePtr(p driver.DevicePtr, v []complex128) {
	r.t.Helper()
	r.check(p.MemcpyHtoD(unsafe.Pointer(unsafe.SliceData(v)), uint(len(v))*elemSize))
	runtime.KeepAlive(v)
}

func (r *rig) readPtr(p driver.DevicePtr, n int) []complex128 {
	r.t.Helper()
	v := make([]complex128, n)
	r.check(p.MemcpyDtoH(unsafe.Pointer(unsafe.SliceData(v)), uint(n)*elemSize))
	runtime.KeepAlive(v)
	return v
}

// state pairs a State with the caller-owned storage attached to it, since the
// two have to be freed in that order and read back together.
type state struct {
	handle  cudensitymat.State
	storage driver.DevicePtr
}

// newState creates a mixed state over one mode of extent 2, allocates the
// storage the library asks for, seeds it with m, and attaches it. The buffer
// stays owned here for the life of the state.
func (r *rig) newState(m mat2) state {
	r.t.Helper()
	s, err := r.handle.CreateState(cudensitymat.StatePurityMixed, 1, &r.extents[0], 1, cudensitymat.C64F)
	r.check(err)

	components, err := s.GetNumComponents(r.handle)
	r.check(err)
	if components != 1 {
		r.t.Fatalf("dense state reported %d components, want 1", components)
	}
	var size uint
	r.check(s.GetComponentStorageSize(r.handle, 1, &size))
	if size != 4*elemSize {
		r.t.Fatalf("component storage size = %d bytes, want %d for a 2x2 complex128 density matrix", size, 4*elemSize)
	}
	r.size = size

	buf := r.alloc(4)
	r.writePtr(buf, m.flat())
	ptr := uintptr(buf)
	r.check(s.AttachComponentStorage(r.handle, 1, &ptr, &size))
	// Destroy the state before the test frees the buffer it is reading.
	r.t.Cleanup(func() { r.check(s.Destroy()) })
	return state{handle: s, storage: buf}
}

func (r *rig) write(s state, m mat2) { r.writePtr(s.storage, m.flat()) }

func (r *rig) read(s state) mat2 {
	r.t.Helper()
	r.sync()
	return unflat(r.readPtr(s.storage, 4))
}

func (r *rig) zero(s state) {
	r.t.Helper()
	r.check(s.handle.InitializeZero(r.handle, r.stream))
	r.sync()
}

func (r *rig) trace(s state) complex128 {
	r.t.Helper()
	p := r.alloc(1)
	r.check(s.handle.ComputeTrace(r.handle, uintptr(p), r.stream))
	r.sync()
	return r.readPtr(p, 1)[0]
}

// elementary builds a 1-body dense elementary tensor operator holding m. Its
// GPU storage is caller-owned and must outlive the operator, and the header
// warns that two elementary operators must not share or overlap storage, so
// each call allocates its own buffer.
func (r *rig) elementary(m mat2) cudensitymat.ElementaryOperator {
	r.t.Helper()
	buf := r.alloc(4)
	r.writePtr(buf, m.flat())
	op, err := r.handle.CreateElementaryOperator(1, &r.extents[0],
		cudensitymat.OperatorSparsityNone, 0, nil, cudensitymat.C64F, uintptr(buf),
		cudensitymat.TensorCallbackNone, cudensitymat.TensorGradientCallbackNone)
	r.check(err)
	r.t.Cleanup(func() { r.check(op.Destroy()) })
	return op
}

// term wraps one elementary operator acting on state mode 0 into an operator
// term. duality zero applies it to the state's ket mode, that is, from the
// left; a positive duality applies it to the bra mode, from the right.
func (r *rig) term(elem cudensitymat.ElementaryOperator, duality int32, coefficient cudensitymat.Complex128) cudensitymat.OperatorTerm {
	r.t.Helper()
	term, err := r.handle.CreateOperatorTerm(1, &r.extents[0])
	r.check(err)
	modes := []int32{0}
	dualities := []int32{duality}
	elems := []cudensitymat.ElementaryOperator{elem}
	r.check(term.AppendElementaryProduct(r.handle, 1, &elems[0], &modes[0], &dualities[0],
		coefficient, cudensitymat.ScalarCallbackNone, cudensitymat.ScalarGradientCallbackNone))
	runtime.KeepAlive(modes)
	runtime.KeepAlive(dualities)
	runtime.KeepAlive(elems)
	r.t.Cleanup(func() { r.check(term.Destroy()) })
	return term
}

// operator collects terms into an operator with unit coefficients.
func (r *rig) operator(terms ...cudensitymat.OperatorTerm) cudensitymat.Operator {
	r.t.Helper()
	op, err := r.handle.CreateOperator(1, &r.extents[0])
	r.check(err)
	for _, term := range terms {
		r.check(op.AppendTerm(r.handle, term, 0, cudensitymat.Complex128{X: 1},
			cudensitymat.ScalarCallbackNone, cudensitymat.ScalarGradientCallbackNone))
	}
	// The operator has to outlive nothing here, but it must be destroyed
	// before the terms it holds, and Cleanup runs last-registered-first.
	r.t.Cleanup(func() { r.check(op.Destroy()) })
	return op
}

// commutator builds -i[H, .] as two terms: (-i) H acting from the left and
// (+i) H acting from the right. Each side gets its own elementary operator
// because the two must not share GPU storage.
func (r *rig) commutator(h mat2) cudensitymat.Operator {
	r.t.Helper()
	left := r.term(r.elementary(h), 0, cudensitymat.Complex128{X: 0, Y: -1})
	right := r.term(r.elementary(h), 1, cudensitymat.Complex128{X: 0, Y: 1})
	return r.operator(left, right)
}

// prepareAndApply sizes a workspace for op against the given state shapes,
// attaches the scratch it asks for, and runs one ComputeAction.
func (r *rig) prepareAndApply(op cudensitymat.Operator, in, out state) {
	r.t.Helper()
	ws, err := r.handle.CreateWorkspace()
	r.check(err)
	r.t.Cleanup(func() { r.check(ws.Destroy()) })
	r.check(op.PrepareAction(r.handle, in.handle, out.handle, cudensitymat.Compute64F,
		workspaceLimit, ws, r.stream))
	r.attachWorkspace(ws)
	r.prepared[op] = ws
	r.apply(op, in, out)
}

// apply runs ComputeAction with the workspace prepareAndApply prepared for op.
func (r *rig) apply(op cudensitymat.Operator, in, out state) {
	r.t.Helper()
	ws, ok := r.prepared[op]
	if !ok {
		r.t.Fatal("apply called before prepareAndApply for this operator")
	}
	r.check(op.ComputeAction(r.handle, 0, 1, 0, nil, in.handle, out.handle, ws, r.stream))
	r.sync()
}

// attachWorkspace allocates and attaches the scratch the preceding Prepare
// call asked for. The header says the requirement may be zero, in which case
// there is nothing to attach. The buffer is caller-owned and stays alive until
// the test ends, which is after the last stream synchronization.
func (r *rig) attachWorkspace(ws cudensitymat.WorkspaceDescriptor) {
	r.t.Helper()
	need, err := r.handle.WorkspaceGetMemorySize(ws, cudensitymat.MemspaceDevice, cudensitymat.WorkspaceScratch)
	r.check(err)
	r.t.Logf("workspace scratch: %d bytes", need)
	if need == 0 {
		return
	}
	buf, err := driver.MemAlloc(need)
	r.check(err)
	r.t.Cleanup(func() { r.check(buf.MemFree()) })
	r.check(r.handle.WorkspaceSetMemory(ws, cudensitymat.MemspaceDevice,
		cudensitymat.WorkspaceScratch, uintptr(buf), need))
	// Reading it back is the only check available that SetMemory recorded
	// what it was given rather than silently ignoring it.
	got, size, err := r.handle.WorkspaceGetMemory(ws, cudensitymat.MemspaceDevice, cudensitymat.WorkspaceScratch)
	r.check(err)
	if got != uintptr(buf) || size != need {
		r.t.Fatalf("WorkspaceGetMemory = (%#x, %d), want (%#x, %d)", got, size, uintptr(buf), need)
	}
}

// mat2 is a 2x2 matrix indexed [row][column].
type mat2 [2][2]complex128

// flat returns the generalized column-major storage cuDensityMat uses for a
// dense 2x2 tensor: the ket mode varies fastest, the bra mode slowest.
func (m mat2) flat() []complex128 {
	return []complex128{m[0][0], m[1][0], m[0][1], m[1][1]}
}

func unflat(v []complex128) mat2 {
	return mat2{{v[0], v[2]}, {v[1], v[3]}}
}

func mul(a, b mat2) mat2 {
	var out mat2
	for i := range 2 {
		for j := range 2 {
			for k := range 2 {
				out[i][j] += a[i][k] * b[k][j]
			}
		}
	}
	return out
}

func sub(a, b mat2) mat2 {
	var out mat2
	for i := range 2 {
		for j := range 2 {
			out[i][j] = a[i][j] - b[i][j]
		}
	}
	return out
}

func scale(s complex128, a mat2) mat2 {
	var out mat2
	for i := range 2 {
		for j := range 2 {
			out[i][j] = s * a[i][j]
		}
	}
	return out
}

func dagger(a mat2) mat2 {
	var out mat2
	for i := range 2 {
		for j := range 2 {
			out[i][j] = cmplx.Conj(a[j][i])
		}
	}
	return out
}

func trace(a mat2) complex128 { return a[0][0] + a[1][1] }

func maxAbs(a mat2) float64 {
	worst := 0.0
	for i := range 2 {
		for j := range 2 {
			worst = math.Max(worst, cmplx.Abs(a[i][j]))
		}
	}
	return worst
}

func maxAbsDiff(a, b mat2) float64 { return maxAbs(sub(a, b)) }
