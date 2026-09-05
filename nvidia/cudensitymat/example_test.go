package cudensitymat_test

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/tmc/lib/nvidia/cudensitymat"
	"github.com/tmc/lib/nvidia/driver"
)

// Example builds the Liouvillian superoperator -i[H, .] for a single
// two-level system and applies it to a density matrix held in caller-owned
// GPU memory. It returns early, printing why, when libcudensitymat or a CUDA
// device is not available, so it can be compiled and read on a machine that
// has neither.
//
// The shape it shows is the whole bound surface: a handle, a mixed [State]
// whose component storage the caller allocates and keeps, an [Operator]
// assembled from [ElementaryOperator] products, a [WorkspaceDescriptor] sized
// by PrepareAction and backed by caller-owned memory, and one ComputeAction
// that accumulates into a second state.
func Example() {
	// Every CUDA call below is bound to the context current on this thread.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if err := cudensitymat.Load(""); err != nil {
		fmt.Println("cuDensityMat unavailable:", err)
		return
	}
	if err := driver.Init(0); err != nil {
		fmt.Println("CUDA driver unavailable:", err)
		return
	}
	dev, err := driver.NewDeviceGet(0)
	if err != nil {
		fmt.Println("no CUDA device:", err)
		return
	}
	ctx, err := dev.PrimaryCtxRetain()
	if err != nil {
		fmt.Println("primary context:", err)
		return
	}
	defer dev.PrimaryCtxRelease()
	if err := ctx.SetCurrent(); err != nil {
		fmt.Println("set context:", err)
		return
	}
	stream, err := driver.NewStream(0)
	if err != nil {
		fmt.Println("stream:", err)
		return
	}
	defer stream.Destroy()

	handle, err := cudensitymat.Create()
	if err != nil {
		fmt.Println("create handle:", err)
		return
	}
	defer handle.Destroy()

	// One space mode of extent 2: a qubit. A mixed state over it is a 2x2
	// density matrix stored in the generalized column-major layout.
	extents := []int64{2}
	rho := []complex128{
		0.7, complex(0.1, -0.2), // column j=0: rho[0][0], rho[1][0]
		complex(0.1, 0.2), 0.3, // column j=1: rho[0][1], rho[1][1]
	}

	in, err := handle.CreateState(cudensitymat.StatePurityMixed, 1, &extents[0], 1, cudensitymat.C64F)
	if err != nil {
		fmt.Println("create state:", err)
		return
	}
	defer in.Destroy()
	out, err := handle.CreateState(cudensitymat.StatePurityMixed, 1, &extents[0], 1, cudensitymat.C64F)
	if err != nil {
		fmt.Println("create state:", err)
		return
	}
	defer out.Destroy()

	// The library reports the storage it needs; the caller allocates it, owns
	// it for the life of the state, and frees it only after State.Destroy.
	// The buffer's initial contents are the state's initial value.
	var size uint
	if err := in.GetComponentStorageSize(handle, 1, &size); err != nil {
		fmt.Println("component storage size:", err)
		return
	}
	inBuf, err := driver.MemAlloc(size)
	if err != nil {
		fmt.Println("device alloc:", err)
		return
	}
	defer inBuf.MemFree()
	outBuf, err := driver.MemAlloc(size)
	if err != nil {
		fmt.Println("device alloc:", err)
		return
	}
	defer outBuf.MemFree()
	if err := inBuf.MemcpyHtoD(unsafe.Pointer(unsafe.SliceData(rho)), size); err != nil {
		fmt.Println("copy state in:", err)
		return
	}
	inPtr, outPtr := uintptr(inBuf), uintptr(outBuf)
	if err := in.AttachComponentStorage(handle, 1, &inPtr, &size); err != nil {
		fmt.Println("attach storage:", err)
		return
	}
	if err := out.AttachComponentStorage(handle, 1, &outPtr, &size); err != nil {
		fmt.Println("attach storage:", err)
		return
	}
	// ComputeAction accumulates, so the output starts at zero.
	if err := out.InitializeZero(handle, cudensitymat.Stream(stream)); err != nil {
		fmt.Println("zero output:", err)
		return
	}

	// H = sigma_y, Hermitian, stored column-major like the state. Its GPU
	// storage is caller-owned too and must outlive the elementary operator.
	sigmaY := []complex128{0, complex(0, 1), complex(0, -1), 0}
	hBuf, err := driver.MemAlloc(uint(len(sigmaY)) * 16)
	if err != nil {
		fmt.Println("device alloc:", err)
		return
	}
	defer hBuf.MemFree()
	if err := hBuf.MemcpyHtoD(unsafe.Pointer(unsafe.SliceData(sigmaY)), uint(len(sigmaY))*16); err != nil {
		fmt.Println("copy operator:", err)
		return
	}
	h, err := handle.CreateElementaryOperator(1, &extents[0],
		cudensitymat.OperatorSparsityNone, 0, nil, cudensitymat.C64F, uintptr(hBuf),
		cudensitymat.TensorCallbackNone, cudensitymat.TensorGradientCallbackNone)
	if err != nil {
		fmt.Println("create elementary operator:", err)
		return
	}
	defer h.Destroy()

	// -i[H, rho] = (-i)(H rho) + (+i)(rho H). Mode action duality 0 applies H
	// to the state's ket mode, a nonzero duality to its bra mode.
	modes := []int32{0}
	left, err := handle.CreateOperatorTerm(1, &extents[0])
	if err != nil {
		fmt.Println("create term:", err)
		return
	}
	defer left.Destroy()
	ket := []int32{0}
	if err := left.AppendElementaryProduct(handle, 1, &h, &modes[0], &ket[0],
		cudensitymat.Complex128{X: 0, Y: -1},
		cudensitymat.ScalarCallbackNone, cudensitymat.ScalarGradientCallbackNone); err != nil {
		fmt.Println("append product:", err)
		return
	}
	right, err := handle.CreateOperatorTerm(1, &extents[0])
	if err != nil {
		fmt.Println("create term:", err)
		return
	}
	defer right.Destroy()
	bra := []int32{1}
	if err := right.AppendElementaryProduct(handle, 1, &h, &modes[0], &bra[0],
		cudensitymat.Complex128{X: 0, Y: 1},
		cudensitymat.ScalarCallbackNone, cudensitymat.ScalarGradientCallbackNone); err != nil {
		fmt.Println("append product:", err)
		return
	}

	liouvillian, err := handle.CreateOperator(1, &extents[0])
	if err != nil {
		fmt.Println("create operator:", err)
		return
	}
	defer liouvillian.Destroy()
	one := cudensitymat.Complex128{X: 1}
	for _, term := range []cudensitymat.OperatorTerm{left, right} {
		if err := liouvillian.AppendTerm(handle, term, 0, one,
			cudensitymat.ScalarCallbackNone, cudensitymat.ScalarGradientCallbackNone); err != nil {
			fmt.Println("append term:", err)
			return
		}
	}

	// PrepareAction reports the scratch it needs; the caller supplies it and
	// keeps it alive until the stream that used it is synchronized.
	ws, err := handle.CreateWorkspace()
	if err != nil {
		fmt.Println("create workspace:", err)
		return
	}
	defer ws.Destroy()
	if err := liouvillian.PrepareAction(handle, in, out, cudensitymat.Compute64F,
		1<<26, ws, cudensitymat.Stream(stream)); err != nil {
		fmt.Println("prepare:", err)
		return
	}
	need, err := handle.WorkspaceGetMemorySize(ws, cudensitymat.MemspaceDevice, cudensitymat.WorkspaceScratch)
	if err != nil {
		fmt.Println("workspace size:", err)
		return
	}
	if need > 0 {
		scratch, err := driver.MemAlloc(need)
		if err != nil {
			fmt.Println("workspace alloc:", err)
			return
		}
		defer scratch.MemFree()
		if err := handle.WorkspaceSetMemory(ws, cudensitymat.MemspaceDevice,
			cudensitymat.WorkspaceScratch, uintptr(scratch), need); err != nil {
			fmt.Println("workspace attach:", err)
			return
		}
	}

	if err := liouvillian.ComputeAction(handle, 0, 1, 0, nil, in, out, ws,
		cudensitymat.Stream(stream)); err != nil {
		fmt.Println("compute action:", err)
		return
	}
	// ComputeAction only enqueued work. Nothing in outBuf is readable until
	// the stream has been synchronized.
	if err := stream.Synchronize(); err != nil {
		fmt.Println("synchronize:", err)
		return
	}
	result := make([]complex128, len(rho))
	if err := outBuf.MemcpyDtoH(unsafe.Pointer(unsafe.SliceData(result)), size); err != nil {
		fmt.Println("copy result:", err)
		return
	}
	fmt.Printf("-i[H,rho][0][0] = %.3f\n", result[0])
}

// ExampleResult shows how a failed call reports itself. Every bound function
// returns the library's cudensitymatStatus_t as an error, and the concrete
// type is [Result], so a caller can compare against a specific code.
func ExampleResult() {
	err := error(cudensitymat.ErrInsufficientWorkspace)
	fmt.Println(err)
	fmt.Println(err == error(cudensitymat.ErrInsufficientWorkspace))
	// Output:
	// cudensitymat: ErrInsufficientWorkspace (19)
	// true
}
