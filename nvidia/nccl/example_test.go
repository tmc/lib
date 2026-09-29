package nccl_test

import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nccl"
)

// Example queries the NCCL version and creates a single-process
// communicator spanning the local GPUs. It returns early when the NCCL
// library is not available.
func Example() {
	version, err := nccl.GetVersion()
	if err != nil {
		fmt.Println("NCCL unavailable:", err)
		return
	}
	fmt.Println("NCCL version:", version)

	devs := []int32{0}
	comms := make([]nccl.Comm, len(devs))
	err = nccl.CommInitAll(&comms[0], int32(len(devs)), &devs[0])
	if err != nil {
		fmt.Println("comm init:", err)
		return
	}
	defer comms[0].Destroy()

	n, _ := comms[0].Count()
	fmt.Println("communicator ranks:", n)
}

// Example_allReduce sums a float32 buffer across a one-rank communicator on
// the first CUDA device, in place. With one rank the sum is the input. It
// returns early when the CUDA driver or NCCL is not available. It has no
// Output comment, so it is compiled but not run: it needs a CUDA device.
func Example_allReduce() {
	// A CUDA context is current on an OS thread, not on a goroutine.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if err := driver.Init(0); err != nil {
		fmt.Println("CUDA driver unavailable:", err)
		return
	}
	dev, err := driver.DeviceGet(0)
	if err != nil {
		fmt.Println("device:", err)
		return
	}
	ctx, err := dev.PrimaryCtxRetain()
	if err != nil {
		fmt.Println("context:", err)
		return
	}
	defer dev.PrimaryCtxRelease()
	if err := ctx.SetCurrent(); err != nil {
		fmt.Println("set context:", err)
		return
	}

	devs := []int32{0}
	comms := make([]nccl.Comm, len(devs))
	if err := nccl.CommInitAll(&comms[0], int32(len(devs)), &devs[0]); err != nil {
		fmt.Println("comm init:", err)
		return
	}
	defer comms[0].Destroy()

	values := []float32{3, 5, 7, 11}
	size := uint(len(values)) * uint(unsafe.Sizeof(values[0]))
	p, err := driver.MemAllocBytes(size)
	if err != nil {
		fmt.Println("alloc:", err)
		return
	}
	defer p.Free()
	if err := p.CopyFromHost(values); err != nil {
		fmt.Println("copy in:", err)
		return
	}
	err = comms[0].AllReduce(p, p, uint(len(values)), nccl.Float32, nccl.Sum, 0)
	if err != nil {
		fmt.Println("all-reduce:", err)
		return
	}
	if err := p.CopyToHost(values); err != nil {
		fmt.Println("copy out:", err)
		return
	}
	fmt.Println("all-reduce =", values)
}

// ExampleParamBind reads an NCCL parameter by its environment variable name.
func ExampleParamBind() {
	p, err := nccl.ParamBind("NCCL_IB_ROCE_VERSION_NUM")
	if err != nil {
		fmt.Println("bind:", err)
		return
	}
	v, err := p.I32()
	if err != nil {
		fmt.Println("get:", err)
		return
	}
	fmt.Println("NCCL_IB_ROCE_VERSION_NUM =", v)
}

// ExampleWindow_PeerDevicePointer registers a symmetric window and prints,
// for each rank, a device pointer this rank can load from and store to. A
// zero pointer means the peer is not reachable by load/store.
func ExampleWindow_PeerDevicePointer() {
	var comm nccl.Comm // this rank's communicator, e.g. from CommInitRank

	const size = 1 << 20
	buf, err := nccl.MemAlloc(size)
	if err != nil {
		fmt.Println("alloc:", err)
		return
	}
	defer nccl.MemFree(buf)

	// WindowRegister is collective: every rank registers its own buffer.
	// Outside a group on a blocking communicator, win is set on return.
	var win nccl.Window
	if err := comm.WindowRegister(buf, size, &win, nccl.WinCollSymmetric); err != nil {
		fmt.Println("register:", err)
		return
	}
	defer comm.WindowDeregister(win)

	n, err := comm.Count()
	if err != nil {
		fmt.Println("count:", err)
		return
	}
	for peer := range n {
		p, err := win.PeerDevicePointer(0, peer)
		if err != nil {
			fmt.Println("peer pointer:", err)
			return
		}
		if p == 0 {
			fmt.Printf("rank %d: not load/store accessible\n", peer)
			continue
		}
		fmt.Printf("rank %d: %#x\n", peer, p)
	}
}
