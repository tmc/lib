package nvlshm

// Multi-process GPU integration test. The parent test discovers the GPU
// count, then re-execs itself once per rank with NVLSHM_TEST_RANK set; each
// child drives one GPU and they rendezvous over tcpboot on localhost.
//
// Run on a multi-GPU Linux box with NVLink peer access:
//
//	go test -run TestDistBufferGPU -v ./nvidia/x/nvlshm/
//
// The test skips when libcuda is unavailable or fewer than 2 devices exist.

import (
	"fmt"
	"net"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/nvlshm/tcpboot"
)

func TestMain(m *testing.M) {
	if os.Getenv("NVLSHM_TEST_RANK") != "" {
		os.Exit(gpuChildMain())
	}
	os.Exit(m.Run())
}

func TestDistBufferGPU(t *testing.T) {
	if err := driver.Init(0); err != nil {
		t.Skipf("CUDA unavailable: %v", err)
	}
	count, err := driver.DeviceGetCount()
	if err != nil {
		t.Skipf("cuDeviceGetCount: %v", err)
	}
	if count < 2 {
		t.Skipf("need >= 2 GPUs, have %d", count)
	}
	world := count
	if world > 8 {
		world = 8
	}

	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := ln.Addr().String()
	ln.Close()

	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	procs := make([]*exec.Cmd, world)
	for r := range world {
		cmd := exec.Command(exe, "-test.run", "TestDistBufferGPU")
		cmd.Env = append(os.Environ(),
			fmt.Sprintf("NVLSHM_TEST_RANK=%d", r),
			fmt.Sprintf("NVLSHM_TEST_WORLD=%d", world),
			fmt.Sprintf("NVLSHM_TEST_ADDR=%s", addr),
		)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		if err := cmd.Start(); err != nil {
			t.Fatal(err)
		}
		procs[r] = cmd
	}
	for r, cmd := range procs {
		if err := cmd.Wait(); err != nil {
			t.Errorf("rank %d: %v", r, err)
		}
	}
}

// gpuChildMain is one rank of the integration test. Exit code 0 = pass.
func gpuChildMain() int {
	rank, _ := strconv.Atoi(os.Getenv("NVLSHM_TEST_RANK"))
	world, _ := strconv.Atoi(os.Getenv("NVLSHM_TEST_WORLD"))
	addr := os.Getenv("NVLSHM_TEST_ADDR")
	fail := func(format string, args ...any) int {
		fmt.Fprintf(os.Stderr, "rank %d: "+format+"\n", append([]any{rank}, args...)...)
		return 1
	}

	runtime.LockOSThread()
	if err := driver.Init(0); err != nil {
		return fail("init: %v", err)
	}
	dev := driver.Device(rank)
	ctx, err := dev.PrimaryCtxRetain()
	if err != nil {
		return fail("primary ctx: %v", err)
	}
	if err := ctx.SetCurrent(); err != nil {
		return fail("set current: %v", err)
	}

	b, err := tcpboot.New(rank, world, addr)
	if err != nil {
		return fail("bootstrap: %v", err)
	}
	defer b.Close()

	gran, err := Granularity(dev)
	if err != nil {
		return fail("granularity: %v", err)
	}
	chunk := PadTo(1<<20, gran)

	buf, err := DistAlloc(b, dev, chunk)
	if err != nil {
		return fail("DistAlloc: %v", err)
	}

	// Each rank writes its pattern into the NEXT rank's chunk (a real
	// cross-GPU NVLink write), then everyone verifies its own chunk.
	peer := (rank + 1) % world
	pattern := uint8(0xA0 + peer)
	if err := buf.ChunkPtr(peer).MemsetD8(pattern, uint(chunk)); err != nil {
		return fail("remote memset: %v", err)
	}
	if err := driver.CtxSynchronize(); err != nil {
		return fail("sync: %v", err)
	}
	if err := b.Barrier(); err != nil {
		return fail("barrier: %v", err)
	}

	got := make([]byte, 4096)
	if err := buf.LocalPtr().MemcpyDtoH(unsafe.Pointer(&got[0]), uint(len(got))); err != nil {
		return fail("copy back: %v", err)
	}
	want := uint8(0xA0 + rank)
	for i, v := range got {
		if v != want {
			return fail("byte %d = %#x, want %#x", i, v, want)
		}
	}

	// Multicast: one store visible in every rank's chunk.
	if ok, _ := MulticastSupported(dev); ok {
		mcGran, err := MulticastGranularity(world)
		if err != nil {
			return fail("mc granularity: %v", err)
		}
		mcChunk := PadTo(chunk, mcGran)
		mcBuf, err := DistAllocMulticast(b, dev, mcChunk)
		if err != nil {
			return fail("DistAllocMulticast: %v", err)
		}
		if rank == 0 {
			mc, _ := mcBuf.McPtr()
			if err := mc.MemsetD8(0x5C, 4096); err != nil {
				return fail("multicast store: %v", err)
			}
			if err := driver.CtxSynchronize(); err != nil {
				return fail("mc sync: %v", err)
			}
		}
		if err := b.Barrier(); err != nil {
			return fail("mc barrier: %v", err)
		}
		if err := mcBuf.LocalPtr().MemcpyDtoH(unsafe.Pointer(&got[0]), 4096); err != nil {
			return fail("mc copy back: %v", err)
		}
		for i, v := range got {
			if v != 0x5C {
				return fail("mc byte %d = %#x, want 0x5c", i, v)
			}
		}
		if err := b.Barrier(); err != nil {
			return fail("pre-close barrier: %v", err)
		}
		if err := mcBuf.Close(); err != nil {
			return fail("mc close: %v", err)
		}
	} else if rank == 0 {
		fmt.Fprintln(os.Stderr, "multicast unsupported; skipping multicast phase")
	}

	if err := b.Barrier(); err != nil {
		return fail("final barrier: %v", err)
	}
	if err := buf.Close(); err != nil {
		return fail("close: %v", err)
	}
	if err := buf.Close(); err != nil { // idempotent
		return fail("second close: %v", err)
	}
	return 0
}
