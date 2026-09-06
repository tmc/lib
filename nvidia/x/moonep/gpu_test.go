//go:build !windows

package moonep

// Multi-process GPU integration test, same re-exec pattern as package
// nvlshm: the parent discovers GPUs and spawns one child per rank. Needs a
// multi-GPU Linux box with NVLink and NVSwitch multicast; skips otherwise.
//
//	go test -run TestBufferGPU -v ./nvidia/x/moonep/

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
	if os.Getenv("MOONEP_TEST_RANK") != "" {
		os.Exit(bufferChildMain())
	}
	os.Exit(m.Run())
}

func TestBufferGPU(t *testing.T) {
	if err := driver.Init(0); err != nil {
		t.Skipf("CUDA unavailable: %v", err)
	}
	count, err := driver.DeviceGetCount()
	if err != nil || count < 2 {
		t.Skipf("need >= 2 GPUs (err=%v count=%d)", err, count)
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
		cmd := exec.Command(exe, "-test.run", "TestBufferGPU")
		cmd.Env = append(os.Environ(),
			fmt.Sprintf("MOONEP_TEST_RANK=%d", r),
			fmt.Sprintf("MOONEP_TEST_WORLD=%d", world),
			fmt.Sprintf("MOONEP_TEST_ADDR=%s", addr),
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

func bufferChildMain() int {
	rank, _ := strconv.Atoi(os.Getenv("MOONEP_TEST_RANK"))
	world, _ := strconv.Atoi(os.Getenv("MOONEP_TEST_WORLD"))
	addr := os.Getenv("MOONEP_TEST_ADDR")
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

	cfg := Config{S: 512, H: 1024, K: 4, E: world * 4, R: world}
	buf, err := NewBuffer(b, dev, cfg)
	if err != nil {
		return fail("NewBuffer: %v", err)
	}

	// The barrier region must read zero in this rank's own chunk (every
	// rank zeroed every chunk during NewBuffer).
	var slots [barrierSlots]int32
	local, err := buf.Meta.LocalPtr()
	if err != nil {
		return fail("local meta: %v", err)
	}
	barrier := local + driver.DevicePtr(buf.Layout.BarrierOff*int32Bytes)
	if err := barrier.MemcpyDtoH(unsafe.Pointer(&slots[0]), barrierSlots*int32Bytes); err != nil {
		return fail("read barrier region: %v", err)
	}
	for i, v := range slots {
		if v != 0 {
			return fail("barrier slot %d = %d, want 0", i, v)
		}
	}

	// Cross-rank visibility through the meta buffer: rank r stamps its
	// rank into peer (r+1)'s src_info region; each rank verifies.
	peer := (rank + 1) % world
	src, err := buf.Meta.ChunkPtr(peer)
	if err != nil {
		return fail("peer meta: %v", err)
	}
	src += driver.DevicePtr(buf.Layout.SrcInfoOff * int32Bytes)
	if err := src.MemsetD32(uint32(rank+100), 4); err != nil {
		return fail("remote src_info write: %v", err)
	}
	if err := driver.CtxSynchronize(); err != nil {
		return fail("sync: %v", err)
	}
	if err := b.Barrier(); err != nil {
		return fail("barrier: %v", err)
	}
	var got [4]int32
	mine, err := buf.Meta.LocalPtr()
	if err != nil {
		return fail("local meta: %v", err)
	}
	mine += driver.DevicePtr(buf.Layout.SrcInfoOff * int32Bytes)
	if err := mine.MemcpyDtoH(unsafe.Pointer(&got[0]), 4*int32Bytes); err != nil {
		return fail("read src_info: %v", err)
	}
	wantFrom := int32((rank+world-1)%world + 100)
	for i, v := range got {
		if v != wantFrom {
			return fail("src_info[%d] = %d, want %d", i, v, wantFrom)
		}
	}

	// Multicast plan broadcast path: rank 0 stores through the multicast
	// address at PlanOff; every rank sees it in its own chunk.
	if rank == 0 {
		mc := buf.MetaMc() + driver.DevicePtr(buf.Layout.PlanOff*int32Bytes)
		if err := mc.MemsetD32(0x77, 4); err != nil {
			return fail("multicast store: %v", err)
		}
		if err := driver.CtxSynchronize(); err != nil {
			return fail("mc sync: %v", err)
		}
	}
	if err := b.Barrier(); err != nil {
		return fail("mc barrier: %v", err)
	}
	plan, err := buf.Meta.LocalPtr()
	if err != nil {
		return fail("local meta: %v", err)
	}
	plan += driver.DevicePtr(buf.Layout.PlanOff * int32Bytes)
	if err := plan.MemcpyDtoH(unsafe.Pointer(&got[0]), 4*int32Bytes); err != nil {
		return fail("read plan region: %v", err)
	}
	for i, v := range got {
		if v != 0x77 {
			return fail("plan[%d] = %#x, want 0x77", i, v)
		}
	}

	// Plan allocation round-trips.
	p, err := AllocatePlan(buf.Layout)
	if err != nil {
		return fail("AllocatePlan: %v", err)
	}
	if err := p.Close(); err != nil {
		return fail("plan close: %v", err)
	}

	if err := buf.Close(); err != nil {
		return fail("close: %v", err)
	}
	if err := buf.Close(); err != nil {
		return fail("second close: %v", err)
	}
	return 0
}
