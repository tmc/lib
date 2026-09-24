package cuptisass_test

import (
	"context"
	"encoding/binary"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"testing"
	"time"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cuptisass"
	"github.com/tmc/lib/nvidia/x/internal/cudatest"
)

// channelKernels is NVBit's ChannelDev (core/utils/channel.hpp:60-124) over
// offsets rather than pointers, a producer in which lane 0 of every warp pushes
// one 16-byte record {cta_id_x, id, 0, 0}, and record_reg_vals' flush kernel
// (tool_func/flush_channel.cu:5-14), which pushes the sentinel and flushes.
const channelKernels = `
#include <stdint.h>
struct Chan {
  uint8_t *buf;
  unsigned long long head, tail, size;
  volatile int *doorbell;
};
__device__ Chan ch;
struct Rec { int cta_id_x; unsigned id; int pad[2]; };

__device__ unsigned long long load(unsigned long long *p) { return *(volatile unsigned long long *)p; }
__device__ void store(unsigned long long *p, unsigned long long v) { *(volatile unsigned long long *)p = v; }

__device__ void flush() {
  unsigned long long n = load(&ch.tail);
  if (n == 0) return;
  __threadfence_system();
  *ch.doorbell = (int)n;
  while (*ch.doorbell != 0) {}
  store(&ch.tail, 0);
  __threadfence();
  store(&ch.head, 0);
}

__device__ void push(const void *rec, unsigned n) {
  unsigned long long curr;
  for (;;) {
    curr = atomicAdd(&ch.head, (unsigned long long)n);
    if (curr + n <= ch.size) break;
    if (curr <= ch.size) {
      while (load(&ch.tail) != curr) {}
      flush();
    } else {
      while (load(&ch.head) > ch.size) {}
    }
  }
  memcpy(ch.buf + curr, rec, n);
  atomicAdd(&ch.tail, (unsigned long long)n);
}

extern "C" __global__ void chan_init(uint8_t *buf, unsigned long long size, int *doorbell) {
  ch.buf = buf; ch.head = 0; ch.tail = 0; ch.size = size; ch.doorbell = doorbell;
}

extern "C" __global__ void chan_produce() {
  if ((threadIdx.x & 31) == 0) {
    Rec r = {(int)blockIdx.x, blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, {0, 0}};
    push(&r, sizeof r);
  }
}

extern "C" __global__ void chan_finish(int sentinel) {
  if (sentinel) {
    Rec r = {-1, 0, {0, 0}};
    push(&r, sizeof r);
  }
  flush();
}
`

// TestTelemetryChannel_NativeMappedChannel runs the channel against a device:
// kernels push into page-locked host memory mapped into the device, and
// Receive drains it while they run. The no-sentinel run is the device mutant:
// the same kernels with the flush kernel's sentinel removed must leave the
// stream unterminated.
func TestTelemetryChannel_NativeMappedChannel(t *testing.T) {
	if os.Getenv("CUPTISASS_NATIVE_CHANNEL") != "1" {
		t.Skip("native qualification pending: CUPTISASS_NATIVE_CHANNEL=1 required")
	}
	// gpu-lock exports GPU_LOCK_HELD=$$, the holder's pid.
	if os.Getenv("GPU_LOCK_HELD") == "" {
		t.Fatal("native gate enabled but no GPU_LOCK_HELD: run under gpu-lock")
	}
	target := os.Getenv("CUPTISASS_NATIVE_CHANNEL_ARCH")
	if target == "" {
		target = "sm_121a"
	}
	if err := cuptisass.CheckDeviceChannel(target); !errors.Is(err, cudaerrors.ErrHardwarePending) {
		t.Fatalf("CheckDeviceChannel(%q) = %v; this run qualifies a pending target", target, err)
	}
	nvcc, _ := cudatest.NVCC(t)
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	dir, err := os.MkdirTemp(filepath.Join(home, "tmp"), "cuptisass-native-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(dir)
	src := filepath.Join(dir, "chan.cu")
	if err := os.WriteFile(src, []byte(channelKernels), 0o644); err != nil {
		t.Fatal(err)
	}
	cubin := filepath.Join(dir, "chan.cubin")
	bctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
	defer cancel()
	if out, err := exec.CommandContext(bctx, nvcc, "-cubin", "-arch="+target, "-o", cubin, src).CombinedOutput(); err != nil {
		t.Fatalf("nvcc: %v\n%s", err, out)
	}
	image, err := os.ReadFile(cubin)
	if err != nil {
		t.Fatal(err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := driver.Init(0); err != nil {
		t.Fatalf("driver.Init: %v", err)
	}
	dev, err := driver.DeviceGet(0)
	if err != nil {
		t.Fatalf("DeviceGet: %v", err)
	}
	cuctx, err := dev.PrimaryCtxRetain()
	if err != nil {
		t.Fatalf("PrimaryCtxRetain: %v", err)
	}
	defer dev.PrimaryCtxRelease()
	if err := cuctx.SetCurrent(); err != nil {
		t.Fatalf("SetCurrent: %v", err)
	}
	mod, err := driver.ModuleLoadData(image)
	if err != nil {
		t.Fatalf("ModuleLoadData: %v", err)
	}
	fn := func(name string) driver.Function {
		f, err := mod.Function(name)
		if err != nil {
			t.Fatalf("Function(%s): %v", name, err)
		}
		return f
	}
	initK, produce, finish := fn("chan_init"), fn("chan_produce"), fn("chan_finish")

	const blocks, threads = 64, 256
	const want = blocks * threads / 32
	// 100 records: the 512 records need at least six flushes.
	const size = 16 * 100

	hostAlloc := func(n uint) (unsafe.Pointer, driver.DevicePtr) {
		var raw uintptr
		if err := driver.MemHostAlloc(&raw, n, driver.MemhostallocDevicemap); err != nil {
			t.Fatalf("MemHostAlloc(%d, DEVICEMAP): %v", n, err)
		}
		t.Cleanup(func() { driver.MemFreeHost(raw) })
		d, err := driver.MemHostGetDevicePointer(raw, 0)
		if err != nil {
			t.Fatalf("MemHostGetDevicePointer: %v", err)
		}
		p := *(*unsafe.Pointer)(unsafe.Pointer(&raw))
		clear(unsafe.Slice((*byte)(p), n))
		return p, d
	}

	run := func(t *testing.T, sentinel int32, timeout time.Duration) (cuptisass.ChannelStats, []uint32, error) {
		bufp, bufd := hostAlloc(size)
		dbp, dbd := hostAlloc(4)
		host, err := cuptisass.NewChannel((*int32)(dbp), unsafe.Slice((*byte)(bufp), size))
		if err != nil {
			t.Fatal(err)
		}
		sz := uint64(size)
		if err := initK.LaunchKernel(1, 1, 1, 1, 1, 1, 0, 0, []unsafe.Pointer{unsafe.Pointer(&bufd), unsafe.Pointer(&sz), unsafe.Pointer(&dbd)}); err != nil {
			t.Fatalf("launch chan_init: %v", err)
		}
		if err := produce.LaunchKernel(blocks, 1, 1, threads, 1, 1, 0, 0, nil); err != nil {
			t.Fatalf("launch chan_produce: %v", err)
		}
		if err := finish.LaunchKernel(1, 1, 1, 1, 1, 1, 0, 0, []unsafe.Pointer{unsafe.Pointer(&sentinel)}); err != nil {
			t.Fatalf("launch chan_finish: %v", err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		defer cancel()
		var ids []uint32
		st, rerr := host.Receive(ctx, cuptisass.RecordLayout{Size: 16}, func(rec []byte) error {
			cta := binary.LittleEndian.Uint32(rec)
			id := binary.LittleEndian.Uint32(rec[4:])
			if cta != id/(threads/32) {
				t.Errorf("record %d has cta_id_x %d, want %d", id, cta, id/(threads/32))
			}
			ids = append(ids, id)
			return nil
		})
		if err := driver.CtxSynchronize(); err != nil {
			t.Fatalf("CtxSynchronize: %v", err)
		}
		return st, ids, rerr
	}

	t.Run("sentinel", func(t *testing.T) {
		st, ids, err := run(t, 1, time.Minute)
		if err != nil {
			t.Fatalf("Receive: %v (stats %+v)", err, st)
		}
		slices.Sort(ids)
		if len(ids) != want || ids[0] != 0 || ids[len(ids)-1] != want-1 || len(slices.Compact(ids)) != want {
			t.Errorf("received %d records, want ids 0-%d once each", len(ids), want-1)
		}
		if !st.Terminated || st.Batches < (want+1+99)/100 {
			t.Errorf("stats %+v, want Terminated and at least %d batches", st, (want+1+99)/100)
		}
		t.Logf("%s: %+v", target, st)
	})
	t.Run("no sentinel", func(t *testing.T) {
		st, ids, err := run(t, 0, 5*time.Second)
		if !errors.Is(err, context.DeadlineExceeded) || st.Terminated {
			t.Errorf("without the sentinel: %v, stats %+v; want DeadlineExceeded and not Terminated", err, st)
		}
		if len(ids) != want {
			t.Errorf("without the sentinel: %d records, want all %d delivered before the deadline", len(ids), want)
		}
	})
}
