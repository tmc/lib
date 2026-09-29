//go:build darwin || linux

package driver

import (
	"os"
	"os/exec"
	"runtime"
	"runtime/debug"
	"sync"
	"sync/atomic"
	"testing"
	"time"
	"unsafe"

	"github.com/ebitengine/purego"
)

// lifetimeStub stands in for libcuda. Each function first calls back into
// Go, where the hook collects garbage and moves the goroutine stack, and only
// then reads its pointer arguments or writes its outputs. A wrapper that let
// an argument die or move before the C call returned fails the check, reads
// clobbered memory, or trips a finalizer while the call is running.
const lifetimeStub = `
#include <stddef.h>
#include <stdint.h>
static void (*hook)(void);
void stubSetHook(void (*f)(void)) { hook = f; }
static void churn(void) { if (hook) hook(); }

// gridDimX is the parameter count and gridDimY their expected sum.
int cuLaunchKernel(void *f, unsigned gx, unsigned gy, unsigned gz,
	unsigned bx, unsigned by, unsigned bz, unsigned shmem,
	void *stream, void **params, void **extra) {
	churn();
	int64_t sum = 0;
	for (unsigned i = 0; i < gx; i++) sum += *(int64_t *)params[i];
	return sum == (int64_t)gy && extra == NULL ? 0 : 1;
}

typedef struct { int id; char pad[4]; union { char pad[64]; int i; unsigned d[3]; } v; } attr;
typedef struct {
	unsigned grid[3], block[3], shmem;
	void *stream;
	attr *attrs;
	unsigned numAttrs;
} config;
int cuLaunchKernelEx(const config *c, void *f, void **params, void **extra) {
	churn();
	int64_t sum = 0;
	for (unsigned i = 0; i < c->grid[0]; i++) sum += *(int64_t *)params[i];
	for (unsigned i = 0; i < c->numAttrs; i++) sum += c->attrs[i].v.i;
	return sum == (int64_t)c->grid[1] ? 0 : 1;
}

int cuDeviceGet(int *d, int ordinal) { churn(); *d = ordinal + 100; return 0; }

int cuMemcpyHtoDAsync_v2(uint64_t dst, const void *src, size_t n, void *stream) {
	churn();
	const unsigned char *p = src;
	for (size_t i = 0; i < n; i++)
		if (p[i] != (unsigned char)(i * 7)) return 1;
	return dst == 0x7f0000000000ull ? 0 : 1;
}

// The host function runs before the launch returns, on the calling thread,
// so a Go callback reenters the bridge while the outer call is in C.
int cuLaunchHostFunc_v2(void *stream, void (*fn)(void *), void *user, unsigned mode) {
	churn();
	fn(user);
	churn();
	return 0;
}
`

var (
	symStubSetHook uintptr

	// earlyFree counts arguments finalized while their own call was still
	// in progress: the collector freed memory the C side was using.
	earlyFree atomic.Int32

	hookOnce sync.Once
	hookFn   uintptr
)

// churnHook runs on the calling goroutine, inside the C call. It collects
// twice, lets finalizers run, and grows the stack so that it moves.
func churnHook() {
	runtime.GC()
	runtime.GC()
	time.Sleep(time.Millisecond)
	growStackBy(64)
}

//go:noinline
func growStackBy(n int) byte {
	var pad [512]byte
	pad[n%len(pad)] = byte(n)
	if n == 0 {
		return pad[0]
	}
	return growStackBy(n-1) + pad[n%len(pad)]
}

// watch sets a finalizer on p that records a free while inCall is set.
// inCall belongs to one call, so an argument of an earlier call that is
// collected during a later one does not count.
func watch[T any](inCall *atomic.Bool, p *T) {
	runtime.SetFinalizer(p, func(*T) {
		if inCall.Load() {
			earlyFree.Add(1)
		}
	})
}

func bindLifetimeStub(t *testing.T) {
	bindStub(t, lifetimeStub, map[string]*uintptr{
		"stubSetHook":          &symStubSetHook,
		"cuLaunchKernel":       &symlaunchKernel,
		"cuLaunchKernelEx":     &symlaunchKernelEx,
		"cuDeviceGet":          &symdeviceGet,
		"cuMemcpyHtoDAsync_v2": &symmemcpyHtoDAsync,
		"cuLaunchHostFunc_v2":  &symlaunchHostFunc,
	})
	hookOnce.Do(func() { hookFn = purego.NewCallback(churnHook) })
	purego.SyscallN(symStubSetHook, hookFn)
	t.Cleanup(func() { purego.SyscallN(symStubSetHook, 0) })
	earlyFree.Store(0)
}

// kernelArgs returns n fresh heap parameters, each watched, whose values
// sum to the returned total. Nothing else refers to them once the call
// starts, so only the bridge keeps them alive.
func kernelArgs(inCall *atomic.Bool, n int) ([]unsafe.Pointer, int64) {
	params := make([]unsafe.Pointer, n)
	var sum int64
	for i := range params {
		v := new(int64)
		*v = int64(i*i + 3)
		sum += *v
		watch(inCall, v)
		params[i] = unsafe.Pointer(v)
	}
	return params, sum
}

// TestLifetime checks, without a GPU, that arguments the generated wrappers
// hand to C stay valid and in place while C runs Go code that collects
// garbage and moves the stack, and that pooled output cells are not shared
// between concurrent calls.
func TestLifetime(t *testing.T) {
	bindLifetimeStub(t)
	defer debug.SetGCPercent(debug.SetGCPercent(1))

	for _, tt := range []struct {
		name string
		call func(inCall *atomic.Bool) error
	}{
		{"LaunchKernel", func(inCall *atomic.Bool) error {
			params, sum := kernelArgs(inCall, 5)
			return Function(1).LaunchKernel(Dim3{X: 5, Y: uint32(sum), Z: 1}, Dim3{X: 1, Y: 1, Z: 1}, 0, 0, params)
		}},
		{"LaunchKernelStackParam", func(*atomic.Bool) error {
			// x starts as a stack variable; the bridge must move it to the heap.
			var x int64 = 42
			return Function(1).LaunchKernel(Dim3{X: 1, Y: 42, Z: 1}, Dim3{X: 1, Y: 1, Z: 1}, 0, 0, []unsafe.Pointer{unsafe.Pointer(&x)})
		}},
		{"LaunchKernelEx", func(inCall *atomic.Bool) error {
			params, sum := kernelArgs(inCall, 3)
			attrs := make([]LaunchAttribute, 2)
			watch(inCall, &attrs[0])
			attrs[0].Id = LaunchAttributeCooperative
			*attrs[0].Value.Cooperative() = 1
			attrs[1].Id = LaunchAttributeCooperative
			*attrs[1].Value.Cooperative() = 1
			cfg := &LaunchConfig{GridDim: Dim3{X: 3, Y: uint32(sum + 2), Z: 1}, BlockDim: Dim3{X: 1, Y: 1, Z: 1}, Attrs: attrs}
			// cfg itself may be collected during the call: the wrapper
			// copies it into a pooled C-layout cell, which refers to attrs.
			return Function(1).LaunchKernelEx(cfg, params)
		}},
		{"MemcpyHtoDAsync", func(inCall *atomic.Bool) error {
			buf := make([]byte, 4096)
			for i := range buf {
				buf[i] = byte(i * 7)
			}
			watch(inCall, &buf[0])
			return DevicePtr(0x7f0000000000).CopyFromHostAsyncPtr(unsafe.Pointer(&buf[0]), uint(len(buf)), 0)
		}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			earlyFree.Store(0)
			for range 20 {
				inCall := new(atomic.Bool)
				inCall.Store(true)
				err := tt.call(inCall)
				inCall.Store(false)
				if err != nil {
					t.Fatal(err)
				}
			}
			runtime.GC()
			time.Sleep(time.Millisecond)
			if n := earlyFree.Load(); n != 0 {
				t.Fatalf("%d arguments freed during the call", n)
			}
		})
	}
}

// TestLifetimeConcurrentOut checks that the pooled output cell of one call is
// never handed to another while C still owns it.
func TestLifetimeConcurrentOut(t *testing.T) {
	bindLifetimeStub(t)
	var wg sync.WaitGroup
	for g := range 8 {
		wg.Go(func() {
			for i := range 20 {
				ord := g*1000 + i
				d, err := DeviceGet(ord)
				if err != nil {
					t.Error(err)
					return
				}
				if d != Device(ord+100) {
					t.Errorf("DeviceGet(%d) = %d, want %d", ord, d, ord+100)
				}
			}
		})
	}
	wg.Wait()
}

var reentered atomic.Int32

// reenter is a host function that calls back into the driver while the
// outer launch is still in C.
func reenter(user uintptr) {
	d, err := DeviceGet(int(user))
	if err == nil && d == Device(user+100) {
		reentered.Add(1)
	}
}

var reenterOnce = sync.OnceValue(func() uintptr { return purego.NewCallback(reenter) })

// TestLifetimeReentry checks that a callback run from inside a bridged
// call can itself make bridged calls, which reuse the pools concurrently.
func TestLifetimeReentry(t *testing.T) {
	bindLifetimeStub(t)
	reentered.Store(0)
	for i := range 10 {
		if err := Stream(0).LaunchHostFunc(reenterOnce(), uintptr(i+1), 0); err != nil {
			t.Fatal(err)
		}
	}
	if n := reentered.Load(); n != 10 {
		t.Fatalf("reentered %d times, want 10", n)
	}
}

// TestLifetimeClobberFree reruns the lifetime tests with the runtime
// overwriting freed memory, so a use after free in C reads garbage and fails
// the stub's checks instead of reading stale but intact values.
func TestLifetimeClobberFree(t *testing.T) {
	if os.Getenv("LIFETIME_CHILD") != "" {
		t.Skip("child")
	}
	if testing.Short() {
		t.Skip("short")
	}
	exe, err := os.Executable()
	if err != nil {
		t.Skip(err)
	}
	cmd := exec.Command(exe, "-test.run=^TestLifetime(ConcurrentOut|Reentry)?$", "-test.count=1")
	cmd.Env = append(os.Environ(), "LIFETIME_CHILD=1", "GODEBUG=clobberfree=1", "GOGC=1")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("%v\n%s", err, out)
	}
}
