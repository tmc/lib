package cudagreen

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"slices"
	"testing"
	"time"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// nativeDevice opens the native gate and returns device 0 with its
// architecture, SM count and a granularity whose minimum equals its
// alignment. It skips unless CUDA_GREEN_NATIVE=1 and fails without an
// inherited GPU lock. The caller's goroutine is locked to its thread.
func nativeDevice(t *testing.T) (driver.Device, string, int, Granularity) {
	t.Helper()
	if os.Getenv("CUDA_GREEN_NATIVE") != "1" {
		t.Skip("set CUDA_GREEN_NATIVE=1 to create green contexts on device 0")
	}
	if _, err := cudalock.Inherited(os.Getenv); err != nil {
		t.Fatalf("native gate open without inherited gpu lock: %v", err)
	}
	runtime.LockOSThread()
	t.Cleanup(runtime.UnlockOSThread)
	if err := driver.Init(0); err != nil {
		t.Fatal(err)
	}
	v, _ := driver.DriverGetVersion()
	t.Logf("driver version %d", v)
	if err := Supported(); err != nil {
		t.Fatal(err)
	}
	dev, err := driver.DeviceGet(0)
	if err != nil {
		t.Fatal(err)
	}
	major, minor := attr(t, dev, driver.DevAttrComputeCapabilityMajor), attr(t, dev, driver.DevAttrComputeCapabilityMinor)
	arch := fmt.Sprintf("sm_%d%d", major, minor)
	total, g, err := DeviceGranularity(dev)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%s: %d SMs, reported granularity %+v", arch, total, g)
	if g.Min == 0 || g.Alignment == 0 {
		if g, err = DefaultGranularity(arch); err != nil {
			t.Fatal(err)
		}
	}
	g.Alignment = roundUp(g.Min, g.Alignment)
	g.Min = g.Alignment
	return dev, arch, total, g
}

func attr(t *testing.T, dev driver.Device, a driver.DeviceAttribute) int {
	t.Helper()
	v, err := dev.Attribute(a)
	if err != nil {
		t.Fatalf("attribute %v: %v", a, err)
	}
	return int(v)
}

// TestNativeOpen creates the green contexts for the prefill-heavy and
// decode-heavy plans on device 0 and checks each reports its planned SM
// count. It runs only with CUDA_GREEN_NATIVE=1 and an inherited GPU lock.
func TestNativeOpen(t *testing.T) {
	dev, arch, total, g := nativeDevice(t)
	for _, regime := range []struct {
		name  string
		specs []Spec
	}{
		{"prefill", PrefillHeavy()},
		{"decode", DecodeHeavy()},
	} {
		t.Run(regime.name, func(t *testing.T) {
			plan, err := ComputePlan(arch, total, g, regime.specs)
			if err != nil {
				t.Fatal(err)
			}
			p, err := Open(dev, plan)
			if err != nil {
				t.Fatal(err)
			}
			defer p.Close()
			for i, q := range plan.Partitions {
				t.Logf("%v: %d SMs", q.Spec.Role, q.SMs)
				if err := p.Stream(i).Synchronize(); err != nil {
					t.Errorf("%v stream: %v", q.Spec.Role, err)
				}
			}
			if err := p.Close(); err != nil {
				t.Fatal(err)
			}
		})
	}
}

// jitterSource holds the two kernels of TestNativeJitter. spin keeps every
// thread of every block busy until a shared deadline: the first block to
// start sets the start time, so blocks that start late exit at the same
// deadline. micro writes a pattern that names its launch.
const jitterSource = `
__device__ __forceinline__ unsigned long long globaltimer() {
	unsigned long long t;
	asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
	return t;
}

extern "C" __global__ void spin(unsigned long long* start, unsigned long long ns, unsigned int* done) {
	__shared__ unsigned long long deadline;
	if (threadIdx.x == 0) {
		atomicCAS(start, 0ull, globaltimer());
		deadline = *(volatile unsigned long long*)start + ns;
	}
	__syncthreads();
	while (globaltimer() < deadline) {
	}
	__syncthreads();
	if (threadIdx.x == 0) {
		atomicAdd(done, 1u);
	}
}

extern "C" __global__ void micro(unsigned int* out, unsigned int n, unsigned int iter) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		out[i] = i * 3u + iter;
	}
}
`

const (
	jitterSpin     = 100 * time.Millisecond
	jitterLaunches = 200
	jitterBlock    = 256
	jitterMicroN   = 1024
)

// lane is a context, a stream in it, and the jitter kernels loaded in it.
type lane struct {
	ctx    driver.Context
	stream driver.Stream
	mod    driver.Module
	spin   driver.Function
	micro  driver.Function
}

func loadLane(t *testing.T, ctx driver.Context, stream driver.Stream, ptx string) *lane {
	t.Helper()
	if err := ctx.SetCurrent(); err != nil {
		t.Fatal(err)
	}
	mod, err := driver.ModuleLoadData([]byte(ptx))
	if err != nil {
		t.Fatalf("load module: %v", err)
	}
	t.Cleanup(func() {
		ctx.SetCurrent()
		mod.Close()
	})
	l := &lane{ctx: ctx, stream: stream, mod: mod}
	if l.spin, err = mod.Function("spin"); err != nil {
		t.Fatal(err)
	}
	if l.micro, err = mod.Function("micro"); err != nil {
		t.Fatal(err)
	}
	return l
}

func launch(t *testing.T, f driver.Function, grid, block uint32, s driver.Stream, args ...any) {
	t.Helper()
	ptrs := make([]unsafe.Pointer, len(args))
	for i, a := range args {
		switch a := a.(type) {
		case driver.DevicePtr:
			ptrs[i] = unsafe.Pointer(&a)
		case uint64:
			ptrs[i] = unsafe.Pointer(&a)
		case uint32:
			ptrs[i] = unsafe.Pointer(&a)
		default:
			t.Fatalf("argument %d: unsupported type %T", i, a)
		}
	}
	if err := f.LaunchKernel(driver.Dim3{X: grid, Y: 1, Z: 1}, driver.Dim3{X: block, Y: 1, Z: 1}, 0, s, ptrs); err != nil {
		t.Fatalf("launch: %v", err)
	}
}

func event(t *testing.T) driver.Event {
	t.Helper()
	e, err := driver.NewEvent(driver.EventDefault)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { e.Close() })
	return e
}

func alloc(t *testing.T, bytes int) driver.DevicePtr {
	t.Helper()
	p, err := driver.MemAllocBytes(uint(bytes))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { p.Free() })
	if err := p.MemsetD8(0, uint(bytes)); err != nil {
		t.Fatal(err)
	}
	return p
}

// measureJitter runs spinGrid blocks of the spin kernel on a's stream and,
// while it runs, times jitterLaunches micro launches on b's stream, one at a
// time, with events. It checks both kernels' outputs and returns the micro
// latencies in milliseconds and the spin kernel's duration.
func measureJitter(t *testing.T, a, b *lane, spinGrid int) ([]float64, float64) {
	t.Helper()
	if err := a.ctx.SetCurrent(); err != nil {
		t.Fatal(err)
	}
	start, done := alloc(t, 8), alloc(t, 4)
	s0, s1 := event(t), event(t)
	if err := s0.Record(a.stream); err != nil {
		t.Fatal(err)
	}
	launch(t, a.spin, uint32(spinGrid), jitterBlock, a.stream, start, uint64(jitterSpin.Nanoseconds()), done)
	if err := s1.Record(a.stream); err != nil {
		t.Fatal(err)
	}
	time.Sleep(5 * time.Millisecond) // let the spin kernel reach the device

	if err := b.ctx.SetCurrent(); err != nil {
		t.Fatal(err)
	}
	out := alloc(t, 4*jitterMicroN)
	e0, e1 := event(t), event(t)
	lat := make([]float64, jitterLaunches)
	for i := range lat {
		if err := e0.Record(b.stream); err != nil {
			t.Fatal(err)
		}
		launch(t, b.micro, (jitterMicroN+jitterBlock-1)/jitterBlock, jitterBlock, b.stream, out, uint32(jitterMicroN), uint32(i))
		if err := e1.Record(b.stream); err != nil {
			t.Fatal(err)
		}
		if err := e1.Synchronize(); err != nil {
			t.Fatalf("micro launch %d: %v", i, err)
		}
		ms, err := e0.ElapsedTime(e1)
		if err != nil {
			t.Fatal(err)
		}
		lat[i] = float64(ms)
	}
	got := make([]uint32, jitterMicroN)
	if err := out.CopyToHost(got); err != nil {
		t.Fatal(err)
	}
	for i, v := range got {
		if want := uint32(i)*3 + jitterLaunches - 1; v != want {
			t.Fatalf("micro out[%d] = %d, want %d", i, v, want)
		}
	}

	if err := a.ctx.SetCurrent(); err != nil {
		t.Fatal(err)
	}
	if err := a.stream.Synchronize(); err != nil {
		t.Fatalf("spin: %v", err)
	}
	spinMs, err := s0.ElapsedTime(s1)
	if err != nil {
		t.Fatal(err)
	}
	var blocks [1]uint32
	if err := done.CopyToHost(blocks[:]); err != nil {
		t.Fatal(err)
	}
	if int(blocks[0]) != spinGrid {
		t.Fatalf("spin: %d of %d blocks finished", blocks[0], spinGrid)
	}
	return lat, float64(spinMs)
}

func logJitter(t *testing.T, name string, lat []float64, spinMs float64) {
	t.Helper()
	s := slices.Sorted(slices.Values(lat))
	p50 := s[len(s)/2]
	p99 := s[int(math.Ceil(0.99*float64(len(s))))-1]
	t.Logf("%s: spin %.1f ms; micro p50 %.4f ms, p99 %.4f ms, max %.4f ms, p99/p50 %.2f", name, spinMs, p50, p99, s[len(s)-1], p99/p50)
}

// TestNativeJitter runs a 100 ms spin kernel that fills the device's thread
// slots and times 200 small kernels launched beside it, first with the
// two on the prefill and decode green contexts of the prefill-heavy plan,
// then on two streams of the primary context. It checks both kernels'
// outputs and logs the latency percentiles; it asserts nothing about them.
// It runs only with CUDA_GREEN_NATIVE=1 and an inherited GPU lock.
func TestNativeJitter(t *testing.T) {
	dev, arch, total, g := nativeDevice(t)
	major, minor := attr(t, dev, driver.DevAttrComputeCapabilityMajor), attr(t, dev, driver.DevAttrComputeCapabilityMinor)
	cm, err := nvrtcx.Compile(jitterSource, nvrtcx.WithName("jitter.cu"), nvrtcx.WithArch(major, minor))
	if err != nil {
		t.Fatalf("nvrtc: %v", err)
	}
	ptx := cm.PTX()
	// Enough blocks to fill every thread slot of every SM.
	spinGrid := total * attr(t, dev, driver.DevAttrMaxThreadsPerMultiprocessor) / jitterBlock
	t.Logf("spin grid %d x %d", spinGrid, jitterBlock)

	t.Run("green", func(t *testing.T) {
		plan, err := ComputePlan(arch, total, g, PrefillHeavy())
		if err != nil {
			t.Fatal(err)
		}
		p, err := Open(dev, plan)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { p.Close() })
		var lanes [2]*lane
		for i := range lanes {
			ctx, err := driver.CtxFromGreenCtx(p.GreenCtx(i))
			if err != nil {
				t.Fatal(err)
			}
			lanes[i] = loadLane(t, ctx, p.Stream(i), ptx)
		}
		lat, spinMs := measureJitter(t, lanes[0], lanes[1], spinGrid)
		logJitter(t, fmt.Sprintf("green (%v %d SMs, %v %d SMs)", plan.Partitions[0].Spec.Role, plan.Partitions[0].SMs, plan.Partitions[1].Spec.Role, plan.Partitions[1].SMs), lat, spinMs)
	})

	t.Run("shared", func(t *testing.T) {
		ctx, err := dev.PrimaryCtxRetain()
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { dev.PrimaryCtxRelease() })
		if err := ctx.SetCurrent(); err != nil {
			t.Fatal(err)
		}
		var lanes [2]*lane
		for i := range lanes {
			s, err := driver.NewStream(driver.StreamNonBlocking)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() {
				ctx.SetCurrent()
				s.Close()
			})
			lanes[i] = loadLane(t, ctx, s, ptx)
		}
		lat, spinMs := measureJitter(t, lanes[0], lanes[1], spinGrid)
		logJitter(t, fmt.Sprintf("shared primary context (%d SMs)", total), lat, spinMs)
	})
}
