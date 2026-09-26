//go:build linux

package cudabench_test

import (
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudabench"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

const evictSource = `
extern "C" __global__ void stream_read(const unsigned *in, unsigned *out, unsigned n) {
  unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned stride = blockDim.x * gridDim.x;
  unsigned acc = 0;
  for (; i < n; i += stride) acc += in[i];
  if (threadIdx.x == 0) out[blockIdx.x] = acc;
}
`

func evictDevice(tb testing.TB) *cuda.Device {
	tb.Helper()
	if os.Getenv("CUDABENCH_PHYSICAL_TEST") != "1" {
		tb.Skip("unqualified: set CUDABENCH_PHYSICAL_TEST=1 on an exclusively owned CUDA device")
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		tb.Fatal(err)
	}
	tb.Cleanup(func() { dev.Close() })
	return dev
}

// TestPhysicalEvictorChangesTheMeasurement is the positive control for the
// cache-control hook: a working set chosen to fit in L2 must measure slower,
// and at lower GB/s, once each iteration is preceded by an eviction. A hook
// that leaves the number unchanged is not controlling anything, and would let a
// caller believe they had taken a cold measurement when they had not.
//
// It reports both numbers rather than asserting a ratio: the size of the effect
// depends on the device, and the claim under test is only that the two differ
// in the direction the mechanism predicts.
func TestPhysicalEvictorChangesTheMeasurement(t *testing.T) {
	dev := evictDevice(t)
	l2, err := dev.Attribute(driver.DevAttrL2CacheSize)
	if err != nil {
		t.Fatal(err)
	}
	// A quarter of L2, so the set is comfortably resident when nothing evicts it.
	words := l2 / 4 / 4
	t.Logf("L2=%d bytes, working set=%d bytes (%d uint32)", l2, words*4, words)

	cm, err := nvrtcx.Compile(evictSource, nvrtcx.WithDeviceArch(dev), nvrtcx.WithNoCache())
	if err != nil {
		t.Fatal(err)
	}
	module, err := cm.Load(dev)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { dev.WithContext(module.Close) })
	var read driver.Function
	if err := dev.WithContext(func() error {
		var e error
		read, e = module.Function("stream_read")
		return e
	}); err != nil {
		t.Fatal(err)
	}

	run := func(evict bool) (nsPerOp, gbPerSec float64) {
		res := testing.Benchmark(func(b *testing.B) {
			cudabench.Run(b, dev, func(cb *cudabench.B) {
				in, err := cudamem.Alloc[uint32](dev, words)
				if err != nil {
					b.Fatal(err)
				}
				defer in.Free()
				out, err := cudamem.Alloc[uint32](dev, 1024)
				if err != nil {
					b.Fatal(err)
				}
				defer out.Free()
				if evict {
					free, err := cudabench.EvictL2(cb, 2)
					if err != nil {
						b.Fatal(err)
					}
					defer free()
				}
				cb.SetBytes(int64(words) * 4)
				cb.Time(func(s *cuda.Stream) {
					cudalaunch.Kernel(read).Grid(256, 1, 1).Block(256, 1, 1).On(s).
						Launch(in, out, uint32(words))
				})
			})
		})
		return res.Extra["gpu-ns/op"], res.Extra["GB/s"]
	}

	warmNs, warmGB := run(false)
	coldNs, coldGB := run(true)
	t.Logf("uncontrolled cache: %.0f gpu-ns/op, %.1f GB/s", warmNs, warmGB)
	t.Logf("evicted per iteration: %.0f gpu-ns/op, %.1f GB/s", coldNs, coldGB)

	if warmNs == 0 || coldNs == 0 {
		t.Fatal("a run reported no device time; the measurement did not resolve")
	}
	if coldNs <= warmNs {
		t.Errorf("eviction did not slow a cache-resident working set: warm %.0f ns/op, evicted %.0f ns/op; "+
			"the hook is not controlling cache state", warmNs, coldNs)
	}
	if coldGB >= warmGB {
		t.Errorf("eviction did not lower reported throughput: warm %.1f GB/s, evicted %.1f GB/s", warmGB, coldGB)
	}
}
