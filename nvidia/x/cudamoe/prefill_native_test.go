package cudamoe

import (
	"context"
	"fmt"
	"math"
	"os"
	"strconv"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/nvrtc"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudair"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudaref"
)

// Small integer halves have exact products and sums for these fixture sizes.
// Exact output comparison is a fixture property, not the general WMMA policy.
func prefillFixture(t *testing.T, k, n, round int) ([]uint16, []uint16, []float32) {
	t.Helper()
	encode := func(v int) uint16 {
		original := v
		sign := uint16(0)
		if v < 0 {
			sign = 0x8000
			v = -v
		}
		bits := sign | [...]uint16{0, 0x3c00, 0x4000, 0x4200, 0x4400, 0x4500}[v]
		var decoded float64
		if bits&0x7fff != 0 {
			decoded = math.Ldexp(float64(1024+(bits&1023)), int((bits>>10)&31)-25)
		}
		if bits&0x8000 != 0 {
			decoded = -decoded
		}
		if decoded != float64(original) {
			t.Fatalf("half encoding %04x does not represent %d", bits, original)
		}
		return bits
	}
	a, b := make([]float32, 256*k), make([]float32, k*n)
	ah, bh := make([]uint16, len(a)), make([]uint16, len(b))
	for i := range a {
		v := (i+round)%7 - 3
		a[i] = float32(v)
		ah[i] = encode(v)
	}
	for i := range b {
		v := (i*3+round)%11 - 5
		b[i] = float32(v)
		bh[i] = encode(v)
	}
	g := cudair.NewGraph()
	x := g.AddInput("a", []int{256, k}, cudair.Float32)
	y := g.AddInput("b", []int{k, n}, cudair.Float32)
	z := g.AddMatMul(x, y)
	if err := g.SetOutputs(z); err != nil {
		t.Fatal(err)
	}
	snap, err := g.Snapshot()
	if err != nil {
		t.Fatal(err)
	}
	result, err := cudaref.EvaluateSnapshot(snap, map[string][]float32{"a": a, "b": b})
	if err != nil {
		t.Fatal(err)
	}
	want := result[z.Name]
	if len(want) != 256*n {
		t.Fatal("missing reference output")
	}
	for r := 0; r < 256; r++ {
		for c := 0; c < n; c++ {
			var sum float64
			for p := 0; p < k; p++ {
				sum += float64(a[r*k+p]) * float64(b[p*n+c])
			}
			if want[r*n+c] != float32(sum) {
				t.Fatalf("reference mismatch row=%d col=%d", r, c)
			}
		}
	}
	return ah, bh, want
}

func comparePrefill(got, want []float32) error {
	if len(got) != len(want) {
		return fmt.Errorf("output length %d, want %d", len(got), len(want))
	}
	for i, v := range got {
		if math.IsNaN(float64(v)) || v != want[i] {
			return fmt.Errorf("output %d = %g, want %g", i, v, want[i])
		}
	}
	return nil
}

func TestPrefillNativeFixture(t *testing.T) {
	for round, shape := range [][2]int{{16, 16}, {32, 48}, {48, 32}} {
		a, b, want := prefillFixture(t, shape[0], shape[1], round)
		if len(a) != 256*shape[0] || len(b) != shape[0]*shape[1] {
			t.Fatal("wrong input extents")
		}
		if err := comparePrefill(want, want); err != nil {
			t.Fatal(err)
		}
		bad := append([]float32(nil), want...)
		bad[len(bad)-1]++
		if comparePrefill(bad, want) == nil {
			t.Fatal("accepted corrupted last output")
		}
		bad[0] = float32(math.NaN())
		if comparePrefill(bad, want) == nil {
			t.Fatal("accepted poisoned output")
		}
	}
}

func TestNativePrefillW256Reference(t *testing.T) {
	if os.Getenv("CUDAMOE_W256_NATIVE") != "1" {
		t.Skip("CUDAMOE_W256_NATIVE=1 requires scoped hardware authorization")
	}
	target := os.Getenv("CUDAMOE_TARGET")
	if target != "sm_100a" && target != "sm_121a" {
		t.Fatal("CUDAMOE_TARGET must name sm_100a or sm_121a")
	}
	ordinal, err := strconv.Atoi(os.Getenv("CUDAMOE_DEVICE"))
	if err != nil || ordinal < 0 {
		t.Fatal("CUDAMOE_DEVICE must name an explicit ordinal")
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	lease, err := cudalock.Acquire(ctx, cudalock.WithLabel(t.Name()))
	if err != nil {
		t.Fatal(err)
	}
	var dev *cuda.Device
	var stream *cuda.Stream
	var raw driver.Module
	var module *cuda.Module
	var halfBuffers []cudamem.Slice[uint16]
	var floatBuffers []cudamem.Slice[float32]
	t.Cleanup(func() {
		drainCtx, cancelDrain := context.WithTimeout(context.Background(), 5*time.Minute)
		defer cancelDrain()
		// Even a failed launch may have submitted work. Stop cleanup on a failed
		// drain and retain all dependent resources and the device lease.
		if stream != nil {
			if err := stream.Synchronize(drainCtx); err != nil {
				t.Errorf("retain after drain failure: %v", err)
				return
			}
		}
		if module != nil {
			if err := module.Close(); err != nil {
				t.Error(err)
				return
			}
		}
		if raw != 0 {
			if err := dev.WithContext(func() error {
				if err := raw.Close(); err != nil {
					return err
				}
				raw = 0
				return nil
			}); err != nil {
				t.Error(err)
				return
			}
		}
		if stream != nil {
			if err := stream.Close(); err != nil {
				t.Error(err)
				return
			}
		}
		for _, b := range halfBuffers {
			if err := b.Free(); err != nil {
				t.Error(err)
				return
			}
		}
		for _, b := range floatBuffers {
			if err := b.Free(); err != nil {
				t.Error(err)
				return
			}
		}
		if dev != nil {
			if err := dev.Close(); err != nil {
				t.Error(err)
				return
			}
		}
		if err := lease.Release(); err != nil {
			t.Error(err)
		}
	})
	dev, err = cuda.NewDevice(ordinal)
	if err != nil {
		t.Fatal(err)
	}
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	if fmt.Sprintf("sm_%d%da", major, minor) != target {
		t.Fatalf("device SM%d%d does not match %s", major, minor, target)
	}
	artifact, err := PrefillW256Artifact(target)
	if err != nil {
		t.Fatal(err)
	}
	raw, err = artifact.Load(dev)
	if err != nil {
		t.Fatal(err)
	}
	module, err = cuda.OwnModule(dev, raw)
	if err != nil {
		t.Fatal(err)
	}
	raw = 0
	kernel, err := module.Function(artifact.EntryPoint)
	if err != nil {
		t.Fatal(err)
	}
	stream, err = dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	driverVersion, err := driver.DriverGetVersion()
	if err != nil {
		t.Fatal(err)
	}
	rtcMajor, rtcMinor, err := nvrtc.Version()
	if err != nil {
		t.Fatal(err)
	}
	uuid, err := dev.Raw().Uuid()
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("device=%d uuid=%v target=%s driver=%d nvrtc=%d.%d image=%s resources=%+v flags=%v lease=%+v", ordinal, uuid, target, driverVersion, rtcMajor, rtcMinor, artifact.Digest, artifact.Resources, artifact.CompilerOpts, lease.Owner())
	runCtx, cancelRun := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancelRun()
	for round, shape := range [][2]int{{16, 16}, {32, 48}, {48, 32}} {
		k, n := shape[0], shape[1]
		if _, err := PrefillW256Layouts(k, n); err != nil {
			t.Fatal(err)
		}
		ah, bh, want := prefillFixture(t, k, n, round)
		a, err := cudamem.Alloc[uint16](dev, len(ah))
		halfBuffers = append(halfBuffers, a)
		if err != nil {
			t.Fatal(err)
		}
		b, err := cudamem.Alloc[uint16](dev, len(bh))
		halfBuffers = append(halfBuffers, b)
		if err != nil {
			t.Fatal(err)
		}
		storage, err := cudamem.Alloc[float32](dev, len(want)+16)
		floatBuffers = append(floatBuffers, storage)
		if err != nil {
			t.Fatal(err)
		}
		out := storage.Slice(8, len(want)+8)
		if a.Ptr()%32 != 0 || b.Ptr()%32 != 0 || out.Ptr()%32 != 0 {
			t.Fatal("unaligned native buffers")
		}
		if err := a.CopyFromHost(ah); err != nil {
			t.Fatal(err)
		}
		if err := b.CopyFromHost(bh); err != nil {
			t.Fatal(err)
		}
		readback := make([]float32, storage.Len())
		for i := range readback {
			readback[i] = float32(math.NaN())
		}
		const guardValue = float32(1234567)
		for i := range 8 {
			readback[i] = guardValue
			readback[len(readback)-1-i] = guardValue
		}
		if err := storage.CopyFromHost(readback); err != nil {
			t.Fatal(err)
		}
		args := []any{a, b, out, uint32(k), uint32(n)}
		if _, err := artifact.Schema.Pack(args...); err != nil {
			t.Fatal(err)
		}
		if err := cudalaunch.OwnedKernel(kernel).Grid(uint32(n/16), 4, 1).Block(128, 1, 1).On(stream).Launch(args...); err != nil {
			t.Fatal(err)
		}
		if err := stream.Synchronize(runCtx); err != nil {
			t.Fatal(err)
		}
		if err := storage.CopyToHost(readback); err != nil {
			t.Fatal(err)
		}
		if err := comparePrefill(readback[8:8+len(want)], want); err != nil {
			t.Fatalf("k=%d n=%d: %v", k, n, err)
		}
		for i := range 8 {
			if readback[i] != guardValue || readback[len(readback)-1-i] != guardValue {
				t.Fatal("output guard overwritten")
			}
		}
		readback[8+len(want)-1]++
		if comparePrefill(readback[8:8+len(want)], want) == nil {
			t.Fatal("accepted corrupted native readback")
		}
		t.Logf("round=%d K=%d N=%d outputs=%d reference=exact-small-integers guards=preserved", round, k, n, len(want))
	}
}
