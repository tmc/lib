package breakout_test

import (
	"encoding/binary"
	"fmt"
	"hash/fnv"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/puffernet/breakout"
)

// A newFunc is breakout.New or breakout.NewOpt.
type newFunc = func(*cuda.Device, breakout.Config, int) (*breakout.Vec, error)

// impls are the implementations, which step identically.
var impls = []struct {
	name string
	new  newFunc
}{
	{"ref", breakout.New},
	{"opt", breakout.NewOpt},
}

// lcg is the action generator in testdata/cref.cu.
type lcg uint32

func (s *lcg) actions(a []float32) {
	for i := range a {
		*s = *s*1664525 + 1013904223
		a[i] = float32((*s >> 16) % 3)
	}
}

// buffers are the device buffers of n environments.
type buffers struct {
	arena     *cudamem.Arena
	obs       cudamem.Slice[uint16]
	actions   cudamem.Slice[float32]
	rewards   cudamem.Slice[float32]
	terminals cudamem.Slice[float32]
}

func newBuffers(t testing.TB, dev *cuda.Device, n int) *buffers {
	t.Helper()
	a := cudamem.NewArena(dev)
	t.Cleanup(func() { a.Free() })
	b := &buffers{
		arena:     a,
		obs:       cudamem.AllocIn[uint16](a, "obs", n*breakout.ObsSize),
		actions:   cudamem.AllocIn[float32](a, "actions", n),
		rewards:   cudamem.AllocIn[float32](a, "rewards", n),
		terminals: cudamem.AllocIn[float32](a, "terminals", n),
	}
	if err := a.Err(); err != nil {
		t.Fatal(err)
	}
	// Neither kernel writes observations 8 and 9, which are zero in
	// testdata/cref.cu's fresh allocation.
	if err := b.obs.MemsetAsync(0, dev.DefaultStream()); err != nil {
		t.Fatal(err)
	}
	return b
}

func newVec(t testing.TB, dev *cuda.Device, open newFunc, n int) *breakout.Vec {
	t.Helper()
	v, err := open(dev, breakout.DefaultConfig, n)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { v.Close() })
	return v
}

// hash is the hash testdata/cref.cu prints: FNV-1a of the observations,
// rewards and terminals.
func (b *buffers) hash(t *testing.T) string {
	t.Helper()
	obs := make([]uint16, b.obs.Len())
	rewards := make([]float32, b.rewards.Len())
	terminals := make([]float32, b.terminals.Len())
	for _, err := range []error{
		b.obs.CopyToHost(obs),
		b.rewards.CopyToHost(rewards),
		b.terminals.CopyToHost(terminals),
	} {
		if err != nil {
			t.Fatal(err)
		}
	}
	h := fnv.New64a()
	binary.Write(h, binary.LittleEndian, obs)
	binary.Write(h, binary.LittleEndian, rewards)
	binary.Write(h, binary.LittleEndian, terminals)
	return fmt.Sprintf("%016x", h.Sum64())
}

// TestTrajectory checks that the environments step exactly as PufferLib's
// build of breakout.cu does, from the hashes testdata/cref.cu printed with
//
//	cref trajectory 256 2000 > testdata/trajectory.txt
func TestTrajectory(t *testing.T) {
	data, err := os.ReadFile("testdata/trajectory.txt")
	if err != nil {
		t.Fatal(err)
	}
	want := strings.Fields(string(data))
	for _, impl := range impls {
		t.Run(impl.name, func(t *testing.T) {
			testTrajectory(t, impl.new, want)
		})
	}
}

func testTrajectory(t *testing.T, open newFunc, want []string) {
	const n = 256
	dev := cudatest.Require(t)
	v := newVec(t, dev, open, n)
	b := newBuffers(t, dev, n)
	stream := dev.DefaultStream()
	if err := v.Reset(stream, b.obs, b.rewards, b.terminals); err != nil {
		t.Fatal(err)
	}
	if got := b.hash(t); got != want[0] {
		t.Fatalf("reset: hash %s, want %s", got, want[0])
	}
	s := lcg(1)
	actions := make([]float32, n)
	for step, w := range want[1:] {
		s.actions(actions)
		if err := b.actions.CopyFromHost(actions); err != nil {
			t.Fatal(err)
		}
		if err := v.Step(stream, b.actions, b.obs, b.rewards, b.terminals); err != nil {
			t.Fatal(err)
		}
		if got := b.hash(t); got != w {
			t.Fatalf("step %d: hash %s, want %s", step+1, got, w)
		}
	}

	l, err := v.Log(t.Context(), stream)
	if err != nil {
		t.Fatal(err)
	}
	if l.N == 0 || l.Score == 0 {
		t.Errorf("Log() = %+v, want finished episodes that scored", l)
	}
}

func TestConfig(t *testing.T) {
	tests := []struct {
		name       string
		rows, cols int
		ok         bool
	}{
		{"default", 6, 18, true},
		{"one", 1, 1, true},
		{"none", 0, 18, false},
		{"too many", 7, 18, false},
	}
	for _, impl := range impls {
		for _, tt := range tests {
			t.Run(impl.name+"/"+tt.name, func(t *testing.T) {
				dev := cudatest.Require(t)
				cfg := breakout.DefaultConfig
				cfg.BrickRows, cfg.BrickCols = tt.rows, tt.cols
				v, err := impl.new(dev, cfg, 1)
				if (err == nil) != tt.ok {
					t.Fatalf("%d×%d bricks: err = %v, want ok %v", tt.rows, tt.cols, err, tt.ok)
				}
				if v != nil {
					v.Close()
				}
			})
		}
	}
}

// BenchmarkStep times Vec.Step as cref bench does: fixed random actions,
// 100 warm-up steps, and a synchronize after the timed steps.
func BenchmarkStep(b *testing.B) {
	for _, impl := range impls {
		for _, n := range []int{4096, 65536, 1 << 20} {
			b.Run(fmt.Sprintf("%s/%d", impl.name, n), func(b *testing.B) {
				benchmarkStep(b, impl.new, n)
			})
		}
	}
}

func benchmarkStep(b *testing.B, open newFunc, n int) {
	dev := cudatest.Require(b)
	v := newVec(b, dev, open, n)
	buf := newBuffers(b, dev, n)
	actions := make([]float32, n)
	s := lcg(1)
	s.actions(actions)
	if err := buf.actions.CopyFromHost(actions); err != nil {
		b.Fatal(err)
	}
	stream := dev.DefaultStream()
	if err := v.Reset(stream, buf.obs, buf.rewards, buf.terminals); err != nil {
		b.Fatal(err)
	}
	for range 100 {
		if err := v.Step(stream, buf.actions, buf.obs, buf.rewards, buf.terminals); err != nil {
			b.Fatal(err)
		}
	}
	ctx := b.Context()
	if err := stream.Synchronize(ctx); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for range b.N {
		if err := v.Step(stream, buf.actions, buf.obs, buf.rewards, buf.terminals); err != nil {
			b.Fatal(err)
		}
	}
	if err := stream.Synchronize(ctx); err != nil {
		b.Fatal(err)
	}
	b.StopTimer()
	b.ReportMetric(float64(n)*float64(b.N)/b.Elapsed().Seconds(), "env-steps/s")
}
