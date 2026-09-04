package cudasample_test

import (
	"math"
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudasample"
	"github.com/tmc/lib/nvidia/x/cudatest"
)

func TestArgmaxCUDAEdgeParity(t *testing.T) {
	if os.Getenv("LIB_CUDA_HARDWARE_TEST") == "" {
		t.Skip("LIB_CUDA_HARDWARE_TEST is empty")
	}
	dev := cudatest.Require(t)
	defer dev.Close()
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	compiled, err := cudasample.Compile(major, minor)
	if err != nil {
		t.Fatal(err)
	}
	mod, err := compiled.Load(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer mod.Unload()
	fn, err := mod.GetFunction("ArgmaxLogits")
	if err != nil {
		t.Fatal(err)
	}
	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer stream.Destroy()
	wideTie := make([]float32, 259)
	wideTie[1], wideTie[258] = 5, 5
	tests := []struct {
		name   string
		logits []float32
	}{
		{"leading NaN", []float32{float32(math.NaN()), 4, 3}},
		{"large negative", []float32{-2e20, -3e20}},
		{"all NaN", []float32{float32(math.NaN()), float32(math.NaN())}},
		{"all negative infinity", []float32{float32(math.Inf(-1)), float32(math.Inf(-1))}},
		{"adjacent tie", []float32{5, 5, 3}},
		{"cross-warp tie", wideTie},
		{"positive infinity", []float32{1, float32(math.Inf(1)), 2}},
	}
	if err := dev.WithContext(func() error {
		for _, tt := range tests {
			logits := tt.logits
			in, err := cudamem.Alloc[float32](dev, len(logits))
			if err != nil {
				return err
			}
			out, err := cudamem.Alloc[int32](dev, 1)
			if err != nil {
				_ = in.Free()
				return err
			}
			if err := in.CopyFromHost(logits); err != nil {
				return err
			}
			if err := cudalaunch.Kernel(fn).On(stream).Grid(1).Block(256).Launch(in, out, int32(len(logits))); err != nil {
				return err
			}
			got := []int32{0}
			if err := out.CopyToHost(got); err != nil {
				return err
			}
			want := cudasample.SampleArgmaxCPU(logits)
			_ = in.Free()
			_ = out.Free()
			if got[0] != want {
				t.Fatalf("%s: GPU=%d CPU=%d", tt.name, got[0], want)
			}
		}
		return nil
	}); err != nil {
		t.Fatal(err)
	}
}
