package main

import (
	"context"
	"fmt"
	"math"
	"math/rand"
	"os"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudafuse"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudalineinfo"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// fastMath recompiles the first kernel with -use_fast_math and reports, per
// Go line, which instructions the flag removed and added. It then runs both
// builds on the same inputs and reports how far the fast result drifts from
// the IEEE one, so the instruction diff and its numerical cost sit next to
// each other.
func fastMath(ctx context.Context, opts cudalineinfo.Options) error {
	k := kernels[0]
	base, fk, err := attribute(ctx, k, opts)
	if err != nil {
		return err
	}
	opts.NVRTC = append(opts.NVRTC, nvrtcx.WithFastMath())
	fast, _, err := attribute(ctx, k, opts)
	if err != nil {
		return err
	}

	fmt.Println()
	if err := cudalineinfo.WriteDiff(os.Stdout, cudalineinfo.Side{Attribution: base, Label: "default"}, cudalineinfo.Side{Attribution: fast, Label: "-use_fast_math"}); err != nil {
		return err
	}

	maxErr, scale, err := compare(fk, base, fast)
	if err != nil {
		return fmt.Errorf("%s: run both builds: %w", k.name, err)
	}
	fmt.Printf("  max |fast - default| over %d elements: %.3g (max |default| %.3g)\n", shape[0], maxErr, scale)
	return nil
}

// compare runs two builds of the same fused kernel on identical inputs and
// returns the largest absolute difference between their outputs, with the
// largest magnitude the first build produced for scale.
func compare(fk cudafuse.FusedKernel, builds ...*cudalineinfo.Attribution) (maxErr, scale float64, err error) {
	if err := cuda.Init(); err != nil {
		return 0, 0, err
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		return 0, 0, err
	}
	host := inputs(fk, shape[0])

	outs := make([][]float32, len(builds))
	for i, b := range builds {
		out, err := runOnce(dev, b, fk, host)
		if err != nil {
			return 0, 0, err
		}
		outs[i] = out
	}
	for j, v := range outs[0] {
		scale = math.Max(scale, math.Abs(float64(v)))
		for _, out := range outs[1:] {
			maxErr = math.Max(maxErr, math.Abs(float64(out[j]-v)))
		}
	}
	return maxErr, scale, nil
}

// inputs draws n normally distributed values for each input of fk, the same
// values on every call.
func inputs(fk cudafuse.FusedKernel, n int) [][]float32 {
	rng := rand.New(rand.NewSource(1))
	host := make([][]float32, len(fk.Inputs))
	for i := range host {
		host[i] = make([]float32, n)
		for j := range host[i] {
			host[i][j] = float32(rng.NormFloat64() * 4)
		}
	}
	return host
}

// runOnce launches one build over the inputs once and copies its output back.
func runOnce(dev *cuda.Device, a *cudalineinfo.Attribution, fk cudafuse.FusedKernel, host [][]float32) ([]float32, error) {
	r, err := newRun(dev, a, fk, host)
	if err != nil {
		return nil, err
	}
	defer r.close()
	if err := r.launch(); err != nil {
		return nil, err
	}
	return r.output()
}

// A loadedKernel is one build of a fused kernel loaded on the device with its
// buffers, ready to be launched as many times as a measurement needs.
type loadedKernel struct {
	dev  *cuda.Device
	mod  *cuda.Module
	fn   *cuda.Kernel
	bufs []cudamem.Slice[float32]
	out  cudamem.Slice[float32]
	n    int
}

func newRun(dev *cuda.Device, a *cudalineinfo.Attribution, fk cudafuse.FusedKernel, host [][]float32) (*loadedKernel, error) {
	mod, err := a.Module.LoadModule(dev)
	if err != nil {
		return nil, fmt.Errorf("load module: %w", err)
	}
	r := &loadedKernel{dev: dev, mod: mod, n: len(host[0])}
	r.fn, err = mod.Function(fk.Name)
	if err != nil {
		r.close()
		return nil, err
	}
	for _, in := range host {
		buf, err := cudamem.Alloc[float32](dev, r.n)
		if err != nil {
			r.close()
			return nil, err
		}
		r.bufs = append(r.bufs, buf)
		if err := buf.CopyFromHost(in); err != nil {
			r.close()
			return nil, err
		}
	}
	if r.out, err = cudamem.Alloc[float32](dev, r.n); err != nil {
		r.close()
		return nil, err
	}
	r.bufs = append(r.bufs, r.out)
	return r, nil
}

// launch runs the kernel once on the default stream and waits for it.
func (r *loadedKernel) launch() error {
	var args []any
	for _, b := range r.bufs {
		args = append(args, b)
	}
	args = append(args, int32(r.n))
	stream := r.dev.DefaultStream()
	if err := cudalaunch.OwnedKernel(r.fn).Grid(uint32((r.n+255)/256), 1, 1).Block(256, 1, 1).On(stream).Launch(args...); err != nil {
		return fmt.Errorf("launch: %w", err)
	}
	return stream.Synchronize(context.Background())
}

func (r *loadedKernel) output() ([]float32, error) {
	res := make([]float32, r.n)
	if err := r.out.CopyToHost(res); err != nil {
		return nil, err
	}
	return res, nil
}

func (r *loadedKernel) close() {
	for _, b := range r.bufs {
		r.dev.WithContext(b.Free)
	}
	r.mod.Close()
}
