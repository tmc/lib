package main

import (
	"context"
	"errors"
	"fmt"
	"math"
	"runtime"
	"runtime/trace"
	"strconv"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudaprof"
)

type workloadResult struct {
	Intervals map[string][]int64
	Verified  int
	MaxError  float64
}

func runWorkload(ctx context.Context, dev *cuda.Device, mod *cuda.Module, prof *cudaprof.Session, code *cudacapture.Code, cfg config) (res workloadResult, err error) {
	n := cfg.elements
	a, b, c := make([]float32, n), make([]float32, n), make([]float32, n)
	for i := range a {
		a[i], b[i] = float32(i%1024), float32(2*(i%1024)+1)
	}
	wantCompute := reference(cfg.rounds)
	da, err := cudamem.Alloc[float32](dev, n)
	if err != nil {
		return res, err
	}
	defer func() { err = errors.Join(err, da.Free()) }()
	db, err := cudamem.Alloc[float32](dev, n)
	if err != nil {
		return res, err
	}
	defer func() { err = errors.Join(err, db.Free()) }()
	dc, err := cudamem.Alloc[float32](dev, n)
	if err != nil {
		return res, err
	}
	defer func() { err = errors.Join(err, dc.Free()) }()
	if err := da.CopyFromHost(a); err != nil {
		return res, err
	}
	if err := db.CopyFromHost(b); err != nil {
		return res, err
	}
	stream, err := dev.NewStream()
	if err != nil {
		return res, err
	}
	defer func() { err = errors.Join(err, stream.Destroy()) }()
	start, err := dev.NewEvent()
	if err != nil {
		return res, err
	}
	defer func() { err = errors.Join(err, start.Destroy()) }()
	end, err := dev.NewEvent()
	if err != nil {
		return res, err
	}
	defer func() { err = errors.Join(err, end.Destroy()) }()

	res.Intervals = make(map[string][]int64)

	for pass := -1; pass < cfg.passes; pass++ {
		for _, name := range []string{"vector_add", "compute"} {
			k := code.Kernel(name)
			if k == nil {
				return res, fmt.Errorf("missing compiled kernel %s", name)
			}
			fn, err := mod.Function(name)
			if err != nil {
				return res, err
			}
			launcher := cudalaunch.OwnedKernel(fn).Grid(uint32((n + 255) / 256)).Block(256).On(stream)
			args := []any{da, db, dc, int32(n)}
			phase := "memory"
			if name == "compute" {
				args = []any{dc, int32(n), int32(cfg.rounds)}
				phase = "compute"
			}
			var ns int64
			cudaprof.Do(ctx, cudaprof.Labels("kernel", name, "phase", phase, "pass", strconv.Itoa(pass)), func(ctx context.Context) {
				ns, err = runKernel(ctx, launcher, args, stream, start, end, k, prof, pass >= 0, n, cfg.rounds)
			})
			if err != nil {
				return res, err
			}
			if err := dc.CopyToHost(c); err != nil {
				return res, err
			}
			for i, got := range c {
				want := a[i] + b[i]
				tolerance := float64(0)
				if name == "compute" {
					want = wantCompute[i%len(wantCompute)]
					tolerance = 1e-6*math.Abs(float64(want)) + 1e-6
				}
				diff := math.Abs(float64(got) - float64(want))
				if math.IsNaN(float64(got)) || math.IsInf(float64(got), 0) || diff > tolerance {
					return res, fmt.Errorf("%s pass %d: c[%d] = %g, want %g (tolerance %g)", name, pass, i, got, want, tolerance)
				}
				res.MaxError = math.Max(res.MaxError, diff)
			}
			if pass >= 0 {
				res.Intervals[name] = append(res.Intervals[name], ns)
				res.Verified += n
			}
			trace.Logf(ctx, "cuda.validation", "%s pass=%d: verified %d results", name, pass, n)
		}
	}
	return res, nil
}

func runKernel(ctx context.Context, launcher *cudalaunch.Launcher, args []any, stream *cuda.Stream, start, end *cuda.Event, k *cudacapture.Kernel, prof *cudaprof.Session, record bool, n, rounds int) (int64, error) {
	region := trace.StartRegion(ctx, k.Name+": host launch and wait")
	defer region.End()
	if err := start.Record(stream); err != nil {
		return 0, err
	}
	if err := launcher.Launch(args...); err != nil {
		return 0, err
	}
	if err := end.Record(stream); err != nil {
		return 0, err
	}
	if err := end.Synchronize(ctx); err != nil {
		return 0, err
	}
	elapsed, err := end.ElapsedSince(start)
	if err != nil {
		return 0, err
	}
	if elapsed <= 0 {
		return 0, fmt.Errorf("%s: nonpositive event interval %s", k.Name, elapsed)
	}
	trace.Logf(ctx, "cuda.event", "kernel=%s measured=%v gpu_time=%s", k.Name, record, elapsed)
	if record {
		vals := cudaprof.SampleValues{
			GPUNanos: elapsed.Nanoseconds(),
		}
		_, _, line, _ := runtime.Caller(0)
		prof.RecordStackSample(ctx, vals, []cudaprof.Frame{
			{FunctionName: k.Name + ".SASS", FileName: "kernels.sass", Line: int64(k.Instructions[0].SASSLine), Address: cudaprof.DefaultMappingStart + k.Offset},
			{FunctionName: k.Name + ".PTX", FileName: "kernels.ptx", Line: int64(k.PTXLine)},
			{FunctionName: k.Name, FileName: "kernels.cu", Line: int64(k.SourceLine)},
			{FunctionName: "main.runKernel", FileName: "workload.go", Line: int64(line)},
		})
	}
	return elapsed.Nanoseconds(), nil
}

// The inputs repeat every 1024 elements, so compute one CPU reference for each
// distinct input, using the same float32 coefficients and a fused operation.
func reference(rounds int) []float32 {
	out := make([]float32, 1024)
	for i := range out {
		v := float32(3*i + 1)
		for j := 0; j < rounds; j++ {
			v = float32(math.FMA(float64(v), float64(float32(1.000001)), float64(float32(0.0001))))
		}
		out[i] = v
	}
	return out
}
