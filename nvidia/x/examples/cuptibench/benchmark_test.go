package cuptibenchexample

import (
	_ "embed"
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cuptibench"
	"github.com/tmc/lib/nvidia/x/cuptirange"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

//go:embed kernels.cu
var source string

func TestMain(m *testing.M) {
	cuptibench.Init()
	code := m.Run() // Benchmark cleanup completes before profile publication.
	if code == 0 {
		if err := cuptibench.Finish(); err != nil {
			fmt.Fprintln(os.Stderr, err)
			code = 1
		}
	}
	os.Exit(code)
}

func check(b *testing.B, err error) {
	b.Helper()
	if err != nil {
		b.Fatal(err)
	}
}

// Each leaf owns its resources, so selecting an isolated stage never needs a
// sibling benchmark to run first. The full leaf measures both stages together.
func BenchmarkPipeline(b *testing.B) {
	for _, view := range []string{"full", "isolated/rmsnorm", "isolated/project", "isolated/silu"} {
		b.Run(view, func(b *testing.B) { benchmarkPipeline(b, view) })
	}
}

func benchmarkPipeline(b *testing.B, view string) {
	// Cleanup runs in reverse dependency order. Stop if a resource still owns
	// native state; freeing its dependencies after failed cleanup is unsafe.
	var closers []func() error
	cleanup := func(close func() error) { closers = append(closers, close) }
	b.Cleanup(func() {
		for i := len(closers) - 1; i >= 0; i-- {
			if err := closers[i](); err != nil {
				b.Error(err)
				return
			}
		}
	})
	if os.Getenv("CUPTIBENCH_EXAMPLE") != "1" {
		b.Skip("set CUPTIBENCH_EXAMPLE=1 on an exclusively owned CUDA device")
	}
	dev, err := cuda.NewDevice(0)
	check(b, err)
	cleanup(dev.Close)
	stream, err := dev.NewStreamWithFlags(uint32(driver.StreamNonBlocking))
	check(b, err)
	cleanup(stream.Destroy)
	const width, rows = 1024, 256
	input, err := cudamem.Alloc[float32](dev, width)
	check(b, err)
	cleanup(input.Free)
	weights, err := cudamem.Alloc[float32](dev, width*rows)
	check(b, err)
	cleanup(weights.Free)
	norm, err := cudamem.Alloc[float32](dev, width)
	check(b, err)
	cleanup(norm.Free)
	projection, err := cudamem.Alloc[float32](dev, rows)
	check(b, err)
	cleanup(projection.Free)
	output, err := cudamem.Alloc[float32](dev, rows)
	check(b, err)
	cleanup(output.Free)
	hostInput, hostWeights, wantNorm, wantProjection, wantOutput := reference(width, rows)
	check(b, input.CopyFromHost(hostInput))
	check(b, weights.CopyFromHost(hostWeights))

	compiled, err := nvrtcx.Compile(source, nvrtcx.WithDeviceArch(dev))
	check(b, err)
	module, err := compiled.Load(dev)
	check(b, err)
	cleanup(func() error { return dev.WithContext(module.Unload) })
	var kernels struct {
		Norm    driver.Function `cuda:"rmsnorm"`
		Project driver.Function `cuda:"project"`
		SiLU    driver.Function `cuda:"silu"`
	}
	check(b, dev.WithContext(func() error { return cudalaunch.Bind(module, &kernels) }))
	stages := []struct {
		name string
		run  func() error
	}{
		{"rmsnorm", func() error {
			return cudalaunch.Kernel(kernels.Norm).Grid(1).Block(256).On(stream).Launch(input, norm, int32(width))
		}},
		{"project", func() error {
			return cudalaunch.Kernel(kernels.Project).Grid(rows).Block(128).On(stream).Launch(weights, norm, projection, int32(width))
		}},
		{"silu", func() error {
			return cudalaunch.Kernel(kernels.SiLU).Grid(1).Block(rows).On(stream).Launch(projection, output)
		}},
	}
	selected := -1
	for i, stage := range stages {
		if view == "isolated/"+stage.name {
			selected = i
		}
	}
	zeroNorm, zeroOutput := make([]float32, width), make([]float32, rows)
	workload := cuptibench.Workload{
		View:        "full",
		Streams:     []*cuda.Stream{stream},
		Synchronize: func() error { return stream.Synchronize(nil) },
		Preparation: "immutable input and dense weights; clear scratch; isolated stages run their GPU prefix before timing and profiling; caches uncontrolled",
		Prepare: func() error {
			if err := norm.CopyFromHost(zeroNorm); err != nil {
				return err
			}
			if err := projection.CopyFromHost(zeroOutput); err != nil {
				return err
			}
			if err := output.CopyFromHost(zeroOutput); err != nil {
				return err
			}
			for i := 0; i < selected; i++ {
				if err := stages[i].run(); err != nil {
					return err
				}
			}
			return nil
		},
		Run: func(s *cuptibench.Scope) error {
			for i, stage := range stages {
				if selected < 0 || selected == i {
					if err := s.Region(stage.name, stage.run); err != nil {
						return err
					}
				}
			}
			return nil
		},
	}
	if selected >= 0 {
		workload.View = "isolated"
	}
	metrics, err := cuptibench.Resolve("launches") // Counts thread blocks, not kernel calls.
	check(b, err)
	plan, err := cuptibench.New(dev, cuptibench.Config{
		Mode: cuptibench.Regions, Metrics: metrics, Operations: 1,
		CachePolicy: cuptirange.CachePreserve, MaxDepth: 2, MaxConfigBytes: 1 << 20,
		Limits: cuptirange.Limits{MaxRanges: 8, MaxRangeTreeNodes: 8, MaxCounterBytes: 8 << 20, MaxPasses: 8},
	})
	if plan != nil {
		cleanup(plan.Close)
	}
	check(b, err)

	// Warm the selected work once before timing, including any isolated prefix.
	check(b, workload.Prepare())
	for i, stage := range stages {
		if selected < 0 || selected == i {
			check(b, stage.run())
		}
	}
	check(b, workload.Synchronize())

	// Benchmark owns b.Loop. Timing and counter replay are separate executions.
	cuptibench.Benchmark(b, plan, workload)

	// Validate every executed stage, including the isolated prefix, before
	// TestMain may publish a profile. GPU reductions need a numeric tolerance.
	checks := []struct {
		name   string
		device cudamem.Slice[float32]
		want   []float32
	}{
		{"rmsnorm", norm, wantNorm}, {"project", projection, wantProjection}, {"silu", output, wantOutput},
	}
	for i, item := range checks {
		if selected >= 0 && i > selected {
			break
		}
		got := make([]float32, len(item.want))
		check(b, item.device.CopyToHost(got))
		for j, value := range got {
			want := float64(item.want[j])
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || math.Abs(float64(value)-want) > 2e-5+2e-4*math.Abs(want) {
				b.Fatalf("%s element %d: got %g, want %g", item.name, j, value, want)
			}
		}
	}
}
