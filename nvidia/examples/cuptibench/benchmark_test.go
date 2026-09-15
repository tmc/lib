package cuptibenchexample

import (
	_ "embed"
	"fmt"
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
	for _, view := range []string{"full", "isolated/project", "isolated/scale"} {
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
	const blocks, threads = 4, 128
	const n = blocks * threads
	input, err := cudamem.Alloc[uint32](dev, n)
	check(b, err)
	cleanup(input.Free)
	mid, err := cudamem.Alloc[uint32](dev, n)
	check(b, err)
	cleanup(mid.Free)
	output, err := cudamem.Alloc[uint32](dev, n)
	check(b, err)
	cleanup(output.Free)
	host := make([]uint32, n)
	for i := range host {
		host[i] = uint32(i)
	}
	check(b, input.CopyFromHost(host))

	compiled, err := nvrtcx.Compile(source, nvrtcx.WithDeviceArch(dev))
	check(b, err)
	module, err := compiled.Load(dev)
	check(b, err)
	cleanup(func() error { return dev.WithContext(module.Unload) })
	var project, scale driver.Function
	check(b, dev.WithContext(func() error {
		var err error
		project, err = module.GetFunction("project")
		if err != nil {
			return err
		}
		scale, err = module.GetFunction("scale")
		return err
	}))
	projectStage := func() error {
		return cudalaunch.Kernel(project).Grid(blocks).Block(threads).On(stream).Launch(input, mid)
	}
	scaleStage := func() error {
		return cudalaunch.Kernel(scale).Grid(blocks).Block(threads).On(stream).Launch(mid, output)
	}
	zero := make([]uint32, n)
	workload := cuptibench.Workload{
		View:        "full",
		Streams:     []*cuda.Stream{stream},
		Synchronize: func() error { return stream.Synchronize(nil) },
		Preparation: "immutable input; clear intermediate and output before each sequence; isolated scale prepares its project prefix; caches uncontrolled",
		Prepare: func() error {
			if err := mid.CopyFromHost(zero); err != nil {
				return err
			}
			if err := output.CopyFromHost(zero); err != nil {
				return err
			}
			if view == "isolated/scale" {
				return projectStage()
			}
			return nil
		},
		Run: func(s *cuptibench.Scope) error {
			if view != "isolated/scale" {
				if err := s.Region("project", projectStage); err != nil {
					return err
				}
			}
			if view != "isolated/project" {
				return s.Region("scale", scaleStage)
			}
			return nil
		},
	}
	if view != "full" {
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

	// Benchmark owns b.Loop. Timing and counter replay are separate executions.
	cuptibench.Benchmark(b, plan, workload)

	// Validate the final replay before TestMain may publish a profile.
	got := make([]uint32, n)
	if view == "isolated/project" {
		check(b, mid.CopyToHost(got))
	} else {
		check(b, output.CopyToHost(got))
	}
	for i, v := range got {
		want := uint32(i) + 3
		if view != "isolated/project" {
			want *= 2
		}
		if v != want {
			b.Fatalf("element %d: got %d, want %d", i, v, want)
		}
	}
}
