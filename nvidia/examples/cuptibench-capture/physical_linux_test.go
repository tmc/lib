package cuptibenchcapture_test

import (
	"context"
	"crypto/sha256"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"
	cuptibenchcapture "github.com/tmc/lib/nvidia/examples/cuptibench-capture"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudacapture"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/cuptibench"
	"github.com/tmc/lib/nvidia/x/cuptirange"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// TestMain registers -cuptiprofile and -cuptimetrics before flag parsing, which
// is the only point at which they can be defined. BenchmarkComposeFromDevice
// calls Finish itself, so this does not publish.
func TestMain(m *testing.M) {
	cuptibench.Init()
	os.Exit(m.Run())
}

// extern "C" keeps the entry name the profile and the disassembly agree on.
const deviceSource = `extern "C" __global__ void compute(float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = out[i] * 1.5f + 1.0f;
}
`

// BenchmarkComposeFromDevice runs the whole path on hardware: compile, load,
// disassemble, benchmark with counter collection, export, and compose. It needs
// an exclusively owned CUDA device, CUPTI counter permission, NVRTC and
// nvdisasm, and it skips without them. A skip is not a pass: the composition is
// unverified against a real cuptibench export until this runs on a device.
//
// This is a benchmark rather than a test because the export records b.Name() as
// the workload identity, and cuptibench refuses to export a result whose
// benchmark name is empty. testing.Benchmark supplies an unnamed B, so driving
// cuptibench.Benchmark from a test fails that check and discards the reason.
//
// cuptibench.Benchmark owns b.Loop, which calls this body exactly once at any
// -benchtime, so the export and the composition below run once. TestMain
// registers the flags; this sets the destination and calls Finish itself
// because it consumes the exported files in the same run. Finish is process
// wide and runs once, so no other benchmark in this package may export.
func BenchmarkComposeFromDevice(b *testing.B) {
	dev := cudatest.Require(b)
	b.Cleanup(func() {
		if err := dev.Close(); err != nil {
			b.Errorf("close device: %v", err)
		}
	})
	cudatest.RequireNVRTC(b)
	nvdisasm := cudasass.Disassembler{TempDir: b.TempDir()}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	if _, err := nvdisasm.Version(ctx); err != nil {
		b.Skipf("nvdisasm unavailable: %v", err)
	}

	compiled, err := nvrtcx.Compile(deviceSource,
		nvrtcx.WithName("kernels.cu"),
		nvrtcx.WithDeviceArch(dev),
		nvrtcx.WithFlag("--generate-line-info"),
		nvrtcx.WithNoCache())
	if err != nil {
		b.Fatalf("compile: %v", err)
	}
	cubin := compiled.CUBIN()
	if len(cubin) == 0 || compiled.PTX() == "" {
		b.Fatal("compiler produced no PTX or CUBIN")
	}
	assembly, err := nvdisasm.Disassemble(ctx, cubin, cudasass.Options{PrintCode: true, InlineLines: true, PTXLines: true})
	if err != nil {
		b.Fatalf("disassemble: %v", err)
	}

	// The digest is taken from the bytes handed to the loader, so the capture
	// records the image that ran rather than a recompilation of the source.
	image := fmt.Sprintf("%x", sha256.Sum256(cubin))
	module, err := compiled.LoadOutput(dev, nvrtcx.OutputCUBIN)
	if err != nil {
		b.Fatalf("load cubin: %v", err)
	}
	b.Cleanup(func() {
		if err := dev.WithContext(module.Unload); err != nil {
			b.Errorf("unload module: %v", err)
		}
	})
	var kernels struct {
		Compute driver.Function `cuda:"compute"`
	}
	if err := dev.WithContext(func() error { return cudalaunch.Bind(module, &kernels) }); err != nil {
		b.Fatalf("bind kernel: %v", err)
	}

	const n = 1024
	stream, err := dev.NewStreamWithFlags(uint32(driver.StreamNonBlocking))
	if err != nil {
		b.Fatalf("stream: %v", err)
	}
	b.Cleanup(func() {
		if err := stream.Destroy(); err != nil {
			b.Errorf("destroy stream: %v", err)
		}
	})
	out, err := cudamem.Alloc[float32](dev, n)
	if err != nil {
		b.Fatalf("allocate: %v", err)
	}
	b.Cleanup(func() {
		if err := out.Free(); err != nil {
			b.Errorf("free: %v", err)
		}
	})
	zero := make([]float32, n)
	workload := cuptibench.Workload{
		View:        "full",
		Streams:     []*cuda.Stream{stream},
		Synchronize: func() error { return stream.Synchronize(nil) },
		Preparation: "scratch buffer cleared before each sequence; caches uncontrolled",
		Prepare:     func() error { return out.CopyFromHost(zero) },
		Run: func(s *cuptibench.Scope) error {
			return s.Region("compute", func() error {
				return cudalaunch.Kernel(kernels.Compute).Grid(n/128).Block(128).On(stream).Launch(out, int32(n))
			})
		},
	}
	metrics, err := cuptibench.Resolve(cuptibench.DefaultSet)
	if err != nil {
		b.Fatalf("resolve metrics: %v", err)
	}
	plan, err := cuptibench.New(dev, cuptibench.Config{
		Mode: cuptibench.Regions, Metrics: metrics, Operations: 1,
		CachePolicy: cuptirange.CachePreserve, MaxDepth: 2, MaxConfigBytes: 1 << 20,
		Limits: cuptirange.Limits{MaxRanges: 8, MaxRangeTreeNodes: 8, MaxCounterBytes: 8 << 20, MaxPasses: 8},
	})
	if plan == nil && err != nil {
		b.Fatalf("plan: %v", err)
	}
	if err != nil {
		b.Errorf("plan: %v", err)
	}
	// Close is idempotent, so the explicit close before the export and this
	// cleanup do not conflict when the test fails in between.
	b.Cleanup(func() {
		if err := plan.Close(); err != nil {
			b.Errorf("close plan: %v", err)
		}
	})

	dir := b.TempDir()
	// Honor an explicit -cuptiprofile so the exported profile can be kept;
	// otherwise export into the temp dir, which is discarded with the run.
	profile := flag.CommandLine.Lookup("cuptiprofile").Value.String()
	if profile == "" {
		profile = filepath.Join(dir, "device.pprof")
		if err := flag.CommandLine.Set("cuptiprofile", profile); err != nil {
			b.Fatalf("set export destination: %v", err)
		}
	}
	cuptibench.Benchmark(b, plan, workload)
	// Finish refuses results from a plan that did not close cleanly, so the
	// plan closes before the export rather than in a cleanup function.
	if err := plan.Close(); err != nil {
		b.Fatalf("close plan: %v", err)
	}
	if err := cuptibench.Finish(); err != nil {
		b.Fatalf("export profile: %v", err)
	}

	profileBytes, err := os.ReadFile(profile)
	if err != nil {
		b.Fatalf("read exported profile: %v", err)
	}
	resultBytes, err := os.ReadFile(profile + ".json")
	if err != nil {
		b.Fatalf("read exported results: %v", err)
	}
	c, err := cuptibenchcapture.Compose(
		cuptibenchcapture.Export{Name: profile, Profile: profileBytes, Results: resultBytes},
		cuptibenchcapture.Artifacts{
			Prefix:      "kernels",
			Source:      []byte(deviceSource),
			PTX:         []byte(compiled.PTX()),
			CUBIN:       cubin,
			SASS:        assembly.Stdout,
			ImageSHA256: image,
			Name:        "compute",
			Generation:  1,
			Kernels:     []string{"compute"},
		})
	if err != nil {
		b.Fatalf("Compose: %v", err)
	}

	// Publish unpacked and read it back the way cmd/cudaprof opens a directory.
	unpacked := filepath.Join(dir, "capture")
	if err := cudacapture.WriteDir(unpacked, c); err != nil {
		b.Fatalf("WriteDir: %v", err)
	}
	read, err := cudacapture.ReadDir(unpacked)
	if err != nil {
		b.Fatalf("ReadDir: %v", err)
	}
	code, err := cudacapture.VerifyModule(read, "sha256:"+image, image, []string{"compute"})
	if err != nil {
		b.Fatalf("VerifyModule: %v", err)
	}
	k := code.Kernel("compute")
	if k == nil || len(k.Instructions) == 0 || k.SourceLine == 0 {
		b.Fatalf("kernel compute did not resolve to source and instructions: %+v", k)
	}
}
