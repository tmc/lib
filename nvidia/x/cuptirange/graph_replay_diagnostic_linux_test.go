//go:build linux

package cuptirange

import (
	contextlib "context"
	"encoding/json"
	"errors"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/internal/cuptiowner"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// This diagnostic deliberately bypasses graph replay admission only in this
// test. It does not grant production support. CUDA 13.2 release notes 1.2.1.1
// explicitly exclude AutoRange+UserReplay graphs for both Range Profiling and
// legacy Profiling APIs. Run kernel and user in separate processes because a
// native rejection may prevent normal profiler cleanup.
// https://docs.nvidia.com/cupti/13.2.0/release-notes/release-notes.html#range-profiling-api-and-profiling-api
func TestNativeGraphReplayDiagnostic(t *testing.T) {
	mode := os.Getenv("CUPTI_GRAPH_REPLAY_DIAGNOSTIC")
	if mode == "" {
		t.Skip("diagnostic only: CUPTI_GRAPH_REPLAY_DIAGNOSTIC=kernel or user required")
	}
	replayMode := UserReplay
	switch mode {
	case "kernel":
		replayMode = KernelReplay
	case "user":
	default:
		t.Fatal("unknown diagnostic mode")
	}
	chip := os.Getenv("CUPTI_CHIP")
	if chip == "" {
		t.Fatal("CUPTI_CHIP must identify the measured device")
	}
	check := func(err error) {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
	}
	cleanup := func(fn func() error) {
		t.Cleanup(func() {
			if err := fn(); err != nil {
				t.Error(err)
			}
		})
	}
	dev, err := cuda.NewDevice(0)
	check(err)
	cleanup(dev.Close)
	stream, err := dev.NewStreamWithFlags(uint32(driver.StreamNonBlocking))
	check(err)
	cleanup(stream.Destroy)
	data, err := cudamem.Alloc[uint32](dev, 160)
	check(err)
	cleanup(data.Free)
	compiled, err := nvrtcx.Compile(`extern "C" __global__ void graph_first(unsigned* out){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;out[i]=i+7;}
extern "C" __global__ void graph_second(unsigned* out){unsigned i=blockIdx.x*blockDim.x+threadIdx.x;out[i]++;}`, nvrtcx.WithDeviceArch(dev), nvrtcx.WithNoCache())
	check(err)
	module, err := compiled.Load(dev)
	check(err)
	cleanup(func() error { return dev.WithContext(module.Unload) })
	var first, second driver.Function
	check(dev.WithContext(func() error {
		var err error
		first, err = module.GetFunction("graph_first")
		if err != nil {
			return err
		}
		second, err = module.GetFunction("graph_second")
		return err
	}))
	var captured error
	graph, err := cudagraph.Capture(stream, func() {
		captured = cudalaunch.Kernel(first).Grid(5).Block(32).On(stream).Launch(data)
		if captured == nil {
			captured = cudalaunch.Kernel(second).Grid(2).Block(32).On(stream).Launch(data)
		}
	})
	check(err)
	cleanup(graph.Destroy)
	check(captured)
	executable, err := graph.InstantiateWithOptions(cudagraph.InstantiateOptions{RetainKernelOrder: true})
	check(err)
	cleanup(executable.Destroy)
	check(graph.Destroy())
	availability, err := CounterAvailability(dev, 16<<20)
	check(err)
	metrics := []string{"sm__ctas_launched.sum"}
	plan, err := NewPlan(chip, availability, metrics, 16<<20)
	check(err)
	cleanup(plan.Close)
	config, passes := plan.Configuration()
	if passes != 1 {
		t.Fatalf("diagnostic needs actual single-pass CTA config, got %d", passes)
	}
	context, err := dev.PrimaryContext()
	check(err)
	cleanup(context.Release)
	lease, err := cuptiowner.AcquireDevice(uint32(dev.Ordinal()))
	check(err)
	var store *counterStore
	var names *kernelNames
	// Keep the owner until native cleanup succeeds, even for this unsupported probe.
	cleanup(func() error {
		err := context.Run(func() error {
			var err error
			if names != nil {
				err = names.Close()
			}
			if store != nil {
				err = errors.Join(err, store.close())
			}
			return err
		})
		if err == nil {
			lease.Release()
		}
		return err
	})
	var records []RangeInfo
	prepares, runs := 0, 0
	err = context.Run(func() error {
		var err error
		store, err = newCounterStore(cupti.Context(context.Raw()), metrics, 8, 16, 16<<20, counterAPI{cupti.RangeProfilerEnable, cupti.RangeProfilerDisable, cupti.RangeProfilerGetCounterDataSize, cupti.RangeProfilerCounterDataImageInitialize})
		if err != nil {
			return err
		}
		names, err = newKernelNames(cupti.Context(context.Raw()), 8)
		if err != nil {
			return err
		}
		// Test-only override: enable the proven graph-order observer while selecting
		// the requested native replay mode independently below. No production path
		// can reach this combination; count, ordinal and symbol checks still apply.
		if err := names.configureGraphs([]*cudagraph.Exec{executable}, true); err != nil {
			return err
		}
		store.kernelNames = names
		_, records, err = store.captureReplay(config, Limits{MaxRanges: 8, MaxRangeTreeNodes: 16, MaxCounterBytes: 16 << 20, MaxPasses: 8}, CaptureOptions{Mode: Kernels, MaxDepth: 1, ReplayMode: replayMode}, Replay{Prepare: func(uint) error { prepares++; return data.CopyFromHost(make([]uint32, 160)) }, Run: func(uint, *Scope) error { runs++; return executable.Launch(stream) }, Synchronize: func() error { return stream.Synchronize(contextlib.Background()) }}, true)
		return err
	})
	var quality CollectionQuality
	if store != nil {
		quality = store.quality
	}
	raw, encodeErr := json.Marshal(struct {
		Mode                string
		Version             uint32
		ConfigurationPasses uint
		Quality             CollectionQuality
		Records             []RangeInfo
		Prepares, Runs      int
	}{mode, plan.version, passes, quality, records, prepares, runs})
	check(encodeErr)
	t.Logf("unsupported-combination diagnostic=%s native_error=%v", raw, err)
	check(err)
	if len(records) != 2 || prepares != int(quality.PassesStarted) || runs != prepares {
		t.Fatalf("incomplete native graph evidence: %s", raw)
	}
	for i, r := range records {
		symbol, want, ordinal := "graph_first", float64(5), "0"
		if i == 1 {
			symbol, want, ordinal = "graph_second", 2, "1"
		}
		values, err := plan.Evaluate(store.image, r.Index)
		check(err)
		t.Logf("native graph ordinal=%s symbol=%s values=%v", r.NativeName, r.Name, values)
		if !strings.Contains(r.Name, symbol) || r.NativeName != ordinal || len(values) != 1 || values[0] != want {
			t.Fatalf("graph mapping %v values=%v", r, values)
		}
	}
	check(stream.Synchronize(contextlib.Background()))
	output := make([]uint32, 160)
	check(data.CopyToHost(output))
	for i, v := range output {
		want := uint32(i + 7)
		if i < 64 {
			want++
		}
		if v != want {
			t.Fatalf("output[%d]=%d want=%d", i, v, want)
		}
	}
	t.Log("diagnostic counters and output agree; this is not production support for an undocumented combination")
}
