//go:build linux

package cuptibench_test

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"runtime"
	"strings"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/cublas"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cuptibench"
	"github.com/tmc/lib/nvidia/x/cuptirange"
)

// TestPhysicalCUBLASHCProjection isolates the DS4 HC GEMM shape from its other
// kernels and HMM model mapping. Select one case per fresh process using
// CUPTIBENCH_CUBLAS_CASE=direct-user,direct-kernel,graph-whole,or graph-kernel.
// Inputs are device-allocated binary16 ones; each of 24 outputs must be 16384.
func TestPhysicalCUBLASHCProjection(t *testing.T) {
	selected := os.Getenv("CUPTIBENCH_CUBLAS_CASE")
	if selected == "" {
		t.Skip("set CUPTIBENCH_CUBLAS_CASE; run each case in a fresh process")
	}
	switch selected {
	case "direct-user", "direct-kernel", "graph-whole", "graph-kernel":
	default:
		t.Fatalf("unknown CUPTIBENCH_CUBLAS_CASE %q", selected)
	}
	p := newPhysicalPipeline(t, 1)
	const rows, k = 24, 16384
	weights, err := cudamem.Alloc[uint16](p.dev, rows*k)
	physicalCheck(t, err)
	physicalClose(t, weights.Free)
	input, err := cudamem.Alloc[uint16](p.dev, k)
	physicalCheck(t, err)
	physicalClose(t, input.Free)
	output, err := cudamem.Alloc[float32](p.dev, rows)
	physicalCheck(t, err)
	physicalClose(t, output.Free)
	ones := make([]uint16, rows*k)
	for i := range ones {
		ones[i] = 0x3c00
	}
	physicalCheck(t, weights.CopyFromHost(ones))
	physicalCheck(t, input.CopyFromHost(ones[:k]))
	var handle cublas.Handle
	physicalCheck(t, p.dev.WithContext(func() error { var err error; handle, err = cublas.Create(); return err }))
	physicalClose(t, func() error { return p.dev.WithContext(handle.Destroy) })
	physicalCheck(t, p.dev.WithContext(func() error {
		version, err := handle.GetVersion()
		if err != nil {
			return err
		}
		t.Logf("CUPTIBENCH_CUBLAS version=%d case=%s m=%d n=1 k=%d transA=T transB=N input=R16F output=R32F compute=Compute32F algorithm=GemmDefault", version, selected, rows, k)
		if err := handle.SetMathMode(cublas.TF32TensorOpMath); err != nil {
			return err
		}
		return handle.SetStream(cublas.Stream(p.streams[0].Raw()))
	}))
	gemm := func() error {
		return p.dev.WithContext(func() error {
			alpha, beta := float32(1), float32(0)
			var pin runtime.Pinner
			pin.Pin(&alpha)
			pin.Pin(&beta)
			defer pin.Unpin()
			err := handle.GemmEx(cublas.OpT, cublas.OpN, rows, 1, k,
				uintptr(unsafe.Pointer(&alpha)), uintptr(weights.Ptr()), cublas.R16F, k,
				uintptr(input.Ptr()), cublas.R16F, k, uintptr(unsafe.Pointer(&beta)),
				uintptr(output.Ptr()), cublas.R32F, rows, cublas.Compute32F, cublas.GemmDefault)
			runtime.KeepAlive(alpha)
			runtime.KeepAlive(beta)
			return err
		})
	}
	run := gemm
	var graphs []*cudagraph.Exec
	var expectedSymbols []string
	if strings.HasPrefix(selected, "graph-") {
		var captureErr error
		graph, err := cudagraph.Capture(p.streams[0], func() { captureErr = gemm() })
		if graph != nil {
			physicalClose(t, graph.Destroy)
		}
		physicalCheck(t, errors.Join(err, captureErr))
		nodes, err := graph.Raw().NodesBounded(65536)
		physicalCheck(t, err)
		for i, node := range nodes {
			kind, err := node.GetType()
			physicalCheck(t, err)
			t.Logf("CUPTIBENCH_CUBLAS_GRAPH node=%d handle=%#x kind=%s", i, uintptr(node), kind)
		}
		executable, err := graph.InstantiateWithOptions(cudagraph.InstantiateOptions{RetainKernelOrder: true})
		if executable != nil {
			physicalClose(t, executable.Destroy)
		}
		physicalCheck(t, err)
		raw, names, err := executable.KernelOrder()
		physicalCheck(t, err)
		expectedSymbols = names
		t.Logf("CUPTIBENCH_CUBLAS_GRAPH exec=%#x kernels=%d symbols=%q", uintptr(raw), len(names), names)
		physicalCheck(t, graph.Destroy())
		graphs = []*cudagraph.Exec{executable}
		run = func() error { return executable.Launch(p.streams[0]) }
	}
	// Cleanup executes before graph, handle or device allocations are released.
	physicalClose(t, p.synchronize)
	checkOutput := func(label string) error {
		if err := p.synchronize(); err != nil {
			return err
		}
		got := make([]float32, rows)
		if err := output.CopyToHost(got); err != nil {
			return err
		}
		t.Logf("CUPTIBENCH_CUBLAS_OUTPUT label=%s bits=%x", label, cublasOutputBits(got))
		for i, v := range got {
			if v != k {
				return fmt.Errorf("%s output[%d]=%g want %d", label, i, v, k)
			}
		}
		return nil
	}
	physicalCheck(t, run())
	physicalCheck(t, checkOutput("warmup"))
	mode, replay := cuptibench.Kernels, cuptirange.UserReplay
	if selected == "graph-whole" {
		mode = cuptibench.Whole
	}
	if strings.HasSuffix(selected, "-kernel") {
		replay = cuptirange.KernelReplay
	}
	plan, err := cuptibench.New(p.dev, cuptibench.Config{Mode: mode, ReplayMode: replay, Metrics: []string{physicalMetric}, CachePolicy: cuptirange.CachePreserve, Operations: 1, MaxDepth: 1, MaxConfigBytes: 16 << 20, Limits: cuptirange.Limits{MaxRanges: 16384, MaxRangeTreeNodes: 32768, MaxCounterBytes: 64 << 20, MaxPasses: 64}})
	if plan != nil {
		physicalClose(t, plan.Close)
	}
	physicalCheck(t, err)
	w := cuptibench.Workload{Streams: p.streams[:1], Graphs: graphs, View: "isolated", Preparation: "device binary16 ones; reset output to -1; HC projection shape", Synchronize: p.synchronize, Prepare: func() error {
		values := make([]float32, rows)
		for i := range values {
			values[i] = -1
		}
		return output.CopyFromHost(values)
	}, Run: func(*cuptibench.Scope) error { return run() }}
	result, collectErr := plan.Collect(w)
	receipt, err := json.Marshal(result)
	physicalCheck(t, err)
	t.Logf("CUPTIBENCH_CUBLAS_RESULT case=%s error=%v result=%s", selected, collectErr, receipt)
	outputErr := checkOutput("collected")
	physicalCheck(t, errors.Join(collectErr, outputErr))
	if result == nil || len(result.Ranges) == 0 || !result.Quality.Decoded || !result.Quality.AllPassesSubmitted || result.Quality.DroppedRanges != 0 {
		t.Fatalf("incomplete native result: %+v", result)
	}
	if mode == cuptibench.Whole && len(result.Ranges) != 1 {
		t.Fatalf("whole ranges=%d want 1", len(result.Ranges))
	}
	if selected == "graph-kernel" && len(result.Ranges) != len(expectedSymbols) {
		t.Fatalf("native kernel ranges=%d graph kernels=%d", len(result.Ranges), len(expectedSymbols))
	}
	var total float64
	for _, r := range result.Ranges {
		v := physicalValue(t, r)
		if v <= 0 || math.Trunc(v) != v {
			t.Fatalf("invalid native CTA count %g range=%s", v, r.Name)
		}
		total += v
	}
	t.Logf("CUPTIBENCH_CUBLAS_CTA case=%s ranges=%d observed_total=%g", selected, len(result.Ranges), total)
}

func cublasOutputBits(values []float32) []uint32 {
	bits := make([]uint32, len(values))
	for i, v := range values {
		bits[i] = math.Float32bits(v)
	}
	return bits
}
