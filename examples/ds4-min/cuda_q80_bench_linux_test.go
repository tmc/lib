//go:build linux

package main

// Q8_0 matvec cost, measured per kernel and summed to a per-token total.
//
// The Q8_0 family is the largest single row of the decode parity gap: the C
// oracle spends 30.993 ms/token on it and ds4-min spends 40.453, a difference
// of 9.459 ms/token. That figure came from an nsys capture of the whole
// workload, differenced across generation lengths, which costs a GPU lease and
// a 2048-token prefill to reproduce. This benchmark reproduces the same row
// from kernel launches alone, on synthetic weights, so a kernel change can be
// priced in seconds instead of in lease slots.
//
// It measures the three matvec kernels the decode actually issues:
//
//	ds4_q8_0_matvec_q8_1                  216 launches/token   20.000 ms/token
//	ds4_q8_0_matvec_quantized              86 launches/token   11.747 ms/token
//	ds4_q8_0_matvec_quantized_blockdiag    43 launches/token    8.311 ms/token
//
// Weights are synthetic. These kernels are data independent -- no branch reads
// a weight value -- so timing does not depend on the bytes, only on the shape
// and the traffic. What must be faithful is the shape, and the shapes below are
// derived rather than read; see cudaQ80BenchShapes.
//
// Reported per shape: microseconds per launch from CUDA events, the resulting
// GB/s against the weight bytes that launch must read, and the contribution to
// a token at the decode's launch count. BenchmarkCUDAQ80MatvecToken sums those
// contributions and compares the total against the census row, which is the
// conservation check: a sum that no longer lands near 40.453 means the shape
// table has drifted from the model, not that the kernels got faster.

import (
	"context"
	"fmt"
	"math/rand"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudamem"
)

// cudaQ80CensusTotal is the Q8_0 family cost from the decode census: positions
// 2048-2175, weights read through host page tables, decode isolated by
// differencing a 40-token run against an 8-token run.
const cudaQ80CensusTotal = 40.453

// cudaQ80BenchShape is one matvec the decode issues, at the shape it issues it.
//
// PROVENANCE, because every number here is load-bearing and none of it was read
// out of the checkpoint. The byte size of each tensor is measured -- it comes
// from the GGUF tensor inventory. Columns and rows are DERIVED from that size
// through the Q8_0 block layout, which stores each 32-element block in 34
// bytes, so bytes = rows * cols/32 * 34. Fixing cols determines rows.
//
// cols=4096 is not a guess: output.weight measures 562.63 MB, and at cols=4096
// that yields 129280 rows, which is exactly the vocabulary size. Every other
// row count derived at these column counts lands on a power of two. A wrong
// cols would have to produce eight coincidences at once.
//
// The check that matters is at the bottom: these shapes, driven at the launch
// counts below, predict the three census kernels to within 6% each, using
// per-launch times from a different instrument in a different regime. Treat
// that as corroboration of the table, not as a measurement of the kernels.
//
// TestCUDAQ80BenchShapesMatchModel validates the table against the real
// checkpoint when DS4_MODEL is set. Until that has run on a given checkpoint,
// the table is a declared premise, and it is the first thing to doubt if a
// result here disagrees with a full-workload capture.
type cudaQ80BenchShape struct {
	name     string // the GGUF tensor this stands in for
	kernel   string // which of the three matvecs consumes it
	cols     int
	rows     int // total rows; for blockdiag this is rows*diag
	diag     int // block-diagonal count, 0 for the other two kernels
	perToken int // launches per token in the decode, from the census
}

// cudaQ80BenchShapes is the decode's Q8_0 working set. The per-token launch
// counts sum to 216/86/43, matching the census exactly; 43 is the layer count,
// and the +1 on the q8_1 row is the output head, which runs once per token
// rather than once per layer.
var cudaQ80BenchShapes = []cudaQ80BenchShape{
	// ds4_q8_0_matvec_q8_1: 5 per layer + 1 output head = 216.
	{"attn_q_a", "q8_1", 4096, 1024, 0, 43},
	{"attn_kv", "q8_1", 4096, 512, 0, 43},
	{"attn_q_b", "q8_1", 1024, 32768, 0, 43},
	{"ffn_gate_shexp", "q8_1", 4096, 2048, 0, 43},
	{"ffn_up_shexp", "q8_1", 4096, 2048, 0, 43},
	{"output.weight", "q8_1", 4096, 129280, 0, 1},

	// ds4_q8_0_matvec_quantized: 2 per layer = 86.
	{"attn_out_b", "quantized", 32768, 1024, 0, 43},
	{"ffn_down_shexp", "quantized", 2048, 4096, 0, 43},

	// ds4_q8_0_matvec_quantized_blockdiag: 1 per layer = 43.
	// UNVERIFIED beyond the byte count: 35.65 MB is consistent with several
	// (cols, rows, diag) splits and only the product is pinned. diag=64 is the
	// head count. If this row disagrees with a capture, suspect it first.
	{"attn_output_a", "blockdiag", 512, 65536, 64, 43},
}

// cudaQ80BenchWeightBytes is the device footprint of one weight tensor.
func (s cudaQ80BenchShape) weightBytes() int { return s.rows * s.cols / 32 * 34 }

// activationBytes is the quantized activation the kernel reads beside the
// weights: 36 bytes per 32-element block, one block set per diagonal block.
func (s cudaQ80BenchShape) activationBytes() int {
	blocks := s.cols / 32 * 36
	if s.diag > 0 {
		return blocks * s.diag
	}
	return blocks
}

func (s cudaQ80BenchShape) outputLen() int { return s.rows }

// cudaQ80BenchFoundation binds device 0 with no model mapping, like
// cudaSynthFoundation but for a benchmark.
func cudaQ80BenchFoundation(b *testing.B) *cudaFoundation {
	b.Helper()
	dev, err := cuda.NewDevice(0)
	if err != nil {
		b.Skipf("no CUDA device: %v", err)
	}
	name, err := dev.Name()
	if err != nil {
		dev.Close()
		b.Fatal(err)
	}
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		dev.Close()
		b.Fatal(err)
	}
	r := &cudaFoundation{device: dev, name: name, major: major, minor: minor}
	b.Cleanup(func() {
		if err := r.Close(); err != nil {
			b.Error(err)
		}
	})
	return r
}

// cudaQ80BenchBuffers allocates one shape's weights, activation and output.
// Weight content is pseudorandom rather than zero: a zero weight buffer would
// still time correctly here, but it makes the buffer trivially compressible for
// any future memory path that notices, and a benchmark should not depend on
// that staying untrue.
func cudaQ80BenchBuffers(b *testing.B, r *cudaFoundation, s cudaQ80BenchShape) (cudaModelTensor, cudamem.Slice[byte], cudamem.Slice[float32]) {
	b.Helper()
	w, err := cudamem.Alloc[byte](r.device, s.weightBytes())
	if err != nil {
		b.Skipf("allocating %d weight bytes for %s: %v", s.weightBytes(), s.name, err)
	}
	b.Cleanup(func() { _ = w.Free() })
	host := make([]byte, s.weightBytes())
	rng := rand.New(rand.NewSource(1))
	rng.Read(host)
	if err := w.CopyFromHost(host); err != nil {
		b.Fatal(err)
	}
	in, err := cudamem.Alloc[byte](r.device, s.activationBytes())
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { _ = in.Free() })
	act := make([]byte, s.activationBytes())
	rng.Read(act)
	if err := in.CopyFromHost(act); err != nil {
		b.Fatal(err)
	}
	out, err := cudamem.Alloc[float32](r.device, s.outputLen())
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { _ = out.Free() })
	t := cudaModelTensor{
		name:       s.name,
		typeID:     8,
		dimensions: []uint64{uint64(s.cols), uint64(s.rows)},
		device:     w.Ptr(),
		bytes:      uint64(s.weightBytes()),
	}
	return t, in, out
}

// cudaQ80BenchLaunch returns the encode call for one shape.
func cudaQ80BenchLaunch(b *testing.B, r *cudaFoundation, s cudaQ80BenchShape, t cudaModelTensor, in cudamem.Slice[byte], out cudamem.Slice[float32]) func() error {
	b.Helper()
	switch s.kernel {
	case "q8_1":
		return func() error { return r.encodeQ80MatvecQ81(t, in, out) }
	case "quantized":
		return func() error { return r.encodeQ80MatvecQuantized(t, in, out) }
	case "blockdiag":
		return func() error { return r.encodeQ80MatvecQuantizedBlockDiag(t, in, out, s.diag) }
	}
	b.Fatalf("unknown kernel %q for %s", s.kernel, s.name)
	return nil
}

// cudaQ80BenchTime brackets n launches with device events and returns the mean
// microseconds per launch.
//
// Device events, not wall clock: this host has been measured swinging wall
// clock by 15x under load while device span held, and a benchmark that cannot
// tell those apart reports the host's mood as a kernel change.
func cudaQ80BenchTime(b *testing.B, r *cudaFoundation, n int, launch func() error) float64 {
	b.Helper()
	var elapsed float64
	if err := r.device.WithContext(func() error {
		start, err := r.device.NewEvent()
		if err != nil {
			return err
		}
		defer start.Destroy()
		end, err := r.device.NewEvent()
		if err != nil {
			return err
		}
		defer end.Destroy()
		stream := r.stream()
		if err := start.Record(stream); err != nil {
			return err
		}
		for i := 0; i < n; i++ {
			if err := launch(); err != nil {
				return err
			}
		}
		if err := end.Record(stream); err != nil {
			return err
		}
		if err := end.Synchronize(context.Background()); err != nil {
			return err
		}
		d, err := end.ElapsedSince(start)
		if err != nil {
			return err
		}
		elapsed = float64(d.Microseconds()) / float64(n)
		return nil
	}); err != nil {
		b.Fatal(err)
	}
	if elapsed <= 0 {
		b.Fatalf("device event duration %v over %d launches does not establish execution", elapsed, n)
	}
	return elapsed
}

// BenchmarkCUDAQ80Matvec prices each Q8_0 matvec the decode issues.
func BenchmarkCUDAQ80Matvec(b *testing.B) {
	r := cudaQ80BenchFoundation(b)
	for _, s := range cudaQ80BenchShapes {
		b.Run(fmt.Sprintf("%s/%s", s.kernel, s.name), func(b *testing.B) {
			t, in, out := cudaQ80BenchBuffers(b, r, s)
			launch := cudaQ80BenchLaunch(b, r, s, t, in, out)
			// Warm the module load and the first-touch page faults out of the
			// measured region: loadMathKernels compiles on first call, and a
			// first launch in a fresh context is not the launch we are pricing.
			cudaQ80BenchTime(b, r, 2, launch)
			us := cudaQ80BenchTime(b, r, b.N, launch)
			bytes := float64(s.weightBytes())
			b.ReportMetric(us, "us/launch")
			b.ReportMetric(bytes/us/1e3, "GB/s")
			b.ReportMetric(us*float64(s.perToken)/1e3, "ms/token")
		})
	}
}

// BenchmarkCUDAQ80MatvecToken sums the shapes into the per-token family cost
// and checks it against the census.
//
// The comparison is the point. A per-kernel benchmark that never adds up is how
// a ledger drifts from the workload it claims to describe without anybody
// noticing: every row stays individually defensible while the total stops
// meaning anything. If this reports a total far from cudaQ80CensusTotal, the
// first suspect is cudaQ80BenchShapes, not the kernels.
func BenchmarkCUDAQ80MatvecToken(b *testing.B) {
	r := cudaQ80BenchFoundation(b)
	perKernel := map[string]float64{}
	total := 0.0
	for _, s := range cudaQ80BenchShapes {
		t, in, out := cudaQ80BenchBuffers(b, r, s)
		launch := cudaQ80BenchLaunch(b, r, s, t, in, out)
		cudaQ80BenchTime(b, r, 2, launch)
		us := cudaQ80BenchTime(b, r, b.N, launch)
		ms := us * float64(s.perToken) / 1e3
		perKernel[s.kernel] += ms
		total += ms
	}
	for _, k := range []string{"q8_1", "quantized", "blockdiag"} {
		b.ReportMetric(perKernel[k], k+"-ms/token")
	}
	b.ReportMetric(total, "ms/token")
	b.ReportMetric(total/cudaQ80CensusTotal, "x-census")
	b.Logf("Q8_0 family %.3f ms/token against census %.3f (%.3fx); C oracle 30.993",
		total, cudaQ80CensusTotal, total/cudaQ80CensusTotal)
}
