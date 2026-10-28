package autotune_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudasass"
	"github.com/tmc/lib/nvidia/x/egraph/autotune"
	"github.com/tmc/lib/nvidia/x/internal/cudatool"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

const autotuneNativeEnv = "CUDA_EGRAPH_AUTOTUNE_NATIVE"

type autotuneNativeGateClosedError struct{}

func (autotuneNativeGateClosedError) Error() string {
	return "autotune: native gate is closed: CUDA_EGRAPH_AUTOTUNE_NATIVE=1 unset, no device examined"
}

func (autotuneNativeGateClosedError) Unwrap() error { return cudaerrors.ErrHardwarePending }

func checkAutotuneNativeGate(getenv func(string) string) error {
	if getenv(autotuneNativeEnv) != "1" {
		return autotuneNativeGateClosedError{}
	}
	if _, err := cudalock.Inherited(getenv); err != nil {
		return fmt.Errorf("autotune: native gate open without inherited gpu lock: %w", err)
	}
	return nil
}

const gemmEntry = "autotune_gemm"

// gemmSource is a shared-memory tiled SGEMM. A 16×16 thread block computes
// an AUTOTUNE_TILE_M square tile, each thread a (TILE_M/16)² register tile,
// staging AUTOTUNE_TILE_K-deep slices of A and B through
// AUTOTUNE_TMA_STAGES shared buffers. M, N and K must be multiples of the
// tile sizes.
const gemmSource = `
#define T 16
#define TM AUTOTUNE_TILE_M
#define TK AUTOTUNE_TILE_K
#define ST AUTOTUNE_TMA_STAGES
#define R (TM / T)
extern "C" __global__ void __launch_bounds__(T * T, MIN_BLOCKS)
autotune_gemm(const float *__restrict__ A, const float *__restrict__ B, float *__restrict__ C, int M, int N, int K) {
    __shared__ float As[ST][TK][TM];
    __shared__ float Bs[ST][TK][TM];
    const int tx = threadIdx.x, ty = threadIdx.y, tid = ty * T + tx;
    const int bm = blockIdx.y * TM, bn = blockIdx.x * TM;
    float acc[R][R];
#pragma unroll
    for (int i = 0; i < R; i++)
#pragma unroll
        for (int j = 0; j < R; j++) acc[i][j] = 0.0f;
    const int nk = K / TK;
    for (int idx = tid; idx < TK * TM; idx += T * T) {
        int k = idx / TM, m = idx % TM;
        As[0][k][m] = A[(bm + m) * K + k];
        Bs[0][k][m] = B[k * N + bn + m];
    }
    __syncthreads();
    for (int kt = 0; kt < nk; kt++) {
        const int buf = kt % ST;
        if (kt + 1 < nk) {
            const int nb = (kt + 1) % ST, k0 = (kt + 1) * TK;
            for (int idx = tid; idx < TK * TM; idx += T * T) {
                int k = idx / TM, m = idx % TM;
                As[nb][k][m] = A[(bm + m) * K + k0 + k];
                Bs[nb][k][m] = B[(k0 + k) * N + bn + m];
            }
        }
#pragma unroll
        for (int k = 0; k < TK; k++) {
            float a[R], b[R];
#pragma unroll
            for (int i = 0; i < R; i++) a[i] = As[buf][k][ty + i * T];
#pragma unroll
            for (int j = 0; j < R; j++) b[j] = Bs[buf][k][tx + j * T];
#pragma unroll
            for (int i = 0; i < R; i++)
#pragma unroll
                for (int j = 0; j < R; j++) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < R; i++)
#pragma unroll
        for (int j = 0; j < R; j++) C[(bm + ty + i * T) * N + bn + tx + j * T] = acc[i][j];
}
`

// spillCandidate caps registers at 32 per thread with an 8×8 register
// tile, which must spill.
const spillCandidate = "t128_k8_s2_spill"

func gemmSourceFor(c autotune.ScheduleCandidate) (string, error) {
	if c.TileM%16 != 0 || c.TileN != c.TileM || c.TileK <= 0 {
		return "", fmt.Errorf("gemm: tile %dx%dx%d, want square multiple of 16", c.TileM, c.TileN, c.TileK)
	}
	minBlocks := 1
	if c.Name == spillCandidate {
		minBlocks = 8
	}
	return fmt.Sprintf("#define MIN_BLOCKS %d\n", minBlocks) + gemmSource, nil
}

func gemmCandidates() []autotune.ScheduleCandidate {
	c := func(tm, tk, st int, cost float64) autotune.ScheduleCandidate {
		return autotune.ScheduleCandidate{
			Name: fmt.Sprintf("t%d_k%d_s%d", tm, tk, st), TileM: tm, TileN: tm, TileK: tk,
			Stages: st, CTAGroup: 1, AnalyticalCostUS: cost,
		}
	}
	spill := c(128, 8, 2, 10.5)
	spill.Name = spillCandidate
	return []autotune.ScheduleCandidate{
		c(32, 8, 2, 10), spill,
		c(16, 8, 2, 12), c(16, 16, 3, 12.5),
		c(32, 16, 2, 11), c(32, 16, 3, 11.5),
		c(64, 8, 2, 13), c(64, 16, 2, 13.5), c(64, 16, 3, 14),
	}
}

// knobCheckSource calls the preamble's setmaxnreg helpers when present.
const knobCheckSource = `
extern "C" __global__ void knob_check(int *out) {
#if defined(AUTOTUNE_PRODUCER_MAXNREG) && defined(AUTOTUNE_CONSUMER_MAXNREG)
    if (threadIdx.x < 128) autotune_producer_regs(); else autotune_consumer_regs();
#endif
    out[threadIdx.x] = AUTOTUNE_TMA_STAGES * 10 + AUTOTUNE_CTA_GROUP;
}
`

func TestAutotuneNativeClosedGateControl(t *testing.T) {
	if os.Getenv(autotuneNativeEnv) == "1" {
		t.Skip("closed-gate control applies when CUDA_EGRAPH_AUTOTUNE_NATIVE != 1")
	}
	if err := checkAutotuneNativeGate(os.Getenv); !errors.Is(err, cudaerrors.ErrHardwarePending) {
		t.Fatalf("checkAutotuneNativeGate(closed) = %v, want ErrHardwarePending", err)
	}
	presence := func(k string) string {
		switch k {
		case autotuneNativeEnv, "GPU_LOCK_HELD":
			return "1"
		}
		return ""
	}
	if err := checkAutotuneNativeGate(presence); err == nil || errors.Is(err, cudaerrors.ErrHardwarePending) {
		t.Fatalf("checkAutotuneNativeGate(GPU_LOCK_HELD=1) = %v, want lock error", err)
	}
	// Every candidate's source is well formed for GB10 without compiling.
	gb10, _ := autotune.TwinForTarget("sm_121a")
	for _, c := range gemmCandidates() {
		pre, err := autotune.KnobPreamble(gb10, c)
		if err != nil {
			t.Fatalf("%s: %v", c.Name, err)
		}
		src, err := gemmSourceFor(c)
		if err != nil {
			t.Fatalf("%s: %v", c.Name, err)
		}
		for _, want := range []string{fmt.Sprintf("#define AUTOTUNE_TILE_M %d\n", c.TileM), fmt.Sprintf("#define AUTOTUNE_TILE_K %d\n", c.TileK)} {
			if !strings.Contains(pre, want) {
				t.Errorf("%s: preamble missing %q", c.Name, want)
			}
		}
		if wantSpill := c.Name == spillCandidate; strings.Contains(src, "MIN_BLOCKS 8") != wantSpill {
			t.Errorf("%s: MIN_BLOCKS 8 present = %v, want %v", c.Name, !wantSpill, wantSpill)
		}
	}
}

// TestAutotuneNativeQualification tunes the tiled SGEMM above on the GPU:
// NVRTC builds each candidate for the device, nvdisasm counts spills in
// the real cubin, CUDA events time it and CUPTI PC sampling collects its
// stalls. It asserts that the register-capped candidate is rejected for
// spilling, that some stall histogram is non-empty, and that the chosen
// candidate's output matches a float64 host reference. It does not assert
// a speedup. It also compiles KnobPreamble output with NVRTC for sm_121a
// and sm_90a; sm_90a is compiled only.
func TestAutotuneNativeQualification(t *testing.T) {
	if os.Getenv(autotuneNativeEnv) != "1" {
		t.Skip("native qualification pending: CUDA_EGRAPH_AUTOTUNE_NATIVE=1 required")
	}
	if err := checkAutotuneNativeGate(os.Getenv); err != nil {
		t.Fatalf("%v: run under gpu-lock", err)
	}
	nvdisasm, err := cudatool.LookTool("nvdisasm", "")
	if err != nil {
		t.Fatalf("nvdisasm: %v", err)
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := dev.Close(); err != nil {
			t.Error(err)
		}
	})
	// Flush any attached profiler before the Go process exits.
	t.Cleanup(func() {
		if err := dev.WithContext(driver.ProfilerStop); err != nil {
			t.Errorf("ProfilerStop: %v", err)
		}
	})
	major, minor, err := dev.ComputeCapability()
	if err != nil {
		t.Fatal(err)
	}
	twin, err := autotune.AutoDiscoverTarget(major, minor, false, false)
	if err != nil {
		t.Fatal(err)
	}
	target := os.Getenv("CUDA_EGRAPH_AUTOTUNE_NATIVE_ARCH")
	if target == "" {
		target = twin.Target
	}

	t.Run("knob_preamble_nvrtc", func(t *testing.T) {
		tests := []struct {
			target string
			c      autotune.ScheduleCandidate
			ptx    []string
		}{
			{"sm_121a", autotune.ScheduleCandidate{Name: "gb10", Stages: 3, CTAGroup: 1, TileM: 64, TileN: 64, TileK: 16}, nil},
			{"sm_90a", autotune.ScheduleCandidate{Name: "h100", Stages: 4, CTAGroup: 1, ProducerMaxRegs: 40, ConsumerMaxRegs: 232},
				[]string{"setmaxnreg.dec.sync.aligned.u32 40", "setmaxnreg.inc.sync.aligned.u32 232"}},
		}
		for _, tt := range tests {
			tw, err := autotune.TwinForTarget(tt.target)
			if err != nil {
				t.Fatal(err)
			}
			pre, err := autotune.KnobPreamble(tw, tt.c)
			if err != nil {
				t.Fatal(err)
			}
			cm, err := nvrtcx.Compile(pre+knobCheckSource, nvrtcx.WithName("knob_"+tt.target+".cu"),
				nvrtcx.WithPhysicalArchTarget(tt.target), nvrtcx.WithNoCache())
			if err != nil {
				t.Fatalf("%s: NVRTC: %v", tt.target, err)
			}
			if len(cm.CUBIN()) == 0 {
				t.Errorf("%s: no cubin", tt.target)
			}
			for _, want := range tt.ptx {
				if !strings.Contains(cm.PTX(), want) {
					t.Errorf("%s: PTX missing %q", tt.target, want)
				}
			}
		}
	})

	const m, n, k = 256, 256, 256
	rng := rand.New(rand.NewPCG(1, 2))
	ha, hb := make([]float32, m*k), make([]float32, k*n)
	for i := range ha {
		ha[i] = rng.Float32()*2 - 1
	}
	for i := range hb {
		hb[i] = rng.Float32()*2 - 1
	}
	want := make([]float64, m*n)
	for i := range m {
		for j := range n {
			var s float64
			for l := range k {
				s += float64(ha[i*k+l]) * float64(hb[l*n+j])
			}
			want[i*n+j] = s
		}
	}
	a, err := cudamem.Alloc[float32](dev, m*k)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Free()
	b, err := cudamem.Alloc[float32](dev, k*n)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Free()
	c, err := cudamem.Alloc[float32](dev, m*n)
	if err != nil {
		t.Fatal(err)
	}
	defer c.Free()
	if err := a.CopyFromHost(ha); err != nil {
		t.Fatal(err)
	}
	if err := b.CopyFromHost(hb); err != nil {
		t.Fatal(err)
	}

	col := autotune.CUPTICollector{
		Device: dev,
		Run: func(fn driver.Function, cand autotune.ScheduleCandidate) error {
			tm := uint32(cand.TileM)
			return cudalaunch.Kernel(fn).Grid(n/tm, m/tm, 1).Block(16, 16, 1).
				Launch(a, b, c, int32(m), int32(n), int32(k))
		},
	}
	output := func(v autotune.Variant) ([]float32, error) {
		if err := c.CopyFromHost(make([]float32, m*n)); err != nil {
			return nil, err
		}
		if err := col.Execute(t.Context(), v); err != nil {
			return nil, err
		}
		got := make([]float32, m*n)
		return got, c.CopyToHost(got)
	}
	check := func(got []float32) (float64, error) {
		worst := 0.0
		for i, g := range got {
			d := math.Abs(float64(g) - want[i])
			worst = max(worst, d)
			if d > 1e-3 {
				return worst, fmt.Errorf("C[%d] = %g, want %g", i, g, want[i])
			}
		}
		return worst, nil
	}
	builder := autotune.NVRTCBuilder{
		Twin:         twin,
		Target:       target,
		Entry:        gemmEntry,
		Source:       gemmSourceFor,
		Disassembler: cudasass.Disassembler{Path: nvdisasm},
	}
	d, err := autotune.NewDaemon(autotune.DaemonConfig{
		Twin:      twin,
		ShapeKey:  fmt.Sprintf("sgemm_%dx%dx%d", m, n, k),
		MaxTrials: 5,
		Iters:     20,
		Builder:   builder,
		Agreement: agreeFunc(func(_ context.Context, serving, cand autotune.Variant) error {
			got, err := output(cand)
			if err != nil {
				return err
			}
			_, err = check(got)
			return err
		}),
	}, col)
	if err != nil {
		t.Fatal(err)
	}

	cands := gemmCandidates()
	var res autotune.StepResult
	var trials []autotune.Trial
	for step := range 2 {
		res, err = d.Step(t.Context(), cands)
		if err != nil {
			t.Fatalf("step %d: %v", step, err)
		}
		trials = append(trials, res.Trials...)
	}
	sampled := false
	spillRejected := false
	for _, tr := range trials {
		t.Logf("%-18s median=%8.2fus regs=%3d samples=%6d mbar=%5d lsb=%5d mio=%5d math=%5d err=%v",
			tr.Candidate.Name, tr.Observation.MedianUS, tr.Observation.Metrics.RegisterCount,
			tr.Stalls.TotalSamples, tr.Stalls.WaitMBarrier, tr.Stalls.LongScoreboard,
			tr.Stalls.MIOThrottle, tr.Stalls.MathThrottle, tr.Err)
		if tr.Stalls.TotalSamples > 0 {
			sampled = true
		}
		if tr.Candidate.Name == spillCandidate {
			spillRejected = errors.Is(tr.Err, autotune.ErrSpillDisqualified)
		}
	}
	t.Logf("winner %s %.2fus, baseline %.2fus, speedup %.2f, serving %s, weights %+v",
		res.Winner.Name, res.WinnerUS, res.BaselineUS, res.Speedup, res.Serving.Name, res.Weights)
	if !spillRejected {
		t.Errorf("%s was not rejected with ErrSpillDisqualified", spillCandidate)
	}
	if res.Winner.Name == spillCandidate || res.Serving.Name == spillCandidate {
		t.Errorf("spilling candidate selected")
	}
	if !sampled {
		t.Errorf("no candidate produced a non-empty PC-sampling histogram")
	}
	v, err := builder.Build(t.Context(), res.Winner)
	if err != nil {
		t.Fatal(err)
	}
	got, err := output(v)
	if err != nil {
		t.Fatal(err)
	}
	worst, err := check(got)
	if err != nil {
		t.Fatalf("winner %s output: %v", res.Winner.Name, err)
	}
	t.Logf("winner %s output max abs error %.3g vs float64 reference (arch %s)", res.Winner.Name, worst, target)
}

type agreeFunc func(ctx context.Context, serving, cand autotune.Variant) error

func (f agreeFunc) Agree(ctx context.Context, serving, cand autotune.Variant) error {
	return f(ctx, serving, cand)
}
