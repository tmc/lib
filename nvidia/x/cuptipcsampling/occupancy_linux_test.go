//go:build linux

package cuptipcsampling_test

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"os"
	"runtime"
	"strings"
	"sync"
	"testing"

	"github.com/tmc/lib/nvidia/cupti"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cuptipcsampling"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// occupancyEnv gates every hardware test in this file.
const occupancyEnv = "CUPTI_PC_OCCUPANCY_TEST"

// reasonPrefix is the prefix CUPTI puts on every stall reason name. It is
// trimmed rather than suffix-matched, because "selected" is a suffix of
// "not_selected" and confusing those two would defeat the point of the file.
const reasonPrefix = "smsp__pcsamp_warps_issue_stalled_"

// Iteration counts are the smallest that gave a clear margin on GB10 while
// keeping every kernel's samples inside one session's output and name budgets.
// They are deliberately short: the sweep's assertions are about shares and
// orderings, and neither needs a long run.
const (
	sweepIters = 20000
	// chain_dep is one dependent FMA at a time on a single block, so it needs
	// the same iteration count as the sweep to produce four figures of samples;
	// at 2000 it collected 137 and wait merely tied with selected.
	chainIters = 20000
	chaseSteps = 2000
	floodIters = 20
	// One cycle over 1<<24 int32 is 64 MiB, past any cache on this class of
	// part. It need not reach DRAM: long_scoreboard covers a variable-latency
	// dependency wherever the data comes from, so the test does not depend on
	// sizing the working set exactly.
	chaseElems = 1 << 24
	floodElems = 1 << 24
)

// occupancySource holds the microkernels the tests launch. Each is written so
// that its dominant stall reason follows from its instruction mix rather than
// from what any reason happens to be named.
//
// The sweep needs one distinct symbol per block size rather than one kernel
// launched six ways, because all six launches share a single sampling session
// and the kernel symbol CUPTI attributes each PC to is what separates them. See
// the comment on occupancyHarness for why there is only one session.
const occupancySource = `
// WIDE_INDEP keeps eight independent fp32 accumulators, so a warp always has
// another FMA ready to issue and never waits on a dependency. With one warp per
// scheduler it can neither lose arbitration nor outrun the FMA pipe; both become
// possible only once several warps share a scheduler. __launch_bounds__ pins the
// block size into the symbol so each row's geometry is part of the kernel.
#define WIDE_INDEP(NAME, BLOCK)                                                \
  extern "C" __global__ void __launch_bounds__(BLOCK)                          \
      NAME(const float *__restrict__ in, float *__restrict__ out, int iters) {  \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                             \
    float a0 = in[i], a1 = a0 + 1.f, a2 = a0 + 2.f, a3 = a0 + 3.f;             \
    float a4 = a0 + 4.f, a5 = a0 + 5.f, a6 = a0 + 6.f, a7 = a0 + 7.f;          \
    for (int k = 0; k < iters; ++k) {                                          \
      _Pragma("unroll") for (int j = 0; j < 4; ++j) {                          \
        a0 = __fmaf_rn(a0, 1.0000001f, 1e-7f);                                 \
        a1 = __fmaf_rn(a1, 1.0000001f, 1e-7f);                                 \
        a2 = __fmaf_rn(a2, 1.0000001f, 1e-7f);                                 \
        a3 = __fmaf_rn(a3, 1.0000001f, 1e-7f);                                 \
        a4 = __fmaf_rn(a4, 1.0000001f, 1e-7f);                                 \
        a5 = __fmaf_rn(a5, 1.0000001f, 1e-7f);                                 \
        a6 = __fmaf_rn(a6, 1.0000001f, 1e-7f);                                 \
        a7 = __fmaf_rn(a7, 1.0000001f, 1e-7f);                                 \
      }                                                                        \
    }                                                                          \
    out[i] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;                            \
  }

WIDE_INDEP(wide_indep_b32, 32)
WIDE_INDEP(wide_indep_b64, 64)
WIDE_INDEP(wide_indep_b128, 128)
WIDE_INDEP(wide_indep_b256, 256)
WIDE_INDEP(wide_indep_b512, 512)
WIDE_INDEP(wide_indep_b1024, 1024)

// chain_dep carries one float through dependent FMAs. Every FMA needs the
// previous result, so the latency is fixed and known to the compiler, which
// covers it with stall cycles rather than with a scoreboard.
extern "C" __global__ void chain_dep(const float *__restrict__ in,
                                     float *__restrict__ out, int iters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float x = in[i];
  for (int k = 0; k < iters; ++k) {
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      x = __fmaf_rn(x, 1.0000001f, 1e-7f);
    }
  }
  out[i] = x;
}

// chase walks a permutation one element at a time. Each load's address is the
// previous load's value, so the latency is variable and unknowable to the
// compiler, and a scoreboard is the only thing that can cover it.
extern "C" __global__ void chase(const int *__restrict__ next,
                                 int *__restrict__ out, int steps, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int p = (int)(((long long)i * 2654435761LL) % (long long)n);
  for (int k = 0; k < steps; ++k) {
    p = next[p];
  }
  out[i] = p;
}

// load_flood issues sixteen independent loads per iteration at high occupancy.
// Nothing waits on a value before computing the next address, so the limit it
// reaches is the load/store queue rather than any single load's latency.
extern "C" __global__ void load_flood(const float *__restrict__ in,
                                      float *__restrict__ out, int iters,
                                      int mask) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int base = (i * 64) & mask;
  float s = 0.f;
  for (int k = 0; k < iters; ++k) {
#pragma unroll
    for (int j = 0; j < 16; ++j) {
      s += in[(base + j * 8192 + k * 131) & mask];
    }
  }
  out[i] = s;
}
`

// TestStallReasonOccupancySweep establishes that two stall reasons behave as
// their meanings require across a pure occupancy sweep. It is the only check in
// this package that is not downstream of the index->name map.
//
// Every other check compares our reason names against names CUPTI reported, so
// both sides read the same table and an agreement tests spelling rather than
// meaning. The reconciliation identity is weaker still: it is a conservation
// check, so it closes exactly even when a reason is filed in the wrong bucket.
// This test instead fixes the instruction stream and varies only the block size,
// which makes the expectation a consequence of how warp schedulers work:
//
//   - not_selected means the scheduler picked a different warp. Below two warps
//     per scheduler there is no other warp to pick, so zero is forced by
//     arbitration rather than merely expected to be small.
//   - math_pipe_throttle means a warp could not issue because the arithmetic
//     pipe was busy. A scheduler issues at most one instruction per cycle and
//     the fp32 FMA pipe accepts one per cycle, so a single warp cannot outrun it
//     and zero is forced by throughput.
//
// The forced zeros alone are not sufficient, which is worth stating because the
// obvious version of this test stops there. Both reasons are zero in the forced
// regime, so a map that swapped these two names would satisfy every equality
// below. What separates them is that above the knee they move in opposite
// directions: contention for a scheduler rises as warps are added, while the
// per-warp share of a fixed-throughput pipe falls. A swapped map predicts the
// opposite pair of slopes and fails. So the slopes are the discriminating
// assertion and the zeros are the forced one, and neither is redundant.
//
// The slopes are asserted on each reason's share of base samples, not on its raw
// count. Both counts rise with warp count simply because more warps produce more
// samples; only the share falls for a reason whose ceiling is pipe throughput.
func TestStallReasonOccupancySweep(t *testing.T) {
	h := requireOccupancyHarness(t)

	// grid is 1 on every row: a single block cannot be split across SMs, so
	// residency is definitional rather than an occupancy-API inference, and
	// below 256 threads no second warp can reach a scheduler. warpsPerSched
	// assumes four warp schedulers per SM, true of every NVIDIA part since
	// Maxwell.
	tests := []struct {
		name            string
		kernel          string
		warpsPerSched   float64
		forcedZero      bool // below two warps per scheduler
		minNotSelected  float64
		minMathThrottle float64
	}{
		{name: "quarter warp per scheduler", kernel: "wide_indep_b32", warpsPerSched: 0.25, forcedZero: true},
		{name: "half warp per scheduler", kernel: "wide_indep_b64", warpsPerSched: 0.5, forcedZero: true},
		{name: "one warp per scheduler", kernel: "wide_indep_b128", warpsPerSched: 1, forcedZero: true},
		{name: "two warps per scheduler", kernel: "wide_indep_b256", warpsPerSched: 2},
		{name: "four warps per scheduler", kernel: "wide_indep_b512", warpsPerSched: 4},
		{name: "eight warps per scheduler", kernel: "wide_indep_b1024", warpsPerSched: 8, minNotSelected: 0.05, minMathThrottle: 0.01},
	}

	type row struct {
		name                      string
		notSelected, mathThrottle float64
	}
	var knee []row

	for _, tt := range tests {
		got := h.counts(t, tt.kernel)
		ns := got.fraction("not_selected")
		mp := got.fraction("math_pipe_throttle")
		t.Logf("%s: %s warps/scheduler=%.2f base=%d not_selected=%d (%.2f%%) math_pipe_throttle=%d (%.2f%%)",
			tt.name, tt.kernel, tt.warpsPerSched, got.base,
			got.count("not_selected"), 100*ns, got.count("math_pipe_throttle"), 100*mp)

		if tt.forcedZero {
			// Exact equality on purpose. These zeros are forced by the geometry,
			// so a tolerance would admit a real regression, and an exact zero
			// cannot be explained away by a different part's issue rates.
			if n := got.count("not_selected"); n != 0 {
				t.Errorf("%s: not_selected = %d, want exactly 0: at %.2f warps/scheduler there is no second warp for a scheduler to prefer",
					tt.name, n, tt.warpsPerSched)
			}
			if n := got.count("math_pipe_throttle"); n != 0 {
				t.Errorf("%s: math_pipe_throttle = %d, want exactly 0: one warp issues at most one instruction per cycle and cannot outrun the FMA pipe",
					tt.name, n)
			}
			continue
		}

		knee = append(knee, row{tt.name, ns, mp})
		if ns <= 0 {
			t.Errorf("%s: not_selected = 0, want nonzero: %.0f warps share each scheduler", tt.name, tt.warpsPerSched)
		}
		if mp <= 0 {
			t.Errorf("%s: math_pipe_throttle = 0, want nonzero: %.0f warps share one FMA pipe", tt.name, tt.warpsPerSched)
		}
		if ns < tt.minNotSelected {
			t.Errorf("%s: not_selected share = %.4f, want >= %.2f", tt.name, ns, tt.minNotSelected)
		}
		if mp < tt.minMathThrottle {
			t.Errorf("%s: math_pipe_throttle share = %.4f, want >= %.2f", tt.name, mp, tt.minMathThrottle)
		}
	}

	// The discriminating assertion. Magnitudes above are this part's and are
	// left loose; the directions are what the two meanings require, and a map
	// that swapped these names would invert both.
	for i := 1; i < len(knee); i++ {
		prev, cur := knee[i-1], knee[i]
		if cur.notSelected <= prev.notSelected {
			t.Errorf("not_selected share did not rise from %s (%.4f) to %s (%.4f): scheduler contention must grow with warps per scheduler",
				prev.name, prev.notSelected, cur.name, cur.notSelected)
		}
		if cur.mathThrottle >= prev.mathThrottle {
			t.Errorf("math_pipe_throttle share did not fall from %s (%.4f) to %s (%.4f): a fixed-throughput pipe's share per warp must shrink as warps are added",
				prev.name, prev.mathThrottle, cur.name, cur.mathThrottle)
		}
	}
}

// TestStallReasonDominance establishes that the reason dominating each of three
// kernels is the one that kernel's construction requires, which tests the
// meaning of three more reasons without consulting the index->name map.
//
// The three separate mechanisms that are easy to conflate: dependent arithmetic
// has a fixed latency the compiler covers with stall cycles (wait); a pointer
// chase has a variable latency only a scoreboard can cover (long_scoreboard);
// and parallel loads saturate the load/store queue rather than any one load's
// latency (lg_throttle). The chase row carries the sharpest claim: it is
// memory-bound yet must NOT be dominated by any stall_mem reason, because
// stall_mem counts throttling and never counts waiting for data.
//
// Dominance is asserted as strictly greater than every other base reason rather
// than as a percentage, so the test pins the ordering and not this part's ratios.
func TestStallReasonDominance(t *testing.T) {
	h := requireOccupancyHarness(t)

	tests := []struct {
		name   string
		kernel string
		want   string
		// forbidden reasons must not dominate, because the kernel's
		// construction rules each of them out.
		forbidden []string
	}{
		{
			name:      "dependent arithmetic stalls on fixed latency",
			kernel:    "chain_dep",
			want:      "wait",
			forbidden: []string{"long_scoreboard", "lg_throttle", "math_pipe_throttle"},
		},
		{
			name:      "serial chase stalls on a scoreboard, not on throttling",
			kernel:    "chase",
			want:      "long_scoreboard",
			forbidden: []string{"lg_throttle", "mio_throttle", "tex_throttle", "wait"},
		},
		{
			name:      "parallel loads stall on the load/store queue",
			kernel:    "load_flood",
			want:      "lg_throttle",
			forbidden: []string{"wait", "short_scoreboard", "math_pipe_throttle"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := h.counts(t, tt.kernel)
			top, topN := got.dominant()
			t.Logf("%s: base=%d dominant=%s (%d, %.2f%%)", tt.kernel, got.base, top, topN, 100*got.fraction(top))
			if top != tt.want {
				t.Errorf("%s: dominant reason = %s (%d samples), want %s (%d samples)",
					tt.kernel, top, topN, tt.want, got.count(tt.want))
			}
			for _, f := range tt.forbidden {
				if f == top {
					t.Errorf("%s: %s dominates, but the kernel's construction rules it out", tt.kernel, f)
				}
			}
		})
	}
}

// reasonCounts holds one kernel's base reason samples, keyed by short name.
// Companions are excluded: a _not_issued companion counts a subset of the cycles
// its base already counted, so including both double counts the same samples.
type reasonCounts struct {
	counts map[string]int64
	base   int64
}

func (r reasonCounts) count(short string) int64 { return r.counts[short] }

func (r reasonCounts) fraction(short string) float64 {
	if r.base == 0 {
		return 0
	}
	return float64(r.count(short)) / float64(r.base)
}

// dominant returns the short name of the base reason with the most samples.
// Ties break on name so that a tie reports deterministically rather than
// following map iteration order.
func (r reasonCounts) dominant() (string, int64) {
	var top string
	var topN int64
	for name, n := range r.counts {
		if n > topN || (n == topN && name < top) {
			top, topN = name, n
		}
	}
	return top, topN
}

// occupancyHarness owns the single sampling session every test in this file
// reads from, along with its per-kernel reason counts.
//
// There is exactly one session per process on purpose. Starting a second PC
// sampling session after closing the first corrupts the native heap on CUPTI
// 2025.3.1: a later CUPTI_PCSamplingEnable aborts the process inside malloc,
// nondeterministically after the second or third session. So every kernel is
// launched inside one session and the samples are separated afterwards by the
// kernel symbol CUPTI attributed each PC to, which is why the sweep needs six
// distinct symbols rather than one kernel launched with six block sizes.
type occupancyHarness struct {
	byKernel map[string]reasonCounts
	err      error
}

var (
	harnessOnce sync.Once
	harness     *occupancyHarness
)

// requireOccupancyHarness captures every kernel once per process, or skips.
func requireOccupancyHarness(t *testing.T) *occupancyHarness {
	t.Helper()
	if os.Getenv(occupancyEnv) != "1" {
		t.Skipf("%s=1 required: needs an NVIDIA GPU with CUPTI PC sampling and NVRTC", occupancyEnv)
	}
	harnessOnce.Do(func() { harness = captureAll() })
	if harness.err != nil {
		t.Fatalf("capture: %v", harness.err)
	}
	return harness
}

// counts returns one kernel's base reason counts.
func (h *occupancyHarness) counts(t *testing.T, kernel string) reasonCounts {
	t.Helper()
	got, ok := h.byKernel[kernel]
	if !ok || got.base == 0 {
		t.Fatalf("%s: no samples attributed to this kernel", kernel)
	}
	return got
}

// launchSpec is one kernel and the geometry that makes its expectation an
// expectation rather than a hope.
type launchSpec struct {
	kernel      string
	grid, block uint32
	args        func(dev *cuda.Device, threads int) (args []any, free func(), err error)
}

func captureAll() *occupancyHarness {
	h := &occupancyHarness{byKernel: make(map[string]reasonCounts)}
	h.err = h.run()
	return h
}

func (h *occupancyHarness) run() error {
	// The CUDA context is per-thread and the sampling session is tied to it.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	dev, err := cuda.NewDevice(0)
	if err != nil {
		return fmt.Errorf("open device: %w", err)
	}
	defer dev.Close()

	major, minor, err := dev.ComputeCapability()
	if err != nil {
		return err
	}
	// A real architecture, not compute_XX: the latter emits PTX only and the
	// module load then fails with an unavailable output.
	arch := fmt.Sprintf("sm_%d%d", major, minor)

	cuCtx, err := dev.PrimaryContext()
	if err != nil {
		return fmt.Errorf("retain primary context: %w", err)
	}
	defer cuCtx.Release()
	ctxHandle := cupti.Context(cuCtx.Raw())

	// The inventory cannot be read while sampling is disabled, and the session
	// config is built from it, so this enable is scoped to the query alone.
	inv, err := readInventory(ctxHandle)
	if err != nil {
		return fmt.Errorf("query stall reasons: %w", err)
	}

	compiled, err := nvrtcx.Compile(occupancySource,
		nvrtcx.WithName("occupancy.cu"),
		nvrtcx.WithFlag("--gpu-architecture="+arch),
		nvrtcx.WithNoCache())
	if err != nil {
		return fmt.Errorf("compile kernels: %w", err)
	}

	config, err := sweepBudget(len(inv.Reasons))
	if err != nil {
		return err
	}
	for _, r := range inv.Reasons {
		config.ReasonIDs = append(config.ReasonIDs, r.Index)
	}

	session, err := cuptipcsampling.Start(ctxHandle, config)
	if err != nil {
		if session != nil {
			err = errors.Join(err, session.Close())
		}
		// A foreign PC sampling client holds the hardware exclusively, so this
		// is a busy device rather than a failing assertion.
		return fmt.Errorf("start pc sampling (another process may hold the sampling hardware): %w", err)
	}

	mod, _, err := compiled.LoadOwnedOutput(dev, nvrtcx.OutputCUBIN)
	if err != nil {
		return errors.Join(fmt.Errorf("load module: %w", err), session.Close())
	}
	defer mod.Close()

	stream, err := dev.NewStream()
	if err != nil {
		return errors.Join(err, session.Close())
	}

	ctx := context.Background()
	floats := func(dev *cuda.Device, threads int) ([]any, func(), error) {
		in, err := cudamem.Alloc[float32](dev, threads)
		if err != nil {
			return nil, nil, err
		}
		out, err := cudamem.Alloc[float32](dev, threads)
		if err != nil {
			in.Free()
			return nil, nil, err
		}
		return []any{in, out, int32(sweepIters)}, func() { in.Free(); out.Free() }, nil
	}

	specs := []launchSpec{
		{"wide_indep_b32", 1, 32, floats},
		{"wide_indep_b64", 1, 64, floats},
		{"wide_indep_b128", 1, 128, floats},
		{"wide_indep_b256", 1, 256, floats},
		{"wide_indep_b512", 1, 512, floats},
		{"wide_indep_b1024", 1, 1024, floats},
		{"chain_dep", 1, 128, func(dev *cuda.Device, threads int) ([]any, func(), error) {
			in, err := cudamem.Alloc[float32](dev, threads)
			if err != nil {
				return nil, nil, err
			}
			out, err := cudamem.Alloc[float32](dev, threads)
			if err != nil {
				in.Free()
				return nil, nil, err
			}
			return []any{in, out, int32(chainIters)}, func() { in.Free(); out.Free() }, nil
		}},
		{"chase", 1, 128, func(dev *cuda.Device, threads int) ([]any, func(), error) {
			next, err := cudamem.Alloc[int32](dev, chaseElems)
			if err != nil {
				return nil, nil, err
			}
			if err := next.CopyFromHost(shuffledRing(chaseElems)); err != nil {
				next.Free()
				return nil, nil, err
			}
			out, err := cudamem.Alloc[int32](dev, threads)
			if err != nil {
				next.Free()
				return nil, nil, err
			}
			return []any{next, out, int32(chaseSteps), int32(chaseElems)}, func() { next.Free(); out.Free() }, nil
		}},
		{"load_flood", 72, 1024, func(dev *cuda.Device, threads int) ([]any, func(), error) {
			in, err := cudamem.Alloc[float32](dev, floodElems)
			if err != nil {
				return nil, nil, err
			}
			out, err := cudamem.Alloc[float32](dev, threads)
			if err != nil {
				in.Free()
				return nil, nil, err
			}
			return []any{in, out, int32(floodIters), int32(floodElems - 1)}, func() { in.Free(); out.Free() }, nil
		}},
	}

	var workErr error
	for _, s := range specs {
		fn, err := mod.Function(s.kernel)
		if err != nil {
			workErr = errors.Join(workErr, err)
			break
		}
		args, free, err := s.args(dev, int(s.grid*s.block))
		if err != nil {
			workErr = errors.Join(workErr, err)
			break
		}
		err = cudalaunch.OwnedKernel(fn).Grid(s.grid).Block(s.block).On(stream).Launch(args...)
		if err == nil {
			err = stream.Synchronize(ctx)
		}
		free()
		if err != nil {
			workErr = errors.Join(workErr, fmt.Errorf("%s: %w", s.kernel, err))
			break
		}
		// Read after every kernel rather than once at the end: the native
		// output holds a fixed number of PCs and the high-occupancy rows fill
		// it well before the sweep is over, which CUPTI reports as an
		// out-of-memory error on the next read rather than as a partial batch.
		if err := h.drain(session, inv); err != nil {
			workErr = errors.Join(workErr, fmt.Errorf("%s: %w", s.kernel, err))
			break
		}
	}

	closeErr := session.Close()
	// Final is populated only once Close has stopped the worker and taken the
	// last native output, so it is read after Close and not before.
	if final, ok := session.Final(); ok && final.Valid {
		h.collect(final, inv)
	}
	return errors.Join(workErr, closeErr)
}

// drain reads every batch CUPTI has ready.
func (h *occupancyHarness) drain(session *cuptipcsampling.Session, inv cuptipcsampling.ReasonInventory) error {
	for {
		batch, err := session.Read()
		if err != nil {
			return err
		}
		if batch.Valid {
			h.collect(batch, inv)
		}
		if batch.RemainingPCs == 0 {
			return nil
		}
	}
}

// collect splits one batch's samples by the kernel symbol CUPTI attributed each
// PC to. Reason names come from the inventory the session was configured with,
// so a kernel's counts are assembled without consulting the stall classifier.
func (h *occupancyHarness) collect(batch cuptipcsampling.Batch, inv cuptipcsampling.ReasonInventory) {
	for _, pc := range batch.PCs {
		if pc.Name == "" {
			continue
		}
		got, ok := h.byKernel[pc.Name]
		if !ok {
			got = reasonCounts{counts: make(map[string]int64)}
		}
		for _, r := range pc.Reasons {
			name := inv.Lookup(r.Index)
			if name == "" || strings.HasSuffix(name, "_not_issued") {
				continue
			}
			got.counts[strings.TrimPrefix(name, reasonPrefix)] += int64(r.Samples)
			got.base += int64(r.Samples)
		}
		h.byKernel[pc.Name] = got
	}
}

// sweepBudget sizes the native buffers for a reason count.
//
// PCsPerBuffer must stay strictly below MaxNames/2. Every read reserves
// 2*PCsPerBuffer name entries against a session-lifetime count of distinct
// native name pointers, so at PCsPerBuffer == MaxNames/2 the reservation equals
// the whole budget and the session fails its next read as soon as it has
// retained one name. Start permits that equality, so the guard has to be here.
// Entries are keyed by pointer and CUPTI returns one stable pointer per
// function, so the headroom below covers distinct kernels, not PCs or reads.
func sweepBudget(reasons int) (cuptipcsampling.Config, error) {
	const (
		outputBytes = 1 << 20
		nameBytes   = 256
		maxNames    = 4096
	)
	stride := uint64(64) + uint64(reasons)*8
	capacity := (outputBytes/2 - 80) / stride
	if capacity < 1 {
		return cuptipcsampling.Config{}, fmt.Errorf("%d reasons exceed the output budget", reasons)
	}
	if headroom := maxNames - 2*int(capacity); headroom < 1 {
		return cuptipcsampling.Config{}, fmt.Errorf("name budget leaves %d entries after one read's reservation; raise MaxNames above %d", headroom, 2*capacity)
	}
	return cuptipcsampling.Config{
		Period:            10,
		WorkerSleepMS:     10,
		MaxReasons:        reasons,
		ScratchBytes:      8 << 20,
		HardwareBytes:     8 << 20,
		MaxNativeBytes:    64 << 20,
		PCsPerBuffer:      int(capacity),
		MaxOutputBytes:    outputBytes,
		MaxReads:          1 << 12,
		MaxNames:          maxNames,
		MaxNameBytes:      nameBytes,
		MaxTotalNameBytes: maxNames * nameBytes,
	}, nil
}

func readInventory(ctx cupti.Context) (inv cuptipcsampling.ReasonInventory, err error) {
	disable, err := cuptipcsampling.Enable(ctx)
	if err != nil {
		return cuptipcsampling.ReasonInventory{}, err
	}
	defer func() { err = errors.Join(err, disable()) }()
	return cuptipcsampling.StallReasons(ctx, 1<<10)
}

// shuffledRing returns next such that following it from any index walks all n
// elements before returning. Chaining next along a shuffled order yields one
// cycle of length n for any permutation, so no lane falls into a short cycle
// whose few lines would simply stay cached.
func shuffledRing(n int) []int32 {
	perm := make([]int32, n)
	for i := range perm {
		perm[i] = int32(i)
	}
	r := rand.New(rand.NewSource(1))
	r.Shuffle(n, func(i, j int) { perm[i], perm[j] = perm[j], perm[i] })
	next := make([]int32, n)
	for i := 0; i < n; i++ {
		next[perm[i]] = perm[(i+1)%n]
	}
	return next
}
