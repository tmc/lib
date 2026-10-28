package qualify

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudalock"
	"github.com/tmc/lib/nvidia/x/internal/cudatest"
	"github.com/tmc/lib/nvidia/x/internal/cudatool"
)

const qualifyNativeEnv = "CUDA_QUALIFY_NATIVE"

type qualifyNativeGateClosedError struct{}

func (qualifyNativeGateClosedError) Error() string {
	return "qualify: native gate is closed: CUDA_QUALIFY_NATIVE=1 unset, no device examined"
}

func (qualifyNativeGateClosedError) Unwrap() error { return cudaerrors.ErrHardwarePending }

func checkQualifyNativeGate(getenv func(string) string) error {
	if getenv(qualifyNativeEnv) != "1" {
		return qualifyNativeGateClosedError{}
	}
	if _, err := cudalock.Inherited(getenv); err != nil {
		return fmt.Errorf("qualify: native gate open without inherited gpu lock: %w", err)
	}
	return nil
}

func qualifyNativeArch() string {
	if arch := os.Getenv("CUDA_QUALIFY_NATIVE_ARCH"); arch != "" {
		return arch
	}
	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader").Output()
	if err == nil {
		if f := strings.Fields(string(out)); len(f) > 0 {
			return "sm_" + strings.ReplaceAll(f[0], ".", "") + "a"
		}
	}
	return "sm_121a"
}

// TestQualifyNativeClosedGateControl checks, on any host, that the closed
// gate reports cudaerrors.ErrHardwarePending, that a presence-only
// GPU_LOCK_HELD=1 is refused, and that the native harness and the PTX the
// native test assembles are generated.
func TestQualifyNativeClosedGateControl(t *testing.T) {
	if os.Getenv(qualifyNativeEnv) == "1" {
		t.Skip("closed-gate control applies when CUDA_QUALIFY_NATIVE != 1")
	}
	if err := checkQualifyNativeGate(os.Getenv); !errors.Is(err, cudaerrors.ErrHardwarePending) {
		t.Fatalf("checkQualifyNativeGate(closed) = %v, want ErrHardwarePending", err)
	}
	bad := func(k string) string {
		if k == qualifyNativeEnv || k == "GPU_LOCK_HELD" {
			return "1"
		}
		return ""
	}
	if err := checkQualifyNativeGate(bad); err == nil || errors.Is(err, cudaerrors.ErrHardwarePending) {
		t.Fatalf("checkQualifyNativeGate(GPU_LOCK_HELD=1) = %v, want a lock error", err)
	}
	src := buildQualifyNativeHarness(t)
	for _, want := range []string{"qd_stochastic_round", "qd_det_gemv", "qs_abft_gemm", "qw_watchdog", "qp_paced_axpy", "QUALIFY_NATIVE_OK", "mutant_rejected="} {
		if !strings.Contains(src, want) {
			t.Errorf("native harness lacks %q", want)
		}
	}
	if len(qualifyNativePTX(t)) != 4 {
		t.Error("qualifyNativePTX did not produce four modules")
	}
}

// TestQualifyNativeQualification builds and runs the deterministic-mode,
// ABFT, watchdog and pacing kernels on the local GPU and assembles the
// emitted PTX with ptxas. Run it on GB10 (sm_121a) under gpu-lock:
//
//	CUDA_QUALIFY_NATIVE=1 go test -run TestQualifyNativeQualification ./x/egraph/qualify
func TestQualifyNativeQualification(t *testing.T) {
	if os.Getenv(qualifyNativeEnv) != "1" {
		t.Skip("native qualification pending: CUDA_QUALIFY_NATIVE=1 required")
	}
	if err := checkQualifyNativeGate(os.Getenv); err != nil {
		t.Fatalf("%v: run under gpu-lock", err)
	}
	nvcc, ver := cudatest.NVCC(t)
	target := qualifyNativeArch()
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatal(err)
	}
	base := filepath.Join(home, "tmp", "nvidia-native")
	if err := os.MkdirAll(base, 0o755); err != nil {
		t.Fatal(err)
	}
	dir, err := os.MkdirTemp(base, "qualify-native-")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })

	// Assemble the emitted PTX: modules for sm_90 run through ptxas for
	// the local target, sm_100a modules for sm_100a.
	ptxas, err := cudatool.LookTool("ptxas", "")
	if err != nil {
		t.Fatalf("ptxas: %v", err)
	}
	for name, mod := range qualifyNativePTX(t) {
		arch := target
		if strings.Contains(mod, ".target sm_100a") {
			arch = "sm_100a"
		}
		p := filepath.Join(dir, name+".ptx")
		if err := os.WriteFile(p, []byte(mod), 0o600); err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(t.Context(), time.Minute)
		out, err := exec.CommandContext(ctx, ptxas, "-arch="+arch, "-o", filepath.Join(dir, name+".cubin"), p).CombinedOutput()
		cancel()
		if err != nil {
			t.Errorf("ptxas -arch=%s %s: %v\n%s", arch, name, err, out)
			continue
		}
		t.Logf("ptxas -arch=%s %s: ok", arch, name)
	}

	cu := filepath.Join(dir, "qualify_native.cu")
	bin := filepath.Join(dir, "qualify_native")
	if err := os.WriteFile(cu, []byte(buildQualifyNativeHarness(t)), 0o600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 3*time.Minute)
	defer cancel()
	if out, err := exec.CommandContext(ctx, nvcc, "-std=c++17", "-O2", "-arch="+target, cu, "-o", bin).CombinedOutput(); err != nil {
		t.Fatalf("nvcc (%s, CUDA %s): %v\n%s", target, ver, err, out)
	}
	rctx, rcancel := context.WithTimeout(t.Context(), time.Minute)
	defer rcancel()
	out, err := exec.CommandContext(rctx, bin).CombinedOutput()
	if err != nil {
		t.Fatalf("native qualification binary (%s): %v\n%s", target, err, out)
	}
	s := string(out)
	if !strings.Contains(s, "QUALIFY_NATIVE_OK") || !strings.Contains(s, "mutant_rejected=1") {
		t.Fatalf("native qualification output lacks markers:\n%s", s)
	}
	t.Logf("native qualification passed (arch=%s, CUDA=%s):\n%s", target, ver, strings.TrimSpace(s))
}

// qualifyNativePTX returns the emitted PTX modules the native test assembles.
func qualifyNativePTX(t *testing.T) map[string]string {
	t.Helper()
	cfg := ABFTConfig{TileM: 128, TileN: 256, TileK: 64, WatchdogTimeoutNs: 50_000_000}
	sr, err := EmitDeterministicStochasticRoundPTX(DeterministicConfig{GlobalSeed: 42}, "qd_sr")
	if err != nil {
		t.Fatal(err)
	}
	abft, err := EmitABFTAndWatchdogPTX(cfg, "qs")
	if err != nil {
		t.Fatal(err)
	}
	wd, err := EmitWatchdogPTX(cfg, "qw")
	if err != nil {
		t.Fatal(err)
	}
	pace, err := EmitPowerPacedProloguePTX(PacingConfig{NumClusters: 48, RampBuckets: 8, BucketStepNs: 40, CoreVoltage: 0.85}, "qp_prologue")
	if err != nil {
		t.Fatal(err)
	}
	return map[string]string{"stochastic_round_sm100a": sr, "abft_sm100a": abft, "watchdog": wd, "pacing": pace}
}

func cU32s(v []uint32) string {
	var b strings.Builder
	for i, x := range v {
		if i > 0 {
			b.WriteString(", ")
		}
		fmt.Fprintf(&b, "%#xu", x)
	}
	return b.String()
}

func cF32s(v []float32) string { return cU32s(f32Words(v)) }

func f32Words(v []float32) []uint32 {
	w := make([]uint32, len(v))
	for i, x := range v {
		w[i] = math.Float32bits(x)
	}
	return w
}

const qualifyNativeWatchdogSrc = `
// qw_watchdog: thread 0 waits on a two-arrival mbarrier with the
// %globaltimer bound of EmitWatchdogPTX; 'arrivals' threads arrive. On
// timeout it appends a mbarrier-timeout FaultRecord.
__global__ void qw_watchdog(uint32_t* ring, int arrivals, uint64_t timeout_ns, uint64_t* elapsed_out) {
    __shared__ uint64_t bar;
    uint32_t addr = (uint32_t)__cvta_generic_to_shared(&bar);
    if (threadIdx.x == 0) asm volatile("mbarrier.init.shared::cta.b64 [%0], 2;" :: "r"(addr));
    __syncthreads();
    if ((int)threadIdx.x < arrivals) asm volatile("{ .reg .b64 state; mbarrier.arrive.shared::cta.b64 state, [%0]; }" :: "r"(addr) : "memory");
    if (threadIdx.x != 0) return;
    uint64_t t0 = qp_globaltimer(), el = 0;
    for (;;) {
        uint32_t done;
        asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2; selp.u32 %0, 1, 0, p; }"
                     : "=r"(done) : "r"(addr), "r"(0u) : "memory");
        if (done) { el = 0; break; }
        el = qp_globaltimer() - t0;
        if (el > timeout_ns) break;
    }
    *elapsed_out = el;
    if (el) qs_report_fault(ring, 3u | (5u << 16), 6u | (7u << 16), 0u, 0u, 0xffffffffu, 0.0f, el);
}
`

// buildQualifyNativeHarness returns a self-contained CUDA program that runs
// the kernels of this package and compares them with Go reference values.
func buildQualifyNativeHarness(t *testing.T) string {
	t.Helper()
	dcfg := DeterministicConfig{GlobalSeed: 0x5eed}
	det, err := DeterministicCUDA(dcfg)
	if err != nil {
		t.Fatal(err)
	}
	acfg := ABFTConfig{TileM: 32, TileN: 64, TileK: 64}
	abft, err := ABFTCUDA(acfg)
	if err != nil {
		t.Fatal(err)
	}
	pcfg := PacingConfig{NumClusters: 48, RampBuckets: 8, BucketStepNs: 40, CoreVoltage: 0.85}
	pace, err := PacedPrologueCUDA(pcfg)
	if err != nil {
		t.Fatal(err)
	}

	// Stochastic rounding: 8 tokens x 64 columns.
	const tokens, cols = 8, 64
	srIn := make([]float32, tokens*cols)
	for i := range srIn {
		srIn[i] = float32(i%29)*0.41 - 5.9
	}
	srWant := make([]uint32, tokens*cols/4)
	for i := range srWant {
		tok, col := i/(cols/4), i%(cols/4)*4
		srWant[i] = uint32(StochasticRoundE2M1x4([4]float32(srIn[tok*cols+col:]), StochasticRoundBits(dcfg, 3, uint32(100+tok), uint32(col))))
	}

	// Deterministic GEMV: 4 rows x 16 columns, K = 4096, 64-element chunks.
	const rows, n, k, kchunk = 4, 16, 4096, 64
	x, w := randomGEMV(121, rows*k, n)
	wFlat := make([]float32, 0, n*k)
	for _, r := range w {
		wFlat = append(wFlat, r[:k]...)
	}
	gemvWant := make([]float32, rows*n)
	for r := range rows {
		for c := range n {
			gemvWant[r*n+c] = FixedTreeSum(chunkSums(x[r*k:(r+1)*k], wFlat[c*k:(c+1)*k], kchunk))
		}
	}

	// ABFT tile.
	ar := rand.New(rand.NewPCG(14, 14))
	aa, ab := randomTile(ar, "f32", acfg.TileM, acfg.TileN, acfg.TileK)

	// Paced axpy: 48 blocks of 32.
	const pn = 48 * 32
	px, py, pw := make([]float32, pn), make([]float32, pn), make([]float32, pn)
	for i := range pn {
		px[i] = float32(i)*0.37 - 11
		py[i] = float32(i%17) * 1.5
		pw[i] = float32(float32(2.5*px[i]) + py[i])
	}

	var b strings.Builder
	b.WriteString("#include <cstdio>\n#include <cstdint>\n#include <cstring>\n#include <cuda_runtime.h>\n")
	b.WriteString(det)
	b.WriteString(abft)
	b.WriteString(pace)
	b.WriteString(qualifyNativeWatchdogSrc)
	fmt.Fprintf(&b, `
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("cuda error %%s at %%d\n", cudaGetErrorString(e), __LINE__); return 1; } } while (0)

// Float inputs are their IEEE bits, copied to the device as is.
static const uint32_t sr_in[] = {%s};
static const uint32_t sr_want[] = {%s};
static const uint32_t gx[] = {%s};
static const uint32_t gw[] = {%s};
static const uint32_t gemv_want[] = {%s};
static const uint32_t abft_a[] = {%s};
static const uint32_t abft_b[] = {%s};
static const uint32_t px[] = {%s};
static const uint32_t py[] = {%s};
static const uint32_t pw[] = {%s};

static int run_gemv(const char* kernel, int split, float* dx, float* dw, float* dy, float* host) {
    CK(cudaMemset(dy, 0, sizeof(float) * %d));
    if (strcmp(kernel, "det") == 0) qd_det_gemv<<<dim3(%d, %d), split>>>(dx, dw, dy, %d, %d, %d);
    else qd_naive_splitk_gemv<<<dim3(%d, %d), split>>>(dx, dw, dy, %d, %d, %d);
    CK(cudaGetLastError());
    CK(cudaMemcpy(host, dy, sizeof(float) * %d, cudaMemcpyDeviceToHost));
    return 0;
}

int main() {
    int fails = 0;

    // 1. Philox stochastic rounding, two launch shapes.
    float* din; uint16_t* dout; uint16_t hout[%d];
    CK(cudaMalloc(&din, sizeof(sr_in))); CK(cudaMalloc(&dout, sizeof(hout)));
    CK(cudaMemcpy(din, sr_in, sizeof(sr_in), cudaMemcpyHostToDevice));
    int shapes[2][2] = {{1, 128}, {5, 27}};
    for (int s = 0; s < 2; ++s) {
        CK(cudaMemset(dout, 0xff, sizeof(hout)));
        qd_stochastic_round<<<shapes[s][0], shapes[s][1]>>>(din, dout, 3u, 100u, %du, %du);
        CK(cudaGetLastError());
        CK(cudaMemcpy(hout, dout, sizeof(hout), cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int i = 0; i < %d; ++i) bad += hout[i] != sr_want[i];
        printf("stochastic_round shape=%%dx%%d mismatches=%%d\n", shapes[s][0], shapes[s][1], bad);
        fails += bad != 0;
    }

    // 2. Fixed-tree GEMV across split counts; naive split-K as the mutant.
    float *dx, *dw, *dy; float host[%d];
    CK(cudaMalloc(&dx, sizeof(gx))); CK(cudaMalloc(&dw, sizeof(gw))); CK(cudaMalloc(&dy, sizeof(host)));
    CK(cudaMemcpy(dx, gx, sizeof(gx), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dw, gw, sizeof(gw), cudaMemcpyHostToDevice));
    int splits[] = {1, 7, 32, 64};
    for (int s = 0; s < 4; ++s) {
        for (int rep = 0; rep < 50; ++rep) {
            if (run_gemv("det", splits[s], dx, dw, dy, host)) return 1;
            int bad = 0;
            for (int i = 0; i < %d; ++i) { uint32_t u; memcpy(&u, &host[i], 4); bad += u != gemv_want[i]; }
            if (bad || rep == 0) printf("det_gemv split=%%d rep=%%d mismatches=%%d\n", splits[s], rep, bad);
            fails += bad != 0;
        }
    }
    if (run_gemv("naive", 8, dx, dw, dy, host)) return 1;
    int naive_bad = 0;
    for (int i = 0; i < %d; ++i) { uint32_t u; memcpy(&u, &host[i], 4); naive_bad += u != gemv_want[i]; }
    printf("naive_splitk mismatches=%%d (nonzero expected)\n", naive_bad);

    // 3. ABFT: clean tile, then an exponent flip at (5, 9).
    float *da, *db, *dd; uint32_t* dring; uint32_t ring[8 + 8 * 4];
    CK(cudaMalloc(&da, sizeof(abft_a))); CK(cudaMalloc(&db, sizeof(abft_b)));
    CK(cudaMalloc(&dd, sizeof(float) * %d)); CK(cudaMalloc(&dring, sizeof(ring)));
    CK(cudaMemcpy(da, abft_a, sizeof(abft_a), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db, abft_b, sizeof(abft_b), cudaMemcpyHostToDevice));
    int abft_ok = 1;
    for (int inj = 0; inj < 2; ++inj) {
        memset(ring, 0, sizeof(ring)); ring[1] = 4;
        CK(cudaMemcpy(dring, ring, sizeof(ring), cudaMemcpyHostToDevice));
        qs_abft_gemm<<<1, %d>>>(da, db, dd, dring, %d, %d, %d, inj ? 5 : -1, 9, 1u << 28, 11u);
        CK(cudaGetLastError());
        CK(cudaMemcpy(ring, dring, sizeof(ring), cudaMemcpyDeviceToHost));
        int want = inj ? 1 : 0;
        int row = ring[0] ? (int)(int16_t)(ring[8 + 4] & 0xffff) : -1;
        printf("abft inject=%%d records=%%u kind=%%u row=%%d\n", inj, ring[0], ring[0] ? ring[8] & 0xff : 0, row);
        if ((int)ring[0] != want || (inj && ((ring[8] & 0xff) != 1 || row != 5))) abft_ok = 0;
    }
    fails += !abft_ok;

    // 4. Watchdog: both arrive (no record), one arrives (timeout record).
    uint64_t* del; uint64_t el;
    CK(cudaMalloc(&del, 8));
    int wd_ok = 1;
    for (int arrivals = 2; arrivals >= 1; --arrivals) {
        memset(ring, 0, sizeof(ring)); ring[1] = 4;
        CK(cudaMemcpy(dring, ring, sizeof(ring), cudaMemcpyHostToDevice));
        qw_watchdog<<<1, 32>>>(dring, arrivals, 200000ull, del);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(&el, del, 8, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(ring, dring, sizeof(ring), cudaMemcpyDeviceToHost));
        uint64_t rec_el = ring[0] ? ((uint64_t)ring[8 + 7] << 32 | ring[8 + 6]) : 0;
        printf("watchdog arrivals=%%d elapsed_ns=%%llu records=%%u kind=%%u record_elapsed=%%llu\n",
               arrivals, (unsigned long long)el, ring[0], ring[0] ? ring[8] & 0xff : 0, (unsigned long long)rec_el);
        if (arrivals == 2 && (el != 0 || ring[0] != 0)) wd_ok = 0;
        if (arrivals == 1 && (el <= 200000ull || ring[0] != 1 || (ring[8] & 0xff) != 3 || rec_el != el)) wd_ok = 0;
    }
    fails += !wd_ok;

    // 5. Paced prologue: bit-identical axpy; log the per-block start spread.
    const int pn = %d;
    static uint32_t pout[pn];
    float *dpx, *dpy; uint64_t* dstarts; uint64_t starts[48];
    CK(cudaMalloc(&dpx, sizeof(px))); CK(cudaMalloc(&dpy, sizeof(py))); CK(cudaMalloc(&dstarts, sizeof(starts)));
    CK(cudaMemcpy(dpx, px, sizeof(px), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dpy, py, sizeof(py), cudaMemcpyHostToDevice));
    qp_paced_axpy<<<48, 32>>>(dpx, dpy, 2.5f, pn, dstarts);
    CK(cudaGetLastError());
    CK(cudaMemcpy(pout, dpy, sizeof(pout), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(starts, dstarts, sizeof(starts), cudaMemcpyDeviceToHost));
    int pbad = 0;
    for (int i = 0; i < pn; ++i) pbad += pout[i] != pw[i];
    uint64_t lo = starts[0], hi = starts[0];
    for (int i = 1; i < 48; ++i) { if (starts[i] < lo) lo = starts[i]; if (starts[i] > hi) hi = starts[i]; }
    printf("paced_axpy mismatches=%%d start_spread_ns=%%llu (modeled ramp %d ns; nanosleep allows [0, 2t])\n", pbad, (unsigned long long)(hi - lo));
    fails += pbad != 0;

    printf("mutant_rejected=%%d\n", naive_bad != 0);
    if (fails == 0) printf("QUALIFY_NATIVE_OK\n");
    return fails ? 1 : 0;
}
`,
		cF32s(srIn), cU32s(srWant), cF32s(x), cF32s(wFlat), cU32s(f32Words(gemvWant)), cF32s(aa), cF32s(ab), cF32s(px), cF32s(py), cF32s(pw),
		rows*n, n, rows, k, n, kchunk, n, rows, k, n, kchunk, rows*n,
		len(srWant), tokens, cols, len(srWant),
		rows*n, rows*n, rows*n,
		acfg.TileM*acfg.TileN, acfg.TileM, acfg.TileM, acfg.TileN, acfg.TileK,
		pn, (pcfg.RampBuckets-1)*int(pcfg.BucketStepNs))
	return b.String()
}
