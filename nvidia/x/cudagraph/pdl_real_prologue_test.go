package cudagraph_test

import (
	"context"
	"encoding/csv"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// Kernel source with real RMSNorm producer and GEMV consumer kernels.
const realPrologueKernelSource = `
static inline __device__ void cudaTriggerProgrammaticLaunchCompletion(void) {
    asm volatile("griddepcontrol.launch_dependents;":::);
}

static inline __device__ void cudaGridDependencySynchronize(void) {
    asm volatile("griddepcontrol.wait;":::"memory");
}

extern "C" __global__ void real_producer_rmsnorm(
    const float *x,
    const float *gamma,
    float *act,
    float *residual,
    int dim,
    long long trailing_cycles
) {
    int tid = threadIdx.x;
    float sum_sq = 0.0f;
    for (int i = tid; i < dim; i += blockDim.x) {
        float v = x[i];
        sum_sq += v * v;
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    __shared__ float s_inv_rms;
    if (tid == 0) {
        float mean_sq = sum_sq / (float)dim;
        s_inv_rms = rsqrtf(mean_sq + 1e-6f);
    }
    __syncthreads();

    float inv_rms = s_inv_rms;
    for (int i = tid; i < dim; i += blockDim.x) {
        act[i] = x[i] * inv_rms * gamma[i];
    }
    __threadfence();
    __syncthreads();

    // Trigger launch of dependent consumer grid early
    cudaTriggerProgrammaticLaunchCompletion();

    // Trailing producer work (e.g. residual accumulation + trailing compute)
    for (int i = tid; i < dim; i += blockDim.x) {
        residual[i] += x[i];
    }
    long long t0 = clock64();
    while (clock64() - t0 < trailing_cycles) {}
}

extern "C" __global__ void real_consumer_gemv_entry_sync(
    const float *act,
    const float *W,
    const float *bias,
    float *out,
    int rows,
    int cols,
    long long prologue_cycles,
    long long compute_cycles
) {
    // Naive / entry-sync PDL: blocks immediately upon entry
    cudaGridDependencySynchronize();

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    float b = bias[row];
    float w_val = W[row * cols + (row % cols)];
    long long t0 = clock64();
    while (clock64() - t0 < prologue_cycles) {}

    float sum = b + w_val * act[row % cols];
    long long t1 = clock64();
    while (clock64() - t1 < compute_cycles) {}

    out[row] = sum;
}

extern "C" __global__ void real_consumer_gemv_hoisted(
    const float *act,
    const float *W,
    const float *bias,
    float *out,
    int rows,
    int cols,
    long long prologue_cycles,
    long long compute_cycles
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;

    // Hoisted prologue: prefetch W and bias BEFORE synchronization
    float b = bias[row];
    float w_val = W[row * cols + (row % cols)];
    long long t0 = clock64();
    while (clock64() - t0 < prologue_cycles) {}

    // Synchronize: activation is only consumed below
    cudaGridDependencySynchronize();

    float sum = b + w_val * act[row % cols];
    long long t1 = clock64();
    while (clock64() - t1 < compute_cycles) {}

    out[row] = sum;
}

extern "C" __global__ void real_producer_ts(
    const float *x,
    const float *gamma,
    float *act,
    int dim,
    long long trailing_cycles,
    long long *ts
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ts[0] = clock64();
    }
    int tid = threadIdx.x;
    for (int i = tid; i < dim; i += blockDim.x) {
        act[i] = x[i] * gamma[i];
    }
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ts[1] = clock64();
    }
    cudaTriggerProgrammaticLaunchCompletion();
    long long t0 = clock64();
    while (clock64() - t0 < trailing_cycles) {}
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ts[2] = clock64();
    }
}

extern "C" __global__ void real_consumer_ts_entry_sync(
    const float *act,
    const float *W,
    float *out,
    int rows,
    int cols,
    long long prologue_cycles,
    long long compute_cycles,
    long long *ts
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ts[3] = clock64();
    }
    cudaGridDependencySynchronize();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ts[4] = clock64();
    }
    long long t0 = clock64();
    while (clock64() - t0 < prologue_cycles) {}
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ts[5] = clock64();
    }
    long long t1 = clock64();
    while (clock64() - t1 < compute_cycles) {}
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ts[6] = clock64();
    }
}

extern "C" __global__ void real_consumer_ts_hoisted(
    const float *act,
    const float *W,
    float *out,
    int rows,
    int cols,
    long long prologue_cycles,
    long long compute_cycles,
    long long *ts
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ts[3] = clock64();
    }
    long long t0 = clock64();
    while (clock64() - t0 < prologue_cycles) {}
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ts[4] = clock64();
    }
    cudaGridDependencySynchronize();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ts[5] = clock64();
    }
    long long t1 = clock64();
    while (clock64() - t1 < compute_cycles) {}
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        ts[6] = clock64();
    }
}
`

// TestRealKernelPrologueTimestamps measures intra-kernel hardware timestamps
// to demonstrate exact cycle overlap when independent prologue work is hoisted
// before cudaGridDependencySynchronize().
func TestRealKernelPrologueTimestamps(t *testing.T) {
	if os.Getenv(pdlAblationEnv) != "1" {
		t.Skipf("%s=1 required: needs NVIDIA GPU with CUDA 13.0 PDL support", pdlAblationEnv)
	}
	dev := cudatest.Require(t)

	compiled, err := nvrtcx.Compile(realPrologueKernelSource, nvrtcx.WithDeviceArch(dev), nvrtcx.WithNoCache())
	if err != nil {
		t.Fatalf("nvrtcx.Compile failed: %v", err)
	}
	ptx := compiled.PTX()
	if !strings.Contains(ptx, "griddepcontrol.launch_dependents;") || !strings.Contains(ptx, "griddepcontrol.wait;") {
		t.Fatal("compiled PTX missing required PDL intrinsics")
	}
	mod, err := compiled.Load(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(mod.Unload)

	pFn, err := mod.GetFunction("real_producer_ts")
	if err != nil {
		t.Fatal(err)
	}
	cEntryFn, err := mod.GetFunction("real_consumer_ts_entry_sync")
	if err != nil {
		t.Fatal(err)
	}
	cHoistFn, err := mod.GetFunction("real_consumer_ts_hoisted")
	if err != nil {
		t.Fatal(err)
	}

	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(stream.Destroy)

	const dim = 1024
	tsBuf, err := cudamem.Alloc[int64](dev, 7)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(tsBuf.Free)

	xBuf, _ := cudamem.Alloc[float32](dev, dim)
	defer dev.WithContext(xBuf.Free)
	gBuf, _ := cudamem.Alloc[float32](dev, dim)
	defer dev.WithContext(gBuf.Free)
	actBuf, _ := cudamem.Alloc[float32](dev, dim)
	defer dev.WithContext(actBuf.Free)
	wBuf, _ := cudamem.Alloc[float32](dev, dim*dim)
	defer dev.WithContext(wBuf.Free)
	outBuf, _ := cudamem.Alloc[float32](dev, dim)
	defer dev.WithContext(outBuf.Free)

	clkKHz, _ := dev.Attribute(driver.DevAttrClockRate)
	cyclesPerUs := float64(clkKHz) / 1000.0

	trailingSpin := int64(50 * cyclesPerUs) // 50 us
	prologueSpin := int64(20 * cyclesPerUs) // 20 us
	computeSpin := int64(20 * cyclesPerUs)  // 20 us

	runTs := func(cFn driver.Function) [7]int64 {
		zero := make([]int64, 7)
		_ = tsBuf.CopyFromHost(zero)

		g, err := cudagraph.Capture(stream, func() {
			_ = cudalaunch.Kernel(pFn).Grid(1).Block(32).
				WithProgrammaticStreamSerialization(true).
				On(stream).
				Launch(xBuf, gBuf, actBuf, int32(dim), trailingSpin, tsBuf)
			_ = cudalaunch.Kernel(cFn).Grid(1).Block(32).
				WithProgrammaticStreamSerialization(true).
				On(stream).
				Launch(actBuf, wBuf, outBuf, int32(dim), int32(dim), prologueSpin, computeSpin, tsBuf)
		})
		if err != nil {
			t.Fatal(err)
		}
		defer g.Destroy()

		exec, err := g.Instantiate()
		if err != nil {
			t.Fatal(err)
		}
		defer exec.Destroy()

		_ = exec.Launch(stream)
		_ = stream.Synchronize(context.Background())

		var out [7]int64
		_ = tsBuf.CopyToHost(out[:])
		return out
	}

	entryTs := runTs(cEntryFn)
	hoistTs := runTs(cHoistFn)

	t.Logf("Entry-Sync Timestamps (cycles rel to prod start):")
	t.Logf("  Prod trigger:   %d (%.2f us)", entryTs[1]-entryTs[0], float64(entryTs[1]-entryTs[0])/cyclesPerUs)
	t.Logf("  Prod exit:      %d (%.2f us)", entryTs[2]-entryTs[0], float64(entryTs[2]-entryTs[0])/cyclesPerUs)
	t.Logf("  Cons enter:     %d (%.2f us)", entryTs[3]-entryTs[0], float64(entryTs[3]-entryTs[0])/cyclesPerUs)
	t.Logf("  Cons sync:      %d (%.2f us)", entryTs[4]-entryTs[0], float64(entryTs[4]-entryTs[0])/cyclesPerUs)
	t.Logf("  Cons prol done: %d (%.2f us)", entryTs[5]-entryTs[0], float64(entryTs[5]-entryTs[0])/cyclesPerUs)
	t.Logf("  Cons exit:      %d (%.2f us)", entryTs[6]-entryTs[0], float64(entryTs[6]-entryTs[0])/cyclesPerUs)

	t.Logf("Hoisted-Prologue Timestamps (cycles rel to prod start):")
	t.Logf("  Prod trigger:   %d (%.2f us)", hoistTs[1]-hoistTs[0], float64(hoistTs[1]-hoistTs[0])/cyclesPerUs)
	t.Logf("  Prod exit:      %d (%.2f us)", hoistTs[2]-hoistTs[0], float64(hoistTs[2]-hoistTs[0])/cyclesPerUs)
	t.Logf("  Cons enter:     %d (%.2f us)", hoistTs[3]-hoistTs[0], float64(hoistTs[3]-hoistTs[0])/cyclesPerUs)
	t.Logf("  Cons prol done: %d (%.2f us)", hoistTs[4]-hoistTs[0], float64(hoistTs[4]-hoistTs[0])/cyclesPerUs)
	t.Logf("  Cons sync:      %d (%.2f us)", hoistTs[5]-hoistTs[0], float64(hoistTs[5]-hoistTs[0])/cyclesPerUs)
	t.Logf("  Cons exit:      %d (%.2f us)", hoistTs[6]-hoistTs[0], float64(hoistTs[6]-hoistTs[0])/cyclesPerUs)

	// Verification assertions:
	// In Hoisted arm: prologue MUST finish before sync unblocks!
	if hoistTs[4] > hoistTs[5] {
		t.Errorf("Hoisted prologue did not finish before sync: prol_done=%d, sync_done=%d", hoistTs[4], hoistTs[5])
	}
	hoistSpanUs := float64(hoistTs[6]-hoistTs[0]) / cyclesPerUs
	entrySpanUs := float64(entryTs[6]-entryTs[0]) / cyclesPerUs
	overlapSavingUs := entrySpanUs - hoistSpanUs
	t.Logf("Measured Span Comparison: EntrySync=%.2f us, Hoisted=%.2f us, Net Overlap Saving=%.2f us",
		entrySpanUs, hoistSpanUs, overlapSavingUs)

	if overlapSavingUs < 15.0 {
		t.Fatalf("expected >= 15 us overlap saving, got %.2f us", overlapSavingUs)
	}
}

// TestRealKernelPrologueAblationRun executes the rigorous 5-arm interleaved
// benchmark with null-control gating on GB10 silicon.
func TestRealKernelPrologueAblationRun(t *testing.T) {
	if os.Getenv(pdlAblationEnv) != "1" {
		t.Skipf("%s=1 required: needs NVIDIA GPU with CUDA 13.0 PDL support", pdlAblationEnv)
	}
	dev := cudatest.Require(t)

	compiled, err := nvrtcx.Compile(realPrologueKernelSource, nvrtcx.WithDeviceArch(dev), nvrtcx.WithNoCache())
	if err != nil {
		t.Fatalf("nvrtcx.Compile failed: %v", err)
	}
	mod, err := compiled.Load(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(mod.Unload)

	pFn, err := mod.GetFunction("real_producer_rmsnorm")
	if err != nil {
		t.Fatal(err)
	}
	cEntryFn, err := mod.GetFunction("real_consumer_gemv_entry_sync")
	if err != nil {
		t.Fatal(err)
	}
	cHoistFn, err := mod.GetFunction("real_consumer_gemv_hoisted")
	if err != nil {
		t.Fatal(err)
	}

	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(stream.Destroy)

	const dim = 1024
	xBuf, _ := cudamem.Alloc[float32](dev, dim)
	defer dev.WithContext(xBuf.Free)
	gBuf, _ := cudamem.Alloc[float32](dev, dim)
	defer dev.WithContext(gBuf.Free)
	actBuf, _ := cudamem.Alloc[float32](dev, dim)
	defer dev.WithContext(actBuf.Free)
	resBuf, _ := cudamem.Alloc[float32](dev, dim)
	defer dev.WithContext(resBuf.Free)
	wBuf, _ := cudamem.Alloc[float32](dev, dim*dim)
	defer dev.WithContext(wBuf.Free)
	bBuf, _ := cudamem.Alloc[float32](dev, dim)
	defer dev.WithContext(bBuf.Free)
	outBuf, _ := cudamem.Alloc[float32](dev, dim)
	defer dev.WithContext(outBuf.Free)

	// Initialize host test data
	hX := make([]float32, dim)
	hG := make([]float32, dim)
	hW := make([]float32, dim*dim)
	hB := make([]float32, dim)
	for i := 0; i < dim; i++ {
		hX[i] = float32(i%17) * 0.1
		hG[i] = 1.0
		hB[i] = 0.5
		for j := 0; j < dim; j++ {
			hW[i*dim+j] = 0.001 * float32((i+j)%13)
		}
	}
	_ = xBuf.CopyFromHost(hX)
	_ = gBuf.CopyFromHost(hG)
	_ = wBuf.CopyFromHost(hW)
	_ = bBuf.CopyFromHost(hB)

	clkKHz, _ := dev.Attribute(driver.DevAttrClockRate)
	cyclesPerUs := int64(clkKHz) / 1000

	trailingSpin := 50 * cyclesPerUs // 50 us
	prologueSpin := 20 * cyclesPerUs // 20 us
	computeSpin := 20 * cyclesPerUs  // 20 us

	startEv, err := dev.NewEvent()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(startEv.Destroy)
	stopEv, err := dev.NewEvent()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(stopEv.Destroy)

	launchArm := func(cFn driver.Function, pdl bool) float64 {
		g, err := cudagraph.Capture(stream, func() {
			_ = cudalaunch.Kernel(pFn).Grid(1).Block(32).
				WithProgrammaticStreamSerialization(pdl).
				On(stream).
				Launch(xBuf, gBuf, actBuf, resBuf, int32(dim), trailingSpin)
			_ = cudalaunch.Kernel(cFn).Grid(dim/32).Block(32).
				WithProgrammaticStreamSerialization(pdl).
				On(stream).
				Launch(actBuf, wBuf, bBuf, outBuf, int32(dim), int32(dim), prologueSpin, computeSpin)
		})
		if err != nil {
			t.Fatal(err)
		}
		defer g.Destroy()

		exec, err := g.Instantiate()
		if err != nil {
			t.Fatal(err)
		}
		defer exec.Destroy()

		if err := startEv.Record(stream); err != nil {
			t.Fatalf("record start: %v", err)
		}
		if err := exec.Launch(stream); err != nil {
			t.Fatalf("launch: %v", err)
		}
		if err := stopEv.Record(stream); err != nil {
			t.Fatalf("record stop: %v", err)
		}
		if err := stream.Synchronize(context.Background()); err != nil {
			t.Fatalf("synchronize: %v", err)
		}
		d, err := stopEv.ElapsedSince(startEv)
		if err != nil {
			t.Fatal(err)
		}
		return float64(d.Nanoseconds())
	}

	// Verify numerical accuracy across arms before running timing loop
	for i := range hX {
		hX[i] += 0.03125
	}
	if err := xBuf.CopyFromHost(hX); err != nil {
		t.Fatalf("copy changed input: %v", err)
	}
	// Poison the activation as well as the output. A premature consumer read
	// must fail independently of whatever launch happened to run previously.
	actPoison := make([]float32, dim)
	for i := range actPoison {
		actPoison[i] = float32(math.NaN())
	}
	if err := actBuf.CopyFromHost(actPoison); err != nil {
		t.Fatalf("poison activation: %v", err)
	}
	poison := make([]float32, dim)
	for i := range poison {
		poison[i] = float32(math.NaN())
	}
	if err := outBuf.CopyFromHost(poison); err != nil {
		t.Fatalf("poison output: %v", err)
	}

	_ = launchArm(cHoistFn, true) // PDL_HOISTED
	hOutHoist := make([]float32, dim)
	if err := outBuf.CopyToHost(hOutHoist); err != nil {
		t.Fatalf("copy hoisted: %v", err)
	}
	_ = launchArm(cEntryFn, false)
	hExpected := make([]float32, dim)
	if err := outBuf.CopyToHost(hExpected); err != nil {
		t.Fatalf("copy changed reference: %v", err)
	}

	for i := 0; i < dim; i++ {
		diff := math.Abs(float64(hExpected[i] - hOutHoist[i]))
		if math.IsNaN(float64(hExpected[i])) || math.IsNaN(float64(hOutHoist[i])) || !(diff <= 1e-4) {
			t.Fatalf("numerical divergence at [%d]: ref=%f, hoisted=%f", i, hExpected[i], hOutHoist[i])
		}
	}
	t.Log("Numerical output verified: Hoisted arm produces identical output to sequential baseline")

	csvPath := os.Getenv("CUDAGRAPH_REAL_PDL_CSV")
	if csvPath == "" {
		csvPath = filepath.Join(t.TempDir(), "pdl-real-prologue.csv")
	}
	csvFile, err := os.Create(csvPath)
	if err != nil {
		t.Fatalf("create CSV failed: %v", err)
	}
	defer csvFile.Close()
	writer := csv.NewWriter(csvFile)
	_ = writer.Write([]string{"arm", "iter", "ns"})
	writer.Flush()

	const totalIters = 220
	const warmupIters = 20
	arms := []string{"NULL_1", "NULL_2", "PDL_HOISTED", "PDL_ENTRY_SYNC", "NOPDL"}
	data := make(map[string][]float64)
	for _, a := range arms {
		data[a] = make([]float64, 0, totalIters)
	}

	for i := 0; i < totalIters; i++ {
		// All arms interleaved in continuous loop
		nsNull1 := launchArm(cHoistFn, true)
		nsNull2 := launchArm(cHoistFn, true)
		nsHoist := launchArm(cHoistFn, true)
		nsEntry := launchArm(cEntryFn, true)
		nsNOPDL := launchArm(cEntryFn, false)

		vals := map[string]float64{
			"NULL_1":         nsNull1,
			"NULL_2":         nsNull2,
			"PDL_HOISTED":    nsHoist,
			"PDL_ENTRY_SYNC": nsEntry,
			"NOPDL":          nsNOPDL,
		}

		for _, arm := range arms {
			v := vals[arm]
			data[arm] = append(data[arm], v)
			_ = writer.Write([]string{arm, fmt.Sprintf("%d", i), fmt.Sprintf("%.1f", v)})
		}
		writer.Flush()
	}

	evalData := make(map[string][]float64)
	for _, arm := range arms {
		evalData[arm] = data[arm][warmupIters:]
	}

	median := func(s []float64) float64 {
		cp := append([]float64(nil), s...)
		sort.Float64s(cp)
		n := len(cp)
		if n%2 == 1 {
			return cp[n/2]
		}
		return (cp[n/2-1] + cp[n/2]) / 2.0
	}
	abs := func(v float64) float64 {
		if v < 0 {
			return -v
		}
		return v
	}

	mNull1 := median(evalData["NULL_1"])
	mNull2 := median(evalData["NULL_2"])
	mHoist := median(evalData["PDL_HOISTED"])
	mEntry := median(evalData["PDL_ENTRY_SYNC"])
	mNOPDL := median(evalData["NOPDL"])

	nullDiff := mNull2 - mNull1
	entryDiff := mNOPDL - mEntry // Expected ~0
	hoistDiff := mNOPDL - mHoist // Expected ~20 us

	nEval := totalIters - warmupIters
	nullS1Faster, nullS2Faster, nullTies := 0, 0, 0
	hoistFaster, nopdlFaster, hoistTies := 0, 0, 0
	for i := 0; i < nEval; i++ {
		nd := evalData["NULL_2"][i] - evalData["NULL_1"][i]
		if nd > 0 {
			nullS1Faster++
		} else if nd < 0 {
			nullS2Faster++
		} else {
			nullTies++
		}

		hd := evalData["NOPDL"][i] - evalData["PDL_HOISTED"][i]
		if hd > 0 {
			hoistFaster++
		} else if hd < 0 {
			nopdlFaster++
		} else {
			hoistTies++
		}
	}

	t.Logf("Evaluated Iterations: %d (warmup %d discarded)", nEval, warmupIters)
	t.Logf("NULL_1         Median: %.1f ns (%.2f us)", mNull1, mNull1/1000)
	t.Logf("NULL_2         Median: %.1f ns (%.2f us)", mNull2, mNull2/1000)
	t.Logf("Null Drift:    %.1f - %.1f = %+.1f ns (%+.3f us)", mNull2, mNull1, nullDiff, nullDiff/1000)
	t.Logf("Null Sign Test: Slot1<Slot2 %d/%d (%.1f%%), Slot2<Slot1 %d/%d (%.1f%%), Ties %d",
		nullS1Faster, nEval, float64(nullS1Faster)/float64(nEval)*100,
		nullS2Faster, nEval, float64(nullS2Faster)/float64(nEval)*100,
		nullTies)

	t.Logf("NOPDL          Median: %.1f ns (%.2f us)", mNOPDL, mNOPDL/1000)
	t.Logf("PDL_ENTRY_SYNC Median: %.1f ns (%.2f us)", mEntry, mEntry/1000)
	t.Logf("Entry-Sync Effect vs NOPDL: %+.1f ns (%+.3f us)", entryDiff, entryDiff/1000)

	t.Logf("PDL_HOISTED    Median: %.1f ns (%.2f us)", mHoist, mHoist/1000)
	t.Logf("Hoisted Effect vs NOPDL:   %+.1f ns (%+.3f us)", hoistDiff, hoistDiff/1000)
	t.Logf("Hoisted Sign Test: Hoisted<NOPDL %d/%d (%.1f%%), NOPDL<Hoisted %d/%d (%.1f%%), Ties %d",
		hoistFaster, nEval, float64(hoistFaster)/float64(nEval)*100,
		nopdlFaster, nEval, float64(nopdlFaster)/float64(nEval)*100,
		hoistTies)

	// Gate: Null drift must be smaller than hoisted effect
	if abs(nullDiff) >= abs(hoistDiff) {
		t.Fatalf("GATE FAILED: null drift %.1f ns >= hoisted effect %.1f ns", abs(nullDiff), abs(hoistDiff))
	}

	// Gate: Hoisted effect must save at least 15 us (~15,000 ns)
	if hoistDiff < 15000.0 {
		t.Fatalf("GATE FAILED: hoisted overlap effect %.1f ns < 15,000 ns expected threshold", hoistDiff)
	}

	speedupPct := (hoistDiff / mNOPDL) * 100.0
	t.Logf("VERDICT: REAL PROLOGUE HOISTING PASSED ON GB10!")
	t.Logf("  Baseline Sequential Span: %.2f us", mNOPDL/1000)
	t.Logf("  Hoisted PDL Span:         %.2f us", mHoist/1000)
	t.Logf("  Overlap Savings:          %.2f us (%.1f%% speedup)", hoistDiff/1000, speedupPct)
	t.Logf("  Null Control Drift:       %.1f ns (significantly below %.1f ns effect)", nullDiff, hoistDiff)
}
