package cudagraph_test

import (
	"context"
	"encoding/csv"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/tmc/lib/nvidia/driver"

	"github.com/tmc/lib/nvidia/x/cudagraph"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
	"github.com/tmc/lib/nvidia/x/cudamem"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

// pdlAblationEnv gates hardware ablation benchmarks in this file.
const pdlAblationEnv = "CUDAGRAPH_PDL_ABLATION"

// Kernel source with programmatic dependent launch (PDL) intrinsics.
const pdlAblationKernelSource = `
static inline __device__ void cudaTriggerProgrammaticLaunchCompletion(void) {
    asm volatile("griddepcontrol.launch_dependents;":::);
}

static inline __device__ void cudaGridDependencySynchronize(void) {
    asm volatile("griddepcontrol.wait;":::"memory");
}

extern "C" __global__ void producer(int *out, long long spin_cycles) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out = 42;
    }
    cudaTriggerProgrammaticLaunchCompletion();
    long long start = clock64();
    while (clock64() - start < spin_cycles) {}
}

extern "C" __global__ void consumer(const int *in, int *out, long long spin_cycles) {
    cudaGridDependencySynchronize();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *out = *in + 1;
    }
    long long start = clock64();
    while (clock64() - start < spin_cycles) {}
}

extern "C" __global__ void producer_ts(long long *timestamps, long long spin_cycles) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        timestamps[0] = clock64();
    }
    cudaTriggerProgrammaticLaunchCompletion();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        timestamps[1] = clock64();
    }
    long long start = clock64();
    while (clock64() - start < spin_cycles) {}
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        timestamps[2] = clock64();
    }
}

extern "C" __global__ void consumer_ts(long long *timestamps, long long spin_cycles) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        timestamps[3] = clock64();
    }
    cudaGridDependencySynchronize();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        timestamps[4] = clock64();
    }
    long long start = clock64();
    while (clock64() - start < spin_cycles) {}
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        timestamps[5] = clock64();
    }
}
`

// TestPDLCompilationAndIntrinsics verifies that NVRTC emits the required
// griddepcontrol.launch_dependents and griddepcontrol.wait instructions.
func TestPDLCompilationAndIntrinsics(t *testing.T) {
	if os.Getenv(pdlAblationEnv) != "1" {
		t.Skipf("%s=1 required: needs NVIDIA GPU with CUDA 13.0 PDL support", pdlAblationEnv)
	}
	dev := cudatest.Require(t)

	compiled, err := nvrtcx.Compile(pdlAblationKernelSource, nvrtcx.WithDeviceArch(dev), nvrtcx.WithNoCache())
	if err != nil {
		t.Fatalf("nvrtcx.Compile failed: %v", err)
	}
	ptx := compiled.PTX()
	if len(ptx) == 0 {
		t.Fatal("empty PTX output from nvrtcx")
	}

	const reqLaunch = "griddepcontrol.launch_dependents;"
	const reqWait = "griddepcontrol.wait;"

	if !strings.Contains(ptx, reqLaunch) {
		t.Fatalf("PTX missing required instruction %q", reqLaunch)
	}
	if !strings.Contains(ptx, reqWait) {
		t.Fatalf("PTX missing required instruction %q", reqWait)
	}
	t.Logf("PTX verified: found %q and %q in %d bytes PTX", reqLaunch, reqWait, len(ptx))
}

// TestPDLPositiveControl verifies that the measurement instrument (device clock events)
// reliably resolves a known +20 us variation in kernel execution time.
func TestPDLPositiveControl(t *testing.T) {
	if os.Getenv(pdlAblationEnv) != "1" {
		t.Skipf("%s=1 required: needs NVIDIA GPU with CUDA 13.0 PDL support", pdlAblationEnv)
	}
	dev := cudatest.Require(t)

	compiled, err := nvrtcx.Compile(pdlAblationKernelSource, nvrtcx.WithDeviceArch(dev), nvrtcx.WithNoCache())
	if err != nil {
		t.Fatalf("nvrtcx.Compile failed: %v", err)
	}
	mod, err := compiled.Load(dev)
	if err != nil {
		t.Fatalf("module load failed: %v", err)
	}
	defer dev.WithContext(mod.Unload)

	pFn, err := mod.GetFunction("producer")
	if err != nil {
		t.Fatal(err)
	}
	cFn, err := mod.GetFunction("consumer")
	if err != nil {
		t.Fatal(err)
	}

	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(stream.Destroy)

	buf1, err := cudamem.Alloc[int32](dev, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(buf1.Free)
	buf2, err := cudamem.Alloc[int32](dev, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(buf2.Free)

	clkKHz, err := dev.Attribute(driver.DevAttrClockRate)
	if err != nil {
		t.Fatal(err)
	}
	cyclesPerUs := int64(clkKHz) / 1000

	baseSpin := 30 * cyclesPerUs  // 30 us
	extraSpin := 50 * cyclesPerUs // 50 us (+20 us)
	consSpin := 20 * cyclesPerUs  // 20 us

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

	measure := func(prodSpin int64) float64 {
		g, err := cudagraph.Capture(stream, func() {
			_ = cudalaunch.Kernel(pFn).Grid(1).Block(32).On(stream).Launch(buf1, prodSpin)
			_ = cudalaunch.Kernel(cFn).Grid(1).Block(32).On(stream).Launch(buf1, buf2, consSpin)
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

		var total float64
		const iters = 15
		for i := 0; i < iters; i++ {
			_ = startEv.Record(stream)
			_ = exec.Launch(stream)
			_ = stopEv.Record(stream)
			_ = stream.Synchronize(context.Background())
			d, _ := stopEv.ElapsedSince(startEv)
			if i >= 5 {
				total += float64(d.Nanoseconds())
			}
		}
		return total / float64(iters-5)
	}

	baseSpanNs := measure(baseSpin)
	extraSpanNs := measure(extraSpin)
	deltaNs := extraSpanNs - baseSpanNs
	expectedNs := float64(20 * time.Microsecond)

	t.Logf("Base (%d cycles, 30 us): %.1f ns (%.2f us)", baseSpin, baseSpanNs, baseSpanNs/1000)
	t.Logf("Extra (%d cycles, 50 us): %.1f ns (%.2f us)", extraSpin, extraSpanNs, extraSpanNs/1000)
	t.Logf("Measured delta: %.1f ns (%.2f us), Expected: %.1f ns (%.2f us)", deltaNs, deltaNs/1000, expectedNs, expectedNs/1000)

	// Check that measured delta is within 25% of expected 20 us
	ratio := deltaNs / expectedNs
	if ratio < 0.75 || ratio > 1.25 {
		t.Fatalf("positive control failed: measured delta ratio %.2f outside [0.75, 1.25]", ratio)
	}
	t.Log("Positive control PASSED: instrument accurately detects 20 us shift")
}

// TestPDLHardwareTimestamps directly measures intra-kernel hardware timestamps
// to demonstrate exact CTA launch and synchronization boundaries.
func TestPDLHardwareTimestamps(t *testing.T) {
	if os.Getenv(pdlAblationEnv) != "1" {
		t.Skipf("%s=1 required: needs NVIDIA GPU with CUDA 13.0 PDL support", pdlAblationEnv)
	}
	dev := cudatest.Require(t)

	compiled, err := nvrtcx.Compile(pdlAblationKernelSource, nvrtcx.WithDeviceArch(dev), nvrtcx.WithNoCache())
	if err != nil {
		t.Fatalf("nvrtcx.Compile failed: %v", err)
	}
	mod, err := compiled.Load(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(mod.Unload)

	pFn, err := mod.GetFunction("producer_ts")
	if err != nil {
		t.Fatal(err)
	}
	cFn, err := mod.GetFunction("consumer_ts")
	if err != nil {
		t.Fatal(err)
	}

	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(stream.Destroy)

	tsBuf, err := cudamem.Alloc[int64](dev, 6)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(tsBuf.Free)

	clkKHz, _ := dev.Attribute(driver.DevAttrClockRate)
	cyclesPerUs := float64(clkKHz) / 1000.0

	prodSpin := int64(50 * cyclesPerUs) // 50 us
	consSpin := int64(20 * cyclesPerUs) // 20 us

	runTs := func(pdl bool) [6]int64 {
		zero := make([]int64, 6)
		_ = tsBuf.CopyFromHost(zero)

		g, err := cudagraph.Capture(stream, func() {
			_ = cudalaunch.Kernel(pFn).Grid(1).Block(32).
				WithProgrammaticStreamSerialization(pdl).
				On(stream).
				Launch(tsBuf, prodSpin)
			_ = cudalaunch.Kernel(cFn).Grid(1).Block(32).
				WithProgrammaticStreamSerialization(pdl).
				On(stream).
				Launch(tsBuf, consSpin)
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

		var out [6]int64
		_ = tsBuf.CopyToHost(out[:])
		return out
	}

	nopdlTs := runTs(false)
	pdlTs := runTs(true)

	t.Logf("NOPDL Timestamps (cycles rel to prod start):")
	t.Logf("  Prod exit:     %d (%.2f us)", nopdlTs[2]-nopdlTs[0], float64(nopdlTs[2]-nopdlTs[0])/cyclesPerUs)
	t.Logf("  Cons enter:    %d (%.2f us)", nopdlTs[3]-nopdlTs[0], float64(nopdlTs[3]-nopdlTs[0])/cyclesPerUs)
	t.Logf("  Cons sync:     %d (%.2f us)", nopdlTs[4]-nopdlTs[0], float64(nopdlTs[4]-nopdlTs[0])/cyclesPerUs)
	t.Logf("  Cons done:     %d (%.2f us)", nopdlTs[5]-nopdlTs[0], float64(nopdlTs[5]-nopdlTs[0])/cyclesPerUs)

	t.Logf("PDL Timestamps (cycles rel to prod start):")
	t.Logf("  Prod exit:     %d (%.2f us)", pdlTs[2]-pdlTs[0], float64(pdlTs[2]-pdlTs[0])/cyclesPerUs)
	t.Logf("  Cons enter:    %d (%.2f us)", pdlTs[3]-pdlTs[0], float64(pdlTs[3]-pdlTs[0])/cyclesPerUs)
	t.Logf("  Cons sync:     %d (%.2f us)", pdlTs[4]-pdlTs[0], float64(pdlTs[4]-pdlTs[0])/cyclesPerUs)
	t.Logf("  Cons done:     %d (%.2f us)", pdlTs[5]-pdlTs[0], float64(pdlTs[5]-pdlTs[0])/cyclesPerUs)
}

// TestPDLAblationRun runs the 4-arm interleaved ablation experiment and writes
// raw data to pdl-ablation-redo.csv.
func TestPDLAblationRun(t *testing.T) {
	if os.Getenv(pdlAblationEnv) != "1" {
		t.Skipf("%s=1 required: needs NVIDIA GPU with CUDA 13.0 PDL support", pdlAblationEnv)
	}
	dev := cudatest.Require(t)

	compiled, err := nvrtcx.Compile(pdlAblationKernelSource, nvrtcx.WithDeviceArch(dev), nvrtcx.WithNoCache())
	if err != nil {
		t.Fatalf("nvrtcx.Compile failed: %v", err)
	}
	mod, err := compiled.Load(dev)
	if err != nil {
		t.Fatalf("module load failed: %v", err)
	}
	defer dev.WithContext(mod.Unload)

	pFn, err := mod.GetFunction("producer")
	if err != nil {
		t.Fatal(err)
	}
	cFn, err := mod.GetFunction("consumer")
	if err != nil {
		t.Fatal(err)
	}

	stream, err := dev.NewStream()
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(stream.Destroy)

	buf1, err := cudamem.Alloc[int32](dev, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(buf1.Free)
	buf2, err := cudamem.Alloc[int32](dev, 1)
	if err != nil {
		t.Fatal(err)
	}
	defer dev.WithContext(buf2.Free)

	clkKHz, _ := dev.Attribute(driver.DevAttrClockRate)
	cyclesPerUs := int64(clkKHz) / 1000

	prodSpin := 50 * cyclesPerUs // 50 us
	consSpin := 20 * cyclesPerUs // 20 us

	// Raw per-iteration spans are written as they are produced, so a run that
	// dies still leaves its data. CUDAGRAPH_PDL_CSV names the file; without it
	// the rows land in the test's temporary directory.
	csvPath := os.Getenv("CUDAGRAPH_PDL_CSV")
	if csvPath == "" {
		csvPath = filepath.Join(t.TempDir(), "pdl-ablation.csv")
	}
	csvFile, err := os.Create(csvPath)
	if err != nil {
		t.Fatalf("create CSV failed: %v", err)
	}
	t.Logf("raw spans: %s", csvPath)
	defer csvFile.Close()
	writer := csv.NewWriter(csvFile)
	_ = writer.Write([]string{"arm", "iter", "ns"})
	writer.Flush()

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

	launchArm := func(pdl bool) float64 {
		g, err := cudagraph.Capture(stream, func() {
			_ = cudalaunch.Kernel(pFn).Grid(1).Block(32).
				WithProgrammaticStreamSerialization(pdl).
				On(stream).
				Launch(buf1, prodSpin)
			_ = cudalaunch.Kernel(cFn).Grid(1).Block(32).
				WithProgrammaticStreamSerialization(pdl).
				On(stream).
				Launch(buf1, buf2, consSpin)
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

		_ = startEv.Record(stream)
		_ = exec.Launch(stream)
		_ = stopEv.Record(stream)
		_ = stream.Synchronize(context.Background())
		d, err := stopEv.ElapsedSince(startEv)
		if err != nil {
			t.Fatal(err)
		}
		return float64(d.Nanoseconds())
	}

	const totalIters = 220
	const warmupIters = 20
	arms := []string{"NULL_1", "NULL_2", "PDL", "NOPDL"}
	data := make(map[string][]float64)
	for _, a := range arms {
		data[a] = make([]float64, 0, totalIters)
	}

	for i := 0; i < totalIters; i++ {
		// All 4 arms interleaved in ONE continuous loop
		// NULL_1: PDL config slot 1
		nsNull1 := launchArm(true)
		// NULL_2: PDL config slot 2 (identical config)
		nsNull2 := launchArm(true)
		// PDL: PDL config
		nsPDL := launchArm(true)
		// NOPDL: Standard non-PDL config
		nsNOPDL := launchArm(false)

		vals := map[string]float64{
			"NULL_1": nsNull1,
			"NULL_2": nsNull2,
			"PDL":    nsPDL,
			"NOPDL":  nsNOPDL,
		}

		for _, arm := range arms {
			v := vals[arm]
			data[arm] = append(data[arm], v)
			_ = writer.Write([]string{arm, fmt.Sprintf("%d", i), fmt.Sprintf("%.1f", v)})
		}
		writer.Flush()
	}

	// Discard first 20 as warmup
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

	mNull1 := median(evalData["NULL_1"])
	mNull2 := median(evalData["NULL_2"])
	mPDL := median(evalData["PDL"])
	mNOPDL := median(evalData["NOPDL"])

	nullDiff := mNull2 - mNull1
	pdlDiff := mNOPDL - mPDL

	t.Logf("Evaluated Iterations: %d (warmup %d discarded)", totalIters-warmupIters, warmupIters)
	t.Logf("NULL_1 Median: %.1f ns (%.2f us)", mNull1, mNull1/1000)
	t.Logf("NULL_2 Median: %.1f ns (%.2f us)", mNull2, mNull2/1000)
	t.Logf("Null Drift: %.1f - %.1f = %+.1f ns (%+.3f us)", mNull2, mNull1, nullDiff, nullDiff/1000)
	t.Logf("PDL    Median: %.1f ns (%.2f us)", mPDL, mPDL/1000)
	t.Logf("NOPDL  Median: %.1f ns (%.2f us)", mNOPDL, mNOPDL/1000)
	t.Logf("PDL Effect: %.1f - %.1f = %+.1f ns (%+.3f us)", mNOPDL, mPDL, pdlDiff, pdlDiff/1000)
}
