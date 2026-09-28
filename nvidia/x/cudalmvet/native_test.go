//go:build native

package cudalmvet

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"testing"
	"unsafe"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func TestTraceFoldPreservesConcurrentRaceSignal(t *testing.T) {
	dev := cudatest.Require(t)
	if dev == nil {
		return
	}
	defer dev.Close()
	source := nvrtcx.TraceHeader() + `
__device__ volatile unsigned int race_cells[1024];
__device__ inline float value_for(int idx, int racing) {
    if (!racing) return (float)idx;
    race_cells[idx] = (unsigned int)(idx + 1);
    __threadfence();
    for (volatile int delay = 0; delay < 512; ++delay) { }
    return (float)race_cells[(idx + 1) % 512];
}
extern "C" __global__ void output_only(int n, float *out, int racing) {
    int idx = threadIdx.x;
    if (idx < n) out[idx] = value_for(idx, racing);
}
extern "C" __global__ void output_and_trace(int n, float *out, int racing) {
    int idx = threadIdx.x;
    if (idx < n) {
        float value = value_for(idx, racing);
        out[idx] = value;
        trace_ordered_float(value, (unsigned int)idx, 0xffffffffU);
    }
}`
	cmod, err := nvrtcx.Compile(source, nvrtcx.WithStandard("c++17"), nvrtcx.WithNoCache())
	if err != nil {
		t.Fatal(err)
	}
	mod, err := cmod.Load(dev)
	if err != nil {
		t.Fatal(err)
	}
	defer mod.Close()

	const n, runs = 512, 64
	observed := map[bool]map[string]bool{false: {}, true: {}}
	traceHashes := map[bool]map[uint64]bool{false: {}, true: {}}
	for _, instrumented := range []bool{false, true} {
		for _, racing := range []bool{false, true} {
			for run := 0; run < runs; run++ {
				output, trace, err := runRaceProbe(dev, mod, n, racing, instrumented)
				if err != nil {
					t.Fatalf("instrumented=%t racing=%t run=%d: %v", instrumented, racing, run, err)
				}
				if instrumented {
					traceHashes[racing][trace] = true
				} else {
					observed[racing][output] = true
				}
			}
		}
	}
	t.Logf("race output: %d distinct outputs in %d uninstrumented runs; %d trace hashes in %d instrumented runs", len(observed[true]), runs, len(traceHashes[true]), runs)
	t.Logf("race-free output: %d distinct outputs in %d uninstrumented runs; %d trace hashes in %d instrumented runs", len(observed[false]), runs, len(traceHashes[false]), runs)
	if len(observed[true]) <= 1 {
		t.Fatalf("planted race did not vary uninstrumented output; cannot measure preservation")
	}
	if len(traceHashes[true]) <= 1 {
		t.Fatalf("instrumented trace suppressed observed race: %d output vectors but one trace hash", len(observed[true]))
	}
	if len(observed[false]) != 1 || len(traceHashes[false]) != 1 {
		t.Fatalf("race-free control varied: outputs=%d traces=%d", len(observed[false]), len(traceHashes[false]))
	}
}

func runRaceProbe(dev *cuda.Device, mod driver.Module, n int, racing, instrumented bool) (string, uint64, error) {
	var key string
	var hash uint64
	err := dev.WithContext(func() error {
		if instrumented {
			if err := nvrtcx.ResetDeviceTrace(mod); err != nil {
				return err
			}
		}
		out, err := driver.MemAllocBytes(uint(n * 4))
		if err != nil {
			return err
		}
		defer out.Free()
		entry := "output_only"
		if instrumented {
			entry = "output_and_trace"
		}
		fn, err := mod.Function(entry)
		if err != nil {
			return err
		}
		nValue := int32(n)
		raceValue := int32(0)
		if racing {
			raceValue = 1
		}
		args := []unsafe.Pointer{unsafe.Pointer(&nValue), unsafe.Pointer(&out), unsafe.Pointer(&raceValue)}
		if err := fn.LaunchKernel(driver.Dim3{X: 1, Y: 1, Z: 1}, driver.Dim3{X: uint32(n), Y: 1, Z: 1}, 0, 0, args); err != nil {
			return err
		}
		if err := driver.CtxSynchronize(); err != nil {
			return err
		}
		values := make([]float32, n)
		if err := out.CopyToHostPtr(unsafe.Pointer(unsafe.SliceData(values)), uint(n*4)); err != nil {
			return err
		}
		buf := make([]byte, n*4)
		for i, value := range values {
			binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(value))
		}
		key = fmt.Sprintf("%x", buf)
		if instrumented {
			hash, err = nvrtcx.ReadDeviceTrace(mod)
			if err != nil {
				return err
			}
		}
		return nil
	})
	return key, hash, err
}

func TestDifferential_AgreesOnReference(t *testing.T) {
	dev := cudatest.Require(t)
	if dev == nil {
		return
	}
	defer dev.Close()
	g, inputs := buildTestGraph()
	res, err := Compare(context.Background(), g, WithInputs(inputs), WithCudaDevice(dev))
	if err != nil {
		t.Fatal(err)
	}
	if res.Status != StatusAgreed {
		t.Fatalf("Compare = %v, want Agreed", res)
	}
	if res.CPUHash != res.DeviceHash {
		t.Fatalf("CPU hash 0x%016x != device hash 0x%016x", res.CPUHash, res.DeviceHash)
	}
}

func TestPerturbation_ReportsTable(t *testing.T) {
	dev := cudatest.Require(t)
	if dev == nil {
		return
	}
	defer dev.Close()
	g, inputs := buildTestGraph()
	perts := append(DefaultPerturbations(), Perturbation{Name: "compile_failure", JITOptLevel: 4})
	calls := 0
	compile := func(src string, opts []nvrtcx.Option, _ cudakernel.JITOptions) (*nvrtcx.CompiledModule, error) {
		calls++
		if calls == len(perts) {
			return nil, errors.New("injected compile failure")
		}
		return nvrtcx.Compile(src, opts...)
	}
	report, err := RunPerturbations(context.Background(), g, perts,
		WithInputs(inputs), WithCudaDevice(dev), withCompileKernel(compile))
	if err != nil {
		t.Fatal(err)
	}
	t.Log("\n" + report.String())
	if len(report.Results) != len(perts) {
		t.Fatalf("got %d rows, want %d", len(report.Results), len(perts))
	}
	for _, row := range report.Results[:len(report.Results)-1] {
		if row.CompileErr != nil || row.RunErr != nil || row.Status != StatusAgreed {
			t.Errorf("perturbation %q: status=%v compile=%v run=%v", row.Perturbation.Name, row.Status, row.CompileErr, row.RunErr)
		}
	}
	failed := report.Results[len(report.Results)-1]
	if failed.CompileErr == nil || failed.RunErr != nil || failed.Status != StatusUnmeasured {
		t.Fatalf("compile failure row = %#v", failed)
	}
	if errors.Is(failed.CompileErr, ErrPerturbationArtifactMismatch) {
		t.Fatal("injected compile error misreported as artifact mismatch")
	}
}
