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
__device__ volatile unsigned int race_cells[512];
extern "C" __global__ void reset_race_cells() {
    int idx = threadIdx.x;
    if (idx < 512) race_cells[idx] = 0xdeadbeefU;
}
__device__ inline float value_for(int idx, int racing, unsigned int schedule_seed) {
    unsigned int delay = ((unsigned int)idx * 1103515245U + schedule_seed * 12345U) & 2047U;
    for (volatile unsigned int i = 0; i < delay; ++i) { }
    int target = racing ? ((idx + 1) % 512) : idx;
    unsigned int value = race_cells[target];
    for (volatile int i = 0; i < ((idx + (int)schedule_seed) & 127); ++i) { }
    race_cells[idx] = (unsigned int)(idx + 1);
    return (float)value;
}
extern "C" __global__ void output_only(int n, float *out, int racing, unsigned int schedule_seed) {
    int idx = threadIdx.x;
    if (idx < n) out[idx] = value_for(idx, racing, schedule_seed);
}
extern "C" __global__ void output_and_trace(int n, float *out, int racing, unsigned int schedule_seed) {
    int idx = threadIdx.x;
    if (idx < n) {
        float value = value_for(idx, racing, schedule_seed);
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
	type arm struct{ racing, instrumented bool }
	arms := []arm{{true, false}, {true, true}, {false, false}, {false, true}}
	outputs := make(map[arm]map[string]bool, len(arms))
	hashes := make(map[arm]map[uint64]bool, len(arms))
	for _, a := range arms {
		outputs[a] = make(map[string]bool)
		hashes[a] = make(map[uint64]bool)
	}
	for run := 0; run < runs; run++ {
		start := run % len(arms)
		for step := 0; step < len(arms); step++ {
			a := arms[(start+step)%len(arms)]
			output, trace, err := runRaceProbe(dev, mod, n, a.racing, a.instrumented, uint32(run))
			if err != nil {
				t.Fatalf("racing=%t instrumented=%t run=%d: %v", a.racing, a.instrumented, run, err)
			}
			outputs[a][output] = true
			if a.instrumented {
				hashes[a][trace] = true
			}
		}
	}
	for _, a := range arms {
		label := "race-free"
		if a.racing {
			label = "race"
		}
		mode := "uninstrumented"
		if a.instrumented {
			mode = "instrumented"
		}
		t.Logf("%s %s: %d distinct outputs and %d distinct trace hashes in %d runs", label, mode, len(outputs[a]), len(hashes[a]), runs)
	}
	if len(outputs[arm{racing: true}]) <= 1 {
		t.Fatalf("race did not vary uninstrumented (inconclusive): %d distinct outputs in %d runs", len(outputs[arm{racing: true}]), runs)
	}
	if len(hashes[arm{racing: true, instrumented: true}]) <= 1 {
		t.Fatalf("race varied uninstrumented but gave one hash instrumented (suppressed): %d distinct outputs and %d hashes", len(outputs[arm{racing: true}]), len(hashes[arm{racing: true, instrumented: true}]))
	}
	if len(outputs[arm{racing: false}]) != 1 || len(outputs[arm{racing: false, instrumented: true}]) != 1 || len(hashes[arm{racing: false, instrumented: true}]) != 1 {
		t.Fatalf("race-free control varied (broken control): uninstrumented outputs=%d instrumented outputs=%d instrumented hashes=%d", len(outputs[arm{racing: false}]), len(outputs[arm{racing: false, instrumented: true}]), len(hashes[arm{racing: false, instrumented: true}]))
	}
}

func runRaceProbe(dev *cuda.Device, mod driver.Module, n int, racing, instrumented bool, scheduleSeed uint32) (string, uint64, error) {
	var key string
	var hash uint64
	err := dev.WithContext(func() error {
		reset, err := mod.Function("reset_race_cells")
		if err != nil {
			return err
		}
		if err := reset.LaunchKernel(driver.Dim3{X: 1, Y: 1, Z: 1}, driver.Dim3{X: 512, Y: 1, Z: 1}, 0, 0, nil); err != nil {
			return err
		}
		if err := driver.CtxSynchronize(); err != nil {
			return err
		}
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
		args := []unsafe.Pointer{unsafe.Pointer(&nValue), unsafe.Pointer(&out), unsafe.Pointer(&raceValue), unsafe.Pointer(&scheduleSeed)}
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
