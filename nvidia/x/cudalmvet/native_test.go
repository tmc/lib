//go:build native

package cudalmvet

import (
	"context"
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/x/cudakernel"
	"github.com/tmc/lib/nvidia/x/cudatest"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

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
