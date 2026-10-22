//go:build linux

package main

import (
	"errors"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/nvrtcx"
)

func TestCUDAProbeCompilePolicy(t *testing.T) {
	failure := errors.New("forced CUBIN refusal")
	for _, name := range []string{"ds4_probe.cu", "ds4_model_probe.cu", "ds4_q8_0_row.cu"} {
		t.Run(name, func(t *testing.T) {
			physical, outputs, compiles, loads := 0, 0, 0, 0
			ops := cudaProbeCompileOperations{
				physical: func(*cuda.Device) nvrtcx.Option { physical++; return nvrtcx.WithPhysicalArch(12, 1) },
				output: func(kind nvrtcx.OutputKind) nvrtcx.Option {
					outputs++
					if kind != nvrtcx.OutputCUBIN {
						t.Fatalf("compile output %v", kind)
					}
					return nvrtcx.WithOutput(kind)
				},
				compile: func(source string, opts ...nvrtcx.Option) (*nvrtcx.CompiledModule, error) {
					compiles++
					if source != cudaProbeSource || len(opts) != 3 {
						t.Fatal("probe source/options")
					}
					return &nvrtcx.CompiledModule{}, nil
				},
				load: func(*nvrtcx.CompiledModule, *cuda.Device, nvrtcx.OutputKind) (driver.Module, error) {
					loads++
					return 0, failure
				},
			}
			compiled, err := compileCUDAProbe(nil, name, ops)
			if err != nil || compiled == nil {
				t.Fatal(compiled, err)
			}
			ops.load = func(_ *nvrtcx.CompiledModule, _ *cuda.Device, kind nvrtcx.OutputKind) (driver.Module, error) {
				loads++
				if kind != nvrtcx.OutputCUBIN {
					t.Fatalf("load output %v", kind)
				}
				return 0, failure
			}
			module, err := loadCUDAProbe(compiled, nil, ops)
			if module != 0 || !errors.Is(err, failure) || physical != 1 || outputs != 1 || compiles != 1 || loads != 1 {
				t.Fatalf("module=%v err=%v physical/output/compile/load=%d/%d/%d/%d", module, err, physical, outputs, compiles, loads)
			}
			ops.compile = func(string, ...nvrtcx.Option) (*nvrtcx.CompiledModule, error) { return nil, failure }
			if _, err := compileCUDAProbe(nil, name, ops); !errors.Is(err, failure) {
				t.Fatal("compile refusal lost", err)
			}
		})
	}
}
