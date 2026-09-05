package nvrtcx

import (
	"errors"
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/driver"
	"github.com/tmc/lib/nvidia/x/cuda"
)

func TestDriverJITDiagnostics(t *testing.T) {
	if os.Getenv("CUDA_JIT_TEST") != "1" {
		t.Skip("CUDA_JIT_TEST=1 requires a CUDA device and driver JIT")
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := dev.Close(); err != nil {
			t.Error(err)
		}
	}()
	const header = ".version 8.0\n.target sm_90\n.address_size 64\n.visible .entry probe() {\n"
	for _, tt := range []struct {
		name, body string
		invalid    bool
	}{
		{"invalid", "not_a_ptx_instruction;\nret;\n}", true},
		{"valid_after_failure", "ret;\n}", false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			cm := &CompiledModule{output: OutputPTX, ptx: header + tt.body}
			mod, log, err := cm.LoadOutputWithLog(dev, OutputPTX)
			if mod != 0 {
				defer func() {
					if err := dev.WithContext(mod.Unload); err != nil {
						t.Error(err)
					}
				}()
			}
			t.Logf("output=%d info=%q error=%q load=%v", log.Output, log.Info, log.Error, err)
			if tt.invalid {
				if !errors.Is(err, driver.ErrInvalidPtx) || log.Error == "" || mod != 0 {
					t.Fatalf("invalid PTX result: module=%v log=%+v err=%v", mod, log, err)
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if err := dev.WithContext(func() error {
				fn, err := mod.GetFunction("probe")
				if err != nil {
					return err
				}
				if err := fn.LaunchKernel(1, 1, 1, 1, 1, 1, 0, 0, nil); err != nil {
					return err
				}
				return driver.CtxSynchronize()
			}); err != nil {
				t.Fatal(err)
			}
		})
	}
}
