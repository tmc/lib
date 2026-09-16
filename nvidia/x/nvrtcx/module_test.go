package nvrtcx

import (
	"fmt"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuda"
	"github.com/tmc/lib/nvidia/x/cudalaunch"
)

type testValidKernels struct {
	Add     *cuda.Kernel `cuda:"vector_add"`
	Scale   *cuda.Kernel
	Ignored *cuda.Kernel `cuda:"-"`
}

func TestModuleNilReceiver(t *testing.T) {
	var cm *CompiledModule
	var k testValidKernels

	tests := []struct {
		name string
		fn   func() error
	}{
		{
			name: "LoadModule",
			fn: func() error {
				_, err := cm.LoadModule(nil)
				return err
			},
		},
		{
			name: "LoadOutputModule",
			fn: func() error {
				_, err := cm.LoadOutputModule(nil, OutputPTX)
				return err
			},
		},
		{
			name: "Bind",
			fn: func() error {
				_, err := cm.Bind(nil, &k)
				return err
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.fn()
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if !strings.Contains(err.Error(), "nil compiled module") {
				t.Fatalf("unexpected error message: %v", err)
			}
		})
	}
}

func TestModuleNilDevice(t *testing.T) {
	cm := &CompiledModule{
		output: OutputPTX,
		ptx:    "// dummy ptx",
	}
	var k testValidKernels

	tests := []struct {
		name string
		fn   func() error
	}{
		{
			name: "LoadModule",
			fn: func() error {
				_, err := cm.LoadModule(nil)
				return err
			},
		},
		{
			name: "LoadOutputModule",
			fn: func() error {
				_, err := cm.LoadOutputModule(nil, OutputPTX)
				return err
			},
		},
		{
			name: "Bind",
			fn: func() error {
				_, err := cm.Bind(nil, &k)
				return err
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := tt.fn()
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if !strings.Contains(err.Error(), "nil device passed to Load") {
				t.Fatalf("unexpected error message: %v", err)
			}
		})
	}
}

func TestLoadOutputModuleDiagnostics(t *testing.T) {
	tests := []struct {
		name    string
		cm      *CompiledModule
		kind    OutputKind
		wantErr string
	}{
		{
			name:    "optix_ir_rejected",
			cm:      &CompiledModule{output: OutputOptiXIR, optixir: []byte("optix")},
			kind:    OutputOptiXIR,
			wantErr: "OptiX IR must be passed to optixModuleCreate",
		},
		{
			name:    "unavailable_kind",
			cm:      &CompiledModule{output: OutputPTX, ptx: "// ptx"},
			kind:    OutputCUBIN,
			wantErr: "output 2 is unavailable",
		},
		{
			name:    "unsupported_output_kind",
			cm:      &CompiledModule{output: OutputPTX, ptx: "// ptx"},
			kind:    OutputKind(99),
			wantErr: "cannot be loaded directly",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tt.cm.LoadOutputModule(nil, tt.kind)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("err = %v, want substring %q", err, tt.wantErr)
			}
		})
	}
}

func TestBindOwnedCompatibility(t *testing.T) {
	// Verify that struct types used with Bind are structurally valid for cudalaunch.BindOwned.
	mod := new(cuda.Module) // zero module is closed
	var valid testValidKernels
	err := cudalaunch.BindOwned(mod, &valid)
	if err == nil {
		t.Fatal("expected error on closed module, got nil")
	}
	// Error should be from lookup on closed module, not from struct field validation.
	if !strings.Contains(err.Error(), "vector_add") || !strings.Contains(err.Error(), "closed module") {
		t.Fatalf("unexpected BindOwned error: %v", err)
	}

	// Verify that invalid struct targets fail during validation.
	invalidTargets := []struct {
		name    string
		target  any
		wantErr string
	}{
		{
			name:    "nil_target",
			target:  nil,
			wantErr: "want non-nil pointer to struct",
		},
		{
			name:    "non_pointer",
			target:  valid,
			wantErr: "want non-nil pointer to struct",
		},
		{
			name:    "pointer_to_non_struct",
			target:  new(int),
			wantErr: "want pointer to struct",
		},
		{
			name:    "no_kernel_fields",
			target:  &struct{ Count int }{},
			wantErr: "has no *cuda.Kernel fields",
		},
		{
			name: "unexported_kernel_field",
			target: &struct {
				kernel *cuda.Kernel
			}{},
			wantErr: "is unexported",
		},
		{
			name: "invalid_field_type_with_cuda_tag",
			target: &struct {
				Bad int `cuda:"bad"`
			}{},
			wantErr: "want *cuda.Kernel",
		},
	}

	for _, tt := range invalidTargets {
		t.Run(tt.name, func(t *testing.T) {
			err := cudalaunch.BindOwned(mod, tt.target)
			if err == nil {
				t.Fatal("expected error, got nil")
			}
			if !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("err = %v, want substring %q", err, tt.wantErr)
			}
		})
	}
}

func TestPhysicalModuleLoadAndBind(t *testing.T) {
	if os.Getenv("CUDA_COMPILER_TEST") != "1" {
		t.Skip("CUDA_COMPILER_TEST=1 required for live compilation and device testing")
	}
	dev, err := cuda.NewDevice(0)
	if err != nil {
		t.Skipf("no CUDA device available: %v", err)
	}
	defer func() {
		if err := dev.Close(); err != nil {
			t.Error(err)
		}
	}()

	const src = `extern "C" __global__ void add_one(int *p) { *p += 1; }`
	cm, err := Compile(src, WithDeviceArch(dev), WithNoCache())
	if err != nil {
		t.Fatalf("Compile failed: %v", err)
	}

	t.Run("LoadModule", func(t *testing.T) {
		mod, err := cm.LoadModule(dev)
		if err != nil {
			t.Fatalf("LoadModule failed: %v", err)
		}
		if mod == nil {
			t.Fatal("expected non-nil Module")
		}
		if err := mod.Close(); err != nil {
			t.Fatalf("mod.Close failed: %v", err)
		}
	})

	t.Run("LoadOutputModule", func(t *testing.T) {
		mod, err := cm.LoadOutputModule(dev, OutputPTX)
		if err != nil {
			t.Fatalf("LoadOutputModule failed: %v", err)
		}
		if mod == nil {
			t.Fatal("expected non-nil Module")
		}
		if err := mod.Close(); err != nil {
			t.Fatalf("mod.Close failed: %v", err)
		}
	})

	t.Run("Bind_Success", func(t *testing.T) {
		var k struct {
			AddOne *cuda.Kernel `cuda:"add_one"`
		}
		mod, err := cm.Bind(dev, &k)
		if err != nil {
			t.Fatalf("Bind failed: %v", err)
		}
		if mod == nil {
			t.Fatal("expected non-nil Module")
		}
		defer mod.Close()

		if k.AddOne == nil {
			t.Fatal("expected non-nil AddOne kernel")
		}
	})

	t.Run("Bind_MissingKernel", func(t *testing.T) {
		var k struct {
			Missing *cuda.Kernel `cuda:"non_existent_kernel"`
		}
		mod, err := cm.Bind(dev, &k)
		if err == nil {
			if mod != nil {
				_ = mod.Close()
			}
			t.Fatal("expected Bind to fail for missing kernel")
		}
		if mod != nil {
			t.Fatal("expected nil Module on failed Bind")
		}
		if !strings.Contains(err.Error(), "non_existent_kernel") {
			t.Fatalf("unexpected error: %v", err)
		}
	})
}

func ExampleCompiledModule_LoadModule() {
	var cm *CompiledModule
	_, err := cm.LoadModule(nil)
	fmt.Println(err)
	// Output: nvrtcx: nil compiled module
}

func ExampleCompiledModule_LoadOutputModule() {
	var cm *CompiledModule
	_, err := cm.LoadOutputModule(nil, OutputPTX)
	fmt.Println(err)
	// Output: nvrtcx: nil compiled module
}

func ExampleCompiledModule_Bind() {
	var cm *CompiledModule
	var kernels struct {
		Add *cuda.Kernel `cuda:"VectorAdd"`
	}
	_, err := cm.Bind(nil, &kernels)
	fmt.Println(err)
	// Output: nvrtcx: nil compiled module
}
