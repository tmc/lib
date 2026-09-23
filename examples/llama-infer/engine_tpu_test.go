//go:build linux && tpu

package main

import (
	"context"
	"errors"
	"os"
	"strings"
	"testing"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/tmc/lib/openxla"
)

func skipIfNoTPU(t *testing.T, err error) {
	if err == nil {
		return
	}
	// Hardware testing gate: when RUN_TPU_HARDWARE_TESTS=1 is set, construction failures MUST fail the test.
	if os.Getenv("RUN_TPU_HARDWARE_TESTS") != "" {
		t.Fatalf("TPU engine initialization failed on enabled hardware test: %v", err)
	}

	// Only skip when libtpu is genuinely not found or no addressable devices are present on host
	if errors.Is(err, ErrLibTPUNotFound) || errors.Is(err, ErrNoAddressableTPUDevices) {
		t.Skipf("Skipping TPU test due to absent hardware/library: %v", err)
	}
	// Any ABI mismatch, compilation failure, or unexpected initialization error MUST fail the test!
	t.Fatalf("TPU engine initialization failed unexpectedly: %v", err)
}

func TestTPUEngineAvailabilityClassifier(t *testing.T) {
	// Unit-test the availability classifier separately without hardware
	tests := []struct {
		name      string
		err       error
		wantSkip  bool
		wantFatal bool
	}{
		{
			name:     "libtpu_not_found",
			err:      ErrLibTPUNotFound,
			wantSkip: true,
		},
		{
			name:     "no_devices",
			err:      ErrNoAddressableTPUDevices,
			wantSkip: true,
		},
		{
			name:      "abi_or_compile_error",
			err:       errors.New("compile stablehlo: syntax error in MLIR"),
			wantFatal: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			isAvailable := errors.Is(tt.err, ErrLibTPUNotFound) || errors.Is(tt.err, ErrNoAddressableTPUDevices)
			if tt.wantSkip && !isAvailable {
				t.Errorf("expected error %v to be recognized as absent hardware/library", tt.err)
			}
			if tt.wantFatal && isAvailable {
				t.Errorf("expected error %v NOT to be classified as absent hardware", tt.err)
			}
		})
	}
}

func TestTPUEngineGenerationClosed(t *testing.T) {
	// Test directly on a zero ModelEngine without requiring hardware or library presence
	var eng ModelEngine
	cfg := DefaultTinyLlamaConfig()

	err := eng.Generate(context.Background(), cfg, "test", 10, func(id int32, text string) bool {
		return true
	})
	if err == nil {
		t.Fatal("Expected Generate to reject a closed engine")
	}
	if !strings.Contains(err.Error(), "closed") {
		t.Fatalf("Expected closed error, got %v", err)
	}
}

func TestTPUEngineCompilation(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	eng, err := GetOrCompileTPUEngine(cfg)
	skipIfNoTPU(t, err)
	if eng == nil {
		t.Fatal("Expected non-nil engine")
	}
	defer eng.Free()

	count := 0
	if err := eng.GenerateFromTokenIDs(context.Background(), []int32{1, 2}, 3, func(int32) bool { count++; return true }); err != nil {
		t.Fatal(err)
	}
	if count != 3 {
		t.Fatalf("generated %d tokens, want 3", count)
	}
}

func TestTPUEngineSmokeMatmul(t *testing.T) {
	cfg := DefaultTinyLlamaConfig()
	err := RunTPUSmokeTest(cfg)
	skipIfNoTPU(t, err)
}

// mockBufferData provides distinct non-zero backing storage for buffer identification.
type mockBufferData struct {
	id  string
	tag [16]byte
}

type mockDeviceData struct {
	id  int32
	tag [16]byte
}

// buildMockTPUEngine sets up a ModelEngine wired to a mock PJRT Api table.
func buildMockTPUEngine(t *testing.T, opts mockTPUOpts) (*ModelEngine, *openxla.Device, *bool, *bool, *bool) {
	api := &openxla.Api{
		StructSize: 2048,
	}

	mockErr := &openxla.ErrorHandle{}
	api.Error_Destroy = purego.NewCallback(func(args *openxla.Error_Destroy_Args) {})
	api.Error_Message = purego.NewCallback(func(args *openxla.Error_Message_Args) {
		msg := "mock injected error"
		args.Message = &([]byte(msg))[0]
		args.Message_size = uintptr(len(msg))
	})

	// Retain addressable devices pointer array storage
	dummyDevMem := &mockDeviceData{id: 42}
	dummyDev := (*openxla.Device)(unsafe.Pointer(dummyDevMem))
	devList := []*openxla.Device{dummyDev}

	api.Client_AddressableDevices = purego.NewCallback(func(args *openxla.Client_AddressableDevices_Args) *openxla.ErrorHandle {
		args.Addressable_devices = (*openxla.Device)(unsafe.Pointer(&devList[0]))
		args.Num_addressable_devices = 1
		return nil
	})

	api.Event_Await = purego.NewCallback(func(args *openxla.Event_Await_Args) *openxla.ErrorHandle {
		return nil
	})
	api.Event_Destroy = purego.NewCallback(func(args *openxla.Event_Destroy_Args) *openxla.ErrorHandle {
		return nil
	})

	// Distinct stable backing storage for mock buffers
	b1Mem := &mockBufferData{id: "b1"}
	b2Mem := &mockBufferData{id: "b2"}
	outMem := &mockBufferData{id: "out"}

	b1Ptr := (*openxla.Buffer)(unsafe.Pointer(b1Mem))
	b2Ptr := (*openxla.Buffer)(unsafe.Pointer(b2Mem))
	outPtr := (*openxla.Buffer)(unsafe.Pointer(outMem))

	b1Destroyed := false
	b2Destroyed := false
	outBufDestroyed := false

	bufAllocCount := 0
	api.Client_BufferFromHostBuffer = purego.NewCallback(func(args *openxla.Client_BufferFromHostBuffer_Args) *openxla.ErrorHandle {
		if args.Device != dummyDev {
			t.Errorf("Client_BufferFromHostBuffer received unexpected device: got %p, want %p", args.Device, dummyDev)
		}
		bufAllocCount++
		if bufAllocCount == 1 {
			args.Buffer = b1Ptr
			return nil
		}
		if bufAllocCount == 2 {
			if opts.b2AllocFail {
				return mockErr
			}
			args.Buffer = b2Ptr
			return nil
		}
		return mockErr
	})

	api.Buffer_Destroy = purego.NewCallback(func(args *openxla.Buffer_Destroy_Args) *openxla.ErrorHandle {
		switch args.Buffer {
		case b1Ptr:
			b1Destroyed = true
			if opts.b1DestroyFail {
				return mockErr
			}
		case b2Ptr:
			b2Destroyed = true
			if opts.b2DestroyFail {
				return mockErr
			}
		case outPtr:
			outBufDestroyed = true
			if opts.outDestroyFail {
				return mockErr
			}
		}
		return nil
	})

	api.LoadedExecutable_GetExecutable = purego.NewCallback(func(args *openxla.LoadedExecutable_GetExecutable_Args) *openxla.ErrorHandle {
		args.Executable = &openxla.Executable{}
		return nil
	})
	api.Executable_NumOutputs = purego.NewCallback(func(args *openxla.Executable_NumOutputs_Args) *openxla.ErrorHandle {
		args.Num_outputs = 1
		return nil
	})
	api.Executable_Destroy = purego.NewCallback(func(args *openxla.Executable_Destroy_Args) *openxla.ErrorHandle {
		return nil
	})

	api.LoadedExecutable_Execute = purego.NewCallback(func(args *openxla.LoadedExecutable_Execute_Args) *openxla.ErrorHandle {
		if opts.execFail {
			return mockErr
		}
		if args.Output_lists != nil {
			outPtrs := *(*[]**openxla.Buffer)(unsafe.Pointer(&struct {
				p   unsafe.Pointer
				len int
				cap int
			}{args.Output_lists, 1, 1}))
			if len(outPtrs) > 0 && outPtrs[0] != nil {
				*outPtrs[0] = outPtr
			}
		}
		return nil
	})

	api.Buffer_ToHostBuffer = purego.NewCallback(func(args *openxla.Buffer_ToHostBuffer_Args) *openxla.ErrorHandle {
		if opts.toHostFail {
			return mockErr
		}
		// Write expected 6 float32s: [11.0, 11.5, 9.0, 27.0, 27.5, 27.0]
		expected := []float32{11.0, 11.5, 9.0, 27.0, 27.5, 27.0}
		dstSlice := unsafe.Slice((*float32)(args.Dst), 6)
		copy(dstSlice, expected)
		return nil
	})

	client := newClientWrapperForTest(api, &openxla.Client{})
	exec := newExecutableWrapperForTest(api, &openxla.LoadedExecutable{})

	eng := &ModelEngine{
		client: client,
		exec:   exec,
	}
	return eng, dummyDev, &b1Destroyed, &b2Destroyed, &outBufDestroyed
}

type mockTPUOpts struct {
	b2AllocFail    bool
	execFail       bool
	toHostFail     bool
	b1DestroyFail  bool
	b2DestroyFail  bool
	outDestroyFail bool
}

func TestForwardPassMockTPU_ProductionLifecycle(t *testing.T) {
	// 1. Success path: all cleanups pass, no errors
	t.Run("success_all_clean", func(t *testing.T) {
		eng, dummyDev, b1D, b2D, outD := buildMockTPUEngine(t, mockTPUOpts{})
		devs, devErr := eng.client.AddressableDevices()
		if devErr != nil || len(devs) != 1 || devs[0] != dummyDev {
			t.Fatalf("expected addressable device %p, got %v (%v)", dummyDev, devs, devErr)
		}
		err := eng.ForwardPassMockTPU()
		if err != nil {
			t.Fatalf("expected nil error on clean mock run, got: %v", err)
		}
		if !*b1D || !*b2D || !*outD {
			t.Fatalf("expected all buffers destroyed: b1=%v b2=%v out=%v", *b1D, *b2D, *outD)
		}
	})

	// 2. Input b1 destructor failure on successful forward pass
	t.Run("b1_destructor_failure", func(t *testing.T) {
		eng, _, b1D, b2D, outD := buildMockTPUEngine(t, mockTPUOpts{b1DestroyFail: true})
		err := eng.ForwardPassMockTPU()
		if err == nil || !strings.Contains(err.Error(), "destroy b1") {
			t.Fatalf("expected destroy b1 error, got: %v", err)
		}
		if !*b1D || !*b2D || !*outD {
			t.Fatalf("expected all destructors called: b1=%v b2=%v out=%v", *b1D, *b2D, *outD)
		}
	})

	// 3. Input b2 destructor failure on successful forward pass
	t.Run("b2_destructor_failure", func(t *testing.T) {
		eng, _, b1D, b2D, outD := buildMockTPUEngine(t, mockTPUOpts{b2DestroyFail: true})
		err := eng.ForwardPassMockTPU()
		if err == nil || !strings.Contains(err.Error(), "destroy b2") {
			t.Fatalf("expected destroy b2 error, got: %v", err)
		}
		if !*b1D || !*b2D || !*outD {
			t.Fatalf("expected all destructors called: b1=%v b2=%v out=%v", *b1D, *b2D, *outD)
		}
	})

	// 4. Output buffer destructor failure on successful forward pass
	t.Run("output_destructor_failure", func(t *testing.T) {
		eng, _, b1D, b2D, outD := buildMockTPUEngine(t, mockTPUOpts{outDestroyFail: true})
		err := eng.ForwardPassMockTPU()
		if err == nil || !strings.Contains(err.Error(), "destroy output buffer") {
			t.Fatalf("expected destroy output buffer error, got: %v", err)
		}
		if !*b1D || !*b2D || !*outD {
			t.Fatalf("expected all destructors called: b1=%v b2=%v out=%v", *b1D, *b2D, *outD)
		}
	})

	// 5. b2 allocation failure + b1 destructor failure: b1 destructor error must be preserved
	t.Run("b2_alloc_fail_and_b1_destroy_fail", func(t *testing.T) {
		eng, _, b1D, b2D, _ := buildMockTPUEngine(t, mockTPUOpts{b2AllocFail: true, b1DestroyFail: true})
		err := eng.ForwardPassMockTPU()
		if err == nil {
			t.Fatal("expected error on b2 alloc failure, got nil")
		}
		if !strings.Contains(err.Error(), "bufferFromHost in2") {
			t.Errorf("expected primary b2 alloc error in joined error, got %v", err)
		}
		if !strings.Contains(err.Error(), "destroy b1") {
			t.Errorf("expected b1 cleanup error in joined error, got %v", err)
		}
		if !*b1D {
			t.Errorf("expected b1 destructor to be called")
		}
		if *b2D {
			t.Errorf("b2 should not be destroyed since it was never allocated")
		}
	})

	// 6. Execution failure + destructor failure: both primary and destructor errors must be joined
	t.Run("exec_fail_and_destructor_fail", func(t *testing.T) {
		eng, _, b1D, b2D, outD := buildMockTPUEngine(t, mockTPUOpts{execFail: true, b1DestroyFail: true})
		err := eng.ForwardPassMockTPU()
		if err == nil {
			t.Fatal("expected error on exec failure, got nil")
		}
		if !strings.Contains(err.Error(), "execute:") {
			t.Errorf("expected primary execute error in joined error, got %v", err)
		}
		if !strings.Contains(err.Error(), "destroy b1") {
			t.Errorf("expected b1 cleanup error in joined error, got %v", err)
		}
		if !*b1D || !*b2D {
			t.Errorf("expected b1 and b2 to be destroyed: b1=%v b2=%v", *b1D, *b2D)
		}
		if *outD {
			t.Errorf("output should not be destroyed since execute failed")
		}
	})
}

func TestModelEngine_Free_ProductionErrorPropagation(t *testing.T) {
	api := &openxla.Api{
		StructSize: 2048,
	}
	mockErr := &openxla.ErrorHandle{}
	api.Error_Destroy = purego.NewCallback(func(args *openxla.Error_Destroy_Args) {})
	api.Error_Message = purego.NewCallback(func(args *openxla.Error_Message_Args) {
		msg := "client destroy failed"
		args.Message = &([]byte(msg))[0]
		args.Message_size = uintptr(len(msg))
	})
	api.Client_Destroy = purego.NewCallback(func(args *openxla.Client_Destroy_Args) *openxla.ErrorHandle {
		return mockErr
	})

	client := newClientWrapperForTest(api, &openxla.Client{})
	eng := &ModelEngine{client: client}

	err := eng.Free()
	if err == nil || !strings.Contains(err.Error(), "destroy client") {
		t.Fatalf("expected Free() to return joined client destroy error, got: %v", err)
	}
}
