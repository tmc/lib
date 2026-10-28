package cudacc

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/tmc/lib/nvidia/nvml"
	"github.com/tmc/lib/nvidia/x/cudaerrors"
	"github.com/tmc/lib/nvidia/x/cudalock"
)

const cudaccNativeEnv = "CUDA_CUDACC_NATIVE"

type cudaccNativeGateClosedError struct{}

func (cudaccNativeGateClosedError) Error() string {
	return "cudacc: native gate is closed: CUDA_CUDACC_NATIVE=1 unset, no device examined"
}

func (cudaccNativeGateClosedError) Unwrap() error { return cudaerrors.ErrHardwarePending }

func checkCudaccNativeGate(getenv func(string) string) error {
	if getenv(cudaccNativeEnv) != "1" {
		return cudaccNativeGateClosedError{}
	}
	if _, err := cudalock.Inherited(getenv); err != nil {
		return fmt.Errorf("cudacc: native gate open without inherited gpu lock: %w", err)
	}
	return nil
}

func TestCudaccNativeClosedGateControl(t *testing.T) {
	if os.Getenv(cudaccNativeEnv) == "1" {
		t.Skip("closed-gate control applies when CUDA_CUDACC_NATIVE != 1")
	}
	if err := checkCudaccNativeGate(os.Getenv); !errors.Is(err, cudaerrors.ErrHardwarePending) {
		t.Fatalf("closed gate = %v, want ErrHardwarePending", err)
	}
	presenceOnly := func(k string) string {
		switch k {
		case cudaccNativeEnv, "GPU_LOCK_HELD":
			return "1"
		}
		return ""
	}
	if err := checkCudaccNativeGate(presenceOnly); err == nil || errors.Is(err, cudaerrors.ErrHardwarePending) {
		t.Fatalf("GPU_LOCK_HELD=1 gate = %v, want lock error", err)
	}
	lockDir := filepath.Join(t.TempDir(), "gpu-lock")
	h, err := cudalock.Acquire(t.Context(), cudalock.WithDir(lockDir), cudalock.WithLabel("cudacc-native-gate-test"))
	if err != nil {
		t.Fatalf("cudalock.Acquire: %v", err)
	}
	t.Cleanup(func() { _ = h.Release() })
	held := func(k string) string {
		switch k {
		case cudaccNativeEnv:
			return "1"
		case "GPU_LOCK_HELD":
			return strconv.Itoa(os.Getpid())
		case "GPU_LOCKDIR":
			return lockDir
		}
		return ""
	}
	if err := checkCudaccNativeGate(held); err != nil {
		t.Fatalf("gate with inherited lock: %v", err)
	}
}

// TestCudaccNativeMode reads the driver's confidential-computing settings
// through NVML and checks that they map to a CCMode. It asserts only that
// the values parse; it does not require any particular mode.
func TestCudaccNativeMode(t *testing.T) {
	if os.Getenv(cudaccNativeEnv) != "1" {
		t.Skip("native check pending: CUDA_CUDACC_NATIVE=1 required")
	}
	if err := checkCudaccNativeGate(os.Getenv); err != nil {
		t.Fatalf("%v: run under gpu-lock", err)
	}
	if err := nvml.Init(); err != nil {
		t.Fatalf("nvml init: %v", err)
	}
	t.Cleanup(func() {
		if err := nvml.Shutdown(); err != nil {
			t.Errorf("nvml shutdown: %v", err)
		}
	})

	settings := nvml.SystemConfComputeSettings{Version: nvml.SystemConfComputeSettingsV1Version}
	err := nvml.SystemGetConfComputeSettings(&settings)
	source := "nvmlSystemGetConfComputeSettings"
	if nvml.IsUnsupported(err) {
		// Older drivers lack the settings query; the state query has no
		// multi-GPU mode, so a CC-enabled result maps to single-GPU CC-On.
		state, serr := nvml.SystemGetConfComputeState()
		switch {
		case nvml.IsUnsupported(serr):
			t.Logf("confidential computing queries unsupported (%v; %v): device has no CC, mode %v", err, serr, CCModeOff)
			return
		case serr != nil:
			t.Fatalf("nvmlSystemGetConfComputeState: %v", serr)
		}
		settings = nvml.SystemConfComputeSettings{Environment: state.Environment, CcFeature: state.CcFeature, DevToolsMode: state.DevToolsMode}
		source = "nvmlSystemGetConfComputeState"
	} else if err != nil {
		t.Fatalf("nvmlSystemGetConfComputeSettings: %v", err)
	}
	mode, err := ModeFromNVML(settings.CcFeature, settings.MultiGpuMode)
	if err != nil {
		t.Fatalf("%s returned %+v: %v", source, settings, err)
	}
	t.Logf("%s: feature=%d multi_gpu=%d environment=%d devtools=%d -> mode %v",
		source, settings.CcFeature, settings.MultiGpuMode, settings.Environment, settings.DevToolsMode, mode)
}
