//go:build linux

package nvrtc_test

import (
	"errors"
	"os"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/ebitengine/purego"
	"github.com/tmc/lib/nvidia/nvrtc"
)

// TestNativeFlowCallback compiles host-side only; no GPU or driver is required.
// Opt-in execution requires NVRTC with flow callback support and matched builtins.
func TestNativeFlowCallback(t *testing.T) {
	if os.Getenv("NVRTC_FLOW_TEST") != "1" {
		t.Skip("NVRTC_FLOW_TEST=1 requires NVRTC flow callbacks and matched builtins")
	}
	const source = `extern "C" __global__ void flow(int* out) { out[threadIdx.x] = threadIdx.x + 1; }`
	for _, cancel := range []bool{false, true} {
		name := "continue"
		if cancel {
			name = "cancel"
		}
		t.Run(name, func(t *testing.T) {
			program, err := nvrtc.CreateProgram(source, "flow.cu", nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				if err := nvrtc.DestroyProgram(&program); err != nil {
					t.Error(err)
				}
			}()
			// This opaque token is never dereferenced, so no Go pointer crosses into C.
			const token uintptr = 73
			var calls, mismatches atomic.Uint64
			var stop atomic.Int32
			if cancel {
				stop.Store(1)
			}
			// purego callback trampolines last for the process lifetime; this test creates
			// only two. Captured atomic state remains reachable throughout compilation.
			callback := purego.NewCallback(func(payload, reserved uintptr) int32 {
				calls.Add(1)
				if payload != token || reserved != 0 {
					mismatches.Add(1)
				}
				// No compiler APIs or testing methods may be invoked from this callback.
				// stop never returns to zero during a compilation.
				return stop.Load()
			})
			if err := program.SetFlowCallback(callback, token); err != nil {
				t.Fatal(err)
			}
			err = program.Compile([]string{"--gpu-architecture=compute_80"})
			log, logErr := program.GetProgramLog()
			if logErr != nil {
				t.Fatal(logErr)
			}
			t.Logf("callbacks=%d cancel=%v compile=%v log=%q", calls.Load(), cancel, err, log)
			if calls.Load() == 0 {
				t.Fatal("compiler never invoked callback")
			}
			if mismatches.Load() != 0 {
				t.Fatalf("callback payload/reserved mismatches=%d", mismatches.Load())
			}
			if cancel {
				if !errors.Is(err, nvrtc.ErrCancelled) {
					t.Fatalf("compile=%v, want ErrCancelled", err)
				}
			} else {
				if err != nil {
					t.Fatal(err)
				}
				ptx, err := program.GetPTX()
				if err != nil {
					t.Fatal(err)
				}
				if !strings.Contains(string(ptx), ".entry flow(") {
					t.Fatal("successful compilation has no flow entry")
				}
			}
		})
	}
}
