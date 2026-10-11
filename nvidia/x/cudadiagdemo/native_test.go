package cudadiagdemo

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuptiactivity"
)

func TestNativeBoundsRefuseBeforeAdmission(t *testing.T) {
	for _, cfg := range []NativeConfig{{Elements: 0, Rounds: 1}, {Elements: 1 << 21, Rounds: 1}, {Elements: 1, Rounds: 65}, {Device: -1, Elements: 1, Rounds: 1}} {
		if result, err := RunNative(context.Background(), cfg); err == nil || result != nil {
			t.Fatalf("invalid bounds admitted: %+v", cfg)
		}
	}
}
func TestNativeOracle(t *testing.T) {
	for _, output := range [][]float32{{3, 5}, {3, 6}, {3, float32(math.NaN())}, {3, float32(math.Inf(1))}, {}} {
		err := checkNativeOutput([]float32{1, 2}, output)
		if len(output) == 2 && output[1] == 5 {
			if err != nil {
				t.Fatal(err)
			}
		} else if err == nil {
			t.Fatalf("incorrect output accepted: %v", output)
		}
	}
}
func TestNativeCaptureCompleteness(t *testing.T) {
	if nativeCaptureComplete(cuptiactivity.Result{}) || nativeCaptureComplete(cuptiactivity.Result{Closed: true}) {
		t.Fatal("empty capture qualified")
	}
}
func TestNativePhysical(t *testing.T) {
	if os.Getenv("GPUDIAGDEMO_NATIVE_TEST") != "1" {
		t.Skip("opt-in physical GPU execution requires GPUDIAGDEMO_NATIVE_TEST=1 and canonical lease")
	}
	result, err := RunNative(context.Background(), NativeConfig{Elements: 4096, Rounds: 2})
	if err != nil {
		t.Fatal(err)
	}
	if result.Synthetic || result.Qualification != "partial" || len(result.Trials) != 4 {
		t.Fatalf("invalid native receipt: %+v", result)
	}
	for _, trial := range result.Trials {
		if !trial.Correct || len(trial.Calls) != 4 || trial.OutputSHA256 == "" {
			t.Fatalf("invalid physical trial: %+v", trial)
		}
	}
}

func TestNativeCleanupRetainsResourcesAndLease(t *testing.T) {
	calls := make(map[string]int)
	failedSession, failedBuffer := true, true
	cleanup := &nativeCleanup{}
	for _, name := range []string{"lease", "device", "module", "buffer", "session"} {
		name := name
		cleanup.add(name, func() error {
			calls[name]++
			if name == "session" && failedSession {
				return errors.New("session close")
			}
			if name == "buffer" && failedBuffer {
				return errors.New("buffer free")
			}
			return nil
		})
	}
	result := &NativeResult{cleanup: cleanup}
	if err := result.Close(); err == nil || !result.CleanupPending {
		t.Fatal("failed session lost ownership")
	}
	if calls["session"] != 1 || calls["buffer"] != 0 || calls["device"] != 0 || calls["lease"] != 0 {
		t.Fatalf("cleanup advanced past failed gate: %v", calls)
	}
	failedSession = false
	if err := result.Close(); err == nil {
		t.Fatal("buffer failure ignored")
	}
	if calls["session"] != 2 || calls["buffer"] != 1 || calls["module"] != 0 || calls["lease"] != 0 {
		t.Fatalf("resources released after failed buffer: %v", calls)
	}
	failedBuffer = false
	if err := result.Close(); err != nil || result.CleanupPending {
		t.Fatalf("explicit cleanup failed: %v", err)
	}
	if calls["session"] != 2 || calls["buffer"] != 2 || calls["module"] != 1 || calls["device"] != 1 || calls["lease"] != 1 {
		t.Fatalf("cleanup order/count: %v", calls)
	}
	if err := result.Close(); err != nil || calls["lease"] != 1 {
		t.Fatal("successful cleanup retried")
	}
}

func ExampleNativeResult_Close() {
	var result NativeResult
	fmt.Println(result.Close())
	// Output: <nil>
}

func TestNativeActivityWriterBound(t *testing.T) {
	var b bytes.Buffer
	writer := nativeActivityWriter{buffer: &b, remaining: 3}
	if n, err := writer.Write([]byte("ok")); err != nil || n != 2 {
		t.Fatalf("write %d %v", n, err)
	}
	if n, err := writer.Write([]byte("too large")); err == nil || n != 0 || b.String() != "ok" {
		t.Fatalf("overflow %d %v %q", n, err, b.String())
	}
}
