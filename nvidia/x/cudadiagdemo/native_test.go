package cudadiagdemo

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/x/cuptiactivity"
)

func TestNativeBoundsRefuseBeforeAdmission(t *testing.T) {
	for _, cfg := range []NativeConfig{{Elements: 0, Rounds: 1}, {Elements: 1 << 21, Rounds: 1}, {Elements: 1, Rounds: 65}, {Device: -1, Elements: 1, Rounds: 1}, {Elements: 1, Rounds: 1, Concurrency: -1}, {Elements: 1, Rounds: 64, Concurrency: 9}, {Elements: 1 << 20, Rounds: 2, Concurrency: 5}} {
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
	cfg := NativeConfig{Elements: 4096, Rounds: 2, DeviceIterations: 4}
	result, err := RunNative(context.Background(), cfg)
	if err != nil {
		t.Fatal(err)
	}
	if result.Synthetic || result.Qualification != "partial" || result.PlannedRequests != 3*cfg.Rounds || result.Concurrency != 4 || len(result.Trials) != 3*cfg.Rounds || len(result.Requests) != 3*cfg.Rounds || result.Composition == nil || len(result.Composition.Requests) != 3*cfg.Rounds {
		t.Fatalf("invalid native receipt: %+v", result)
	}
	seen := map[string]bool{}
	for _, trial := range result.Trials {
		launches := 1
		switch trial.Arm {
		case "baseline", "candidate":
		case "device":
			launches = cfg.DeviceIterations
		default:
			t.Fatalf("unexpected physical arm: %+v", trial)
		}
		key := fmt.Sprintf("%d/%s", trial.Round, trial.Arm)
		if trial.Round < 1 || trial.Round > cfg.Rounds || seen[key] || trial.RequestID == "" || trial.Attempt != trial.Arm || trial.RequestedLaunches != launches || trial.KernelLaunches != launches || !trial.Correct || len(trial.Calls) != launches+3 || trial.OutputSHA256 == "" || trial.Error != "" {
			t.Fatalf("invalid physical trial: %+v", trial)
		}
		seen[key] = true
	}
	if err := checkNativeInputDigests(result.Trials); err != nil {
		t.Fatalf("physical input digests: %v", err)
	}
	for _, request := range result.Composition.Requests {
		if request.Diagnosis.Admitted || request.Refusal == "" {
			t.Fatalf("partial native evidence promoted: %+v", request)
		}
	}
}

// checkNativeInputDigests checks the launch-ready-v1 input rule: every trial
// has an input digest, all arms of a round share it, and rounds differ.
func checkNativeInputDigests(trials []NativeTrial) error {
	rounds := map[int]string{}
	for _, trial := range trials {
		if trial.InputSHA256 == "" {
			return fmt.Errorf("%s/%s: empty input digest", trial.RequestID, trial.Attempt)
		}
		if d, ok := rounds[trial.Round]; ok && d != trial.InputSHA256 {
			return fmt.Errorf("round %d: %s/%s input %s, want %s", trial.Round, trial.RequestID, trial.Attempt, trial.InputSHA256, d)
		}
		rounds[trial.Round] = trial.InputSHA256
	}
	byDigest := map[string]int{}
	for round, d := range rounds {
		if other, ok := byDigest[d]; ok {
			return fmt.Errorf("rounds %d and %d share input %s", min(round, other), max(round, other), d)
		}
		byDigest[d] = round
	}
	return nil
}

func TestNativeInputDigestsByRound(t *testing.T) {
	// trials returns two rounds of planned arms with the given input
	// digests, in plan order.
	trials := func(digests ...string) []NativeTrial {
		var out []NativeTrial
		for i, d := range digests {
			round := i/len(nativeArms) + 1
			arm := nativeArms[i%len(nativeArms)]
			out = append(out, NativeTrial{RequestID: fmt.Sprintf("request-%d", round), Attempt: arm, Arm: arm, Round: round, InputSHA256: d})
		}
		return out
	}
	tests := []struct {
		name   string
		trials []NativeTrial
		want   string
	}{
		{"matched rounds", trials("a", "a", "a", "b", "b", "b"), ""},
		{"empty digest", trials("a", "a", "a", "b", "b", ""), "request-2/device: empty input digest"},
		{"within-round mismatch", trials("a", "c", "a", "b", "b", "b"), "round 1: request-1/candidate input c, want a"},
		{"cross-round collision", trials("a", "a", "a", "a", "a", "a"), "rounds 1 and 2 share input a"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := checkNativeInputDigests(tt.trials)
			if tt.want == "" {
				if err != nil {
					t.Fatalf("checkNativeInputDigests = %v, want nil", err)
				}
				return
			}
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("checkNativeInputDigests = %v, want error containing %q", err, tt.want)
			}
		})
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
