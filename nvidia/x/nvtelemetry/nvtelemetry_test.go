package nvtelemetry_test

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/tmc/lib/nvidia/nvml"
	"github.com/tmc/lib/nvidia/x/nvtelemetry"
)

func TestClassifyNVMLError(t *testing.T) {
	tests := []struct {
		name string
		err  error
		want string
	}{
		{
			name: "nil error",
			err:  nil,
			want: "",
		},
		{
			name: "not supported",
			err:  nvml.ErrNotSupported,
			want: nvtelemetry.ReasonUnsupported,
		},
		{
			name: "wrapped not supported",
			err:  fmt.Errorf("query: %w", nvml.ErrNotSupported),
			want: nvtelemetry.ReasonUnsupported,
		},
		{
			name: "permission denied",
			err:  nvml.ErrNoPermission,
			want: nvtelemetry.ReasonPermissionDenied,
		},
		{
			name: "gpu lost",
			err:  nvml.ErrGpuIsLost,
			want: nvtelemetry.ReasonGPULost,
		},
		{
			name: "unknown failure",
			err:  errors.New("connection failed"),
			want: nvtelemetry.ReasonQueryFailure,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := nvtelemetry.ClassifyNVMLError(tt.err)
			if got != tt.want {
				t.Errorf("ClassifyNVMLError(%v) = %q, want %q", tt.err, got, tt.want)
			}
		})
	}
}

func TestFakeSamplerScripted(t *testing.T) {
	dev := nvtelemetry.DeviceMetadata{
		UUID:  "GPU-1234",
		Index: 0,
		Model: "NVIDIA L4",
	}
	s := nvtelemetry.NewFakeSampler(dev)

	if devs := s.Devices(); len(devs) != 1 || devs[0].UUID != "GPU-1234" {
		t.Fatalf("Devices() = %+v, want 1 device with UUID GPU-1234", devs)
	}

	batch1 := []nvtelemetry.Reading{
		{
			Name:       nvtelemetry.MetricGPUUtilization,
			Unit:       nvtelemetry.UnitPercent,
			Backend:    "fake",
			DeviceUUID: "GPU-1234",
			Evidence:   nvtelemetry.EvidenceMeasured,
			Value:      42.0,
		},
	}
	batch2 := []nvtelemetry.Reading{
		{
			Name:       nvtelemetry.MetricPowerUsageMW,
			Unit:       nvtelemetry.UnitMilliwatts,
			Backend:    "fake",
			DeviceUUID: "GPU-1234",
			Evidence:   nvtelemetry.EvidenceMeasured,
			IntValue:   75000,
			IsInt:      true,
		},
	}
	s.Enqueue(batch1, batch2)

	ctx := context.Background()

	got1, err := s.Sample(ctx)
	if err != nil {
		t.Fatalf("Sample 1 failed: %v", err)
	}
	if len(got1) != 1 || got1[0].Value != 42.0 {
		t.Errorf("Sample 1 = %+v, want Value=42.0", got1)
	}

	got2, err := s.Sample(ctx)
	if err != nil {
		t.Fatalf("Sample 2 failed: %v", err)
	}
	if len(got2) != 1 || got2[0].IntValue != 75000 {
		t.Errorf("Sample 2 = %+v, want IntValue=75000", got2)
	}

	// Fallback call
	got3, err := s.Sample(ctx)
	if err != nil {
		t.Fatalf("Sample 3 fallback failed: %v", err)
	}
	if len(got3) != 1 || got3[0].Value != 0 {
		t.Errorf("Sample 3 = %+v, want Value=0", got3)
	}
}

func TestReadingPreservesExactInt(t *testing.T) {
	// Value exceeding 2^53 (loss of precision in float64)
	var largeVal uint64 = 1<<55 + 12345
	r := nvtelemetry.Reading{
		Name:     nvtelemetry.MetricTotalEnergyMJ,
		Unit:     nvtelemetry.UnitMillijoules,
		Evidence: nvtelemetry.EvidenceMeasured,
		IntValue: largeVal,
		IsInt:    true,
	}

	if !r.IsAvailable() {
		t.Errorf("IsAvailable() = false, want true")
	}
	if r.IntValue != largeVal {
		t.Errorf("IntValue = %d, want %d", r.IntValue, largeVal)
	}
}

func TestRealClock(t *testing.T) {
	var c nvtelemetry.RealClock
	m := c.NowMonotonic()
	u := c.NowUnix()

	if m <= 0 {
		t.Errorf("NowMonotonic() = %d, want > 0", m)
	}
	if u <= 0 {
		t.Errorf("NowUnix() = %d, want > 0", u)
	}
}

func TestFakeClock_EpochUniquenessAndZeroValue(t *testing.T) {
	// Zero-value FakeClock returns explicitly unqualified empty epoch ID
	var zeroClock nvtelemetry.FakeClock
	if id := zeroClock.MonotonicEpochID(); id != "" {
		t.Errorf("zeroClock.MonotonicEpochID() = %q, want empty string", id)
	}

	// Distinct instances within process have distinct epoch IDs with process prefix
	fc1 := nvtelemetry.NewFakeClock(100, 200)
	fc2 := nvtelemetry.NewFakeClock(100, 200)

	id1 := fc1.MonotonicEpochID()
	id2 := fc2.MonotonicEpochID()

	if id1 == "" {
		t.Errorf("fc1 epoch ID is empty")
	}
	if id2 == "" {
		t.Errorf("fc2 epoch ID is empty")
	}
	if id1 == id2 {
		t.Errorf("fc1 and fc2 have identical epoch IDs: %q", id1)
	}
	if !strings.HasPrefix(id1, "fake-proc-") {
		t.Errorf("fc1 epoch ID %q missing process prefix", id1)
	}

	// Explicit SetEpochID
	fc1.SetEpochID("explicit-epoch-test")
	if fc1.MonotonicEpochID() != "explicit-epoch-test" {
		t.Errorf("SetEpochID failed: got %q, want explicit-epoch-test", fc1.MonotonicEpochID())
	}
}

// TestFakeClock_SubprocessEpochDisambiguation verifies that independent subprocesses
// generate distinct FakeClock epoch IDs despite starting sequence numbers at 1.
func TestFakeClock_SubprocessEpochDisambiguation(t *testing.T) {
	if os.Getenv("NVTELEMETRY_TEST_SUBPROCESS") == "1" {
		fc := nvtelemetry.NewFakeClock(0, 0)
		fmt.Println(fc.MonotonicEpochID())
		return
	}

	cmd := exec.Command(os.Args[0], "-test.run=^TestFakeClock_SubprocessEpochDisambiguation$")
	cmd.Env = append(os.Environ(), "NVTELEMETRY_TEST_SUBPROCESS=1")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("subprocess failed: %v, output: %s", err, string(out))
	}

	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	var subprocessEpoch string
	for _, l := range lines {
		l = strings.TrimSpace(l)
		if strings.HasPrefix(l, "fake-proc-") {
			subprocessEpoch = l
			break
		}
	}
	if subprocessEpoch == "" {
		t.Fatalf("failed to find subprocess epoch in output: %s", string(out))
	}

	localClock := nvtelemetry.NewFakeClock(0, 0)
	localEpoch := localClock.MonotonicEpochID()

	if subprocessEpoch == localEpoch {
		t.Fatalf("subprocess and local process generated identical FakeClock epoch IDs: %q", localEpoch)
	}
	if !strings.HasPrefix(subprocessEpoch, "fake-proc-") {
		t.Errorf("subprocess epoch %q does not have expected prefix", subprocessEpoch)
	}
}
