package cudaschedule

import (
	"fmt"
	"os"
	"time"
)

type recordedLimits struct {
	Queue     time.Duration `json:"queue"`
	Admission time.Duration `json:"admission"`
	Runtime   time.Duration `json:"runtime"`
	Cleanup   time.Duration `json:"cleanup"`
	Lease     time.Duration `json:"lease"`
}

func (l recordedLimits) validate() (time.Duration, error) {
	// The cap keeps every sum representable. It is a refusal boundary, not a
	// default or permission to spend the full budget for a particular case.
	const maximum = 24 * time.Hour
	for _, field := range []struct {
		name  string
		value time.Duration
	}{
		{"queue", l.Queue}, {"admission", l.Admission}, {"runtime", l.Runtime},
		{"cleanup", l.Cleanup}, {"lease", l.Lease},
	} {
		if field.value <= 0 || field.value > maximum {
			return 0, fmt.Errorf("recorded run: %s limit out of range", field.name)
		}
	}
	if l.Lease < l.Admission+l.Runtime+l.Cleanup {
		return 0, fmt.Errorf("recorded run: lease does not cover admission, runtime and cleanup")
	}
	// Keeper custody begins before queue admission and ends only after bounded
	// retirement. The native runtime budget does not include queue wait.
	return l.Queue + l.Admission + l.Runtime + l.Cleanup, nil
}

func recordedOutsideLease() error {
	for _, key := range []string{"GPU_LOCK_HELD", "GPU_LOCK_DISABLED", "CUDATEST_NO_LOCK"} {
		if os.Getenv(key) != "" {
			return fmt.Errorf("recorded run: controller must be outside lease; %s is set", key)
		}
	}
	return nil
}
